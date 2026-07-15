from __future__ import annotations

import re
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from .analyzer import RepoAnalyzer
from .block_store import BlockStore
from .docker_runtime import DockerRuntime
from .llm_planner import make_planner, merge_usage_summaries
from .models import (
    BlockExecution,
    BuildRequest,
    BuildResult,
    Checkpoint,
    CommandBlock,
    CommandResult,
    RepairContext,
    RepairProbeResult,
    RepoContext,
    progress_control_for_ablation,
)
from .oracle import load_oracle_commands
from .planner import BlockPlanner
from .repair import RepairCommand, RepairPlanner, RepairProbeCommand, make_repair_planner
from .utils import new_run_id, normalize_posix_source, tail_text

RuntimeFactory = Callable[[BuildRequest, str], DockerRuntime]


def _should_continue_llm_repair_failure(error: str | None) -> bool:
    if not error:
        return False
    if error == "LLM repair returned no usable suggestions":
        return True
    normalized = error.lower()
    if "llm repair request failed" not in normalized:
        return False
    transient_markers = (
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "timeout",
        "timed out",
        "rate limit",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection",
    )
    return any(marker in normalized for marker in transient_markers)


def _first_failed_block(blocks: list[CommandBlock]) -> CommandBlock | None:
    return next((block for block in blocks if block.status == "failed"), None)


class EnvironmentBuilder:
    def __init__(
        self,
        request: BuildRequest,
        *,
        analyzer: RepoAnalyzer | None = None,
        planner: BlockPlanner | None = None,
        repair_planner: RepairPlanner | None = None,
        runtime_factory: RuntimeFactory = DockerRuntime,
    ):
        self.request = request.normalized()
        self.run_id = self.request.run_id or new_run_id()
        self.run_dir = (self.request.state_dir or Path(".pheragent")) / "runs" / self.run_id
        self.store = BlockStore(self.run_dir)
        self.analyzer = analyzer or RepoAnalyzer()
        self.planner = planner or make_planner(
            mode=self.request.planner_mode,
            model=self.request.llm_model,
            api_key_env=self.request.openai_api_key_env,
            base_url_env=self.request.openai_base_url_env,
            base_url=self.request.openai_base_url,
            timeout=self.request.llm_timeout,
            max_tokens=self.request.llm_max_tokens,
            retries=self.request.llm_retries,
            retry_delay=self.request.llm_retry_delay,
            api_mode=self.request.llm_api,
        )
        self.repair_planner = repair_planner or make_repair_planner(
            mode=self.request.planner_mode,
            model=self.request.llm_model,
            api_key_env=self.request.openai_api_key_env,
            base_url_env=self.request.openai_base_url_env,
            base_url=self.request.openai_base_url,
            timeout=self.request.llm_timeout,
            max_tokens=self.request.llm_max_tokens,
            retries=self.request.llm_retries,
            retry_delay=self.request.llm_retry_delay,
            api_mode=self.request.llm_api,
        )
        self.runtime_factory = runtime_factory
        self.progress_control = progress_control_for_ablation(self.request.ablation_mode)
        self._whole_script_recovery_completed = False
        self._final_replay_whole_script = False

    def plan_only(self) -> BuildResult:
        self._emit(f"run {self.run_id}: analyze repo {self.request.repo_path}")
        context = self._analyze_repo_context()
        self.store.save_context(context)
        self._emit(f"run {self.run_id}: plan blocks")
        blocks = self.store.write_blocks(
            _ensure_inherited_block_preludes(
                self.planner.plan(context),
                workdir=self.request.container_workdir,
                context=context,
            )
        )
        self.store.save_context(context)
        result = BuildResult(
            ok=True,
            run_id=self.run_id,
            state_dir=self.run_dir,
            scripts_dir=self.store.scripts_dir,
            manifest_path=self.store.manifest_path,
            blocks=blocks,
        )
        self._save_manifest(result)
        return result

    def build(self) -> BuildResult:
        self._emit(f"run {self.run_id}: analyze repo {self.request.repo_path}")
        context = self._analyze_repo_context()
        self.store.save_context(context)
        checkpoints: list[Checkpoint] = []
        executions: list[BlockExecution] = []
        blocks: list[CommandBlock] = []
        current_image: str | None = None
        workspace_image: str | None = None
        result = BuildResult(
            ok=False,
            run_id=self.run_id,
            state_dir=self.run_dir,
            scripts_dir=self.store.scripts_dir,
            manifest_path=self.store.manifest_path,
            blocks=blocks,
            checkpoints=checkpoints,
            executions=executions,
            ablation_mode=self.request.ablation_mode,
            progress_control=self.progress_control,
            final_clean_replay_enabled=self.progress_control.final_clean_replay,
        )
        if self.request.resume_from and self.progress_control.final_clean_replay:
            result.error = (
                "--resume-from cannot be used with an ablation mode that requires "
                "final clean replay; rerun from the base Dockerfile or use "
                "--ablation without-final-clean-replay"
            )
            result.final_clean_replay_ok = False
            result.final_clean_replay_failure_stage = "resume-from"
            self._save_manifest(result)
            return result

        runtime = self.runtime_factory(self.request, self.run_id)

        try:
            start_index = 0
            if self.request.resume_from:
                current_image = self.request.resume_from
                self._emit(f"run {self.run_id}: resume from {current_image}")
                runtime.start(image_ref=current_image, seed_repo=False)
                self._collect_container_context(
                    runtime=runtime,
                    context=context,
                    executions=executions,
                    checkpoint_before=current_image,
                )
                blocks = self._load_or_plan_blocks(context)
                result.blocks = blocks
                checkpoints.append(
                    Checkpoint(
                        id="resume-from",
                        image_ref=current_image,
                        block_id=self._infer_resume_block_id(blocks, current_image),
                        parent_image_ref=None,
                        kind="resume",
                    )
                )
                start_index = self._resume_start_index(blocks, current_image)
                self._mark_resume_skipped_blocks(blocks, start_index, current_image)
            else:
                self._emit(f"run {self.run_id}: build base image")
                build_result = runtime.build_base_image()
                executions.append(
                    self._execution_from_result(
                        block_id="base-image",
                        phase="docker_build",
                        attempt=1,
                        command_result=build_result,
                        checkpoint_before=None,
                    )
                )
                self.store.append_execution(executions[-1])
                if not build_result.ok:
                    self._emit(f"run {self.run_id}: base image build failed")
                    result.error = "base Docker image build failed"
                    self._save_manifest(result)
                    return result

                self._emit(f"run {self.run_id}: start container and copy repo")
                runtime.start(seed_repo=True)
                self._collect_container_context(
                    runtime=runtime,
                    context=context,
                    executions=executions,
                    checkpoint_before=runtime.base_image,
                )
                blocks = self._load_or_plan_blocks(context)
                result.blocks = blocks
                workspace_checkpoint = runtime.commit(
                    block_id=None,
                    parent_image_ref=runtime.base_image,
                    kind="workspace",
                )
                checkpoints.append(workspace_checkpoint)
                current_image = workspace_checkpoint.image_ref
                workspace_image = workspace_checkpoint.image_ref

            self.store.save_context(context)

            if self.progress_control.forward_granularity == "whole-script":
                whole_ok, current_image = self._run_whole_script_forward(
                    runtime=runtime,
                    blocks=blocks,
                    baseline_image=current_image,
                    checkpoints=checkpoints,
                    executions=executions,
                )
                if not whole_ok:
                    self._emit(f"run {self.run_id}: whole-script forward failed")
                    result.error = "whole-script forward failed"
                    result.final_image = current_image
                    self._save_manifest(result)
                    return result
            else:
                for block_index, block in enumerate(blocks[start_index:], start=start_index):
                    self._emit(f"run {self.run_id}: block {block.id} start")
                    block.baseline_checkpoint = current_image
                    self.store.update_block(block)
                    block_ok, current_image = self._run_block_with_repair(
                        runtime=runtime,
                        block=block,
                        baseline_image=current_image,
                        repo_context=context,
                        completed_blocks=blocks[:block_index],
                        all_blocks=blocks,
                        workspace_image=workspace_image,
                        checkpoints=checkpoints,
                        executions=executions,
                    )
                    if not block_ok:
                        failed_result_block = _first_failed_block(blocks) or block
                        self._emit(
                            f"run {self.run_id}: block {failed_result_block.id} failed"
                        )
                        result.error = f"block failed: {failed_result_block.id}"
                        result.final_image = current_image
                        self._save_manifest(result)
                        return result
                    if self._whole_script_recovery_completed:
                        break

            if self.progress_control.final_clean_replay:
                if workspace_image is None:
                    result.error = "final clean replay requires an initial workspace checkpoint"
                    result.final_image = current_image
                    result.final_clean_replay_ok = False
                    result.final_clean_replay_failure_stage = "missing-workspace-checkpoint"
                    self._save_manifest(result)
                    return result
                self._emit(f"run {self.run_id}: final clean replay")
                replay_ok, replay_image, replay_failure_stage = self._run_final_clean_replay(
                    runtime=runtime,
                    blocks=blocks,
                    workspace_image=workspace_image,
                    checkpoints=checkpoints,
                    executions=executions,
                )
                result.final_clean_replay_ok = replay_ok
                result.final_clean_replay_image = replay_image
                result.final_clean_replay_failure_stage = replay_failure_stage
                if not replay_ok:
                    self._emit(f"run {self.run_id}: final clean replay failed")
                    result.error = "final clean replay failed"
                    result.final_image = replay_image or current_image
                    self._save_manifest(result)
                    return result
                current_image = replay_image or current_image

            if self.request.oracle_file is not None:
                self._emit(f"run {self.run_id}: oracle validation")
                oracle_ok = self._run_oracle_validation(
                    runtime=runtime,
                    checkpoint_image=current_image,
                    executions=executions,
                )
                if not oracle_ok:
                    self._emit(f"run {self.run_id}: oracle validation failed")
                    result.error = "oracle validation failed"
                    result.final_image = current_image
                    self._save_manifest(result)
                    return result

            result.ok = True
            result.final_image = current_image
            self._save_manifest(result)
            self._emit(f"run {self.run_id}: ok")
            return result
        except Exception as exc:
            self._emit(f"run {self.run_id}: failed: {exc}")
            result.error = str(exc)
            result.final_image = current_image
            self._save_manifest(result)
            return result
        finally:
            runtime.cleanup()

    def _save_manifest(self, result: BuildResult) -> None:
        result.ablation_mode = self.request.ablation_mode
        result.progress_control = self.progress_control
        result.final_clean_replay_enabled = self.progress_control.final_clean_replay
        result.llm_usage = self._llm_usage_summary()
        result.llm_usage_path = self.store.llm_usage_path
        self.store.save_llm_usage(result)
        self.store.save_manifest(result)

    def _llm_usage_summary(self) -> dict[str, dict[str, int]]:
        summaries: list[dict[str, dict[str, int]]] = []
        for component in (self.planner, self.repair_planner):
            usage_summary = getattr(component, "usage_summary", None)
            if callable(usage_summary):
                summaries.append(usage_summary())
        return merge_usage_summaries(*summaries)

    def _analyze_repo_context(self) -> RepoContext:
        context = self.analyzer.analyze(self.request.repo_path)
        context.task_description = self.request.task_description
        return context

    def _load_or_plan_blocks(self, context: RepoContext) -> list[CommandBlock]:
        if self.request.resume_from:
            existing_blocks = self.store.list_blocks()
            if existing_blocks:
                self._emit(f"run {self.run_id}: reuse existing block scripts")
                return self.store.write_blocks(
                    _ensure_inherited_block_preludes(
                        existing_blocks,
                        workdir=self.request.container_workdir,
                        context=context,
                    )
                )
        self._emit(f"run {self.run_id}: plan blocks")
        return self.store.write_blocks(
            _ensure_inherited_block_preludes(
                self.planner.plan(context),
                workdir=self.request.container_workdir,
                context=context,
            )
        )

    def _collect_container_context(
        self,
        *,
        runtime: DockerRuntime,
        context: RepoContext,
        executions: list[BlockExecution],
        checkpoint_before: str | None,
    ) -> None:
        self._emit(f"run {self.run_id}: collect container context")
        result = runtime.execute_command(
            _CONTAINER_PREFLIGHT_COMMAND,
            timeout=min(self.request.command_timeout, 120.0),
        )
        execution = self._execution_from_result(
            block_id="container-preflight",
            phase="container_preflight",
            attempt=1,
            command_result=result,
            checkpoint_before=checkpoint_before,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        context.runtime_notes = _runtime_notes_from_preflight(result)
        self.store.save_context(context)

    def _resume_start_index(self, blocks: list[CommandBlock], image_ref: str) -> int:
        if self.request.start_at_block:
            for index, block in enumerate(blocks):
                if block.id == self.request.start_at_block:
                    return index
            raise ValueError(f"start block not found: {self.request.start_at_block}")

        completed_index = self._infer_completed_block_index(blocks, image_ref)
        if completed_index is None:
            return 0
        return completed_index + 1

    def _mark_resume_skipped_blocks(
        self,
        blocks: list[CommandBlock],
        start_index: int,
        resume_image: str,
    ) -> None:
        for block in blocks[:start_index]:
            block.status = "skipped"
            block.success_checkpoint = block.success_checkpoint or resume_image
            self.store.update_block(block)

    def _infer_resume_block_id(self, blocks: list[CommandBlock], image_ref: str) -> str | None:
        completed_index = self._infer_completed_block_index(blocks, image_ref)
        if completed_index is None:
            return None
        return blocks[completed_index].id

    def _infer_completed_block_index(
        self,
        blocks: list[CommandBlock],
        image_ref: str,
    ) -> int | None:
        image_tag = image_ref.rsplit(":", 1)[-1]
        for index in range(len(blocks) - 1, -1, -1):
            block = blocks[index]
            if image_tag.endswith(f"-{block.id}-success") or image_tag.endswith(
                f"-{block.id}-repaired"
            ):
                return index
        return None

    def _run_block_with_repair(
        self,
        *,
        runtime: DockerRuntime,
        block: CommandBlock,
        baseline_image: str,
        repo_context: RepoContext,
        completed_blocks: list[CommandBlock],
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        all_blocks: list[CommandBlock] | None = None,
        workspace_image: str | None = None,
        allow_earlier_root_recovery: bool = True,
    ) -> tuple[bool, str]:
        attempt = 1
        last_result = self._execute_block(runtime, block, attempt, baseline_image, executions)
        validation_result = self._validate_block(
            runtime,
            block,
            attempt,
            baseline_image,
            executions,
        )
        if last_result.ok and (validation_result is None or validation_result.ok):
            finalize_result = self._finalize_checkpoint_tools(
                runtime=runtime,
                block=block,
                attempt=attempt,
                baseline_image=baseline_image,
                executions=executions,
            )
            if not finalize_result.ok:
                block.status = "failed"
                block.last_error = tail_text(finalize_result.combined_output, max_chars=4000)
                self.store.update_block(block)
                self._recreate_from_checkpoint(runtime, baseline_image)
                return False, baseline_image
            checkpoint = runtime.commit(
                block_id=block.id,
                parent_image_ref=baseline_image,
                kind="success",
            )
            checkpoints.append(checkpoint)
            block.success_checkpoint = checkpoint.image_ref
            block.status = "succeeded"
            self.store.update_block(block)
            self._emit(f"run {self.run_id}: block {block.id} checkpoint {checkpoint.image_ref}")
            return True, checkpoint.image_ref
        last_failure_phase = "block"
        if last_result.ok and validation_result is not None:
            last_result = validation_result
            last_failure_phase = "validation"

        if not self.progress_control.local_repair:
            block.status = "failed"
            block.last_error = tail_text(last_result.combined_output, max_chars=4000)
            self.store.update_block(block)
            self._recreate_from_checkpoint(runtime, baseline_image)
            return False, baseline_image

        if self.progress_control.recovery_granularity == "command":
            return self._run_block_command_recovery(
                runtime=runtime,
                block=block,
                baseline_image=baseline_image,
                repo_context=repo_context,
                completed_blocks=completed_blocks,
                executions=executions,
                checkpoints=checkpoints,
                last_result=last_result,
            )

        if self.progress_control.recovery_granularity == "whole-script":
            if workspace_image is None or all_blocks is None:
                block.status = "failed"
                block.last_error = "whole-script recovery requires a workspace checkpoint"
                self.store.update_block(block)
                return False, baseline_image
            return self._run_whole_script_recovery(
                runtime=runtime,
                failed_block=block,
                all_blocks=all_blocks,
                workspace_image=workspace_image,
                repo_context=repo_context,
                completed_blocks=completed_blocks,
                checkpoints=checkpoints,
                executions=executions,
                last_result=last_result,
            )

        if allow_earlier_root_recovery:
            earlier_root_block = self._localized_earlier_root_block(
                failed_block=block,
                failure_result=last_result,
                repo_context=repo_context,
                baseline_image=baseline_image,
                completed_blocks=completed_blocks,
                executions=executions,
                all_blocks=all_blocks,
            )
            if earlier_root_block is not None and all_blocks is not None:
                return self._run_earlier_block_recovery(
                    runtime=runtime,
                    failed_block=block,
                    root_block=earlier_root_block,
                    all_blocks=all_blocks,
                    repo_context=repo_context,
                    completed_blocks=completed_blocks,
                    checkpoints=checkpoints,
                    executions=executions,
                    failure_result=last_result,
                )

        probe_failures = 0
        max_probe_failures = max(0, self.request.max_probe_failures)
        probe_disabled = max_probe_failures == 0
        for repair_attempt in range(1, self.request.max_repair_attempts + 1):
            repair_context = self._repair_context(
                repo_context=repo_context,
                baseline_image=baseline_image,
                completed_blocks=completed_blocks,
                executions=executions,
            )
            probes: list[RepairProbeCommand] = []
            if not probe_disabled and probe_failures < max_probe_failures:
                probes = self.repair_planner.propose_probes(
                    block,
                    last_result,
                    context=repair_context,
                )
                self._record_llm_probe_status(
                    block=block,
                    attempt=repair_attempt,
                    baseline_image=baseline_image,
                    executions=executions,
                )
                probe_error = getattr(self.repair_planner, "last_probe_error", None)
                if probe_error:
                    probe_failures += 1
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM probe attempt "
                        f"{repair_attempt} failed ({probe_failures}/{max_probe_failures}): "
                        f"{tail_text(probe_error, max_chars=500)}"
                    )
                    if (
                        probe_failures < max_probe_failures
                        and repair_attempt < self.request.max_repair_attempts
                    ):
                        continue
                    self._emit(
                        f"run {self.run_id}: block {block.id} continue repair without probes"
                    )
                    probes = []
                elif (
                    not probes
                    and getattr(self.repair_planner, "last_probe_raw_response", None)
                    is not None
                ):
                    probe_disabled = True
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM requested no probes; "
                        "skip future probes"
                    )
            probe_results = self._run_repair_probes(
                runtime=runtime,
                block=block,
                baseline_image=baseline_image,
                attempt=repair_attempt,
                probes=probes,
                executions=executions,
            )
            if probe_results:
                repair_context = self._repair_context(
                    repo_context=repo_context,
                    baseline_image=baseline_image,
                    completed_blocks=completed_blocks,
                    executions=executions,
                    probe_results=probe_results,
                )
            suggestions = self.repair_planner.suggest(
                block,
                last_result,
                context=repair_context,
            )
            self._record_llm_repair_status(
                block=block,
                attempt=repair_attempt,
                baseline_image=baseline_image,
                executions=executions,
                ok=bool(suggestions),
            )
            if not suggestions:
                error = getattr(self.repair_planner, "last_llm_error", None)
                if error:
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM repair attempt "
                        f"{repair_attempt} failed: {tail_text(error, max_chars=500)}"
                    )
                if _should_continue_llm_repair_failure(error):
                    continue
                break
            repair = suggestions[min(repair_attempt - 1, len(suggestions) - 1)]
            self._emit(f"run {self.run_id}: block {block.id} repair attempt {repair_attempt}")
            self._recreate_from_checkpoint(runtime, baseline_image)
            if (
                self.progress_control.checkpoint_rollback
                and last_failure_phase == "validation"
                and block.repair_attempts > 0
            ):
                prep_result = runtime.execute_script(
                    self.store.script_path(block.id),
                    timeout=self.request.command_timeout,
                )
                prep_execution = self._execution_from_result(
                    block_id=block.id,
                    phase="repair_prep",
                    attempt=repair_attempt,
                    command_result=prep_result,
                    checkpoint_before=baseline_image,
                )
                executions.append(prep_execution)
                self.store.append_execution(prep_execution)
                if not prep_result.ok:
                    last_result = prep_result
                    last_failure_phase = "block"
                    continue
            repair_result = runtime.execute_command(
                repair.command,
                timeout=self.request.command_timeout,
            )
            execution = self._execution_from_result(
                block_id=block.id,
                phase="repair",
                attempt=repair_attempt,
                command_result=repair_result,
                checkpoint_before=baseline_image,
                repair_command=repair.command,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not repair_result.ok:
                last_result = repair_result
                last_failure_phase = "repair"
                continue

            if self.progress_control.patch_back:
                block = self.repair_planner.patch_block(block, repair)
                self.store.update_block(block)
            self._recreate_from_checkpoint(runtime, baseline_image)
            attempt += 1
            last_result = self._execute_block(runtime, block, attempt, baseline_image, executions)
            last_failure_phase = "block"
            validation_result = self._validate_block(
                runtime, block, attempt, baseline_image, executions
            )
            if last_result.ok and (validation_result is None or validation_result.ok):
                finalize_result = self._finalize_checkpoint_tools(
                    runtime=runtime,
                    block=block,
                    attempt=attempt,
                    baseline_image=baseline_image,
                    executions=executions,
                )
                if not finalize_result.ok:
                    block.status = "failed"
                    block.last_error = tail_text(finalize_result.combined_output, max_chars=4000)
                    self.store.update_block(block)
                    self._recreate_from_checkpoint(runtime, baseline_image)
                    return False, baseline_image
                checkpoint = runtime.commit(
                    block_id=block.id,
                    parent_image_ref=baseline_image,
                    kind="repaired",
                )
                checkpoints.append(checkpoint)
                block.success_checkpoint = checkpoint.image_ref
                block.status = "succeeded"
                self.store.update_block(block)
                self._emit(
                    f"run {self.run_id}: block {block.id} repaired checkpoint "
                    f"{checkpoint.image_ref}"
                )
                return True, checkpoint.image_ref
            if last_result.ok and validation_result is not None:
                last_result = validation_result
                last_failure_phase = "validation"

        block.status = "failed"
        block.last_error = tail_text(last_result.combined_output, max_chars=4000)
        self.store.update_block(block)
        self._recreate_from_checkpoint(runtime, baseline_image)
        return False, baseline_image

    def _localized_earlier_root_block(
        self,
        *,
        failed_block: CommandBlock,
        failure_result: CommandResult,
        repo_context: RepoContext,
        baseline_image: str,
        completed_blocks: list[CommandBlock],
        executions: list[BlockExecution],
        all_blocks: list[CommandBlock] | None,
    ) -> CommandBlock | None:
        if not self.progress_control.checkpoint_rollback or all_blocks is None:
            return None
        if not completed_blocks:
            return None
        repair_context = self._repair_context(
            repo_context=repo_context,
            baseline_image=baseline_image,
            completed_blocks=completed_blocks,
            executions=executions,
            strategy_notes=[
                "Before repairing the failed block, localize whether an earlier "
                "block introduced the faulty state. Return the failed block id "
                "when evidence is weak.",
            ],
        )
        localization = self.repair_planner.localize_failure(
            failed_block,
            failure_result,
            context=repair_context,
        )
        if localization is None or localization.root_cause_block_id == failed_block.id:
            return None
        completed_by_id = {block.id: block for block in completed_blocks}
        root_block = completed_by_id.get(localization.root_cause_block_id)
        if root_block is None or not root_block.baseline_checkpoint:
            return None
        root_index = all_blocks.index(root_block)
        failed_index = all_blocks.index(failed_block)
        if root_index >= failed_index:
            return None
        self._emit(
            f"run {self.run_id}: block {failed_block.id} failure localized to "
            f"earlier block {root_block.id}; rollback to {root_block.baseline_checkpoint}"
        )
        return root_block

    def _run_earlier_block_recovery(
        self,
        *,
        runtime: DockerRuntime,
        failed_block: CommandBlock,
        root_block: CommandBlock,
        all_blocks: list[CommandBlock],
        repo_context: RepoContext,
        completed_blocks: list[CommandBlock],
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        failure_result: CommandResult,
    ) -> tuple[bool, str]:
        root_index = all_blocks.index(root_block)
        failed_index = all_blocks.index(failed_block)
        root_baseline = root_block.baseline_checkpoint
        if root_baseline is None:
            root_block.status = "failed"
            root_block.last_error = "earlier block recovery requires root baseline checkpoint"
            self.store.update_block(root_block)
            return False, failed_block.baseline_checkpoint or ""

        suffix = all_blocks[root_index : failed_index + 1]
        invalidated_ids = {block.id for block in suffix}
        checkpoints[:] = [
            checkpoint for checkpoint in checkpoints if checkpoint.block_id not in invalidated_ids
        ]
        for block in suffix:
            block.status = "planned"
            block.success_checkpoint = None
            block.last_error = None
            self.store.update_block(block)

        stable_prefix = all_blocks[:root_index]
        last_result = failure_result
        probe_failures = 0
        max_probe_failures = max(0, self.request.max_probe_failures)
        probe_disabled = max_probe_failures == 0
        for repair_attempt in range(1, self.request.max_repair_attempts + 1):
            repair_context = self._repair_context(
                repo_context=repo_context,
                baseline_image=root_baseline,
                completed_blocks=stable_prefix,
                executions=executions,
                strategy_notes=[
                    f"Failure appeared in {failed_block.id}, but localization selected "
                    f"{root_block.id} as the root-cause block. Repair the root block, "
                    "then replay and validate the invalidated suffix.",
                ],
            )
            probes: list[RepairProbeCommand] = []
            if not probe_disabled and probe_failures < max_probe_failures:
                probes = self.repair_planner.propose_probes(
                    root_block,
                    last_result,
                    context=repair_context,
                )
                self._record_llm_probe_status(
                    block=root_block,
                    attempt=repair_attempt,
                    baseline_image=root_baseline,
                    executions=executions,
                )
                probe_error = getattr(self.repair_planner, "last_probe_error", None)
                if probe_error:
                    probe_failures += 1
                    self._emit(
                        f"run {self.run_id}: root-cause block {root_block.id} "
                        f"LLM probe attempt {repair_attempt} failed "
                        f"({probe_failures}/{max_probe_failures}): "
                        f"{tail_text(probe_error, max_chars=500)}"
                    )
                    if (
                        probe_failures < max_probe_failures
                        and repair_attempt < self.request.max_repair_attempts
                    ):
                        continue
                    self._emit(
                        f"run {self.run_id}: root-cause block {root_block.id} "
                        "continue repair without probes"
                    )
                    probes = []
                elif (
                    not probes
                    and getattr(self.repair_planner, "last_probe_raw_response", None)
                    is not None
                ):
                    probe_disabled = True
                    self._emit(
                        f"run {self.run_id}: root-cause block {root_block.id} "
                        "LLM requested no probes; skip future probes"
                    )
            probe_results = self._run_repair_probes(
                runtime=runtime,
                block=root_block,
                baseline_image=root_baseline,
                attempt=repair_attempt,
                probes=probes,
                executions=executions,
            )
            if probe_results:
                repair_context = self._repair_context(
                    repo_context=repo_context,
                    baseline_image=root_baseline,
                    completed_blocks=stable_prefix,
                    executions=executions,
                    probe_results=probe_results,
                    strategy_notes=[
                        f"Failure appeared in {failed_block.id}, but localization selected "
                        f"{root_block.id} as the root-cause block. Repair the root block, "
                        "then replay and validate the invalidated suffix.",
                    ],
                )
            suggestions = self.repair_planner.suggest(
                root_block,
                last_result,
                context=repair_context,
            )
            self._record_llm_repair_status(
                block=root_block,
                attempt=repair_attempt,
                baseline_image=root_baseline,
                executions=executions,
                ok=bool(suggestions),
            )
            if not suggestions:
                error = getattr(self.repair_planner, "last_llm_error", None)
                if _should_continue_llm_repair_failure(error):
                    continue
                break

            repair = suggestions[min(repair_attempt - 1, len(suggestions) - 1)]
            self._emit(
                f"run {self.run_id}: root-cause block {root_block.id} repair "
                f"attempt {repair_attempt}"
            )
            self._recreate_from_checkpoint(runtime, root_baseline)
            repair_result = runtime.execute_command(
                repair.command,
                timeout=self.request.command_timeout,
            )
            execution = self._execution_from_result(
                block_id=root_block.id,
                phase="repair",
                attempt=repair_attempt,
                command_result=repair_result,
                checkpoint_before=root_baseline,
                repair_command=repair.command,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not repair_result.ok:
                last_result = repair_result
                continue

            if self.progress_control.patch_back:
                root_block = self.repair_planner.patch_block(root_block, repair)
                self.store.update_block(root_block)

            replay_ok, replay_image, replay_failed_block, replay_failure = (
                self._replay_block_suffix(
                    runtime=runtime,
                    blocks=suffix,
                    baseline_image=root_baseline,
                    repaired_root_id=root_block.id,
                    checkpoints=checkpoints,
                    executions=executions,
                    attempt=repair_attempt + 1,
                )
            )
            if replay_ok:
                return True, replay_image
            if (
                replay_failed_block is not None
                and replay_failed_block.id != root_block.id
                and replay_failure is not None
            ):
                return self._repair_replayed_suffix_failure(
                    runtime=runtime,
                    root_block=root_block,
                    failed_index=failed_index,
                    failed_block=replay_failed_block,
                    failure_result=replay_failure,
                    failure_baseline_image=replay_image,
                    all_blocks=all_blocks,
                    repo_context=repo_context,
                    checkpoints=checkpoints,
                    executions=executions,
                    attempt=repair_attempt + 1,
                )
            if replay_failure is not None:
                last_result = replay_failure

        failed_block.status = "failed"
        failed_block.last_error = tail_text(last_result.combined_output, max_chars=4000)
        self.store.update_block(failed_block)
        self._recreate_from_checkpoint(runtime, root_baseline)
        return False, root_baseline

    def _repair_replayed_suffix_failure(
        self,
        *,
        runtime: DockerRuntime,
        root_block: CommandBlock,
        failed_index: int,
        failed_block: CommandBlock,
        failure_result: CommandResult,
        failure_baseline_image: str,
        all_blocks: list[CommandBlock],
        repo_context: RepoContext,
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        attempt: int,
    ) -> tuple[bool, str]:
        current_failed_block = failed_block
        current_failure_result = failure_result
        current_image = failure_baseline_image
        replay_attempt = attempt

        while True:
            failed_suffix_index = all_blocks.index(current_failed_block)
            if failed_suffix_index > failed_index:
                current_failed_block.status = "failed"
                current_failed_block.last_error = tail_text(
                    current_failure_result.combined_output,
                    max_chars=4000,
                )
                self.store.update_block(current_failed_block)
                return False, current_image

            self._emit(
                f"run {self.run_id}: suffix replay failed at block "
                f"{current_failed_block.id}; repair that block instead of "
                f"root-cause block {root_block.id}"
            )
            block_ok, current_image = self._run_block_with_repair(
                runtime=runtime,
                block=current_failed_block,
                baseline_image=current_image,
                repo_context=repo_context,
                completed_blocks=all_blocks[:failed_suffix_index],
                checkpoints=checkpoints,
                executions=executions,
                all_blocks=all_blocks,
                workspace_image=None,
                allow_earlier_root_recovery=False,
            )
            if not block_ok:
                return False, current_image

            next_index = failed_suffix_index + 1
            if next_index > failed_index:
                return True, current_image

            replay_attempt += 1
            replay_ok, current_image, next_failed_block, next_failure_result = (
                self._replay_block_suffix(
                    runtime=runtime,
                    blocks=all_blocks[next_index : failed_index + 1],
                    baseline_image=current_image,
                    repaired_root_id="",
                    checkpoints=checkpoints,
                    executions=executions,
                    attempt=replay_attempt,
                )
            )
            if replay_ok:
                return True, current_image
            if next_failed_block is None or next_failure_result is None:
                return False, current_image
            current_failed_block = next_failed_block
            current_failure_result = next_failure_result

    def _replay_block_suffix(
        self,
        *,
        runtime: DockerRuntime,
        blocks: list[CommandBlock],
        baseline_image: str,
        repaired_root_id: str,
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        attempt: int,
    ) -> tuple[bool, str, CommandBlock | None, CommandResult | None]:
        current_image = baseline_image
        self._recreate_from_checkpoint(runtime, current_image)
        for block in blocks:
            block.baseline_checkpoint = current_image
            self.store.update_block(block)
            result = self._execute_block(runtime, block, attempt, current_image, executions)
            validation_result = self._validate_block(
                runtime,
                block,
                attempt,
                current_image,
                executions,
            )
            if result.ok and (validation_result is None or validation_result.ok):
                finalize_result = self._finalize_checkpoint_tools(
                    runtime=runtime,
                    block=block,
                    attempt=attempt,
                    baseline_image=current_image,
                    executions=executions,
                )
                if not finalize_result.ok:
                    block.status = "failed"
                    block.last_error = tail_text(finalize_result.combined_output, max_chars=4000)
                    self.store.update_block(block)
                    self._recreate_from_checkpoint(runtime, current_image)
                    return False, current_image, block, finalize_result
                checkpoint = runtime.commit(
                    block_id=block.id,
                    parent_image_ref=current_image,
                    kind="repaired" if block.id == repaired_root_id else "success",
                )
                checkpoints.append(checkpoint)
                block.success_checkpoint = checkpoint.image_ref
                block.status = "succeeded"
                self.store.update_block(block)
                current_image = checkpoint.image_ref
                continue
            if result.ok and validation_result is not None:
                result = validation_result
            block.status = "failed"
            block.last_error = tail_text(result.combined_output, max_chars=4000)
            self.store.update_block(block)
            self._recreate_from_checkpoint(runtime, current_image)
            return False, current_image, block, result
        return True, current_image, None, None

    def _run_whole_script_forward(
        self,
        *,
        runtime: DockerRuntime,
        blocks: list[CommandBlock],
        baseline_image: str,
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
    ) -> tuple[bool, str]:
        whole_script_path = self._write_whole_script(blocks)
        for block in blocks:
            block.baseline_checkpoint = baseline_image
            block.status = "running"
            self.store.update_block(block)

        self._emit(f"run {self.run_id}: whole-script forward")
        result = runtime.execute_script(whole_script_path, timeout=self.request.command_timeout)
        execution = self._execution_from_result(
            block_id="whole-script",
            phase="whole_script",
            attempt=1,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        if not result.ok:
            self._mark_whole_script_blocks_failed(blocks, result)
            return False, baseline_image

        for index, block in enumerate(blocks, start=1):
            validation_result = self._validate_block(
                runtime,
                block,
                index,
                baseline_image,
                executions,
            )
            if validation_result is not None and not validation_result.ok:
                block.status = "failed"
                block.last_error = tail_text(validation_result.combined_output, max_chars=4000)
                self.store.update_block(block)
                return False, baseline_image

        finalize_block = CommandBlock(
            id="whole-script",
            title="Whole Script",
            goal="finalize whole-script artifact",
            script=whole_script_path.read_text(encoding="utf-8"),
        )
        finalize_result = self._finalize_checkpoint_tools(
            runtime=runtime,
            block=finalize_block,
            attempt=1,
            baseline_image=baseline_image,
            executions=executions,
        )
        if not finalize_result.ok:
            self._mark_whole_script_blocks_failed(blocks, finalize_result)
            return False, baseline_image

        checkpoint = runtime.commit(
            block_id="whole-script",
            parent_image_ref=baseline_image,
            kind="success",
        )
        checkpoints.append(checkpoint)
        for block in blocks:
            block.success_checkpoint = checkpoint.image_ref
            block.status = "succeeded"
            self.store.update_block(block)
        self._emit(f"run {self.run_id}: whole-script checkpoint {checkpoint.image_ref}")
        return True, checkpoint.image_ref

    def _mark_whole_script_blocks_failed(
        self,
        blocks: list[CommandBlock],
        result: CommandResult,
    ) -> None:
        error = tail_text(result.combined_output, max_chars=4000)
        for block in blocks:
            block.status = "failed"
            block.last_error = error
            self.store.update_block(block)

    def _run_whole_script_recovery(
        self,
        *,
        runtime: DockerRuntime,
        failed_block: CommandBlock,
        all_blocks: list[CommandBlock],
        workspace_image: str,
        repo_context: RepoContext,
        completed_blocks: list[CommandBlock],
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        last_result: CommandResult,
    ) -> tuple[bool, str]:
        probe_failures = 0
        max_probe_failures = max(0, self.request.max_probe_failures)
        probe_disabled = max_probe_failures == 0
        strategy_notes = [
            "whole-script-recovery mode: do not repair only the failed block.",
            "Return a durable shell patch for a regenerated whole setup artifact. "
            "The orchestrator will replay the whole artifact from the workspace "
            "checkpoint, discarding already validated block progress.",
        ]
        for repair_attempt in range(1, self.request.max_repair_attempts + 1):
            repair_context = self._repair_context(
                repo_context=repo_context,
                baseline_image=workspace_image,
                completed_blocks=completed_blocks,
                executions=executions,
                strategy_notes=strategy_notes,
            )
            probes: list[RepairProbeCommand] = []
            if not probe_disabled and probe_failures < max_probe_failures:
                probes = self.repair_planner.propose_probes(
                    failed_block,
                    last_result,
                    context=repair_context,
                )
                self._record_llm_probe_status(
                    block=failed_block,
                    attempt=repair_attempt,
                    baseline_image=workspace_image,
                    executions=executions,
                )
                probe_error = getattr(self.repair_planner, "last_probe_error", None)
                if probe_error:
                    probe_failures += 1
                    self._emit(
                        f"run {self.run_id}: block {failed_block.id} LLM probe attempt "
                        f"{repair_attempt} failed ({probe_failures}/{max_probe_failures}): "
                        f"{tail_text(probe_error, max_chars=500)}"
                    )
                    if (
                        probe_failures < max_probe_failures
                        and repair_attempt < self.request.max_repair_attempts
                    ):
                        continue
                    self._emit(
                        f"run {self.run_id}: block {failed_block.id} continue repair "
                        "without probes"
                    )
                    probes = []
                elif (
                    not probes
                    and getattr(self.repair_planner, "last_probe_raw_response", None)
                    is not None
                ):
                    probe_disabled = True
                    self._emit(
                        f"run {self.run_id}: block {failed_block.id} LLM requested no "
                        "probes; skip future probes"
                    )
            probe_results = self._run_repair_probes(
                runtime=runtime,
                block=failed_block,
                baseline_image=workspace_image,
                attempt=repair_attempt,
                probes=probes,
                executions=executions,
            )
            if probe_results:
                repair_context = self._repair_context(
                    repo_context=repo_context,
                    baseline_image=workspace_image,
                    completed_blocks=completed_blocks,
                    executions=executions,
                    probe_results=probe_results,
                    strategy_notes=strategy_notes,
                )
            suggestions = self.repair_planner.suggest(
                failed_block,
                last_result,
                context=repair_context,
            )
            self._record_llm_repair_status(
                block=failed_block,
                attempt=repair_attempt,
                baseline_image=workspace_image,
                executions=executions,
                ok=bool(suggestions),
            )
            if not suggestions:
                error = getattr(self.repair_planner, "last_llm_error", None)
                if error:
                    self._emit(
                        f"run {self.run_id}: block {failed_block.id} LLM repair attempt "
                        f"{repair_attempt} failed: {tail_text(error, max_chars=500)}"
                    )
                if _should_continue_llm_repair_failure(error):
                    continue
                break

            repair = suggestions[min(repair_attempt - 1, len(suggestions) - 1)]
            self._emit(
                f"run {self.run_id}: block {failed_block.id} whole-script recovery "
                f"attempt {repair_attempt}"
            )
            whole_script_path = self._write_whole_script(
                all_blocks,
                repair=repair,
            )
            runtime.recreate_from(workspace_image)
            script_result = runtime.execute_script(
                whole_script_path,
                timeout=self.request.command_timeout,
            )
            execution = self._execution_from_result(
                block_id="whole-script",
                phase="whole_script_recovery",
                attempt=repair_attempt,
                command_result=script_result,
                checkpoint_before=workspace_image,
                repair_command=repair.patch_script,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not script_result.ok:
                last_result = script_result
                continue

            validation_ok = True
            for index, block in enumerate(all_blocks, start=1):
                validation_result = self._validate_block(
                    runtime,
                    block,
                    index,
                    workspace_image,
                    executions,
                )
                if validation_result is not None and not validation_result.ok:
                    last_result = validation_result
                    validation_ok = False
                    break
            if not validation_ok:
                continue

            finalize_result = runtime.execute_command(
                _CHECKPOINT_TOOL_EXPORT_COMMAND,
                timeout=min(self.request.command_timeout, 120.0),
            )
            finalize_execution = self._execution_from_result(
                block_id="whole-script",
                phase="whole_script_recovery_finalize",
                attempt=repair_attempt,
                command_result=finalize_result,
                checkpoint_before=workspace_image,
            )
            executions.append(finalize_execution)
            self.store.append_execution(finalize_execution)
            if not finalize_result.ok:
                last_result = finalize_result
                continue

            checkpoint = runtime.commit(
                block_id="whole-script",
                parent_image_ref=workspace_image,
                kind="repaired",
            )
            checkpoints.append(checkpoint)
            for block in all_blocks:
                block.success_checkpoint = checkpoint.image_ref
                block.status = "succeeded"
                self.store.update_block(block)
            self._whole_script_recovery_completed = True
            self._final_replay_whole_script = True
            self._emit(
                f"run {self.run_id}: whole-script recovery checkpoint "
                f"{checkpoint.image_ref}"
            )
            return True, checkpoint.image_ref

        self._mark_whole_script_blocks_failed(all_blocks, last_result)
        return False, workspace_image

    def _run_block_command_recovery(
        self,
        *,
        runtime: DockerRuntime,
        block: CommandBlock,
        baseline_image: str,
        repo_context: RepoContext,
        completed_blocks: list[CommandBlock],
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
        last_result: CommandResult,
    ) -> tuple[bool, str]:
        probe_failures = 0
        max_probe_failures = max(0, self.request.max_probe_failures)
        probe_disabled = max_probe_failures == 0
        strategy_notes = [
            "single-command-recovery mode: the failed block already ran in the "
            "current live container.",
            "Return a local recovery command for this live container state. "
            "The orchestrator will not roll back to the block checkpoint, will "
            "not patch the block script, and will not replay the block before "
            "validation.",
        ]
        for repair_attempt in range(1, self.request.max_repair_attempts + 1):
            repair_context = self._repair_context(
                repo_context=repo_context,
                baseline_image=baseline_image,
                completed_blocks=completed_blocks,
                executions=executions,
                strategy_notes=strategy_notes,
            )
            probes: list[RepairProbeCommand] = []
            if not probe_disabled and probe_failures < max_probe_failures:
                probes = self.repair_planner.propose_probes(
                    block,
                    last_result,
                    context=repair_context,
                )
                self._record_llm_probe_status(
                    block=block,
                    attempt=repair_attempt,
                    baseline_image=baseline_image,
                    executions=executions,
                )
                probe_error = getattr(self.repair_planner, "last_probe_error", None)
                if probe_error:
                    probe_failures += 1
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM probe attempt "
                        f"{repair_attempt} failed ({probe_failures}/{max_probe_failures}): "
                        f"{tail_text(probe_error, max_chars=500)}"
                    )
                    if (
                        probe_failures < max_probe_failures
                        and repair_attempt < self.request.max_repair_attempts
                    ):
                        continue
                    self._emit(
                        f"run {self.run_id}: block {block.id} continue repair without probes"
                    )
                    probes = []
                elif (
                    not probes
                    and getattr(self.repair_planner, "last_probe_raw_response", None)
                    is not None
                ):
                    probe_disabled = True
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM requested no probes; "
                        "skip future probes"
                    )
            probe_results = self._run_repair_probes(
                runtime=runtime,
                block=block,
                baseline_image=baseline_image,
                attempt=repair_attempt,
                probes=probes,
                executions=executions,
            )
            if probe_results:
                repair_context = self._repair_context(
                    repo_context=repo_context,
                    baseline_image=baseline_image,
                    completed_blocks=completed_blocks,
                    executions=executions,
                    probe_results=probe_results,
                    strategy_notes=strategy_notes,
                )
            suggestions = self.repair_planner.suggest(
                block,
                last_result,
                context=repair_context,
            )
            self._record_llm_repair_status(
                block=block,
                attempt=repair_attempt,
                baseline_image=baseline_image,
                executions=executions,
                ok=bool(suggestions),
            )
            if not suggestions:
                error = getattr(self.repair_planner, "last_llm_error", None)
                if error:
                    self._emit(
                        f"run {self.run_id}: block {block.id} LLM repair attempt "
                        f"{repair_attempt} failed: {tail_text(error, max_chars=500)}"
                    )
                if _should_continue_llm_repair_failure(error):
                    continue
                break

            repair = suggestions[min(repair_attempt - 1, len(suggestions) - 1)]
            self._emit(
                f"run {self.run_id}: block {block.id} command recovery attempt "
                f"{repair_attempt}"
            )
            repair_result = runtime.execute_command(
                repair.command,
                timeout=self.request.command_timeout,
            )
            execution = self._execution_from_result(
                block_id=block.id,
                phase="command_recovery",
                attempt=repair_attempt,
                command_result=repair_result,
                checkpoint_before=baseline_image,
                repair_command=repair.command,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not repair_result.ok:
                last_result = repair_result
                continue

            block.repair_attempts += 1
            block.status = "repaired"
            self.store.update_block(block)
            validation_result = self._validate_block(
                runtime,
                block,
                repair_attempt,
                baseline_image,
                executions,
            )
            if validation_result is not None and not validation_result.ok:
                last_result = validation_result
                continue

            finalize_result = self._finalize_checkpoint_tools(
                runtime=runtime,
                block=block,
                attempt=repair_attempt,
                baseline_image=baseline_image,
                executions=executions,
            )
            if not finalize_result.ok:
                block.status = "failed"
                block.last_error = tail_text(finalize_result.combined_output, max_chars=4000)
                self.store.update_block(block)
                return False, baseline_image

            checkpoint = runtime.commit(
                block_id=block.id,
                parent_image_ref=baseline_image,
                kind="repaired",
            )
            checkpoints.append(checkpoint)
            block.success_checkpoint = checkpoint.image_ref
            block.status = "succeeded"
            self.store.update_block(block)
            self._emit(
                f"run {self.run_id}: block {block.id} command-recovered checkpoint "
                f"{checkpoint.image_ref}"
            )
            return True, checkpoint.image_ref

        block.status = "failed"
        block.last_error = tail_text(last_result.combined_output, max_chars=4000)
        self.store.update_block(block)
        return False, baseline_image

    def _recreate_from_checkpoint(self, runtime: DockerRuntime, image_ref: str) -> None:
        if self.progress_control.checkpoint_rollback:
            runtime.recreate_from(image_ref)

    def _run_final_clean_replay(
        self,
        *,
        runtime: DockerRuntime,
        blocks: list[CommandBlock],
        workspace_image: str,
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
    ) -> tuple[bool, str | None, str | None]:
        blocks = self.store.write_blocks(blocks)
        runtime.recreate_from(workspace_image)
        if (
            self.progress_control.forward_granularity == "whole-script"
            or self._final_replay_whole_script
        ):
            return self._run_whole_script_clean_replay(
                runtime=runtime,
                blocks=blocks,
                workspace_image=workspace_image,
                checkpoints=checkpoints,
                executions=executions,
            )
        for index, block in enumerate(blocks, start=1):
            self._emit(f"run {self.run_id}: clean replay block {block.id}")
            script_result = runtime.execute_script(
                self.store.script_path(block.id),
                timeout=self.request.command_timeout,
            )
            execution = self._execution_from_result(
                block_id=block.id,
                phase="clean_replay",
                attempt=index,
                command_result=script_result,
                checkpoint_before=workspace_image,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not script_result.ok:
                return False, workspace_image, "clean_replay"

            if block.validation_command:
                validation_result = runtime.execute_command(
                    block.validation_command,
                    timeout=self.request.command_timeout,
                )
                validation_execution = self._execution_from_result(
                    block_id=block.id,
                    phase="clean_replay_validation",
                    attempt=index,
                    command_result=validation_result,
                    checkpoint_before=workspace_image,
                )
                executions.append(validation_execution)
                self.store.append_execution(validation_execution)
                if not validation_result.ok:
                    return False, workspace_image, "clean_replay_validation"

            finalize_result = runtime.execute_command(
                _CHECKPOINT_TOOL_EXPORT_COMMAND,
                timeout=min(self.request.command_timeout, 120.0),
            )
            finalize_execution = self._execution_from_result(
                block_id=block.id,
                phase="clean_replay_finalize",
                attempt=index,
                command_result=finalize_result,
                checkpoint_before=workspace_image,
            )
            executions.append(finalize_execution)
            self.store.append_execution(finalize_execution)
            if not finalize_result.ok:
                return False, workspace_image, "clean_replay_finalize"

        try:
            checkpoint = runtime.commit(
                block_id="final-clean-replay",
                parent_image_ref=workspace_image,
                kind="clean-replay",
            )
        except Exception:
            return False, workspace_image, "clean_replay_commit"
        checkpoints.append(checkpoint)
        return True, checkpoint.image_ref, None

    def _run_whole_script_clean_replay(
        self,
        *,
        runtime: DockerRuntime,
        blocks: list[CommandBlock],
        workspace_image: str,
        checkpoints: list[Checkpoint],
        executions: list[BlockExecution],
    ) -> tuple[bool, str | None, str | None]:
        whole_script_path = self.store.scripts_dir / "whole-setup.sh"
        if not whole_script_path.is_file() or not self._final_replay_whole_script:
            whole_script_path = self._write_whole_script(blocks)
        self._emit(f"run {self.run_id}: clean replay whole-script")
        script_result = runtime.execute_script(
            whole_script_path,
            timeout=self.request.command_timeout,
        )
        execution = self._execution_from_result(
            block_id="whole-script",
            phase="clean_replay",
            attempt=1,
            command_result=script_result,
            checkpoint_before=workspace_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        if not script_result.ok:
            return False, workspace_image, "clean_replay"

        for index, block in enumerate(blocks, start=1):
            if not block.validation_command:
                continue
            validation_result = runtime.execute_command(
                block.validation_command,
                timeout=self.request.command_timeout,
            )
            validation_execution = self._execution_from_result(
                block_id=block.id,
                phase="clean_replay_validation",
                attempt=index,
                command_result=validation_result,
                checkpoint_before=workspace_image,
            )
            executions.append(validation_execution)
            self.store.append_execution(validation_execution)
            if not validation_result.ok:
                return False, workspace_image, "clean_replay_validation"

        finalize_result = runtime.execute_command(
            _CHECKPOINT_TOOL_EXPORT_COMMAND,
            timeout=min(self.request.command_timeout, 120.0),
        )
        finalize_execution = self._execution_from_result(
            block_id="whole-script",
            phase="clean_replay_finalize",
            attempt=1,
            command_result=finalize_result,
            checkpoint_before=workspace_image,
        )
        executions.append(finalize_execution)
        self.store.append_execution(finalize_execution)
        if not finalize_result.ok:
            return False, workspace_image, "clean_replay_finalize"

        try:
            checkpoint = runtime.commit(
                block_id="final-clean-replay",
                parent_image_ref=workspace_image,
                kind="clean-replay",
            )
        except Exception:
            return False, workspace_image, "clean_replay_commit"
        checkpoints.append(checkpoint)
        return True, checkpoint.image_ref, None

    def _write_whole_script(
        self,
        blocks: list[CommandBlock],
        *,
        repair: RepairCommand | None = None,
    ) -> Path:
        script_path = self.store.scripts_dir / "whole-setup.sh"
        workdir = shlex.quote(self.request.container_workdir)
        lines = ["#!/bin/sh", "set -eu", f"cd {workdir}"]
        if repair is not None and repair.patch_script:
            lines.append("")
            lines.append("echo " + shlex.quote(f"[pheragent] repair: {repair.title}"))
            lines.extend(repair.patch_script.splitlines())
        for block in blocks:
            lines.append("")
            lines.append(f"cd {workdir}")
            lines.append(
                "echo "
                + shlex.quote(f"[pheragent] whole-script block {block.id}: {block.title}")
            )
            lines.extend(_shell_script_body_lines(block.script))
        script_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        return script_path

    def _repair_context(
        self,
        *,
        repo_context: RepoContext,
        baseline_image: str,
        completed_blocks: list[CommandBlock],
        executions: list[BlockExecution],
        probe_results: list[RepairProbeResult] | None = None,
        strategy_notes: list[str] | None = None,
    ) -> RepairContext:
        return RepairContext(
            repo_context=repo_context,
            checkpoint_before=baseline_image,
            previous_blocks=[
                block
                for block in completed_blocks
                if block.status in {"succeeded", "repaired", "skipped"}
            ],
            recent_executions=executions[-8:],
            probe_results=probe_results or [],
            strategy_notes=strategy_notes or [],
        )

    def _run_repair_probes(
        self,
        *,
        runtime: DockerRuntime,
        block: CommandBlock,
        baseline_image: str,
        attempt: int,
        probes: list[RepairProbeCommand],
        executions: list[BlockExecution],
    ) -> list[RepairProbeResult]:
        if not probes:
            return []
        self._emit(
            f"run {self.run_id}: block {block.id} probe attempt {attempt} "
            f"({len(probes)} commands)"
        )
        self._recreate_from_checkpoint(runtime, baseline_image)
        probe_results: list[RepairProbeResult] = []
        timeout = min(self.request.command_timeout, 60.0)
        for index, probe in enumerate(probes, start=1):
            self._emit(f"run {self.run_id}: block {block.id} probe {index}/{len(probes)}")
            command_result = runtime.execute_command(probe.command, timeout=timeout)
            execution = self._execution_from_result(
                block_id=block.id,
                phase="probe",
                attempt=attempt,
                command_result=command_result,
                checkpoint_before=baseline_image,
                repair_command=probe.command,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            probe_results.append(
                RepairProbeResult(
                    title=probe.title,
                    command=probe.command,
                    exit_code=command_result.exit_code,
                    timed_out=command_result.timed_out,
                    stdout_tail=tail_text(command_result.stdout, max_chars=4000),
                    stderr_tail=tail_text(command_result.stderr, max_chars=4000),
                    duration_s=command_result.duration_s,
                )
            )
        self._recreate_from_checkpoint(runtime, baseline_image)
        return probe_results

    def _execute_block(
        self,
        runtime: DockerRuntime,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
    ) -> CommandResult:
        block.status = "running"
        self.store.update_block(block)
        self._emit(f"run {self.run_id}: block {block.id} execute attempt {attempt}")
        if self.progress_control.forward_granularity == "command":
            return self._execute_block_commands(
                runtime=runtime,
                block=block,
                attempt=attempt,
                baseline_image=baseline_image,
                executions=executions,
            )
        result = runtime.execute_script(
            self.store.script_path(block.id),
            timeout=self.request.command_timeout,
        )
        execution = self._execution_from_result(
            block_id=block.id,
            phase="block",
            attempt=attempt,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        return result

    def _execute_block_commands(
        self,
        *,
        runtime: DockerRuntime,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
    ) -> CommandResult:
        script_path = self.store.script_path(block.id)
        commands = _split_shell_script_commands(script_path.read_text(encoding="utf-8"))
        if not commands:
            return CommandResult(
                exit_code=0,
                stdout="[pheragent] command forward: no commands\n",
                command=["pheragent-command-forward", block.id],
            )
        stdout_parts: list[str] = []
        command_results = runtime.execute_command_sequence(
            commands,
            timeout=self.request.command_timeout,
        )
        for index, (command, result) in enumerate(
            zip(commands, command_results, strict=False),
            start=1,
        ):
            self._emit(
                f"run {self.run_id}: block {block.id} command {index}/{len(commands)}"
            )
            if not result.command:
                result.command = ["pheragent-command-forward-persistent", str(index), command]
            execution = self._execution_from_result(
                block_id=block.id,
                phase="command_forward",
                attempt=(attempt * 1000) + index,
                command_result=result,
                checkpoint_before=baseline_image,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            stdout_parts.append(f"[{index}/{len(commands)}] {command}\n{result.stdout}")
            if not result.ok:
                return result
        if len(command_results) < len(commands):
            return CommandResult(
                exit_code=1,
                stderr=(
                    "[pheragent] persistent command forward ended before all commands "
                    f"completed ({len(command_results)}/{len(commands)})"
                ),
                command=["pheragent-command-forward-persistent", block.id],
            )
        return CommandResult(
            exit_code=0,
            stdout="\n".join(stdout_parts),
            command=["pheragent-command-forward-persistent", block.id],
        )

    def _finalize_checkpoint_tools(
        self,
        *,
        runtime: DockerRuntime,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
    ) -> CommandResult:
        self._emit(f"run {self.run_id}: block {block.id} finalize")
        result = runtime.execute_command(
            _CHECKPOINT_TOOL_EXPORT_COMMAND,
            timeout=min(self.request.command_timeout, 120.0),
        )
        execution = self._execution_from_result(
            block_id=block.id,
            phase="finalize",
            attempt=attempt,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        return result

    def _validate_block(
        self,
        runtime: DockerRuntime,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
    ) -> CommandResult | None:
        if not block.validation_command:
            return None
        self._emit(f"run {self.run_id}: block {block.id} validate")
        result = runtime.execute_command(
            block.validation_command,
            timeout=self.request.command_timeout,
        )
        execution = self._execution_from_result(
            block_id=block.id,
            phase="validation",
            attempt=attempt,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)
        return result

    def _run_oracle_validation(
        self,
        *,
        runtime: DockerRuntime,
        checkpoint_image: str,
        executions: list[BlockExecution],
    ) -> bool:
        if self.request.oracle_file is None:
            return True
        commands = load_oracle_commands(self.request.oracle_file)
        if not commands:
            raise ValueError(
                f"oracle file did not contain test commands: {self.request.oracle_file}"
            )
        timeout = self.request.oracle_timeout or self.request.command_timeout
        for index, command in enumerate(commands, start=1):
            self._emit(f"run {self.run_id}: oracle command {index}/{len(commands)}")
            result = runtime.execute_command(command, timeout=timeout)
            execution = self._execution_from_result(
                block_id="oracle",
                phase="oracle",
                attempt=index,
                command_result=result,
                checkpoint_before=checkpoint_image,
            )
            executions.append(execution)
            self.store.append_execution(execution)
            if not result.ok:
                return False
        return True

    def _record_llm_repair_status(
        self,
        *,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
        ok: bool = False,
    ) -> None:
        error = getattr(self.repair_planner, "last_llm_error", None)
        raw_response = getattr(self.repair_planner, "last_llm_raw_response", None)
        diagnostics = getattr(self.repair_planner, "last_llm_parse_diagnostics", [])
        stdout = _llm_repair_debug_output(raw_response, diagnostics)
        if not error and not stdout:
            return
        result = CommandResult(
            exit_code=0 if ok and not error else 1,
            stdout=stdout,
            stderr=error or "",
            command=["llm-repair", block.id],
        )
        execution = self._execution_from_result(
            block_id=block.id,
            phase="llm_repair",
            attempt=attempt,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)

    def _record_llm_probe_status(
        self,
        *,
        block: CommandBlock,
        attempt: int,
        baseline_image: str,
        executions: list[BlockExecution],
    ) -> None:
        error = getattr(self.repair_planner, "last_probe_error", None)
        raw_response = getattr(self.repair_planner, "last_probe_raw_response", None)
        diagnostics = getattr(self.repair_planner, "last_probe_parse_diagnostics", [])
        stdout = _llm_repair_debug_output(raw_response, diagnostics)
        if not error and not stdout:
            return
        result = CommandResult(
            exit_code=0 if not error else 1,
            stdout=stdout,
            stderr=error or "",
            command=["llm-probe", block.id],
        )
        execution = self._execution_from_result(
            block_id=block.id,
            phase="llm_probe",
            attempt=attempt,
            command_result=result,
            checkpoint_before=baseline_image,
        )
        executions.append(execution)
        self.store.append_execution(execution)

    def _execution_from_result(
        self,
        *,
        block_id: str,
        phase: str,
        attempt: int,
        command_result: CommandResult,
        checkpoint_before: str | None,
        checkpoint_after: str | None = None,
        repair_command: str | None = None,
    ) -> BlockExecution:
        log_path = self.store.write_execution_log(
            block_id=block_id,
            phase=phase,
            attempt=attempt,
            command_result=command_result,
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            repair_command=repair_command,
        )
        return BlockExecution(
            block_id=block_id,
            phase=phase,
            attempt=attempt,
            exit_code=command_result.exit_code,
            timed_out=command_result.timed_out,
            stdout_tail=tail_text(command_result.stdout),
            stderr_tail=tail_text(command_result.stderr),
            checkpoint_before=checkpoint_before,
            checkpoint_after=checkpoint_after,
            repair_command=repair_command,
            duration_s=command_result.duration_s,
            command=command_result.command,
            log_path=str(log_path),
        )

    def _emit(self, message: str) -> None:
        if self.request.stream_logs:
            print(f"[pheragent] {message}", file=sys.stderr, flush=True)


_INHERITED_PRELUDE_BEGIN = "# [pheragent] inherited environment prelude begin"
_INHERITED_PRELUDE_END = "# [pheragent] inherited environment prelude end"


def _ensure_inherited_block_preludes(
    blocks: list[CommandBlock],
    *,
    workdir: str,
    context: RepoContext,
) -> list[CommandBlock]:
    return [
        _ensure_inherited_block_prelude(block, workdir=workdir, context=context)
        for block in blocks
    ]


def _ensure_inherited_block_prelude(
    block: CommandBlock,
    *,
    workdir: str,
    context: RepoContext,
) -> CommandBlock:
    del context
    block.script = normalize_posix_source(block.script)
    if _INHERITED_PRELUDE_BEGIN in block.script:
        return block
    block.script = _prepend_inherited_prelude(block.script, workdir=workdir)
    return block


def _prepend_inherited_prelude(script: str, *, workdir: str) -> str:
    prelude = _inherited_environment_prelude(workdir)
    body_lines = _shell_script_body_lines(script)
    return "#!/bin/sh\nset -eu\n\n" + prelude + "\n\n" + "\n".join(body_lines).strip() + "\n"


def _inherited_environment_prelude(workdir: str) -> str:
    quoted_workdir = shlex.quote(workdir)
    return f"""
{_INHERITED_PRELUDE_BEGIN}
PHERAGENT_WORKDIR={quoted_workdir}
cd "$PHERAGENT_WORKDIR"
mkdir -p "$PHERAGENT_WORKDIR/.cache/pip" "$PHERAGENT_WORKDIR/.cache/uv"
export PIP_CACHE_DIR="$PHERAGENT_WORKDIR/.cache/pip"
export UV_CACHE_DIR="$PHERAGENT_WORKDIR/.cache/uv"
PHERAGENT_PATH_PREFIX="$PHERAGENT_WORKDIR/.venv/bin:$PHERAGENT_WORKDIR/.pheragent-tools/bin"
PHERAGENT_PATH_PREFIX="$PHERAGENT_PATH_PREFIX:$PHERAGENT_WORKDIR/node_modules/.bin"
export PATH="$PHERAGENT_PATH_PREFIX:/usr/local/bin:$PATH"

if [ -x "$PHERAGENT_WORKDIR/.venv/bin/python" ]; then
  ln -sf "$PHERAGENT_WORKDIR/.venv/bin/python" /usr/local/bin/python || true
  ln -sf "$PHERAGENT_WORKDIR/.venv/bin/python" /usr/local/bin/python3 || true
fi
if [ -x "$PHERAGENT_WORKDIR/.venv/bin/pip" ]; then
  ln -sf "$PHERAGENT_WORKDIR/.venv/bin/pip" /usr/local/bin/pip || true
  ln -sf "$PHERAGENT_WORKDIR/.venv/bin/pip" /usr/local/bin/pip3 || true
fi
if [ -x "$PHERAGENT_WORKDIR/.venv/bin/pytest" ]; then
  ln -sf "$PHERAGENT_WORKDIR/.venv/bin/pytest" /usr/local/bin/pytest || true
fi
if [ -x "$PHERAGENT_WORKDIR/.pheragent-tools/bin/uv" ]; then
  ln -sf "$PHERAGENT_WORKDIR/.pheragent-tools/bin/uv" /usr/local/bin/uv || true
fi
{_INHERITED_PRELUDE_END}
""".strip()


def _split_shell_script_commands(script: str) -> list[str]:
    lines = _shell_script_body_lines(script)
    commands: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue

        chunk, index = _take_shell_command_chunk(lines, index)
        command = "\n".join(chunk).strip()
        if command:
            commands.append(command)

    return commands


def _shell_script_body_lines(script: str) -> list[str]:
    body: list[str] = []
    for raw_line in script.splitlines():
        stripped = raw_line.strip()
        if stripped.startswith("#!"):
            continue
        if stripped in {"set -e", "set -eu", "set -eux", "set -ex"}:
            continue
        if stripped == "set -o pipefail":
            continue
        body.append(raw_line)
    return body


def _take_shell_command_chunk(lines: list[str], start_index: int) -> tuple[list[str], int]:
    chunk: list[str] = []
    depth = 0
    index = start_index
    while index < len(lines):
        line = lines[index]
        chunk.append(line)
        depth += _shell_compound_depth_delta(line)
        delimiter = _heredoc_delimiter(line)
        if delimiter is not None:
            index += 1
            while index < len(lines):
                heredoc_line = lines[index]
                chunk.append(heredoc_line)
                index += 1
                if heredoc_line.strip() == delimiter:
                    break
            if depth <= 0 and not _shell_line_continues(line):
                break
            continue

        index += 1
        if depth <= 0 and not _shell_line_continues(line):
            break
    return chunk, index


def _heredoc_delimiter(line: str) -> str | None:
    match = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", line)
    if not match:
        return None
    return match.group(1)


def _shell_line_continues(line: str) -> bool:
    stripped = line.rstrip()
    return stripped.endswith(("\\", "&&", "||", "|"))


def _shell_compound_depth_delta(line: str) -> int:
    clean = line.strip()
    if not clean or clean.startswith("#"):
        return 0
    delta = 0
    for match in re.finditer(r"\b(if|for|while|until|case|fi|done|esac)\b|[{}]", clean):
        word = match.group(1) or match.group(0)
        if word in {"if", "for", "while", "until", "case", "{"}:
            delta += 1
        elif word in {"fi", "done", "esac", "}"}:
            delta -= 1
    return delta


def _is_shell_context_command(command: str) -> bool:
    stripped = command.strip()
    if _is_shell_function_definition(stripped):
        return True
    if "\n" in stripped:
        return False
    return (
        _is_shell_assignment(stripped)
        or _is_shell_cd(stripped)
        or _is_shell_source(stripped)
    )


def _is_shell_function_definition(command: str) -> bool:
    lines = [
        line.strip()
        for line in command.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not lines:
        return False
    function_start = re.match(
        r"^(?:function\s+)?[A-Za-z_][A-Za-z0-9_]*(?:\s*\(\))?\s*\{",
        lines[0],
    )
    return bool(function_start) and command.rstrip().endswith("}")


def _is_shell_assignment(command: str) -> bool:
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    if not tokens:
        return False
    if tokens[0] == "export":
        return len(tokens) > 1 and all(_is_assignment_token(token) for token in tokens[1:])
    return all(_is_assignment_token(token) for token in tokens)


def _is_assignment_token(token: str) -> bool:
    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", token) is not None


def _is_shell_cd(command: str) -> bool:
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    return bool(tokens) and tokens[0] == "cd" and all(
        operator not in command for operator in ("&&", "||", "|", ";")
    )


def _is_shell_source(command: str) -> bool:
    try:
        tokens = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return False
    return bool(tokens) and tokens[0] in {".", "source"} and all(
        operator not in command for operator in ("&&", "||", "|", ";")
    )


def _llm_repair_debug_output(raw_response: object, diagnostics: object) -> str:
    parts: list[str] = []
    if isinstance(raw_response, str):
        parts.append("--- raw_llm_response ---\n" + raw_response)
    if diagnostics:
        if isinstance(diagnostics, str):
            diagnostic_lines = [diagnostics]
        else:
            try:
                diagnostic_lines = [str(item) for item in diagnostics]
            except TypeError:
                diagnostic_lines = [str(diagnostics)]
        parts.append("--- parse_diagnostics ---\n" + "\n".join(diagnostic_lines))
    return "\n\n".join(parts)


_CONTAINER_PREFLIGHT_COMMAND = r"""
echo "[pheragent] container preflight"
printf 'workdir=%s\n' "$(pwd -P)" || true
printf 'user=%s uid=%s gid=%s\n' \
  "$(id -un 2>/dev/null || true)" \
  "$(id -u 2>/dev/null || true)" \
  "$(id -g 2>/dev/null || true)" || true
uname -a || true
cat /etc/os-release 2>/dev/null || true

for cmd in \
  sh bash python python3 pip pip3 uv gcc g++ make cmake ninja bazel node npm \
  java go rustc cargo nvidia-smi nvcc apt-get yum apk
do
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'tool:%s=%s\n' "$cmd" "$(command -v "$cmd")"
    "$cmd" --version 2>&1 | sed -n '1,2p' || true
  else
    printf 'tool:%s=missing\n' "$cmd"
  fi
done

if command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3 2>/dev/null || command -v python)"
  "$PYTHON_BIN" - <<'PY' || true
import os
import platform
import sys
print(f"python.executable={sys.executable}")
print(f"python.version={sys.version}")
print(f"python.platform={platform.platform()}")
print(f"python.prefix={sys.prefix}")
print(f"python.base_prefix={getattr(sys, 'base_prefix', '')}")
print(f"env.PATH={os.environ.get('PATH', '')}")
PY
fi

printf '[pheragent] repo markers\n'
find . -maxdepth 2 -type f \
  \( -name 'pyproject.toml' \
     -o -name 'setup.py' \
     -o -name 'setup.cfg' \
     -o -name 'requirements*.txt' \
     -o -name 'uv.lock' \
     -o -name 'WORKSPACE' \
     -o -name 'WORKSPACE.bazel' \
     -o -name 'MODULE.bazel' \
     -o -name '.bazelrc' \
     -o -name 'package.json' \
     -o -name 'go.mod' \
     -o -name 'Cargo.toml' \) \
  | sort | sed -n '1,120p' || true

exit 0
""".strip()


def _runtime_notes_from_preflight(result: CommandResult) -> list[str]:
    notes: list[str] = []
    if result.timed_out:
        notes.append("container preflight timed out")
    if not result.ok:
        notes.append(f"container preflight exit_code={result.exit_code}")
    output = tail_text(result.combined_output, max_chars=12000)
    for line in output.splitlines():
        clean = line.strip()
        if not clean:
            continue
        notes.append(clean[:300])
        if len(notes) >= 160:
            notes.append("container preflight output truncated")
            break
    return notes


_CHECKPOINT_TOOL_EXPORT_COMMAND = """
repo_dir="$(pwd -P)"
if ! command -v uv >/dev/null 2>&1; then
  for rel in \
    .pheragent/uv-bootstrap/bin/uv \
    .pheragent-tools/bin/uv \
    .venv/bin/uv
  do
    candidate="$repo_dir/$rel"
    if [ -x "$candidate" ]; then
      if [ -w /usr/local/bin ] || [ "$(id -u)" = "0" ]; then
        ln -sf "$candidate" /usr/local/bin/uv
        echo "[pheragent] exported uv to /usr/local/bin/uv"
        exit 0
      fi
      echo "[pheragent] uv found at $candidate but /usr/local/bin is not writable" >&2
      exit 1
    fi
  done
fi
exit 0
""".strip()
