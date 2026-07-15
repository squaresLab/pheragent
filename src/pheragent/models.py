from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

BlockStatus = Literal["planned", "running", "succeeded", "failed", "repaired", "skipped"]
PlannerMode = Literal["auto", "rules", "llm"]
LLMApiMode = Literal["responses", "chat-completions"]
AblationMode = Literal[
    "full",
    "without-local-repair",
    "without-checkpoint-rollback",
    "without-final-clean-replay",
    "single-command-forward",
    "single-command-recovery",
    "whole-script-forward",
    "whole-script-recovery",
]
DEFAULT_ABLATION_MODE: AblationMode = "full"


@dataclass(frozen=True, slots=True)
class ProgressControl:
    forward_granularity: Literal["block", "command", "whole-script"] = "block"
    recovery_granularity: Literal["block", "command", "whole-script", "none"] = "block"
    local_repair: bool = True
    patch_back: bool = True
    checkpoint_rollback: bool = True
    final_clean_replay: bool = False


def progress_control_for_ablation(mode: AblationMode) -> ProgressControl:
    if mode == "full":
        return ProgressControl(final_clean_replay=True)
    if mode == "without-local-repair":
        return ProgressControl(
            recovery_granularity="none",
            local_repair=False,
            patch_back=False,
            final_clean_replay=True,
        )
    if mode == "without-checkpoint-rollback":
        return ProgressControl(checkpoint_rollback=False, final_clean_replay=True)
    if mode == "without-final-clean-replay":
        return ProgressControl(final_clean_replay=False)
    if mode == "single-command-forward":
        return ProgressControl(forward_granularity="command", final_clean_replay=True)
    if mode == "single-command-recovery":
        return ProgressControl(
            recovery_granularity="command",
            patch_back=False,
            checkpoint_rollback=False,
            final_clean_replay=True,
        )
    if mode == "whole-script-forward":
        return ProgressControl(
            forward_granularity="whole-script",
            recovery_granularity="whole-script",
            local_repair=False,
            patch_back=False,
            checkpoint_rollback=False,
            final_clean_replay=True,
        )
    if mode == "whole-script-recovery":
        return ProgressControl(
            recovery_granularity="whole-script",
            patch_back=False,
            final_clean_replay=True,
        )
    raise ValueError(f"unsupported ablation mode: {mode}")


def _normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


@dataclass(slots=True)
class BuildRequest:
    repo_path: Path
    base_dockerfile: Path | None = None
    state_dir: Path | None = None
    run_id: str | None = None
    task_description: str | None = None
    container_workdir: str = "/workspace/repo"
    image_prefix: str = "pheragent"
    max_repair_attempts: int = 2
    max_probe_failures: int = 5
    command_timeout: float = 900.0
    docker_build_timeout: float = 1800.0
    keep_container: bool = False
    cleanup_images: bool = False
    stream_logs: bool = False
    planner_mode: PlannerMode = "auto"
    llm_api: LLMApiMode = "responses"
    llm_model: str | None = None
    openai_base_url: str | None = None
    openai_api_key_env: str = "OPENAI_API_KEY"
    openai_base_url_env: str = "OPENAI_BASE_URL"
    llm_timeout: float = 120.0
    llm_max_tokens: int = 4096
    llm_retries: int = 3
    llm_retry_delay: float = 1.0
    oracle_file: Path | None = None
    oracle_timeout: float | None = None
    resume_from: str | None = None
    start_at_block: str | None = None
    ablation_mode: AblationMode = DEFAULT_ABLATION_MODE

    def normalized(self) -> BuildRequest:
        repo_path = self.repo_path.expanduser().resolve()
        base_dockerfile = (
            None if self.base_dockerfile is None else self.base_dockerfile.expanduser().resolve()
        )
        state_dir = self.state_dir
        oracle_file = None if self.oracle_file is None else self.oracle_file.expanduser().resolve()
        if state_dir is None:
            state_dir = repo_path / ".pheragent"
        return BuildRequest(
            repo_path=repo_path,
            base_dockerfile=base_dockerfile,
            state_dir=state_dir.expanduser().resolve(),
            run_id=self.run_id,
            task_description=_normalize_optional_text(self.task_description),
            container_workdir=self.container_workdir,
            image_prefix=self.image_prefix,
            max_repair_attempts=self.max_repair_attempts,
            max_probe_failures=self.max_probe_failures,
            command_timeout=self.command_timeout,
            docker_build_timeout=self.docker_build_timeout,
            keep_container=self.keep_container,
            cleanup_images=self.cleanup_images,
            stream_logs=self.stream_logs,
            planner_mode=self.planner_mode,
            llm_api=self.llm_api,
            llm_model=self.llm_model,
            openai_base_url=self.openai_base_url,
            openai_api_key_env=self.openai_api_key_env,
            openai_base_url_env=self.openai_base_url_env,
            llm_timeout=self.llm_timeout,
            llm_max_tokens=self.llm_max_tokens,
            llm_retries=self.llm_retries,
            llm_retry_delay=self.llm_retry_delay,
            oracle_file=oracle_file,
            oracle_timeout=self.oracle_timeout,
            resume_from=self.resume_from,
            start_at_block=self.start_at_block,
            ablation_mode=self.ablation_mode,
        )


@dataclass(slots=True)
class SetupLocation:
    path: str
    language: str
    package_files: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RepoContext:
    repo_path: Path
    task_description: str | None = None
    package_files: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    test_commands: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    runtime_notes: list[str] = field(default_factory=list)
    setup_locations: list[SetupLocation] = field(default_factory=list)


@dataclass(slots=True)
class CommandBlock:
    id: str
    title: str
    goal: str
    script: str
    order: int = 0
    validation_command: str | None = None
    status: BlockStatus = "planned"
    last_error: str | None = None
    baseline_checkpoint: str | None = None
    success_checkpoint: str | None = None
    repair_attempts: int = 0


@dataclass(slots=True)
class CommandResult:
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_s: float = 0.0
    command: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def combined_output(self) -> str:
        if self.stderr:
            return f"{self.stdout}\n{self.stderr}".strip()
        return self.stdout


@dataclass(slots=True)
class Checkpoint:
    id: str
    image_ref: str
    block_id: str | None
    parent_image_ref: str | None
    kind: str


@dataclass(slots=True)
class BlockExecution:
    block_id: str
    phase: str
    attempt: int
    exit_code: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    checkpoint_before: str | None = None
    checkpoint_after: str | None = None
    repair_command: str | None = None
    duration_s: float = 0.0
    command: list[str] = field(default_factory=list)
    log_path: str | None = None


@dataclass(slots=True)
class RepairProbeResult:
    title: str
    command: str
    exit_code: int | None
    timed_out: bool
    stdout_tail: str
    stderr_tail: str
    duration_s: float = 0.0


@dataclass(slots=True)
class RepairContext:
    repo_context: RepoContext
    checkpoint_before: str | None
    previous_blocks: list[CommandBlock] = field(default_factory=list)
    recent_executions: list[BlockExecution] = field(default_factory=list)
    probe_results: list[RepairProbeResult] = field(default_factory=list)
    strategy_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BuildResult:
    ok: bool
    run_id: str
    state_dir: Path
    scripts_dir: Path
    manifest_path: Path
    llm_usage_path: Path | None = None
    blocks: list[CommandBlock] = field(default_factory=list)
    checkpoints: list[Checkpoint] = field(default_factory=list)
    executions: list[BlockExecution] = field(default_factory=list)
    final_image: str | None = None
    error: str | None = None
    llm_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    ablation_mode: AblationMode = DEFAULT_ABLATION_MODE
    progress_control: ProgressControl | None = None
    final_clean_replay_enabled: bool = False
    final_clean_replay_ok: bool | None = None
    final_clean_replay_image: str | None = None
    final_clean_replay_failure_stage: str | None = None


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    return value
