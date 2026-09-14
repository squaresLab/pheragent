from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_deployment_execution import _write_workflow

from pheragent.deployment.analysis_llm import ClassificationOutcome
from pheragent.deployment.execution import CommandOutcome, prepare_execution
from pheragent.deployment.probes import ProbeRequest, ProbeRunner
from pheragent.deployment.recovery import (
    PatchSandbox,
    RecoveryAgent,
    RecoveryDecision,
    RecoveryDecisionStatus,
    RecoveryFailure,
    RecoveryRisk,
    RecoveryScope,
    RecoveryStatus,
    _unsafe_patch_reason,
)


def test_failure_bundle_is_compact_and_redacted(tmp_path: Path) -> None:
    workflow, source = _write_workflow(tmp_path)
    prepared = prepare_execution(workflow, {"fixture": source})
    output = tmp_path / "execution"
    final_error = "Error: repo bitnami not found"

    report = prepared.execute(
        approval_token=prepared.approval_token,
        timeout=10,
        output_directory=output,
        command_runner=lambda *_args: CommandOutcome.failed(
            1,
            "token=unsafe\n" + ("noisy output\n" * 3000) + final_error,
        ),
    )

    bundle_path = output / "failure-bundle.json"
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    rendered = bundle_path.read_text(encoding="utf-8")
    assert report.failed
    assert len(rendered) < 15_000
    assert "unsafe" not in rendered
    assert "[REDACTED]" in rendered
    assert final_error in payload["failures"][0]["output_excerpt"]
    assert (output / "logs" / "S001-attempt-1.log").is_file()


def test_probe_runner_uses_fixed_read_only_command(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def run(arguments: list[str], _cwd: Path, _timeout: float) -> tuple[int, str]:
        calls.append(arguments)
        return 0, "mosip https://example.invalid/charts"

    runner = ProbeRunner(command_runner=run)
    result = runner.run(ProbeRequest(name="helm.repo_list"), root=tmp_path)

    assert calls == [["helm", "repo", "list"]]
    assert result.succeeded is True
    assert result.output.startswith("mosip")


def test_probe_runner_rejects_file_outside_source_root(tmp_path: Path) -> None:
    runner = ProbeRunner()

    result = runner.run(
        ProbeRequest(name="shell.file_excerpt", path="../private-key"),
        root=tmp_path,
    )

    assert result.succeeded is False
    assert "outside source root" in result.error


def test_patch_sandbox_validates_without_changing_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text("#!/bin/bash\nprintf old\\n\n", encoding="utf-8")
    patch = """\
--- a/install.sh
+++ b/install.sh
@@ -1,2 +1,2 @@
 #!/bin/bash
-printf old\\n
+printf new\\n
"""

    validation = PatchSandbox(tmp_path / "sandboxes").validate(source, patch)

    assert validation.succeeded is True
    assert script.read_text(encoding="utf-8") == "#!/bin/bash\nprintf old\\n\n"
    assert (validation.workspace / "install.sh").read_text(encoding="utf-8") == (
        "#!/bin/bash\nprintf new\\n\n"
    )


@pytest.mark.parametrize(
    "added_line",
    ["sudo apt install postgresql", "kubectl apply -f manifest.yaml", "curl example.invalid"],
)
def test_automatic_repair_rejects_high_impact_commands(added_line: str) -> None:
    patch = f"--- a/install.sh\n+++ b/install.sh\n@@ -1 +1 @@\n-old\n+{added_line}\n"

    assert _unsafe_patch_reason(patch) is not None


def test_recovery_agent_returns_only_a_sandbox_validated_fix(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    script = source / "install.sh"
    script.write_text("#!/bin/bash\nprintf old\\n\n", encoding="utf-8")
    patch = """\
--- a/install.sh
+++ b/install.sh
@@ -1,2 +1,2 @@
 #!/bin/bash
-printf old\\n
+printf new\\n
"""
    decision = RecoveryDecision(
        status=RecoveryDecisionStatus.PROPOSED_FIX,
        scope=RecoveryScope.COMPONENT,
        root_cause="The installer uses the old command.",
        patch=patch,
        risk=RecoveryRisk.LOW,
        requires_human_approval=False,
    )

    class Classifier:
        def classify(self, **_kwargs: object) -> ClassificationOutcome[RecoveryDecision]:
            return ClassificationOutcome(
                decision,
                "deployment_recovery",
                "llm",
                {"input_tokens": 100, "output_tokens": 20, "requests": 1},
                90,
                duration_seconds=0.25,
            )

    agent = RecoveryAgent(
        classifier=Classifier(),
        probe_runner=ProbeRunner(command_runner=lambda *_args: (0, "ok")),
        sandbox=PatchSandbox(tmp_path / "sandboxes"),
        model="test-model",
    )
    failure = RecoveryFailure(
        id="S001-attempt-1",
        step_id="S001",
        block_id="B7",
        component_ids=("C001_postgresql",),
        executor="shell",
        command="./install.sh",
        repo_id="fixture",
        source_path="install.sh",
        source_root=source,
        working_directory=source,
        attempt=1,
        exit_code=1,
        timed_out=False,
        duration_seconds=1.0,
        output_excerpt="old command failed",
    )

    resolution = agent.resolve(failure)

    assert resolution.status == RecoveryStatus.RESOLVED
    assert resolution.validation is not None
    assert resolution.validation.succeeded
    assert resolution.workspace_root is not None
    assert (resolution.workspace_root / "install.sh").read_text(encoding="utf-8").endswith(
        "printf new\\n\n"
    )
    assert resolution.llm_calls == (
        {
            "stage": "deployment_recovery",
            "status": "llm",
            "usage": {"input_tokens": 100, "output_tokens": 20, "requests": 1},
            "input_tokens_estimate": 90,
            "duration_seconds": 0.25,
        },
    )
