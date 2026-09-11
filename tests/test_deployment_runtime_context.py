from __future__ import annotations

import json
from pathlib import Path

from pheragent.deployment.runtime_context import (
    RuntimeInspectionConfig,
    compact_runtime_context,
    inspect_runtime_context,
    load_runtime_context,
)
from pheragent.deployment.serialization import write_json
from pheragent.models import CommandResult


def test_runtime_inspection_uses_fixed_read_only_probes(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], *, timeout: float) -> CommandResult:
        commands.append(command)
        if "get-caller-identity" in command:
            payload = {"Account": "123", "Arn": "arn:aws:iam::123:user/test"}
        elif "describe-instances" in command:
            payload = {
                "Reservations": [{"Instances": [{
                    "InstanceId": "i-1",
                    "State": {"Name": "running"},
                    "Tags": [{"Key": "Name", "Value": "worker-1"}],
                }]}]
            }
        elif "version" in command:
            payload = {"serverVersion": {"gitVersion": "v1.36.3"}}
        elif "namespaces" in command:
            payload = {"items": [{"kind": "Namespace", "metadata": {"name": "keycloak"}}]}
        elif "deployments,statefulsets,daemonsets,jobs,cronjobs" in command:
            payload = {"items": [{
                "kind": "Deployment",
                "metadata": {"name": "keycloak", "namespace": "keycloak"},
                "spec": {"replicas": 1, "template": {"spec": {"containers": [
                    {"image": "keycloak:latest"}
                ]}}},
                "status": {"readyReplicas": 1},
            }]}
        elif command[0] == "helm":
            payload = [{"name": "istio", "namespace": "istio-system", "status": "deployed"}]
        else:
            payload = {"items": []}
        return CommandResult(exit_code=0, stdout=json.dumps(payload), duration_s=0.01)

    snapshot = inspect_runtime_context(
        RuntimeInspectionConfig(aws_region="us-east-1", kube_context="mosip"),
        runner=run,
    )

    assert snapshot.aws.instances[0].name == "worker-1"
    assert snapshot.kubernetes.server_version == "v1.36.3"
    assert snapshot.kubernetes.workloads[0].ready == 1
    assert snapshot.kubernetes.helm_releases[0].name == "istio"
    assert all(probe.succeeded for probe in snapshot.probes)
    assert all(
        not {"apply", "create", "delete", "install", "patch", "start", "stop", "upgrade"}
        & set(command)
        for command in commands
    )

    path = tmp_path / "runtime-context.json"
    write_json(path, snapshot)
    assert load_runtime_context(path) == snapshot
    assert "probes" not in compact_runtime_context(snapshot)


def test_runtime_inspection_records_partial_failures() -> None:
    def fail(command: list[str], *, timeout: float) -> CommandResult:
        return CommandResult(exit_code=1, stderr="token=unsafe-value")

    snapshot = inspect_runtime_context(RuntimeInspectionConfig(), runner=fail)

    assert not snapshot.aws.available
    assert not snapshot.kubernetes.available
    assert len(snapshot.probes) == 12
    assert all(probe.error == "token=[REDACTED]" for probe in snapshot.probes)
