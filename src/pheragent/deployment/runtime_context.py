from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from pheragent.models import CommandResult
from pheragent.process import run_command

from .models import ContractModel
from .redaction import redact_secrets
from .serialization import load_yaml

_MAX_ITEMS = 500
_MAX_PROMPT_ITEMS = 25
_MAX_ERROR_CHARACTERS = 500


class RuntimeProbe(ContractModel):
    provider: str
    name: str
    command: list[str]
    succeeded: bool
    duration_seconds: float = Field(ge=0)
    error: str | None = None


class AwsInstance(ContractModel):
    id: str
    name: str | None = None
    state: str
    instance_type: str | None = None
    availability_zone: str | None = None
    private_ip: str | None = None
    public_ip: str | None = None
    project: str | None = None
    role: str | None = None


class AwsRuntimeContext(ContractModel):
    available: bool = False
    profile: str | None = None
    region: str | None = None
    account: str | None = None
    principal_arn: str | None = None
    instance_count: int = Field(default=0, ge=0)
    instances: list[AwsInstance] = Field(default_factory=list)


class KubernetesResource(ContractModel):
    kind: str
    name: str
    namespace: str | None = None
    state: str | None = None
    ready: int | None = Field(default=None, ge=0)
    desired: int | None = Field(default=None, ge=0)
    images: list[str] = Field(default_factory=list)


class KubernetesRuntimeContext(ContractModel):
    available: bool = False
    context: str | None = None
    server_version: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    namespaces: list[str] = Field(default_factory=list)
    nodes: list[KubernetesResource] = Field(default_factory=list)
    workloads: list[KubernetesResource] = Field(default_factory=list)
    services: list[KubernetesResource] = Field(default_factory=list)
    config_maps: list[KubernetesResource] = Field(default_factory=list)
    storage_classes: list[KubernetesResource] = Field(default_factory=list)
    ingress_classes: list[KubernetesResource] = Field(default_factory=list)
    custom_resource_definitions: list[KubernetesResource] = Field(default_factory=list)
    helm_releases: list[KubernetesResource] = Field(default_factory=list)


class RuntimeContextSnapshot(ContractModel):
    snapshot_version: str = Field(default="0.1", pattern=r"^0\.1$")
    captured_at: str
    aws: AwsRuntimeContext
    kubernetes: KubernetesRuntimeContext
    probes: list[RuntimeProbe] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class RuntimeInspectionConfig:
    aws_profile: str | None = None
    aws_region: str | None = None
    kube_context: str | None = None
    timeout: float = 30.0


CommandRunner = Callable[..., CommandResult]


def inspect_runtime_context(
    config: RuntimeInspectionConfig,
    *,
    runner: CommandRunner = run_command,
) -> RuntimeContextSnapshot:
    """Capture bounded environment facts using a fixed read-only command set."""
    probes: list[RuntimeProbe] = []
    warnings: list[str] = []
    aws = _inspect_aws(config, runner, probes, warnings)
    kubernetes = _inspect_kubernetes(config, runner, probes, warnings)
    return RuntimeContextSnapshot(
        captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        aws=aws,
        kubernetes=kubernetes,
        probes=probes,
        warnings=warnings,
    )


def load_runtime_context(path: Path) -> RuntimeContextSnapshot:
    return RuntimeContextSnapshot.model_validate(load_yaml(path.expanduser().resolve()))


def compact_runtime_context(snapshot: RuntimeContextSnapshot | None) -> dict[str, Any] | None:
    """Return only deployment-relevant observations for LLM prompts."""
    if snapshot is None:
        return None
    aws = snapshot.aws
    kubernetes = snapshot.kubernetes
    return {
        "captured_at": snapshot.captured_at,
        "aws": {
            "available": aws.available,
            "region": aws.region,
            "instances": [
                instance.model_dump(mode="json", exclude_none=True)
                for instance in aws.instances[:_MAX_PROMPT_ITEMS]
            ],
        },
        "kubernetes": {
            "available": kubernetes.available,
            "context": kubernetes.context,
            "server_version": kubernetes.server_version,
            "counts": kubernetes.counts,
            "namespaces": kubernetes.namespaces[:_MAX_PROMPT_ITEMS],
            "nodes": _prompt_resources(kubernetes.nodes),
            "workloads": _prompt_resources(kubernetes.workloads, limit=75),
            "config_maps": _prompt_resources(kubernetes.config_maps),
            "storage_classes": _prompt_resources(kubernetes.storage_classes),
            "ingress_classes": _prompt_resources(kubernetes.ingress_classes),
            "helm_releases": _prompt_resources(kubernetes.helm_releases),
        },
        "interpretation": (
            "Observed resource existence is not proof of health or deployment intent."
        ),
    }


def _inspect_aws(
    config: RuntimeInspectionConfig,
    runner: CommandRunner,
    probes: list[RuntimeProbe],
    warnings: list[str],
) -> AwsRuntimeContext:
    identity = _json_probe(
        "aws",
        "caller_identity",
        _aws_command(config, "sts", "get-caller-identity"),
        config,
        runner,
        probes,
        warnings,
    )
    inventory = _json_probe(
        "aws",
        "ec2_instances",
        _aws_command(config, "ec2", "describe-instances"),
        config,
        runner,
        probes,
        warnings,
    )
    instances = _aws_instances(inventory)
    return AwsRuntimeContext(
        available=identity is not None or inventory is not None,
        profile=config.aws_profile,
        region=config.aws_region,
        account=_string(identity, "Account"),
        principal_arn=_string(identity, "Arn"),
        instance_count=len(instances),
        instances=instances[:_MAX_ITEMS],
    )


def _inspect_kubernetes(
    config: RuntimeInspectionConfig,
    runner: CommandRunner,
    probes: list[RuntimeProbe],
    warnings: list[str],
) -> KubernetesRuntimeContext:
    version = _json_probe(
        "kubernetes",
        "version",
        _kubectl_command(config, "version"),
        config,
        runner,
        probes,
        warnings,
    )
    queries = {
        "nodes": ("get", "nodes"),
        "namespaces": ("get", "namespaces"),
        "workloads": ("get", "deployments,statefulsets,daemonsets,jobs,cronjobs", "-A"),
        "services": ("get", "services", "-A"),
        "config_maps": ("get", "configmaps", "-A"),
        "storage_classes": ("get", "storageclasses"),
        "ingress_classes": ("get", "ingressclasses"),
        "custom_resource_definitions": ("get", "crds"),
    }
    documents = {
        name: _json_probe(
            "kubernetes",
            name,
            _kubectl_command(config, *arguments),
            config,
            runner,
            probes,
            warnings,
        )
        for name, arguments in queries.items()
    }
    helm = _json_probe(
        "kubernetes",
        "helm_releases",
        _helm_command(config),
        config,
        runner,
        probes,
        warnings,
    )
    resources = {
        name: _kubernetes_resources(document)
        for name, document in documents.items()
        if name != "namespaces"
    }
    namespaces = sorted(
        resource.name for resource in _kubernetes_resources(documents.get("namespaces"))
    )
    helm_releases = _helm_releases(helm)
    counts = {name: len(items) for name, items in resources.items()}
    counts["namespaces"] = len(namespaces)
    counts["helm_releases"] = len(helm_releases)
    return KubernetesRuntimeContext(
        available=(
            version is not None
            or helm is not None
            or any(document is not None for document in documents.values())
        ),
        context=config.kube_context,
        server_version=_nested_string(version, "serverVersion", "gitVersion"),
        counts=counts,
        namespaces=namespaces[:_MAX_ITEMS],
        nodes=resources.get("nodes", [])[:_MAX_ITEMS],
        workloads=resources.get("workloads", [])[:_MAX_ITEMS],
        services=resources.get("services", [])[:_MAX_ITEMS],
        config_maps=resources.get("config_maps", [])[:_MAX_ITEMS],
        storage_classes=resources.get("storage_classes", [])[:_MAX_ITEMS],
        ingress_classes=resources.get("ingress_classes", [])[:_MAX_ITEMS],
        custom_resource_definitions=resources.get("custom_resource_definitions", [])[:_MAX_ITEMS],
        helm_releases=helm_releases[:_MAX_ITEMS],
    )


def _json_probe(
    provider: str,
    name: str,
    command: list[str],
    config: RuntimeInspectionConfig,
    runner: CommandRunner,
    probes: list[RuntimeProbe],
    warnings: list[str],
) -> Any | None:
    result = runner(command, timeout=config.timeout)
    error = None
    payload = None
    if result.ok:
        try:
            payload = json.loads(result.stdout or "null")
        except json.JSONDecodeError as exc:
            error = f"invalid JSON output: {exc}"
    else:
        error = result.combined_output or "command failed without output"
    safe_error = redact_secrets(error)[:_MAX_ERROR_CHARACTERS] if error else None
    probes.append(
        RuntimeProbe(
            provider=provider,
            name=name,
            command=command,
            succeeded=payload is not None,
            duration_seconds=result.duration_s,
            error=safe_error,
        )
    )
    if safe_error:
        warnings.append(f"{provider} {name}: {safe_error}")
    return payload


def _aws_command(config: RuntimeInspectionConfig, service: str, operation: str) -> list[str]:
    command = ["aws"]
    if config.aws_profile:
        command.extend(("--profile", config.aws_profile))
    if config.aws_region:
        command.extend(("--region", config.aws_region))
    command.extend((service, operation, "--output", "json", "--no-cli-pager"))
    return command


def _kubectl_command(config: RuntimeInspectionConfig, *arguments: str) -> list[str]:
    command = ["kubectl"]
    if config.kube_context:
        command.extend(("--context", config.kube_context))
    command.extend((*arguments, "-o", "json"))
    return command


def _helm_command(config: RuntimeInspectionConfig) -> list[str]:
    command = ["helm"]
    if config.kube_context:
        command.extend(("--kube-context", config.kube_context))
    command.extend(("list", "--all-namespaces", "--output", "json"))
    return command


def _aws_instances(document: Any) -> list[AwsInstance]:
    result = []
    reservations = document.get("Reservations", []) if isinstance(document, dict) else []
    for reservation in reservations:
        for instance in reservation.get("Instances", []):
            tags = {
                item.get("Key"): item.get("Value")
                for item in instance.get("Tags", [])
                if item.get("Key") in {"Name", "project", "role"}
            }
            instance_id = instance.get("InstanceId")
            if not isinstance(instance_id, str):
                continue
            result.append(
                AwsInstance(
                    id=instance_id,
                    name=tags.get("Name"),
                    state=_nested_string(instance, "State", "Name") or "unknown",
                    instance_type=instance.get("InstanceType"),
                    availability_zone=_nested_string(instance, "Placement", "AvailabilityZone"),
                    private_ip=instance.get("PrivateIpAddress"),
                    public_ip=instance.get("PublicIpAddress"),
                    project=tags.get("project"),
                    role=tags.get("role"),
                )
            )
    return sorted(result, key=lambda item: (item.name or "", item.id))


def _kubernetes_resources(document: Any) -> list[KubernetesResource]:
    items = document.get("items", []) if isinstance(document, dict) else []
    resources = []
    for item in items:
        metadata = item.get("metadata", {})
        name = metadata.get("name")
        if not isinstance(name, str):
            continue
        kind = str(item.get("kind") or "Resource")
        ready, desired = _readiness(kind, item)
        resources.append(
            KubernetesResource(
                kind=kind,
                name=name,
                namespace=metadata.get("namespace"),
                state=_resource_state(kind, item),
                ready=ready,
                desired=desired,
                images=_container_images(item),
            )
        )
    return sorted(resources, key=lambda item: (item.namespace or "", item.kind, item.name))


def _helm_releases(document: Any) -> list[KubernetesResource]:
    if not isinstance(document, list):
        return []
    releases = []
    for item in document:
        name = item.get("name")
        if not isinstance(name, str):
            continue
        releases.append(
            KubernetesResource(
                kind="HelmRelease",
                name=name,
                namespace=item.get("namespace"),
                state=item.get("status"),
                images=[],
            )
        )
    return sorted(releases, key=lambda item: (item.namespace or "", item.name))


def _readiness(kind: str, item: dict[str, Any]) -> tuple[int | None, int | None]:
    status = item.get("status", {})
    spec = item.get("spec", {})
    if kind == "DaemonSet":
        return status.get("numberReady", 0), status.get("desiredNumberScheduled", 0)
    if kind == "Job":
        return status.get("succeeded", 0), spec.get("completions", 1)
    if kind in {"Deployment", "StatefulSet"}:
        return status.get("readyReplicas", 0), spec.get("replicas", 1)
    if kind == "Node":
        ready = any(
            condition.get("type") == "Ready" and condition.get("status") == "True"
            for condition in status.get("conditions", [])
        )
        return int(ready), 1
    return None, None


def _resource_state(kind: str, item: dict[str, Any]) -> str | None:
    if kind == "Namespace":
        return _nested_string(item, "status", "phase")
    if kind == "StorageClass":
        return item.get("provisioner")
    if kind == "IngressClass":
        return _nested_string(item, "spec", "controller")
    return None


def _container_images(item: dict[str, Any]) -> list[str]:
    pod_spec = item.get("spec", {}).get("template", {}).get("spec", {})
    return sorted(
        {
            image
            for container in pod_spec.get("containers", [])
            if isinstance((image := container.get("image")), str)
        }
    )


def _string(document: Any, key: str) -> str | None:
    value = document.get(key) if isinstance(document, dict) else None
    return value if isinstance(value, str) else None


def _nested_string(document: Any, *keys: str) -> str | None:
    value = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value if isinstance(value, str) else None


def _prompt_resources(
    resources: list[KubernetesResource],
    *,
    limit: int = _MAX_PROMPT_ITEMS,
) -> list[dict[str, Any]]:
    return [
        resource.model_dump(mode="json", exclude_none=True)
        for resource in resources[:limit]
    ]
