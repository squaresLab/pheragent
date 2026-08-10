from __future__ import annotations

from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from ..enums import DeterministicFindingKind, InventoryCategory
from ..evidence import EvidenceStore
from ..models import InventoryEntry
from ..source_manager import AcquiredSource
from .base import ParseResult, make_finding, source_file_path


class YamlDeploymentParser:
    name = "yaml"

    def parse(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
    ) -> ParseResult:
        text = source_file_path(source, entry).read_text(encoding="utf-8")
        result = ParseResult(parser=self.name)
        try:
            documents = [document for document in yaml.compose_all(text) if document is not None]
        except yaml.YAMLError as exc:
            result.warnings.append(f"YAML parse failed: {exc}")
            return result

        for document in documents:
            if entry.category == InventoryCategory.CI_WORKFLOW:
                self._github_actions(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.COMPOSE:
                self._compose(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.KUBERNETES:
                self._kubernetes(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.KUSTOMIZE:
                self._kustomize(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.HELM:
                self._helm(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.HELMSMAN:
                self._helmsman(source, entry, evidence, document, result)
            elif entry.category == InventoryCategory.ANSIBLE:
                self._ansible(source, entry, evidence, document, result)
            else:
                self._generic_configuration(source, entry, evidence, document, result)
        return result

    def _github_actions(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        jobs = _mapping(document).get("jobs")
        for job_name, job_node in _mapping(jobs).items():
            job = _mapping(job_node)
            needs = _string_list(job.get("needs"))
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    job_node,
                    DeterministicFindingKind.COMPONENT,
                    job_name,
                    {"workflow_job": job_name, "needs": needs},
                )
            )
            for dependency in needs:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        job.get("needs") or job_node,
                        DeterministicFindingKind.DEPENDENCY,
                        f"{job_name} needs {dependency}",
                        {"source": job_name, "target": dependency, "relation": "needs"},
                    )
                )
            steps = job.get("steps")
            if isinstance(steps, SequenceNode):
                for index, step_node in enumerate(steps.value, start=1):
                    step = _mapping(step_node)
                    step_name = _scalar(step.get("name")) or f"{job_name} step {index}"
                    run = _scalar(step.get("run"))
                    uses = _scalar(step.get("uses"))
                    if run:
                        result.findings.append(
                            _finding(
                                source,
                                entry,
                                evidence,
                                step.get("run") or step_node,
                                DeterministicFindingKind.COMMAND,
                                step_name,
                                {"command": run, "workflow_job": job_name},
                            )
                        )
                    elif uses:
                        result.findings.append(
                            _finding(
                                source,
                                entry,
                                evidence,
                                step_node,
                                DeterministicFindingKind.CONFIGURATION,
                                step_name,
                                {"uses": uses, "workflow_job": job_name},
                            )
                        )

    def _compose(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        services = _mapping(document).get("services")
        for service_name, service_node in _mapping(services).items():
            service = _mapping(service_node)
            depends_on = _mapping_keys_or_strings(service.get("depends_on"))
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    service_node,
                    DeterministicFindingKind.COMPONENT,
                    service_name,
                    {
                        "compose_service": service_name,
                        "image": _scalar(service.get("image")),
                        "depends_on": depends_on,
                    },
                )
            )
            for dependency in depends_on:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        service.get("depends_on") or service_node,
                        DeterministicFindingKind.DEPENDENCY,
                        f"{service_name} depends on {dependency}",
                        {"source": service_name, "target": dependency},
                    )
                )
            command = _node_value(service.get("command"))
            if command:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        service.get("command") or service_node,
                        DeterministicFindingKind.COMMAND,
                        f"{service_name} command",
                        {"command": command, "component": service_name},
                    )
                )

    def _kubernetes(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        root = _mapping(document)
        kind = _scalar(root.get("kind")) or "Unknown"
        api_version = _scalar(root.get("apiVersion"))
        metadata = _mapping(root.get("metadata"))
        name = _scalar(metadata.get("name")) or kind
        namespace = _scalar(metadata.get("namespace"))
        result.findings.append(
            _finding(
                source,
                entry,
                evidence,
                document,
                DeterministicFindingKind.RESOURCE,
                name,
                {
                    "api_version": api_version,
                    "kubernetes_kind": kind,
                    "name": name,
                    "namespace": namespace,
                },
            )
        )
        for probe_name, probe_node in _walk_named_nodes(
            document, {"livenessProbe", "readinessProbe", "startupProbe"}
        ):
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    probe_node,
                    DeterministicFindingKind.VALIDATION,
                    f"{name} {probe_name}",
                    {"resource": name, "probe": probe_name},
                )
            )
        for volume_field, volumes_node in _walk_named_nodes(
            document, {"volumes", "volumeClaimTemplates"}
        ):
            if not isinstance(volumes_node, SequenceNode):
                continue
            for index, volume_node in enumerate(volumes_node.value, start=1):
                volume = _mapping(volume_node)
                volume_metadata = _mapping(volume.get("metadata"))
                volume_name = (
                    _scalar(volume.get("name"))
                    or _scalar(volume_metadata.get("name"))
                    or f"{name} volume {index}"
                )
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        volume_node,
                        DeterministicFindingKind.RESOURCE,
                        volume_name,
                        {"resource": name, "kubernetes_field": volume_field},
                    )
                )

    def _helm(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        root = _mapping(document)
        chart_name = _scalar(root.get("name"))
        if chart_name and "apiVersion" in root:
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    document,
                    DeterministicFindingKind.COMPONENT,
                    chart_name,
                    {"chart": chart_name, "version": _scalar(root.get("version"))},
                )
            )
        dependencies = root.get("dependencies")
        if isinstance(dependencies, SequenceNode):
            for dependency_node in dependencies.value:
                dependency = _mapping(dependency_node)
                name = _scalar(dependency.get("name"))
                if name:
                    result.findings.append(
                        _finding(
                            source,
                            entry,
                            evidence,
                            dependency_node,
                            DeterministicFindingKind.DEPENDENCY,
                            name,
                            {
                                "chart": chart_name,
                                "repository": _scalar(dependency.get("repository")),
                                "version": _scalar(dependency.get("version")),
                            },
                        )
                    )

    def _kustomize(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        root = _mapping(document)
        for field in ("resources", "bases", "components"):
            node = root.get(field)
            for resource in _string_list(node):
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        node or document,
                        DeterministicFindingKind.DEPENDENCY,
                        resource,
                        {"kustomize_field": field, "resource": resource},
                    )
                )

    def _helmsman(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        root = _mapping(document)
        for app_name, app_node in _mapping(root.get("apps")).items():
            app = _mapping(app_node)
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    app_node,
                    DeterministicFindingKind.COMPONENT,
                    app_name,
                    {
                        "chart": _scalar(app.get("chart")),
                        "namespace": _scalar(app.get("namespace")),
                    },
                )
            )

    def _ansible(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        plays = document.value if isinstance(document, SequenceNode) else [document]
        for index, play_node in enumerate(plays, start=1):
            play = _mapping(play_node)
            imported_playbook = _scalar(play.get("import_playbook"))
            if imported_playbook:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        play.get("import_playbook") or play_node,
                        DeterministicFindingKind.DEPENDENCY,
                        imported_playbook,
                        {"ansible_import_playbook": imported_playbook},
                    )
                )
                continue
            name = _scalar(play.get("name")) or f"play {index}"
            hosts = _node_value(play.get("hosts"))
            roles = _string_list(play.get("roles"))
            if play:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        play_node,
                        DeterministicFindingKind.PROCEDURE_STEP,
                        name,
                        {"hosts": hosts, "roles": roles},
                    )
                )
            for role in roles:
                result.findings.append(
                    _finding(
                        source,
                        entry,
                        evidence,
                        play.get("roles") or play_node,
                        DeterministicFindingKind.DEPENDENCY,
                        role,
                        {"play": name, "role": role},
                    )
                )

    def _generic_configuration(
        self,
        source: AcquiredSource,
        entry: InventoryEntry,
        evidence: EvidenceStore,
        document: Node,
        result: ParseResult,
    ) -> None:
        for key, value_node in _mapping(document).items():
            result.findings.append(
                _finding(
                    source,
                    entry,
                    evidence,
                    value_node,
                    DeterministicFindingKind.CONFIGURATION,
                    key,
                    {"key": key},
                )
            )


def _finding(
    source: AcquiredSource,
    entry: InventoryEntry,
    evidence: EvidenceStore,
    node: Node,
    kind: DeterministicFindingKind,
    name: str,
    attributes: dict[str, Any],
):
    start_line = node.start_mark.line + 1
    end_line = node.end_mark.line + (1 if node.end_mark.column else 0)
    end_line = max(start_line, end_line)
    return make_finding(
        source=source,
        entry=entry,
        evidence=evidence,
        kind=kind,
        name=name,
        start_line=start_line,
        end_line=end_line,
        attributes={key: value for key, value in attributes.items() if value is not None},
    )


def _mapping(node: Node | None) -> dict[str, Node]:
    if not isinstance(node, MappingNode):
        return {}
    return {key.value: value for key, value in node.value if isinstance(key, ScalarNode)}


def _scalar(node: Node | None) -> str | None:
    if isinstance(node, ScalarNode):
        return node.value
    return None


def _string_list(node: Node | None) -> list[str]:
    if isinstance(node, ScalarNode):
        return [node.value]
    if isinstance(node, SequenceNode):
        return [value.value for value in node.value if isinstance(value, ScalarNode)]
    return []


def _mapping_keys_or_strings(node: Node | None) -> list[str]:
    if isinstance(node, MappingNode):
        return sorted(_mapping(node))
    return _string_list(node)


def _node_value(node: Node | None) -> Any:
    if isinstance(node, ScalarNode):
        return node.value
    if isinstance(node, SequenceNode):
        return [_node_value(value) for value in node.value]
    if isinstance(node, MappingNode):
        return {key: _node_value(value) for key, value in _mapping(node).items()}
    return None


def _walk_named_nodes(node: Node, names: set[str]):
    if isinstance(node, MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, ScalarNode) and key_node.value in names:
                yield key_node.value, value_node
            yield from _walk_named_nodes(value_node, names)
    elif isinstance(node, SequenceNode):
        for value_node in node.value:
            yield from _walk_named_nodes(value_node, names)
