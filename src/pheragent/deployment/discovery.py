from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import urlsplit

import yaml

from pheragent.utils import slugify

from .analysis_models import (
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisQuestion,
    AnalysisRelation,
    AnalysisRelationType,
    AnalysisSourceRef,
    ArtifactRole,
    ArtifactRoleCandidate,
    ArtifactScope,
    CandidateComponent,
    ComponentClassification,
    ComponentDeployment,
    ComponentDisposition,
    ComponentEvidenceStrength,
    DeploymentAction,
    DeploymentContext,
    DeploymentRoot,
    DeploymentSignalBundle,
    ExternalRequirement,
    ReferenceNode,
    RepositoryFile,
    SignalStrength,
    StageSignal,
    ValidationSignal,
)
from .enums import InventoryCategory
from .investigation_models import (
    ArtifactOutline,
    EvidenceKind,
    EvidenceObservation,
    FactPredicate,
    InvestigationSynthesis,
    SourcePurpose,
)
from .models import RepositoryInventory
from .retrieval import is_installer_path
from .source_manager import AcquiredSource

_VERSION = re.compile(r"(?<!\d)(\d+\.\d+\.\d+(?:\.\d+)?)(?!\d)")
_MARKDOWN_LINK = re.compile(r"\[[^]]+]\((?P<target>[^)]+)\)")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s+(?P<title>.+?)\s*$")
_HELM_RELEASE = re.compile(
    r"\bhelm\s+(?:upgrade\s+--install|install)\s+(?P<name>[A-Za-z0-9_.-]+)",
    re.IGNORECASE,
)
_ANSIBLE_PLAYBOOK = re.compile(
    r"\bansible-playbook\s+(?P<path>[^\s]+\.(?:ya?ml))",
    re.IGNORECASE,
)
_SHELL_INVOCATION = re.compile(
    r"(?:^|[;&|]\s*)(?P<target>(?:\./|\.\./)[A-Za-z0-9_./${}\"'-]+\.sh)"
    r"(?P<args>(?:[ \t]+[^;&|\n]+)?)?(?=$|[;&|])"
)
_MODULE_ARRAY = re.compile(
    r"declare\s+-a\s+module\s*=\s*\((?P<body>.*?)\)", re.DOTALL | re.IGNORECASE
)
_QUOTED_ITEM = re.compile(r"[\"'](?P<item>[A-Za-z0-9_.-]+)[\"']")
_TERRAFORM_MODULE = re.compile(r'(?m)^\s*module\s+"(?P<name>[^"]+)"\s*\{')
_TERRAFORM_RESOURCE = re.compile(r'(?m)^\s*resource\s+"(?P<kind>[^"]+)"\s+"(?P<name>[^"]+)"\s*\{')
_DEPLOYMENT_TOOL = re.compile(
    r"\b(?:helm(?:file)?|kubectl\s+apply|terraform\s+(?:apply|plan)|ansible-playbook|"
    r"docker\s+compose\s+up|kustomize\s+build)\b",
    re.IGNORECASE,
)
_VALIDATION_COMMAND = re.compile(
    r"\b(?:kubectl\s+(?:wait|rollout\s+status)|helm\s+test|docker\s+compose\s+ps|"
    r"ansible\s+.*--check|curl\s+[^\n]*(?:health|ready)|(?:check|verify|status)[-_a-z0-9]*\.sh)\b",
    re.IGNORECASE,
)
_FORBIDDEN_COMPONENT = re.compile(
    r"(?:^|\b)(?:command|step\s*\d*|kubectl|configmap|namespace|secret|configuration|"
    r"deployment)(?:$|\b)|\.(?:sh|ya?ml|json|ini)$|[/\\]",
    re.IGNORECASE,
)
_WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}
_MAX_ARTIFACT_ROLE_CARDS = 100
_REQUIRED_ENV_INPUT = re.compile(r"\$\{(?P<name>[A-Z][A-Z0-9_]*)\s*(?::?\?)")
_POSITIONAL_INPUT = re.compile(r"\$\{(?P<number>[1-9])\s*(?::?\?)")
_COMPOSE_FILE_ARGUMENT = re.compile(
    r"(?:^|\s)(?:-f|--file)(?:\s+|=)"
    r"(?P<path>\"[^\"]+\"|'[^']+'|[^\s\\]+)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class _ShellInvocation:
    target: str
    line: int
    command: str
    working_directory: str


@dataclass(frozen=True, slots=True)
class _ComposeInvocation:
    command: str
    line: int
    working_directory: str
    artifacts: tuple[str, ...]


@dataclass(slots=True)
class _ParsedArtifact:
    source_id: str
    path: str
    roles: set[ArtifactRole] = field(default_factory=set)
    scope: ArtifactScope = ArtifactScope.SUPPORTING
    references: list[tuple[str, int, AnalysisRelationType]] = field(default_factory=list)
    invocation_sequence: list[_ShellInvocation] = field(default_factory=list)
    validations: list[ValidationSignal] = field(default_factory=list)
    structured_components: list[tuple[str, AnalysisExecutor, int]] = field(default_factory=list)
    explicit_relations: list[AnalysisRelation] = field(default_factory=list)
    structural_score: float = 0.0
    compose_invocation: _ComposeInvocation | None = None


@dataclass(frozen=True, slots=True)
class RepositoryDiscovery:
    """Deterministic deployment knowledge exposed to the analysis coordinator."""

    files: tuple[RepositoryFile, ...]
    reference_nodes: tuple[ReferenceNode, ...]
    reference_relations: tuple[AnalysisRelation, ...]
    roots: tuple[DeploymentRoot, ...]
    selected_paths: frozenset[str]
    signals: DeploymentSignalBundle
    outlines: tuple[ArtifactOutline, ...]
    searchable_paths: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...]


def discover_repository(
    inventory: RepositoryInventory,
    context: DeploymentContext,
    sources: dict[str, AcquiredSource],
    source_purposes: dict[str, SourcePurpose],
    *,
    node_budget: int,
) -> RepositoryDiscovery:
    """Discover deployment roots, components, and relations using bounded local evidence."""
    files = _repository_files(inventory, context)
    parsed = _parse_artifacts(files, sources, context)
    nodes, relations = _reference_graph(parsed)
    files = _files_with_roles(files, parsed)
    roots = _discover_roots(files, parsed, relations, context)
    selected = _traverse_roots(roots, relations, node_budget)
    selected.update(_supplemental_materialization_artifacts(parsed))
    signals, warnings = _discover_signals(context, roots, selected, parsed, sources)
    candidates = _artifact_role_candidates(files, parsed, relations, context)
    return RepositoryDiscovery(
        files=tuple(files),
        reference_nodes=tuple(nodes),
        reference_relations=tuple(relations),
        roots=tuple(roots),
        selected_paths=frozenset(selected),
        signals=signals,
        outlines=tuple(_investigation_outlines(candidates, source_purposes)),
        searchable_paths=_searchable_paths(files, context),
        warnings=tuple(warnings),
    )


def bind_retrieved_installation_routes(
    signals: DeploymentSignalBundle,
    observations: tuple[EvidenceObservation, ...],
    sources: dict[str, AcquiredSource],
) -> tuple[DeploymentSignalBundle, list[str]]:
    """Attach source-grounded installer commands recovered during retrieval."""
    return _bind_retrieved_installation_routes(signals, observations, sources)


def apply_investigation_synthesis(
    signals: DeploymentSignalBundle,
    synthesis: InvestigationSynthesis,
    observations: tuple[EvidenceObservation, ...],
) -> DeploymentSignalBundle:
    """Merge semantic decisions without discarding deterministic candidates."""
    return _apply_investigation_synthesis(signals, synthesis, observations)


def _repository_files(
    inventory: RepositoryInventory,
    context: DeploymentContext,
) -> list[RepositoryFile]:
    result = []
    for entry in inventory.entries:
        version = _path_version(entry.path)
        profile = _path_profile(entry.path)
        result.append(
            RepositoryFile(
                repo_id=entry.source_id,
                path=entry.path,
                file_type=entry.category.value,
                size=entry.size_bytes,
                version_context=version,
                profile_context=profile,
                deployment_roles=[],
            )
        )
    return result


_INVESTIGATION_FILE_TYPES = {
    InventoryCategory.DOCUMENTATION.value,
    InventoryCategory.SHELL.value,
    InventoryCategory.ANSIBLE.value,
    InventoryCategory.TERRAFORM.value,
    InventoryCategory.HELM.value,
    InventoryCategory.HELMSMAN.value,
    InventoryCategory.KUSTOMIZE.value,
    InventoryCategory.KUBERNETES.value,
    InventoryCategory.COMPOSE.value,
    InventoryCategory.CI_WORKFLOW.value,
    InventoryCategory.CONFIGURATION.value,
}
_INVESTIGATION_EXCLUDED_PARTS = {
    ".git",
    ".gitbook",
    "node_modules",
    "vendor",
}


def _searchable_paths(
    files: list[RepositoryFile],
    context: DeploymentContext,
) -> dict[str, list[str]]:
    """Return ranked, text-like deployment evidence paths.

    Search is deliberately narrower than inventory. Inventory records everything;
    investigation reads only files that can plausibly explain deployment.
    """
    paths: dict[str, list[str]] = defaultdict(list)
    for file in files:
        path = PurePosixPath(file.path)
        parts = {part.casefold() for part in path.parts}
        if file.file_type not in _INVESTIGATION_FILE_TYPES:
            continue
        if parts & _INVESTIGATION_EXCLUDED_PARTS:
            continue
        if (
            ".github" in parts
            and file.file_type == InventoryCategory.CI_WORKFLOW.value
            and ArtifactRole.ORCHESTRATOR not in file.deployment_roles
        ):
            continue
        if path.name.casefold().startswith(("license", "notice", "changelog")):
            continue
        if (
            file.profile_context
            and context.deployment.profile
            and file.profile_context != context.deployment.profile
        ):
            continue
        if "upgrade" in parts and not any(
            "upgrade" in objective.casefold() for objective in context.objectives
        ):
            continue
        if (
            "api" in parts
            and file.size > 100_000
            and not file.deployment_roles
            and path.suffix.casefold() in {".json", ".yaml", ".yml"}
        ):
            continue
        paths[file.repo_id].append(file.path)
    file_by_key = {(file.repo_id, file.path): file for file in files}
    return {
        source_id: sorted(
            items,
            key=lambda path: _investigation_path_rank(
                file_by_key[(source_id, path)],
                context,
            ),
        )
        for source_id, items in paths.items()
    }


def _investigation_path_rank(
    file: RepositoryFile,
    context: DeploymentContext,
) -> tuple[int, int, str]:
    """Put profile/version deployment guides and executable roots first."""
    path = file.path.casefold()
    score = 0
    if file.version_context and file.version_context == context.deployment.version:
        score += 30
    if file.profile_context and file.profile_context == context.deployment.profile:
        score += 25
    if context.deployment.version and context.deployment.version.casefold() in path:
        score += 20
    if context.deployment.profile and context.deployment.profile.casefold() in path:
        score += 15
    if any(part in path for part in ("deploy", "install", "setup", "infra")):
        score += 12
    if file.deployment_roles:
        score += 8
    if PurePosixPath(file.path).name.casefold() in {"readme.md", "install.sh", "deploy.sh"}:
        score += 5
    return (-score, len(PurePosixPath(file.path).parts), file.path)


def _investigation_outlines(
    candidates: list[ArtifactRoleCandidate],
    purposes: dict[str, SourcePurpose],
) -> list[ArtifactOutline]:
    return [
        ArtifactOutline(
            id=candidate.id,
            source_id=candidate.repo_id,
            source_purpose=purposes[candidate.repo_id],
            path=candidate.path,
            file_type=candidate.file_type,
            roles=[role.value for role in candidate.deterministic_roles],
            references=candidate.references,
            referenced_by=candidate.referenced_by,
            materialized_names=candidate.materialized_names,
            validation_count=candidate.validation_count,
            context_hint=candidate.context_hint,
        )
        for candidate in candidates
    ]


def _artifact_role_candidates(
    files: list[RepositoryFile],
    parsed: dict[str, _ParsedArtifact],
    relations: list[AnalysisRelation],
    context: DeploymentContext,
) -> list[ArtifactRoleCandidate]:
    """Build compact, source-grounded cards for the artifact-role classifier."""
    incoming: dict[str, int] = defaultdict(int)
    for relation in relations:
        incoming[relation.target] += 1
    file_by_key = {_node_key(item.repo_id, item.path): item for item in files}
    hints = {_normalize_path(PurePosixPath(path)) for path in context.hints.documentation}
    candidates = []
    for key, artifact in parsed.items():
        file = file_by_key.get(key)
        if file is None:
            continue
        candidates.append(
            ArtifactRoleCandidate(
                id=_node_id(artifact.source_id, artifact.path),
                repo_id=artifact.source_id,
                path=artifact.path,
                file_type=file.file_type,
                deterministic_roles=sorted(artifact.roles),
                references=len(artifact.references),
                referenced_by=incoming[key],
                materialized_names=[
                    name for name, _executor, _line in artifact.structured_components[:20]
                ],
                validation_count=len(artifact.validations),
                context_hint=artifact.path in hints,
            )
        )
    ranked = sorted(
        candidates,
        key=lambda item: (
            -_artifact_role_card_score(item),
            item.repo_id,
            item.path,
        ),
    )
    return ranked[:_MAX_ARTIFACT_ROLE_CARDS]


def _artifact_role_card_score(candidate: ArtifactRoleCandidate) -> int:
    root_roles = {
        ArtifactRole.DEPLOYMENT_GUIDE,
        ArtifactRole.DEPLOYMENT_INDEX,
        ArtifactRole.ORCHESTRATOR,
        ArtifactRole.DESIRED_STATE,
    }
    return (
        int(candidate.context_hint) * 1000
        + len(root_roles & set(candidate.deterministic_roles)) * 200
        + min(candidate.references, 20) * 10
        + min(candidate.referenced_by, 20) * 5
        + int(bool(candidate.materialized_names)) * 25
        + min(candidate.validation_count, 10) * 3
    )


def _files_with_roles(
    files: list[RepositoryFile],
    parsed: dict[str, _ParsedArtifact],
) -> list[RepositoryFile]:
    return [
        file.model_copy(
            update={
                "deployment_roles": sorted(
                    parsed.get(
                        _node_key(file.repo_id, file.path),
                        _ParsedArtifact("", ""),
                    ).roles
                )
            }
        )
        for file in files
    ]


def _supplemental_materialization_artifacts(
    parsed: dict[str, _ParsedArtifact],
) -> set[str]:
    """Keep non-Compose materialization sites not reached from a selected root.

    Compose repositories commonly contain mutually exclusive overlays. Those
    artifacts must enter the selected set through a root or an explicit file
    reference rather than merely existing in the repository.
    """
    return {
        key
        for key, artifact in parsed.items()
        if artifact.structured_components
        and not any(
            executor == AnalysisExecutor.DOCKER_COMPOSE
            for _name, executor, _line in artifact.structured_components
        )
    }


def _parse_artifacts(
    files: list[RepositoryFile],
    sources: dict[str, AcquiredSource],
    context: DeploymentContext,
) -> dict[str, _ParsedArtifact]:
    parsed: dict[str, _ParsedArtifact] = {}
    for file in files:
        if file.size > 2 * 1024 * 1024 or _wrong_context(file, context):
            continue
        if file.file_type not in {
            InventoryCategory.DOCUMENTATION.value,
            InventoryCategory.SHELL.value,
            InventoryCategory.ANSIBLE.value,
            InventoryCategory.TERRAFORM.value,
            InventoryCategory.HELM.value,
            InventoryCategory.KUSTOMIZE.value,
            InventoryCategory.KUBERNETES.value,
            InventoryCategory.COMPOSE.value,
            InventoryCategory.CI_WORKFLOW.value,
            InventoryCategory.CONFIGURATION.value,
        }:
            continue
        try:
            path = sources[file.repo_id].resolve_path(file.path)
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError, ValueError:
            continue
        artifact = _parse_file(file, text)
        if artifact.roles or artifact.references or artifact.structured_components:
            parsed[_node_key(file.repo_id, file.path)] = artifact
    return parsed


def _parse_file(file: RepositoryFile, text: str) -> _ParsedArtifact:
    artifact = _ParsedArtifact(source_id=file.repo_id, path=file.path)
    if file.file_type == InventoryCategory.SHELL.value:
        _parse_shell(artifact, text)
    elif file.file_type == InventoryCategory.DOCUMENTATION.value:
        _parse_markdown(artifact, text)
    elif file.file_type == InventoryCategory.TERRAFORM.value:
        _parse_terraform(artifact, text)
    elif file.file_type in {
        InventoryCategory.ANSIBLE.value,
        InventoryCategory.HELM.value,
        InventoryCategory.KUSTOMIZE.value,
        InventoryCategory.KUBERNETES.value,
        InventoryCategory.COMPOSE.value,
        InventoryCategory.CI_WORKFLOW.value,
        InventoryCategory.CONFIGURATION.value,
    }:
        _parse_yaml_artifact(artifact, text, file.file_type)
    return artifact


def _parse_shell(artifact: _ParsedArtifact, text: str) -> None:
    path = PurePosixPath(artifact.path)
    lowered = artifact.path.casefold()
    logical_lines = _logical_shell_lines(text)
    modules = []
    module_match = _MODULE_ARRAY.search(text)
    if module_match:
        modules = [
            match.group("item") for match in _QUOTED_ITEM.finditer(module_match.group("body"))
        ]
    root_dir = path.parent.parent
    if modules:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
        module_line = text.count("\n", 0, module_match.start()) + 1 if module_match else 1
        for module in modules:
            target = _normalize_path(root_dir / module / "install.sh")
            artifact.references.append((target, module_line, AnalysisRelationType.INVOKES))
            artifact.invocation_sequence.append(
                _ShellInvocation(
                    target=target,
                    line=module_line,
                    command="./install.sh",
                    working_directory=PurePosixPath(target).parent.as_posix(),
                )
            )

    current_dir = path.parent
    for line_number, command in logical_lines:
        stripped = command.strip()
        cd_target = _shell_cd_target(stripped, root_dir, path.parent)
        if cd_target is not None:
            current_dir = cd_target
            continue
        for match in _SHELL_INVOCATION.finditer(stripped):
            raw_target = match.group("target").strip("\"'")
            target = _normalize_path(current_dir / raw_target)
            command = f"{raw_target}{match.group('args') or ''}".strip()
            artifact.references.append((target, line_number, AnalysisRelationType.INVOKES))
            artifact.invocation_sequence.append(
                _ShellInvocation(
                    target=target,
                    line=line_number,
                    command=command,
                    working_directory=current_dir.as_posix(),
                )
            )
        if _VALIDATION_COMMAND.search(stripped) and not re.match(
            r"^[A-Za-z_][A-Za-z0-9_]*=", stripped
        ):
            artifact.validations.append(
                ValidationSignal(
                    subject=path.parent.name,
                    check=stripped[:500],
                    source_ref=AnalysisSourceRef(
                        repo_id=artifact.source_id,
                        path=artifact.path,
                        start_line=line_number,
                        end_line=line_number,
                    ),
                )
            )

    if len({invocation.target for invocation in artifact.invocation_sequence}) >= 2:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
    if path.name.casefold() in {"install.sh", "deploy.sh", "setup.sh"}:
        artifact.roles.add(ArtifactRole.COMPONENT_INSTALLER)
    if any(term in lowered for term in ("init", "migrat", "bootstrap", "onboard", "seed")):
        artifact.roles.add(ArtifactRole.INITIALIZATION)
    if any(term in lowered for term in ("backup", "restore", "monitor", "logging")):
        artifact.roles.add(ArtifactRole.OPERATIONS)
    if artifact.validations or any(term in lowered for term in ("check", "verify", "status")):
        artifact.roles.add(ArtifactRole.VALIDATION)
    artifact.structural_score += (
        len(artifact.invocation_sequence) * 8 + len(artifact.validations) * 3
    )


def _parse_markdown(artifact: _ParsedArtifact, text: str) -> None:
    lowered_path = artifact.path.casefold()
    local_links = 0
    deployment_links = 0
    current_heading: str | None = None
    for line_number, line in enumerate(text.splitlines(), start=1):
        if heading := _MARKDOWN_HEADING.match(line.strip()):
            current_heading = heading.group("title").strip(" `#")
        for match in _MARKDOWN_LINK.finditer(line):
            target = match.group("target").split("#", 1)[0].strip()
            if not target or urlsplit(target).scheme or target.startswith("#"):
                continue
            resolved = _normalize_path(PurePosixPath(artifact.path).parent / target)
            artifact.references.append((resolved, line_number, AnalysisRelationType.REFERENCES))
            local_links += 1
            if _looks_deployment_path(resolved):
                deployment_links += 1
        if _DEPLOYMENT_TOOL.search(line):
            artifact.structural_score += 2
            documented = _documented_component(line, current_heading)
            if documented is not None:
                component_name, executor = documented
                artifact.structured_components.append((component_name, executor, line_number))
    compose_invocation = _select_compose_invocation(artifact.path, text)
    if compose_invocation is not None:
        artifact.compose_invocation = compose_invocation
        artifact.roles.add(ArtifactRole.DEPLOYMENT_GUIDE)
        artifact.structural_score += 8
        _add_compose_references(artifact, compose_invocation)
    is_global_index = PurePosixPath(artifact.path).name.casefold() in {
        "summary.md",
        "summary.markdown",
    }
    if not is_global_index and (_looks_deployment_path(artifact.path) or deployment_links):
        artifact.roles.add(ArtifactRole.DEPLOYMENT_GUIDE)
    if (
        PurePosixPath(artifact.path).name.casefold().startswith("readme") or is_global_index
    ) and local_links >= 2:
        artifact.roles.add(ArtifactRole.DEPLOYMENT_INDEX)
    if re.search(r"(?im)^#{1,4}\s+.*(?:prerequisite|requirement|before you begin)", text):
        artifact.roles.add(ArtifactRole.HUMAN_PREREQUISITE)
    if re.search(r"(?im)^#{1,4}\s+.*(?:verify|validation|health|test)", text):
        artifact.roles.add(ArtifactRole.VALIDATION)
    # A repository-wide documentation index can contain hundreds of deployment
    # links.  It is useful for reachability, but link volume is not evidence that
    # it is a better deployment entrypoint than a focused installation guide.
    artifact.structural_score += min(deployment_links, 20) * 5 + min(local_links, 10)
    if "deployment" in lowered_path or "install" in lowered_path:
        artifact.structural_score += 5


def _documented_component(
    line: str,
    heading: str | None,
) -> tuple[str, AnalysisExecutor] | None:
    """Extract only component names explicitly present in a deployment command."""
    if match := _HELM_RELEASE.search(line):
        return match.group("name"), AnalysisExecutor.HELM
    if match := _ANSIBLE_PLAYBOOK.search(line):
        return PurePosixPath(match.group("path").strip("`'\"")).stem, AnalysisExecutor.ANSIBLE
    if not heading or _invalid_component_name(heading):
        return None
    if re.search(r"\bterraform\s+(?:apply|plan)\b", line, re.IGNORECASE):
        return heading, AnalysisExecutor.TERRAFORM
    if re.search(r"\bkubectl\s+apply\b", line, re.IGNORECASE):
        return heading, AnalysisExecutor.KUBERNETES
    if re.search(r"\bkustomize\s+build\b", line, re.IGNORECASE):
        return heading, AnalysisExecutor.KUSTOMIZE
    return None


def _select_compose_invocation(path: str, text: str) -> _ComposeInvocation | None:
    """Return one explicit Compose file set, preferring a documented default."""
    lines = text.splitlines()
    candidates: list[tuple[bool, _ComposeInvocation]] = []
    index = 0
    while index < len(lines):
        start = index
        command_line = _comment_content(lines[index])
        if not re.search(r"\bdocker\s+compose\b", command_line, re.IGNORECASE):
            index += 1
            continue
        parts = []
        while True:
            continued = command_line.rstrip().endswith("\\")
            parts.append(command_line.rstrip().removesuffix("\\").strip())
            if not continued or index + 1 >= len(lines):
                break
            index += 1
            command_line = _comment_content(lines[index])
        command = " ".join(part for part in parts if part)
        if not re.search(r"\bup\b", command, re.IGNORECASE):
            index += 1
            continue
        raw_files = [
            match.group("path").strip("'\"") for match in _COMPOSE_FILE_ARGUMENT.finditer(command)
        ]
        if not raw_files:
            index += 1
            continue
        parent = PurePosixPath(path).parent
        artifacts = tuple(
            dict.fromkeys(_normalize_path(parent / raw_path) for raw_path in raw_files)
        )
        context = _compose_invocation_label(lines, start)
        candidates.append(
            (
                bool(re.search(r"\bdefault\b", context, re.IGNORECASE)),
                _ComposeInvocation(
                    command=command,
                    line=start + 1,
                    working_directory=parent.as_posix(),
                    artifacts=artifacts,
                ),
            )
        )
        index += 1

    defaults = [invocation for is_default, invocation in candidates if is_default]
    if len(defaults) == 1:
        return defaults[0]
    if len(candidates) == 1:
        return candidates[0][1]
    return None


def _comment_content(line: str) -> str:
    stripped = line.strip()
    return stripped[1:].strip() if stripped.startswith("#") else stripped


def _compose_invocation_label(lines: list[str], command_index: int) -> str:
    for index in range(command_index - 1, max(-1, command_index - 4), -1):
        candidate = _comment_content(lines[index])
        if candidate and not candidate.startswith("```"):
            return candidate
    return ""


def _add_compose_references(
    artifact: _ParsedArtifact,
    invocation: _ComposeInvocation,
) -> None:
    for target in invocation.artifacts:
        if target != artifact.path:
            artifact.references.append((target, invocation.line, AnalysisRelationType.REFERENCES))


def _parse_terraform(artifact: _ParsedArtifact, text: str) -> None:
    artifact.roles.add(ArtifactRole.DESIRED_STATE)
    declarations: list[tuple[int, str]] = []
    for match in _TERRAFORM_MODULE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        component_id = slugify(match.group("name"))
        artifact.structured_components.append(
            (match.group("name"), AnalysisExecutor.TERRAFORM, line)
        )
        declarations.append((match.start(), component_id))
    for match in _TERRAFORM_RESOURCE.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        component_name = f"{match.group('kind')}.{match.group('name')}"
        artifact.structured_components.append((component_name, AnalysisExecutor.TERRAFORM, line))
        declarations.append((match.start(), slugify(component_name)))
    declarations.sort()
    for match in re.finditer(r"(?ms)depends_on\s*=\s*\[(?P<body>.*?)]", text):
        owners = [item for item in declarations if item[0] < match.start()]
        if not owners:
            continue
        owner = owners[-1][1]
        line = text.count("\n", 0, match.start()) + 1
        for dependency in re.findall(
            r"(?:module\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)",
            match.group("body"),
        ):
            artifact.explicit_relations.append(
                AnalysisRelation(
                    source=owner,
                    target=slugify(dependency.removeprefix("module.")),
                    relation=AnalysisRelationType.REQUIRES,
                    strength=SignalStrength.EXPLICIT_DEPENDENCY,
                    evidence=[
                        AnalysisSourceRef(
                            repo_id=artifact.source_id,
                            path=artifact.path,
                            start_line=line,
                            end_line=line,
                        )
                    ],
                )
            )
    for match in re.finditer(r'(?m)^\s*source\s*=\s*"(?P<path>\.{1,2}/[^"]+)"', text):
        target = _normalize_path(PurePosixPath(artifact.path).parent / match.group("path"))
        if not PurePosixPath(target).suffix:
            target = _normalize_path(PurePosixPath(target) / "main.tf")
        line = text.count("\n", 0, match.start()) + 1
        artifact.references.append((target, line, AnalysisRelationType.REFERENCES))
    artifact.structural_score += len(artifact.structured_components) * 4


def _parse_yaml_artifact(artifact: _ParsedArtifact, text: str, file_type: str) -> None:
    try:
        loaded = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return
    documents = [
        document
        for item in loaded
        for document in (item if isinstance(item, list) else [item])
        if isinstance(document, dict)
    ]
    name = PurePosixPath(artifact.path).name.casefold()
    if name == "chart.yaml":
        artifact.roles.add(ArtifactRole.DESIRED_STATE)
        for document in documents:
            chart = document.get("name")
            if isinstance(chart, str):
                artifact.structured_components.append((chart, AnalysisExecutor.HELM, 1))
                for dependency in document.get("dependencies", []):
                    if not isinstance(dependency, dict) or not isinstance(
                        dependency.get("name"), str
                    ):
                        continue
                    dependency_name = dependency["name"]
                    artifact.structured_components.append(
                        (dependency_name, AnalysisExecutor.HELM, 1)
                    )
                    artifact.explicit_relations.append(
                        AnalysisRelation(
                            source=slugify(chart),
                            target=slugify(dependency_name),
                            relation=AnalysisRelationType.REQUIRES,
                            strength=SignalStrength.EXPLICIT_DEPENDENCY,
                            evidence=[
                                AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)
                            ],
                        )
                    )
    if file_type == InventoryCategory.COMPOSE.value:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
        if name in {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"}:
            artifact.structural_score += 20
        compose_invocation = _select_compose_invocation(artifact.path, text)
        if compose_invocation is not None:
            artifact.compose_invocation = compose_invocation
            artifact.structural_score += 8
            _add_compose_references(artifact, compose_invocation)
        for document in documents:
            services = document.get("services")
            if not isinstance(services, dict):
                continue
            for service_name, service in services.items():
                artifact.structured_components.append(
                    (str(service_name), AnalysisExecutor.DOCKER_COMPOSE, 1)
                )
                if not isinstance(service, dict):
                    continue
                depends = service.get("depends_on", [])
                dependency_names = depends.keys() if isinstance(depends, dict) else depends
                iterable_dependencies = (
                    dependency_names if isinstance(dependency_names, Iterable) else []
                )
                for dependency in iterable_dependencies:
                    condition = depends.get(dependency, {}) if isinstance(depends, dict) else {}
                    health = (
                        isinstance(condition, dict)
                        and condition.get("condition") == "service_healthy"
                    )
                    artifact.explicit_relations.append(
                        AnalysisRelation(
                            source=str(service_name),
                            target=str(dependency),
                            relation=(
                                AnalysisRelationType.HEALTH_GATED_BY
                                if health
                                else AnalysisRelationType.REQUIRES
                            ),
                            strength=(
                                SignalStrength.EXPLICIT_HEALTH_DEPENDENCY
                                if health
                                else SignalStrength.EXPLICIT_DEPENDENCY
                            ),
                            evidence=[
                                AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)
                            ],
                        )
                    )
                if isinstance(service, dict) and isinstance(service.get("healthcheck"), dict):
                    artifact.validations.append(
                        ValidationSignal(
                            subject=str(service_name),
                            check="compose healthcheck",
                            readiness="healthy",
                            source_ref=AnalysisSourceRef(
                                repo_id=artifact.source_id, path=artifact.path
                            ),
                        )
                    )
    if name in {"kustomization.yaml", "kustomization.yml"}:
        artifact.roles.update({ArtifactRole.ORCHESTRATOR, ArtifactRole.DESIRED_STATE})
        component_name = PurePosixPath(artifact.path).parent.name or "kustomization"
        if not _invalid_component_name(component_name):
            artifact.structured_components.append((component_name, AnalysisExecutor.KUSTOMIZE, 1))
        for document in documents:
            for field_name in ("resources", "bases", "components"):
                entries = document.get(field_name, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, str) or urlsplit(entry).scheme:
                        continue
                    target = _normalize_path(PurePosixPath(artifact.path).parent / entry)
                    if not PurePosixPath(target).suffix:
                        target = _normalize_path(PurePosixPath(target) / "kustomization.yaml")
                    artifact.references.append((target, 1, AnalysisRelationType.COMPOSES))
    if file_type == InventoryCategory.ANSIBLE.value:
        _parse_ansible_documents(artifact, documents)
    if file_type == InventoryCategory.CI_WORKFLOW.value:
        _parse_github_workflow(artifact, documents)
    for document in documents:
        kind = document.get("kind")
        metadata = document.get("metadata")
        resource_name = metadata.get("name") if isinstance(metadata, dict) else None
        if kind in _WORKLOAD_KINDS and isinstance(resource_name, str):
            artifact.roles.add(ArtifactRole.DESIRED_STATE)
            artifact.structured_components.append((resource_name, AnalysisExecutor.KUBERNETES, 1))
            serialized = yaml.safe_dump(document, sort_keys=False)
            for probe in ("livenessProbe", "readinessProbe", "startupProbe"):
                if probe in serialized:
                    artifact.validations.append(
                        ValidationSignal(
                            subject=resource_name,
                            check=probe,
                            readiness=probe.removesuffix("Probe"),
                            source_ref=AnalysisSourceRef(
                                repo_id=artifact.source_id, path=artifact.path
                            ),
                        )
                    )
        if kind in {"Kustomization", "HelmRelease"} and isinstance(resource_name, str):
            _parse_flux_object(artifact, document, kind, resource_name)
        if kind in {"Application", "ApplicationSet"} and isinstance(resource_name, str):
            _parse_argo_object(artifact, document, resource_name)
    artifact.structural_score += len(artifact.structured_components) * 4


def _parse_ansible_documents(
    artifact: _ParsedArtifact, documents: list[dict[object, object]]
) -> None:
    artifact.roles.add(ArtifactRole.DESIRED_STATE)
    ordered_roles: list[str] = []
    for document in documents:
        imported = document.get("import_playbook")
        if isinstance(imported, str):
            target = _normalize_path(PurePosixPath(artifact.path).parent / imported)
            artifact.references.append((target, 1, AnalysisRelationType.INVOKES))
            artifact.roles.add(ArtifactRole.ORCHESTRATOR)
        roles = document.get("roles", [])
        if isinstance(roles, list):
            for role in roles:
                role_name = (
                    role
                    if isinstance(role, str)
                    else role.get("role")
                    if isinstance(role, dict)
                    else None
                )
                if isinstance(role_name, str):
                    ordered_roles.append(role_name)
                    artifact.structured_components.append((role_name, AnalysisExecutor.ANSIBLE, 1))
        for task_group in ("pre_tasks", "tasks", "post_tasks"):
            tasks = document.get(task_group, [])
            if not isinstance(tasks, list):
                continue
            for task in tasks:
                if not isinstance(task, dict):
                    continue
                for action in ("import_role", "include_role"):
                    role_spec = task.get(action)
                    role_name = role_spec.get("name") if isinstance(role_spec, dict) else role_spec
                    if isinstance(role_name, str):
                        ordered_roles.append(role_name)
                        artifact.structured_components.append(
                            (role_name, AnalysisExecutor.ANSIBLE, 1)
                        )
                for action in ("import_tasks", "include_tasks"):
                    task_path = task.get(action)
                    if isinstance(task_path, str):
                        target = _normalize_path(PurePosixPath(artifact.path).parent / task_path)
                        artifact.references.append((target, 1, AnalysisRelationType.INVOKES))
    if len(ordered_roles) >= 2:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
    for left, right in zip(ordered_roles, ordered_roles[1:], strict=False):
        artifact.explicit_relations.append(
            AnalysisRelation(
                source=slugify(left),
                target=slugify(right),
                relation=AnalysisRelationType.ORDERED_BEFORE,
                strength=SignalStrength.DECLARED_ORDER,
                evidence=[AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)],
            )
        )


def _parse_github_workflow(
    artifact: _ParsedArtifact, documents: list[dict[object, object]]
) -> None:
    for document in documents:
        jobs = document.get("jobs")
        if not isinstance(jobs, dict):
            continue
        deployment_jobs: set[str] = set()
        for job_name, job in jobs.items():
            if not isinstance(job, dict):
                continue
            steps = job.get("steps", [])
            deploys = any(
                isinstance(step, dict)
                and (
                    _DEPLOYMENT_TOOL.search(str(step.get("run", ""))) is not None
                    or re.search(r"(?:^|\s)\.{0,2}/[^\s]+\.sh(?:\s|$)", str(step.get("run", "")))
                    is not None
                )
                for step in (steps if isinstance(steps, list) else [])
            )
            if deploys:
                deployment_jobs.add(str(job_name))
                artifact.structured_components.append(
                    (str(job_name), AnalysisExecutor.GITHUB_ACTIONS, 1)
                )
        if deployment_jobs:
            artifact.roles.add(ArtifactRole.ORCHESTRATOR)
        for job_name in deployment_jobs:
            job = jobs[job_name]
            needs = job.get("needs", [])
            dependency_names = [needs] if isinstance(needs, str) else needs
            if not isinstance(dependency_names, list):
                continue
            for dependency in dependency_names:
                if str(dependency) not in deployment_jobs:
                    continue
                artifact.explicit_relations.append(
                    AnalysisRelation(
                        source=slugify(job_name),
                        target=slugify(str(dependency)),
                        relation=AnalysisRelationType.REQUIRES,
                        strength=SignalStrength.EXPLICIT_DEPENDENCY,
                        evidence=[
                            AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)
                        ],
                    )
                )


def _parse_flux_object(
    artifact: _ParsedArtifact,
    document: dict[object, object],
    kind: str,
    resource_name: str,
) -> None:
    artifact.roles.update({ArtifactRole.ORCHESTRATOR, ArtifactRole.DESIRED_STATE})
    executor = AnalysisExecutor.HELM if kind == "HelmRelease" else AnalysisExecutor.KUSTOMIZE
    artifact.structured_components.append((resource_name, executor, 1))
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return
    for dependency in spec.get("dependsOn", []):
        dependency_name = dependency.get("name") if isinstance(dependency, dict) else None
        if not isinstance(dependency_name, str):
            continue
        artifact.explicit_relations.append(
            AnalysisRelation(
                source=slugify(resource_name),
                target=slugify(dependency_name),
                relation=AnalysisRelationType.HEALTH_GATED_BY,
                strength=SignalStrength.EXPLICIT_HEALTH_DEPENDENCY,
                evidence=[AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)],
            )
        )


def _parse_argo_object(
    artifact: _ParsedArtifact,
    document: dict[object, object],
    resource_name: str,
) -> None:
    artifact.roles.update({ArtifactRole.ORCHESTRATOR, ArtifactRole.DESIRED_STATE})
    artifact.structured_components.append((resource_name, AnalysisExecutor.GITOPS, 1))
    spec = document.get("spec")
    if not isinstance(spec, dict):
        return
    source = spec.get("source")
    path = source.get("path") if isinstance(source, dict) else None
    if isinstance(path, str) and not urlsplit(path).scheme:
        target = _normalize_path(PurePosixPath(path) / "kustomization.yaml")
        artifact.references.append((target, 1, AnalysisRelationType.REFERENCES))


def _reference_graph(
    parsed: dict[str, _ParsedArtifact],
) -> tuple[list[ReferenceNode], list[AnalysisRelation]]:
    nodes = [
        ReferenceNode(
            id=_node_id(item.source_id, item.path),
            repo_id=item.source_id,
            path=item.path,
            node_type=_node_type(item),
            roles=sorted(item.roles),
        )
        for item in parsed.values()
    ]
    relations: list[AnalysisRelation] = []
    for key, artifact in parsed.items():
        for target_path, line, relation in artifact.references:
            target_key = _node_key(artifact.source_id, target_path)
            if target_key not in parsed:
                continue
            relations.append(
                AnalysisRelation(
                    source=key,
                    target=target_key,
                    relation=relation,
                    strength=SignalStrength.STRUCTURAL_REFERENCE,
                    evidence=[
                        AnalysisSourceRef(
                            repo_id=artifact.source_id,
                            path=artifact.path,
                            start_line=line,
                            end_line=line,
                        )
                    ],
                )
            )
    return nodes, relations


def _discover_roots(
    files: list[RepositoryFile],
    parsed: dict[str, _ParsedArtifact],
    relations: list[AnalysisRelation],
    context: DeploymentContext,
) -> list[DeploymentRoot]:
    incoming = defaultdict(int)
    for relation in relations:
        incoming[relation.target] += 1
    hints = {_normalize_path(PurePosixPath(item)) for item in context.hints.documentation}
    role_candidates: dict[ArtifactRole, list[tuple[float, str, list[str]]]] = defaultdict(list)
    allowed_roles = {
        ArtifactRole.DEPLOYMENT_GUIDE,
        ArtifactRole.DEPLOYMENT_INDEX,
        ArtifactRole.ORCHESTRATOR,
        ArtifactRole.DESIRED_STATE,
        ArtifactRole.INITIALIZATION,
        ArtifactRole.OPERATIONS,
        ArtifactRole.VALIDATION,
    }
    for key, artifact in parsed.items():
        if artifact.scope == ArtifactScope.EXCLUDE:
            continue
        for role in artifact.roles & allowed_roles:
            score = artifact.structural_score + incoming[key] * 3
            reasons = [f"structural score {artifact.structural_score:.1f}"]
            if artifact.path in hints:
                score += 100
                reasons.append("deployment context hint")
            if _looks_deployment_path(artifact.path):
                score += 10
                reasons.append("deployment path")
            if (
                context.deployment.version
                and _path_version(artifact.path) == context.deployment.version
            ):
                score += 20
                reasons.append("version match")
            effective_profile = _effective_profile(context)
            artifact_profile = _path_profile(artifact.path)
            if (
                effective_profile
                and artifact_profile is not None
                and _profile_matches(artifact_profile, effective_profile)
            ):
                score += 100
                reasons.append("profile match")
            if any(
                term in artifact.path.casefold() for term in ("example", "sample", "deprecated")
            ):
                score -= 20
                reasons.append("example/deprecated penalty")
            if any(
                term in PurePosixPath(artifact.path).name.casefold()
                for term in ("delete", "remove", "uninstall", "cleanup")
            ):
                score -= 500
                reasons.append("destructive-entrypoint penalty")
            role_candidates[role].append((score, key, reasons))
    roots: list[DeploymentRoot] = []
    has_shell_orchestrator = any(
        key.split(":", 1)[-1].casefold().endswith(".sh")
        for _score, key, _reasons in role_candidates[ArtifactRole.ORCHESTRATOR]
    )
    for role, candidates in role_candidates.items():
        # A structural shell orchestrator already identifies the concrete units
        # it materializes. Treating arbitrary nested charts as additional roots
        # in that case creates subchart noise. Declarative-only repositories still
        # use DESIRED_STATE roots (Helm, Terraform, Kubernetes, and Kustomize).
        if role == ArtifactRole.DESIRED_STATE and has_shell_orchestrator:
            continue
        ranked = sorted(candidates, key=lambda item: (-item[0], item[1]))
        selected = _select_role_roots(role, ranked)
        for score, key, reasons in selected:
            artifact = parsed[key]
            roots.append(
                DeploymentRoot(
                    id=f"root-{_digest(key)[:16]}",
                    role=role,
                    source_ref=AnalysisSourceRef(
                        repo_id=artifact.source_id,
                        path=artifact.path,
                    ),
                    score=score,
                    selection_reasons=reasons,
                )
            )
    roots.sort(key=lambda item: (item.role, -item.score, item.source_ref.path))
    return roots


def _select_role_roots(
    role: ArtifactRole,
    ranked: list[tuple[float, str, list[str]]],
) -> list[tuple[float, str, list[str]]]:
    """Select at most two roots while preserving distinct deployment stages."""
    if role != ArtifactRole.ORCHESTRATOR:
        return ranked[:2]

    selected: list[tuple[float, str, list[str]]] = []
    selected_buckets: set[str] = set()
    for candidate in ranked:
        path = candidate[1].split(":", 1)[-1].casefold()
        if "/external/" in f"/{path}":
            bucket = "shared-services"
        elif any(
            marker in f"/{path}"
            for marker in ("/application/", "/apps/", "/services/", "/modules/", "/core/")
        ):
            bucket = "application"
        else:
            bucket = "other"
        if bucket in selected_buckets:
            continue
        selected.append(candidate)
        selected_buckets.add(bucket)
        if len(selected) == 2:
            return selected
    for candidate in ranked:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == 2:
            break
    return selected


def _traverse_roots(
    roots: list[DeploymentRoot],
    relations: list[AnalysisRelation],
    node_budget: int,
) -> set[str]:
    if node_budget < 1:
        raise ValueError("analysis node budget must be at least one")
    outgoing: dict[str, list[str]] = defaultdict(list)
    for relation in relations:
        outgoing[relation.source].append(relation.target)
    queue = deque(_node_key(root.source_ref.repo_id, root.source_ref.path) for root in roots)
    visited: set[str] = set()
    while queue and len(visited) < node_budget:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        queue.extend(
            sorted(target for target in outgoing.get(current, []) if target not in visited)
        )
    return visited


def _discover_signals(
    context: DeploymentContext,
    roots: list[DeploymentRoot],
    selected: set[str],
    parsed: dict[str, _ParsedArtifact],
    sources: dict[str, AcquiredSource],
) -> tuple[DeploymentSignalBundle, list[str]]:
    components: dict[str, CandidateComponent] = {}
    relations: list[AnalysisRelation] = []
    stages: list[StageSignal] = []
    validations: list[ValidationSignal] = []
    warnings: list[str] = []
    compose_invocations = _compose_invocation_bindings(selected, parsed)
    selected_orchestrators = [
        artifact
        for key, artifact in parsed.items()
        if key in selected and ArtifactRole.ORCHESTRATOR in artifact.roles
    ]
    for artifact in sorted(selected_orchestrators, key=lambda item: (item.source_id, item.path)):
        ordered_ids: list[str] = []
        for invocation in artifact.invocation_sequence:
            target_path = invocation.target
            name = _component_name_from_entrypoint(target_path)
            if not name:
                continue
            target_artifact = parsed.get(_node_key(artifact.source_id, target_path))
            candidate = _candidate_component(
                source_id=artifact.source_id,
                name=name,
                entrypoint=target_path,
                line=1,
                artifact_roles=(sorted(target_artifact.roles) if target_artifact else []),
                artifact_scope=(
                    target_artifact.scope if target_artifact else ArtifactScope.SUPPORTING
                ),
                evidence_strength=ComponentEvidenceStrength.EXPLICIT_ORCHESTRATOR,
                existence_locked=True,
                command=invocation.command,
                working_directory=invocation.working_directory,
                operation_source_ref=AnalysisSourceRef(
                    repo_id=artifact.source_id,
                    path=artifact.path,
                    start_line=invocation.line,
                    end_line=invocation.line,
                ),
            )
            _merge_candidate(components, candidate)
            if not ordered_ids or ordered_ids[-1] != candidate.id:
                ordered_ids.append(candidate.id)
        if ordered_ids:
            stages.append(
                StageSignal(
                    id=f"stage-{_digest(_node_key(artifact.source_id, artifact.path))[:16]}",
                    entrypoint=AnalysisSourceRef(
                        repo_id=artifact.source_id,
                        path=artifact.path,
                    ),
                    component_ids=ordered_ids,
                )
            )
            for source_id, target_id in zip(ordered_ids, ordered_ids[1:], strict=False):
                relations.append(
                    AnalysisRelation(
                        source=source_id,
                        target=target_id,
                        relation=AnalysisRelationType.ORDERED_BEFORE,
                        strength=SignalStrength.OBSERVED_EXECUTION_ORDER,
                        evidence=[
                            AnalysisSourceRef(repo_id=artifact.source_id, path=artifact.path)
                        ],
                    )
                )

    for key in sorted(selected):
        artifact = parsed.get(key)
        if artifact is None:
            continue
        validations.extend(artifact.validations)
        for candidate in _artifact_component_candidates(
            artifact,
            compose_binding=compose_invocations.get(key),
        ):
            _merge_candidate(components, candidate)
        relations.extend(artifact.explicit_relations)

    for component in components.values():
        component.validation_candidates.extend(
            signal
            for signal in validations
            if signal.subject.casefold() in {component.id.casefold(), component.name.casefold()}
        )
        if component.deployment is None and component.deployable:
            warnings.append(f"deployable component has no entrypoint: {component.name}")

    if not components:
        warnings.append("no deployable components were discovered from selected deployment roots")
    numbered_components, component_id_map = _number_components(list(components.values()))
    numbered_relations = [
        relation.model_copy(
            update={
                "source": component_id_map.get(relation.source, relation.source),
                "target": component_id_map.get(relation.target, relation.target),
            }
        )
        for relation in relations
    ]
    numbered_stages = [
        stage.model_copy(
            update={
                "component_ids": [
                    component_id_map.get(component_id, component_id)
                    for component_id in stage.component_ids
                ]
            }
        )
        for stage in stages
    ]
    signals = DeploymentSignalBundle(
        context_blocks=context.provided_blocks,
        deployment_roots=roots,
        candidate_components=numbered_components,
        relations=_deduplicate_relations(numbered_relations),
        deployment_stages=numbered_stages,
        validation_signals=validations,
        unresolved=(
            []
            if components
            else [
                AnalysisQuestion(
                    question="Which artifacts are the deployment entrypoints?",
                    reason="No component materialization path was discovered.",
                )
            ]
        ),
    )
    _validate_grounding(signals, sources)
    return signals, warnings


def _compose_invocation_bindings(
    selected: set[str],
    parsed: dict[str, _ParsedArtifact],
) -> dict[str, tuple[_ComposeInvocation, AnalysisSourceRef]]:
    bindings: dict[str, tuple[_ComposeInvocation, AnalysisSourceRef]] = {}
    for key in sorted(selected):
        artifact = parsed.get(key)
        if artifact is None or artifact.compose_invocation is None:
            continue
        invocation = artifact.compose_invocation
        source_ref = AnalysisSourceRef(
            repo_id=artifact.source_id,
            path=artifact.path,
            start_line=invocation.line,
            end_line=invocation.line,
        )
        for target in invocation.artifacts:
            target_key = _node_key(artifact.source_id, target)
            if target_key in selected:
                bindings.setdefault(target_key, (invocation, source_ref))
    return bindings


def _artifact_component_candidates(
    artifact: _ParsedArtifact,
    *,
    compose_binding: tuple[_ComposeInvocation, AnalysisSourceRef] | None = None,
) -> list[CandidateComponent]:
    ordinary = [
        item
        for item in artifact.structured_components
        if item[1] != AnalysisExecutor.TERRAFORM or "." not in item[0]
    ]
    terraform_resources = [
        item
        for item in artifact.structured_components
        if item[1] == AnalysisExecutor.TERRAFORM and "." in item[0]
    ]
    candidates = []
    for name, executor, line in ordinary:
        if _invalid_component_name(name):
            continue
        invocation, operation_source_ref = (
            compose_binding if compose_binding is not None else (None, None)
        )
        candidates.append(
            _candidate_component(
                source_id=artifact.source_id,
                name=name,
                entrypoint=artifact.path,
                line=line,
                executor=executor,
                artifact_roles=sorted(artifact.roles),
                artifact_scope=artifact.scope,
                command=(
                    invocation.command
                    if executor == AnalysisExecutor.DOCKER_COMPOSE and invocation is not None
                    else None
                ),
                working_directory=(
                    invocation.working_directory
                    if executor == AnalysisExecutor.DOCKER_COMPOSE and invocation is not None
                    else None
                ),
                operation_source_ref=(
                    operation_source_ref if executor == AnalysisExecutor.DOCKER_COMPOSE else None
                ),
            )
        )
    if terraform_resources:
        resource_names = [name for name, _executor, _line in terraform_resources]
        candidates.append(
            _candidate_component(
                source_id=artifact.source_id,
                name=_terraform_stack_name(artifact.source_id, artifact.path),
                entrypoint=artifact.path,
                line=min(line for _name, _executor, line in terraform_resources),
                executor=AnalysisExecutor.TERRAFORM,
                artifact_roles=sorted(artifact.roles),
                artifact_scope=artifact.scope,
                aliases=resource_names,
                materialized_names=resource_names,
            )
        )
    return candidates


def _merge_candidate(
    components: dict[str, CandidateComponent],
    candidate: CandidateComponent,
) -> None:
    existing = components.get(candidate.id)
    if existing is None:
        components[candidate.id] = candidate
        return
    existing.aliases = sorted(
        {*existing.aliases, *candidate.aliases},
        key=str.casefold,
    )
    existing.materialized_names = sorted(
        {*existing.materialized_names, *candidate.materialized_names},
        key=str.casefold,
    )
    evidence_rank = {
        ComponentEvidenceStrength.INFERRED: 0,
        ComponentEvidenceStrength.DOCUMENTED_COMMAND: 1,
        ComponentEvidenceStrength.DECLARED_DEPLOYMENT: 2,
        ComponentEvidenceStrength.EXPLICIT_ORCHESTRATOR: 3,
    }
    if evidence_rank[candidate.evidence_strength] > evidence_rank[existing.evidence_strength]:
        existing.evidence_strength = candidate.evidence_strength
        existing.source_ref = candidate.source_ref
        existing.deployment = candidate.deployment
        existing.artifact_roles = candidate.artifact_roles
        existing.artifact_scope = candidate.artifact_scope
    existing.existence_locked = existing.existence_locked or candidate.existence_locked


def _terraform_stack_name(source_id: str, path: str) -> str:
    artifact = PurePosixPath(path)
    generic = {
        "deploy",
        "deployment",
        "infra",
        "infrastructure",
        "terraform",
        "modules",
        "environments",
        "production",
        "prod",
    }
    for part in reversed(artifact.parent.parts):
        if part.casefold() not in generic and not part.startswith("."):
            return f"{part} infrastructure"
    if artifact.stem.casefold() not in {"main", "resources"}:
        return f"{artifact.stem} infrastructure"
    return f"{source_id} infrastructure stack"


def _candidate_component(
    *,
    source_id: str,
    name: str,
    entrypoint: str,
    line: int,
    executor: AnalysisExecutor = AnalysisExecutor.SHELL,
    artifact_roles: list[ArtifactRole] | None = None,
    artifact_scope: ArtifactScope = ArtifactScope.SUPPORTING,
    aliases: list[str] | None = None,
    materialized_names: list[str] | None = None,
    evidence_strength: ComponentEvidenceStrength | None = None,
    existence_locked: bool | None = None,
    command: str | None = None,
    working_directory: str | None = None,
    operation_source_ref: AnalysisSourceRef | None = None,
) -> CandidateComponent:
    normalized_name = _display_name(name)
    path = entrypoint.casefold()
    block_type, subtype, domain = _classify_unknown(path, normalized_name)
    component_id = slugify(normalized_name)
    classification = ComponentClassification(
        block_type=block_type,
        subtype=subtype,
        confidence=0.25,
        domain=domain,
    )
    implementation = normalized_name
    display_name = normalized_name
    capabilities: list[str] = []
    resolved_strength = evidence_strength or (
        ComponentEvidenceStrength.DOCUMENTED_COMMAND
        if ArtifactRole.DEPLOYMENT_GUIDE in (artifact_roles or [])
        else ComponentEvidenceStrength.DECLARED_DEPLOYMENT
    )
    locked = (
        existence_locked
        if existence_locked is not None
        else resolved_strength
        in {
            ComponentEvidenceStrength.EXPLICIT_ORCHESTRATOR,
            ComponentEvidenceStrength.DECLARED_DEPLOYMENT,
        }
    )
    return CandidateComponent(
        id=f"C000_{component_id}",
        name=display_name,
        implementation=implementation,
        deployable=True,
        external=False,
        aliases=sorted(
            {
                *(aliases or []),
                *([] if display_name.casefold() == name.casefold() else [name]),
            },
            key=str.casefold,
        ),
        capabilities=capabilities,
        materialized_names=materialized_names or [name],
        artifact_roles=artifact_roles or [],
        artifact_scope=artifact_scope,
        source_ref=AnalysisSourceRef(
            repo_id=source_id,
            path=entrypoint,
            start_line=line,
            end_line=line,
        ),
        deployment=ComponentDeployment(
            executor=executor,
            entrypoint=entrypoint,
            command=command,
            working_directory=working_directory,
            operation_source_ref=operation_source_ref,
        ),
        classification=classification,
        evidence_strength=resolved_strength,
        existence_locked=locked,
    )


def _number_components(
    components: list[CandidateComponent],
) -> tuple[list[CandidateComponent], dict[str, str]]:
    width = max(3, len(str(len(components))))
    result: list[CandidateComponent] = []
    id_map: dict[str, str] = {}
    for index, component in enumerate(components, start=1):
        name_slug = slugify(component.name)
        component_id = f"C{index:0{width}d}_{name_slug}"
        result.append(component.model_copy(update={"id": component_id}))
        id_map[component.id] = component_id
        for alias in {
            component.name,
            component.implementation or "",
            *component.aliases,
        }:
            if alias:
                id_map.setdefault(slugify(alias), component_id)
    return [
        component.model_copy(
            update={"installed_by": id_map.get(component.installed_by, component.installed_by)}
        )
        for component in result
    ], id_map


def _bind_retrieved_installation_routes(
    signals: DeploymentSignalBundle,
    observations: tuple[EvidenceObservation, ...],
    sources: dict[str, AcquiredSource],
) -> tuple[DeploymentSignalBundle, list[str]]:
    """Bind manifest-like components to nearby or calling installers.

    This is intentionally structural: it uses repository paths and explicit file
    references, never product names or a system-specific technology catalogue.
    """
    components = [
        component.model_copy(update={"deployment": _deployment_with_inputs(component, sources)})
        for component in signals.candidate_components
    ]
    by_entrypoint = {
        (component.source_ref.repo_id, component.deployment.entrypoint): component
        for component in components
        if component.deployment is not None
    }
    observations_by_component: dict[str, list[EvidenceObservation]] = defaultdict(list)
    for observation in observations:
        if observation.kind != EvidenceKind.INSTALLATION_ROUTE or not observation.query_id:
            continue
        component_id = observation.query_id.removeprefix("component-")
        observations_by_component[component_id].append(observation)

    updated: list[CandidateComponent] = []
    warnings: list[str] = []
    for component in components:
        deployment = component.deployment
        if deployment is None or deployment.command:
            updated.append(component)
            continue
        route = _best_installation_route(
            component,
            observations_by_component.get(component.id, []),
            sources,
        )
        if route is None:
            updated.append(component)
            continue
        owner = by_entrypoint.get((route.repo_id, route.path))
        owner_id = owner.id if owner is not None and owner.id != component.id else None
        route_path = PurePosixPath(route.path)
        route_deployment = ComponentDeployment(
            executor=AnalysisExecutor.SHELL,
            entrypoint=route.path,
            command=f"./{route_path.name}",
            working_directory=str(route_path.parent),
            operation_source_ref=route,
            required_inputs=_required_shell_inputs(
                _read_source_text_lines(sources[route.repo_id], route.path)
            ),
        )
        updated.append(
            component.model_copy(
                update={
                    "deployment": route_deployment,
                    "installed_by": owner_id,
                    "investigation_evidence_ids": sorted(
                        {
                            *component.investigation_evidence_ids,
                            *(
                                observation.id
                                for observation in observations_by_component.get(component.id, [])
                                if observation.path == route.path
                            ),
                        }
                    ),
                }
            )
        )
        relation = f" through {owner.name}" if owner_id and owner else ""
        warnings.append(f"bound {component.name} to installer {route.path}{relation}")
    return signals.model_copy(update={"candidate_components": updated}), warnings


def _best_installation_route(
    component: CandidateComponent,
    observations: list[EvidenceObservation],
    sources: dict[str, AcquiredSource],
) -> AnalysisSourceRef | None:
    assert component.deployment is not None
    original = PurePosixPath(component.deployment.entrypoint)
    names = {
        original.name.casefold(),
        original.stem.casefold(),
        component.name.casefold(),
        *(item.casefold() for item in component.aliases),
        *(item.casefold() for item in component.materialized_names),
    }
    ranked: list[tuple[int, EvidenceObservation, int]] = []
    for observation in observations:
        if observation.source_id != component.source_ref.repo_id:
            continue
        path = PurePosixPath(observation.path)
        if not is_installer_path(observation.path):
            continue
        lines = _read_source_text_lines(sources[observation.source_id], observation.path)
        matching_line = next(
            (
                number
                for number, line in enumerate(lines, start=1)
                if any(name and name in line.casefold() for name in names)
            ),
            1,
        )
        same_directory = path.parent == original.parent
        explicitly_references = matching_line != 1 or any(
            name and name in (lines[0].casefold() if lines else "") for name in names
        )
        score = (4 if same_directory else 0) + (3 if explicitly_references else 0)
        if path.parent == original.parent.parent:
            score += 2
        ranked.append((score, observation, matching_line))
    if not ranked:
        return None
    _score, observation, line = max(ranked, key=lambda item: (item[0], item[1].path))
    return AnalysisSourceRef(
        repo_id=observation.source_id,
        path=observation.path,
        start_line=line,
        end_line=line,
    )


def _read_source_text_lines(source: AcquiredSource, path: str) -> list[str]:
    try:
        candidate = source.resolve_path(path)
        return candidate.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError, ValueError:
        return []


def _deployment_with_inputs(
    component: CandidateComponent,
    sources: dict[str, AcquiredSource],
) -> ComponentDeployment | None:
    deployment = component.deployment
    if (
        deployment is None
        or deployment.executor != AnalysisExecutor.SHELL
        or deployment.required_inputs
        or component.source_ref.repo_id not in sources
    ):
        return deployment
    lines = _read_source_text_lines(sources[component.source_ref.repo_id], deployment.entrypoint)
    return deployment.model_copy(update={"required_inputs": _required_shell_inputs(lines)})


def _required_shell_inputs(lines: list[str]) -> list[str]:
    text = "\n".join(lines)
    environment = {f"env:{match.group('name')}" for match in _REQUIRED_ENV_INPUT.finditer(text)}
    positional = {f"arg:{match.group('number')}" for match in _POSITIONAL_INPUT.finditer(text)}
    return sorted(environment | positional)


def _apply_investigation_synthesis(
    signals: DeploymentSignalBundle,
    synthesis: InvestigationSynthesis,
    observations: tuple[EvidenceObservation, ...],
) -> DeploymentSignalBundle:
    """Apply complete, grounded classifications while retaining every invoked operation."""
    group_by_id = {
        component_id: group
        for group in synthesis.classification_groups
        for component_id in group.component_ids
    }
    rename_by_id = {rename.component_id: rename for rename in synthesis.renames}
    action_by_id = {action.candidate_id: action for action in synthesis.deployment_actions}
    original_by_id = {component.id: component for component in signals.candidate_components}
    classified: list[CandidateComponent] = []
    for component in signals.candidate_components:
        group = group_by_id.get(component.id)
        action = action_by_id.get(component.id)
        rename = rename_by_id.get(component.id)
        canonical_name = rename.canonical_name.strip() if rename else component.name
        if _invalid_component_name(canonical_name):
            canonical_name = component.name
        aliases = {*component.aliases, *(rename.aliases if rename else [])}
        if canonical_name.casefold() != component.name.casefold():
            aliases.add(component.name)
        disposition = (
            ComponentDisposition.IMPLEMENTATION_DETAIL
            if action
            else group.disposition
            if group
            else component.disposition
        )
        classification = (
            ComponentClassification(
                block_type=group.classification.block_type,
                subtype=group.classification.subtype,
                domain=group.domain,
                confidence=group.confidence,
            )
            if group
            else component.classification
        )
        evidence_ids = (
            action.evidence_ids
            if action
            else group.evidence_ids
            if group
            else component.investigation_evidence_ids
        )
        reason_code = (
            action.reason_code
            if action
            else group.reason_code
            if group
            else "DETERMINISTIC_CANDIDATE_RETAINED"
        )
        classified.append(
            component.model_copy(
                update={
                    "name": canonical_name,
                    "aliases": sorted(aliases, key=str.casefold),
                    "disposition": disposition,
                    "deployable": disposition
                    in {
                        ComponentDisposition.DEPLOYMENT_COMPONENT,
                        ComponentDisposition.UNCERTAIN,
                    },
                    "external": disposition == ComponentDisposition.EXTERNAL_DEPENDENCY,
                    "classification": classification,
                    "classification_source": (
                        "investigation_llm" if group or action else component.classification_source
                    ),
                    "classification_reason_code": reason_code,
                    "investigation_evidence_ids": sorted(set(evidence_ids)),
                }
            )
        )

    observation_by_id = {observation.id: observation for observation in observations}
    existing_names = {
        name.casefold() for component in classified for name in (component.name, *component.aliases)
    }
    external_requirements = list(signals.external_requirements)
    for entity in synthesis.implied_entities:
        if entity.canonical_name.casefold() in existing_names:
            continue
        if not entity.required_for_initial_deployment:
            # Operational and recovery capabilities are useful facts, but they
            # are not forward-deployment components unless the selected workflow
            # actually requires them.
            continue
        evidence = observation_by_id[entity.evidence_ids[0]]
        entrypoint_evidence = (
            observation_by_id.get(entity.entrypoint_evidence_id)
            if entity.entrypoint_evidence_id
            else None
        )
        disposition = entity.disposition
        deployment = None
        source_evidence = entrypoint_evidence or evidence
        if disposition == ComponentDisposition.DEPLOYMENT_COMPONENT:
            if entrypoint_evidence is None or entity.executor == AnalysisExecutor.UNKNOWN:
                disposition = ComponentDisposition.UNCERTAIN
            else:
                deployment = ComponentDeployment(
                    executor=entity.executor,
                    entrypoint=entrypoint_evidence.path,
                )
        candidate = CandidateComponent(
            id=f"C000_{slugify(entity.canonical_name)}",
            name=entity.canonical_name,
            implementation=entity.canonical_name,
            deployable=disposition
            in {ComponentDisposition.DEPLOYMENT_COMPONENT, ComponentDisposition.UNCERTAIN},
            external=disposition == ComponentDisposition.EXTERNAL_DEPENDENCY,
            aliases=entity.aliases,
            capabilities=entity.capabilities,
            materialized_names=[],
            disposition=disposition,
            source_ref=AnalysisSourceRef(
                repo_id=source_evidence.source_id,
                path=source_evidence.path,
                start_line=source_evidence.start_line,
                end_line=source_evidence.end_line,
            ),
            deployment=deployment,
            classification=ComponentClassification(
                block_type=entity.classification.block_type,
                subtype=entity.classification.subtype,
                confidence=entity.confidence,
            ),
            classification_source="investigation_llm",
            classification_reason_code="IMPLIED_BY_GROUNDED_EVIDENCE",
            evidence_strength=ComponentEvidenceStrength.INFERRED,
            existence_locked=False,
            investigation_evidence_ids=entity.evidence_ids,
        )
        classified.append(candidate)
        existing_names.update(
            {entity.canonical_name.casefold(), *(alias.casefold() for alias in entity.aliases)}
        )
        if disposition == ComponentDisposition.EXTERNAL_DEPENDENCY:
            external_requirements.append(
                ExternalRequirement(
                    name=entity.canonical_name,
                    source_ref=candidate.source_ref,
                )
            )

    numbered, id_map = _number_components(classified)
    numbered_actions = [
        DeploymentAction(
            id=f"A{index:03d}_{slugify(original_by_id[binding.candidate_id].name)}",
            name=next(
                component.name for component in classified if component.id == binding.candidate_id
            ),
            source_candidate_id=id_map[binding.candidate_id],
            owner_component_id=id_map[binding.owner_component_id],
            action_type=binding.action_type,
            source_ref=original_by_id[binding.candidate_id].source_ref,
            deployment=original_by_id[binding.candidate_id].deployment,
            evidence_ids=binding.evidence_ids,
        )
        for index, binding in enumerate(synthesis.deployment_actions, start=1)
    ]
    relations = [
        relation.model_copy(
            update={
                "source": id_map.get(relation.source, relation.source),
                "target": id_map.get(relation.target, relation.target),
            }
        )
        for relation in signals.relations
    ]
    component_lookup = _component_lookup(numbered)
    for fact in synthesis.facts:
        relation_type = {
            FactPredicate.REQUIRES: AnalysisRelationType.REQUIRES,
            FactPredicate.CONSUMES: AnalysisRelationType.REQUIRES,
            FactPredicate.PROVIDES: AnalysisRelationType.REQUIRES,
            FactPredicate.ORDERED_BEFORE: AnalysisRelationType.ORDERED_BEFORE,
        }.get(fact.predicate)
        if relation_type is None or fact.confidence < 0.7:
            continue
        source_id = component_lookup.get(fact.subject.casefold())
        target_id = component_lookup.get(fact.object.casefold())
        if fact.predicate == FactPredicate.PROVIDES:
            source_id, target_id = target_id, source_id
        if source_id is None or target_id is None or source_id == target_id:
            continue
        relations.append(
            AnalysisRelation(
                source=source_id,
                target=target_id,
                relation=relation_type,
                strength=SignalStrength.LLM_INFERRED,
                evidence=[
                    AnalysisSourceRef(
                        repo_id=observation_by_id[evidence_id].source_id,
                        path=observation_by_id[evidence_id].path,
                        start_line=observation_by_id[evidence_id].start_line,
                        end_line=observation_by_id[evidence_id].end_line,
                    )
                    for evidence_id in fact.evidence_ids
                ],
            )
        )
    stages = [
        stage.model_copy(
            update={
                "component_ids": [
                    id_map.get(component_id, component_id) for component_id in stage.component_ids
                ]
            }
        )
        for stage in signals.deployment_stages
    ]
    return signals.model_copy(
        update={
            "candidate_components": numbered,
            "deployment_actions": numbered_actions,
            "relations": _deduplicate_relations(relations),
            "deployment_stages": stages,
            "external_requirements": external_requirements,
            "unresolved": [*signals.unresolved, *synthesis.unresolved],
        }
    )


def _component_lookup(components: list[CandidateComponent]) -> dict[str, str]:
    result: dict[str, str] = {}
    for component in components:
        for name in (
            component.id,
            component.name,
            component.implementation or "",
            *component.aliases,
        ):
            if name:
                result.setdefault(name.casefold(), component.id)
    return result


def _classify_unknown(path: str, name: str) -> tuple[AnalysisBlockType, str, str | None]:
    if path.endswith(".tf") or "terraform" in path:
        return AnalysisBlockType.BASE_INFRASTRUCTURE, "cloud_infrastructure", None
    if any(term in path for term in ("monitor", "logging", "backup", "restore", "report")):
        return AnalysisBlockType.OPERATIONS, "observability", None
    if any(term in path for term in ("application", "service", "apps/", "modules/", "core/")):
        return AnalysisBlockType.APPLICATION, "domain_service", None
    if "/external/" in path:
        return AnalysisBlockType.SHARED_SERVICES, "external_integration", None
    if any(term in name.casefold() for term in ("ingress", "storage", "cluster")):
        return AnalysisBlockType.RUNTIME_ENVIRONMENT, "platform_services", None
    return AnalysisBlockType.APPLICATION, "core_application", None


def _component_name_from_entrypoint(path: str) -> str | None:
    if "$" in path or "{" in path or "}" in path:
        return None
    candidate = PurePosixPath(path)
    if candidate.name.casefold() not in {
        "install.sh",
        "deploy.sh",
        "setup.sh",
        "init_db.sh",
        "keycloak_init.sh",
        "cred.sh",
    }:
        return None
    parent = candidate.parent.name
    if parent in {"ansible", "all"}:
        parent = candidate.parent.parent.name
    return parent if parent and not _invalid_component_name(parent) else None


def _display_name(value: str) -> str:
    special = {
        "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL",
        "minio": "MinIO",
        "activemq": "ActiveMQ",
        "clamav": "ClamAV",
        "softhsm": "SoftHSM",
        "rke2": "RKE2",
        "iam": "Keycloak",
        "smtp": "SMTP",
    }
    return special.get(value.casefold(), value.replace("_", " ").replace("-", " ").title())


def _invalid_component_name(name: str) -> bool:
    return not name.strip() or bool(_FORBIDDEN_COMPONENT.search(name.strip()))


def _logical_shell_lines(text: str) -> list[tuple[int, str]]:
    result = []
    parts: list[str] = []
    start = 1
    for line_number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not parts:
            start = line_number
        if not stripped or stripped.startswith("#"):
            continue
        parts.append(stripped.removesuffix("\\").strip())
        if stripped.endswith("\\"):
            continue
        result.append((start, " ".join(parts)))
        parts = []
    if parts:
        result.append((start, " ".join(parts)))
    return result


def _shell_cd_target(
    command: str, root_dir: PurePosixPath, script_dir: PurePosixPath
) -> PurePosixPath | None:
    match = re.match(r"^cd\s+(?P<target>[^;&|]+)", command)
    if not match:
        return None
    raw = match.group("target").strip().strip("\"'")
    if "$ROOT_DIR" in raw or "${ROOT_DIR}" in raw:
        suffix = raw.replace("${ROOT_DIR}", "").replace("$ROOT_DIR", "").lstrip("/")
        return PurePosixPath(_normalize_path(root_dir / suffix))
    if raw.startswith("/") or "$" in raw:
        return script_dir
    return PurePosixPath(_normalize_path(script_dir / raw))


def _source_ref_exists(source: AcquiredSource, relative_path: str) -> bool:
    return source.contains_file(relative_path)


def _validate_grounding(
    signals: DeploymentSignalBundle, sources: dict[str, AcquiredSource]
) -> None:
    for component in signals.candidate_components:
        if component.source_ref.repo_id not in sources:
            raise ValueError(f"component references unknown source: {component.name}")
        if component.deployable and not _source_ref_exists(
            sources[component.source_ref.repo_id], component.source_ref.path
        ):
            raise ValueError(
                f"component entrypoint does not exist: {component.source_ref.repo_id}:"
                f"{component.source_ref.path}"
            )


def _deduplicate_relations(relations: list[AnalysisRelation]) -> list[AnalysisRelation]:
    strength_rank = {
        SignalStrength.LLM_INFERRED: 0,
        SignalStrength.STRUCTURAL_REFERENCE: 1,
        SignalStrength.DECLARED_ORDER: 2,
        SignalStrength.OBSERVED_EXECUTION_ORDER: 3,
        SignalStrength.EXPLICIT_DEPENDENCY: 4,
        SignalStrength.EXPLICIT_HEALTH_DEPENDENCY: 5,
    }
    result: dict[tuple[str, str, AnalysisRelationType], AnalysisRelation] = {}
    for relation in relations:
        if relation.source == relation.target:
            continue
        key = (relation.source, relation.target, relation.relation)
        existing = result.get(key)
        if existing is None or strength_rank[relation.strength] > strength_rank[existing.strength]:
            result[key] = relation
    return sorted(result.values(), key=lambda item: (item.source, item.target, item.relation))


def _node_type(artifact: _ParsedArtifact) -> str:
    if ArtifactRole.ORCHESTRATOR in artifact.roles:
        return "SCRIPT" if artifact.path.endswith(".sh") else "DEPLOYMENT_UNIT"
    if ArtifactRole.DEPLOYMENT_GUIDE in artifact.roles:
        return "DOCUMENT"
    if artifact.path.endswith(".tf"):
        return "TERRAFORM_MODULE"
    if PurePosixPath(artifact.path).name.casefold() == "chart.yaml":
        return "HELM_CHART"
    return "DEPLOYMENT_UNIT"


def _wrong_context(file: RepositoryFile, context: DeploymentContext) -> bool:
    expected_version = context.deployment.version
    if expected_version and file.version_context and file.version_context != expected_version:
        return True
    expected_profile = _effective_profile(context)
    return bool(
        expected_profile
        and file.profile_context
        and not _profile_matches(file.profile_context, expected_profile)
    )


def _effective_profile(context: DeploymentContext) -> str | None:
    profile = context.deployment.profile
    if not profile or profile.casefold().replace("_", "-") != "cloud":
        return profile
    provider = next(
        (
            block.implementation
            for block in context.provided_blocks
            if block.type == AnalysisBlockType.BASE_INFRASTRUCTURE
            and (block.implementation or "").casefold()
            in {"aws", "azure", "gcp", "google-cloud", "openstack"}
        ),
        None,
    )
    return provider or profile


def _profile_matches(detected: str | None, expected: str) -> bool:
    if detected is None:
        return True
    normalized = expected.casefold().replace("_", "-")
    if normalized.startswith("on-prem"):
        return detected == "on-premises"
    if normalized == "cloud":
        return detected in {"aws", "azure", "gcp"}
    if normalized == "google-cloud":
        return detected == "gcp"
    return detected == normalized


def _path_version(path: str) -> str | None:
    match = _VERSION.search(path)
    return match.group(1) if match else None


def _path_profile(path: str) -> str | None:
    lowered = path.casefold()
    if "on-prem" in lowered or "on_prem" in lowered:
        return "on-premises"
    if "aws" in lowered:
        return "aws"
    if "azure" in lowered:
        return "azure"
    if "gcp" in lowered or "google-cloud" in lowered:
        return "gcp"
    return None


def _looks_deployment_path(path: str) -> bool:
    lowered = path.casefold()
    return any(
        term in lowered
        for term in ("deploy", "install", "setup", "k8s", "terraform", "ansible", "helm")
    )


def _normalize_path(path: PurePosixPath) -> str:
    parts: list[str] = []
    for part in path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def _node_key(source_id: str, path: str) -> str:
    return f"{source_id}:{path}"


def _node_id(source_id: str, path: str) -> str:
    return f"artifact-{_digest(_node_key(source_id, path))[:20]}"


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
