from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import yaml

from pheragent.utils import slugify

from .analysis_llm import (
    AnalysisLLMConfig,
    ComponentClassificationOutcome,
    classify_components_with_llm,
)
from .analysis_models import (
    AnalysisBlockType,
    AnalysisEvaluation,
    AnalysisExecutor,
    AnalysisQuestion,
    AnalysisRelation,
    AnalysisRelationType,
    AnalysisSourceRef,
    ArtifactRole,
    CandidateComponent,
    ComponentClassification,
    ComponentDeployment,
    DeploymentContext,
    DeploymentRoot,
    DeploymentSignalBundle,
    FunctionalBlock,
    FunctionalBlocksDocument,
    FunctionalComponent,
    FunctionalDeployRef,
    GoldDefinition,
    ReferenceNode,
    RepositoryFile,
    SignalStrength,
    StageSignal,
    ValidationSignal,
)
from .enums import InventoryCategory, SourceKind
from .inventory import RepositoryInventoryBuilder
from .models import RepositoryInventory, SourcesConfig, SourceSpec
from .source_manager import AcquiredSource, AcquisitionResult, SourceManager

ProgressCallback = Callable[[str], None]

_VERSION = re.compile(r"(?<!\d)(\d+\.\d+\.\d+(?:\.\d+)?)(?!\d)")
_MARKDOWN_LINK = re.compile(r"\[[^]]+]\((?P<target>[^)]+)\)")
_SHELL_INVOCATION = re.compile(
    r"(?:^|[;&|]\s*)(?P<target>(?:\./|\.\./)[A-Za-z0-9_./${}\"'-]+\.sh)(?:\s|$)"
)
_MODULE_ARRAY = re.compile(
    r"declare\s+-a\s+module\s*=\s*\((?P<body>.*?)\)", re.DOTALL | re.IGNORECASE
)
_QUOTED_ITEM = re.compile(r"[\"'](?P<item>[A-Za-z0-9_.-]+)[\"']")
_TERRAFORM_MODULE = re.compile(r'(?m)^\s*module\s+"(?P<name>[^"]+)"\s*\{')
_TERRAFORM_RESOURCE = re.compile(
    r'(?m)^\s*resource\s+"(?P<kind>[^"]+)"\s+"(?P<name>[^"]+)"\s*\{'
)
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


@dataclass(slots=True)
class AnalysisConfig:
    repositories: list[str]
    documentation: list[str]
    context_path: Path
    cache_dir: Path
    node_budget: int = 200
    strict: bool = False
    source_timeout: float = 900.0
    gold_path: Path | None = None
    synthesizer: str = "auto"
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    base_url: str | None = None
    llm_timeout: float = 120.0
    llm_max_output_tokens: int = 3000
    llm_cache_dir: Path | None = None
    retry_failed_llm: bool = False


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    acquisition: AcquisitionResult
    inventory: RepositoryInventory
    repository_files: tuple[RepositoryFile, ...]
    reference_nodes: tuple[ReferenceNode, ...]
    reference_relations: tuple[AnalysisRelation, ...]
    signals: DeploymentSignalBundle
    document: FunctionalBlocksDocument
    selected_paths: tuple[str, ...]
    warnings: tuple[str, ...]
    used_synthesizer: str
    llm_usage: dict[str, int]
    llm_failure_history_path: Path | None


@dataclass(slots=True)
class _ParsedArtifact:
    source_id: str
    path: str
    roles: set[ArtifactRole] = field(default_factory=set)
    references: list[tuple[str, int, AnalysisRelationType]] = field(default_factory=list)
    invocation_sequence: list[tuple[str, int]] = field(default_factory=list)
    validations: list[ValidationSignal] = field(default_factory=list)
    structured_components: list[tuple[str, AnalysisExecutor, int]] = field(default_factory=list)
    explicit_relations: list[AnalysisRelation] = field(default_factory=list)
    structural_score: float = 0.0


@dataclass(frozen=True, slots=True)
class _Technology:
    canonical: str
    aliases: tuple[str, ...]
    block_type: AnalysisBlockType
    subtype: str
    capabilities: tuple[str, ...]


def run_repository_analysis(
    config: AnalysisConfig,
    *,
    progress: ProgressCallback | None = None,
) -> AnalysisResult:
    notify = progress or (lambda _message: None)
    context = _load_context(config.context_path)
    sources_config = _sources_config(config.repositories, config.documentation, context.system)
    notify(f"acquiring {len(sources_config.sources)} source(s)")
    acquisition = SourceManager(
        cache_dir=config.cache_dir,
        config_dir=config.context_path.expanduser().resolve().parent,
        strict=config.strict,
        timeout=config.source_timeout,
        progress=notify,
    ).acquire(sources_config)
    notify("inventorying repositories")
    inventory = RepositoryInventoryBuilder().build(acquisition.sources)
    source_by_id = {source.id: source for source in acquisition.sources}
    files = _repository_files(inventory, context)
    notify(f"detecting artifact roles across {len(files)} file(s)")
    parsed = _parse_artifacts(files, source_by_id, context)
    files = [
        file.model_copy(
            update={
                "deployment_roles": sorted(
                    parsed.get(_node_key(file.repo_id, file.path), _ParsedArtifact("", "")).roles
                )
            }
        )
        for file in files
    ]
    nodes, reference_relations = _reference_graph(parsed)
    roots = _discover_roots(files, parsed, reference_relations, context)
    selected = _traverse_roots(roots, reference_relations, config.node_budget)
    notify(f"selected {len(roots)} deployment root(s) and {len(selected)} graph node(s)")
    signals, warnings = _discover_signals(
        context,
        roots,
        selected,
        parsed,
        source_by_id,
    )
    classification_outcome = classify_components_with_llm(
        signals,
        config=AnalysisLLMConfig(
            mode=config.synthesizer,
            model=config.model,
            api_key_env=config.api_key_env,
            base_url_env=config.base_url_env,
            base_url=config.base_url,
            timeout=config.llm_timeout,
            max_output_tokens=config.llm_max_output_tokens,
            cache_dir=config.llm_cache_dir,
            retry_failed=config.retry_failed_llm,
        ),
    )
    if classification_outcome.warning:
        warnings.append(classification_outcome.warning)
    gold = _load_gold(config.gold_path)
    document = _synthesize(context, signals, source_by_id, gold, classification_outcome)
    return AnalysisResult(
        acquisition=acquisition,
        inventory=inventory,
        repository_files=tuple(files),
        reference_nodes=tuple(nodes),
        reference_relations=tuple(reference_relations),
        signals=signals,
        document=document,
        selected_paths=tuple(sorted(selected)),
        warnings=tuple(warnings),
        used_synthesizer=classification_outcome.used,
        llm_usage=classification_outcome.usage,
        llm_failure_history_path=classification_outcome.failure_history_path,
    )


def render_analysis_report(result: AnalysisResult) -> str:
    document = result.document
    lines = [
        f"# Deployment Analysis: {document.system}",
        "",
        "## Repositories inspected",
        "",
    ]
    for source in result.acquisition.sources:
        revision = source.manifest.resolved_revision or source.manifest.content_hash
        lines.append(f"- `{source.id}`: `{revision}`")
    lines.extend(
        [
            "",
            "## Technologies detected",
            "",
            ", ".join(result.inventory.detected_technologies) or "None.",
            "",
            "## Deployment entrypoints selected",
            "",
        ]
    )
    for root in result.signals.deployment_roots:
        lines.append(
            f"- `{root.role}`: `{root.source_ref.repo_id}:{root.source_ref.path}` "
            f"(score {root.score:.1f})"
        )
    lines.extend(
        [
            "",
            "## Candidate path",
            "",
            f"- Graph nodes traversed: {len(result.selected_paths)}",
            f"- Candidate components: {len(result.signals.candidate_components)}",
            f"- Source-derived relations: {document.evaluation.source_derived_relation_count}",
        ]
    )
    for index, level in enumerate(document.levels):
        lines.append(
            f"- DAG level {index}: {', '.join(f'`{item}`' for item in level)}"
        )
    lines.extend(["", "## Unresolved questions", ""])
    lines.extend(
        [f"- **{item.question}** {item.reason}" for item in document.unresolved] or ["None."]
    )
    lines.extend(["", "## Warnings", ""])
    lines.extend([f"- {warning}" for warning in result.warnings] or ["None."])
    evaluation = document.evaluation
    lines.extend(
        [
            "",
            "## Evaluation summary",
            "",
            f"- Components: {evaluation.component_count}",
            f"- Deployability coverage: {evaluation.deployability_coverage:.1%}",
            f"- Grounded component rate: {evaluation.grounded_component_rate:.1%}",
            f"- Forbidden components: {evaluation.forbidden_component_count}",
            f"- Relations: {evaluation.relation_count}",
            f"- Artifact lines: {evaluation.artifact_line_count}",
            f"- LLM input tokens: {evaluation.llm_input_tokens}",
            f"- LLM requests this run: {evaluation.llm_requests}",
            f"- Synthesizer: {result.used_synthesizer}",
        ]
    )
    if result.llm_failure_history_path:
        lines.append(f"- LLM failure history: `{result.llm_failure_history_path}`")
    for label, value in (
        ("Component precision", evaluation.component_precision),
        ("Component recall", evaluation.component_recall),
        ("Classification accuracy", evaluation.classification_accuracy),
        ("Edge precision", evaluation.edge_precision),
        ("Edge recall", evaluation.edge_recall),
        ("Entrypoint accuracy", evaluation.entrypoint_accuracy),
        ("Pairwise grouping F1", evaluation.grouping_f1),
    ):
        if value is not None:
            lines.append(f"- {label}: {value:.1%}")
    lines.append(f"- Hallucination rate: {evaluation.hallucination_rate:.1%}")
    lines.append("")
    return "\n".join(lines)


def _load_context(path: Path) -> DeploymentContext:
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deployment context must contain a YAML mapping")
    return DeploymentContext.model_validate(payload)


def _load_gold(path: Path | None) -> GoldDefinition | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("gold definition must contain a YAML mapping")
    return GoldDefinition.model_validate(payload)


def _sources_config(
    repositories: list[str], documentation: list[str], system: str
) -> SourcesConfig:
    specs: list[SourceSpec] = []
    used_ids: set[str] = set()
    for kind, locations in (("repo", repositories), ("docs", documentation)):
        for index, location in enumerate(locations, start=1):
            candidate = _source_id(location, fallback=f"{kind}-{index}")
            source_id = candidate
            suffix = 2
            while source_id in used_ids:
                source_id = f"{candidate}-{suffix}"
                suffix += 1
            used_ids.add(source_id)
            source_kind = _source_kind(location)
            resolved_location = (
                str(Path(location).expanduser().resolve())
                if source_kind != SourceKind.GIT
                else location
            )
            specs.append(
                SourceSpec(
                    id=source_id,
                    kind=source_kind,
                    location=resolved_location,
                )
            )
    if not specs:
        raise ValueError("deployment analysis requires at least one repository or docs source")
    return SourcesConfig(system=system, sources=specs)


def _source_kind(location: str) -> SourceKind:
    if urlsplit(location).scheme in {"git", "http", "https", "ssh"} or location.startswith(
        "git@"
    ):
        return SourceKind.GIT
    path = Path(location).expanduser()
    return SourceKind.LOCAL_FILE if path.is_file() else SourceKind.LOCAL_DIRECTORY


def _source_id(location: str, *, fallback: str) -> str:
    parsed = urlsplit(location)
    raw = Path(parsed.path or location).name.removesuffix(".git") or fallback
    return slugify(raw, fallback=fallback)


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
        path = _source_path(sources[file.repo_id], file.path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
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
            match.group("item")
            for match in _QUOTED_ITEM.finditer(module_match.group("body"))
        ]
    root_dir = path.parent.parent
    if modules:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
        for index, module in enumerate(modules, start=1):
            target = _normalize_path(root_dir / module / "install.sh")
            artifact.references.append((target, index, AnalysisRelationType.INVOKES))
            artifact.invocation_sequence.append((target, index))

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
            artifact.references.append((target, line_number, AnalysisRelationType.INVOKES))
            artifact.invocation_sequence.append((target, line_number))
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

    if len({target for target, _ in artifact.invocation_sequence}) >= 2:
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
    for line_number, line in enumerate(text.splitlines(), start=1):
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
        artifact.structured_components.append(
            (component_name, AnalysisExecutor.TERRAFORM, line)
        )
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
                                AnalysisSourceRef(
                                    repo_id=artifact.source_id, path=artifact.path
                                )
                            ],
                        )
                    )
    if name in {"compose.yaml", "compose.yml", "docker-compose.yaml", "docker-compose.yml"}:
        artifact.roles.add(ArtifactRole.ORCHESTRATOR)
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
            artifact.structured_components.append(
                (component_name, AnalysisExecutor.KUSTOMIZE, 1)
            )
        for document in documents:
            for field_name in ("resources", "bases", "components"):
                entries = document.get(field_name, [])
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if not isinstance(entry, str) or urlsplit(entry).scheme:
                        continue
                    target = _normalize_path(
                        PurePosixPath(artifact.path).parent / entry
                    )
                    if not PurePosixPath(target).suffix:
                        target = _normalize_path(
                            PurePosixPath(target) / "kustomization.yaml"
                        )
                    artifact.references.append(
                        (target, 1, AnalysisRelationType.COMPOSES)
                    )
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
            artifact.structured_components.append(
                (resource_name, AnalysisExecutor.KUBERNETES, 1)
            )
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
                    else role.get("role") if isinstance(role, dict) else None
                )
                if isinstance(role_name, str):
                    ordered_roles.append(role_name)
                    artifact.structured_components.append(
                        (role_name, AnalysisExecutor.ANSIBLE, 1)
                    )
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
                        target = _normalize_path(
                            PurePosixPath(artifact.path).parent / task_path
                        )
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
                term in artifact.path.casefold()
                for term in ("example", "sample", "deprecated")
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
        elif any(marker in f"/{path}" for marker in ("/mosip/", "/application/", "/apps/")):
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
    queue = deque(
        _node_key(root.source_ref.repo_id, root.source_ref.path) for root in roots
    )
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
    registry = _technology_registry()
    alias_registry = {
        alias.casefold(): item
        for item in registry.values()
        for alias in (item.canonical, *item.aliases)
    }
    components: dict[str, CandidateComponent] = {}
    relations: list[AnalysisRelation] = []
    stages: list[StageSignal] = []
    validations: list[ValidationSignal] = []
    warnings: list[str] = []
    selected_orchestrators = [
        artifact
        for key, artifact in parsed.items()
        if key in selected and ArtifactRole.ORCHESTRATOR in artifact.roles
    ]
    for artifact in sorted(selected_orchestrators, key=lambda item: (item.source_id, item.path)):
        ordered_ids: list[str] = []
        for target_path, line in artifact.invocation_sequence:
            name = _component_name_from_entrypoint(target_path)
            if not name:
                continue
            candidate = _candidate_component(
                source_id=artifact.source_id,
                name=name,
                entrypoint=target_path,
                line=line,
                registry=alias_registry,
            )
            if candidate.id not in components:
                components[candidate.id] = candidate
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
        for name, executor, line in artifact.structured_components:
            if _invalid_component_name(name):
                continue
            candidate = _candidate_component(
                source_id=artifact.source_id,
                name=name,
                entrypoint=artifact.path,
                line=line,
                executor=executor,
                registry=alias_registry,
            )
            components.setdefault(candidate.id, candidate)
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


def _candidate_component(
    *,
    source_id: str,
    name: str,
    entrypoint: str,
    line: int,
    registry: dict[str, _Technology],
    executor: AnalysisExecutor = AnalysisExecutor.SHELL,
) -> CandidateComponent:
    normalized_name = _display_name(name)
    technology = registry.get(name.casefold()) or registry.get(normalized_name.casefold())
    path = entrypoint.casefold()
    if technology is not None:
        component_id = slugify(technology.canonical)
        classification = ComponentClassification(
            block_type=technology.block_type,
            subtype=technology.subtype,
            confidence=1.0,
        )
        implementation = technology.canonical
        display_name = _display_name(technology.canonical)
        capabilities = list(technology.capabilities)
    else:
        block_type, subtype, domain = _classify_unknown(path, normalized_name)
        component_id = slugify(normalized_name)
        classification = ComponentClassification(
            block_type=block_type,
            subtype=subtype,
            confidence=0.8 if block_type == AnalysisBlockType.APPLICATION else 0.65,
            domain=domain,
        )
        implementation = normalized_name
        display_name = normalized_name
        capabilities = []
    return CandidateComponent(
        id=f"C000_{component_id}",
        name=display_name,
        implementation=implementation,
        deployable=True,
        external=False,
        aliases=[] if display_name.casefold() == name.casefold() else [name],
        capabilities=capabilities,
        source_ref=AnalysisSourceRef(
            repo_id=source_id,
            path=entrypoint,
            start_line=line,
            end_line=line,
        ),
        deployment=ComponentDeployment(executor=executor, entrypoint=entrypoint),
        classification=classification,
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
    return result, id_map


def _synthesize(
    context: DeploymentContext,
    signals: DeploymentSignalBundle,
    sources: dict[str, AcquiredSource],
    gold: GoldDefinition | None,
    classification_outcome: ComponentClassificationOutcome,
) -> FunctionalBlocksDocument:
    blocks = [
        FunctionalBlock(
            id=item.id,
            name=_display_name(item.subtype),
            type=item.type,
            subtype=item.subtype,
            implementation=item.implementation,
            state=item.state,
            after=item.after,
            provides=item.provides,
        )
        for item in signals.context_blocks
    ]
    next_id = max(
        (int(match.group(1)) for block in blocks if (match := re.fullmatch(r"B(\d+)", block.id))),
        default=-1,
    ) + 1
    for name, block_type, subtype, components in _synthesis_groups(
        signals, classification_outcome
    ):
        block_id = f"B{next_id}"
        next_id += 1
        blocks.append(
            FunctionalBlock(
                id=block_id,
                name=name,
                type=block_type,
                subtype=subtype,
                provides=sorted(
                    {
                        capability
                        for component in components
                        for capability in component.capabilities
                    }
                ),
                components=[
                    FunctionalComponent(
                        id=component.id,
                        name=component.name,
                        implementation=component.implementation,
                        deployable=component.deployable,
                        external=component.external,
                        deploy=(
                            FunctionalDeployRef(
                                executor=component.deployment.executor,
                                ref=component.deployment.entrypoint,
                                repo_id=component.source_ref.repo_id,
                            )
                            if component.deployment
                            else None
                        ),
                    )
                    for component in components
                ],
            )
        )
    _attach_block_dependencies(blocks)
    levels = _topological_levels(blocks)
    evaluation = _evaluate(blocks, signals, sources, gold)
    evaluation = evaluation.model_copy(
        update={
            "llm_input_tokens": int(classification_outcome.usage.get("input_tokens", 0)),
            "llm_requests": int(classification_outcome.usage.get("requests", 0)),
            "llm_status": classification_outcome.used,
        }
    )
    document = FunctionalBlocksDocument(
        system=context.system,
        deployment=context.deployment,
        blocks=blocks,
        levels=levels,
        unresolved=[
            *signals.unresolved,
            *(
                classification_outcome.classification.unresolved
                if classification_outcome.classification
                else []
            ),
        ],
        evaluation=evaluation,
    )
    rendered = yaml.safe_dump(
        document.model_dump(mode="json", exclude_none=True), sort_keys=False, allow_unicode=True
    )
    updated_evaluation = evaluation.model_copy(
        update={"artifact_line_count": len(rendered.splitlines())}
    )
    return document.model_copy(update={"evaluation": updated_evaluation})


def _synthesis_groups(
    signals: DeploymentSignalBundle,
    classification_outcome: ComponentClassificationOutcome,
) -> list[tuple[str, AnalysisBlockType, str, list[CandidateComponent]]]:
    grouped: dict[
        tuple[AnalysisBlockType, str, str | None], list[CandidateComponent]
    ] = defaultdict(list)
    for component in signals.candidate_components:
        classification = (
            classification_outcome.classification.assignments[component.id]
            if classification_outcome.classification
            else component.classification
        )
        grouped[
            (classification.block_type, classification.subtype, classification.domain)
        ].append(component)
    return [
        (_block_name(subtype, domain), block_type, subtype, components)
        for (block_type, subtype, domain), components in sorted(
            grouped.items(),
            key=lambda item: (item[0][0].value, item[0][1], item[0][2] or ""),
        )
    ]


def _attach_block_dependencies(blocks: list[FunctionalBlock]) -> None:
    by_type: dict[AnalysisBlockType, list[FunctionalBlock]] = defaultdict(list)
    for block in blocks:
        by_type[block.type].append(block)
    base_ids = [item.id for item in by_type[AnalysisBlockType.BASE_INFRASTRUCTURE]]
    runtime_ids = [item.id for item in by_type[AnalysisBlockType.RUNTIME_ENVIRONMENT]]
    shared_ids = [item.id for item in by_type[AnalysisBlockType.SHARED_SERVICES]]
    for block in blocks:
        if block.after:
            continue
        if block.type == AnalysisBlockType.RUNTIME_ENVIRONMENT:
            block.after.extend(base_ids)
        elif block.type in {AnalysisBlockType.SHARED_SERVICES, AnalysisBlockType.OPERATIONS}:
            block.after.extend(runtime_ids or base_ids)
        elif block.type == AnalysisBlockType.APPLICATION:
            block.after.extend(shared_ids or runtime_ids or base_ids)
        block.after = sorted(set(item for item in block.after if item != block.id))


def _topological_levels(blocks: list[FunctionalBlock]) -> list[list[str]]:
    block_ids = {block.id for block in blocks}
    dependencies = {
        block.id: {item for item in block.after if item in block_ids} for block in blocks
    }
    levels: list[list[str]] = []
    emitted: set[str] = set()
    while len(emitted) < len(blocks):
        ready = sorted(
            block_id
            for block_id, required in dependencies.items()
            if block_id not in emitted and required <= emitted
        )
        if not ready:
            raise ValueError("functional block graph contains a cycle")
        levels.append(ready)
        emitted.update(ready)
    return levels


def _evaluate(
    blocks: list[FunctionalBlock],
    signals: DeploymentSignalBundle,
    sources: dict[str, AcquiredSource],
    gold: GoldDefinition | None,
) -> AnalysisEvaluation:
    components = [component for block in blocks for component in block.components]
    deployable = [component for component in components if component.deployable]
    grounded = [
        component
        for component in components
        if component.deploy is not None
        and _source_ref_exists(
            sources[component.deploy.repo_id], component.deploy.ref
        )
    ]
    forbidden = [component for component in components if _invalid_component_name(component.name)]
    total = len(components)
    evaluation = AnalysisEvaluation(
        component_count=total,
        deployable_component_count=len(deployable),
        deployability_coverage=(
            sum(component.deploy is not None for component in deployable) / len(deployable)
            if deployable
            else 1.0
        ),
        grounded_component_rate=len(grounded) / total if total else 1.0,
        forbidden_component_count=len(forbidden),
        relation_count=len(signals.relations),
        source_derived_relation_count=sum(
            relation.strength != SignalStrength.LLM_INFERRED for relation in signals.relations
        ),
        hallucination_rate=1 - (len(grounded) / total if total else 1.0),
    )
    if gold is None:
        return evaluation
    predicted = {component.name.casefold(): component for component in components}
    expected = {name.casefold() for name in gold.expected_components}
    forbidden_names = {name.casefold() for name in gold.forbidden_components}
    correct = len(set(predicted) & expected)
    false_positive = len(set(predicted) - expected - forbidden_names) + len(
        set(predicted) & forbidden_names
    )
    precision = correct / (correct + false_positive) if correct + false_positive else 1.0
    recall = correct / len(expected) if expected else 1.0
    classifications = []
    for name, expected_classification in gold.expected_classifications.items():
        predicted_component = predicted.get(name.casefold())
        classifications.append(
            predicted_component is not None
            and _component_block(blocks, predicted_component.id).type
            == expected_classification.block_type
            and _component_block(blocks, predicted_component.id).subtype
            == expected_classification.subtype
        )
    entrypoints = []
    for name, expected_path in gold.expected_entrypoints.items():
        component = predicted.get(name.casefold())
        entrypoints.append(
            component is not None
            and component.deploy is not None
            and component.deploy.ref == expected_path
        )
    edge_precision, edge_recall = _edge_metrics(blocks, gold)
    grouping_f1 = _grouping_f1(blocks, gold.expected_groups)
    return evaluation.model_copy(
        update={
            "component_precision": precision,
            "component_recall": recall,
            "classification_accuracy": (
                sum(classifications) / len(classifications) if classifications else None
            ),
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "entrypoint_accuracy": sum(entrypoints) / len(entrypoints) if entrypoints else None,
            "grouping_f1": grouping_f1,
        }
    )


def _edge_metrics(
    blocks: list[FunctionalBlock], gold: GoldDefinition
) -> tuple[float | None, float | None]:
    if not gold.expected_major_edges:
        return None, None
    block_by_id = {block.id: block for block in blocks}
    predicted = {
        (block_by_id[parent].type.value, block.type.value)
        for block in blocks
        for parent in block.after
        if parent in block_by_id
    }
    expected = {(edge.source, edge.target) for edge in gold.expected_major_edges}
    correct = len(predicted & expected)
    return (
        correct / len(predicted) if predicted else 0.0,
        correct / len(expected) if expected else 1.0,
    )


def _grouping_f1(blocks: list[FunctionalBlock], expected_groups: list[list[str]]) -> float | None:
    if not expected_groups:
        return None
    # Gold files may intentionally score only representative components. Pairs
    # containing components outside that declared universe are not labeled and
    # therefore must not be counted as false positives.
    universe = {name.casefold() for group in expected_groups for name in group}
    expected_pairs = _pairs(expected_groups)
    predicted_pairs = _pairs(
        [
            [item.name for item in block.components if item.name.casefold() in universe]
            for block in blocks
        ]
    )
    true_positive = len(expected_pairs & predicted_pairs)
    precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
    recall = true_positive / len(expected_pairs) if expected_pairs else 1.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _pairs(groups: list[list[str]]) -> set[tuple[str, str]]:
    result = set()
    for group in groups:
        normalized = sorted(item.casefold() for item in group)
        for index, left in enumerate(normalized):
            for right in normalized[index + 1 :]:
                result.add((left, right))
    return result


def _component_block(blocks: list[FunctionalBlock], component_id: str) -> FunctionalBlock:
    for block in blocks:
        if any(component.id == component_id for component in block.components):
            return block
    raise ValueError(f"component is not assigned to a block: {component_id}")


def _technology_registry() -> dict[str, _Technology]:
    path = Path(__file__).with_name("technologies.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    result = {}
    for canonical, details in payload.items():
        result[canonical] = _Technology(
            canonical=canonical,
            aliases=tuple(details.get("aliases", [])),
            block_type=AnalysisBlockType(details["block_type"]),
            subtype=str(details["subtype"]),
            capabilities=tuple(details.get("capabilities", [])),
        )
    return result
def _classify_unknown(
    path: str, name: str
) -> tuple[AnalysisBlockType, str, str | None]:
    if path.endswith(".tf") or "terraform" in path:
        return AnalysisBlockType.BASE_INFRASTRUCTURE, "cloud_infrastructure", None
    if any(term in path for term in ("monitor", "logging", "backup", "restore", "report")):
        return AnalysisBlockType.OPERATIONS, "observability", None
    if "/mosip/" in path or any(term in path for term in ("application", "service", "apps/")):
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


def _block_name(subtype: str, domain: str | None = None) -> str:
    return _display_name(domain or subtype)


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


def _source_path(source: AcquiredSource, relative_path: str) -> Path:
    if source.path.is_file():
        if relative_path != source.path.name:
            raise ValueError(f"local file source has no artifact: {relative_path}")
        return source.path.resolve()
    candidate = source.path.joinpath(*PurePosixPath(relative_path).parts).resolve()
    candidate.relative_to(source.path.resolve())
    return candidate


def _source_ref_exists(source: AcquiredSource, relative_path: str) -> bool:
    try:
        return _source_path(source, relative_path).is_file()
    except (OSError, ValueError):
        return False


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
    seen = set()
    result = []
    for relation in relations:
        key = (relation.source, relation.target, relation.relation, relation.strength)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return sorted(result, key=lambda item: (item.source, item.target, item.relation))


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
