from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from pheragent.utils import slugify

from .enums import (
    BlockType,
    ComponentRelationType,
    Confidence,
    DeterministicFindingKind,
    Executor,
    FactPredicate,
    InventoryCategory,
    OperationPhase,
    ProvenanceOrigin,
    TargetKind,
    ValidationLevel,
)
from .fact_extractor import FactExtractionResult
from .graph import build_dependency_graph
from .inspection import DeterministicInspectionResult
from .models import (
    ArtifactMetadata,
    Capability,
    Component,
    ComponentRelation,
    DependencyGraph,
    DeploymentArtifact,
    DeploymentBlock,
    DeploymentScope,
    DeploymentValidation,
    Operation,
    Provenance,
    RecoveryBoundary,
    Requirement,
    ScopeClaim,
    SourceArtifactReference,
    SourceRecord,
    Target,
    UnresolvedQuestion,
)

_BLOCK_ORDER = {block_type: index for index, block_type in enumerate(BlockType)}
_BLOCK_DETAILS = {
    BlockType.HOST_READINESS: (
        "Host Readiness",
        "Establish that the supplied hosts meet deployment prerequisites.",
        "reachable-hosts",
    ),
    BlockType.PLATFORM: (
        "Platform",
        "Establish the deployment platform and its cluster-level services.",
        "container-orchestration",
    ),
    BlockType.SHARED_SERVICES: (
        "Shared Services",
        "Provide reusable infrastructure services consumed by the application.",
        "shared-services",
    ),
    BlockType.APPLICATION: (
        "Application",
        "Deploy the domain application services.",
        "application-services",
    ),
    BlockType.INITIALIZATION: (
        "Initialization",
        "Perform one-time initialization required to make the application usable.",
        "initialized-application",
    ),
    BlockType.OPERATIONS: (
        "Operations",
        "Provide monitoring, logging, management, and recovery capabilities.",
        "operational-observability",
    ),
    BlockType.VERIFICATION: (
        "Verification",
        "Verify readiness and behavior across deployment boundaries.",
        "deployment-verified",
    ),
    BlockType.HUMAN_GATE: (
        "Human Gate",
        "Record external inputs or approvals that automation cannot safely derive.",
        "human-input-satisfied",
    ),
}
_CLASSIFICATION_TERMS = (
    (BlockType.HUMAN_GATE, ("manual", "approval", "certificate input", "human gate")),
    (
        BlockType.VERIFICATION,
        ("test rig", "integration test", "end-to-end", "e2e", "smoke test", "verify"),
    ),
    (
        BlockType.INITIALIZATION,
        ("migration", "initialize", "initialise", "bootstrap", "onboard", "seed", "masterdata"),
    ),
    (
        BlockType.OPERATIONS,
        (
            "monitor",
            "logging",
            "grafana",
            "prometheus",
            "rancher",
            "observability",
            "backup",
            "alert",
            "fluent",
        ),
    ),
    (
        BlockType.HOST_READINESS,
        ("host readiness", "preflight", "prerequisite", "ssh", "prepare host", "inventory"),
    ),
    (
        BlockType.PLATFORM,
        (
            "kubernetes",
            "rke2",
            "k8s",
            "ingress",
            "cert-manager",
            "istio",
            "storageclass",
            "cluster platform",
            "container runtime",
            "cni",
        ),
    ),
    (
        BlockType.SHARED_SERVICES,
        (
            "postgres",
            "database",
            "kafka",
            "activemq",
            "rabbitmq",
            "minio",
            "keycloak",
            "redis",
            "mysql",
            "vault",
            "softhsm",
            "object storage",
            "identity provider",
            "message queue",
        ),
    ),
)


@dataclass(slots=True)
class _ComponentCandidate:
    name: str
    implementation: str
    aliases: set[str] = field(default_factory=set)
    evidence_refs: set[str] = field(default_factory=set)
    confidence: Confidence = Confidence.MEDIUM
    paths: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    artifact: DeploymentArtifact
    graph: DependencyGraph


def synthesize_artifact(
    inspection: DeterministicInspectionResult,
    extraction: FactExtractionResult,
) -> SynthesisResult:
    candidates = _component_candidates(inspection, extraction)
    component_types = {name: _classify_candidate(item) for name, item in candidates.items()}
    block_types = set(component_types.values())
    for finding in inspection.findings:
        if finding.kind in {
            DeterministicFindingKind.COMMAND,
            DeterministicFindingKind.VALIDATION,
        }:
            block_types.add(_classify_text(f"{finding.path} {finding.name}"))
    if any(fact.predicate == FactPredicate.HUMAN_INPUT_REQUIRED for fact in extraction.facts):
        block_types.add(BlockType.HUMAN_GATE)

    components_by_block, component_lookup = _build_components(candidates, component_types)
    component_block_types = {
        component.id: block_type
        for block_type, components in components_by_block.items()
        for component in components
    }
    operations_by_block = _build_operations(inspection, component_lookup, component_block_types)
    validations_by_block = _build_validations(inspection, component_lookup, component_block_types)
    relations_by_block, cross_requirements, fact_capabilities = _build_relations(
        extraction, component_lookup, component_block_types
    )
    blocks: list[DeploymentBlock] = []
    for block_type in sorted(block_types, key=_BLOCK_ORDER.__getitem__):
        components = components_by_block.get(block_type, [])
        operations = operations_by_block.get(block_type, [])
        validations = validations_by_block.get(block_type, [])
        evidence_refs = sorted(
            {
                ref
                for item in (*components, *operations, *validations)
                for ref in item.provenance.evidence_refs
            }
        )
        block_id = block_type.value.replace("_", "-")
        name, purpose, capability = _BLOCK_DETAILS[block_type]
        requirements = _base_requirements(block_type, block_types, evidence_refs)
        requirements.extend(cross_requirements.get(block_type, []))
        requirements = _deduplicate_requirements(requirements)
        provides = [
            Capability(
                capability=capability,
                provenance=_inferred(evidence_refs),
            )
        ]
        for component in components:
            role = component.role
            if role != capability and all(item.capability != role for item in provides):
                provides.append(Capability(capability=role, provenance=component.provenance))
        provides.extend(fact_capabilities.get(block_type, []))
        provides = _deduplicate_capabilities(provides)
        blocks.append(
            DeploymentBlock(
                id=block_id,
                name=name,
                type=block_type,
                target="deployment-environment",
                purpose=purpose,
                grouping_rationale=(
                    "Components share a functional purpose, target environment, and "
                    "deployment/recovery lifecycle."
                ),
                grouping_provenance=_inferred(evidence_refs),
                components=components,
                component_relations=relations_by_block.get(block_type, []),
                requires=requirements,
                provides=provides,
                operations=operations,
                validations=validations,
                recovery_boundary=RecoveryBoundary(
                    local_components=[component.id for component in components],
                    upstream_capabilities=[item.capability for item in requirements],
                ),
                provenance=_inferred(evidence_refs),
            )
        )

    blocks_tuple = tuple(blocks)
    graph = build_dependency_graph(blocks_tuple)
    questions = _artifact_questions(extraction, blocks_tuple, graph)
    artifact = DeploymentArtifact(
        artifact_version="0.1",
        metadata=ArtifactMetadata(
            artifact_id=f"{slugify(inspection.acquisition.manifest.system)}-deployment",
            system_name=inspection.acquisition.manifest.system,
            system_version="unknown",
            profile="unknown",
            generated_at=datetime.now(UTC),
            generator_version=_generator_version(),
        ),
        sources=[_source_record(source) for source in inspection.acquisition.sources],
        scope=DeploymentScope(
            includes=[
                ScopeClaim(
                    description=_BLOCK_DETAILS[block.type][1],
                    provenance=block.provenance,
                )
                for block in blocks_tuple
            ]
        ),
        targets=(
            [
                Target(
                    id="deployment-environment",
                    kind=TargetKind.LOGICAL_ENVIRONMENT,
                    description="Logical environment described by the inspected sources.",
                    provenance=_inferred(
                        sorted(
                            {
                                ref
                                for block in blocks_tuple
                                for ref in block.provenance.evidence_refs
                            }
                        )
                    ),
                )
            ]
            if blocks_tuple
            else []
        ),
        blocks=list(blocks_tuple),
        unresolved_questions=questions,
    )
    return SynthesisResult(artifact=artifact, graph=graph)


def _component_candidates(
    inspection: DeterministicInspectionResult,
    extraction: FactExtractionResult,
) -> dict[str, _ComponentCandidate]:
    paths_by_name: dict[str, set[str]] = defaultdict(set)
    for finding in inspection.findings:
        paths_by_name[finding.name.casefold()].add(finding.path)
    candidates: dict[str, _ComponentCandidate] = {}
    for fact in extraction.facts:
        if fact.predicate == FactPredicate.IMPLEMENTED_BY:
            name = fact.subject
            implementation = fact.object
            aliases = fact.subject_aliases or []
        elif fact.predicate == FactPredicate.CONTAINS:
            name = fact.object
            implementation = "unknown"
            aliases = fact.object_aliases or []
        else:
            continue
        key = name.casefold()
        candidate = candidates.get(key)
        if candidate is None:
            candidate = _ComponentCandidate(name=name, implementation=implementation)
            candidates[key] = candidate
        elif candidate.implementation == "unknown" and implementation != "unknown":
            candidate.implementation = implementation
        candidate.aliases.update(aliases)
        candidate.evidence_refs.update(fact.provenance.evidence_refs)
        candidate.paths.update(paths_by_name.get(key, set()))
        if fact.provenance.confidence == Confidence.HIGH:
            candidate.confidence = Confidence.HIGH
    return candidates


def _build_components(
    candidates: dict[str, _ComponentCandidate],
    component_types: dict[str, BlockType],
) -> tuple[dict[BlockType, list[Component]], dict[str, Component]]:
    by_block: dict[BlockType, list[Component]] = defaultdict(list)
    lookup: dict[str, Component] = {}
    used_ids: set[str] = set()
    for key, candidate in sorted(candidates.items(), key=lambda item: item[1].name.casefold()):
        component_id = slugify(candidate.name, fallback="component")
        if component_id in used_ids:
            component_id = f"{component_id}-{_digest((key,))[:8]}"
        used_ids.add(component_id)
        role = _component_role(candidate)
        component = Component(
            id=component_id,
            name=candidate.name,
            role=role,
            implementation=candidate.implementation,
            aliases=sorted(candidate.aliases, key=str.casefold),
            provenance=Provenance(
                origin=ProvenanceOrigin.EXTRACTED,
                confidence=candidate.confidence,
                evidence_refs=sorted(candidate.evidence_refs),
            ),
        )
        by_block[component_types[key]].append(component)
        lookup[key] = component
        for alias in candidate.aliases:
            lookup.setdefault(alias.casefold(), component)
    return by_block, lookup


def _build_operations(
    inspection: DeterministicInspectionResult,
    component_lookup: dict[str, Component],
    component_block_types: dict[str, BlockType],
) -> dict[BlockType, list[Operation]]:
    result: dict[BlockType, list[Operation]] = defaultdict(list)
    seen: set[str] = set()
    for finding in inspection.findings:
        if finding.kind != DeterministicFindingKind.COMMAND:
            continue
        command = str(finding.attributes.get("command") or "").strip()
        if not command:
            continue
        named_component = str(finding.attributes.get("component") or "").casefold()
        component = component_lookup.get(named_component)
        block_type = (
            component_block_types.get(component.id)
            if component is not None
            else _classify_text(f"{finding.path} {finding.name} {command}")
        ) or BlockType.APPLICATION
        operation_id = f"operation-{_digest((finding.id, command))[:20]}"
        if operation_id in seen:
            continue
        seen.add(operation_id)
        result[block_type].append(
            Operation(
                id=operation_id,
                phase=_operation_phase(command),
                executor=_executor(finding.category, command),
                component=component.id if component is not None else None,
                source_artifact=SourceArtifactReference(
                    source_id=finding.source_id, path=finding.path
                ),
                command=command,
                provenance=_extracted(finding.evidence_refs),
            )
        )
    for operations in result.values():
        operations.sort(key=lambda item: item.id)
    _attach_component_operations(result, component_lookup)
    return result


def _attach_component_operations(
    operations_by_block: dict[BlockType, list[Operation]],
    component_lookup: dict[str, Component],
) -> None:
    operations_by_component: dict[str, list[str]] = defaultdict(list)
    for operations in operations_by_block.values():
        for operation in operations:
            if operation.component:
                operations_by_component[operation.component].append(operation.id)
    visited: set[str] = set()
    for component in component_lookup.values():
        if component.id in visited:
            continue
        visited.add(component.id)
        component.operations.extend(sorted(operations_by_component.get(component.id, [])))


def _build_validations(
    inspection: DeterministicInspectionResult,
    component_lookup: dict[str, Component],
    component_block_types: dict[str, BlockType],
) -> dict[BlockType, list[DeploymentValidation]]:
    result: dict[BlockType, list[DeploymentValidation]] = defaultdict(list)
    for finding in inspection.findings:
        if finding.kind != DeterministicFindingKind.VALIDATION:
            continue
        named_component = str(
            finding.attributes.get("component") or finding.attributes.get("resource") or ""
        ).casefold()
        component = component_lookup.get(named_component)
        block_type = (
            component_block_types.get(component.id)
            if component is not None
            else _classify_text(f"{finding.path} {finding.name}")
        ) or BlockType.VERIFICATION
        command = str(finding.attributes.get("command") or "")
        result[block_type].append(
            DeploymentValidation(
                id=f"validation-{_digest((finding.id,))[:20]}",
                level=_validation_level(finding),
                description=finding.name,
                executor=_executor(finding.category, command),
                source_artifact=SourceArtifactReference(
                    source_id=finding.source_id, path=finding.path
                ),
                provenance=_extracted(finding.evidence_refs),
            )
        )
    for validations in result.values():
        validations.sort(key=lambda item: item.id)
    return result


def _build_relations(
    extraction: FactExtractionResult,
    component_lookup: dict[str, Component],
    component_block_types: dict[str, BlockType],
) -> tuple[
    dict[BlockType, list[ComponentRelation]],
    dict[BlockType, list[Requirement]],
    dict[BlockType, list[Capability]],
]:
    relations: dict[BlockType, list[ComponentRelation]] = defaultdict(list)
    cross_requirements: dict[BlockType, list[Requirement]] = defaultdict(list)
    fact_capabilities: dict[BlockType, list[Capability]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for fact in extraction.facts:
        source_key = fact.subject.casefold()
        source = component_lookup.get(source_key)
        if fact.predicate == FactPredicate.PROVIDES and source is not None:
            fact_capabilities[component_block_types[source.id]].append(
                Capability(capability=fact.object, provenance=fact.provenance)
            )
            continue
        if fact.predicate != FactPredicate.REQUIRES:
            continue
        target_key = fact.object.casefold()
        target = component_lookup.get(target_key)
        if source is None:
            continue
        source_type = component_block_types[source.id]
        if target is None:
            cross_requirements[source_type].append(
                Requirement(capability=fact.object, provenance=fact.provenance)
            )
            continue
        if (source.id, target.id) in seen:
            continue
        seen.add((source.id, target.id))
        target_type = component_block_types[target.id]
        if source_type == target_type:
            relations[source_type].append(
                ComponentRelation(
                    source=source.id,
                    relation=ComponentRelationType.REQUIRES,
                    target=target.id,
                    provenance=fact.provenance,
                )
            )
        else:
            cross_requirements[source_type].append(
                Requirement(
                    capability=target.role,
                    provider_block=target_type.value.replace("_", "-"),
                    provenance=fact.provenance,
                )
            )
    for values in relations.values():
        values.sort(key=lambda item: (item.source, item.target))
    return relations, cross_requirements, fact_capabilities


def _base_requirements(
    block_type: BlockType,
    present: set[BlockType],
    evidence_refs: list[str],
) -> list[Requirement]:
    candidates: list[tuple[BlockType, str]] = []
    if block_type == BlockType.PLATFORM and BlockType.HOST_READINESS in present:
        candidates.append((BlockType.HOST_READINESS, "reachable-hosts"))
    if (
        block_type in {BlockType.SHARED_SERVICES, BlockType.APPLICATION, BlockType.OPERATIONS}
        and BlockType.PLATFORM in present
    ):
        candidates.append((BlockType.PLATFORM, "container-orchestration"))
    if block_type == BlockType.APPLICATION and BlockType.SHARED_SERVICES in present:
        candidates.append((BlockType.SHARED_SERVICES, "shared-services"))
    if block_type == BlockType.INITIALIZATION and BlockType.APPLICATION in present:
        candidates.append((BlockType.APPLICATION, "application-services"))
    if block_type == BlockType.VERIFICATION:
        provider = (
            BlockType.INITIALIZATION
            if BlockType.INITIALIZATION in present
            else BlockType.APPLICATION
        )
        if provider in present:
            capability = (
                "initialized-application"
                if provider == BlockType.INITIALIZATION
                else "application-services"
            )
            candidates.append((provider, capability))
    return [
        Requirement(
            capability=capability,
            provider_block=provider.value.replace("_", "-"),
            provenance=_inferred(evidence_refs),
        )
        for provider, capability in candidates
    ]


def _deduplicate_requirements(requirements: list[Requirement]) -> list[Requirement]:
    unique: dict[tuple[str, str | None], Requirement] = {}
    for item in requirements:
        unique[(item.capability, item.provider_block)] = item
    return sorted(unique.values(), key=lambda item: (item.capability, item.provider_block or ""))


def _deduplicate_capabilities(capabilities: list[Capability]) -> list[Capability]:
    unique: dict[str, Capability] = {}
    for item in capabilities:
        existing = unique.get(item.capability)
        if existing is None or item.provenance.confidence == Confidence.HIGH:
            unique[item.capability] = item
    return sorted(unique.values(), key=lambda item: item.capability)


def _artifact_questions(
    extraction: FactExtractionResult,
    blocks: tuple[DeploymentBlock, ...],
    graph: DependencyGraph,
) -> list[UnresolvedQuestion]:
    questions = {question.id: question for question in extraction.unresolved_questions}
    version_question = UnresolvedQuestion(
        id="question-system-version",
        question="Which system version does this deployment describe?",
        reason="No system version is asserted by the normalized facts.",
    )
    questions[version_question.id] = version_question
    for block in blocks:
        if block.type != BlockType.VERIFICATION and not block.validations:
            question = UnresolvedQuestion(
                id=f"question-{block.id}-validation",
                question=f"What source-backed validation confirms {block.name} is ready?",
                reason="No explicit validation was classified for this block.",
                related_block_ids=[block.id],
                evidence_refs=block.provenance.evidence_refs,
            )
            questions[question.id] = question
    for unmatched in graph.unmatched_capabilities:
        block_id, capability = unmatched.split(":", 1)
        question = UnresolvedQuestion(
            id=f"question-unmatched-{_digest((unmatched,))[:16]}",
            question=f"Which block or external system provides {capability}?",
            reason=f"The required capability for {block_id} has no unique provider.",
            related_block_ids=[block_id],
        )
        questions[question.id] = question
    return sorted(questions.values(), key=lambda item: item.id)


def _classify_candidate(candidate: _ComponentCandidate) -> BlockType:
    text = " ".join(
        (candidate.name, candidate.implementation, *sorted(candidate.aliases), *candidate.paths)
    )
    return _classify_text(text)


def _classify_text(text: str) -> BlockType:
    normalized = text.casefold().replace("_", " ").replace("-", " ")
    for block_type, terms in _CLASSIFICATION_TERMS:
        if any(term.replace("_", " ").replace("-", " ") in normalized for term in terms):
            return block_type
    return BlockType.APPLICATION


def _component_role(candidate: _ComponentCandidate) -> str:
    text = f"{candidate.name} {candidate.implementation}".casefold()
    roles = (
        ("relational-database", ("postgres", "mysql", "database")),
        ("event-streaming", ("kafka",)),
        ("message-queue", ("activemq", "rabbitmq")),
        ("object-storage", ("minio", "object storage")),
        ("identity-provider", ("keycloak", "identity")),
        ("distributed-cache", ("redis", "cache")),
        ("container-orchestration", ("kubernetes", "rke2", "k8s")),
        ("observability", ("grafana", "prometheus", "monitor", "logging")),
    )
    for role, terms in roles:
        if any(term in text for term in terms):
            return role
    return slugify(candidate.name, fallback="application-component")


def _operation_phase(command: str) -> OperationPhase:
    normalized = command.casefold()
    for phase, terms in (
        (OperationPhase.VALIDATE, (" wait ", " get ", "health", "curl ", "test ")),
        (OperationPhase.INITIALIZE, ("migrate", "init", "onboard", "seed")),
        (OperationPhase.UPGRADE, ("upgrade",)),
        (OperationPhase.REMOVE, ("delete", "remove", "uninstall")),
        (OperationPhase.CONFIGURE, ("config", "apply")),
        (OperationPhase.INSTALL, ("install", "create")),
        (OperationPhase.START, ("start", " up")),
    ):
        if any(term in f" {normalized} " for term in terms):
            return phase
    return OperationPhase.PREPARE


def _executor(category: InventoryCategory, command: str) -> Executor:
    normalized = command.strip().casefold()
    for prefix, executor in (
        ("kubectl", Executor.KUBECTL),
        ("helm ", Executor.HELM),
        ("helmsman", Executor.HELMSMAN),
        ("terraform", Executor.TERRAFORM),
        ("ansible-playbook", Executor.ANSIBLE),
        ("kustomize", Executor.KUSTOMIZE),
        ("docker compose", Executor.DOCKER_COMPOSE),
        ("docker-compose", Executor.DOCKER_COMPOSE),
    ):
        if normalized.startswith(prefix):
            return executor
    category_executors = {
        InventoryCategory.SHELL: Executor.SHELL,
        InventoryCategory.ANSIBLE: Executor.ANSIBLE,
        InventoryCategory.TERRAFORM: Executor.TERRAFORM,
        InventoryCategory.HELM: Executor.HELM,
        InventoryCategory.HELMSMAN: Executor.HELMSMAN,
        InventoryCategory.KUSTOMIZE: Executor.KUSTOMIZE,
        InventoryCategory.COMPOSE: Executor.DOCKER_COMPOSE,
        InventoryCategory.CI_WORKFLOW: Executor.GITHUB_ACTIONS,
    }
    return category_executors.get(category, Executor.UNKNOWN)


def _validation_level(finding: Any) -> ValidationLevel:
    normalized = f"{finding.name} {finding.attributes}".casefold()
    if "readiness" in normalized or "ready" in normalized:
        return ValidationLevel.READINESS
    if "integration" in normalized:
        return ValidationLevel.INTEGRATION
    if "end-to-end" in normalized or "e2e" in normalized:
        return ValidationLevel.END_TO_END
    if "health" in normalized:
        return ValidationLevel.FUNCTIONAL
    return ValidationLevel.STRUCTURAL


def _source_record(source: Any) -> SourceRecord:
    manifest = source.manifest
    return SourceRecord(
        id=manifest.id,
        kind=manifest.kind,
        repository=manifest.location,
        revision=manifest.resolved_revision,
        root_path=manifest.root_path,
        content_hash=manifest.content_hash,
    )


def _extracted(evidence_refs: list[str]) -> Provenance:
    return Provenance(
        origin=ProvenanceOrigin.EXTRACTED,
        confidence=Confidence.HIGH,
        evidence_refs=sorted(set(evidence_refs)),
    )


def _inferred(evidence_refs: list[str]) -> Provenance:
    return Provenance(
        origin=ProvenanceOrigin.INFERRED,
        confidence=Confidence.MEDIUM,
        evidence_refs=sorted(set(evidence_refs)),
    )


def _digest(values: tuple[str, ...]) -> str:
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _generator_version() -> str:
    try:
        return version("pheragent")
    except PackageNotFoundError:
        return "0.1.0"
