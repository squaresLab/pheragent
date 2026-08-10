from __future__ import annotations

import hashlib
import json
import posixpath
import re
from collections.abc import Iterable

from .enums import Confidence, DeterministicFindingKind, FactPredicate, ProvenanceOrigin
from .models import (
    DeploymentFact,
    DeterministicFinding,
    ExtractedFactClaim,
    ExtractedQuestionClaim,
    Provenance,
    UnresolvedQuestion,
)

_CONFIDENCE_RANK = {Confidence.LOW: 0, Confidence.MEDIUM: 1, Confidence.HIGH: 2}
_TERM_ALIASES = {
    "docker compose": "docker_compose",
    "docker-compose": "docker_compose",
    "docker_compose": "docker_compose",
    "github actions": "github_actions",
    "github-actions": "github_actions",
    "github_actions": "github_actions",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "postgres-postgresql": "postgresql",
    "postgresql": "postgresql",
}


def deterministic_facts(
    findings: Iterable[DeterministicFinding],
) -> tuple[DeploymentFact, ...]:
    facts: list[DeploymentFact] = []
    for finding in findings:
        claim = _claim_from_finding(finding)
        if claim is not None:
            facts.append(_fact_from_claim(claim))
    return normalize_facts(facts)


def extracted_facts(claims: Iterable[ExtractedFactClaim]) -> tuple[DeploymentFact, ...]:
    return normalize_facts(_fact_from_claim(claim) for claim in claims)


def normalize_facts(facts: Iterable[DeploymentFact]) -> tuple[DeploymentFact, ...]:
    merged: dict[tuple[str, str, str], DeploymentFact] = {}
    for item in facts:
        subject, subject_alias = _canonicalize(item.subject)
        object_value, object_alias = _canonicalize(item.object)
        key = (subject.casefold(), item.predicate.value, object_value.casefold())
        existing = merged.get(key)
        refs = sorted(set(item.provenance.evidence_refs))
        subject_aliases = _aliases(
            subject,
            *(item.subject_aliases or []),
            *([subject_alias] if subject_alias else []),
        )
        object_aliases = _aliases(
            object_value,
            *(item.object_aliases or []),
            *([object_alias] if object_alias else []),
        )
        if existing is not None:
            refs = sorted(set(existing.provenance.evidence_refs) | set(refs))
            confidence = max(
                (existing.provenance.confidence, item.provenance.confidence),
                key=_CONFIDENCE_RANK.__getitem__,
            )
            subject = existing.subject
            object_value = existing.object
            subject_aliases = _aliases(subject, *(existing.subject_aliases or []), *subject_aliases)
            object_aliases = _aliases(
                object_value, *(existing.object_aliases or []), *object_aliases
            )
        else:
            confidence = item.provenance.confidence
        fact_id = _stable_id(
            "fact", (subject.casefold(), item.predicate.value, object_value.casefold())
        )
        merged[key] = DeploymentFact(
            id=fact_id,
            subject=subject,
            subject_aliases=subject_aliases or None,
            predicate=item.predicate,
            object=object_value,
            object_aliases=object_aliases or None,
            provenance=Provenance(
                origin=ProvenanceOrigin.EXTRACTED,
                confidence=confidence,
                evidence_refs=refs,
            ),
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda fact: (
                fact.subject.casefold(),
                fact.predicate,
                fact.object.casefold(),
                fact.id,
            ),
        )
    )


def normalize_questions(
    claims: Iterable[ExtractedQuestionClaim],
) -> tuple[UnresolvedQuestion, ...]:
    merged: dict[str, UnresolvedQuestion] = {}
    for claim in claims:
        question = _normalize_text(claim.question)
        key = question.casefold()
        existing = merged.get(key)
        evidence_refs = sorted(set(claim.evidence_refs))
        subjects = sorted({_normalize_text(value) for value in claim.related_subjects})
        if existing is not None:
            evidence_refs = sorted(set(existing.evidence_refs) | set(evidence_refs))
            subjects = sorted(set(existing.related_subjects) | set(subjects))
        merged[key] = UnresolvedQuestion(
            id=_stable_id("question", (key,)),
            question=question,
            reason=_normalize_text(claim.reason),
            related_subjects=subjects,
            evidence_refs=evidence_refs,
        )
    return tuple(sorted(merged.values(), key=lambda item: (item.question.casefold(), item.id)))


def _claim_from_finding(finding: DeterministicFinding) -> ExtractedFactClaim | None:
    attributes = finding.attributes
    if finding.kind == DeterministicFindingKind.DEPENDENCY:
        source = _optional_text(attributes.get("source"))
        target = _optional_text(attributes.get("target"))
        if source and target:
            return _claim(source, FactPredicate.REQUIRES, target, finding)
    if finding.kind in {
        DeterministicFindingKind.COMPONENT,
        DeterministicFindingKind.RESOURCE,
    }:
        implementation = (
            _optional_text(attributes.get("image"))
            or _optional_text(attributes.get("chart"))
            or _optional_text(attributes.get("kubernetes_kind"))
            or finding.category.value
        )
        return _claim(finding.name, FactPredicate.IMPLEMENTED_BY, implementation, finding)
    if finding.kind == DeterministicFindingKind.VALIDATION:
        subject = _optional_text(attributes.get("component")) or finding.path
        return _claim(subject, FactPredicate.VALIDATED_BY, finding.name, finding)
    if finding.kind == DeterministicFindingKind.CONFIGURATION:
        return _claim(finding.path, FactPredicate.CONFIGURED_BY, finding.name, finding)
    return None


def _claim(
    subject: str,
    predicate: FactPredicate,
    object_value: str,
    finding: DeterministicFinding,
) -> ExtractedFactClaim:
    return ExtractedFactClaim(
        subject=subject,
        predicate=predicate,
        object=object_value,
        confidence=Confidence.HIGH,
        evidence_refs=finding.evidence_refs,
    )


def _fact_from_claim(claim: ExtractedFactClaim) -> DeploymentFact:
    subject, subject_alias = _canonicalize(claim.subject)
    object_value, object_alias = _canonicalize(claim.object)
    return DeploymentFact(
        id=_stable_id("fact", (subject.casefold(), claim.predicate.value, object_value.casefold())),
        subject=subject,
        subject_aliases=[subject_alias] if subject_alias else None,
        predicate=claim.predicate,
        object=object_value,
        object_aliases=[object_alias] if object_alias else None,
        provenance=Provenance(
            origin=ProvenanceOrigin.EXTRACTED,
            confidence=claim.confidence,
            evidence_refs=sorted(set(claim.evidence_refs)),
        ),
    )


def _normalize_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if not normalized:
        raise ValueError("fact and question text must not be blank")
    return normalized


def _canonicalize(value: str) -> tuple[str, str | None]:
    original = _normalize_text(value)
    lookup = original.casefold()
    canonical = _TERM_ALIASES.get(lookup)
    if (
        canonical is None
        and "://" not in original
        and ("/" in original or "\\" in original or original.startswith("."))
    ):
        canonical = posixpath.normpath(original.replace("\\", "/"))
    canonical = canonical or original
    alias = original if original != canonical else None
    return canonical, alias


def _aliases(canonical: str, *values: str) -> list[str]:
    aliases = {
        _normalize_text(value)
        for value in values
        if value and _normalize_text(value).casefold() != canonical.casefold()
    }
    return sorted(aliases, key=str.casefold)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_id(prefix: str, values: tuple[str, ...]) -> str:
    canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return f"{prefix}-{hashlib.sha256(canonical.encode()).hexdigest()[:20]}"
