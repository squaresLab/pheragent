from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from pheragent.deployment.chunking import build_extraction_chunks, rank_extraction_chunks
from pheragent.deployment.enums import (
    Confidence,
    DeterministicFindingKind,
    FactPredicate,
    InventoryCategory,
)
from pheragent.deployment.fact_extractor import FactExtractorConfig, OpenAIFactExtractor
from pheragent.deployment.facts import deterministic_facts, extracted_facts, normalize_facts
from pheragent.deployment.models import (
    DeterministicFinding,
    EvidenceRecord,
    ExtractedFactClaim,
    InventoryEntry,
    RepositoryInventory,
)


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(
        id="evidence-123",
        source_id="fixture",
        path="compose.yaml",
        start_line=1,
        end_line=3,
        excerpt_hash="a" * 64,
        excerpt="api:\n  depends_on:\n    - database",
    )


def _finding() -> DeterministicFinding:
    return DeterministicFinding(
        id="finding-123",
        source_id="fixture",
        path="compose.yaml",
        category=InventoryCategory.COMPOSE,
        kind=DeterministicFindingKind.DEPENDENCY,
        name="api requires database",
        attributes={"source": "api", "target": "database"},
        evidence_refs=["evidence-123"],
    )


def test_chunks_and_deterministic_facts_are_stable_and_evidence_backed() -> None:
    chunks = build_extraction_chunks((_finding(),), (_evidence(),))
    repeated = build_extraction_chunks((_finding(),), (_evidence(),))

    assert chunks == repeated
    assert chunks[0].finding_ids == ["finding-123"]
    assert [record.id for record in chunks[0].evidence] == ["evidence-123"]

    facts = deterministic_facts((_finding(),))
    assert len(facts) == 1
    assert facts[0].subject == "api"
    assert facts[0].predicate == FactPredicate.REQUIRES
    assert facts[0].object == "database"
    assert facts[0].provenance.confidence == Confidence.HIGH
    assert facts[0].provenance.evidence_refs == ["evidence-123"]


def test_fact_normalization_preserves_aliases_and_merges_evidence() -> None:
    claims = [
        ExtractedFactClaim(
            subject="api",
            predicate=FactPredicate.REQUIRES,
            object="postgres",
            confidence=Confidence.MEDIUM,
            evidence_refs=["evidence-123"],
        ),
        ExtractedFactClaim(
            subject="API",
            predicate=FactPredicate.REQUIRES,
            object="postgres-postgresql",
            confidence=Confidence.HIGH,
            evidence_refs=["evidence-456"],
        ),
    ]

    facts = normalize_facts(extracted_facts(claims))

    assert len(facts) == 1
    assert facts[0].subject == "api"
    assert facts[0].object == "postgresql"
    assert facts[0].object_aliases == ["postgres", "postgres-postgresql"]
    assert facts[0].provenance.confidence == Confidence.HIGH
    assert facts[0].provenance.evidence_refs == ["evidence-123", "evidence-456"]


def test_openai_extractor_uses_strict_schema_and_rejects_unknown_evidence(monkeypatch) -> None:
    chunks = build_extraction_chunks((_finding(),), (_evidence(),))
    content = json.dumps(
        {
            "facts": [
                {
                    "subject": "api",
                    "predicate": "requires",
                    "object": "database",
                    "confidence": "high",
                    "evidence_refs": ["evidence-123"],
                },
                {
                    "subject": "api",
                    "predicate": "requires",
                    "object": "invented-service",
                    "confidence": "low",
                    "evidence_refs": ["evidence-unknown"],
                },
            ],
            "unresolved_questions": [
                {
                    "question": "Who provisions the database?",
                    "reason": "The evidence only declares the dependency.",
                    "related_subjects": ["database"],
                    "evidence_refs": ["evidence-123"],
                }
            ],
        }
    )
    captured: dict[str, object] = {}

    class Responses:
        def create(self, **payload):
            captured.update(payload)
            return [
                {"type": "response.output_text.delta", "delta": content},
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 10, "output_tokens": 8, "total_tokens": 18}
                    },
                },
            ]

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    extractor = OpenAIFactExtractor(
        FactExtractorConfig(extractor="llm", api_key_env="TEST_OPENAI_KEY"),
        client_factory=lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )

    facts, questions, warnings = extractor.extract(chunks)

    assert [(fact.subject, fact.predicate, fact.object) for fact in facts] == [
        ("api", FactPredicate.REQUIRES, "database")
    ]
    assert [question.question for question in questions] == ["Who provisions the database?"]
    assert warnings == [
        "discarded unsupported fact in "
        f"{chunks[0].id}: unknown evidence references: evidence-unknown"
    ]
    assert captured["text"]["format"]["type"] == "json_schema"  # type: ignore[index]
    assert captured["text"]["format"]["strict"] is True  # type: ignore[index]
    prompt = json.loads(captured["input"][0]["content"][0]["text"])  # type: ignore[index]
    assert prompt["cost_control"] == {
        "hard_request_limit": 25,
        "selected_chunk_index": 1,
        "selected_chunk_count": 1,
        "remaining_selected_chunks_after_this": 0,
        "instruction": (
            "Complete extraction for this chunk in one response; no follow-up request "
            "is available for this chunk."
        ),
    }
    assert extractor.usage()["requests"] == 1


def test_chunk_ranking_prioritizes_inventory_relevance() -> None:
    low_evidence = _evidence().model_copy(update={"id": "evidence-low", "path": "a.yaml"})
    high_evidence = _evidence().model_copy(update={"id": "evidence-high", "path": "z.yaml"})
    low_finding = _finding().model_copy(
        update={"id": "finding-low", "path": "a.yaml", "evidence_refs": ["evidence-low"]}
    )
    high_finding = _finding().model_copy(
        update={"id": "finding-high", "path": "z.yaml", "evidence_refs": ["evidence-high"]}
    )
    chunks = build_extraction_chunks((low_finding, high_finding), (low_evidence, high_evidence))
    inventory = RepositoryInventory(
        entries=[
            InventoryEntry(
                source_id="fixture",
                path="a.yaml",
                category=InventoryCategory.COMPOSE,
                size_bytes=10,
                selected=True,
                relevance_score=1,
            ),
            InventoryEntry(
                source_id="fixture",
                path="z.yaml",
                category=InventoryCategory.COMPOSE,
                size_bytes=10,
                selected=True,
                relevance_score=50,
            ),
        ]
    )

    ranked = rank_extraction_chunks(chunks, inventory)

    assert [chunk.path for chunk in ranked] == ["z.yaml", "a.yaml"]


def test_openai_extractor_hard_cap_includes_retries(monkeypatch) -> None:
    attempts = 0

    class Responses:
        def create(self, **_payload):
            nonlocal attempts
            attempts += 1
            raise OSError("temporary connection failure")

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    extractor = OpenAIFactExtractor(
        FactExtractorConfig(
            extractor="llm",
            api_key_env="TEST_OPENAI_KEY",
            max_requests=2,
            max_retries=5,
            retry_delay_s=0,
        ),
        client_factory=lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )

    facts, questions, warnings = extractor.extract(
        (build_extraction_chunks((_finding(),), (_evidence(),))[0],)
    )

    assert facts == []
    assert questions == []
    assert attempts == 2
    assert extractor.requests_started == 2
    assert warnings == [
        "LLM request budget exhausted after 2 request(s); skipped 1 selected chunk(s)"
    ]


def test_openai_extractor_does_not_retry_credit_errors(monkeypatch) -> None:
    attempts = 0

    class CreditError(Exception):
        status_code = 429
        code = "credit_balance_exhausted"

    class Responses:
        def create(self, **_payload):
            nonlocal attempts
            attempts += 1
            raise CreditError("You have no credits remaining")

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    extractor = OpenAIFactExtractor(
        FactExtractorConfig(
            extractor="llm",
            api_key_env="TEST_OPENAI_KEY",
            max_requests=10,
            max_retries=5,
            retry_delay_s=0,
        ),
        client_factory=lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )

    with pytest.raises(RuntimeError, match="no credits remaining"):
        extractor.extract(build_extraction_chunks((_finding(),), (_evidence(),)))

    assert attempts == 1


def test_cli_offline_extraction_writes_fact_sidecars(tmp_path: Path, monkeypatch, capsys) -> None:
    from pheragent.cli import main

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "compose.yaml").write_text(
        "services:\n  database:\n    image: postgres:16\n  api:\n    depends_on: [database]\n",
        encoding="utf-8",
    )
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "sources": [
                    {
                        "id": "fixture",
                        "kind": "local_directory",
                        "location": "source",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "output"
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "deployment",
            "inspect",
            "--sources",
            str(sources),
            "--output",
            str(output),
            "--strict",
            "--extractor",
            "deterministic",
        ]
    )

    assert exit_code == 0
    assert "facts:" in capsys.readouterr().out
    fact_lines = (output / "facts.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(json.loads(line)["predicate"] == "requires" for line in fact_lines)
    questions = yaml.safe_load((output / "unresolved-questions.yaml").read_text())
    assert questions["questions_version"] == "0.1"
    assert any(item["id"] == "question-system-version" for item in questions["questions"])
    report = json.loads((output / "fact-extraction.json").read_text())
    assert report["used_extractor"] == "deterministic"
    assert report["max_llm_requests"] == 25
    assert report["llm_requests_made"] == 0
    assert report["selected_chunk_count"] == 0
