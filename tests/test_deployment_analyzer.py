from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from pheragent.cli import main
from pheragent.deployment.analysis_llm import (
    AnalysisLLMConfig,
    classify_components_with_llm,
)
from pheragent.deployment.analysis_models import AnalysisBlockType, DeploymentContext
from pheragent.deployment.analyzer import AnalysisConfig, run_repository_analysis
from pheragent.deployment.output import create_timestamped_run_directory


def _minimal_context(path: Path) -> Path:
    path.write_text(
        yaml.safe_dump(
            {
                "system": "grammar-fixture",
                "deployment": {},
                "provided_blocks": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    repository = root / "repository"
    (repository / "deployment/external/all").mkdir(parents=True)
    (repository / "deployment/mosip/all").mkdir(parents=True)
    for component in ("postgres", "iam", "activemq", "kafka"):
        component_dir = repository / "deployment/external" / component
        component_dir.mkdir(parents=True)
        (component_dir / "install.sh").write_text("#!/bin/bash\nhelm upgrade --install x x\n")
    for component in ("kernel", "idrepo"):
        component_dir = repository / "deployment/mosip" / component
        component_dir.mkdir(parents=True)
        (component_dir / "install.sh").write_text("#!/bin/bash\nhelm upgrade --install x x\n")
    (repository / "deployment/README.md").write_text(
        "# Deployment\n\n"
        "1. [External services](external/all/install-all.sh)\n"
        "2. [Application](mosip/all/install-all.sh)\n",
        encoding="utf-8",
    )
    (repository / "deployment/external/all/install-all.sh").write_text(
        """#!/bin/bash
ROOT_DIR=`pwd`/../
cd $ROOT_DIR/postgres
./install.sh
cd $ROOT_DIR/iam
./install.sh
cd $ROOT_DIR/activemq
./install.sh
cd $ROOT_DIR/kafka
./install.sh
""",
        encoding="utf-8",
    )
    (repository / "deployment/mosip/all/install-all.sh").write_text(
        """#!/bin/bash
ROOT_DIR=`pwd`/../
declare -a module=("kernel" "idrepo")
for i in "${module[@]}"; do
  cd $ROOT_DIR/"$i"
  ./install.sh
done
""",
        encoding="utf-8",
    )
    context = root / "context.yaml"
    context.write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "deployment": {"version": "1.0.0", "profile": "on-premises"},
                "provided_blocks": [
                    {
                        "id": "B0",
                        "type": "base_infrastructure",
                        "subtype": "on_prem_infrastructure",
                        "implementation": "physical-servers",
                        "provides": ["compute-hosts"],
                    },
                    {
                        "id": "B1",
                        "type": "runtime_environment",
                        "subtype": "container_platform",
                        "implementation": "kubernetes",
                        "after": ["B0"],
                        "provides": ["kubernetes-api"],
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    gold = root / "gold.yaml"
    gold.write_text(
        yaml.safe_dump(
            {
                "expected_components": [
                    "PostgreSQL",
                    "Keycloak",
                    "ActiveMQ",
                    "Kafka",
                    "Kernel",
                    "Idrepo",
                ],
                "expected_classifications": {
                    "PostgreSQL": {
                        "block_type": "shared_services",
                        "subtype": "data_services",
                    },
                    "Kafka": {
                        "block_type": "shared_services",
                        "subtype": "messaging_integration",
                    },
                },
                "expected_major_edges": [
                    {"source": "base_infrastructure", "target": "runtime_environment"},
                    {"source": "runtime_environment", "target": "shared_services"},
                    {"source": "shared_services", "target": "application"},
                ],
                "expected_groups": [["ActiveMQ", "Kafka"], ["Kernel", "Idrepo"]],
                "forbidden_components": ["install.sh", "Command", "Step"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return repository, context, gold


def test_analyzer_discovers_grounded_components_and_order(tmp_path: Path) -> None:
    repository, context, gold = _write_fixture(tmp_path)

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context,
            cache_dir=tmp_path / "cache",
            gold_path=gold,
            synthesizer="deterministic",
        )
    )

    components = {
        component.name: component
        for component in result.signals.candidate_components
    }
    assert set(components) == {
        "ActiveMQ",
        "Idrepo",
        "Kafka",
        "Kernel",
        "Keycloak",
        "PostgreSQL",
    }
    assert components["PostgreSQL"].deployment is not None
    assert components["PostgreSQL"].deployment.entrypoint.endswith("postgres/install.sh")
    assert [item.id for item in result.signals.candidate_components] == [
        "C001_postgresql",
        "C002_keycloak",
        "C003_activemq",
        "C004_kafka",
        "C005_kernel",
        "C006_idrepo",
    ]
    assert all(component.name != "install.sh" for component in components.values())
    assert any(
        relation.source == components["ActiveMQ"].id
        and relation.target == components["Kafka"].id
        and relation.relation == "ordered_before"
        for relation in result.signals.relations
    )
    assert result.document.evaluation.component_recall == 1.0
    assert result.document.evaluation.grouping_f1 == 1.0
    assert result.document.evaluation.forbidden_component_count == 0
    assert result.document.evaluation.deployability_coverage == 1.0
    assert any(
        block.type == AnalysisBlockType.SHARED_SERVICES for block in result.document.blocks
    )


def test_timestamped_run_directories_do_not_overwrite(tmp_path: Path) -> None:
    now = datetime(2026, 8, 7, 10, 11, 12, tzinfo=UTC)

    first = create_timestamped_run_directory(tmp_path, name="Apache Airflow", now=now)
    second = create_timestamped_run_directory(tmp_path, name="Apache Airflow", now=now)

    assert first.name == "20260807T101112Z-apache-airflow"
    assert second.name == "20260807T101112Z-apache-airflow-01"
    assert first.is_dir()
    assert second.is_dir()


def test_component_classification_uses_one_request_then_shared_cache(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, context, _gold = _write_fixture(tmp_path)
    analysis = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context,
            cache_dir=tmp_path / "source-cache",
            synthesizer="deterministic",
        )
    )
    component_ids = [item.id for item in analysis.signals.candidate_components]
    response = json.dumps(
        {
            "assignments": {
                component.id: {
                    "block_type": component.classification.block_type,
                    "subtype": component.classification.subtype,
                    "domain": component.classification.domain,
                }
                for component in analysis.signals.candidate_components
            },
            "unresolved": [],
        }
    )
    requests = 0
    request_payloads = []

    class Responses:
        def create(self, **payload):
            nonlocal requests
            requests += 1
            request_payloads.append(payload)
            return [
                {"type": "response.output_text.delta", "delta": response},
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 20,
                            "total_tokens": 120,
                        }
                    },
                },
            ]

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    config = AnalysisLLMConfig(
        mode="llm",
        api_key_env="TEST_OPENAI_KEY",
        cache_dir=tmp_path / "llm-cache",
    )

    first = classify_components_with_llm(analysis.signals, config=config)
    second = classify_components_with_llm(analysis.signals, config=config)

    assert requests == 1
    assert first.used == "llm"
    assert first.usage["input_tokens"] == 100
    assert second.used == "llm-cache"
    assert second.classification == first.classification
    assignments_schema = request_payloads[0]["text"]["format"]["schema"][
        "properties"
    ]["assignments"]
    assert assignments_schema["required"] == component_ids
    assert list(assignments_schema["properties"]) == component_ids
    assert assignments_schema["additionalProperties"] is False
    locked_component = next(
        component
        for component in analysis.signals.candidate_components
        if component.classification.confidence == 1.0
    )
    locked_schema = assignments_schema["properties"][locked_component.id]
    assert locked_schema["properties"]["block_type"]["enum"] == [
        locked_component.classification.block_type
    ]
    assert locked_schema["properties"]["subtype"]["enum"] == [
        locked_component.classification.subtype
    ]
    assert "grouping IDs" in request_payloads[0]["instructions"]


def test_incomplete_llm_classification_falls_back_per_component(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, context, _gold = _write_fixture(tmp_path)
    analysis = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context,
            cache_dir=tmp_path / "source-cache",
            synthesizer="deterministic",
        )
    )
    ambiguous = next(
        component
        for component in analysis.signals.candidate_components
        if component.classification.confidence < 1.0
    )
    response = json.dumps(
        {
            "assignments": {
                ambiguous.id: {
                    "block_type": "application",
                    "subtype": "identity_service",
                    "domain": "identity",
                }
            },
            "unresolved": [],
        }
    )
    requests = 0

    class Responses:
        def create(self, **_payload):
            nonlocal requests
            requests += 1
            return [
                {"type": "response.output_text.delta", "delta": response},
                {"type": "response.completed", "response": {"usage": {}}},
            ]

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    config = AnalysisLLMConfig(
        mode="auto",
        api_key_env="TEST_OPENAI_KEY",
        cache_dir=tmp_path / "llm-cache",
    )

    first = classify_components_with_llm(analysis.signals, config=config)
    second = classify_components_with_llm(analysis.signals, config=config)

    assert requests == 1
    assert first.used == "llm-with-fallback"
    assert second.used == "llm-cache-with-fallback"
    assert first.classification is not None
    assert set(first.classification.assignments) == {
        component.id for component in analysis.signals.candidate_components
    }
    assert first.classification.assignments[ambiguous.id].domain == "identity"
    locked = next(
        component
        for component in analysis.signals.candidate_components
        if component.classification.confidence == 1.0
    )
    assert (
        first.classification.assignments[locked.id].block_type
        == locked.classification.block_type
    )
    assert first.warning is not None
    assert second.warning == first.warning


def test_failed_llm_response_is_recorded_and_not_repeated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, context, _gold = _write_fixture(tmp_path)
    analysis = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context,
            cache_dir=tmp_path / "source-cache",
            synthesizer="deterministic",
        )
    )
    response = "not-json"
    requests = 0

    class Responses:
        def create(self, **_payload):
            nonlocal requests
            requests += 1
            return [
                {"type": "response.output_text.delta", "delta": response},
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 80,
                            "output_tokens": 10,
                            "total_tokens": 90,
                        }
                    },
                },
            ]

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    config = AnalysisLLMConfig(
        mode="auto",
        api_key_env="TEST_OPENAI_KEY",
        cache_dir=tmp_path / "llm-cache",
    )

    first = classify_components_with_llm(analysis.signals, config=config)
    second = classify_components_with_llm(analysis.signals, config=config)

    assert requests == 1
    assert first.used == "llm-failed-deterministic"
    assert first.usage["requests"] == 1
    assert first.usage["input_tokens"] == 80
    assert first.failure_history_path is not None
    assert second.used == "llm-failure-cache"
    assert second.usage == {}
    history = json.loads(first.failure_history_path.read_text(encoding="utf-8"))
    assert history["status"] == "failed"
    assert len(history["attempts"]) == 1
    assert history["attempts"][0]["response"] == response

    config.retry_failed = True
    third = classify_components_with_llm(analysis.signals, config=config)
    assert requests == 2
    assert third.used == "llm-failed-deterministic"
    history = json.loads(first.failure_history_path.read_text(encoding="utf-8"))
    assert len(history["attempts"]) == 2


def test_deployment_context_rejects_profile_infrastructure_conflicts() -> None:
    with pytest.raises(ValidationError, match="on-premises deployment profile conflicts"):
        DeploymentContext.model_validate(
            {
                "system": "conflict",
                "deployment": {"profile": "on-premises"},
                "provided_blocks": [
                    {
                        "id": "B0",
                        "type": "base_infrastructure",
                        "subtype": "cloud_infrastructure",
                        "implementation": "aws",
                    }
                ],
            }
        )


def test_cloud_profile_uses_the_configured_provider(tmp_path: Path) -> None:
    repository = tmp_path / "cloud-docs"
    (repository / "docs/aws").mkdir(parents=True)
    (repository / "docs/on-prem").mkdir(parents=True)
    (repository / "docs/aws/deployment.md").write_text(
        "# AWS deployment\n\nUse Terraform and Helm.\n", encoding="utf-8"
    )
    (repository / "docs/on-prem/deployment.md").write_text(
        "# On-premises deployment\n\nUse Ansible and Helm.\n", encoding="utf-8"
    )
    context = tmp_path / "cloud-context.yaml"
    context.write_text(
        yaml.safe_dump(
            {
                "system": "cloud-fixture",
                "deployment": {"profile": "cloud"},
                "provided_blocks": [
                    {
                        "id": "B0",
                        "type": "base_infrastructure",
                        "subtype": "cloud_infrastructure",
                        "implementation": "aws",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context,
            cache_dir=tmp_path / "cache",
            synthesizer="deterministic",
        )
    )

    root_paths = {root.source_ref.path for root in result.signals.deployment_roots}
    assert "docs/aws/deployment.md" in root_paths
    assert "docs/on-prem/deployment.md" not in root_paths


def test_analyze_cli_writes_two_default_files_in_timestamped_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repository, context, gold = _write_fixture(tmp_path)
    output = tmp_path / "output"
    monkeypatch.chdir(tmp_path)

    exit_code = main(
        [
            "deployment",
            "analyze",
            "--repo",
            str(repository),
            "--context",
            str(context),
            "--gold",
            str(gold),
            "--output",
            str(output),
            "--synthesizer",
            "deterministic",
        ]
    )

    assert exit_code == 0
    run_directories = list((output / "runs").iterdir())
    assert len(run_directories) == 1
    assert run_directories[0].name.endswith("-fixture")
    assert sorted(path.name for path in run_directories[0].iterdir()) == [
        "analysis-report.md",
        "functional-blocks.yaml",
    ]
    assert (output / ".source-cache").is_dir()
    assert f"run: {run_directories[0]}" in capsys.readouterr().out


def test_compose_preserves_health_dependency(tmp_path: Path) -> None:
    repository = tmp_path / "compose-repository"
    repository.mkdir()
    (repository / "compose.yaml").write_text(
        """services:
  database:
    image: postgres:16
    healthcheck:
      test: [CMD, pg_isready]
  api:
    image: example/api
    depends_on:
      database:
        condition: service_healthy
""",
        encoding="utf-8",
    )
    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            synthesizer="deterministic",
        )
    )

    assert {item.name for item in result.signals.candidate_components} == {
        "Api",
        "Database",
    }
    assert any(
        relation.source
        == next(item.id for item in result.signals.candidate_components if item.name == "Api")
        and relation.target
        == next(
            item.id
            for item in result.signals.candidate_components
            if item.name == "Database"
        )
        and relation.relation == "health_gated_by"
        and relation.strength == "explicit_health_dependency"
        for relation in result.signals.relations
    )
    assert any(item.subject == "database" for item in result.signals.validation_signals)


def test_terraform_and_ansible_preserve_explicit_vs_order_semantics(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "structured-repository"
    (repository / "terraform").mkdir(parents=True)
    (repository / "playbooks").mkdir()
    (repository / "terraform/main.tf").write_text(
        """module \"network\" { source = \"./modules/network\" }
module \"cluster\" {
  source = \"./modules/cluster\"
  depends_on = [module.network]
}
""",
        encoding="utf-8",
    )
    (repository / "playbooks/site.yml").write_text(
        """- hosts: all
  roles:
    - database
    - application
""",
        encoding="utf-8",
    )
    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            synthesizer="deterministic",
        )
    )

    relations = result.signals.relations
    component_ids = {
        item.name.casefold(): item.id for item in result.signals.candidate_components
    }
    assert any(
        item.source == component_ids["cluster"]
        and item.target == component_ids["network"]
        and item.relation == "requires"
        and item.strength == "explicit_dependency"
        for item in relations
    )
    assert any(
        item.source == component_ids["database"]
        and item.target == component_ids["application"]
        and item.relation == "ordered_before"
        and item.strength == "declared_order"
        for item in relations
    )


def test_helm_kustomize_flux_and_argo_are_materialized_units(tmp_path: Path) -> None:
    repository = tmp_path / "gitops-repository"
    (repository / "charts/platform").mkdir(parents=True)
    (repository / "overlays/prod").mkdir(parents=True)
    (repository / "charts/platform/Chart.yaml").write_text(
        """apiVersion: v2
name: platform
dependencies:
  - name: redis
    version: 1.0.0
    repository: https://example.invalid/charts
""",
        encoding="utf-8",
    )
    (repository / "overlays/prod/kustomization.yaml").write_text(
        """apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - workload.yaml
""",
        encoding="utf-8",
    )
    (repository / "overlays/prod/workload.yaml").write_text(
        """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web-api
""",
        encoding="utf-8",
    )
    gitops_repository = tmp_path / "flux-argo-repository"
    gitops_repository.mkdir()
    (gitops_repository / "flux.yaml").write_text(
        """apiVersion: kustomize.toolkit.fluxcd.io/v1
kind: Kustomization
metadata:
  name: application
spec:
  dependsOn:
    - name: platform
""",
        encoding="utf-8",
    )
    (gitops_repository / "argo.yaml").write_text(
        """apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: delivery
spec:
  source:
    path: overlays/prod
""",
        encoding="utf-8",
    )
    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            node_budget=50,
            synthesizer="deterministic",
        )
    )

    components = {item.name: item for item in result.signals.candidate_components}
    assert {"Platform", "Redis", "Prod", "Web Api"} <= set(components)
    assert components["Platform"].deployment.executor == "helm"

    gitops_result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(gitops_repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "gitops-context.yaml"),
            cache_dir=tmp_path / "gitops-cache",
            synthesizer="deterministic",
        )
    )
    gitops_components = {
        item.name: item for item in gitops_result.signals.candidate_components
    }
    assert {"Application", "Delivery"} <= set(gitops_components)
    assert gitops_components["Delivery"].deployment.executor == "gitops"
    assert any(
        item.source == gitops_components["Application"].id
        and item.target == "platform"
        and item.relation == "health_gated_by"
        for item in gitops_result.signals.relations
    )
