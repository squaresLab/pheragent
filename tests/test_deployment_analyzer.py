from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from pheragent.cli import main
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
kubectl rollout status deployment/idrepo
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
            llm_enabled=False,
        )
    )

    components = {component.name: component for component in result.signals.candidate_components}
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
    assert any(block.type == AnalysisBlockType.SHARED_SERVICES for block in result.document.blocks)


def test_human_gold_never_changes_discovery_or_synthesis(tmp_path: Path) -> None:
    repository, context, reviewed_gold = _write_fixture(tmp_path)
    incorrect_gold = tmp_path / "incorrect-gold.yaml"
    incorrect_gold.write_text(
        yaml.safe_dump(
            {
                "expected_components": ["Imaginary Component"],
                "forbidden_components": [
                    "PostgreSQL",
                    "Keycloak",
                    "ActiveMQ",
                    "Kafka",
                    "Kernel",
                    "Idrepo",
                ],
                "expected_classifications": {
                    "PostgreSQL": {
                        "block_type": "operations",
                        "subtype": "observability",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    def analyze(gold_path: Path):
        return run_repository_analysis(
            AnalysisConfig(
                repositories=[str(repository)],
                documentation=[],
                context_path=context,
                cache_dir=tmp_path / "cache",
                gold_path=gold_path,
                llm_enabled=False,
            )
        )

    reviewed = analyze(reviewed_gold)
    incorrect = analyze(incorrect_gold)

    assert reviewed.signals == incorrect.signals
    assert reviewed.document.blocks == incorrect.document.blocks
    assert reviewed.document.levels == incorrect.document.levels
    assert reviewed.document.evaluation != incorrect.document.evaluation


def test_timestamped_run_directories_do_not_overwrite(tmp_path: Path) -> None:
    now = datetime(2026, 8, 7, 10, 11, 12, tzinfo=UTC)

    first = create_timestamped_run_directory(tmp_path, name="Apache Airflow", now=now)
    second = create_timestamped_run_directory(tmp_path, name="Apache Airflow", now=now)

    assert first.name == "20260807T101112Z-apache-airflow"
    assert second.name == "20260807T101112Z-apache-airflow-01"
    assert first.is_dir()
    assert second.is_dir()


def test_analysis_pipeline_makes_two_bounded_llm_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, context, _gold = _write_fixture(tmp_path)
    requests: list[str] = []
    response_formats: list[dict] = []

    class Responses:
        def create(self, **payload):
            request = json.loads(payload["input"][0]["content"][0]["text"])
            requests.append(request["task"])
            response_formats.append(payload["text"]["format"])
            if request["task"] == "plan_deployment_investigation":
                response = {
                    "queries": [
                        {
                            "id": "query-services",
                            "purpose": "missing_components",
                            "terms": ["service"],
                            "source_scope": "both",
                            "path_prefix": None,
                            "reason_code": "BROAD_SERVICE_SWEEP",
                        }
                    ],
                    "hypotheses": [],
                }
            else:
                evidence_ids = [item["id"] for item in request["evidence"]]

                def evidence_for(component):
                    return next(
                        (
                            item["id"]
                            for item in request["evidence"]
                            if item["path"] == component["entrypoint"]
                        ),
                        evidence_ids[0],
                    )

                response = {
                    "classification_groups": [
                        {
                            "component_ids": [component["id"]],
                            "disposition": "deployment_component",
                            "classification": (
                                "shared_services.data_services"
                                if component["name"] == "PostgreSQL"
                                else "shared_services.messaging_integration"
                                if component["name"] in {"ActiveMQ", "Kafka"}
                                else "shared_services.identity_security"
                                if component["name"] == "Keycloak"
                                else "shared_services.external_integration"
                                if "/external/" in component["entrypoint"]
                                else "application.domain_service"
                            ),
                            "domain": None,
                            "evidence_ids": [evidence_for(component)],
                            "confidence": 0.9,
                            "reason_code": "DEPLOYMENT_ENTRYPOINT",
                        }
                        for component in request["components"]
                    ],
                    "deployment_actions": [],
                    "renames": [],
                    "implied_entities": [],
                    "facts": [],
                    "disagreements": [],
                    "unresolved": [],
                }
            return [
                {
                    "type": "response.output_text.delta",
                    "delta": json.dumps(response),
                },
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "total_tokens": 150,
                        }
                    },
                },
            ]

    monkeypatch.setenv("TEST_OPENAI_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    config = AnalysisConfig(
        repositories=[str(repository)],
        documentation=[],
        context_path=context,
        cache_dir=tmp_path / "source-cache",
        api_key_env="TEST_OPENAI_KEY",
        llm_cache_dir=tmp_path / "llm-cache",
    )

    first = run_repository_analysis(config)
    second = run_repository_analysis(config)

    assert requests == [
        "plan_deployment_investigation",
        "synthesize_deployment_investigation",
    ]
    assert all("uniqueItems" not in json.dumps(item) for item in response_formats)
    assert first.llm_stage_statuses == {
        "investigation_plan": "llm",
        "investigation_synthesis": "llm",
    }
    assert first.document.evaluation.llm_requests == 2
    assert first.document.evaluation.llm_input_tokens == 200
    assert len(first.investigation.plan_input["source_outlines"]) <= 30
    assert all(
        "entrypoint" not in component
        for component in first.investigation.plan_input["known_components"]
    )
    assert second.llm_stage_statuses == {
        "investigation_plan": "cache",
        "investigation_synthesis": "cache",
    }
    assert second.document.evaluation.llm_requests == 0
    assert first.workflow.ready_for_execution is True
    workflow_steps = {step.component_name: step for step in first.workflow.steps}
    assert workflow_steps["PostgreSQL"].command == "./install.sh"
    assert workflow_steps["PostgreSQL"].operation_source_ref is not None
    assert set(workflow_steps["Kernel"].after) >= {
        workflow_steps[name].id for name in ("PostgreSQL", "Keycloak", "ActiveMQ", "Kafka")
    }
    assert first.workflow.validation_checks
    assert (
        next(
            component
            for component in first.signals.candidate_components
            if component.name == "Kernel"
        ).disposition
        == "deployment_component"
    )
    assert any(block.subtype == "data_services" for block in first.document.blocks)


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
            llm_enabled=False,
        )
    )

    root_paths = {root.source_ref.path for root in result.signals.deployment_roots}
    assert "docs/aws/deployment.md" in root_paths
    assert "docs/on-prem/deployment.md" not in root_paths


def test_analyze_cli_writes_execution_readiness_outputs_in_timestamped_run(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    repository, context, _gold = _write_fixture(tmp_path)
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
            "--output",
            str(output),
            "--openai-api-key-env",
            "TEST_MISSING_OPENAI_KEY",
        ]
    )

    assert exit_code == 0
    run_directories = list((output / "runs").iterdir())
    assert len(run_directories) == 1
    assert run_directories[0].name.endswith("-fixture")
    assert sorted(path.name for path in run_directories[0].iterdir()) == [
        "analysis-report.md",
        "deployment-workflow.yaml",
        "events.jsonl",
        "functional-blocks.yaml",
        "metrics.json",
        "run-manifest.json",
    ]
    assert (output / ".source-cache").is_dir()
    workflow = yaml.safe_load(
        (run_directories[0] / "deployment-workflow.yaml").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (run_directories[0] / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert workflow["ready_for_execution"] is False
    assert manifest["analysis_method"] == "deployment-analysis-v1"
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
            llm_enabled=False,
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
        == next(item.id for item in result.signals.candidate_components if item.name == "Database")
        and relation.relation == "health_gated_by"
        and relation.strength == "explicit_health_dependency"
        for relation in result.signals.relations
    )
    assert any(item.subject == "database" for item in result.signals.validation_signals)


def test_compose_default_invocation_selects_referenced_overlays_only(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "layered-compose-repository"
    repository.mkdir()
    (repository / "compose.yaml").write_text(
        """# Default deployment:
# docker compose -f compose.yaml -f compose.full.yaml -f compose.observability.yaml up
# Optional test deployment:
# docker compose -f compose.yaml -f compose.tests.yaml up
services:
  api:
    image: example/api
""",
        encoding="utf-8",
    )
    (repository / "compose.full.yaml").write_text(
        """services:
  worker:
    image: example/worker
""",
        encoding="utf-8",
    )
    (repository / "compose.observability.yaml").write_text(
        """services:
  collector:
    image: example/collector
""",
        encoding="utf-8",
    )
    (repository / "compose.tests.yaml").write_text(
        """services:
  test-runner:
    image: example/tests
""",
        encoding="utf-8",
    )
    component_docs = repository / "src/api/README.md"
    component_docs.parent.mkdir(parents=True)
    component_docs.write_text(
        """# API development

## Build locally

```bash
docker compose up
```
""",
        encoding="utf-8",
    )

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            llm_enabled=False,
        )
    )

    components = {item.name: item for item in result.signals.candidate_components}
    assert set(components) == {"Api", "Worker", "Collector"}
    assert all(
        component.deployment.command == "docker compose -f compose.yaml -f compose.full.yaml "
        "-f compose.observability.yaml up"
        for component in components.values()
    )
    assert all(
        component.deployment.operation_source_ref is not None for component in components.values()
    )
    assert len(result.workflow.steps) == 1
    compose_step = result.workflow.steps[0]
    assert {target.name for target in compose_step.targets} == {"Api", "Worker", "Collector"}
    assert compose_step.command == (
        "docker compose -f compose.yaml -f compose.full.yaml "
        "-f compose.observability.yaml up"
    )
    assert compose_step.after == []
    selected = {path.split(":", 1)[-1] for path in result.selected_paths}
    assert {"compose.yaml", "compose.full.yaml", "compose.observability.yaml"} <= selected
    assert "compose.tests.yaml" not in selected


def test_documented_deployment_commands_contribute_grounded_components(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "documented-repository"
    repository.mkdir()
    (repository / "deployment.md").write_text(
        """# Production deployment

## Identity provider

```bash
helm upgrade --install keycloak charts/keycloak
```

## Database stack

```bash
ansible-playbook playbooks/postgresql.yml
```
""",
        encoding="utf-8",
    )

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            llm_enabled=False,
        )
    )

    components = {item.name: item for item in result.signals.candidate_components}
    assert {"Keycloak", "PostgreSQL"} <= set(components)
    assert components["Keycloak"].deployment.executor == "helm"
    assert components["PostgreSQL"].deployment.executor == "ansible"
    assert all(item.source_ref.path == "deployment.md" for item in components.values())


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
            llm_enabled=False,
        )
    )

    relations = result.signals.relations
    component_ids = {item.name.casefold(): item.id for item in result.signals.candidate_components}
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


def test_terraform_resources_are_reduced_to_a_grounded_stack_candidate(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "terraform-repository"
    module = repository / "terraform/modules/database"
    module.mkdir(parents=True)
    (module / "main.tf").write_text(
        """resource "aws_db_instance" "primary" {}
resource "aws_security_group" "database" {}
resource "aws_secretsmanager_secret" "credentials" {}
""",
        encoding="utf-8",
    )

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=_minimal_context(tmp_path / "context.yaml"),
            cache_dir=tmp_path / "cache",
            llm_enabled=False,
        )
    )

    assert len(result.signals.candidate_components) == 1
    candidate = result.signals.candidate_components[0]
    assert candidate.name == "Database Infrastructure"
    assert candidate.deployment.executor == "terraform"
    assert candidate.materialized_names == [
        "aws_db_instance.primary",
        "aws_security_group.database",
        "aws_secretsmanager_secret.credentials",
    ]


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
            llm_enabled=False,
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
            llm_enabled=False,
        )
    )
    gitops_components = {item.name: item for item in gitops_result.signals.candidate_components}
    assert {"Application", "Delivery"} <= set(gitops_components)
    assert gitops_components["Delivery"].deployment.executor == "gitops"
    assert any(
        item.source == gitops_components["Application"].id
        and item.target == "platform"
        and item.relation == "health_gated_by"
        for item in gitops_result.signals.relations
    )
