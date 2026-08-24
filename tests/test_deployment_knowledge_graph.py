from pathlib import Path

from pheragent.deployment.analysis_models import (
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisRelation,
    AnalysisRelationType,
    AnalysisSourceRef,
    CandidateComponent,
    ComponentClassification,
    ComponentDeployment,
    DeploymentSignalBundle,
    ReferenceNode,
    SignalStrength,
)
from pheragent.deployment.analyzer import AnalysisConfig, run_repository_analysis
from pheragent.deployment.enums import AnalysisTreatment
from pheragent.deployment.evidence import schedule_investigation_queries
from pheragent.deployment.investigation_models import (
    EvidenceKind,
    InvestigationPlan,
    InvestigationPurpose,
    InvestigationQuery,
    SourceScope,
)
from pheragent.deployment.knowledge_graph import (
    KnowledgeEdgeKind,
    KnowledgeGapKind,
    graph_gap_queries,
    project_deployment_knowledge,
)
from pheragent.deployment.serialization import load_sources_config


def _component() -> CandidateComponent:
    source_ref = AnalysisSourceRef(repo_id="fixture", path="deploy/cart/service.yaml")
    return CandidateComponent(
        id="C001_cart",
        name="Cart",
        capabilities=["cart-api"],
        source_ref=source_ref,
        deployment=ComponentDeployment(
            executor=AnalysisExecutor.KUBERNETES,
            entrypoint=source_ref.path,
        ),
        classification=ComponentClassification(
            block_type=AnalysisBlockType.APPLICATION,
            subtype="domain_service",
            confidence=1.0,
        ),
    )


def _references() -> list[ReferenceNode]:
    return [
        ReferenceNode(
            id="fixture-service",
            repo_id="fixture",
            path="deploy/cart/service.yaml",
            node_type="kubernetes",
        ),
        ReferenceNode(
            id="fixture-installer",
            repo_id="fixture",
            path="deploy/cart/install.sh",
            node_type="shell",
        ),
    ]


def _analyze_multihop(tmp_path: Path, treatment: AnalysisTreatment):
    case_dir = Path("research/cases/multihop").resolve()
    return run_repository_analysis(
        AnalysisConfig(
            repositories=[],
            documentation=[],
            sources=load_sources_config(case_dir / "sources.yaml"),
            context_path=case_dir / "context.yaml",
            cache_dir=tmp_path / treatment.value,
            treatment=treatment,
            llm_enabled=False,
        )
    )


def test_graph_projection_is_stable_and_evidence_backed() -> None:
    component = _component()
    evidence = [component.source_ref]
    relation = AnalysisRelation(
        source="fixture-installer",
        target="fixture-service",
        relation=AnalysisRelationType.INVOKES,
        strength=SignalStrength.STRUCTURAL_REFERENCE,
        evidence=evidence,
    )
    signals = DeploymentSignalBundle(candidate_components=[component])

    first = project_deployment_knowledge(signals, _references(), [relation])
    second = project_deployment_knowledge(signals, list(reversed(_references())), [relation])

    assert first == second
    assert all(edge.evidence for edge in first.edges)
    assert any(edge.kind == KnowledgeEdgeKind.INSTALLS for edge in first.edges)
    assert any(edge.kind == KnowledgeEdgeKind.INVOKES for edge in first.edges)


def test_graph_gap_becomes_a_bounded_retrieval_query() -> None:
    graph = project_deployment_knowledge(
        DeploymentSignalBundle(candidate_components=[_component()]),
        _references(),
        [],
    )

    assert graph.gaps[0].kind == KnowledgeGapKind.MISSING_OPERATION
    queries = graph_gap_queries(graph, limit=1)
    assert len(queries) == 1
    assert queries[0].component_id == "C001_cart"
    assert queries[0].reason_code == "GRAPH_MISSING_OPERATION"
    assert "Cart" in queries[0].terms


def test_graph_queries_replace_model_queries_under_the_same_budget() -> None:
    plan = InvestigationPlan(
        queries=[
            InvestigationQuery(
                id=f"model-query-{index}",
                purpose=InvestigationPurpose.MISSING_COMPONENTS,
                terms=[f"candidate-{index}"],
                source_scope=SourceScope.BOTH,
                reason_code=f"MODEL_QUERY_{index}",
            )
            for index in range(1, 5)
        ]
    )
    guided = graph_gap_queries(
        project_deployment_knowledge(
            DeploymentSignalBundle(candidate_components=[_component()]),
            _references(),
            [],
        ),
        limit=2,
    )

    hybrid = schedule_investigation_queries(plan)
    graph_hybrid = schedule_investigation_queries(plan, guided_queries=guided)

    assert len(hybrid) == len(graph_hybrid) == 8
    assert sum(query.reason_code.startswith("GRAPH_") for query in hybrid) == 0
    assert sum(query.reason_code.startswith("GRAPH_") for query in graph_hybrid) == 2


def test_graph_guidance_closes_multihop_installation_routes(tmp_path: Path) -> None:
    baseline = _analyze_multihop(tmp_path, AnalysisTreatment.DETERMINISTIC)
    result = _analyze_multihop(tmp_path, AnalysisTreatment.HYBRID_GRAPH)

    assert all(
        component.deployment is not None and component.deployment.command is None
        for component in baseline.signals.candidate_components
    )
    routes = [
        (observation.path, observation.query_id)
        for observation in result.investigation.observations
        if observation.kind == EvidenceKind.INSTALLATION_ROUTE
    ]
    assert {
        ("scripts/install-cart.sh", "component-C001_cart"),
        ("scripts/install-dependencies.sh", "component-C002_redis"),
    } <= set(routes)
    commands = {
        component.name: component.deployment.command
        for component in result.signals.candidate_components
        if component.deployment is not None
    }
    assert commands == {
        "Cart": "./install-cart.sh",
        "Redis": "./install-dependencies.sh",
    }
