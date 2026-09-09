from pheragent.deployment.analysis_models import (
    AnalysisBlockType,
    AnalysisExecutor,
    AnalysisRelation,
    AnalysisRelationType,
    AnalysisSourceRef,
    CandidateComponent,
    ComponentClassification,
    ComponentDeployment,
    DeploymentContext,
    DeploymentSignalBundle,
    SignalStrength,
)
from pheragent.deployment.functional_blocks import build_functional_blocks
from pheragent.deployment.retrieval import (
    DeploymentRetrievalEngine,
    RetrievalDocument,
    RetrievalQuery,
)
from pheragent.deployment.workflow import build_deployment_workflow


def _documents():
    return [
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="deployment/oauth2-proxy/oauth2-proxy.yaml",
            text="kind: Deployment\nmetadata:\n  name: oauth2-proxy\n",
        ),
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="deployment/oauth2-proxy/install.sh",
            text="#!/bin/bash\nkubectl apply -f ./oauth2-proxy.yaml\n",
        ),
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="deployment/iam/istio-addons/Chart.yaml",
            text="name: istio-addons\n",
        ),
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="deployment/iam/install.sh",
            text="#!/bin/bash\nhelm upgrade --install istio-addons ./istio-addons\n",
        ),
        RetrievalDocument(
            source_id="docs",
            source_kind="documentation",
            path="guide.md",
            text="Install the OAuth2 Proxy after identity services.",
        ),
    ]


def _engine() -> DeploymentRetrievalEngine:
    return DeploymentRetrievalEngine(_documents())


def test_retrieval_finds_sibling_installer_without_embeddings() -> None:
    hits = _engine().search(
        RetrievalQuery(
            terms=("OAuth2 Proxy",),
            source_kind="repository",
            seed_paths=("deployment/oauth2-proxy/oauth2-proxy.yaml",),
        ),
        limit=3,
    )

    assert "deployment/oauth2-proxy/install.sh" in [hit.document.path for hit in hits]
    installer = next(hit for hit in hits if hit.document.path.endswith("install.sh"))
    assert installer.line == 2


def test_retrieval_expands_to_parent_orchestrator_and_respects_source_scope() -> None:
    hits = _engine().search(
        RetrievalQuery(
            terms=("istio addons",),
            source_kind="repository",
            seed_paths=("deployment/iam/istio-addons/Chart.yaml",),
        ),
        limit=3,
    )

    assert "deployment/iam/install.sh" in [hit.document.path for hit in hits]
    assert all(hit.document.source_kind == "repository" for hit in hits)


def test_retrieval_is_stable_when_irrelevant_files_are_added() -> None:
    baseline = [hit.document.path for hit in _engine().search(RetrievalQuery(terms=("oauth2",)))]
    noisy = DeploymentRetrievalEngine(
        [
            *_documents(),
            RetrievalDocument(
                source_id="repo",
                source_kind="repository",
                path="notes/unrelated.md",
                text="This file contains no deployment identifiers.",
            ),
        ]
    )

    noisy_result = [hit.document.path for hit in noisy.search(RetrievalQuery(terms=("oauth2",)))]
    assert noisy_result == baseline


def test_retrieval_indexes_markdown_sections_as_source_grounded_passages() -> None:
    documents = DeploymentRetrievalEngine.passages(
        source_id="docs",
        source_kind="documentation",
        path="deployment.md",
        text=(
            "# Overview\nGeneral architecture.\n\n"
            "## Install Kafka\nRun the broker installer.\n"
            "helm upgrade --install kafka ./charts/kafka\n"
        ),
    )

    assert len(documents) == 2
    engine = DeploymentRetrievalEngine(documents)
    hit = engine.search(RetrievalQuery(terms=("install kafka",)), limit=1)[0]

    assert hit.document.start_line == 4
    assert hit.document.end_line == 6
    assert hit.line == 4
    assert "helm upgrade --install kafka" in hit.document.text


def test_retrieval_expands_over_parsed_file_relations() -> None:
    documents = [
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="docs/deployment.md",
            text="Install the message broker.",
        ),
        RetrievalDocument(
            source_id="repo",
            source_kind="repository",
            path="scripts/bootstrap.sh",
            text="execute-the-unrelated-binary --flag",
        ),
    ]
    engine = DeploymentRetrievalEngine(
        documents,
        relations=(("repo:docs/deployment.md", "repo:scripts/bootstrap.sh"),),
    )

    hits = engine.search(
        RetrievalQuery(
            terms=("message broker",),
            seed_paths=("docs/deployment.md",),
        ),
        limit=2,
    )

    assert [hit.document.path for hit in hits] == [
        "docs/deployment.md",
        "scripts/bootstrap.sh",
    ]


def test_owned_component_relation_does_not_create_owner_self_dependency() -> None:
    deployment = ComponentDeployment(
        executor=AnalysisExecutor.SHELL,
        entrypoint="install.sh",
        command="./install.sh",
        operation_source_ref=AnalysisSourceRef(repo_id="repo", path="install.sh"),
    )
    classification = ComponentClassification(
        block_type=AnalysisBlockType.APPLICATION,
        subtype="core_application",
        confidence=1.0,
    )
    owner = CandidateComponent(
        id="C001_owner",
        name="Owner",
        source_ref=AnalysisSourceRef(repo_id="repo", path="install.sh"),
        deployment=deployment,
        classification=classification,
    )
    child = CandidateComponent(
        id="C002_child",
        name="Child",
        source_ref=AnalysisSourceRef(repo_id="repo", path="install.sh"),
        deployment=deployment,
        installed_by=owner.id,
        classification=classification,
    )
    signals = DeploymentSignalBundle(
        candidate_components=[owner, child],
        relations=[
            AnalysisRelation(
                source=owner.id,
                target=child.id,
                relation=AnalysisRelationType.REQUIRES,
                strength=SignalStrength.LLM_INFERRED,
            )
        ],
    )

    workflow = build_deployment_workflow(
        context=DeploymentContext(system="owner-fixture"),
        signals=signals,
        observations=(),
        synthesis=None,
        llm_completed=False,
        mandatory_probes=(),
        documentation_expected=False,
    )

    assert len(workflow.steps) == 1
    assert workflow.steps[0].component_id == owner.id
    assert workflow.steps[0].after == []


def test_functional_block_dependencies_follow_component_relations() -> None:
    database = CandidateComponent(
        id="C001_database",
        name="Database",
        source_ref=AnalysisSourceRef(repo_id="repo", path="database/install.sh"),
        deployment=ComponentDeployment(
            executor=AnalysisExecutor.SHELL,
            entrypoint="database/install.sh",
        ),
        classification=ComponentClassification(
            block_type=AnalysisBlockType.SHARED_SERVICES,
            subtype="data_services",
            confidence=1.0,
        ),
    )
    application = CandidateComponent(
        id="C002_application",
        name="Application",
        source_ref=AnalysisSourceRef(repo_id="repo", path="application/install.sh"),
        deployment=ComponentDeployment(
            executor=AnalysisExecutor.SHELL,
            entrypoint="application/install.sh",
        ),
        classification=ComponentClassification(
            block_type=AnalysisBlockType.APPLICATION,
            subtype="core_application",
            confidence=1.0,
        ),
    )
    signals = DeploymentSignalBundle(
        candidate_components=[database, application],
        relations=[
            AnalysisRelation(
                source=database.id,
                target=application.id,
                relation=AnalysisRelationType.REQUIRES,
                strength=SignalStrength.EXPLICIT_DEPENDENCY,
            )
        ],
    )

    document = build_functional_blocks(
        DeploymentContext(system="relation-fixture"),
        signals,
        {},
        None,
        llm_usage={},
        llm_stage_statuses={},
    )
    block_by_component = {
        component.id: block
        for block in document.blocks
        for component in block.components
    }

    assert block_by_component[database.id].after == [block_by_component[application.id].id]
