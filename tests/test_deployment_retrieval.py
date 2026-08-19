from pathlib import Path

import yaml

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
from pheragent.deployment.analyzer import AnalysisConfig, run_repository_analysis
from pheragent.deployment.retrieval import (
    DeploymentRetrievalEngine,
    RetrievalQuery,
)
from pheragent.deployment.workflow import build_deployment_workflow


def _documents():
    return [
        DeploymentRetrievalEngine.document(
            source_id="repo",
            source_kind="repository",
            path="deployment/oauth2-proxy/oauth2-proxy.yaml",
            text="kind: Deployment\nmetadata:\n  name: oauth2-proxy\n",
        ),
        DeploymentRetrievalEngine.document(
            source_id="repo",
            source_kind="repository",
            path="deployment/oauth2-proxy/install.sh",
            text="#!/bin/bash\nkubectl apply -f ./oauth2-proxy.yaml\n",
        ),
        DeploymentRetrievalEngine.document(
            source_id="repo",
            source_kind="repository",
            path="deployment/iam/istio-addons/Chart.yaml",
            text="name: istio-addons\n",
        ),
        DeploymentRetrievalEngine.document(
            source_id="repo",
            source_kind="repository",
            path="deployment/iam/install.sh",
            text="#!/bin/bash\nhelm upgrade --install istio-addons ./istio-addons\n",
        ),
        DeploymentRetrievalEngine.document(
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
            DeploymentRetrievalEngine.document(
                source_id="repo",
                source_kind="repository",
                path="notes/unrelated.md",
                text="This file contains no deployment identifiers.",
            ),
        ]
    )

    noisy_result = [hit.document.path for hit in noisy.search(RetrievalQuery(terms=("oauth2",)))]
    assert noisy_result == baseline


def test_reference_extraction_handles_large_non_path_tokens() -> None:
    document = DeploymentRetrievalEngine.document(
        source_id="repo",
        source_kind="repository",
        path="large.yaml",
        text=f"{'a' * 500_000}\n./deployment/install.sh\n",
    )

    assert document.references == ("./deployment/install.sh",)


def test_analysis_binds_manifest_to_its_grounded_installer(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "deployment/all").mkdir(parents=True)
    (repository / "deployment/admin").mkdir(parents=True)
    (repository / "deployment/all/install-all.sh").write_text(
        "#!/bin/bash\ncd ../admin\n./install.sh\n",
        encoding="utf-8",
    )
    (repository / "deployment/admin/install.sh").write_text(
        "#!/bin/bash\n: ${NAMESPACE:?required}\n: ${1:?required}\n"
        "kubectl apply -f ./admin-proxy.yaml\n",
        encoding="utf-8",
    )
    (repository / "deployment/admin/admin-proxy.yaml").write_text(
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: admin-proxy\n",
        encoding="utf-8",
    )
    context_path = tmp_path / "context.yaml"
    context_path.write_text(
        yaml.safe_dump(
            {
                "system": "nested-installer-fixture",
                "deployment": {"profile": "kubernetes"},
                "provided_blocks": [],
            }
        ),
        encoding="utf-8",
    )

    result = run_repository_analysis(
        AnalysisConfig(
            repositories=[str(repository)],
            documentation=[],
            context_path=context_path,
            cache_dir=tmp_path / "cache",
            llm_enabled=False,
        )
    )

    proxy = next(
        component
        for component in result.signals.candidate_components
        if component.name == "Admin Proxy"
    )
    assert proxy.deployment is not None
    assert proxy.deployment.entrypoint == "deployment/admin/install.sh"
    proxy_steps = [step for step in result.workflow.steps if step.component_id == proxy.id]
    assert len(proxy_steps) == 1
    assert proxy_steps[0].command == "./install.sh"
    assert proxy_steps[0].required_inputs == ["arg:1", "env:NAMESPACE"]
    assert result.document.evaluation.orphan_component_count == 0


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
