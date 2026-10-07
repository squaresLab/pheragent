from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pheragent.deployment.enums import SourceKind
from pheragent.deployment.models import SourceManifestEntry, SourceSpec
from pheragent.deployment.overview import (
    DeploymentOverview,
    EntrypointSelection,
    active_step,
    apply_change,
    create_overview,
    set_discoveries,
    set_step_status,
)
from pheragent.deployment.source_manager import AcquiredSource
from pheragent.deployment.sources import SourceTools
from pheragent.deployment.task import OverviewChange


def _source(path: Path) -> AcquiredSource:
    spec = SourceSpec(
        id="repository-1",
        kind=SourceKind.LOCAL_DIRECTORY,
        location=str(path),
        purpose="repository",
    )
    return AcquiredSource(
        id=spec.id,
        kind=spec.kind,
        path=path,
        spec=spec,
        manifest=SourceManifestEntry(
            id=spec.id,
            kind=spec.kind,
            location=spec.location,
            content_hash="a" * 64,
        ),
    )


def test_model_selects_route_then_builds_overview_from_complete_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("See [run guide](docs/run.md).\n")
    (tmp_path / "docs").mkdir()
    guide = "\n".join(f"line {line}" for line in range(1, 151))
    (tmp_path / "docs/run.md").write_text(guide)
    sources = SourceTools((_source(tmp_path),))

    class Classifier:
        def classify(self, *, stage, payload, validate, **_kwargs):
            if stage == "deployment_entrypoints":
                value = EntrypointSelection(
                    route="documented local route",
                    files=["repository-1:README.md", "repository-1:docs/run.md"],
                    reason="root guide links to run guide",
                )
            else:
                assert payload["documents"][1]["text"].endswith("line 150")
                value = DeploymentOverview.model_validate(
                    {
                        "route": "placeholder",
                        "route_evidence": ["repository-1:README.md"],
                        "stages": [
                            {
                                "id": "deploy",
                                "kind": "deploy",
                                "goal": "deploy the application",
                                "steps": [
                                    {
                                        "id": "deploy.app",
                                        "goal": "start services",
                                        "success_condition": "services are healthy",
                                        "source_refs": ["repository-1:docs/run.md"],
                                    }
                                ],
                            }
                        ],
                    }
                )
            validate(value)
            return SimpleNamespace(value=value, usage={"requests": 1}, warning=None)

    overview, usage = create_overview(
        Classifier(),
        objective="Deploy sample",
        target={"type": "shell"},
        sources=sources,
    )

    assert overview.route == "documented local route"
    assert overview.route_evidence == ["repository-1:README.md", "repository-1:docs/run.md"]
    assert active_step(overview).id == "deploy.app"
    assert active_step(overview).source_refs == ["repository-1:docs/run.md"]
    assert usage == {"requests": 2}
    set_discoveries(overview, ["database endpoint missing", "database endpoint missing"])
    assert overview.discoveries == ["database endpoint missing"]
    set_step_status(overview, "deploy.app", "verified")
    assert active_step(overview) is None


def test_grounded_prerequisite_is_inserted_before_its_consumer(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Install storage before the service.\n")
    sources = SourceTools((_source(tmp_path),))
    overview = DeploymentOverview.model_validate(
        {
            "route": "local",
            "route_evidence": ["repository-1:README.md"],
            "stages": [
                {
                    "id": "deploy",
                    "kind": "deploy",
                    "goal": "deploy",
                    "steps": [
                        {
                            "id": "service",
                            "goal": "deploy service",
                            "success_condition": "service is ready",
                            "source_refs": ["repository-1:README.md"],
                        }
                    ],
                }
            ],
        }
    )

    apply_change(
        overview,
        OverviewChange(
            operation="insert_before",
            step_id="storage",
            before_step_id="service",
            goal="provide storage",
            success_condition="a dynamic provisioner is ready",
            source_refs=["repository-1:README.md"],
        ),
        sources,
    )

    assert active_step(overview).id == "storage"
    assert [step.id for step in overview.stages[0].steps] == ["storage", "service"]
