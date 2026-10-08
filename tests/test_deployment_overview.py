from __future__ import annotations

from pathlib import Path

from pheragent.deployment.models import SourceKind, SourceManifestEntry, SourceSpec
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
    guide = "See [service details](details.md).\n" + "\n".join(
        f"line {line}" for line in range(2, 151)
    )
    (tmp_path / "docs/run.md").write_text(guide)
    (tmp_path / "docs/details.md").write_text("The frontend is the user-facing service.\n")
    sources = SourceTools((_source(tmp_path),))

    class Client:
        def complete(self, response_model, *, payload, **_kwargs):
            if response_model is EntrypointSelection:
                value = EntrypointSelection(
                    route="documented local route",
                    files=["repository-1:README.md", "repository-1:docs/run.md"],
                    reason="root guide links to run guide",
                )
            else:
                assert payload["documents"][1]["text"].endswith("line 150")
                assert payload["related_sources"] == ["repository-1:docs/details.md"]
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
                                        "components": ["frontend"],
                                        "source_refs": ["repository-1:docs/run.md"],
                                        "related_sources": [
                                            "repository-1:docs/details.md"
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                )
            return value, {"requests": 1}

    overview, usage = create_overview(
        Client(),
        objective="Deploy sample",
        target={"type": "shell"},
        sources=sources,
    )

    assert overview.route == "documented local route"
    assert overview.route_evidence == ["repository-1:README.md", "repository-1:docs/run.md"]
    assert active_step(overview).id == "deploy.app"
    assert active_step(overview).source_refs == ["repository-1:docs/run.md"]
    assert active_step(overview).components == ["frontend"]
    assert active_step(overview).related_sources == ["repository-1:docs/details.md"]
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
