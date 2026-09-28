import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from pheragent.cli import main
from pheragent.deployment.analysis_llm import strict_response_format
from pheragent.deployment.analysis_models import DeploymentContext
from pheragent.deployment.evidence_oracle import _local_link_targets
from pheragent.deployment.recursive_compiler import compile_recursive_plan
from pheragent.deployment.recursive_plan import (
    _PLACEHOLDER,
    ActionKind,
    ExternalSourceRequest,
    PlanNode,
    RecursivePlan,
    RequiredInput,
    StepAction,
    StepState,
    _load_plan,
    _validate_action,
)
from pheragent.deployment.serialization import write_yaml


def _study(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "install.sh").write_text("#!/bin/sh\necho deploy\n", encoding="utf-8")
    (tmp_path / "context.yaml").write_text(
        yaml.safe_dump({"system": "fixture", "deployment": {}, "provided_blocks": []}),
        encoding="utf-8",
    )
    (tmp_path / "sources.yaml").write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "sources": [
                    {
                        "id": "fixture",
                        "kind": "local_directory",
                        "location": str(repository),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    study = tmp_path / "study.yaml"
    study.write_text(
        yaml.safe_dump(
            {
                "version": "0.1",
                "id": "fixture-study",
                "max_total_llm_requests": 0,
                "treatments": ["a0"],
                "cases": [
                    {
                        "id": "fixture",
                        "sources": "sources.yaml",
                        "context": "context.yaml",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return study


def test_research_run_defaults_to_cost_preflight(tmp_path: Path, capsys) -> None:
    exit_code = main(["research", "run", "--study", str(_study(tmp_path))])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "planned runs: 1" in output
    assert "maximum LLM requests: 0" in output
    assert "preflight only" in output


def test_research_run_uses_product_analyzer_and_seals_results(
    tmp_path: Path,
    capsys,
) -> None:
    study = _study(tmp_path)
    output = tmp_path / "results"
    exit_code = main(
        ["research", "run", "--study", str(study), "--output", str(output), "--execute"]
    )

    assert exit_code == 0
    study_root = output / "fixture-study"
    run_dir = next((study_root / "runs").iterdir())
    manifest = json.loads(
        (run_dir / ".heragent" / "run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_kind"] == "research"
    assert manifest["analysis_method"] == "a0"
    assert manifest["status"] == "completed"
    assert (run_dir / "functional-blocks.yaml").is_file()
    assert (study_root / "results.csv").is_file()
    assert "results:" in capsys.readouterr().out


def test_research_one_shot_writes_outline_corpus_and_usage(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    study = _study(tmp_path)
    sources = study.with_name("sources.yaml")
    context = study.with_name("context.yaml")
    output = tmp_path / "one-shot"

    class Responses:
        def create(self, **payload):
            assert "max_output_tokens" not in payload
            assert "install.sh" in payload["input"][0]["content"][0]["text"]
            assert "echo deploy" not in payload["input"][0]["content"][0]["text"]
            document = {
                "system": "fixture",
                "stages": [
                    {
                        "title": "Deploy fixture",
                        "goal": "Make the fixture service ready.",
                        "children": [],
                    }
                ],
            }
            return [
                {"type": "response.output_text.delta", "delta": json.dumps(document)},
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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )

    exit_code = main(
        [
            "research",
            "one-shot",
            "--sources",
            str(sources),
            "--context",
            str(context),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    run_dir = next((output / "runs").iterdir())
    corpus = (run_dir / "corpus.txt").read_text(encoding="utf-8")
    assert "  install.sh" in corpus
    assert "echo deploy" not in corpus
    assert (run_dir / "deployment-outline.yaml").is_file()
    usage = json.loads((run_dir / "usage.json").read_text(encoding="utf-8"))
    assert usage["total_tokens"] == 120
    assert "LLM usage: input=100; output=20" in capsys.readouterr().out


def _recursive_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    (repository / "components" / "postgres").mkdir(parents=True)
    (repository / "README.md").write_text(
        "# Fixture\n\nInstall the PostgreSQL component.\n",
        encoding="utf-8",
    )
    (repository / "components" / "postgres" / "README.md").write_text(
        "# PostgreSQL\n\nFrom this directory run ./install.sh\n",
        encoding="utf-8",
    )
    (repository / "components" / "postgres" / "install.sh").write_text(
        "#!/bin/sh\necho deployed\n",
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
                        "location": str(repository),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    context = tmp_path / "context.yaml"
    context.write_text(
        yaml.safe_dump(
            {
                "system": "fixture",
                "deployment": {"profile": "local"},
                "provided_blocks": [],
            }
        ),
        encoding="utf-8",
    )
    return sources, context


def _action(action: str, **changes) -> dict[str, object]:
    value = {
        "action": action,
        "reason": "Grounded fixture decision.",
        "queries": [],
        "subquestions": [],
        "command": None,
        "working_directory": None,
        "success_check": None,
        "evidence_ids": [],
        "required_inputs": [],
    }
    value.update(changes)
    return value


def _run_recursive(
    tmp_path: Path,
    monkeypatch,
    responses: list[dict[str, object]],
    *,
    extra_args: list[str] | None = None,
    payloads: list[dict[str, object]] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    sources, context = _recursive_fixture(tmp_path)

    class Responses:
        def create(self, **payload):
            text = payload["input"][0]["content"][0]["text"]
            request = json.loads(text)
            if "source_corpus" in request:
                document = {
                    "system": "fixture",
                    "stages": [
                        {
                            "title": "Deploy fixture",
                            "goal": "How do I deploy the fixture system?",
                            "children": [],
                        }
                    ],
                }
                return [
                    {"type": "response.output_text.delta", "delta": json.dumps(document)},
                    {
                        "type": "response.completed",
                        "response": {
                            "usage": {
                                "input_tokens": 10,
                                "output_tokens": 5,
                                "total_tokens": 15,
                            }
                        },
                    },
                ]
            if payloads is not None:
                payloads.append(request)
            document = responses.pop(0)
            if document["action"] == "search":
                assert '"evidence": []' in text
            else:
                assert '"evidence": [' in text
            return [
                {"type": "response.output_text.delta", "delta": json.dumps(document)},
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    },
                },
            ]

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "pheragent.deployment.analysis_llm._openai_client",
        lambda **_kwargs: SimpleNamespace(responses=Responses()),
    )
    output = tmp_path / "recursive"
    arguments = [
            "research",
            "recursive-plan",
            "--sources",
            str(sources),
            "--context",
            str(context),
            "--output",
            str(output),
        ]
    arguments.extend(extra_args or [])
    exit_code = main(arguments)
    assert exit_code == 0
    run_dir = next((output / "runs").iterdir())
    return (
        yaml.safe_load((run_dir / "deployment-tree.yaml").read_text(encoding="utf-8")),
        json.loads((run_dir / "usage.json").read_text(encoding="utf-8")),
    )


def test_recursive_oracle_finds_and_grounds_unlinked_command(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    payloads: list[dict[str, object]] = []
    plan, usage = _run_recursive(
        tmp_path,
        monkeypatch,
        [
            _action(
                "search",
                queries=[
                    {
                        "text": "Fixture PostgreSQL deployment procedure",
                        "path_prefix": None,
                    }
                ],
            ),
            _action(
                "expand",
                subquestions=[
                    {
                        "title": "Install PostgreSQL",
                        "question": "How do I install the PostgreSQL component?",
                    }
                ],
                evidence_ids=["E1"],
            ),
            _action(
                "search",
                queries=[
                    {
                        "text": "PostgreSQL component install command",
                        "path_prefix": "components/postgres",
                    }
                ],
            ),
            _action(
                "executable",
                command="./install.sh",
                working_directory="components/postgres",
                success_check="The installer exits successfully.",
                evidence_ids=["E2"],
            ),
        ],
        payloads=payloads,
    )

    leaf = plan["roots"][0]["children"][0]["children"][0]
    assert leaf["state"] == "executable"
    assert leaf["command"] == "./install.sh"
    assert plan["planning_complete"] is True
    assert usage["requests"] == 5
    assert usage["oracle_searches"] == 2
    assert usage["grounded_commands"] == 1
    assert payloads[1]["evidence"][0]["id"] == "E1"
    run_dir = next((tmp_path / "recursive" / "runs").iterdir())
    trace = json.loads((run_dir / "trace.json").read_text(encoding="utf-8"))
    report = (run_dir / "analysis-trace.md").read_text(encoding="utf-8")
    assert trace[0]["oracle_results"][0]["source"].startswith("fixture:")
    assert trace[1]["subquestions"][0]["title"] == "Install PostgreSQL"
    assert "Oracle query: Fixture PostgreSQL deployment procedure" in report
    assert "Substep: Install PostgreSQL" in report
    output = capsys.readouterr()
    assert "planning complete: true" in output.out
    assert "oracle query: Fixture PostgreSQL deployment procedure" in output.err


def test_recursive_oracle_uses_ancestry_and_verified_runtime_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root_evidence_id = "fixture:README.md:1:3:documentation_section"
    component_evidence_id = "fixture:components/postgres/README.md:1:3:documentation_section"
    runtime_id = "runtime:kubernetes:helm_releases:postgres/postgres"
    runtime = tmp_path / "runtime.yaml"
    write_yaml(
        runtime,
        {
            "captured_at": "2026-09-27T10:00:00+00:00",
            "aws": {"available": False},
            "host": {"available": True, "system": "Linux"},
            "kubernetes": {
                "available": True,
                "context": "fixture",
                "helm_releases": [
                    {
                        "kind": "helm_release",
                        "name": "postgres",
                        "namespace": "postgres",
                        "state": "deployed",
                    }
                ],
            },
            "probes": [],
            "warnings": [],
        },
    )
    payloads: list[dict[str, object]] = []
    plan, usage = _run_recursive(
        tmp_path,
        monkeypatch,
        [
            _action("search", queries=[{"text": "Fixture deployment", "path_prefix": None}]),
            _action(
                "expand",
                subquestions=[
                    {
                        "title": "Install PostgreSQL",
                        "question": "How do I install the PostgreSQL component?",
                    }
                ],
                evidence_ids=[root_evidence_id],
            ),
            _action(
                "search",
                queries=[{"text": "PostgreSQL deployment", "path_prefix": None}],
            ),
            _action(
                "satisfied",
                success_check="The PostgreSQL Helm release is deployed.",
                evidence_ids=[component_evidence_id],
                runtime_evidence_ids=[runtime_id],
            ),
        ],
        extra_args=["--runtime-context", str(runtime)],
        payloads=payloads,
    )

    leaf = plan["roots"][0]["children"][0]["children"][0]
    assert leaf["state"] == "satisfied"
    assert leaf["runtime_evidence_ids"] == [runtime_id]
    assert plan["deployment_ready"] is True
    assert usage["satisfied_leaves"] == 1
    assert payloads[-1]["tree_context"]["ancestors"][0]["id"] == "H1"
    runtime_releases = payloads[-1]["runtime_context"]["kubernetes"]["helm_releases"]
    assert runtime_releases[0]["evidence_id"] == runtime_id


def test_recursive_plan_resumes_only_human_approved_external_source(tmp_path: Path) -> None:
    location = "https://example.invalid/deployment.git"
    request = ExternalSourceRequest(
        suggested_id="deployment",
        location=location,
        revision="0123456789abcdef0123456789abcdef01234567",
        reason="The repository README delegates installation to this source.",
    )
    tree = tmp_path / "tree.yaml"
    write_yaml(
        tree,
        RecursivePlan(
            system="fixture",
            deployment={"profile": "local"},
            planning_complete=True,
            roots=[
                PlanNode(
                    id="H1",
                    title="Deploy fixture",
                    goal="How do I deploy fixture?",
                    depth=0,
                    state=StepState.EXTERNAL_SOURCE_REQUIRED,
                    external_source=request,
                )
            ],
        ),
    )
    context = DeploymentContext.model_validate(
        {"system": "fixture", "deployment": {"profile": "local"}}
    )

    blocked = _load_plan(context, tree, approved_locations=set())
    resumed = _load_plan(context, tree, approved_locations={location})

    assert blocked.roots[0].state == StepState.EXTERNAL_SOURCE_REQUIRED
    assert resumed.roots[0].state == StepState.PENDING
    assert resumed.planning_complete is False


def test_recursive_oracle_rejects_command_missing_from_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence_id = "fixture:components/postgres/README.md:1:3:documentation_section"
    plan, usage = _run_recursive(
        tmp_path,
        monkeypatch,
        [
            _action(
                "search",
                queries=[{"text": "PostgreSQL component install command", "path_prefix": None}],
            ),
            _action(
                "executable",
                command="./missing.sh",
                working_directory="components/postgres",
                success_check="The installer exits successfully.",
                evidence_ids=[evidence_id],
            ),
        ],
    )

    stage = plan["roots"][0]["children"][0]
    assert stage["state"] == "unresolved"
    assert stage["issue"] == "rejected command: not present in cited evidence"
    assert usage["rejected_commands"] == 1


def test_recursive_oracle_stops_after_one_grounded_action(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan, _usage = _run_recursive(
        tmp_path,
        monkeypatch,
        [
            _action("search", queries=[{"text": "fixture deployment", "path_prefix": None}]),
            _action(
                "expand",
                subquestions=[
                    {"title": "Install database", "question": "How is the database installed?"},
                    {"title": "Install API", "question": "How is the API installed?"},
                ],
                evidence_ids=["E1"],
            ),
            _action(
                "search",
                queries=[
                    {
                        "text": "PostgreSQL component install command",
                        "path_prefix": "components/postgres",
                    }
                ],
            ),
            _action(
                "executable",
                command="./install.sh",
                working_directory="components/postgres",
                success_check="The installer exits successfully.",
                evidence_ids=["E2"],
            ),
        ],
        extra_args=["--action-budget", "1"],
    )

    children = plan["roots"][0]["children"][0]["children"]
    assert [child["state"] for child in children] == ["executable", "pending"]
    assert plan["planning_complete"] is False


def test_step_action_keeps_secret_values_out_of_contract() -> None:
    action = StepAction(
        action=ActionKind.HUMAN_INPUT,
        reason="A password must be configured outside the model.",
        evidence_ids=["fixture:README.md:1:3:documentation_section"],
        required_inputs=[
            RequiredInput(
                name="postgres_admin_password",
                reason="Required by the documented installation.",
                sensitive=True,
            )
        ],
    )

    _validate_action(action, has_evidence=True)
    assert "value" not in RequiredInput.model_fields


def test_recursive_plan_schema_preserves_action_fields() -> None:
    schema = strict_response_format(StepAction, name="step_action")["schema"]

    assert "action" in schema["properties"]
    assert "queries" in schema["properties"]
    assert set(schema["required"]) == set(schema["properties"])


def test_recursive_plan_follows_restructured_text_toctree() -> None:
    links = _local_link_targets(
        """.. toctree::
   :hidden:

   quick-start
   Install from source <installing-from-source>
"""
    )
    assert links == ["quick-start", "installing-from-source"]


def test_recursive_plan_detects_documentation_placeholders() -> None:
    assert _PLACEHOLDER.search("./install.sh [kubeconfig]")
    assert _PLACEHOLDER.search("helm install <release>")
    assert not _PLACEHOLDER.search("test [ -f values.yaml ]")


def test_recursive_tree_compiles_ordered_product_artifacts() -> None:
    source = {
        "repo_id": "fixture",
        "path": "deploy/install.sh",
        "start_line": 1,
        "end_line": 3,
    }
    plan = RecursivePlan(
        system="fixture",
        deployment={"profile": "local"},
        roots=[
            PlanNode(
                id="H1",
                title="Deploy fixture",
                goal="Deploy fixture",
                depth=0,
                state=StepState.EXPANDED,
                children=[
                    PlanNode(
                        id="H1.1",
                        title="Install database",
                        goal="Install database",
                        depth=1,
                        state=StepState.EXECUTABLE,
                        command="./install.sh",
                        working_directory="deploy",
                        operation_source_ref=source,
                        success_check="Database is ready.",
                    ),
                    PlanNode(
                        id="H1.2",
                        title="Configure credentials",
                        goal="Configure credentials",
                        depth=1,
                        state=StepState.HUMAN_REQUIRED,
                        issue="A human must provide credentials.",
                    ),
                    PlanNode(
                        id="H1.3",
                        title="Install application",
                        goal="Install application",
                        depth=1,
                        state=StepState.EXECUTABLE,
                        command="./install.sh",
                        working_directory="deploy",
                        operation_source_ref=source,
                        success_check="Application is ready.",
                    ),
                ],
            )
        ],
    )
    context = DeploymentContext.model_validate(
        {"system": "fixture", "deployment": {"profile": "local"}}
    )

    blocks, workflow = compile_recursive_plan(plan, context)

    assert [step.command for step in workflow.steps] == ["./install.sh", "./install.sh"]
    assert workflow.steps[0].status == "ready"
    assert workflow.steps[1].status == "blocked"
    assert workflow.steps[1].after == ["S001"]
    assert len(blocks.blocks) == 3
    assert workflow.ready_for_execution is False
