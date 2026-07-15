from __future__ import annotations

from pathlib import Path

from pheragent.analyzer import RepoAnalyzer
from pheragent.planner import RuleBasedBlockPlanner


def test_analyzer_detects_python_uv_repo(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "demo"

[tool.pytest.ini_options]
testpaths = ["tests"]
""",
        encoding="utf-8",
    )
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "tests").mkdir()

    context = RepoAnalyzer().analyze(tmp_path)

    assert context.languages == ["python"]
    assert "uv" in context.package_managers
    assert "python -m pytest -q" in context.test_commands


def test_rule_based_planner_writes_preflight_and_python_blocks(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)

    blocks = RuleBasedBlockPlanner().plan(context)

    assert [block.id for block in blocks] == [
        "00-preflight",
        "20-python-runtime",
        "30-python-deps",
        "50-test-tooling",
    ]
    deps_block = next(block for block in blocks if block.id == "30-python-deps")
    test_block = next(block for block in blocks if block.id == "50-test-tooling")
    assert "pip install" in deps_block.script
    assert "pip install pytest" not in deps_block.script
    assert "pip install pytest" in test_block.script
    assert deps_block.script.startswith("#!/bin/sh")


def test_rule_based_planner_collects_pytest_without_running_full_suite(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    context = RepoAnalyzer().analyze(tmp_path)

    blocks = RuleBasedBlockPlanner().plan(context)
    test_block = next(block for block in blocks if block.id == "50-test-tooling")

    assert test_block.validation_command is not None
    assert ".venv/bin/python" in test_block.validation_command
    assert "--collect-only" in test_block.validation_command
    assert "no pytest tests collected" in test_block.validation_command


def test_rule_based_planner_validates_go_without_vcs_stamping(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.24\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)

    blocks = RuleBasedBlockPlanner().plan(context)
    go_block = next(block for block in blocks if block.id.endswith("go-deps"))

    assert "go mod download" in go_block.script
    assert go_block.validation_command is not None
    assert "-buildvcs=false" in go_block.validation_command
    assert "go list -mod=mod ./..." in go_block.validation_command


def test_rule_based_planner_installs_node_compatible_pnpm(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text('{"scripts":{"build":"vite build"}}', encoding="utf-8")
    (tmp_path / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)

    blocks = RuleBasedBlockPlanner().plan(context)
    node_block = next(block for block in blocks if block.id.endswith("node-deps"))

    assert "ensure_pnpm" in node_block.script
    assert "PNPM_PACKAGE=pnpm@9" in node_block.script
    assert '.pheragent-tools/bin/npm install -g "$PNPM_PACKAGE"' in node_block.script


def test_analyzer_detects_nested_multi_module_java_repo(tmp_path: Path) -> None:
    reactor = tmp_path / "pre-registration"
    reactor.mkdir()
    (reactor / "pom.xml").write_text("<project/>", encoding="utf-8")
    for service in ("service-a", "service-b"):
        module_dir = reactor / service
        module_dir.mkdir()
        (module_dir / "pom.xml").write_text("<project/>", encoding="utf-8")
    api_test = tmp_path / "api-test"
    api_test.mkdir()
    (api_test / "pom.xml").write_text("<project/>", encoding="utf-8")

    context = RepoAnalyzer().analyze(tmp_path)

    assert context.languages == ["java"]
    assert "maven" in context.package_managers
    java_location_paths = {
        location.path for location in context.setup_locations if location.language == "java"
    }
    assert java_location_paths == {
        "pre-registration",
        "pre-registration/service-a",
        "pre-registration/service-b",
        "api-test",
    }

    blocks = RuleBasedBlockPlanner().plan(context)

    assert any("api-test" in note for note in context.notes)
    assert [block.id for block in blocks] == [
        "00-preflight",
        "20-java-runtime",
        "30-java-deps",
        "50-test-tooling",
    ]
    deps_block = next(block for block in blocks if block.id == "30-java-deps")
    assert "cd pre-registration" in deps_block.script
    assert deps_block.validation_command is not None
    assert "cd pre-registration" in deps_block.validation_command
    assert not deps_block.validation_command.rstrip().endswith("|| true")
    test_block = next(block for block in blocks if block.id == "50-test-tooling")
    assert "cd pre-registration" in test_block.script
    assert test_block.validation_command is not None
    assert not test_block.validation_command.rstrip().endswith("|| true")


def test_rule_based_planner_splits_two_language_repo_but_caps_blocks(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (tmp_path / "package.json").write_text('{"scripts":{"test":"vitest"}}', encoding="utf-8")
    context = RepoAnalyzer().analyze(tmp_path)

    blocks = RuleBasedBlockPlanner().plan(context)

    assert [block.id for block in blocks] == [
        "00-preflight",
        "10-system-packages",
        "20-python-runtime",
        "21-node-runtime",
        "30-python-deps",
        "31-node-deps",
        "50-test-tooling",
    ]
    assert len(blocks) == 7
    python_runtime = next(block for block in blocks if block.id == "20-python-runtime")
    assert "ensuring python runtime" in python_runtime.script
    assert "python3 python3-pip python3-venv" in python_runtime.script
    assert "SYSTEM_PYTHON=python3" in python_runtime.script
    assert '"$SYSTEM_PYTHON" -m venv .venv' in python_runtime.script
    assert "./.venv/bin/python -m pip --version" in python_runtime.script
    assert python_runtime.validation_command is not None
    assert ".venv/bin/python" in python_runtime.validation_command
    node_runtime = next(block for block in blocks if block.id == "21-node-runtime")
    assert "ensuring node runtime" in node_runtime.script
    assert "apt-get install -y --no-install-recommends nodejs npm" in node_runtime.script
    assert "resolve_runtime_bin" in node_runtime.script
    assert 'ln -sf "$NODE_BIN" .pheragent-tools/bin/node' in node_runtime.script
    assert ".pheragent-tools/bin/node --version" in node_runtime.script
    assert node_runtime.validation_command == (
        "test -x .pheragent-tools/bin/node && test -x .pheragent-tools/bin/npm && "
        ".pheragent-tools/bin/node --version && .pheragent-tools/bin/npm --version"
    )
