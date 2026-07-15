from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from .models import RepoContext, SetupLocation

_MAX_WALK_DEPTH = 5

_IGNORED_DIR_NAMES = {
    ".git",
    ".github",
    ".hg",
    ".svn",
    ".pheragent",
    "node_modules",
    "target",
    "build",
    "dist",
    "out",
    ".venv",
    "venv",
    "vendor",
    ".gradle",
    ".dart_tool",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
}


class RepoAnalyzer:
    def analyze(self, repo_path: Path) -> RepoContext:
        repo = repo_path.expanduser().resolve()
        if not repo.is_dir():
            raise ValueError(f"repo path is not a directory: {repo}")

        context = RepoContext(repo_path=repo)
        for directory in self._walk_directories(repo):
            for detector in (
                self._detect_python,
                self._detect_node,
                self._detect_go,
                self._detect_rust,
                self._detect_java,
            ):
                location = detector(repo, directory, context)
                if location is not None:
                    context.setup_locations.append(location)

        if not context.package_files:
            context.notes.append("No common dependency manifest was found.")
        return context

    def _walk_directories(self, repo: Path) -> list[Path]:
        directories: list[Path] = []
        for current_root, dir_names, _file_names in os.walk(repo):
            current = Path(current_root)
            depth = len(current.relative_to(repo).parts)
            directories.append(current)
            if depth >= _MAX_WALK_DEPTH:
                dir_names[:] = []
                continue
            dir_names[:] = sorted(
                name for name in dir_names if name not in _IGNORED_DIR_NAMES
            )
        return directories

    def _add_unique(self, values: list[str], value: str) -> None:
        if value not in values:
            values.append(value)

    def _add_test_commands(self, context: RepoContext, commands: list[str]) -> None:
        for command in commands:
            if command not in context.test_commands:
                context.test_commands.append(command)

    def _record_manifests(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
        manifests: list[str],
    ) -> None:
        for filename in manifests:
            context.package_files.append(_relative_display_path(repo, directory, filename))

    def _detect_python(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
    ) -> SetupLocation | None:
        manifests = [
            filename
            for filename in (
                "pyproject.toml",
                "requirements.txt",
                "setup.py",
                "setup.cfg",
                "uv.lock",
                "poetry.lock",
                "Pipfile",
                "tox.ini",
            )
            if (directory / filename).exists()
        ]
        if not manifests:
            return None

        package_managers: list[str] = []
        if "uv.lock" in manifests:
            package_managers.append("uv")
        if "poetry.lock" in manifests:
            package_managers.append("poetry")
        package_managers.append("pip")

        has_tests_dir = (directory / "tests").is_dir()
        pyproject = directory / "pyproject.toml"
        has_pytest_config = (
            self._read_pyproject(pyproject, context) if pyproject.exists() else False
        )
        test_commands = ["python -m pytest -q"] if (has_tests_dir or has_pytest_config) else []

        self._record_manifests(repo, directory, context, manifests)
        self._add_unique(context.languages, "python")
        for manager in package_managers:
            self._add_unique(context.package_managers, manager)
        self._add_test_commands(context, test_commands)

        return SetupLocation(
            path=_location_path(repo, directory),
            language="python",
            package_files=manifests,
            package_managers=package_managers,
            test_commands=test_commands,
        )

    def _read_pyproject(self, path: Path, context: RepoContext) -> bool:
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            context.notes.append(f"Could not parse pyproject.toml: {exc}")
            return False
        tool = payload.get("tool", {})
        has_pytest_config = isinstance(tool, dict) and "pytest" in tool
        project = payload.get("project", {})
        if isinstance(project, dict) and project.get("scripts"):
            context.notes.append("pyproject.toml defines project scripts.")
        return has_pytest_config

    def _detect_node(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
    ) -> SetupLocation | None:
        package_json = directory / "package.json"
        if not package_json.exists():
            return None

        manifests = ["package.json"]
        package_managers: list[str] = []
        for filename, manager in (
            ("pnpm-lock.yaml", "pnpm"),
            ("yarn.lock", "yarn"),
            ("package-lock.json", "npm"),
        ):
            if (directory / filename).exists():
                manifests.append(filename)
                package_managers.append(manager)
        if not any(manager in package_managers for manager in ("pnpm", "yarn", "npm")):
            package_managers.append("npm")

        build_commands: list[str] = []
        test_commands: list[str] = []
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            context.notes.append(f"Could not parse package.json: {exc}")
            payload = {}
        scripts = payload.get("scripts", {})
        if isinstance(scripts, dict):
            if "build" in scripts:
                build_commands.append("npm run build")
            if "test" in scripts:
                test_commands.append("npm test")

        self._record_manifests(repo, directory, context, manifests)
        self._add_unique(context.languages, "node")
        for manager in package_managers:
            self._add_unique(context.package_managers, manager)
        for command in build_commands:
            if command not in context.build_commands:
                context.build_commands.append(command)
        self._add_test_commands(context, test_commands)

        return SetupLocation(
            path=_location_path(repo, directory),
            language="node",
            package_files=manifests,
            package_managers=package_managers,
            test_commands=test_commands,
        )

    def _detect_go(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
    ) -> SetupLocation | None:
        if not (directory / "go.mod").exists():
            return None
        test_commands = ["go test ./..."]
        self._record_manifests(repo, directory, context, ["go.mod"])
        self._add_unique(context.languages, "go")
        self._add_unique(context.package_managers, "go")
        self._add_test_commands(context, test_commands)
        return SetupLocation(
            path=_location_path(repo, directory),
            language="go",
            package_files=["go.mod"],
            package_managers=["go"],
            test_commands=test_commands,
        )

    def _detect_rust(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
    ) -> SetupLocation | None:
        if not (directory / "Cargo.toml").exists():
            return None
        test_commands = ["cargo test"]
        self._record_manifests(repo, directory, context, ["Cargo.toml"])
        self._add_unique(context.languages, "rust")
        self._add_unique(context.package_managers, "cargo")
        self._add_test_commands(context, test_commands)
        return SetupLocation(
            path=_location_path(repo, directory),
            language="rust",
            package_files=["Cargo.toml"],
            package_managers=["cargo"],
            test_commands=test_commands,
        )

    def _detect_java(
        self,
        repo: Path,
        directory: Path,
        context: RepoContext,
    ) -> SetupLocation | None:
        manifests = [
            filename
            for filename in ("pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle")
            if (directory / filename).exists()
        ]
        if not manifests:
            return None

        package_managers: list[str] = []
        test_commands: list[str] = []
        if "pom.xml" in manifests:
            package_managers.append("maven")
            test_commands.append("mvn test")
        if "build.gradle" in manifests or "build.gradle.kts" in manifests:
            package_managers.append("gradle")
            test_commands.append("./gradlew test || gradle test")

        self._record_manifests(repo, directory, context, manifests)
        self._add_unique(context.languages, "java")
        for manager in package_managers:
            self._add_unique(context.package_managers, manager)
        self._add_test_commands(context, test_commands)

        return SetupLocation(
            path=_location_path(repo, directory),
            language="java",
            package_files=manifests,
            package_managers=package_managers,
            test_commands=test_commands,
        )


def _location_path(repo: Path, directory: Path) -> str:
    relative = directory.relative_to(repo)
    text = relative.as_posix()
    return text if text not in ("", ".") else "."


def _relative_display_path(repo: Path, directory: Path, filename: str) -> str:
    location = _location_path(repo, directory)
    return filename if location == "." else f"{location}/{filename}"
