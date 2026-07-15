from __future__ import annotations

import shlex
from collections.abc import Iterable
from typing import Protocol

from .models import CommandBlock, RepoContext, SetupLocation
from .utils import shell_script, slugify


class BlockPlanner(Protocol):
    def plan(self, context: RepoContext) -> list[CommandBlock]:
        ...


class RuleBasedBlockPlanner:
    """Deterministic bootstrap planner.

    This is intentionally conservative: it emits install-oriented command
    blocks from repo manifests, and leaves richer synthesis to future LLM
    planners that implement the same interface.
    """

    def plan(self, context: RepoContext) -> list[CommandBlock]:
        blocks: list[CommandBlock] = [
            CommandBlock(
                id="00-preflight",
                order=0,
                title="Preflight",
                goal="Capture basic OS, toolchain, and repository context.",
                script=shell_script(_preflight_script()),
                validation_command="test -d .",
            )
        ]

        if "java" in context.languages:
            _note_secondary_java_locations(context)

        languages = [language for language in context.languages if _language_supported(language)]
        if not languages:
            blocks.append(
                CommandBlock(
                    id="10-generic-inspect",
                    order=10,
                    title="Generic Inspect",
                    goal="Provide a minimal generic repository inspection block.",
                    script=shell_script(
                        """
echo "[pheragent] no known package manager detected"
find . -maxdepth 2 -type f | sort | sed -n '1,120p'
"""
                    ),
                    validation_command="test -d .",
                )
            )
            return blocks

        if len(languages) == 1:
            language = languages[0]
            blocks.append(_runtime_block(language, 20, context))
            blocks.append(_dependency_block(language, 30, context))
            blocks.append(_test_tooling_block(50, context))
            return blocks

        if len(languages) == 2:
            blocks.append(_system_packages_block(10, context))
            for offset, language in enumerate(languages):
                blocks.append(_runtime_block(language, 20 + offset, context))
            for offset, language in enumerate(languages):
                blocks.append(_dependency_block(language, 30 + offset, context))
            blocks.append(_test_tooling_block(50, context))
            return blocks

        blocks.append(_system_packages_block(10, context))
        blocks.append(_combined_runtime_block(20, languages, context))
        blocks.append(_combined_dependency_block(30, languages, context))
        if _needs_native_build_config(context):
            blocks.append(_native_build_config_block(40))
        blocks.append(_test_tooling_block(50, context))
        return blocks


def _ordered_id(order: int, title: str) -> str:
    return f"{order:02d}-{slugify(title)}"


def _language_supported(language: str) -> bool:
    return language in {"python", "node", "go", "rust", "java"}


def _runtime_block(language: str, order: int, context: RepoContext) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, f"{language}-runtime"),
        order=order,
        title=f"{language.title()} Runtime",
        goal=f"Verify the {language} runtime and core command-line tools.",
        script=shell_script(_runtime_script(language, context)),
        validation_command=_runtime_validation_command(language, context),
    )


def _dependency_block(language: str, order: int, context: RepoContext) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, f"{language}-deps"),
        order=order,
        title=f"{language.title()} Dependencies",
        goal=f"Install {language} project dependencies from detected manifests.",
        script=shell_script(_dependency_script(language, context)),
        validation_command=_dependency_validation_command(language, context),
    )


def _combined_runtime_block(
    order: int,
    languages: list[str],
    context: RepoContext,
) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, "runtime-toolchain"),
        order=order,
        title="Runtime Toolchain",
        goal="Verify all detected language runtimes and core command-line tools.",
        script=shell_script(
            _join_scripts(_runtime_script(language, context) for language in languages)
        ),
        validation_command=_join_validation(
            _runtime_validation_command(language, context) for language in languages
        ),
    )


def _combined_dependency_block(
    order: int,
    languages: list[str],
    context: RepoContext,
) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, "project-dependencies"),
        order=order,
        title="Project Dependencies",
        goal="Install project dependencies across detected ecosystems.",
        script=shell_script(
            _join_scripts(_dependency_script(language, context) for language in languages)
        ),
        validation_command=_join_validation(
            _dependency_validation_command(language) for language in languages
        ),
    )


def _system_packages_block(order: int, context: RepoContext) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, "system-packages"),
        order=order,
        title="System Packages",
        goal="Install common OS packages shared by the detected ecosystems.",
        script=shell_script(_system_packages_script(context)),
        validation_command="command -v git >/dev/null 2>&1 || true",
    )


def _native_build_config_block(order: int) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, "native-build-config"),
        order=order,
        title="Native Build Config",
        goal="Check native build tools used by compiled dependencies.",
        script=shell_script(_native_build_config_script()),
        validation_command=(
            "command -v cc >/dev/null 2>&1 || "
            "command -v gcc >/dev/null 2>&1 || true"
        ),
    )


def _test_tooling_block(order: int, context: RepoContext) -> CommandBlock:
    return CommandBlock(
        id=_ordered_id(order, "test-tooling"),
        order=order,
        title="Test Tooling",
        goal="Prepare lightweight test and validation tooling without running full suites.",
        script=shell_script(_test_tooling_script(context)),
        validation_command=_test_tooling_validation_command(context),
    )


def _runtime_script(language: str, context: RepoContext) -> str:
    if language == "python":
        return _python_runtime_script()
    if language == "node":
        return _node_runtime_script()
    if language == "go":
        return _go_runtime_script()
    if language == "rust":
        return _rust_runtime_script()
    if language == "java":
        return _java_runtime_script(context)
    raise ValueError(f"unsupported language: {language}")


def _dependency_script(language: str, context: RepoContext) -> str:
    if language == "python":
        return _python_script(use_uv="uv" in context.package_managers)
    if language == "node":
        return _node_script(context.package_managers)
    if language == "go":
        return _go_script()
    if language == "rust":
        return _rust_dependency_script()
    if language == "java":
        return _java_script(context)
    raise ValueError(f"unsupported language: {language}")


def _runtime_validation_command(language: str, context: RepoContext) -> str:
    del context
    if language == "python":
        return _python_runtime_validation_command()
    if language == "node":
        return _node_runtime_validation_command()
    if language == "go":
        return 'PATH="/usr/local/go/bin:$PATH" go version'
    if language == "rust":
        return "cargo --version"
    if language == "java":
        return "java -version || mvn -version || gradle -version"
    raise ValueError(f"unsupported language: {language}")


def _dependency_validation_command(language: str, context: RepoContext) -> str:
    if language == "python":
        return _python_dependency_validation_command()
    if language == "node":
        return _node_dependency_validation_command()
    if language == "go":
        return _safe_go_validation_command()
    if language == "rust":
        return "cargo fetch --locked || cargo fetch"
    if language == "java":
        return _java_build_validation_command(context)
    raise ValueError(f"unsupported language: {language}")


def _first_present(values: list[str], *, prefix: str, fallback: str) -> str:
    for value in values:
        if value.startswith(prefix):
            return value
    return fallback


def _join_scripts(parts: Iterable[str]) -> str:
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def _join_validation(commands: Iterable[str]) -> str:
    return " && ".join(f"( {command} )" for command in commands if command)


def _needs_native_build_config(context: RepoContext) -> bool:
    native_languages = {"go", "rust", "java"}
    return any(language in native_languages for language in context.languages)


def _java_locations(context: RepoContext) -> list[SetupLocation]:
    return [location for location in context.setup_locations if location.language == "java"]


def _select_reactor_root(java_locations: list[SetupLocation]) -> SetupLocation:
    def descendant_count(candidate: SetupLocation) -> int:
        if candidate.path == ".":
            return len(java_locations) - 1
        prefix = candidate.path + "/"
        return sum(
            1
            for other in java_locations
            if other is not candidate and other.path.startswith(prefix)
        )

    def sort_key(candidate: SetupLocation) -> tuple[int, int, str]:
        segments = 0 if candidate.path == "." else candidate.path.count("/") + 1
        return (-descendant_count(candidate), segments, candidate.path)

    return sorted(java_locations, key=sort_key)[0]


def _primary_java_location(context: RepoContext) -> SetupLocation | None:
    java_locations = _java_locations(context)
    if not java_locations:
        return None
    reactor_root = (
        java_locations[0] if len(java_locations) == 1 else _select_reactor_root(java_locations)
    )
    return None if reactor_root.path == "." else reactor_root


def _note_secondary_java_locations(context: RepoContext) -> None:
    primary = _primary_java_location(context)
    if primary is None:
        return
    prefix = primary.path + "/"
    secondary = [
        location.path
        for location in _java_locations(context)
        if location.path != primary.path and not location.path.startswith(prefix)
    ]
    if secondary:
        context.notes.append(
            "Additional Java module(s) outside the primary build reactor "
            f"({primary.path}): {', '.join(secondary)}. Not included in the "
            "generated dependency/test-tooling block; build separately if needed."
        )


def _cd_prefix(location: SetupLocation | None) -> str:
    if location is None or location.path == ".":
        return ""
    return f"cd {shlex.quote(location.path)}\n"


def _java_build_validation_command(context: RepoContext) -> str:
    location = _primary_java_location(context)
    target = shlex.quote(location.path if location is not None else ".")
    return (
        f"(cd {target} && mvn -q -DskipTests test) || "
        f"(cd {target} && ./gradlew testClasses --no-daemon) || "
        f"(cd {target} && gradle testClasses)"
    )


def _safe_python_validation_command(context: RepoContext, fallback: str) -> str:
    command = _first_present(context.test_commands, prefix="python", fallback=fallback)
    normalized = " ".join(command.split()).lower()
    if "pytest" in normalized:
        return _pytest_collect_validation_command()
    return command


def _pytest_collect_validation_command() -> str:
    return (
        "test -x .venv/bin/python; "
        "./.venv/bin/python -m pytest --collect-only -q; "
        "status=$?; "
        'if [ "$status" -eq 5 ]; then '
        'echo "[pheragent] no pytest tests collected"; exit 0; '
        "fi; "
        'exit "$status"'
    )


def _safe_go_validation_command() -> str:
    return (
        'PATH="/usr/local/go/bin:$PATH" '
        'GOFLAGS="${GOFLAGS:-} -buildvcs=false" '
        "go list -mod=mod ./... >/dev/null"
    )


def _python_runtime_validation_command() -> str:
    return (
        "test -x .venv/bin/python && "
        './.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)" '
        "&& ./.venv/bin/python -m pip --version"
    )


def _python_dependency_validation_command() -> str:
    return (
        "test -x .venv/bin/python && "
        './.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)"'
    )


def _node_runtime_validation_command() -> str:
    return (
        "test -x .pheragent-tools/bin/node && test -x .pheragent-tools/bin/npm && "
        ".pheragent-tools/bin/node --version && .pheragent-tools/bin/npm --version"
    )


def _node_dependency_validation_command() -> str:
    return (
        "test -x .pheragent-tools/bin/node && test -x .pheragent-tools/bin/npm && "
        "( test -d node_modules || test ! -f package.json )"
    )


def _preflight_script() -> str:
    return """
echo "[pheragent] preflight"
pwd
uname -a || true
cat /etc/os-release 2>/dev/null || true
find . -maxdepth 2 -type f | sort | sed -n '1,120p'
command -v sh >/dev/null 2>&1 && sh --version 2>/dev/null || true
command -v python3 >/dev/null 2>&1 && python3 --version || true
command -v python >/dev/null 2>&1 && python --version || true
command -v node >/dev/null 2>&1 && node --version || true
command -v npm >/dev/null 2>&1 && npm --version || true
command -v go >/dev/null 2>&1 && go version || true
command -v cargo >/dev/null 2>&1 && cargo --version || true
"""


def _system_packages_script(context: RepoContext) -> str:
    extra_packages = ""
    if _needs_native_build_config(context):
        extra_packages = " build-essential pkg-config"
    return f"""
echo "[pheragent] preparing system packages"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates curl git{extra_packages}
elif command -v apk >/dev/null 2>&1; then
  apk add --no-cache ca-certificates curl git make gcc g++ pkgconf
elif command -v dnf >/dev/null 2>&1; then
  dnf install -y ca-certificates curl git gcc gcc-c++ make pkgconf-pkg-config
else
  echo "[pheragent] no supported OS package manager detected; skipping system packages"
fi
"""


def _python_runtime_script() -> str:
    return """
echo "[pheragent] ensuring python runtime"
ensure_python_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache python3 py3-pip py3-virtualenv
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y python3 python3-pip python3-virtualenv
  elif command -v yum >/dev/null 2>&1; then
    yum install -y python3 python3-pip python3-virtualenv
  else
    echo "python runtime missing components and no supported package manager is available" >&2
    exit 127
  fi
}

if ! command -v python3 >/dev/null 2>&1; then
  ensure_python_packages
fi

if command -v python3 >/dev/null 2>&1; then
  SYSTEM_PYTHON=python3
else
  echo "python executable not found" >&2
  exit 127
fi

if ! "$SYSTEM_PYTHON" -m pip --version >/dev/null 2>&1 || \
   ! "$SYSTEM_PYTHON" -m venv -h >/dev/null 2>&1; then
  ensure_python_packages
  SYSTEM_PYTHON=python3
fi

if [ ! -x .venv/bin/python ]; then
  rm -rf .venv
  "$SYSTEM_PYTHON" -m venv .venv
fi
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
VENV_PYTHON="$(pwd)/.venv/bin/python"
VENV_PIP="$(pwd)/.venv/bin/pip"
ln -sf "$VENV_PYTHON" /usr/local/bin/python || true
ln -sf "$VENV_PYTHON" /usr/local/bin/python3 || true
if [ -x .venv/bin/pip ]; then
  ln -sf "$VENV_PIP" /usr/local/bin/pip || true
  ln -sf "$VENV_PIP" /usr/local/bin/pip3 || true
fi
./.venv/bin/python -c "import sys; print(sys.executable); print(sys.version)"
./.venv/bin/python -m pip --version
"""


def _node_runtime_script() -> str:
    return """
echo "[pheragent] ensuring node runtime"
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends nodejs npm
  elif command -v apk >/dev/null 2>&1; then
    apk add --no-cache nodejs npm
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y nodejs npm
  elif command -v yum >/dev/null 2>&1; then
    yum install -y nodejs npm
  else
    echo "node/npm not found and no supported package manager is available" >&2
    exit 127
  fi
fi
mkdir -p .pheragent-tools/bin
resolve_runtime_bin() {
  name="$1"
  fixed="$(pwd)/.pheragent-tools/bin/$name"
  for candidate in \
    "/usr/bin/$name" \
    "/usr/local/bin/$name" \
    "$(command -v "$name" 2>/dev/null || true)"; do
    if [ -z "$candidate" ] || [ ! -x "$candidate" ]; then
      continue
    fi
    real="$(readlink -f "$candidate" 2>/dev/null || true)"
    if [ -z "$real" ]; then
      real="$candidate"
    fi
    case "$real" in
      "$fixed"|"$fixed"/*) continue ;;
    esac
    if [ -x "$real" ] && "$real" --version >/dev/null 2>&1; then
      printf '%s\\n' "$real"
      return 0
    fi
  done
  return 1
}
NODE_BIN="$(resolve_runtime_bin node)"
NPM_BIN="$(resolve_runtime_bin npm)"
ln -sf "$NODE_BIN" .pheragent-tools/bin/node
ln -sf "$NPM_BIN" .pheragent-tools/bin/npm
ln -sf "$(pwd)/.pheragent-tools/bin/node" /usr/local/bin/node || true
ln -sf "$(pwd)/.pheragent-tools/bin/npm" /usr/local/bin/npm || true
.pheragent-tools/bin/node --version
.pheragent-tools/bin/npm --version
"""


def _go_runtime_script() -> str:
    return """
echo "[pheragent] checking go runtime"
export PATH="/usr/local/go/bin:$PATH"
go version
"""


def _rust_runtime_script() -> str:
    return """
echo "[pheragent] checking rust runtime"
cargo --version
rustc --version
"""


def _java_runtime_script(context: RepoContext) -> str:
    cd_prefix = _cd_prefix(_primary_java_location(context))
    return f"""
echo "[pheragent] checking java runtime"
{cd_prefix}java -version || true
mvn -version || true
if [ -x ./gradlew ]; then
  ./gradlew --version || true
elif command -v gradle >/dev/null 2>&1; then
  gradle -version || true
fi
"""


def _go_script() -> str:
    return """
echo "[pheragent] installing go module dependencies"
export PATH="/usr/local/go/bin:$PATH"
go version
go mod download
"""


def _python_script(*, use_uv: bool) -> str:
    uv_branch = ""
    if use_uv:
        uv_branch = """
if [ -f uv.lock ] && command -v uv >/dev/null 2>&1; then
  echo "[pheragent] uv.lock detected; running uv sync"
  uv sync --all-extras --dev || uv sync
  test -x .venv/bin/python
  exit 0
fi
"""
    return (
        """
echo "[pheragent] installing python dependencies"
if [ ! -x .venv/bin/python ]; then
  if command -v python3 >/dev/null 2>&1; then
    rm -rf .venv
    python3 -m venv .venv
  else
    echo ".venv/bin/python not found and python3 is unavailable" >&2
    exit 127
  fi
fi
"""
        + uv_branch
        + """
./.venv/bin/python -m pip --version >/dev/null 2>&1 || \
  ./.venv/bin/python -m ensurepip --upgrade || true
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
if [ -f requirements.txt ]; then
  ./.venv/bin/python -m pip install -r requirements.txt
fi
if [ -f pyproject.toml ] || [ -f setup.py ] || [ -f setup.cfg ]; then
  ./.venv/bin/python -m pip install -e '.[dev]' || ./.venv/bin/python -m pip install -e .
fi
"""
    )


def _node_script(package_managers: list[str]) -> str:
    prefers_pnpm = "pnpm" in package_managers
    prefers_yarn = "yarn" in package_managers
    return f"""
echo "[pheragent] installing node dependencies"
test -x .pheragent-tools/bin/node
test -x .pheragent-tools/bin/npm
.pheragent-tools/bin/node --version
.pheragent-tools/bin/npm --version
ensure_pnpm() {{
  if command -v pnpm >/dev/null 2>&1 && pnpm --version >/dev/null 2>&1; then
    return 0
  fi
  PNPM_PACKAGE=pnpm@9
  NODE_VERSION=$(.pheragent-tools/bin/node -p "process.versions.node" 2>/dev/null || echo 0.0.0)
  NODE_MAJOR=${{NODE_VERSION%%.*}}
  NODE_REST=${{NODE_VERSION#*.}}
  NODE_MINOR=${{NODE_REST%%.*}}
  if [ "$NODE_MAJOR" -gt 22 ] || \
     {{ [ "$NODE_MAJOR" -eq 22 ] && [ "$NODE_MINOR" -ge 13 ]; }}; then
    PNPM_PACKAGE=pnpm
  fi
  .pheragent-tools/bin/npm install -g "$PNPM_PACKAGE"
  pnpm --version
}}
if command -v corepack >/dev/null 2>&1; then
  corepack enable || true
fi
if [ -f pnpm-lock.yaml ] && {"true" if prefers_pnpm else "false"}; then
  ensure_pnpm
  pnpm install --frozen-lockfile || pnpm install
elif [ -f yarn.lock ] && {"true" if prefers_yarn else "false"}; then
  if command -v yarn >/dev/null 2>&1; then
    yarn install --frozen-lockfile || yarn install
  else
    .pheragent-tools/bin/npm install -g yarn
    yarn install --frozen-lockfile || yarn install
  fi
elif [ -f package-lock.json ]; then
  .pheragent-tools/bin/npm ci || .pheragent-tools/bin/npm install
else
  .pheragent-tools/bin/npm install
fi
"""


def _rust_dependency_script() -> str:
    return """
echo "[pheragent] fetching rust dependencies"
cargo --version
cargo fetch --locked || cargo fetch
"""


def _java_script(context: RepoContext) -> str:
    package_managers = context.package_managers
    has_maven = "maven" in package_managers
    has_gradle = "gradle" in package_managers
    gradle_enabled = "true" if has_gradle else "false"
    cd_prefix = _cd_prefix(_primary_java_location(context))
    return f"""
echo "[pheragent] warming java dependencies"
{cd_prefix}if [ -f pom.xml ] && {"true" if has_maven else "false"}; then
  mvn -q -DskipTests dependency:go-offline
fi
if {{ [ -f build.gradle ] || [ -f build.gradle.kts ]; }} && {gradle_enabled}; then
  if [ -x ./gradlew ]; then
    ./gradlew dependencies --no-daemon || true
  elif [ -f ./gradlew ]; then
    chmod +x ./gradlew
    ./gradlew dependencies --no-daemon || true
  else
    gradle dependencies || true
  fi
fi
"""


def _native_build_config_script() -> str:
    return """
echo "[pheragent] checking native build configuration"
command -v cc >/dev/null 2>&1 && cc --version || true
command -v gcc >/dev/null 2>&1 && gcc --version || true
command -v make >/dev/null 2>&1 && make --version || true
command -v pkg-config >/dev/null 2>&1 && pkg-config --version || true
"""


def _test_tooling_script(context: RepoContext) -> str:
    scripts: list[str] = []
    if "python" in context.languages:
        scripts.append(_python_test_tooling_script())
    if "node" in context.languages:
        scripts.append(_node_test_tooling_script())
    if "go" in context.languages:
        scripts.append(_go_test_tooling_script())
    if "rust" in context.languages:
        scripts.append(_rust_test_tooling_script())
    if "java" in context.languages:
        scripts.append(_java_test_tooling_script(context))
    return _join_scripts(scripts) or 'echo "[pheragent] no test tooling detected"'


def _test_tooling_validation_command(context: RepoContext) -> str:
    commands: list[str] = []
    if "python" in context.languages:
        commands.append(
            _safe_python_validation_command(context, _python_runtime_validation_command())
        )
    if "node" in context.languages:
        commands.append(_node_runtime_validation_command())
    if "go" in context.languages:
        commands.append(_safe_go_validation_command())
    if "rust" in context.languages:
        commands.append("cargo test --no-run || cargo check")
    if "java" in context.languages:
        commands.append(_java_build_validation_command(context))
    return _join_validation(commands) or "test -d ."


def _python_test_tooling_script() -> str:
    return """
echo "[pheragent] preparing python test tooling"
test -x .venv/bin/python
./.venv/bin/python -m pytest --version >/dev/null 2>&1 || ./.venv/bin/python -m pip install pytest
"""


def _node_test_tooling_script() -> str:
    return """
echo "[pheragent] preparing node test tooling"
test -x .pheragent-tools/bin/node
test -x .pheragent-tools/bin/npm
.pheragent-tools/bin/node --version
.pheragent-tools/bin/npm --version
if command -v pnpm >/dev/null 2>&1; then pnpm --version; fi
if command -v yarn >/dev/null 2>&1; then yarn --version; fi
"""


def _go_test_tooling_script() -> str:
    return """
echo "[pheragent] preparing go test tooling"
export PATH="/usr/local/go/bin:$PATH"
go test -run '^$' ./... || go list -mod=mod ./... >/dev/null
"""


def _rust_test_tooling_script() -> str:
    return """
echo "[pheragent] preparing rust test tooling"
cargo test --no-run || cargo check
"""


def _java_test_tooling_script(context: RepoContext) -> str:
    cd_prefix = _cd_prefix(_primary_java_location(context))
    return f"""
echo "[pheragent] preparing java test tooling"
{cd_prefix}if [ -f pom.xml ]; then
  mvn -q -DskipTests test || true
fi
if [ -f build.gradle ] || [ -f build.gradle.kts ]; then
  if [ -x ./gradlew ]; then
    ./gradlew testClasses --no-daemon || true
  elif [ -f ./gradlew ]; then
    chmod +x ./gradlew
    ./gradlew testClasses --no-daemon || true
  elif command -v gradle >/dev/null 2>&1; then
    gradle testClasses || true
  fi
fi
"""
