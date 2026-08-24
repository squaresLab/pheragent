# pheragent

`pheragent` contains two command-driven workflows built on repository analysis:

- **HerAgent-Deploy** analyzes deployment repositories and documentation, produces a
  human-reviewable workflow, and executes only the exact approved operations.
- The original **environment builder** plans setup blocks, executes them in an isolated
  Docker container, and repairs failed blocks when possible.

The environment builder's default planner mode is `auto`: it uses the LLM planner when the
configured OpenAI-compatible API key is present, otherwise it falls back to deterministic rules.
HerAgent-Deploy uses its versioned product analysis policy and reports degraded fallback when model
reasoning is unavailable.

## Requirements

- `uv`
- Python `>=3.14`, as declared in `pyproject.toml`
- Optional: an OpenAI-compatible endpoint for LLM planning and repair

HerAgent-Deploy additionally requires the tools used by the generated workflow, such as
`kubectl` and Helm. The environment builder requires the Docker CLI and a running Docker daemon.

Install dependencies and check the CLI:

```bash
uv sync
uv run pheragent --help
```

Run the local test suite:

```bash
uv run pytest -q
uv run ruff check .
```

## Deployment Repository Analyzer (Phase 1)

The Phase 1 analyzer statically discovers deployable units and produces a compact
functional-block DAG. It parses structural relationships in shell automation,
Terraform, Ansible, Compose, Helm, Kustomize, Kubernetes, GitHub Actions, Flux, and
Argo CD sources. Repository code and deployment commands are never executed.

```bash
uv run pheragent deployment analyze \
  --repo https://github.com/mosip/mosip-infra.git \
  --docs https://github.com/mosip/documentation.git \
  --context configs/deployment/mosip/deployment-context.yaml \
  --output .pheragent/deployment/mosip
```

The product uses the versioned `deployment-analysis-v1` policy. Experimental methods and
human-reviewed invariants belong to the separate `pheragent research` entry point, so operators do
not select research treatments when producing a deployment plan.

Each invocation creates an immutable UTC directory under
`.pheragent/deployment/mosip/runs/`. Source clones and content-addressed LLM synthesis
responses are shared in `.source-cache/` and `.llm-cache/`, so prior runs are retained
without paying repeated acquisition or synthesis costs. A normal run contains:

```text
runs/<timestamp>-<run-name>/
├── run-manifest.json
├── events.jsonl
├── metrics.json
├── functional-blocks.yaml
├── deployment-workflow.yaml
└── analysis-report.md
```

Use `--debug` to add the repository index, reference graph, candidate components,
deployment signals, and compact LLM input beneath `debug/`. Analysis uses at most two
schema-constrained LLM requests by default: one bounded retrieval plan and one grounded
synthesis. The raw repository is never included in either request.

Blocks and components use separate namespaces: blocks are `B0`, `B1`, and so on,
while discovered components are deterministically sequenced as `C001_postgresql`,
`C002_keycloak`, etc. The synthesis JSON Schema enumerates the exact allowed `C*`
IDs, so provided `B*` IDs cannot be returned as components.

Failed synthesis attempts are retained under `.llm-cache/failures/`, including the
error, token usage when available, and rejected structured response. An identical
model/input/prompt request is not charged again; the analyzer reuses the failure
record and falls back deterministically. Use `--retry-failed-llm` only when an
intentional retry is appropriate, such as after changing external account state.

### Dry-run and execution

Review `deployment-workflow.yaml`, then preview the exact local commands and their order:

```bash
pheragent deployment run path/to/deployment-workflow.yaml \
  --source-root mosip-infra=/path/to/mosip-infra
```

Dry-run is the default and never starts a repository command. If the workflow is ready,
the output includes an approval token tied to the workflow, ordered commands, and source
roots. Supply that token to execute the unchanged plan sequentially and stop at the first
failure:

```bash
pheragent deployment run path/to/deployment-workflow.yaml \
  --source-root mosip-infra=/path/to/mosip-infra \
  --execute \
  --approve 'sha256:...'
```

For fail-fast experiments, add `--allow-unready` to the dry-run and execution commands.
HerAgent then selects only grounded, ready steps whose prerequisites are also selected;
blocked steps and their dependents remain excluded. The approval token records this mode.

Commands run on the machine hosting the CLI. Deployment scripts are responsible for
reaching Kubernetes or worker nodes. This first execution slice does not yet perform
automatic health validation, rollback, or repair.

The MOSIP context deliberately supplies only the already-provisioned infrastructure
and Kubernetes blocks. The analyzer discovers installer roots and service/application
components without exact path hints. A human must reconcile the generated workflow with the live
environment before approving execution.

## Deployment Research

Research uses the same analyzer core through a separate CLI. A study selects pinned cases,
experimental treatments, repetitions, budgets, and optional human-reviewed invariants:

```bash
uv run pheragent research run \
  --study research/studies/graph-retrieval-pilot.yaml
```

This defaults to cost preflight and reports the planned run and LLM-request ceilings. Add
`--execute` to run the declared study. Research alone exposes the analysis treatments:

- `a0`: deterministic baseline
- `a1`: bounded hybrid analysis
- `a2`: graph-guided bounded hybrid analysis

Rebuild derived result tables without modifying sealed runs with:

```bash
uv run pheragent research summarize .pheragent/research/graph-retrieval-pilot
```

## Configuration

The CLI loads a local `.env` file from the current working directory before
parsing commands. Existing environment variables are not overwritten.

Common LLM settings:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://example.test/v1"
export PHERAGENT_MODEL="gpt-5.5"
```

You can also pass these through CLI flags:

```bash
uv run pheragent plan \
  --repo /path/to/repo \
  --planner llm \
  --model gpt-5.5 \
  --openai-base-url https://example.test/v1
```

The default `--llm-api responses` mode calls the OpenAI Responses API. For a
Chat Completions compatible endpoint, use:

```bash
uv run pheragent plan \
  --repo /path/to/repo \
  --planner llm \
  --llm-api chat-completions
```

## Core Loop

The build loop is intentionally block-oriented:

1. Analyze the repository on disk.
2. Build a base image from an input Dockerfile.
3. Start an isolated Docker container and copy the target repository into
   `/workspace/repo`.
4. Run a container preflight to capture OS, toolchain, package manager, Python,
   and repository marker facts from the actual runtime image.
5. Render setup blocks as shell scripts from repository and runtime context.
6. Commit the copied workspace as the first checkpoint.
7. Execute one block at a time.
8. After each successful block, create a Docker checkpoint with `docker commit`.
9. Continue in the current container on the success path to avoid unnecessary
   container restarts.
10. If a block fails, roll back to the block baseline checkpoint, apply a local
    repair, and replay the block.
11. Persist the final block scripts, execution records, LLM usage, and manifest.

## Single Repository Usage

Plan scripts without Docker:

```bash
uv run pheragent plan --repo /path/to/repo
```

Force deterministic planning:

```bash
uv run pheragent plan --repo /path/to/repo --planner rules
```

Run a checkpointed build:

```bash
uv run pheragent build \
  --repo /path/to/repo \
  --base-dockerfile /path/to/Dockerfile \
  --planner llm \
  --llm-retries 3 \
  --llm-retry-delay 1 \
  --stream-logs
```

Validate the final checkpoint image with an oracle file:

```bash
uv run pheragent build \
  --repo /path/to/repo \
  --base-dockerfile /path/to/Dockerfile \
  --oracle-file /path/to/project.oracle.json \
  --oracle-timeout 1800
```

Resume from an existing checkpoint image:

```bash
uv run pheragent build \
  --repo /path/to/repo \
  --resume-from pheragent:previous-run-003-30-python-deps-success \
  --start-at-block 50-test-tooling
```

When `--start-at-block` is omitted, `pheragent` tries to infer the completed
block from checkpoint image tags ending in `<block-id>-success` or
`<block-id>-repaired`. If the same `--run-id` is reused and the run directory
still contains `blocks/*.json`, resume mode reuses those block scripts instead
of asking the planner to regenerate them.

Useful build options:

```bash
uv run pheragent build \
  --repo /path/to/repo \
  --base-dockerfile /path/to/Dockerfile \
  --state-dir /path/to/repo/.pheragent \
  --image-prefix pheragent-local \
  --max-repair-attempts 2 \
  --max-probe-failures 5 \
  --command-timeout 900 \
  --docker-build-timeout 1800 \
  --keep-container
```

Use `--stream-logs` when you want live Docker, git, block, validation, repair,
and oracle command output in the terminal while still keeping complete logs
under `logs/`.

## Batch Usage

`build-projects` clones and builds multiple projects from a text file. Each
non-empty, non-comment line must contain:

```text
owner/repo commit-or-ref
```

Example:

```bash
uv run pheragent build-projects \
  --projects-file tests/projects/executionAgent.txt \
  --projects-dir projects \
  --oracles-dir oracles \
  --base-dockerfile tests/dockerfile/Dockerfile.heragent-thin \
  --run-id-prefix execution-agent \
  --planner llm \
  --command-timeout 1800 \
  --llm-timeout 180 \
  --stream-logs
```

Use `--limit N` for a small smoke run, `--jobs N` for concurrent project builds,
and `--stop-on-failure` when you want the batch to stop at the first
clone/build failure.

Project checkout first tries to fetch the requested commit/ref directly. If a
short commit hash cannot be fetched as a remote ref, `pheragent` fetches remote
heads/tags with blob filtering and resolves the short hash locally before
checkout. Failed projects are written to:

```text
<projects-dir>/failed-projects.log
```

Each line is tab-separated: owner/repo, commit/ref, checkout directory, repo
path, and a short failure stage such as `prepare_failed` or `build_failed`.

For `build-projects`, `.github` is treated as oracle data instead of build
context. After checkout, if a cloned project contains `.github`, it is moved to
`<oracles-dir>/<project-name>/.github` before the environment build starts. This
keeps CI/CD validation hints out of the agent's repo context while preserving
them for later manual or oracle-based validation.

## Experiment Wrappers

The `scripts/` directory contains benchmark-oriented wrappers around
`pheragent build-projects`. Start with `--dry-run` to inspect the generated
commands before running Docker builds.

SetupBench:

```bash
uv run python scripts/run_setupbench.py \
  --limit 1 \
  --dry-run
```

ExecutionAgent:

```bash
uv run python scripts/run_executionagent.py \
  --limit 1 \
  --dry-run
```

Repo2Run:

```bash
uv run python scripts/run_repo2run.py \
  --limit 1 \
  --dry-run
```

Typical Repo2Run LLM run:

```bash
uv run python scripts/run_repo2run.py \
  --run-root repo2run-runs-gpt-4o-20241120-r30 \
  --run-id-prefix repo2run-gpt-4o-20241120-r30 \
  --image-prefix pheragent-repo2run-gpt-4o-20241120-r30 \
  --planner llm \
  --model gpt-4o-20241120 \
  --max-repair-attempts 30 \
  --project-retries 3 \
  --limit 10
```

The wrappers write per-project logs under `<run-root>/logs/` and summaries under
`<run-root>/results/`. Common options include `--start`, `--limit`, `--only`,
`--fresh-results`, `--skip-existing-results`, `--skip-existing-success`,
`--stop-on-failure`, `--stream-logs`, and `--no-stream-logs`.

## Ablation Modes

Both `pheragent build` and `pheragent build-projects` accept `--ablation`.
Current `main` supports:

| Mode | Effect |
| --- | --- |
| `full` | Full progress-control setting. Block forward, block recovery, local repair, checkpoint rollback, patch-back, and final clean replay are enabled. This is the default in `main`. |
| `without-local-repair` | Disables local repair and patch-back after a failed block. |
| `without-checkpoint-rollback` | Keeps repair, but does not restore the block checkpoint during repair. |
| `without-final-clean-replay` | Disables the final clean replay step. |
| `single-command-forward` | Splits normal forward execution into command-level steps while preserving block-level recovery. |
| `single-command-recovery` | Keeps block-level forward execution, but recovers around the failed command in the live container. |
| `whole-script-forward` | Executes the planned setup as one whole setup artifact. |
| `whole-script-recovery` | Uses whole-artifact recovery after a block failure. |

Single-repository ablation example:

```bash
uv run pheragent build \
  --repo /path/to/repo \
  --base-dockerfile /path/to/Dockerfile \
  --planner llm \
  --ablation without-checkpoint-rollback \
  --run-id my-project-no-rollback \
  --stream-logs
```

Batch ablation example:

```bash
uv run pheragent build-projects \
  --projects-file tests/projects/repo2run.txt \
  --projects-dir projects/repo2run-no-rollback \
  --state-dir state/repo2run-no-rollback \
  --base-dockerfile tests/dockerfile/Dockerfile.heragent-thin \
  --run-id-prefix repo2run-no-rollback \
  --planner llm \
  --ablation without-checkpoint-rollback \
  --limit 10 \
  --stream-logs
```

When comparing ablation modes, use distinct `--run-id-prefix`, `--projects-dir`,
`--state-dir`, and `--image-prefix` values. This keeps manifests, Docker images,
and success-skip decisions from mixing across variants. `build-projects` records
the ablation mode in `llm-usage-projects.jsonl` and only skips an existing
successful run when its manifest was produced by the same ablation mode.

`scripts/run_setupbench.py` also accepts `--ablation` and passes it through to
`pheragent build-projects`:

```bash
uv run python scripts/run_setupbench.py \
  --ablation full \
  --run-id-prefix setupbench-full \
  --image-prefix pheragent-setupbench-full \
  --limit 10
```

## Branch Guide

Use `feat/mosip-deployment-mvp` for the current HerAgent-Deploy product and research implementation:

```bash
git fetch origin
git switch feat/mosip-deployment-mvp
uv sync
uv run pheragent deployment --help
uv run pheragent research --help
```

Use `main` for the integrated environment-builder implementation:

```bash
git switch main
uv sync
uv run pheragent --help
```

Use `repo2run` when reproducing Repo2Run-focused experiments or changes that
are not yet merged back into `main`:

```bash
git fetch origin
git switch repo2run
uv sync
uv run python scripts/run_repo2run.py --limit 1 --dry-run
```

The Repo2Run branch keeps the same high-level workflow but carries
Repo2Run-specific changes around the runner, repair behavior, and validation
paths. The main entry point is:

```bash
uv run python scripts/run_repo2run.py \
  --planner llm \
  --model gpt-4o-20241120 \
  --max-repair-attempts 30 \
  --project-retries 3 \
  --limit 10
```

Use `single-ablation` for the expanded ablation branch. This is the branch that
currently contains the extra single-command ablation variants beyond `main`:

```bash
git fetch origin
git switch single-ablation
uv sync
uv run pheragent build --help
```

Additional modes on `single-ablation` include:

```text
single-command-forward-recovery
single-command-rollback-regenerate
block-rollback-regenerate
block-live-repair-no-patch
```

Use them through the same `--ablation` flag after switching branches:

```bash
uv run pheragent build-projects \
  --projects-file tests/projects/repo2run.txt \
  --projects-dir projects/repo2run-forward-recovery \
  --state-dir state/repo2run-forward-recovery \
  --base-dockerfile tests/dockerfile/Dockerfile.heragent-thin \
  --run-id-prefix repo2run-forward-recovery \
  --planner llm \
  --ablation single-command-forward-recovery \
  --limit 10 \
  --stream-logs
```

If your remote uses a branch named `ablation`, substitute that branch name for
`single-ablation`; always confirm the accepted modes with
`uv run pheragent build --help` on the branch you are about to run.

## Outputs

For single-repository builds, outputs are written under:

```text
<state-dir>/runs/<run-id>/
  context.json
  blocks/*.json
  scripts/*.sh
  logs/<block-id>/*.log
  executions.jsonl
  llm-usage.json
  manifest.json
```

For `build-projects`, batch-level logs include:

```text
<projects-dir>/llm-usage-projects.jsonl
<projects-dir>/failed-projects.log
<projects-dir>/no-repo-projects.log
<projects-dir>/version-mismatch-projects.log
```

When using `--oracles-dir`, quarantined CI/oracle data is written under:

```text
<oracles-dir>/<project-name>/.github/
```

The most important artifact is `scripts/*.sh`: those files are the final
block-by-block setup plan, including any repair snippets that were committed
back into a failed block.

`executions.jsonl` stores one record per Docker build, block, validation,
repair, finalization, clean replay, or oracle command. Each record points at a
full log file under `logs/`, so failures can be debugged with complete
stdout/stderr instead of only manifest tails.

`context.json` includes both static repository analysis and `runtime_notes` from
the container preflight. Those runtime notes are included in LLM planning and in
LLM-assisted block repair.

## Current Scope

- Repository context analysis is deterministic and file-based.
- Docker execution uses the Docker CLI directly.
- Checkpoints are Docker images created with `docker commit`.
- Successful blocks keep using the current container after committing a
  checkpoint; checkpoints are used for resume and failure/repair rollback.
- Block planning in build mode uses both repository context and container
  preflight runtime context.
- Repair is local to the failed block. When LLM support is active, the failed
  block, stdout/stderr tails, runtime context, probe results, and strategy hints
  are sent to the configured OpenAI SDK repair planner.
- LLM planning uses the OpenAI Python SDK and does not store API keys on disk.
  Transient LLM request failures are retried; `auto` mode falls back to
  deterministic rules if the LLM planner cannot produce a plan.
