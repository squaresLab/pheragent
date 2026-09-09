# HerAgent evaluation

## Purpose

Evaluation is a post-run activity. It measures HerAgent outputs without changing the analyzer,
the sealed run, or the deployment environment. The evaluator uses the same pinned repositories,
documentation, and deployment context as evidence, but it does not use a manually curated gold
artifact.

The package lives at `pheragent.evaluation` because evaluation spans all HerAgent phases. It must
not be hidden inside the product analyzer or the research runner:

- `pheragent.deployment` produces deployment artifacts and execution traces.
- `pheragent.evaluation` evaluates those stable artifacts and traces.
- `pheragent.research` selects treatments, repeats runs, and compares evaluation reports.

This boundary prevents an analyzer from seeing its own expected score and lets product and
research entry points use the same evaluator.

Public projects use similar separations:

- [OpenAI Evals](https://github.com/openai/evals) keeps evaluator code in its package and registry
  definitions and datasets under `evals/registry`.
- [Ragas](https://github.com/vibrantlabsai/ragas) exposes a reusable evaluation engine and keeps
  metrics as dedicated package modules.
- [DeepEval](https://github.com/confident-ai/deepeval) exposes metrics as library code and evaluates
  application outputs or agent traces from tests and integrations.
- [OpenEvals](https://github.com/langchain-ai/openevals) provides reusable evaluators that can run
  directly or through a test runner and external experiment tracking.

HerAgent follows the same principle without adopting their larger registries or framework
abstractions before they are needed.

## Evaluation contract

One Phase 1 evaluation receives:

- One or more sealed run directories.
- The pinned local source roots used by those runs.
- The deployment context used by those runs.

Each run directory must contain `functional-blocks.yaml` and `deployment-workflow.yaml`. Repeated
runs are required only for consistency. Evaluation reports are written outside sealed run
directories so the original scientific record remains unchanged.

Scores use the inclusive range `0.0` to `1.0`. A metric may be unavailable when its required
evidence is absent; unavailable metrics must not be silently converted to zero or one. The five
metrics are reported separately rather than combined into one score.

Each metric exposes `issues` before its complete `findings` ledger. Issues are assessed findings
that lower the metric or remain unresolved. Correctly excluded, not-required entities are not
issues. The CLI prints only this compact issue set and groups identical reasons; the full ledger
remains in JSON for audit and scientific analysis.

Source material is untrusted data. An LLM judge must ignore instructions found inside source
content, cite the evidence supporting its verdict, and be allowed to return `insufficient_evidence`.
Deterministic validation runs before the LLM judge. It checks objective properties such as source
containment, file existence, workflow references, and identifier uniqueness. The judge does not see
the deterministic scores, preventing those results from anchoring its independent semantic review.
Hard deterministic contradictions cannot be overturned by the model.

The evaluator uses bounded structured LLM requests:

1. Judge components in batches of at most 20 for validity, profile relevance, and route support.
2. Audit independently observed entities against both components and workflow actions.

All requests use strict schemas, bounded redacted evidence, caching, request limits, and failed
response history. Invalid or unavailable judge responses never erase deterministic results. They
make `judge.complete` false, preserve the conservative baseline, and make the CLI exit non-zero.
Use `--no-llm` only when an offline deterministic baseline is explicitly wanted.

Every component batch must cover its exact supplied ID set. Component judgments are applied only
when all batches succeed, preventing a partially returned list from changing the score. Completeness
may independently succeed or fail because it has its own contract.

Run it without modifying the sealed analysis outputs:

```bash
uv run pheragent evaluation phase-one \
  --run .pheragent/deployment/<system>/runs/<run-id> \
  --source-root repository=/path/to/pinned/repository \
  --source-root documentation=/path/to/pinned/documentation \
  --context configs/deployment/<system>/deployment-context.yaml \
  --model gpt-5.6-terra \
  --output .pheragent/evaluations/<run-id>.json
```

The default request budget is four: enough for three component batches and one completeness audit.
The 18,000-character component-evidence budget is divided across its batches; completeness has an
independent budget of the same size. Successful responses and failures are cached beside the
evaluation reports. Use
`--refresh-llm` for an intentionally fresh judge sample or `--retry-failed-llm` to retry a matching
failed request.

Repeat `--run` for comparable runs. Consistency is unavailable unless their manifests record the
same source pins, context, analysis method, model, and budgets.

## Phase 1: component discovery

### Validity: component groundedness

For every component in `functional-blocks.yaml`, determine whether pinned source evidence supports
it as an independently deployed system component.

Deterministic checks establish hard facts. The LLM evaluates the semantic component claim from the
deployment context, route card, and cited source excerpts.

```text
supported component claims / all discovered component claims
```

Contradicted and insufficiently evidenced claims remain in the denominator.

### Relevance: profile relevance

Determine whether every discovered component belongs to the deployment profile selected by the
deployment context. Penalize excluded profiles, disabled options, and provided infrastructure
reported as discovered components.

Deterministic checks reject explicit exclusions, mislabelled provided prerequisites, and strong
Compose-versus-Kubernetes conflicts. The LLM evaluates remaining profile relevance.

```text
profile-relevant components / all discovered components
```

### Completeness: deployment entity accountability

Independently identify high-confidence deployment entities in the pinned sources. Every entity
must be accounted for as a component, sidecar, deployment action, provided prerequisite, excluded
profile, non-component artifact, or unresolved entity.

```text
accounted high-confidence entities / all observed high-confidence entities
```

The LLM decides whether each independently observed entity is required and accounted for by a
component or workflow action, required but missing, not required, or insufficiently evidenced. This
remains a reference-free completeness proxy, not a claim of true recall.

High-confidence entities currently mean Compose services, Kubernetes workloads, Helm charts and
dependencies, Ansible roles, and component-scoped deployment installers. The evaluator does not
reuse analyzer candidates or analyzer confidence.

### Deployability: valid route coverage

Determine whether every deployable component is covered by a source-supported route in
`deployment-workflow.yaml`. Validate referenced files, working directories, required inputs,
profile conditions, and safe static checks where supported. One shared route may cover many
components.

```text
components covered by a valid route / all deployable components
```

Static validation checks workflow coverage, readiness, executor agreement, source containment,
working directories, and referenced command files. The LLM evaluates whether the route actually
installs or materializes the claimed component. Evaluation never executes a deployment command.

### Consistency: run stability

For repeated runs with identical pins, context, model, and budgets, compare normalized component
sets, dependency edges, block assignments, and deployment routes. Report the mean pairwise
similarity. Fewer than two comparable runs makes this metric unavailable; three or more runs are
recommended.

## Phase 2

## Phase 3

## Phase 4
