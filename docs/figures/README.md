# HerAgent trajectory figures

These SVGs are slide-ready summaries of downloaded AWS experiment records. They separate the agent's recorded rationale, the proposed action, the harness result, and the observed state change.

- `heragent-progressive-loop.svg` is the minimal architecture slide for the progressive agent loop.
- `trajectory-progressive-successes.svg` is the recommended success slide: PostgreSQL repair, approved database initialization, and partial-state cleanup.
- `trajectory-progressive-failures.svg` is the recommended failure slide: over-broad execution, missing interactive-process support, and unbounded retrieval.
- `trajectory-progressive-postgres.svg` shows the exact progressive recovery that added the missing Bitnami Helm repository and then validated PostgreSQL.
- `trajectory-full-plan-vs-progressive.svg` compares the older full-plan cascade with the later progressive run.
- `trajectory-retrieval-stagnation.svg` shows how semantically identical Kafka prerequisite searches continued under different wording.

## Provenance

The figures use these records:

- `/home/obotu/Downloads/heragent-experiments/mosip-policy-20260929T133639Z.tgz`
  - `run/trajectory.jsonl`
  - `run/actions.jsonl`
  - `run/state.json`
- `/home/obotu/dev/heragent-experiment-results/mosip-full-feedback-20260915T185216Z/deployment-execution.log`
- `/home/obotu/dev/pheragent/.pheragent/imported-runs/aws-progressive-20261012T090524Z/mosip-linked-source-20261007T115820Z-logs/mosip-linked-source-20261007T115820Z/run/events.jsonl`

Quoted rationale is copied from the recorded decision `reason` fields and line-wrapped. Ellipses mark omitted text. Command output is shortened to the state-changing or failure-bearing lines. The figures do not claim to expose hidden model chain-of-thought.

Color key:

- Blue: observation or evidence
- Amber: proposed or approved action
- Green: validated state change
- Red: failure or stagnation

Each figure is supplied as editable SVG and as a 1600×900 PNG. SVG preserves exact text and scales cleanly; PNG is convenient for slide software that handles SVG poorly.
