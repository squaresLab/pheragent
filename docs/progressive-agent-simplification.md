# Progressive agent simplification

- [x] 1. Characterize continuation, context, policy, and action recording.
- [x] 2. Derive active-step context from `events.jsonl`; remove model-written memory.
- [x] 3. Continue after blocked steps and report binary component outcomes.
- [ ] 4. Allow in-scope commands; review only boundary crossings and opaque scripts.
- [ ] 5. Record each started action, result, validation, and checkpoint atomically.
- [ ] 6. Stop after a no-progress pass and derive a replay runbook from events.

Constraints: keep `events.jsonl` authoritative, add no recovery loop or class hierarchy,
and delete more production code than this work adds.
