# Prometheus memory review for HerAgent Deploy

## Scope

This note reviews the anonymous paper *Prometheus: Long-Horizon Codebase Navigation for Repository-Level Problem Solving*. The supplied manuscript is `prometheus_icse_2027.pdf`, dated July 14, 2026. The review focuses on the memory method and its possible use in the progressive deployment agent. No agent code was changed.

## What Prometheus does

Prometheus treats repeated retrieval as a memory problem. Its context engine receives a high level intent and converts it into a structured query with three fields:

1. The essential information that is needed.
2. Constraints on the retrieval, such as preferred files or size limits.
3. The purpose for which the information is needed.

The engine searches working memory before it searches the repository. A memory hit returns context that was retrieved earlier. A miss searches a repository knowledge graph, organizes and deduplicates the returned snippets, stores them in memory, and returns them to the agent. Later queries are matched by semantic intent rather than by exact wording.

Each memory record binds the structured query to one or more context units. A context unit contains an exact source snippet and metadata for its file and line range. Prometheus embeds the three query fields separately and ranks memory records with a weighted combination of their similarities. The four agents in its issue resolution pipeline share the same memory.

The paper reports an ablation on 50 SWE-bench Verified tasks. Working memory increased resolution from 56 percent to 70 percent and reduced inference cost by 45.4 percent. The paper also reports important limits. The ablation used one run per task, covered issue resolution rather than deployment, and depended on a repository graph, embeddings, PostgreSQL, and Neo4j. See Sections II-B, IV-C, and V of the supplied paper.

## How HerAgent works now

HerAgent already has the right durable foundation. `events.jsonl` is an append only record, the deployment overview tracks long term progress, and checkpoints support resume. The model also receives the active step, current state, the latest tool result, and a short history window.

The current history window is not working memory. `RunHistory.context()` keeps only the last events for the active step. `_event_summary()` keeps a few fields such as status, reason, source, and error. It does not retain the passages returned by `search_sources`, or the text and line range returned by `read_file`. When the latest result leaves the prompt, the model knows that it read a file but no longer knows what it learned from that file.

The progressive prompt tells the model to use event history before reading a file again, but the history often lacks the fact it needs. `ContextSettings.mode` defaults to `memory`, but no progressive memory implementation uses that value. The one shot agent has a different memory mechanism in which the model rewrites up to 12 notes on each turn. That mechanism is useful for experiments, but it can omit facts, rewrite provenance, or preserve stale runtime state.

## Parts to adopt

### 1. Put memory in front of source retrieval

HerAgent should query memory before it runs `search_sources` or `read_file`. On a hit, the tool should return the earlier source context with its provenance. On a miss, the existing lexical and structural retriever should run, and its result should be recorded for later reuse.

The memory should be managed by the harness. The model should request information, but it should not decide which exact source text is retained.

### 2. Reuse the current decision fields as a structured query

HerAgent does not need a new planning model call. The current decision already contains useful fields:

* `query` can represent the essential information.
* Active step source references and `source_path` can represent retrieval constraints.
* `focus` can represent the purpose.

The harness can normalize these fields and use them as the memory key. A later request can match the same intent even when its wording changes.

### 3. Store exact context with provenance

A source memory record should contain:

* The structured query.
* The returned source passages or full file content.
* Source ID, path, line range, and content hash.
* The active step and event sequence that produced the record.

Exact source content should remain the source of truth. A short extracted fact can be added for display, but it must not replace the cited text.

### 4. Derive memory from the event log

`events.jsonl` should remain the only durable record. Successful source results are already written as tool events, so memory should not duplicate their text. The memory index can pair each source decision with its tool result and store a reference to that event sequence. On resume, the index can be rebuilt from the events. The index is disposable and should never become another state file that can disagree with the event log.

### 5. Keep source memory separate from live state

Repository content is stable when its revision and content hash are unchanged. Kubernetes state, process state, and cloud state can change at any time. HerAgent must not return an old runtime observation as if it were current.

Source memory can be reused until its content hash changes. Runtime information should remain in the current observation, component ledger, latest result, and event history. Previous runtime results can provide diagnostic history, but the agent must observe again before it treats them as current state.

### 6. Log memory decisions

Each source request should record whether it was a memory hit or miss, which record matched, and which source ranges were returned. These events will let HerAgent Bench measure repeated retrieval, stale reuse, token savings, and the effect on deployment success.

## Parts not to adopt now

HerAgent does not need the Prometheus knowledge graph for the first experiment. The existing retrieval engine already combines lexical ranking with local file relations. Memory-first retrieval can use that engine as its fallback.

HerAgent also does not need Neo4j, PostgreSQL, a four agent pipeline, or three embedding indexes. Those components solve needs specific to Prometheus and would add operational and code complexity. Exact normalized matching and the existing lexical ranking are enough for a first experiment. Embedding based matching can be added behind the same interface for an ablation.

The model maintained list from the one shot agent should not be copied into the progressive agent as its primary memory. It is a lossy summary. It can remain an experimental context condition, but source memory should preserve exact retrieved content automatically.

## Minimal design

Add one deep module with two operations:

* `recall(request)` returns matching source context that is still valid.
* `remember(request, result)` indexes the successful source result that the event log already records.

The module should hide query normalization, matching, deduplication, content hash checks, and event reconstruction. The source tool router should call it around existing source operations. The agent loop should receive the result in the same tool result shape whether it came from memory or fresh retrieval.

This interface avoids changes to the LLM decision schema and keeps memory out of the main loop. It also allows the matching method to change without changing the agent or source tools.

## Proposed experiment

Compare the following conditions with the same task, model, initial target, and budgets:

1. Current progressive agent with recent history only.
2. Memory-first retrieval with normalized lexical matching.
3. Memory-first retrieval with semantic matching across essential query, constraints, and purpose.

Measure source tool calls, repeated reads of the same content hash and range, input tokens, time, memory hit precision, verified components, and deployment success. Inspect false memory hits separately because an incorrect reused passage can reduce calls while harming the deployment.

The Prometheus result provides a hypothesis, not an expected effect size for HerAgent. Deployment is longer, uses several repositories, and includes changing runtime state. HerAgent should claim an improvement only after matched deployment runs show lower retrieval cost without lower component verification.

## Design review

The current progressive memory design scores 5 out of 10 against the software design framework. The append only history and overview have simple responsibilities, and their interfaces are smaller than their implementations. Four diagnostic rows need work:

* Information hiding fails because the prompt, history summarizer, task mode, and one shot loop define different meanings of memory. One memory module should own the concept.
* Implementation independence fails because changing what history retains can change agent behavior even though the `RunHistory.context()` interface does not state that contract. The source memory contract should state exactly what can be recalled.
* Interface comments fail because `ContextSettings.mode` advertises a memory behavior that is not implemented. Remove the flag until there are two real supported modes.
* Newcomer understanding fails because source knowledge, recent actions, runtime state, and model written notes are all called memory. Give each one a distinct name and keep source memory separate from live state.

The smallest route to 10 out of 10 is to remove the unused mode flag, add one harness managed source memory module, and keep the event log authoritative. The module should preserve exact context and provenance while hiding its matching method. Runtime state should remain outside this cache.
