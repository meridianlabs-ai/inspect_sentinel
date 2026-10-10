# Changelog

## 0.1.0 (2026-10-10)

First release. Inspect Sentinel adds monitors and control protocols that watch an agent's tool calls during an Inspect eval and decide whether each call goes ahead. It is under active development and its API may change between 0.x releases.

- Install with `pip install inspect-sentinel`. Requires `inspect-ai>=0.3.278`.
- Give a task a sentinel with `Task(sentinel=...)`, override it with `eval(sentinel=...)` or `eval_set(sentinel=...)`, or pass a YAML/JSON configuration with `inspect eval --sentinel`. The configuration is recorded in the log, so `eval_retry()` rebuilds the same sentinel.
- **Monitors** (`@monitor`) observe a tool call before it runs, or its result afterwards, and report a suspicion score. They never act.
- **Protocols** (`@protocol`) decide what happens: proceed, reject, modify, terminate or escalate. They can consult monitors first; a protocol with no monitors is a rule.
- Shipped protocols: `threshold()`, `observe_only()`, `concurrent()`, `sequential()` and `human()`.
- Helpers for rules: `call_text()`, `tool_matches()`, `find_words()` and `result_text()`.
- Every observation and decision is recorded in the eval log and shown in the viewer, along with the version of the monitor or protocol that produced it.

Documentation: https://meridianlabs-ai.github.io/inspect_sentinel
