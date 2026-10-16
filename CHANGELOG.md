# Changelog

## [0.1.4](https://github.com/meridianlabs-ai/inspect_sentinel/compare/0.1.3...0.1.4) (2026-10-16)


### Features

* Host.score marks every block and message header in its prompt with a nonce ([#105](https://github.com/meridianlabs-ai/inspect_sentinel/issues/105)) ([8e8ff62](https://github.com/meridianlabs-ai/inspect_sentinel/commit/8e8ff62a2a1f81ef67acbbe9a50949fdbc65e48a))

## [0.1.3](https://github.com/meridianlabs-ai/inspect_sentinel/compare/0.1.2...0.1.3) (2026-10-15)


### Features

* Host.score, with excerpts for what a monitor shows a model ([#87](https://github.com/meridianlabs-ai/inspect_sentinel/issues/87)) ([8cdb257](https://github.com/meridianlabs-ai/inspect_sentinel/commit/8cdb257cac864d287c8c156e4ad5772fc2f69710))
* require Python 3.11 and test on Python 3.14 ([#95](https://github.com/meridianlabs-ai/inspect_sentinel/issues/95)) ([dd3f250](https://github.com/meridianlabs-ai/inspect_sentinel/commit/dd3f25094136e698f88f579009816dd3a1a7fe6c))

## [0.1.2](https://github.com/meridianlabs-ai/inspect_sentinel/compare/0.1.1...0.1.2) (2026-10-14)


### Features

* human() can ask a person about a whole assistant turn, before its tool calls run ([#89](https://github.com/meridianlabs-ai/inspect_sentinel/issues/89)) ([a3faeea](https://github.com/meridianlabs-ai/inspect_sentinel/commit/a3faeea182539252abe49cd9f83cc3447d739aab))
* step.history is None when the host can't provide the caller's message list ([#86](https://github.com/meridianlabs-ai/inspect_sentinel/issues/86)) ([5948f57](https://github.com/meridianlabs-ai/inspect_sentinel/commit/5948f5766365f5f284677ccb81231d291c58492c))
* tool-stage monitors see the offered tools and whether the result was truncated ([#88](https://github.com/meridianlabs-ai/inspect_sentinel/issues/88)) ([00130a9](https://github.com/meridianlabs-ai/inspect_sentinel/commit/00130a9a26afd4c14434eed522d78011a53e4235))

## [0.1.1](https://github.com/meridianlabs-ai/inspect_sentinel/compare/0.1.0...0.1.1) (2026-10-12)


### Features

* monitors watch every agent loop's model calls (generate stages) ([#42](https://github.com/meridianlabs-ai/inspect_sentinel/issues/42)) ([3d3ce8e](https://github.com/meridianlabs-ai/inspect_sentinel/commit/3d3ce8e1acb82fa481df57e963385e716b0ad64a))
* add handle_escalation() and terminate an unhandled root escalate ([#74](https://github.com/meridianlabs-ai/inspect_sentinel/issues/74)) ([85234ba](https://github.com/meridianlabs-ai/inspect_sentinel/commit/85234ba34cc8d37e46c1fb188727c434ec0e2d2d))
* step.input is None when no model call produced the call or its record is gone ([#83](https://github.com/meridianlabs-ai/inspect_sentinel/issues/83)) ([f30b236](https://github.com/meridianlabs-ai/inspect_sentinel/commit/f30b236945b537f38189f458d0f22a8b428beb4e))
* threshold escalates or terminates on scores without reject ([#79](https://github.com/meridianlabs-ai/inspect_sentinel/issues/79)) ([9d4ac61](https://github.com/meridianlabs-ai/inspect_sentinel/commit/9d4ac618b51c4452be7778c5021932e799ec0da6))


### Bug Fixes

* run the children a composition was configured with ([#82](https://github.com/meridianlabs-ai/inspect_sentinel/issues/82)) ([559843b](https://github.com/meridianlabs-ai/inspect_sentinel/commit/559843b01197ffe8e30986bb93a6d384a8d83a25))
* let a factory return another factory's configured instance ([#75](https://github.com/meridianlabs-ai/inspect_sentinel/issues/75)) ([e30a7c9](https://github.com/meridianlabs-ai/inspect_sentinel/commit/e30a7c987136229fff8b0521142ef2231c8939ec))
* name a protocol's own escalation in the unhandled explanation ([#78](https://github.com/meridianlabs-ai/inspect_sentinel/issues/78)) ([8afeeb9](https://github.com/meridianlabs-ai/inspect_sentinel/commit/8afeeb98da459b47c676f4527339876f7e6bda86))
* name the action that lets a step go ahead proceed, matching Decision.proceed() ([#84](https://github.com/meridianlabs-ai/inspect_sentinel/issues/84)) ([09913cd](https://github.com/meridianlabs-ai/inspect_sentinel/commit/09913cd1be7ebb8ca13c1a3e0edbadc62872df10))

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
