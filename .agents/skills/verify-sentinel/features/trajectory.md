# Trajectory state

One monitor can have a function per stage, returned as a `MonitorGroup`. `failure_count` (from `examples/trajectory.py`) has `before` (tool call) and `after` (tool result), which share a `StoreModel` through `context.store_as()`. `after` counts failed calls, and `before`'s suspicion rises by 0.2 per earlier failure. `observe_only` records the observations and acts on nothing.

## Sub-features

- `group-stages`: both functions run, at the right stages. Each step records `before` at `tool_call` and `after` at `tool_result`, all at path `failure_count`.
- `store-state`: state that `after` writes is visible to the next `before` in the same sample.
- `observe-only`: no decision acts. Every call runs.

## How to get to it (user POV)

- `Task(sentinel=observe_only(failure_count()))`.

## Driving it with verify

Preconditions:

- `verify doctor` passes on the inspect_sentinel and inspect_ai lines.

- **Run.** `verify run trajectory`. The agent runs `python3 -c '1/0'` twice, then the echo command.
- **Stages.** `report.txt` alternates `tool_call ... fn=before` and `tool_result ... fn=after`.
- **State.** `before` suspicions are `0.0`, `0.2`, `0.4`. `after` explanations are `call failed`, `call failed`, `call succeeded`.
- **Side effect.** `work files: ['allowed.txt']`. Nothing is rejected.
- **Proof.** All checks PASS.

## Gotchas

- inspect_ai's `bash()` returns stderr and a non-zero exit as normal tool output, not as a tool error. `failure_count` treats a call as failed only on a tool error or `Traceback` in the output, so `false` or `ls /missing` count as successes. That is why the scripted failure is a Python traceback.
- State is per sample. A multi-sample variant must not expect counts to carry over.
