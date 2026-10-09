# inspect_sentinel verification map

This directory is the maintained source for verifying inspect_sentinel's user-facing behavior: what a sentinel does inside an `inspect eval`. Read this index, then use the matching feature file as the recipe. `../SKILL.md` covers launch, doctor, evidence and cleanup.

## Baseline preconditions

- Run from the repo root with this checkout's `.venv`.
- `verify doctor` reports the inspect_sentinel and inspect_ai commits you mean to test, and `src/` and `examples/` have no local changes.
- `VERIFY_OUT` points at a scratch directory, or is unset (default `$TMPDIR/verify-sentinel`).
- Each `verify run` gets its own `work/` directory, so runs do not share state and can run side by side.

## Driving conventions

- Drive through `.agents/skills/verify-sentinel/scripts/verify run <feature>`. It runs the real `inspect eval` CLI on a task in `scripts/tasks.py`.
- The sentinels under test are the ones in `examples/`, imported unchanged. Do not copy or edit them for a run. Change the scripted commands in `tasks.py` instead.
- Every command the sentinel should stop also touches a marker file in `$VERIFY_WORK`. The marker proves the call ran.

## Proof and skip reporting

- Proof is a run whose `report.txt` shows all PASS, together with that run's `evidence/` path.
- A check that fails because of a `Host.generate` mismatch flagged by `doctor` is reported as that, with the error string from `report.txt`, not as verified and not as a new regression.
- A regression claim, or a no-regression claim, names the base checkout (`VERIFY_BASE`) and shows the `diff` of the two `report.txt` files.
- Do not report a feature as verified through a different feature's run.

## Feature entry contract

Each feature file starts with an H1 title and one paragraph describing the user-visible behavior. It then uses exactly four H2 sections in this order: `Sub-features`, `How to get to it (user POV)`, `Driving it with verify`, `Gotchas`.

## Features

- [A rule](./rule.md): a protocol with no monitors (`no_network`) rejects a call, and the agent sees why.
- [YAML configuration](./yaml-config.md): `inspect eval --sentinel sentinel.yaml` attaches a sentinel to a task that has none.
- [An LLM monitor under threshold](./llm-monitor.md): `threshold(suspicion(), reject_at=0.8)` asks the monitor model for a score and acts on it.
- [Trajectory state](./trajectory.md): a `MonitorGroup` with `before` and `after` functions shares `store_as()` state across a sample, under `observe_only`.
- [Composition and final decisions](./composition.md): a mapping with a `concurrent()` of rules; `decide_final()` ends a step and bypasses the layers above it.
