---
name: verify-sentinel
description: Prove inspect_sentinel works end to end by driving the real `inspect eval` CLI with the shipped example monitors and protocols watching a scripted agent, then checking the SentinelEvents in the eval log and the files the agent's commands did or did not write. Use to verify a sentinel change or PR, to compare a branch against origin/main for regressions, or when someone asks "does this actually work in an eval".
---

# Verify inspect_sentinel

inspect_sentinel is a library. Its user-facing surface is an Inspect eval: `Task(sentinel=...)` in code, or `inspect eval task.py --sentinel sentinel.yaml` on the command line. The sentinel watches each tool call, and its decisions and observations land in the eval log as `SentinelEvent`s. This skill drives that path for real: the `inspect` CLI from this checkout's `.venv` runs tasks whose sentinels are the ones in `examples/` (`no_network`, `suspicion`, `failure_count`, `protected`), unmodified.

What is faked, and why it is safe to fake:

- **The agent** is `mockllm/model` replaying a fixed list of bash commands, one per turn. No API keys, deterministic.
- **The monitor model** (role `monitor`) is a mockllm that returns `{"score": 0.9}` when the judged call contains `curl`, else `0.1`.
- **The sandbox** is `local`, not Docker: commands run on this machine. Every scripted command is harmless (`echo`, `cat /etc/hostname`, `curl` to a closed local port, `python3 -c '1/0'`). Each command the sentinel should stop also does `touch $VERIFY_WORK/<marker>`, so a marker file proves the call ran and its absence proves it was stopped.

Everything else is real: inspect_ai's dispatcher, the sentinel runner, config resolution, YAML loading, and log writing.

Other surfaces, not driven here: a person answering in the approval panel (`examples/escalate_to_human.py`, interactive), the portability check (covered by `tests/test_portable.py`), and `spikes/wasm_sidecar` (its own `build.sh`).

## Layout

- `scripts/verify`: the entry point. It runs `scripts/verify.py` with this checkout's `.venv/bin/python`.
- `scripts/tasks.py`: the `@task`s that `inspect eval` runs. One task per feature; the scripted commands are at the bottom.
- `features/`: the verification map, one file per feature. Read `features/README.md` first.

## Launch

There is no server. Each `verify run` starts one `inspect eval` subprocess that exits when the eval finishes (a few seconds). Prerequisites:

- `.venv` exists, with `inspect_ai` and `inspect_sentinel` installed. Check with `doctor`.
- Run commands from the repo root.
- Optionally set `VERIFY_OUT` to choose where runs go. The default is `$TMPDIR/verify-sentinel`. Use your scratchpad directory when you have one.

```bash
export VERIFY_OUT=<scratchpad>/verify      # optional
.agents/skills/verify-sentinel/scripts/verify doctor
.agents/skills/verify-sentinel/scripts/verify run all            # or: run rule | yaml-config | llm-monitor | trajectory | composition
```

`run` prints the event table and a PASS/FAIL line per check, and exits 1 if any check fails. Each run makes `$VERIFY_OUT/<timestamp>-<feature>/` with `evidence/` and `work/` inside.

## Doctor

`verify doctor` is read-only. Run it first, and again whenever a result looks wrong. It reports:

- which python, and where `inspect_sentinel` and `inspect_ai` are imported from, with each one's git commit and branch. inspect_ai is normally an editable install from a sibling checkout. Confirm it is the commit you meant to test.
- whether `src/` and `examples/` have uncommitted changes, which would mean you are not testing the commit you think you are.
- whether `inspect_sentinel` imports only `inspect_ai.core`, using inspect_ai's own `check_imports`, the same check as `tests/test_package.py`.
- whether inspect_ai's `_Host.generate` takes the same parameters as sentinel's `Host.generate`. When it doesn't, every monitor that calls `context.host.generate(role=...)` fails at run time. `examples/llm_suspicion.py` does that, so the `llm-monitor` feature and the `audit` monitor in `composition` will fail.

## Drive

Each feature in `features/` names its task in `scripts/tasks.py`, the commands the agent runs, and the checks `verify.py` applies. To add a scenario, add a `@task` to `tasks.py`, then add an entry to `FEATURES` in `verify.py` with the task name and its checks, then add a feature file. Checks read the eval log, never sentinel internals.

### Regression baseline against another checkout

`VERIFY_BASE=<dir>` runs another checkout's `src/` and `examples/` with this checkout's `.venv` and harness. It works by prepending `<dir>/src` to `PYTHONPATH`, ahead of the editable install. Use it to compare a PR against `origin/main`. The base runs against the inspect_ai installed in this `.venv`, so it must be able to import it: when the PR moves with an inspect_ai change the base can't import (for example, an import path the installed inspect_ai no longer has), `doctor` fails on the import and `run` reports that no log was written. Then compare at an inspect_ai commit both can import, or report that no baseline was possible. Extract main without touching git state:

```bash
mkdir -p <scratch>/main && git archive origin/main src examples | tar -x -C <scratch>/main
VERIFY_BASE=<scratch>/main .agents/skills/verify-sentinel/scripts/verify doctor   # must show inspect_sentinel imported from <scratch>/main
VERIFY_BASE=<scratch>/main .agents/skills/verify-sentinel/scripts/verify run all
diff <(sed 1d $VERIFY_OUT/<pr-run>/evidence/report.txt) <(sed 1d $VERIFY_OUT/<base-run>/evidence/report.txt)
```

In `report.txt`, the work path is written as `$VERIFY_WORK`, so a PR run and a base run of the same feature should be identical apart from the first line. Any other difference is a behavior change. Explain it, or report it as a regression.

### Unit tests, types, lint

The repo's own checks are `pytest`, `pyright` (strict, Python 3.10 floor) and `ruff`. CI runs them with `uv sync --group dev` then `uv run ...`, and `make test` / `make typecheck` use `uv run`.

- **Do not run `uv sync`, `uv run` or `make test` / `make typecheck` / `make check` in a checkout whose `.venv` has an editable inspect_ai.** `uv run` syncs first. It replaces the editable inspect_ai with the version pinned in `uv.lock`, so you stop testing the inspect_ai you meant to test.
- If pytest, pyright and ruff are not in `.venv` (check `ls .venv/bin`), ask the user before installing anything. Without them, report that unit tests, types and lint were not run. Don't claim they pass.

## Evidence

Each run keeps these files under `$VERIFY_OUT/<run-id>/evidence/`:

- `command.txt`: the exact `inspect eval` command, the sentinel `src/` that ran, and the exit code.
- `stdout.txt`, `stderr.txt`: the CLI's output.
- `logs/*.json`: the eval log. It is the primary record. Read it with `inspect_ai.log.read_eval_log(path, resolve_attachments=True)`.
- `sentinel.yaml`: for `yaml-config`, the config file that was passed.
- `report.txt`: the flattened event table (sentinel events and tool calls in order), the files in `work/`, and the PASS/FAIL checks.

Proof standards:

- A pass needs both the log and the side effect. The sentinel event says `reject`, and the marker file is absent from `work/`. Either one alone is not proof.
- Check the log's `status`, not the CLI exit code. `inspect eval` exits 0 even when the eval log ends with `status=error`. `verify.py` checks this, under "eval log status is success".
- Report a failing check with its `report.txt` line and the event or error it saw. If a failure is the known `role=` mismatch that `doctor` flags, say so rather than calling it a regression.
- When you claim no regression, name the base you compared against and show the `diff` result.

## Cleanup

`verify run` starts no long-lived processes: each `inspect eval` exits before `run` returns, and the `local` sandbox removes its own temp dir. `cleanup` removes only the run's `work/` dir (the marker files) and keeps `evidence/`:

```bash
.agents/skills/verify-sentinel/scripts/verify cleanup $VERIFY_OUT/<run-id>
```

It refuses a directory that is not a run under `$VERIFY_OUT`. Remove a `git archive` baseline dir yourself when you are done. Never delete `evidence/` as part of cleanup.
