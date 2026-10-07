# YAML configuration

A user can attach a sentinel at run time, without editing the task: `inspect eval task.py --sentinel sentinel.yaml`. The file names registered monitors and protocols. inspect_ai loads it, inspect_sentinel builds the sentinel tree from it, and the log records the result as config.

## Sub-features

- `yaml-lone-entry`: `sentinel: {name: no_network}` is a lone protocol and becomes the root.
- `yaml-registry-lookup`: the name resolves through the registry to the `@protocol` defined in `examples/no_network.py`.
- `yaml-behaves-as-code`: the resulting sentinel decides exactly as `Task(sentinel=no_network())` does (same checks as `rule`).
- `yaml-config-recorded`: the log header records the config.

## How to get to it (user POV)

- Write a YAML file with a top-level `sentinel:` key, then `inspect eval task.py --sentinel sentinel.yaml`.
- The same `--sentinel` flag also takes a registered protocol name instead of a file.

## Driving it with verify

Preconditions:

- `verify doctor` passes on the inspect_sentinel and inspect_ai lines.

- **Run.** `verify run yaml-config`. It writes `evidence/sentinel.yaml` with `sentinel:\n  name: no_network\n` and runs the task `unwatched` (no sentinel in code) with `--sentinel evidence/sentinel.yaml`.
- **Header.** `stdout.txt` shows `sentinel: no_network` in the task banner.
- **Behavior.** `report.txt` matches the `rule` feature: `continue`, then `reject`, then `work files: ['allowed.txt']`.
- **Proof.** All checks PASS. `diff` the event rows against a `rule` run from the same checkout: they should be identical.

## Gotchas

- The name resolves only because `scripts/tasks.py` imports `examples/no_network.py`, which registers it. A YAML entry that names an unimported factory fails at startup with a registry lookup error.
- To try other shapes (lists, mappings, nested `monitors:` / `children:`), edit `YAML_RULE` in `verify.py`. The design reference (`design/sentinel-reference.md`, "Configuration") lists the shapes. Add checks to match.
