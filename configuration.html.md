# Configuration – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

The ways to give a task its sentinel. Covers `Task(sentinel=...)`, overriding it with `eval(sentinel=...)` and `eval_set(sentinel=...)`, configuration files in YAML or JSON passed with `--sentinel`, a registered name on the command line, how factory parameters and nested children are written, and how the configuration is recorded in the log so that [eval_retry()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval_retry) rebuilds the same sentinel.

## Planned Sections

- `Task(sentinel=...)`: one instance, a list, or a mapping
- Overriding a task’s sentinel: `eval(sentinel=...)`, `eval_set(sentinel=...)`
- `inspect eval --sentinel sentinel.yaml` and `--sentinel <name>`
- The file format: `name`, `params`, and nested keys named for the factory’s parameter (`monitors:`, `children:`)
- Lists, mappings and single entries in YAML
- How names are found: registered names and the `inspect_sentinel/` prefix
- Configuration errors and the entry paths they name
- The sentinel in the log, and [eval_retry()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval_retry)
- Model roles from the command line
