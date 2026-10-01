# State – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How a monitor or protocol keeps state across the steps of a sample, such as a running trajectory score. Covers `context.store_as()` with a [StoreModel](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#storemodel), why closure variables are wrong for state, how the store is namespaced by instance path, and reading the state back from a finished log. Per-task state is planned.

## Planned Sections

- `context.store_as(Model)` and [StoreModel](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#storemodel)
- Why not closure variables or the global [store_as()](https://inspect.aisi.org.uk/reference/inspect_ai.util.html#store_as)
- Namespacing by instance path, and what renaming a key does to it
- Sharing state between the functions of one [MonitorGroup](./reference/index.html.md#monitorgroup) (`examples/trajectory.py`)
- The whole sample store: `context.store`
- State in the transcript, checkpointing and resume
- Reading state from a log for analysis
- Per-task state, `scope="task"` (planned)
