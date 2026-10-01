# Composition – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How several monitors and protocols combine into one sentinel. Covers what a list or mapping passed to `Task(sentinel=...)` resolves to, the shipped protocols that combine and threshold, how decisions are ranked, instance names and paths, and nesting. `sequential()`, `defer_to_trusted()` and `resample()` are planned.

## Planned Sections

- What `Task(sentinel=...)` resolves to: a lone protocol, a list or mapping, monitors alone
- Instance names, mapping keys and paths (`llm/suspicion`)
- [concurrent()](./reference/index.html.md#concurrent): every child runs; the strongest decision wins (`terminate` \> `reject` \> `modify` \> `escalate` \> `continue`)
- [threshold()](./reference/index.html.md#threshold): `reject_at` and `terminate_at`, as one number or per monitor and dimension
- [observe()](./reference/index.html.md#observe): recording scores with nothing acting on them, and collecting a benign distribution
- `sequential([...])`: an ordered chain of children that hands escalations forward, e.g. `sequential([rule(), human()])` (planned)
- `defer_to_trusted()` and `resample()` (planned)
- Nesting compositions
- Writing your own composition over the runner
