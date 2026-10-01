# Final Decisions and Humans – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How a protocol makes a decision that no layer above it may weaken, and how a person takes part in a sentinel. Covers [decide_final()](./reference/index.html.md#decide_final), what it cancels and bypasses and how that is recorded, and when to use it. `human()`, a rule that asks a person through Inspect’s approval surfaces, and `sequential()`, which hands it the escalations that led to it, are planned.

## Planned Sections

- [decide_final()](./reference/index.html.md#decide_final): ending a step with a decision
- What it does to siblings (cancelled) and to the layers above (bypassed)
- When to use it, and when returning a decision is enough
- `examples/nested.py`, walked through
- Two final decisions at once (superseded)
- `escalate`, and what happens when it reaches the top
- `human()` and `sequential([rule(), human()])` (planned)
- Panels: a person as one vote with `human(final=False)` (planned)
- Notifications and remote decisions, e.g. Slack (planned)
