# Protocols and Rules – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How to write a protocol: a `@protocol` factory returning a function that returns a [Decision](./reference/index.html.md#decision). Covers rules (protocols with no children), the five actions and where each is legal, what the agent is told (`message`) versus what the log records (`explanation`), modifying a tool call, and protocols that consult monitors or other protocols through the runner.

## Planned Sections

- The `@protocol` decorator
- Rules: a protocol with no children
- Actions: `continue`, `reject`, `modify`, `terminate`, `escalate`, and which stages allow each
- `Decision.reject(explanation, message=...)`: the log versus the agent
- Modifying a call with `dataclasses.replace(step.call, arguments=...)`
- Requesting human review with `audit=True` (recorded; nothing acts on it yet)
- Consulting children: [run_monitors()](./reference/index.html.md#run_monitors), [run_protocols()](./reference/index.html.md#run_protocols), [run_children()](./reference/index.html.md#run_children), and the [Observations](./reference/index.html.md#observations), [Decisions](./reference/index.html.md#decisions) and [Reports](./reference/index.html.md#reports) they return
- Taking children as a parameter, and naming them
- The boundary check: shape errors raised before a decision takes effect
- Running at every stage with [Step](./reference/index.html.md#step)
- Shipping protocols in a package
