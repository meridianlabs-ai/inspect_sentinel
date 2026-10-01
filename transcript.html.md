# Transcript and Viewer – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

What a sentinel leaves in the log and how to read it. Covers the `SentinelEvent` (one per report, per layer), its fields and statuses, how Inspect View shows the checks inside each tool call, and reading the events with [read_eval_log()](https://inspect.aisi.org.uk/reference/inspect_ai.log.html#read_eval_log) for analysis.

## Planned Sections

- `SentinelEvent`: `factory`, `path`, `function`, `kind`, `status`, `suspicion`, `action`, `message`, `explanation`, `references`
- One event per report, the ones that lost included
- Statuses: reported, cancelled, bypassed, superseded
- The viewer: the summary row, the checks tree, *Did not run*, monitor model calls, the flagged chip
- Cites in explanations that link to messages
- Reading events with [read_eval_log()](https://inspect.aisi.org.uk/reference/inspect_ai.log.html#read_eval_log)
- Monitor usage in [ModelUsage](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#modelusage) and the eval summary
- Threshold comparisons in the viewer (planned)
