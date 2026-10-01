# Approval and Review – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How a sentinel sits beside Inspect’s existing `approval=` and `review=` policies. Covers the order in which they run around a tool call, when to use each, the differences in vocabulary (`approve` versus `continue`, and the agent-facing `message`), and what is planned for running approvers as protocols.

## Planned Sections

- The order: approval, then the sentinel, before a call; review, then the sentinel, after it
- When to use approval, review or a sentinel
- Vocabulary: `approve` and `continue`; `Approval.explanation` and `Decision.message`
- Using an approval and a sentinel on one task
- Approvers as protocols, `as_protocol()` (planned)
- Reviewers and after-call protocols (under review)
