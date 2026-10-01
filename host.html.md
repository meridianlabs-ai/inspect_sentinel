# The Host and Model Roles – Inspect Sentinel

> **WARNING: WarningOutline**
>
> This page is an outline: a summary and the sections planned for it, not yet the documentation. See [Getting Started](./index.html.md) for a worked example.

## Overview

How a monitor or protocol calls a model. Covers `context.host.generate()` and why it is used instead of [get_model()](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#get_model), the `model=` and `role=` convention for LLM monitors, the `monitor` model role and its fallback, how monitor usage is recorded and kept off the agent’s limits, and structured output. Outbound HTTP through `fetch` is planned.

## Planned Sections

- `context.host.generate()`: prompts, messages, tools and [GenerateConfig](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#generateconfig)
- `model=` and `role=` on an LLM monitor’s factory
- The `monitor` role: `--model-role monitor=...` and `Task(model_roles=...)`, and the fallback to the agent’s model
- The `trusted` role for `defer_to_trusted()` and `resample()` (planned)
- Structured output with [ResponseSchema](https://inspect.aisi.org.uk/reference/inspect_ai.model.html#responseschema)
- Usage: monitor calls recorded separately and not charged to the agent’s limits
- Treating agent text as untrusted in prompts
- Other effects: `fetch` (planned), and calling inspect_ai APIs directly at the cost of portability
