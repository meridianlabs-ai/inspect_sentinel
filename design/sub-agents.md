# Sub-agents

How a sentinel covers an agent that starts other agents: which of their steps it sees, how a monitor tells agents apart, and what state they share. It extends [sentinel.md](sentinel.md) ("The payloads", open question 22) and [generate-stages.md](generate-stages.md) (decisions 1 and 2, on sentinel #42).

Status: decided in part with the maintainer on 2026-10-10, revised 2026-10-12. Every agent loop's model calls are checked from the first release of the generate stages; the instance record and per-instance store are the [planned direction](#planned-direction), not built.

## Terms

An **agent instance** is one running copy of an agent loop. A sample's main agent is one instance; each `handoff()` or `as_tool()` run of a sub-agent is another, so an `as_tool()` agent called twice is two instances. A scaffold's own sub-agents behind a bridge (Claude Code's Task tool, Codex's multi-agent) are instances too, once the bridge reports them.

A **helper call** is a model call made by agent code that is not an agent turn: a solver's self-critique pass, a classifier, a model-graded check inside a tool. Monitors' and protocols' own calls through `Host.generate`, scorers and compaction summaries are excluded already (generate-stages.md, decision 1).

Example used below:

```
main agent (react)                        instance a1
 ├─ handoff(researcher)   call_3          instance a2  (parent a1, spawned by call_3)
 └─ coder_tool (as_tool)  call_7          instance a3  (parent a1, spawned by call_7)
      └─ tester_tool (as_tool) call_12    instance a4  (parent a3, spawned by call_12)
```

## What v1 covers

Tool stages: every tool call that goes through `execute_tools` is checked, whichever instance makes it, so `a2`, `a3` and `a4`'s tool calls are checked in v1. The `handoff()` call itself is checked as a tool call.

Generate stages (generate-stages.md, decisions 1 and 2, revised 2026-10-12): a call made by an agent loop, that is inside an agent span (`react`, `handoff()`, `as_tool()`, `deepagent`, the bridges) or on the sample's active model during the solver phase, and not inside a tool's own body, a sentinel run, a compaction or a scorer. Sub-agents and role-model agents are therefore checked whatever model they use, matching the tool stages. A helper call made inside a tool's body is not checked; one a solver makes directly on the active model still is.

Bridged agents (inspect_ai #5765): the bridged scaffold's host tool calls are checked; its internal sub-agents are not attributed to separate instances.

## Decided

Decided with the maintainer on 2026-10-10, so that monitors written against v1 do not break when coverage widens.

1. **Every agent loop's model calls are checked** (generate-stages.md decisions 1 and 2, revised 2026-10-12, replacing "v1 coverage is the active model"). The tool stages already checked every agent's tool calls; leaving sub-agent turns on other models unchecked was inconsistent and let an agent hand risky work to a sub-agent. Requires every agent loop to open an agent span consistently, tested per loop.
2. **`step.conversation` identifies one agent instance.** It is public on every step today and holds the agent span id. The contract is the instance, not the span: a new id for each sub-agent run, distinct from its parent's; stable across that instance's compactions and across both stages of one call; the same live and in replay. Monitors may key state and logic on it (for example AISI's "record the opening prompts at a conversation's first monitored step", R7 and R10). If the span id stops meeting the contract (a scaffold whose agents share a span, a bridge with several sub-agent streams), the implementation changes, not the meaning.
3. **`Context.store_as()` stays per sample.** State is namespaced by the instance's path in the sentinel tree (`_context.py`), not by agent instance, so every agent's steps share one monitor's state in v1 and after. Per-instance state comes later as an explicit option, `store_as(Model, scope="conversation")` or similar, keyed by `step.conversation`. Switching the default later would silently change every stateful monitor; an option is additive. Docs say so, with the consequence: a monitor that keeps cross-step state and wants it per agent keys it by `step.conversation` itself until the option exists.
4. **Helper calls:** one made inside a tool's body is not checked; one a solver makes directly on the active model still is, for now. When helper calls can be marked (below), marked calls stop being checked. Docs say so, so nobody builds on seeing them.

## Planned direction

Agreed with the maintainer as the target ("we generally need this"); each item is its own piece of work, ordered roughly by dependency.

### Coverage follows the agent tree, not the model

*Largely decided 2026-10-12 through the agent-loop rule above; what remains is marking helper calls on the active model.*

Every model call and tool call made by an agent instance or any of its descendants is checked, whatever model it uses; helper calls are excluded by marking them, not by model identity. This closes v1's gap that a sub-agent on a cheaper model is unmonitored at the generate stages, which an agent could otherwise exploit by handing risky work to one.

- **Marking helper calls.** A context the caller enters (`with helper_call():` or a `GenerateConfig` flag), or a role that declares it (`model_roles` whose calls are never agent turns). To decide: which, and whether marking is the caller's job or the model role's.
- **Which loops define an instance.** `react`, the `generate()` solver, `handoff()`, `as_tool()`, `deepagent` and the bridges mint an instance when they start. A custom solver calling `model.generate` belongs to the instance it runs in (the main agent if none).

### The instance record

When an instance starts, the code that starts it creates a record, and every step and every model and tool event carries it:

| field | example (`a4`) | meaning |
|---|---|---|
| `id` | `a4` | unique per running instance; this is `step.conversation` (decision 2) |
| `name` | `tester` | the agent's registered name |
| `parent` | `a3` | the instance that started it; none for the main agent |
| `spawned_by` | `call_12` | the parent's tool call that started it, which is that parent's step id at the tool stages |
| `depth` | `2` | distance from the main agent |

On the step as `step.agent` (with `step.conversation` equal to `step.agent.id`); on `ModelEvent`, `ToolEvent` and `CompactionEvent` as the instance id (the `conversation` id sentinel.md asks core to record). Uses:

- A protocol branches by agent: stricter thresholds for `coder`, `terminate` past a depth.
- A monitor judges a sub-agent's action against what it was asked to do, by looking up `spawned_by` (the tool call whose arguments carried the instruction).
- Parallel instances stay apart: their history, state (with the per-instance store option) and decisions do not mix.
- AISI R10 (history as sent, per instance) collects the model calls tagged with one id; R4 (earlier `human()` approvals in the instance) keys by it; R7 (replay) rebuilds each instance from the tagged events.

### Bridged scaffolds

A bridge sees each of a scaffold's sub-agents as a separate request stream (a session, a thread, a parent message id, depending on the client). It mints a child instance per stream, with `parent` the bridged agent's instance and `spawned_by` the scaffold's own tool call where the client reports it. The Codex Multi-Agent V2 bridge work (inspect_ai #4787 onward) and the agent message bridge are where this lands. A proxy host treats each session as an instance, with whatever parent link the client provides.

### Per-agent configuration

Not planned. One sentinel covers the tree and branches on `step.agent`. `handoff(agent, sentinel=...)` or a per-agent mapping comes only if branching proves insufficient.

## Open questions

1. How a helper call is marked (context, config flag or role), and who marks it.
2. Whether a `handoff()` sub-agent, which takes over the conversation rather than running beside it, is a new instance or continues its parent's. Proposed: a new instance, with `parent` set, since its turns are its own; its history includes the parent's messages it was handed.
3. The name of the per-instance store option and whether it also scopes by stage.
4. Whether replay must reproduce instance ids exactly or only their structure (ids are minted at run time).
5. What a monitor sees of a sub-agent's instruction: the `spawned_by` call's arguments, or a rendered view (workstream 3, R8).
