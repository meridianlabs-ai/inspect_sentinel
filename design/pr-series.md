# Implementation: the first PR series

How `sentinel.md` becomes code, in an order where each PR is reviewable alone, has its own tests, and leaves `main` importable. Agreed 2026-09-21; PR 0 opened 2026-09-22.

## Goal

The minimal surface that lets `inspect_ai` connect: the types a dispatcher constructs, the decorators an author writes against, the runner a protocol calls, and the three protocols the compile rule needs. No `inspect_ai` dispatcher yet; that is the next series, on inspect_ai's `feature/sentinel` branch.

## Decisions

- **Surface only.** Sentinel ships types and functions; `inspect_ai` wiring comes after. Verified by unit tests with fake `Host` and `Recorder`.
- **Real registration.** `@monitor` and `@protocol` register under `"monitor"` and `"protocol"`, added to `inspect_ai`'s `RegistryType` by PR 0 ([inspect_ai#5514](https://github.com/UKGovernmentBEIS/inspect_ai/pull/5514), branch `feature/sentinel`). Sentinel's `inspect-ai` git ref points at that branch until it merges.
- **Tool stages only.** `BeforeToolCall` and `AfterToolCall` with the design's full fields; `Step` is their union. The generate stages arrive with the generate-side dispatcher. The schema may change freely until there are users.
- **Recording is the runner's, through a separate interface.** `Host` is the author-facing ABI (`generate` only for now). `Recorder` has four methods: `record`, called by the runner for every participating child; `cancelled`, called for a child cancelled before it reported; `bypassed`, called for a protocol a descendant's `decide_final()` ended the step past; and `superseded`, called for a `decide_final()` decision that lost a race to another in the same layer or was outranked by an exception there. This departs from `sentinel-deployment.md`, whose `Host` ABI carries `record`; that doc is to be updated.
- **Two context types.** `Context` is what authors see. `RunnerContext(Context)` adds `recorder` and `child()`; the dispatcher builds it, `dataclasses.replace` preserves it down the layers, and the runner raises `TypeError` if handed a bare `Context`.
- **Author surface vs integration surface.** `__init__` exports only what an author writes against, including `Reported` and `Report`, since `step.escalations` and the runner's results are typed with them. `inspect_sentinel/_integration.py` re-exports what the dispatcher needs (`RunnerContext`, `Recorder`, `validate_instance_name`, `step_types`, `resolve_sentinel`, `Sentinels`, `run_root`); `inspect_ai` imports from that module only, so refactors have one file to keep stable.
- **Shipped protocols:** `observe`, `concurrent`, `threshold`. `sequential` and `human` are follow-ups.

## Layout

```
src/inspect_sentinel/
  __init__.py       author-facing exports
  _step.py          BeforeToolCall, AfterToolCall, Step, Stage
  _report.py        Suspicion, Action, Observation, Decision, Report, Reported
  _context.py       Host, Recorder, Context, RunnerContext
  _monitor.py       Monitor, Protocol, Monitors, Protocols, Children, @monitor, @protocol
  _runner.py        Observations, Decisions, run_monitors, run_protocols,
                    run_children
  _check.py         validate_decision_shape
  _final.py         final, Final
  _protocols.py     observe, concurrent, threshold
  _resolve.py       resolve_sentinel
  _integration.py   dispatcher-facing re-exports
  _registry.py      inspect_ai entry point; imports _protocols so they register
```

## PR 1: types

`_step.py`, `_report.py`, `_context.py`. Data only.

```python
Stage = Literal["tool_call", "tool_result"]        # grows with the generate stages

@dataclass(frozen=True)
class BeforeToolCall:
    conversation: str
    message: str
    call: ToolCall
    view: ToolCallView
    input: list[ChatMessage]
    history: list[ChatMessage]
    escalations: tuple[Reported[Decision], ...] = ()

@dataclass(frozen=True)
class AfterToolCall:
    conversation: str
    message: str
    call: ToolCall
    result: ChatMessageTool
    output: ToolResult
    view: ToolCallView
    input: list[ChatMessage]
    history: list[ChatMessage]
    escalations: tuple[Reported[Decision], ...] = ()

Step = BeforeToolCall | AfterToolCall
```

`escalations` references `Reported`, so PR 1 defines it in `_report.py` rather than leaving it to the runner PR.

```python
Suspicion = float | dict[str, float]  # finite; a dict has at least one key
Action = Literal["continue", "modify", "reject", "terminate", "escalate"]

class Observation(BaseModel):
    suspicion: Suspicion
    explanation: str | None = None
    metadata: dict[str, Any] | None = None
    @classmethod
    def score(cls, suspicion, explanation=None) -> Observation
    @classmethod
    def flag(cls, explanation=None) -> Observation     # suspicion=1.0

class Decision(BaseModel):
    action: Action
    audit: bool = False
    modified: ToolCall | None = None                    # ChatMessageAssistant joins with AfterGenerate
    explanation: str | None = None                      # log-only; the agent never sees it
    message: str | None = None                          # what the agent reads; reject only, for now
    metadata: dict[str, Any] | None = None
    @classmethod proceed / reject(explanation, *, message) / terminate / escalate

Report = Observation | Decision

R_co = TypeVar("R_co", bound=Report, covariant=True)  # covariant so Reported[Decision] flows into Reported[Report]

@dataclass(frozen=True)
class Reported(Generic[R_co]):
    name: str
    path: str
    report: R_co
```

```python
class Host(typing.Protocol):
    async def generate(self, input: str | list[ChatMessage], *, model: str | None = None,
                       tools: list[ToolInfo] | None = None,
                       config: GenerateConfig | None = None) -> ModelOutput: ...

class Recorder(typing.Protocol):
    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None: ...
    def cancelled(self, context: Context, step: Step, name: str) -> None: ...
    def bypassed(self, context: Context, step: Step, name: str) -> None: ...
    def superseded(self, context: Context, step: Step, reported: Reported[Decision]) -> None: ...

@dataclass(frozen=True, kw_only=True)
class Context:
    task: str | None
    task_description: str | None
    sample_id: str | int | None
    epoch: int | None
    sample_description: str | None
    input: str | list[ChatMessage]
    metadata: dict[str, Any]
    path: str
    store: Store
    host: Host
    target: Target | None = None
    def store_as(self, model_cls: type[SMT]) -> SMT      # instance=self.path

@dataclass(frozen=True, kw_only=True)
class RunnerContext(Context):
    recorder: Recorder
    def child(self, name: str) -> RunnerContext          # path joined with "/"
```

Tests: report constructors set the fields the design says; `Observation` and `Decision` round-trip through pydantic JSON (they are serialized into `SentinelEvent`); `Decision.modified` accepted for `modify` and rejected otherwise is *not* checked here (that is the boundary check, PR 4); `store_as` namespaces by `path`; `child()` composes paths from an empty and a non-empty parent and returns a `RunnerContext`.

## PR 2: decorators and registration

`_monitor.py`. Stage inferred from the second parameter's annotation; kind is the decorator; return annotation must agree, checked at decoration time. `registry_tag`/`registry_add` from `inspect_ai._util.registry`, as `@approver` does. Sentinel's git ref moves to `feature/sentinel`.

Tests: stage inference for each payload and for `Step`; an unannotated second parameter is an error; `@monitor` on a function annotated `-> Decision | None` is an error, and vice versa; `registry_create("protocol", "no_curl")` resolves; factory params recorded in registry info.

Implemented: stages are read at factory-call time from the returned function's second-parameter annotation via `get_type_hints` and stored on the instance; `stages(instance)` exposes them to the runner and is integration surface. A union of stage payloads registers at every member. The return annotation must be the report type, optionally with `None`. `registry_create` is not a construction path for sentinels: it instantiates only factories whose return annotation's class name equals the registry type, and `Monitor`/`Protocol` are union aliases, so it returns the factory uncalled and `inspect_ai` deliberately has no `registry_create` overload for the two types, which makes such a call a type error. Configuration and replay construct through `create_registry_object`.

## PR 3: runner

`_runner.py`. Per child: skip when not annotated for the stage (indistinguishable from abstention); name from mapping key or registry name, duplicate within a layer is an error; derive the child context with `child()`; record every participating report through `context.recorder`; exceptions propagate. Plural forms fan out with anyio (own `tg_collect`, no inspect_ai dependency); `run_protocols` cancels siblings on `terminate`. `run_children` runs both families in one task group.

Tests use a list-backed `Recorder` and a stub `Host`. Cover: filtering, naming and the duplicate error, path derivation, every participating report recorded including ones the caller ignores, `max_suspicion()` over scalar and dict suspicion, `strongest()` precedence, sibling cancellation on `terminate`, a raising child fails the call, `TypeError` on a bare `Context`.

Implemented: the decorators store the accepted payload classes on each instance and `step_types(instance)` (runner-facing, not exported) reads them; the runner filters with `isinstance`, so there is no stage-name vocabulary in the filter. Children are named by mapping key or by `registry_unqualified_name`, since a package-qualified registry name (`acme/suspicion`) would put the path separator inside a segment; two packages' monitors sharing a leaf name in one layer therefore collide, and the duplicate-name error tells the author to name them with a mapping. `run_monitor` and `run_protocol` check the child's registry type; `run_monitors`, `run_protocols` and `run_children` all validate their children once in `named_children` and then fan out through one private `_run_named`, which owns the `terminate` cancellation. The task-group unwrap catches `ExceptionGroup` only, never `BaseExceptionGroup`, so a cancellation travelling with an error propagates intact, as in inspect_ai's `tg_collect`. `Observation` now rejects an empty structured suspicion, since `max_suspicion()` has no answer for it. Exceptions propagate unconditionally: the failure policy (warn and continue for a monitor nothing consumes, `fail="open"`) is not implemented here and is not PR 4's either; it needs a per-child hook in `_run_child`, because by the time an exception leaves the task group the siblings' results are gone, and it will be added with that hook in a later PR. A child cancelled before it reported, whether by a sibling's `terminate` or by cancellation from above, is recorded through `Recorder.cancelled()`, and a recorder that raises there fails the layer exactly as one raising from `record()` does, so no recorder bug is hidden behind a log line; cancellation lands at a child's next await, so one that finishes without awaiting is recorded normally. When several children fail concurrently only the first exception surfaces, and a child raising its own `ExceptionGroup` passes through as a group; both match inspect_ai's `tg_collect` and the deferred failure-policy hook will want the sibling errors.

## PR 4: protocols, boundary check, final decisions, resolve, integration

`_protocols.py`, `_check.py`, `_final.py`, `_resolve.py`, `_integration.py`, `_registry.py`, and the `inspect_ai` entry point in `pyproject.toml`.

- `observe`: records, returns `None`.
- `concurrent`: `run_children`; `strongest()`; a `modify` with more than one deciding child becomes `reject` naming the modifier.
- `threshold(monitors, reject_at, terminate_at=None)`: as written in `sentinel.md`.
- `validate_decision_shape(decision, step)` raises on the deterministic protocol bugs: an action illegal for the stage (`reject` and `modify` are `BeforeToolCall` only), or `modified` not set exactly when `action == "modify"`.
- `decide_final(decision)` ends the step: it raises `Final`, a `BaseException`, which the runner records as the calling protocol's decision after the shape check, then carries past every layer above, recording each through `Recorder.bypassed()` and cancelling siblings as `terminate` does. The dispatcher catches it at the root. It replaces the `binding` field and the floor the runner clamped to.
- `resolve_sentinel(spec)`: monitor or monitors-only collection to `observe`; anything else containing a protocol to `concurrent`; every configuration wrapped, so a lone protocol is `concurrent([protocol])` and a lone monitor `observe([monitor])`, and a lone child records the same paths as a list of one. `observe`, `concurrent` and `threshold` raise `ValueError` when given no children.

Tests: each protocol's rules above; every row of the resolve table; each shape rule raising; a `decide_final()` passing every layer above, recorded, bypassing and cancelling; two racing `decide_final()`s; an error beside a `decide_final()`; `decide_final()` from a monitor.

Implemented: both checks take the `Step` rather than a stage string, so the payload type stays the stage identity; the runner applies the shape check as each protocol returns or calls `decide_final()`, so a shape error names the leaf; `Final` carries its origin (the calling protocol's context, step and `Reported` decision) from its first catch; each layer it passes records itself as bypassed immediately, a `Final` that loses a race is recorded as `superseded` at the group where it lost, and the decision is recorded once, by `run_root`, when it takes effect, so a `Final` escaping `run_protocol` or `run_children` below the root is not yet recorded; nested exception groups, including a protocol's own task group, are flattened to their leaves and resolved by one policy in both places, so a protocol's own group mixing an error and final decisions surfaces the error and supersedes the decided ones, as a runner group does; a protocol or monitor whose own `decide_final()` fails supersedes the child decisions beside it before raising; a `Final` raised by a monitor's own body is a `TypeError` naming the monitor, since a monitor returns observations, while one made by a protocol the monitor ran propagates and records the monitor as bypassed; in a task group an exception outranks a `Final`, and of several `Final`s the first to arrive escapes and the rest are superseded; when an exception outranks them, every one is recorded as superseded; `resolve_sentinel` reuses the runner's `named_children` so duplicate names and uncalled factories fail at configuration time, classifies a single instance with `is_registry_object` so a set, an iterator or a string gets `named_children`'s message rather than being wrapped in a list, and warns through `logging` for a monitors-only configuration, naming each instance as it is configured, so an explicit `observe()` is how to say recording is intended; the three shipped protocols register as `inspect_sentinel/<name>` through the `inspect_ai` entry point in `pyproject.toml`; inspect_ai imports `RunnerContext`, `Recorder`, `validate_instance_name`, `step_types`, `resolve_sentinel`, `Sentinels` and `run_root` from `inspect_sentinel._integration` only; `run_root` records the resolved root as a layer at `path=""` under its unqualified registry name and returns a `decide_final()` decision as the outcome, so the dispatcher never catches `Final`; `concurrent` records every decision it makes, and when more than one protocol decided its explanation lists each one's decision after the winner's own; `validate_decision_shape` does not re-check `modified`'s type, which `Decision` validates. `threshold` is `BeforeToolCall`-only, as designed, because `reject` is not legal after a tool call.

Known limit: a `terminate` outrun by a sibling's `decide_final()` inside a protocol's own task group, rather than `run_children` or `run_protocols`, is not marked superseded; its record stands as written.

## PR 5: multi-function factories

**Multi-function monitors (agreed 2026-09-25).** A `@monitor` or `@protocol` factory may return a sequence of functions instead of one. The group is one configured instance: one instance name, one `path`, one child context, and so one `store_as` namespace that every member shares; that is the point, since accumulating across stages is the trajectory-score case the design argues for and closure state is wrong for it (created once per configuration, shared across samples). Members are told apart by the inner function's `__name__`, recorded as a `function` field on `Reported` and `SentinelEvent`; path plus function identifies a report, and read-mode validation keys on both. Duplicate function names within one factory are a configuration error, checked when the factory is called. The decorators validate each member and record the union of their payload types; the runner treats the group as one child and dispatches to every member that accepts the step, so one instance may contribute several reports at a step, which `Observations` and `Decisions` already accommodate as sequences. Each member still watches exactly one stage, so "One function, one point" stands; what changes is that a factory may emit several functions when they share state, which is the class-with-a-method-per-stage the design rejected, reached through functions so the static scan and per-function portability verdict still work. Two instances of one factory, and two samples, keep separate state as before. `sentinel.md` "One function, one point", "Instance names", and the `SentinelEvent` listing need updating.

Implemented: `run_monitor` and `run_protocol` are removed; `run_monitors`, `run_protocols` and `run_children` accept one instance, a sequence or a mapping and always return a possibly empty sequence, since one instance may now report more than once; `named_children` treats any callable as a single instance, so an uncalled factory or an undecorated function still gets its own message. A factory returning a sequence (not a string) gets a private group object, a `MonitorGroup` or `ProtocolGroup`, registry-tagged and carrying the union of its members' payload types, whose `__call__` is typed so it fits every `Monitor` or `Protocol` alias and raises `TypeError` if called directly. `_run_child` derives one child context and runs the members that accept the step through `_run_member`, which holds the per-function shape checks, the `decide_final()` handling and the recording; a group cancelled part way is recorded once, through `Recorder.cancelled`, at the instance level. The fan-out collects a group's reports as they are recorded, so a group cancelled after one member reported returns that report, as a lone child that finished first does; a member's `terminate` stops the group, since nothing outranks it: the members after it do not run, so none can override it with `decide_final()`, and sibling instances are cancelled as for a lone child. `Reported.function` is set for every report, a lone function's included. `threshold` requires every function of every monitor to watch `BeforeToolCall`, since it runs only there and a member watching another stage would never run. `run_root` rejects a group, since the root is what `resolve_sentinel` returns and a step has one outcome. `concurrent` counts votes by instance, so a group's `modify` is contested only by another instance's decision or by another of its own functions modifying, and its explanation labels a group member's vote `name.function`.

## S6: configuration

`_config.py`. `sentinel_from_config(config)` builds a sentinel from the YAML/JSON shape in `sentinel.md` "Configuration" through `create_registry_object`; `config_from_sentinel(sentinels)` is the inverse, for the log and `eval_retry`. Also drops the unused public `Stage` alias, since `SentinelEvent` declares its own stage literal.

Implemented: the shape is `SentinelConfig`, a pydantic `RootModel` over a list of `SentinelEntry` or a mapping of instance names to them; an entry has `name` and `params`, and any other key is kept as a pydantic extra validated as a nested `SentinelConfig`, so the model serializes to exactly the YAML shape and `entry.nested` returns the nested layers. `sentinel_from_config` accepts a file whose only key is `sentinel`, a bare registered name (a list of one), the model or the plain list or mapping; it walks the plain shape itself rather than through pydantic, so each error names the entry path (`sentinel.attempt.children[1]`), and a `TypeError` or `ValueError` from a factory is re-raised with the path prepended. A nested key is legal exactly when the factory's signature has that parameter, and may not also appear in `params`; there is no `tools` key. Names are looked up as both monitor and protocol, and `@monitor` and `@protocol` refuse a name already registered as the other kind, so at most one matches; a bare name that matches nothing exactly is then tried as `inspect_sentinel/<name>`, as `registry_lookup` does for `inspect_ai/`, so a local factory shadows a shipped one of the same name without making either unrebuildable. A file is parsed as JSON and otherwise as YAML (PyYAML rejects tab-indented JSON), with mappings loaded as pairs so a repeated key is an error naming the file and entry path. A factory with `**kwargs` accepts any nested key, since `registry_tag` records those arguments under their own names. The result is the list or mapping of instances, unresolved. `config_from_sentinel` reads `registry_value` of each instance, whose params already hold nested instances as registry dicts, and writes a non-empty list or mapping of monitor or protocol registry dicts as nested entries, except under a parameter named `name` or `params`, which stays in `params` in registry-dict form; a lone instance is recorded as a list of one, and a package name is written bare when the bare name finds it unambiguously. Both functions and both models are exported from `_integration`, not the author-facing `__init__`: authors write Python or YAML, and the model is for inspect_ai's log and CLI.

## Design sync

The canonical docs on inspect_ai's `design/monitor` branch were synced with this series on 2026-09-28 (inspect_ai 12f555a65): `decide_final()`, the runner-side `Recorder`, `Context.path`, `RunnerContext`, the plural-only runner, and multi-function factories. The copies in `design/` match them; edit both together.

## S7: the factory name for the log

`SentinelEvent.name` is the registry name of the factory and `path` the instance path, but the `Recorder` had no way to learn the registry name: `Reported.name` is the instance name, `cancelled` and `bypassed` receive only the instance name, and `Context` carries no registry name, so for `{"attempt": chain(...)}` the dispatcher could only record `name="attempt"`, repeating the path's last segment.

Implemented: `RunnerContext` gains `factory: str`, the full registry name as `registry_info(child).name` gives it (package prefix included), defaulting to empty so the dispatcher's top-layer context need not know the root; `child(name, factory)` takes it and the runner passes each child's registry name when it derives the child's context; `run_root` sets it on the root's own context, so the root's decision, its `bypassed` record and any `decide_final()` origin at the root carry it. The `Recorder` methods take a `RunnerContext` rather than a `Context`, since the runner always passes one, so the dispatcher reads `context.factory` without a cast. A `terminate` superseded by a sibling's `decide_final()` is re-derived with its factory's name, as the other records are. `Reported` is unchanged: the instance name is what authors compare, and the factory is recording metadata.

## No outcome field

Decided with the user 2026-09-29: `SentinelEvent.outcome` is removed (inspect_ai c6d332ba6). What the layer above did with a report is the decision event at the parent path, so the field only duplicated the log. Which child a top-level result came from stays inferred by the viewer, exact after `decide_final()`.

## Threshold explanations

Implemented, agreed with the user 2026-09-29: a terminate names the highest-scoring monitor (first in configuration order on a tie) and carries its explanation: `suspicion 0.95 from audit: <reason>`. It previously said only `suspicion 0.95`. The reject at first carried no explanation, to keep the reasoning from the agent; superseded by [Decision.message and single-instance protocols](#decisionmessage-and-single-instance-protocols), which makes `explanation` log-only, so the reject now carries the same explanation.

## Decision.message and single-instance protocols

Implemented, agreed with the user 2026-09-29: `Decision` gains `message: str | None`, the only text the agent reads from a decision; `explanation` is log-only. `Decision.reject(explanation, *, message=None)` takes it keyword-only and the other constructors do not. The shape check allows `message` only on a `reject`, since "continue, but tell the model X" needs a way to deliver text while the step proceeds, which does not exist yet. `threshold`'s reject and terminate both carry `suspicion 0.95 from <name>: <reason>` and neither sets `message`, so the agent gets the host's default "not approved" text. `observe`, `threshold` and `concurrent` take a single instance as well as a sequence or mapping (`Monitor | Monitors`, `Monitor | Protocol | Children`), matching `run_monitors`, so `threshold(suspicion_monitor(), reject_at=0.8)` type-checks. The inspect_ai dispatcher must pass `message`, not `explanation`, to the agent.

## decide_final()

Renamed from `final()` with the user 2026-09-30: `final` shadowed `typing.final` and read as a decorator, while the function raises and ends the step. A verb says that nothing after it runs. The exception it raises is still the internal `Final`.

## Public names

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- The protocol alias is `Protocol`, not `ControlProtocol`: it pairs with `@protocol` as `Tool` pairs with `@tool`. The collision with `typing.Protocol` is accepted; the package imports typing's as `TypingProtocol`, and the docs say "a control protocol, not `typing.Protocol`" on first use. The decorator stays `@protocol`, the registry type `"protocol"`, and the plural `Protocols`.
- `Decision.clear()` is `Decision.proceed()`, which says what happens to the step rather than what the protocol concluded; it still produces `continue`.
- `Reports`, what `run_children` returns, is a `NamedTuple` of `observations` and `decisions`, so `observations, decisions = await run_children(...)` works beside named access.
- The planned ordered composition is `sequential`, not `chain`: it pairs with `concurrent` and does not collide with `inspect_ai.solver.chain`. It is not implemented yet.
- `Reported` keeps its name. It was reviewed as a possible verb form of `Report`; read as an adjective on its type parameter, like `typing.Annotated`, `Reported[Decision]` is a reported decision: the decision together with the identity of the instance that reported it. `ChildReport` and `Attributed` were considered.
- Vocabulary: a *step* is the payload a monitor receives, a *stage* is its type (`BeforeToolCall`, `AfterToolCall`, and the generate stages), and "a point in the loop" is only an informal gloss on stage. `Context` and `Step` are unrelated to inspect_ai's `StepEvent` and `step()`. A monitor judges a call against `step.input`, exactly what the model was sent; `context.input` is the sample's input, the assignment.

## Explicit groups, decorator arguments and the call guard

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- A multi-function factory returns `MonitorGroup(before, after)` or `ProtocolGroup(a, b)`, both exported, instead of a list. They are final classes holding the functions in order with no `__call__`, so pyright rejects calling a group; construction checks that there is at least one function and that `__name__`s are distinct, and the decorator's `_register` still validates each member (stage, return annotation, kind), since it knows the kind: a `MonitorGroup` from `@protocol`, or the reverse, is a `TypeError`, and so is a plain list or tuple, with a message saying to return the group. `@monitor` is overloaded so a factory returning `Monitor` gives `Callable[P, Monitor]` and one returning `MonitorGroup` gives `Callable[P, MonitorGroup]`; `@protocol` likewise. `Monitors`, `Protocols`, `Children`, `Sentinels`, the runners' parameters and the shipped protocols' parameters name the groups beside the functions. Runtime behaviour is unchanged.
- `@monitor(name=..., version=...)` and `@protocol(name=..., version=...)` beside the bare form, with typed overloads. `name` overrides the registered name as `@solver(name=)` does; `version` is an `int`, default 0 as for Inspect Scout's scanners, stored under `version` in the registry info's metadata for calibration to record. `portable=` and `fail=` are future arguments on this form.
- A configured function called directly while a runner is invoking a sentinel raises `RuntimeError` ("call children through run_monitors/run_protocols/run_children"). The decorator wraps each inner function in a guard with `functools.wraps`, so `__name__`, `__wrapped__`, the signature and the annotations are preserved; the runner sets a context variable to the function it is about to invoke, and the guard sets it to `None` for the body, so a direct call from inside a body fails and a call through the runner (which sets it again, in its own task) passes. The variable defaults to "no run", so a direct call outside any run, as in a unit test, works.
- The docs say "annotate `-> Observation`, or `-> Observation | None` if it can abstain" (and the same for `Decision`), since both annotations are accepted.

Implemented: the old private `Group`/`MonitorGroup`/`ProtocolGroup` with a raising `__call__` is gone; one internal base, `Group`, remains for `isinstance` checks and holds `functions`. Each member's accepted stages are recorded on its guarded function and `members()` reads them from there. `named_children` treats a group as one instance, as it does a callable.

## Out of scope

The failure policy (`design/sentinel.md`, "Failure semantics"): the runner propagates every exception until a later PR adds the per-child hook described under PR 3.

Generate stages, `sequential`, `human`, `defer_to_trusted`, `resample`, views and rendering helpers, `store_as(scope=)`, `fetch`/`terminate` on `Host`, the `inspect_ai` dispatcher and `SentinelEvent`.
