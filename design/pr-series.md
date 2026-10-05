# Implementation: the first PR series

How `sentinel.md` becomes code, in an order where each PR is reviewable alone, has its own tests, and leaves `main` importable. Agreed 2026-09-21; PR 0 opened 2026-09-22.

## Goal

The minimal surface that lets `inspect_ai` connect: the types a dispatcher constructs, the decorators an author writes against, the runner a protocol calls, and the three protocols the compile rule needs. No `inspect_ai` dispatcher yet; that is the next series, on inspect_ai's `feature/sentinel` branch.

## Decisions

- **Surface only.** Sentinel ships types and functions; `inspect_ai` wiring comes after. Verified by unit tests with fake `Host` and `Recorder`.
- **Real registration.** `@monitor` and `@protocol` register under `"monitor"` and `"protocol"`, added to `inspect_ai`'s `RegistryType` by PR 0 ([inspect_ai#5514](https://github.com/UKGovernmentBEIS/inspect_ai/pull/5514), branch `feature/sentinel`). Sentinel's `inspect-ai` git ref points at that branch until it merges.
- **Tool stages only.** `BeforeToolCall` and `AfterToolCall` with the design's full fields; `Step` is their union. The generate stages arrive with the generate-side dispatcher. The schema may change freely until there are users.
- **Recording is the runner's, through a separate interface.** `Host` is the author-facing ABI (`generate` only for now). `Recorder` has four methods: `record`, called by the runner for every participating child; `cancelled`, called for a child cancelled before it reported; `bypassed`, called for a protocol a descendant's `decide_final()` ended the step past; and `superseded`, called for a `decide_final()` decision that lost a race to another in the same layer or was outranked by an exception there. This departs from the earlier `sentinel-deployment.md`, whose `Host` ABI carried `record`; that doc now says recording is not on the ABI.
- **Two context types.** `Context` is what authors see. `RunnerContext(Context)` adds `recorder` and `child()`; the dispatcher builds it, `dataclasses.replace` preserves it down the layers, and the runner raises `TypeError` if handed a bare `Context`.
- **Author surface vs integration surface.** `__init__` exports only what an author writes against, including `Reported` and `Report`, since `step.escalations` and the runner's results are typed with them. `inspect_sentinel/_integration.py` re-exports what the dispatcher needs (`RunnerContext`, `Recorder`, `validate_instance_name`, `step_types`, `resolve_sentinel`, `Sentinels`, `run_sentinel`, and since S6 `sentinel_from_config` and `config_from_sentinel`); `inspect_ai` imports from that module only, so refactors have one file to keep stable.
- **Shipped protocols:** `observe`, `concurrent`, `threshold`. `sequential` and `human` are follow-ups.

## Layout

```
src/inspect_sentinel/
  __init__.py       author-facing exports
  _step.py          BeforeToolCall, AfterToolCall, Step
  _report.py        Suspicion, Action, Observation, Decision, Report, Reported
  _context.py       Context, validate_instance_name
  _host.py          Host, HumanAnswer, Recorder, HostContext, the step context variable
  _types.py         Monitor, Protocol, Monitors, Protocols, Sentinel, Sentinels,
                    MonitorGroup, ProtocolGroup
  _decorators.py    @monitor, @protocol, registration, signature validation,
                    the direct-call guard, step_types
  _results.py       Observations, Decisions, Reports
  _runner.py        run_monitors, run_protocols, run_children, run_sentinel
  _validate.py      run-time checks: validate_decision_shape, per-child checks,
                    named_children
  _final.py         decide_final, Final
  _protocols/       observe, concurrent, threshold, one module each
  _resolve.py       resolve_sentinel
  _config.py        sentinel_from_config, config_from_sentinel
  _integration.py   the contract inspect_ai imports
  _entrypoint.py    inspect_ai entry point; imports _protocols so they register
```

The layout as of [Module layout](#module-layout); the PR sections below name the modules as they were when each landed.

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

Implemented: both checks take the `Step` rather than a stage string, so the payload type stays the stage identity; the runner applies the shape check as each protocol returns or calls `decide_final()`, so a shape error names the leaf; `Final` carries its origin (the calling protocol's context, step and `Reported` decision) from its first catch; each layer it passes records itself as bypassed immediately, a `Final` that loses a race is recorded as `superseded` at the group where it lost, and the decision is recorded once, by `run_sentinel`, when it takes effect, so a `Final` escaping `run_protocol` or `run_children` below the root is not yet recorded; nested exception groups, including a protocol's own task group, are flattened to their leaves and resolved by one policy in both places, so a protocol's own group mixing an error and final decisions surfaces the error and supersedes the decided ones, as a runner group does; a protocol or monitor whose own `decide_final()` fails supersedes the child decisions beside it before raising; a `Final` raised by a monitor's own body is a `TypeError` naming the monitor, since a monitor returns observations, while one made by a protocol the monitor ran propagates and records the monitor as bypassed; in a task group an exception outranks a `Final`, and of several `Final`s the first to arrive escapes and the rest are superseded; when an exception outranks them, every one is recorded as superseded; `resolve_sentinel` reuses the runner's `named_children` so duplicate names and uncalled factories fail at configuration time, classifies a single instance with `is_registry_object` so a set, an iterator or a string gets `named_children`'s message rather than being wrapped in a list, and warns through `logging` for a monitors-only configuration, naming each instance as it is configured, so an explicit `observe()` is how to say recording is intended; the three shipped protocols register as `inspect_sentinel/<name>` through the `inspect_ai` entry point in `pyproject.toml`; inspect_ai imports `RunnerContext`, `Recorder`, `validate_instance_name`, `step_types`, `resolve_sentinel`, `Sentinels` and `run_sentinel` from `inspect_sentinel._integration` only; `run_sentinel` records the resolved root as a layer at `path=""` under its unqualified registry name and returns a `decide_final()` decision as the outcome, so the dispatcher never catches `Final`; `concurrent` records every decision it makes, and when more than one protocol decided its explanation lists each one's decision after the winner's own; `validate_decision_shape` does not re-check `modified`'s type, which `Decision` validates. `threshold` is `BeforeToolCall`-only, as designed, because `reject` is not legal after a tool call.

Known limit: a `terminate` outrun by a sibling's `decide_final()` inside a protocol's own task group, rather than `run_children` or `run_protocols`, is not marked superseded; its record stands as written.

## PR 5: multi-function factories

**Multi-function monitors (agreed 2026-09-25).** A `@monitor` or `@protocol` factory may return a sequence of functions instead of one. The group is one configured instance: one instance name, one `path`, one child context, and so one `store_as` namespace that every member shares; that is the point, since accumulating across stages is the trajectory-score case the design argues for and closure state is wrong for it (created once per configuration, shared across samples). Members are told apart by the inner function's `__name__`, recorded as a `function` field on `Reported` and `SentinelEvent`; path plus function identifies a report, and read-mode validation keys on both. Duplicate function names within one factory are a configuration error, checked when the factory is called. The decorators validate each member and record the union of their payload types; the runner treats the group as one child and dispatches to every member that accepts the step, so one instance may contribute several reports at a step, which `Observations` and `Decisions` already accommodate as sequences. Each member still watches exactly one stage, so "One function, one point" stands; what changes is that a factory may emit several functions when they share state, which is the class-with-a-method-per-stage the design rejected, reached through functions so the static scan and per-function portability verdict still work. Two instances of one factory, and two samples, keep separate state as before. `sentinel.md` "One function, one point", "Instance names", and the `SentinelEvent` listing need updating.

Implemented: `run_monitor` and `run_protocol` are removed; `run_monitors`, `run_protocols` and `run_children` accept one instance, a sequence or a mapping and always return a possibly empty sequence, since one instance may now report more than once; `named_children` treats any callable as a single instance, so an uncalled factory or an undecorated function still gets its own message. A factory returning a sequence (not a string) gets a private group object, a `MonitorGroup` or `ProtocolGroup`, registry-tagged and carrying the union of its members' payload types, whose `__call__` is typed so it fits every `Monitor` or `Protocol` alias and raises `TypeError` if called directly. `_run_child` derives one child context and runs the members that accept the step through `_run_member`, which holds the per-function shape checks, the `decide_final()` handling and the recording; a group cancelled part way is recorded once, through `Recorder.cancelled`, at the instance level. The fan-out collects a group's reports as they are recorded, so a group cancelled after one member reported returns that report, as a lone child that finished first does; a member's `terminate` stops the group, since nothing outranks it: the members after it do not run, so none can override it with `decide_final()`, and sibling instances are cancelled as for a lone child. `Reported.function` is set for every report, a lone function's included. `threshold` requires every function of every monitor to watch `BeforeToolCall`, since it runs only there and a member watching another stage would never run. `run_sentinel` rejects a group, since the root is what `resolve_sentinel` returns and a step has one outcome. `concurrent` counts votes by instance, so a group's `modify` is contested only by another instance's decision or by another of its own functions modifying, and its explanation labels a group member's vote `name.function`.

## S6: configuration

`_config.py`. `sentinel_from_config(config)` builds a sentinel from the YAML/JSON shape in `sentinel.md` "Configuration" through `create_registry_object`; `config_from_sentinel(sentinels)` is the inverse, for the log and `eval_retry`. Also drops the unused public `Stage` alias, since `SentinelEvent` declares its own stage literal.

Implemented: the shape is `SentinelConfig`, a pydantic `RootModel` over a list of `SentinelEntry` or a mapping of instance names to them; an entry has `name`, `params`, an optional `version` and an optional `meta` (see [Versions in the log](#versions-in-the-log)), and any other key is kept as a pydantic extra validated as a nested `SentinelConfig`, so the model serializes to exactly the YAML shape and `entry.nested` returns the nested layers. `sentinel_from_config` accepts a file whose only key is `sentinel`, a bare registered name (a list of one), the model or the plain list or mapping; it walks the plain shape itself rather than through pydantic, so each error names the entry path (`sentinel.attempt.children[1]`), and a `TypeError` or `ValueError` from a factory is re-raised with the path prepended. A nested key is legal exactly when the factory's signature has that parameter, and may not also appear in `params`; there is no `tools` key. Names are looked up as both monitor and protocol, and `@monitor` and `@protocol` refuse a name already registered as the other kind, so at most one matches; a bare name that matches nothing exactly is then tried as `inspect_sentinel/<name>`, as `registry_lookup` does for `inspect_ai/`, so a local factory shadows a shipped one of the same name without making either unrebuildable. A file is parsed as JSON and otherwise as YAML (PyYAML rejects tab-indented JSON), with mappings loaded as pairs so a repeated key is an error naming the file and entry path. A factory with `**kwargs` accepts any nested key, since `registry_tag` records those arguments under their own names. The result is the list or mapping of instances, unresolved. `config_from_sentinel` reads `registry_value` of each instance, whose params already hold nested instances as registry dicts, and writes a non-empty list or mapping of monitor or protocol registry dicts as nested entries; `@monitor` and `@protocol` refuse a factory with a parameter named `name`, `params`, `version` or `meta`, the entry's own keys, so only a `**kwargs` argument of one of those names reaches it, and that stays in `params` in registry-dict form; a lone instance is recorded as a list of one, and a package name is written bare when the bare name finds it unambiguously. Both functions and both models are exported from `_integration`, not the author-facing `__init__`: authors write Python or YAML, and the model is for inspect_ai's log and CLI.

## Design sync

Decided with the user 2026-10-01: these documents in `design/` are canonical. They were drafted on inspect_ai's `design/monitor` branch and kept in sync with it by hand until then; that branch has a final sync and a note pointing here, and is no longer updated. Edit only the copies in this repository.

## S7: the factory name for the log

`SentinelEvent.name` is the registry name of the factory and `path` the instance path, but the `Recorder` had no way to learn the registry name: `Reported.name` is the instance name, `cancelled` and `bypassed` receive only the instance name, and `Context` carries no registry name, so for `{"attempt": chain(...)}` the dispatcher could only record `name="attempt"`, repeating the path's last segment.

Implemented: `RunnerContext` gains `factory: str`, the full registry name as `registry_info(child).name` gives it (package prefix included), defaulting to empty so the dispatcher's top-layer context need not know the root; `child(name, factory)` takes it and the runner passes each child's registry name when it derives the child's context; `run_sentinel` sets it on the root's own context, so the root's decision, its `bypassed` record and any `decide_final()` origin at the root carry it. The `Recorder` methods take a `RunnerContext` rather than a `Context`, since the runner always passes one, so the dispatcher reads `context.factory` without a cast. A `terminate` superseded by a sibling's `decide_final()` is re-derived with its factory's name, as the other records are. `Reported` is unchanged: the instance name is what authors compare, and the factory is recording metadata.

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
- The planned ordered composition is `sequential`, not `chain`: it pairs with `concurrent` and does not collide with `inspect_ai.solver.chain`. It shipped 2026-10-01 (see "sequential()").
- `Reported` keeps its name. It was reviewed as a possible verb form of `Report`; read as an adjective on its type parameter, like `typing.Annotated`, `Reported[Decision]` is a reported decision: the decision together with the identity of the instance that reported it. `ChildReport` and `Attributed` were considered.
- Vocabulary: a *step* is the payload a monitor receives, a *stage* is its type (`BeforeToolCall`, `AfterToolCall`, and the generate stages), and "a point in the loop" is only an informal gloss on stage. `Context` and `Step` are unrelated to inspect_ai's `StepEvent` and `step()`. A monitor judges a call against `step.input`, exactly what the model was sent; `context.input` is the sample's input, the assignment.

## Explicit groups, decorator arguments and the call guard

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- A multi-function factory returns `MonitorGroup(before, after)` or `ProtocolGroup(a, b)`, both exported, instead of a list. They are final classes holding the functions in order with no `__call__`, so pyright rejects calling a group; construction checks that there is at least one function and that `__name__`s are distinct, and the decorator's `_register` still validates each member (stage, return annotation, kind), since it knows the kind: a `MonitorGroup` from `@protocol`, or the reverse, is a `TypeError`, and so is a plain list or tuple, with a message saying to return the group. `@monitor` is overloaded so a factory returning `Monitor` gives `Callable[P, Monitor]` and one returning `MonitorGroup` gives `Callable[P, MonitorGroup]`; `@protocol` likewise. `Monitors`, `Protocols`, `Sentinels`, the runners' parameters and the shipped protocols' parameters name the groups beside the functions. Runtime behaviour is unchanged.
- `@monitor(name=..., version=...)` and `@protocol(name=..., version=...)` beside the bare form, with typed overloads. `name` overrides the registered name as `@solver(name=)` does; `version` is an `int`, default 0 as for Inspect Scout's scanners, stored under `version` in the registry info's metadata for calibration to record. `portable=` and `fail=` are future arguments on this form. `portable=` will default to True and trigger the runtime portability check ([workstreams.md](workstreams.md), workstream 11).
- A configured function called directly while a runner is invoking a sentinel raises `RuntimeError` ("call children through run_monitors/run_protocols/run_children"). The decorator wraps each inner function in a guard with `functools.wraps`, so `__name__`, `__wrapped__`, the signature and the annotations are preserved; the runner sets a context variable to the function it is about to invoke, and the guard sets it to `None` for the body, so a direct call from inside a body fails and a call through the runner (which sets it again, in its own task) passes. The variable defaults to "no run", so a direct call outside any run, as in a unit test, works.
- The docs say "annotate `-> Observation`, or `-> Observation | None` if it can abstain" (and the same for `Decision`), since both annotations are accepted.

Implemented: the old private `Group`/`MonitorGroup`/`ProtocolGroup` with a raising `__call__` is gone; one internal base, `Group`, remains for `isinstance` checks and holds `functions`. Each member's accepted stages are recorded on its guarded function and `members()` reads them from there. `named_children` treats a group as one instance, as it does a callable.

## References, per-dimension thresholds

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- `Observation` and `Decision` gain `references: list[Reference]`, reusing `inspect_ai.scorer.Reference` (the type Scout's `Result.references` uses), so Inspect View can link cites such as `[M22]` in an explanation. Every named constructor takes `references=` as a keyword. `threshold` copies the references of the monitor its explanation names onto its decision. inspect_ai's `SentinelEvent` records them; empty for `cancelled` and `bypassed`.
- `threshold`'s `reject_at` and `terminate_at` take a float or a mapping keyed by `"<monitor>.<dimension>"`, `"<dimension>"`, a scalar monitor's instance name, or `"*"`; the most specific key wins, unlisted scores never trigger without `"*"`, and a float is `{"*": float}`, normalized at configuration, so a float threshold's decisions and explanations are unchanged. Any score reaching its threshold acts, terminate first. Configuration checks well-formed keys, that a dotted key names a configured monitor (so a dimension containing `.` must be qualified), finite values, and terminate above reject wherever both cover a score. A crossing through a dimension's own key is explained `sabotage 0.72 ≥ 0.60 from audit: ...`; through `"*"`, or for a scalar monitor, as before.
- `audit` is documented as recorded for offline analysis and shown in the viewer as a flag; nothing acts on it yet.
- The docs say why `explanation` is log-only though `Approval.explanation` is shown to the model, and give the `modify` pattern with `dataclasses.replace(step.call, arguments=...)`.

## Context and Host

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- The `Context` docstrings say what fills: `task_description` and `sample_description` are None until inspect_ai has `Task(description=)` and `Sample(description=)`, and `target` is None until the `target=True` opt-in exists (both under [Deferred](#deferred)). `Context.store` is the whole sample store, the agent's state included, and is not namespaced; `store_as` is the namespaced way in.
- `store_as` is keyed by the instance path, so renaming a mapping key or wrapping a monitor in another layer moves its state. The docs say so; there is no `namespace=` override.
- `Host.generate(input, *, model: str | Model | None = None, role: str | None = None, tools=None, config=None)`. `model` is always a model (a name or an instance), `role` always a role, and passing both is an error the host raises. With neither, the host uses the `monitor` role; if that role is not configured, it falls back to the agent's model and logs a warning once per eval naming the role to set. LLM monitors expose `model=` and `role=` on their factory, `role` defaulting to `"monitor"`, and pass them through. `defer_to_trusted` and `resample` will use a `trusted` role, so a task declares `monitor` and `trusted` separately with `Task(model_roles=...)` or `--model-role`. The inspect_ai `Host` implementation follows in inspect_ai.
- The docs no longer call `context.host` the only route out of a monitor. Outbound HTTP is planned as `fetch` through named endpoints (`sentinel-deployment.md`); meanwhile an in-process monitor may call inspect_ai APIs directly, such as `sandbox()` or `logging`, at the cost of portability to a proxy, which a future `portable=False` will declare.

## Root and escalate

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility. This replaces PR 4's "every configuration wrapped" rule.

- **A lone protocol is the root, unwrapped.** `resolve_sentinel` returns a single `Protocol` as the root itself, so `Task(sentinel=threshold(suspicion(), ...))` records `threshold` at the empty path and its monitor at `suspicion`. A list or mapping still resolves to `concurrent(...)`, and monitors alone to `observe(...)` with the warning. A lone `ProtocolGroup` is wrapped in `concurrent`: the root returns the step's one outcome, `run_sentinel` runs one function, and combining several functions' decisions is `concurrent`'s job. The root records its own decision at the empty path, as every layer does. Configuration follows: `SentinelConfig` accepts one entry, `config_from_sentinel` records a lone instance as one entry, and `sentinel_from_config` builds one entry, or a bare registered name, as a lone instance, so a retry records the same paths. A mapping is one entry when its `name` is a string and a mapping of instance names when every value is an entry; anything else is an error naming the layer.
- **`escalate` outranks `continue`.** The precedence is `terminate > reject > modify > escalate > continue`, in `PRECEDENCE`, `Decisions.strongest()` and `concurrent`. A peer's `continue` cannot override an `escalate`, a stronger decision still wins, and a layer whose strongest decision is `escalate` returns `escalate`, its explanation listing the votes when several decided. An `escalate` does not contest a `modify`. `sequential`, when it lands, returns the last escalate when every deciding link escalated, so it passes up the same way.
- **A top-level escalate proceeds.** When the root's decision is `escalate`, nobody is left to hand it to: the inspect_ai dispatcher proceeds as for `continue`, the escalate stays recorded as the root's decision, and the host warns once per eval that `sequential([..., human()])` sends escalations to a person. `run_sentinel` returns the escalate; the host decides what an unresolved escalate means.
- **Runners stay three.** `run_children` runs anything; `run_monitors` and `run_protocols` are typed shortcuts whose parameter and return types catch misuse. Docs only.

## Shared types

Decided with the user 2026-09-30, before the first release, so nothing is kept for compatibility:

- `Action` and `Suspicion` are defined once, in inspect_ai, as `SentinelAction` and `SentinelSuspicion` exported from `inspect_ai.event` beside `SentinelEvent`. inspect_sentinel re-exports them under its own names, `Action` and `Suspicion`, with the same validation. inspect_ai's drift test that compared the two copies is gone.
- `SentinelEntry` and `SentinelConfig` move into inspect_ai (`inspect_ai._sentinel._config`, exported from `inspect_ai.log`), with the parsing of one entry, a list or a mapping. `EvalConfig.sentinel` is typed `SentinelConfig | None`, and logs load without inspect_sentinel. `_config.py` imports them from there; `sentinel_from_config` and `config_from_sentinel` stay in inspect_sentinel.
- `SentinelEvent.kind` is split in two. `kind` is the report family (`observation` or `decision`), and `status` says what happened to the report (`reported`, `cancelled`, `bypassed` or `superseded`). A `cancelled` event's kind comes from the registry type of the layer's factory. A `bypassed` event is always a decision, and only a decision can be `superseded`.

## Module layout

Restructured 2026-09-30, as a pure move and rename with no change in behaviour beyond the two noted:

| Before | After |
|---|---|
| `_monitor.py` | `_types.py` (callable aliases, groups) and `_decorators.py` (decorators, registration, signature validation, the direct-call guard, `members`, `step_types`) |
| `_registry.py` | `_entrypoint.py`, and the `pyproject.toml` entry point with it |
| `_check.py`, the runner's `_check_child`, `named_children` | `_validate.py` |
| `_runner.py` | `_results.py` (`Observations`, `Decisions`, `Reports`, `peak_suspicion`) and `_runner.py` (execution, exception-group handling) |
| `_protocols.py` | `_protocols/` with `observe.py`, `concurrent.py`, `threshold.py` |

- The rule that a monitor and a protocol cannot share a name asks inspect_ai's `registry_has`, which does not load entry points, in place of a module-level map of registered kinds.
- The bare-monitor warning is one line naming every monitor nothing acts on, rather than one line per monitor.
- `_integration.py` says in its docstring that it is the contract inspect_ai depends on and changes in step with it. It no longer re-exports `SentinelConfig` and `SentinelEntry`; inspect_ai imports them from `inspect_ai.log`.
- Stage types stay an attribute on each configured instance rather than moving to registry metadata: per-member stages live in groups, and registry metadata would need a name-to-class mapping.

## One stage per function

Decided 2026-10-01. Each function a `@monitor` or `@protocol` factory returns handles exactly one stage, so no function dispatches on the payload type. A monitor annotates one payload class, as before. A protocol annotates one payload class, or `Step` when it is stage-agnostic and only forwards the step (`concurrent`, `observe`). Any other union is rejected when the factory is called, with an error naming a `MonitorGroup`/`ProtocolGroup` with one function per stage, or `Step`. Since `Step` is a union, a union of every stage written out equals it and is accepted; the check compares the annotated set against all stages and rejects a proper subset with more than one member.

## API friction fixes

Decided with the user 2026-10-01, before the first release, so nothing is kept for compatibility:

- `Host.generate` resolves `model` and `role` as inspect's `get_model(model=, role=)` does instead of raising when both are given: the role's model if that role is configured for the task or eval, otherwise `model`, otherwise the agent's model with the one-time warning. With neither, the role is `monitor`. An LLM monitor's factory takes `model=None, role="monitor"` and passes both straight through.
- `Context.input_text` is the sample input as one string: `input` itself if a string, else its messages' text joined with newlines.
- The examples pass a single instance where they mean one, `threshold(suspicion(), reject_at=0.8)`, rather than a one-element list.
- `concurrent` lists the votes in its explanation only when they are not all `continue`; an all-continue layer keeps the winner's own explanation, if any.

## sequential()

Shipped 2026-10-01, the first of the `sequential()` and `human()` series. `sequential(children)` takes what `concurrent` takes, annotates `Step`, and runs each child through `run_children` with a one-entry mapping, so each link is recorded under its own name and a link that does not watch the stage is not invoked. The first decision other than `escalate` is returned and later links do not run; an `escalate` is appended to `step.escalations` for the next link. The list starts empty for each run of the chain, so escalations do not cross layers. Monitors are recorded and fall through. All-escalate returns the last escalate, monitors only `continue`, nothing participating `None`. An empty children list is a `ValueError` when the factory is called; YAML nests the links under `children:`. With it, decided with the user 2026-10-01, `Child` and `Children` are replaced by `Sentinel` (a monitor, protocol or group) and `Sentinels` (one sentinel, or a sequence or mapping of them), both exported; `Sentinels` is what `Task(sentinel=)`, `concurrent`, `sequential` and `run_children` take. The check that a `human()` directly under `concurrent` is an error is not stubbed: with no `human()` it would be dead code, so it lands with `human()`.

## human()

Shipped 2026-10-01, the second of the series; the inspect_ai host implementation of `Host.ask_human` (the human approval surfaces extracted from `human_approver` into a shared helper, rendering escalations above the call and the result after it) follows in PR 3 on inspect_ai's `feature/sentinel`, and until then inspect_ai's host does not satisfy `Host`. `Host.ask_human(step, choices) -> HumanAnswer` is in the ABI: one method for every stage, the host rendering the step by its type, notifying the person and marking the sample as awaiting a person. `HumanAnswer` is a frozen dataclass of `decision`, `reason` and `modified`. `human(stages, choices=None)` is registered as `inspect_sentinel/human`:

- `stages` is required, `"tool_call"` and/or `"tool_result"`; an empty list, a bare string, a repeated, unknown or unsupported stage is a `ValueError` when the factory is called, naming the valid stages (leaving it out is Python's `TypeError`, which configuration reports with the entry named). One function per stage, `tool_call` and `tool_result`; a `ProtocolGroup` when both.
- Choices use inspect's approval words: default `approve`, `reject`, `terminate` before a call and `approve`, `terminate` after it. Given choices are checked against every chosen stage, so `reject` or `modify` with `"tool_result"` is a `ValueError`; so are an empty list and `escalate`.
- It asks whenever reached: no escalation filter, no `ask=` mode, no warning.
- `approve` maps to `Decision.proceed()`, `reject` to `Decision.reject(explanation=reason, message=reason)`, `terminate` to `Decision.terminate(explanation=reason)`, `modify` to a `modify` with the answer's call, held to the call's `id` and `function` by the shape check. A host answer outside the choices offered, other than `terminate`, or a `modified` call on anything but `modify`, is a `ValueError`; a host answers `reject` if offered, otherwise `terminate`, when the person ends the prompt without a choice.
- **The answer is an ordinary decision (decided by the maintainer in review, 2026-10-01).** The `final=` parameter is gone and `human()` does not call `decide_final()`. Inside a `sequential()` the answer ends that chain; above the chain the usual vote applies, so beside other protocols in a `concurrent()` it is one vote: a peer's `reject` outranks the person's `approve`, and the person's `reject` or `terminate` is not weakened.
- **A `human()` in a `concurrent()` is allowed (decided by the maintainer in review, 2026-10-01).** This replaces the placement check that raised a `ValueError` naming `sequential()`. A `human()` there, beside other protocols or at the root a top-level list or mapping resolves to, asks about every call it is reached on, which can be intended, as with no warning when nothing earlier watches a stage. The docs recommend `sequential()` for asking only on escalation.
- `@protocol` gained an overload for a factory annotated `-> Protocol | ProtocolGroup`, which `human()` is.

## Examples and guide for sequential() and human()

Shipped 2026-10-01, the fourth of the series. `examples/escalate_to_human.py` puts a rule, `triage()`, ahead of `human(stages=["tool_call"])` in a `sequential()`: it rejects `bash()` commands that mention `/etc`, escalates ones that delete files, and proceeds on everything else, non-`bash()` calls included, since a link that abstains passes the step on and the person would be asked about it. The sequential section of `docs/composition.qmd` and the humans section of `docs/final-decisions.qmd` teach it, including the inspect_ai surfaces' limits (the reason is the surface's fixed text; `modify` is not supported). No example runs unattended, since `human()` waits for a person.

## Versions in the log

Decided by the maintainer 2026-10-01: `SentinelEntry` has `version: int | None` (inspect_ai), `config_from_sentinel` records each factory's `version` when not 0, nested entries included, and `sentinel_from_config` warns once per factory and version pair when a recorded version differs from the installed one, and builds anyway. The model's `version` is a `StrictInt`, so `true`, `"3"` and `3.0` are rejected. `@monitor(version=)` and `@protocol(version=)` raise a `TypeError` naming the factory for a version that is not an `int` (a `bool` included) and a `ValueError` for a negative one.

Decided by the maintainer 2026-10-01: `version` stays a top-level key, and an entry reserves `meta`, an optional mapping of JSON values that readers keep but do not interpret, so a field added later does not break readers already released. Nothing writes `meta` yet: `config_from_sentinel` writes none and `sentinel_from_config` ignores it, neither a param nor nested. Since `name`, `params`, `version` and `meta` are the entry's own keys, the decorators refuse a factory with a parameter of one of those names, with a `TypeError` naming the factory and parameter; `*args` and `**kwargs` are exempt.

## Failure policy

Decided with the maintainer 2026-10-01; replaces the design's earlier "fail loud unless nothing consumes it" with `fail="open"`. `sentinel.md`, "Failure semantics", is the policy; what landed:

- **A monitor never fails the sample itself.** `_run_member` catches an `Exception` from a monitor function and returns a `Failed(name, path, function, error)` (exported) in place of the observation, recorded through a new `Recorder.failed(context, step, failed)`; the instance's other functions and its siblings run on. A cancellation (a `BaseException`) is recorded as cancelled, not failed, and so is an exception raised in a cancelled scope (the runner checkpoints before recording a failure, since some SDKs turn cancellation into their own error); a `LimitExceededError` propagates, including one among several errors from a monitor's own task group, since a cost or time limit reached in a monitor's model call ends the sample; a monitor returning a `Decision` or calling `decide_final()` still raises, as a bug the runner detects. A protocol's exception propagates as before.
- **`Observations` carries the failures.** `failed` is a tuple of `Failed`; while it is non-empty, iteration, indexing, `len()` (so `bool()`), `in` and `max_suspicion()` raise `MonitorFailedError` (exported, a `RuntimeError`), which lists each failed path, function and error and is chained to the first error. `succeeded` is the observations without the failures and always reads. `len()` raises too, since a protocol testing `if not observations:` would otherwise abstain when its only monitor failed, which is failing open.
- **Who handles it.** `observe()`, `concurrent()` and `sequential()` read only decisions, so they log a warning per failure, once per (instance path, exception type) per process, and continue; `sequential()` counts a failed monitor link as having taken part, so a chain of only monitors still returns `continue`. `threshold()` does not handle failures (decided), so the read raises and the sample fails. A custom protocol checks `failed` and reads `succeeded`.
- **No `fail=`.** Removed from the design as a future decorator argument; `portable=` stays.
- **Transcript.** inspect_ai's `SentinelEvent` gains `status="error"` (observations only, with `function` and no suspicion, action or references) and an `error: str | None` field holding the exception's type and message, set only on an error event, as `ModelEvent.error` carries a model call's. One event per failure, at the call. Inspect View shows it as a "Failed" row on the tool call's sentinel inset and in the event tree, with the error as its detail.

## Interface cleanup

Decided by the maintainer on 2026-10-02, before the first release, so nothing is kept for compatibility. One change in each repository, since `_integration.py` is the contract between them:

- **`Task(sentinel=)` requires a protocol.** `observe()` is renamed `observe_only()` (`inspect_sentinel/observe_only`); it records its monitors and returns None, as before. `resolve_sentinel` raises a `ValueError` for a monitor, a `MonitorGroup`, or a list or mapping of only monitors, naming each monitor and saying to wrap them in `threshold()` or `observe_only()`; the warning is gone. A list or mapping with at least one protocol still resolves to `concurrent()`, and monitors in it are recorded. inspect_ai's `resolve_sentinel_spec` now also resolves the root, so `Task(sentinel=)`, `eval(sentinel=)`, `eval_set(sentinel=)` and `--sentinel` raise when they are given, not when the task runs. A configuration recorded with `observe` no longer loads.
- **`Context` is immutable.** It was already a frozen dataclass; `Context.store` is removed, so the API no longer hands a monitor or protocol the agent's state; a convention, not isolation, since the `StoreModel` from `store_as()` carries its store. `store_as()` stays, backed by the sample store the host passes in `HostContext` (see "`Context` carries no runner state" below). Giving checks access to the sample store is a separate entry under [Deferred](#deferred), "Later".
- **`Context.input` is `Context.sample_input`, and `input_text` is `sample_input_text`.** Beside `step.input`, what the model was sent, the longer names say which input it is, and the property keeps the field's prefix so the two read as a pair.
- **`RunnerContext` is `HostContext`.** The related names were surveyed in both repositories:

| Name | Outcome | Reason |
|---|---|---|
| `RunnerContext` | `HostContext` | The host builds it and supplies what it adds (the recorder and the store); the runner only reads it. |
| inspect_ai `_context()` | `_host_context()` | It builds the `HostContext`, and `_context` also names inspect_ai's `_sentinel/_context.py`, which holds the active sentinel. |
| `Recorder` | kept | It says what it does; the host implements it beside `Host`, so a `Host` prefix adds nothing. |
| `_runner.py` | kept | It holds the runner (`run_sentinel`, `run_children`, `run_monitors`, `run_protocols`). |
| `_host.py` | added | It holds what the host implements or builds: `Host`, `HumanAnswer`, `Recorder` and `HostContext`, moved out of `_context.py`, and the runner's private step context variable, which keeps `Context` and `validate_instance_name`. |
| `run_root` | renamed `run_sentinel` | Decided by the maintainer on 2026-10-03. It pairs with `resolve_sentinel`: a host resolves the sentinel once and runs it at each step. The runner's "root" stays off the host surface. |
| `_integration.py` | kept | It is the contract inspect_ai imports, and re-exports `HostContext` and `Recorder` from `_host.py`. |
| inspect_ai `_Host`, `_Recorder` | kept | Each is the inspect_ai implementation of the protocol it is named for. |

- **`HostContext` wraps a `Context` instead of subclassing it.** Decided by the maintainer on 2026-10-03. `HostContext(context=..., recorder=...)` is what the host builds and passes to `run_sentinel(protocol, host_context, step)`. The runner's per-layer state is described in the next entry. `HostContext.child()` is replaced by a private runner helper that keeps `validate_instance_name`. The `Recorder` methods take the instance's `Context` and its factory's registry name, `record(context, factory, step, reported)` and likewise for `failed`, `cancelled`, `bypassed` and `superseded`, and `Final`'s origin carries the deciding protocol's `Context`. Why: the `Context` an author is given no longer hides a subclass carrying the recorder, the runner no longer recovers the `HostContext` with `isinstance` and `cast`, and the host contract is a `Context` wrapped in a `HostContext`.

- **`Context` carries no runner state.** Decided by the maintainer on 2026-10-03, replacing the private `_run` and `_store` fields of `Context`. `HostContext(context=..., recorder=..., store=...)`: the host supplies the sample store beside the recorder. `run_sentinel` sets a private context variable for the step holding the recorder, the store, and a mapping from each layer's path to its factory's registry name; it registers the root at path `""` and the runner registers each child's path before invoking it. `run_children`, `run_monitors` and `run_protocols` look up the `path` of the context they are given, and `store_as()` reads the store from the variable. Called outside a step, for example with a context kept after its protocol returned, they raise `RuntimeError`; with a context whose `path` is not a layer of the running step, `ValueError`. anyio copies context variables into the tasks a protocol starts, so a protocol that fans out with its own task group runs children as before, and two samples' steps do not see each other's state. Entries are not removed when a layer finishes: a path is one instance per step, and a second, different factory under a path already used in the step is a `ValueError`, since its records and `store_as()` state would be indistinguishable from the first's; the path is registered before the stage filter, so this fails at every stage, and the same instance run again under its path is allowed. No lock guards the mapping, since tasks run on one event loop and each access is one dict operation. Precedent: inspect_ai keeps per-sample and per-call state in context variables (`util/_store.py` `store()`, `util/_span.py` `_current_span_id`, `log/_transcript.py`, approval's `_tool_approver`), as does inspect_scout's `ObserveContext`. Why: `Context` carries nothing hidden, so an author's context is exactly the data it shows, and a context used outside its step, or one whose path is not a layer of the running step, fails loudly instead of recording somewhere unintended. A context whose path is another running layer's is not detected; the check is by path.

- **`run_root` is `run_sentinel`.** Decided by the maintainer on 2026-10-03; see the table above. Earlier entries in this document use the new name.

"The dispatcher" in inspect_sentinel's docstrings, where it meant whatever builds the context and invokes the root, is now "the host"; the design keeps "dispatcher" for inspect_ai's in-process host.

## Helpers for rules

Decided with the maintainer on 2026-10-03, and cut to this scope on 2026-10-04 (sentinel PR #49): helpers for rules that check tool calls, in `_rules.py` and exported from the top level. They are pure functions over inspect_ai types, with no runtime services, so they work in a proxy host too. The guide's section is "Helpers for Rules" on the Protocols page. On 2026-10-04 the maintainer cut everything that copied built-in tools' argument names or parsed shell and Python text, as surprising and fragile; see "Finding paths in shell and code text" under Deferred.

- **`call_text(call) -> str`.** Every string argument value, in the order the arguments were given, recursing into lists and dicts (a dict's values in insertion order), joined with newlines; empty strings and non-strings are skipped. It knows nothing about particular tools. An earlier version kept a table of inspect_ai's built-in tools and their text arguments, leaving out mode arguments such as `text_editor()`'s `command`; it was dropped because it copied argument names that inspect_ai can change, and a tool of the same name from elsewhere has other arguments. If per-tool knowledge is wanted later, derive it from the tool's schema, supplied by the host, rather than a table in sentinel.
- **`tool_matches(call, *patterns) -> bool`.** Exact names and explicit globs, matched with `fnmatch.fnmatchcase` against the whole name: `"bash"` does not match `bash_session`, `"bash*"` does, and a comma is not a separator. Decided 2026-10-04 for least surprise. An earlier version copied `ApprovalPolicy(tools=...)` (an implicit trailing `*` and comma splitting) with a test against `policy_approver`; the docstring now states the difference in one sentence.
- **`find_words(text, words, *, case_sensitive=True) -> list[str]`.** A word matches where it is not preceded or followed by a word character (`(?<!\w)` and `(?!\w)`, not `\b`, so words that begin or end with punctuation, such as `-delete`, work), and a space in a word matches any run of whitespace. Returns the words as given, without duplicates, in order of first appearance, so an explanation can name them.
- **`result_text(step) -> str`.** The result's text with non-text content as `[<type>]`, or `Error (<type>): <message>`, as inspect_ai's `view_with_result` renders it for human review.
- **`path_matches(path, patterns, *, cwd=None, home=None) -> bool`.** For a path held in a structured argument, such as a file tool's `path`. Decided by the maintainer on 2026-10-03: path helpers must be built on a well-tested library and be tested thoroughly. Matching uses `wcmatch.glob.globmatch` with `GLOBSTAR`, `DOTGLOB` (so `*` sees `.ssh`), `NODOTDIR` (no wildcard matches a `.` or `..` segment), `FORCEUNIX` and `CASE`; character classes work, and `\` escapes. The path is normalised by its text first: repeated slashes collapse and `.`/`..` resolve with `posixpath`; a relative path is joined to `cwd` when given. `home=` (absolute, normalised before use) expands a leading `~` in the path, `cwd` and patterns; without it `~` is literal. Patterns are not joined to `cwd`. A path can keep a `..` (relative without `cwd`, or `~/..` without `home`); it then matches only a pattern spelling the `..` out, so an allow-list rejects it. A pattern ending in `/**` also matches the directory itself (`/etc/**` matches `/etc`), and a pattern's repeated and trailing slashes and `.` segments are dropped; a pattern `..` resolves against a preceding segment that has no wildcard, and is otherwise kept. Tests: tables per behaviour, and Hypothesis properties, including agreement with `PurePosixPath.full_match` on 3.13+ outside the documented differences.
- **`path_resolves(path, *, cwd=None, home=None) -> bool`.** Whether the normalised path is absolute or under `~` and has no `..` segment, so a relative path without `cwd` does not resolve; `~user/x` and `file:` URLs never do. Otherwise the path is literal, as a file tool's argument is, so `$` and `*` are characters of a name. A deny-list escalates a path that does not resolve. Computing it needs the same normalisation as matching, so it is a helper rather than an idiom that reimplements it.
- **No public step builders.** `before_tool_call()` and `after_tool_call()` were public in an earlier revision for unit-testing rules; decided by the maintainer on 2026-10-05, they are test fakes in `tests/_fakes.py` instead (top-level test builders have no precedent in inspect_ai and only cover half of a test, with no `Context`). See "Testing helpers for authors" under Deferred.
- **Examples.** `no_network.py` and Getting Started use `call_text` and `find_words`; `nested.py` uses `call_text`. They check `call.function == "bash"`, which reads more plainly than a pattern there. `escalate_to_human.py`'s `triage()` uses `tool_matches`, `call_text` and `find_words` to reject `bash()` and `python()` calls that mention `/etc` and escalate ones with a delete word. The Protocols page's example applies `path_matches` and `path_resolves` to `text_editor()`'s `path` argument only.
- **Framing.** Decided with the maintainer on 2026-10-03, adjusted 2026-10-04: the docs no longer say rules catch mistakes and not an adversary, or that they are illustrative and easy to evade. They say rules are cheap and run without a model call; they match a call's text and structured arguments and cannot prove a call safe; anything that needs an understanding of shell or code goes to an LLM monitor, or is escalated, typically `sequential([rule(), llm_monitor(), human()])`.

## Deferred

Agreed work that waits for something else. Each entry says what unblocks it. Keep this list current: remove an entry when it lands. Grouped by priority, decided by the maintainer on 2026-10-02, in the order to take them.

### Next: interface changes

Do as soon as possible.

- **`Task(description=)` and `Sample(description=)` in inspect_ai.** No longer held. A separate inspect_ai PR against `main`, not part of the cleanup: `Task(description: str | None = None)`, recorded in the log header and carried by `task_with()` and eval sets; `Sample(description: str | None = None)`, recorded per sample, with `FieldSpec(description=)`. Decided: no default from the `@task` docstring, and a `description` column in `evals_df` and `samples_df`. With it, the schema and ts-mono types regenerated, round-trip tests, docs and a CHANGELOG entry; showing the descriptions in the viewer is a follow-up (below). Opened as inspect_ai #5667 (Fixes #5666), with ts-mono #733 for the types; `evals_df` names its column `task_description`, matching the other `task_*` columns. Then `feature/sentinel` fills `Context.task_description` and `Context.sample_description`, which until then are always None in an eval.

### Then: helpers for tool-stage rules

- **Views and prompt helpers** (`monitor_prompt` and the view helpers): designed, not yet built. Add with them a structured-verdict helper, since today a structured verdict takes a pydantic model, a `GenerateConfig` with `ResponseSchema` and `json_schema`, `model_validate_json` and `Observation.score`; and helpers for escaping untrusted agent text and laying out a trusted/untrusted prompt, which `examples/llm_suspicion.py` writes by hand. Raised by the maintainer on 2026-10-02: views of the messages (for example the last six); the model-facing helpers are [workstreams.md](workstreams.md), workstream 3. The message views come first. The rule helpers, including the call's arguments as text, are done (see "Helpers for rules" above).
- **Shared message rendering, then the prompt helpers. Planned for 2026-10-05, with the generate stages, after the tool-call work lands.** Sentinel PR #48 (closed 2026-10-04, branch `feat/prompt-helpers` kept for its code and tests) built `messages_as_str`, `message_as_str`, `call_as_str`, `last_turns` and `step_as_str` in `src/inspect_sentinel/_prompt.py`. It was closed because message rendering already exists in four places and should be shared rather than written a fifth time: Inspect Scout's public `messages_as_str`/`message_as_str`/`MessageFormatOptions` (`_scanner/extract.py`), inspect_ai's private `format_function_call` (`_util/format.py`, used by approval, review, the scorer and the transcript), a private `messages_as_str`/`message_as_str` in `analysis/_dataframe/extract.py`, and `chat_history` in `scorer/_model.py`. Decided with the maintainer:
  1. An inspect_ai PR moves Scout's sync renderer into inspect_ai as public API (`message_as_str`, `messages_as_str` with Scout's `MessageFormatOptions` options, and a public tool-call renderer built on `format_function_call`), in `inspect_ai.model` and later `inspect_core` (workstream 1). Its output must be exactly Scout's, since scanner prompts and results depend on it.
  2. A Scout PR imports it, keeping what is Scout's own (the async `messages_as_str`, the preprocessor `transform`, message numbering and `include_ids` references), with a test that its output is unchanged.
  3. Sentinel imports the shared functions and keeps only `step_as_str` and `last_turns`. #48 rendered a tool call on one line where Scout uses one argument per line; either adopt Scout's or add an option to the shared renderer.
  `chat_history` stays, so `model_graded_qa` prompts do not change; the dataframe extractor stays unless its text may change.
  What #48 learned, to carry over (two review rounds, recorded in its body):
  - A call whose arguments did not parse renders as a fixed marker, `bash(<arguments did not parse>)`, with none of the model's text: inspect_ai's `ToolCall.parse_error` holds up to 16 KB of raw, unescaped arguments that can forge `TOOL:`/`ASSISTANT:` headers, and the tool message that follows already carries it.
  - Rendering an argument value never raises: `json.dumps(..., default=str)`, falling back to `repr()` with line breaks escaped for non-string keys and circular references.
  - Argument names come from the model, not the tool's schema, when a sentinel runs (it runs before `validate_tool_input`), so a name that is not an identifier is rendered quoted, `**{"name": value}`, so it cannot forge headers.
  - Message content can imitate role headers; say so, and keep the rendering inside the prompt's untrusted section.
  - `last_turns(messages, n)`: a turn runs from one assistant message to the next; never split a call from its result; drop the system message and the task unless the conversation has no more than `n` turns; `n=0` gives `[]`, negative `n` is a `ValueError`.
  - `step_as_str(step, *, turns=None)`: `step.input` (or its last `turns` turns), then an `ASSISTANT:` block with `step.message` and the checked call, then the result after a call; render oldest first so the prompt's prefix is stable for caching. Tool-stage only until the generate stages land.
  - `message_as_str` always returns a `str` and has no `exclude_system`, where Scout's returns `str | None`; reconcile when sharing.
  - Docs: a "Prompt Helpers" section on `docs/monitors.qmd`, and `examples/llm_suspicion.py` and Getting Started building their prompt with `step_as_str` instead of hand-built JSON.
- **A second set of rule helpers**, decided with the maintainer on 2026-10-03: `urls_in(text)`, `host_matches(url, patterns)` and `secrets_in(text)`. Left out of the first set to keep it small; nothing blocks it.
- **Finding paths in shell and code text.** Cut from sentinel PR #49 on 2026-10-04: `paths_in(call)` (paths in a file tool's arguments and in `bash()`, `bash_session()`, `python()` and `code_execution()` text), `unresolved_paths(call, *, cwd=None, home=None)` (references a rule cannot resolve, for a deny-list to escalate) and the private shell lexer `_shell.py`. The code and its tests are kept on the branch `archive/rule-helpers-shell-paths` (d9c2c90, "fix: read arithmetic shifts as arithmetic, and cover the guard branches"). What the work learned:
  - Regex and hand-lexer parsing failed open in case after case found by review: `$@`/`$*`; backslash-newline continuation; `~+`/`~-`; extglob; pathlib `/` and Python implicit string concatenation; directory changes by `git -C`, `tar -C` and `cwd=`; quoting inside a word (`'/e'tc/passwd`); `\cd` and `c''d`; and `<<` inside `$((…))` lexed as a heredoc.
  - False escalations of the deny-list idiom (reject a protected path, escalate anything unresolved) were reduced from 25 to 11 of a 286-command corpus (8.7% to 3.8%) and from 26 to 19 of a reviewer's 187 (14% to 10%), by reading `$HOME`/`$PWD`, following simple `cd`, and treating quoted arguments of a list of literal-reading programs as literal. Each relaxation needed its own guards.
  - A general rule that single-quoted text is literal is unsound: `bash -c`, `eval`, `find -path`, `sed`'s `e` command and `awk`'s `system()` read the text again.
  - A revisit should start from a real parser, such as `bashlex` or tree-sitter-bash, checked for portability to a proxy host, not regexes. Until then, finding what a command touches is an LLM monitor's job.
- **Shell evasion indicators** — a helper naming suspicious shell constructs (command substitution, decode-then-execute, inline interpreters, obfuscated command names) for rules to escalate on. Requested 2026-10-03; not built by an agent (blocked by a safety classifier while drafting test fixtures); to be written by a maintainer.

### Then: design decisions

- **Revisit `decide_final()`: is it the right design at all?** Raised in review: any protocol at any depth, including third-party code inside `observe_only()` or `threshold()`, can call it and bypass every layer above, while the design sells child decisions as advisory to the parent. Options to weigh: a declared capability (`@protocol(final=True)`, recorded in the registry and the log, with an undeclared call raising); parent control (`allow_final=` on a layer, or finalization only at the root); containment (`shadow()`); or a different mechanism. `sentinel.md` open question 23. `human()` no longer uses it, which weakens the case for keeping it. Simplifying the runner's origin-threading and its superseded and bypassed bookkeeping is part of the outcome.
- **Containing `decide_final()`**: a parent configured to hold back a child's final decision, e.g. `shadow(child)` for trialling a protocol on live traffic, recorded as its own event kind. Waits for a user who needs shadow mode.
- **Revisit an `escalate` that reaches the top with nobody to hand it to.** Today it proceeds, is recorded as the root's decision, and the host warns once per eval. The alternatives are rejecting it (fail-closed, as inspect's approval does when every approver escalates) or a configurable setting with a chosen default. The trade-off: rejecting turns every unsure rule into a blocked call. `sentinel.md` open question 24.
- **`Decision.modify(step, arguments=...)`**, a constructor that keeps the call's id and function by construction. A tool-call form can land before the generate stages, with the signature designed for both, since what a generate-stage modify replaces shapes it. Until then, authors write `Decision(action="modify", modified=replace(step.call, arguments={...}))`, and the shape check rejects a changed id or function.

### Then: tool-stage completeness

- **An opt-in tool filter.** Discussed with the user 2026-10-01, replacing the earlier idea of a default filter. There is no default filter: `submit()` is monitored like any tool, because the final answer is where sabotage or sandbagging shows up and silently dropping coverage is a safety failure, and since inspect_ai 34adba814 the sentinel checks `handoff()` calls before and after they run, as approval does for every tool (review still skips them). A monitor not written for a tool abstains in its body (`if step.call.function == "submit": return None`). The proposed opt-in: `tools=` on `@monitor`/`@protocol`, an include list of tool-name patterns as approval policies use (e.g. `"bash*"`), recorded in the registry info; the runner does not invoke a function for a tool outside its list, as it does not invoke one at a stage it does not watch, and nothing is recorded. Rules: it applies only to tool-stage functions; setting it on an instance with no tool-stage function is a configuration error raised when the factory is called; in a mixed group it filters only the tool-stage members. Alternatives weighed: per-instance overrides in `Task(sentinel=)` or YAML (later, if someone needs to override an author's choice; it is the `tools:` key removed earlier), a filter on a composition (too coarse), and a per-function `@tools(...)` decorator (more precise for mixed groups, but a second decorator to learn).
- **Record `step.input` instead of rebuilding it.** The dispatcher scans the transcript to rebuild what the model was sent. Instead, record each generate's input when it finishes, keyed by its assistant message id, scoped to the sample and bounded, and fall back to a scan only for a message no inspect generate produced.
- **Human surfaces: a typed reason and `modify`.** Let the person type a reason, and support `modify`, in the approval panel, the console and ACP. ACP also cannot withdraw a request, and resets an after-call card to pending. Separately, the console's Enter key approves even when approve is not offered, a fail-open bug in inspect_ai's released `human_approver`, to fix as its own inspect_ai issue and PR.
- **Monitor model calls tied to their row**: today they are listed together at the end of the sentinel block, because a `ModelEvent` records no instance path. Needs a span per instance inside the sentinel span, or the model event ids on the `SentinelEvent`.
- **Threshold comparisons in the viewer** (`0.62 < 0.80`): the event does not record the threshold a protocol applied.
- **A failed call under `threshold()` in the viewer**: a call whose sample failed reads "Observed · 1 failed", because the protocol that raised leaves no event.
- **The duplicate-key config loader, for `approval=` and `review=` too.** Move sentinel's YAML/JSON loader that rejects duplicate keys into inspect_ai `_util`, and adopt it for `approval=` and `review=` configs. That changes behaviour for those users, so it is its own PR to inspect_ai's `main`.

### Later

Waits on another workstream, or not urgent.

- **Generate stages** (`BeforeGenerate`, `AfterGenerate`), and with them the design of what a generate-stage `modify` may replace.
- **`compaction_summary(step)`**, a helper returning the latest compaction summary text. Waits on inspect_ai making the `"summary"` message metadata key a documented constant, and must handle a summary merged into the task prompt.
- **An example compaction monitor.** Raised by the maintainer on 2026-10-02: a stateful generate-stage monitor that watches the messages, detects a compaction summary, and runs a check on the summary, keeping what it has seen in `store_as()`. Waits on the generate stages (inspect_sentinel PR #42), and uses `compaction_summary` once it exists.
- **`react` overflow recovery rewriting history**: when a call overflows the context window, `react` replaces `state.messages` with the compacted list, so `step.history` loses the folded turns. To raise with inspect_ai.
- **`resample`, and `fetch`/`terminate` on `Host`**: designed, not yet built. `defer_to_trusted` was removed from the plan by the maintainer on 2026-10-02: an eval has no trusted model to defer to.
- **Bridged agents.** A sentinel does not yet run for a bridged agent's tool calls ([workstreams.md](workstreams.md), workstream 9). With that hook, verify that handoff and bridged agents open agent spans consistently, so conversation ids link as designed, with tests.
- **Migrating `review=` and tool approvers. This is the largest open question in how sentinels relate to inspect_ai.** Three questions are open, and the ordering is settled:
  - **Is `review=` deprecated in favour of after-call protocols?** The design's position ([Relationship to approval and review](sentinel.md#relationship-to-approval-and-review)) is that a childless `AfterToolCall` protocol subsumes `Reviewer`. Deprecating `@reviewer` and retiring `review/` waits on protocols having users.
  - **How do `approval=` and `sentinel=` coexist on one task?** Both stay for now.
  - **Settled: the ordering.** inspect_ai documents it in `docs/review.qmd`, "Approval, Review and Sentinels". Before a call, approval runs and then the sentinel, so a rejected call never reaches the sentinel and a modified one reaches it modified. After a call, review runs and then the sentinel.
  - **Can existing tool approvers run as protocols?** The proposal is an `as_protocol(approver)` adapter, so that approval users can compose approvers with monitors. It would map `approve` to `continue`, the approval's `explanation` to the decision's `message` (what the agent is told), and `reject`, `escalate`, `terminate` and `modify` to the action of the same name.
  - **Parked 2026-10-02: the draft adapter is #40** (`as_protocol()` over registered `approver` and `approval` protocols, with tests comparing it to `approval=`). Not a priority; before it lands:
    - Suppress the duplicate record: each approver call writes an `ApprovalEvent` beside the `SentinelEvent`, so a tool card shows the same decision twice. Show only the sentinel's when an approval runs under one.
    - Export the inspect_ai helpers it imports from private modules (`call_approver`, `policy_approver`, `approval_policies_from_config`, `ApprovalPolicyConfig`) rather than depend on `inspect_ai.approval._call` and `._policy`.
    - Decide whether a lone `as_protocol(approver)` whose escalate reaches the top rejects, as `approval=` does, instead of letting the call run; tied to the top-level escalate policy above.
    - Say in the docs that configuring a human approver in both `approval=` and `as_protocol()` asks the person twice.
    - A reviewer adapter, once deprecating `review=` is decided.

  `sentinel.md` open question 25.
- **Task and sample descriptions in the viewer.** Once inspect_ai #5667 lands, show `EvalSpec.task_description` in the Task tab's task information card (and perhaps as a tooltip on the task name in the title bar and log list), and a sample's `description` in the sample header beside the input and target (perhaps as an optional samples-list column if descriptions are short). ts-mono only, plus the submodule bump and viewer rebuild. Raised by the user 2026-10-02.
- **A sentinel section in the viewer's Task tab.** The eval header records the configuration at `eval.config.sentinel`, and the Task tab's configuration card shows it only as a raw nested value. A section should show the tree of instance names with each one's factory, kind (monitor or protocol), version and parameters, and link an instance to its rows in the transcript. Raised by the user 2026-10-01; the ts-mono work goes with the viewer redesign (ts-mono #716) or after it.
- **Sentinel events in the sample Activity tab.** The Activity tab (ts-mono main) lays a sample's events out by time; `SentinelEvent`s should appear there with the tool calls they check, including the time a monitor or a person took, so a slow monitor or a long wait on a person is visible. Raised by the user 2026-10-01.
- **`store_as(scope="task")`**: per-task state across samples, documented in the design (`sentinel.md`, "Per-task") but not built. Needs a design for concurrency and checkpointing first.
- **The `portable=True` runtime check** ([workstreams.md](workstreams.md), workstream 11; decided by the maintainer on 2026-10-05). `portable=` on `@monitor` and `@protocol`, default True; `portable=False` opts out and is recorded in the registry. When a portable function is registered or configured, sentinel checks its module's import closure against the bundleable allowlist and its body for the ambient escapes (`get_model()`, `sandbox()`, inspect_ai's `store()`), and fails with an error naming the function, the import or call, and `portable=False`. While it runs, it goes through the restricted host a proxy would use, so a banned call raises in every eval. A composition is as portable as its least portable member, and a `portable=False` function does not excuse its module. CI and deployment run the same check over the portable set. Where each half runs is open.
- **Testing helpers for authors.** Raised 2026-10-05 when the step builders were taken out of #49's public API. Decide in the pre-release API review (workstreams.md, workstream 5): likely an `inspect_sentinel.testing` module that builds a step and a `Context` together, and perhaps a fake host, so a rule or protocol can be unit-tested in a few lines; or constructors on the step types, as inspect_ai's `ModelOutput.from_content()`. The test suite's private fakes (`tests/_fakes.py`: `before_tool_call()`, `after_tool_call()`, `host_context()`, `in_step()`, `FakeHost`) are the starting point.
- **The `target` opt-in**: `@monitor(target=True)` and `@protocol(target=True)`, the runner passing `Context.target` only to instances that declared it, and the registry recording that they did, so the log shows which monitors used the answer. Performance with the answer is not achievable in deployment, sandbagging detection legitimately needs it, and a monitor that reads `target` is not portable to a proxy. Until then `target` is always None.
- **Giving checks access to the sample store.** A future consideration, raised by the maintainer on 2026-10-02 when `Context.store` was dropped ([Interface cleanup](#interface-cleanup)). Waits on a monitor or protocol that needs the agent's state; the design would have to say whether access is read-only, how an instance declares it, and what a proxy, which has no sample store, provides.
- **Drop the trusted/untrusted model framing where it implies a trusted model.** Raised by the maintainer on 2026-10-02 with the removal of `defer_to_trusted`. The design and docs describe the agent as the untrusted model and the monitor's model as trusted (`sentinel.md`, "What the names mean", and the `trusted` model role), which assumes a model nobody provides. Treating agent text as untrusted input in a monitor's prompt is a different sense and stays.
