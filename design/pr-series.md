# Implementation: the first PR series

How `sentinel.md` becomes code, in an order where each PR is reviewable alone, has its own tests, and leaves `main` importable. Agreed 2026-09-21; PR 0 opened 2026-09-22.

## Goal

The minimal surface that lets `inspect_ai` connect: the types a dispatcher constructs, the decorators an author writes against, the runner a protocol calls, and the three protocols the compile rule needs. No `inspect_ai` dispatcher yet; that is the next series, on inspect_ai's `feature/sentinel` branch.

## Decisions

- **Surface only.** Sentinel ships types and functions; `inspect_ai` wiring comes after. Verified by unit tests with fake `Host` and `Recorder`.
- **Real registration.** `@monitor` and `@protocol` register under `"monitor"` and `"protocol"`, added to `inspect_ai`'s `RegistryType` by PR 0 ([inspect_ai#5514](https://github.com/UKGovernmentBEIS/inspect_ai/pull/5514), branch `feature/sentinel`). Sentinel's `inspect-ai` git ref points at that branch until it merges.
- **Tool stages only.** `BeforeToolCall` and `AfterToolCall` with the design's full fields; `Step` is their union. The generate stages arrive with the generate-side dispatcher. The schema may change freely until there are users.
- **Recording is the runner's, through a separate interface.** `Host` is the author-facing ABI (`generate` only for now). `Recorder` has two methods: `record`, called by the runner for every participating child, and `cancelled`, called for a child cancelled before it reported. This departs from `sentinel-deployment.md`, whose `Host` ABI carries `record`; that doc is to be updated.
- **Two context types.** `Context` is what authors see. `RunnerContext(Context)` adds `recorder` and `child()`; the dispatcher builds it, `dataclasses.replace` preserves it down the layers, and the runner raises `TypeError` if handed a bare `Context`.
- **Author surface vs integration surface.** `__init__` exports only what an author writes against, including `Reported` and `Report`, since `step.escalations` and the runner's results are typed with them. `inspect_sentinel/_integration.py` re-exports what the dispatcher needs (`RunnerContext`, `Recorder`, `validate_decision`, `compile_sentinel`); `inspect_ai` imports from that module only, so refactors have one file to keep stable.
- **Shipped protocols:** `observe`, `concurrent`, `threshold`. `chain` and `human` are follow-ups.

## Layout

```
src/inspect_sentinel/
  __init__.py       author-facing exports
  _step.py          BeforeToolCall, AfterToolCall, Step, Stage
  _report.py        Suspicion, Action, Observation, Decision, Report, Reported
  _context.py       Host, Recorder, Context, RunnerContext
  _monitor.py       Monitor, ControlProtocol, Monitors, Protocols, Children, @monitor, @protocol
  _runner.py        Observations, Decisions, run_monitor, run_protocol,
                    run_monitors, run_protocols, run_children
  _check.py         validate_decision
  _protocols.py     observe, concurrent, threshold
  _compile.py       compile_sentinel
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
Suspicion = float | dict[str, float]
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
    authoritative: bool = False
    modified: ToolCall | None = None                    # ChatMessageAssistant joins with AfterGenerate
    explanation: str | None = None
    metadata: dict[str, Any] | None = None
    @classmethod clear / reject / terminate / escalate

Report = Observation | Decision

R_co = TypeVar("R_co", bound=Report, covariant=True)  # covariant so Reported[Decision] flows into Reported[Report]

@dataclass(frozen=True)
class Reported(Generic[R_co]):
    name: str
    path: str
    report: R_co
```

```python
class Host(Protocol):
    async def generate(self, input: str | list[ChatMessage], *, model: str | None = None,
                       tools: list[ToolInfo] | None = None,
                       config: GenerateConfig | None = None) -> ModelOutput: ...

class Recorder(Protocol):
    def record(self, context: Context, step: Step, reported: Reported[Report]) -> None: ...
    def cancelled(self, context: Context, step: Step, name: str) -> None: ...

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

Implemented: stages are read at factory-call time from the returned function's second-parameter annotation via `get_type_hints` and stored on the instance; `stages(instance)` exposes them to the runner and is integration surface. A union of stage payloads registers at every member. The return annotation must be the report type, optionally with `None`. `registry_create` is not a construction path for sentinels: it instantiates only factories whose return annotation's class name equals the registry type, and `Monitor`/`ControlProtocol` are union aliases, so it returns the factory uncalled and `inspect_ai` deliberately has no `registry_create` overload for the two types, which makes such a call a type error. Configuration and replay construct through `create_registry_object`.

## PR 3: runner

`_runner.py`. Per child: skip when not annotated for the stage (indistinguishable from abstention); name from mapping key or registry name, duplicate within a layer is an error; derive the child context with `child()`; record every participating report through `context.recorder`; exceptions propagate. Plural forms fan out with anyio (own `tg_collect`, no inspect_ai dependency); `run_protocols` cancels siblings on `terminate`. `run_children` runs both families in one task group.

Tests use a list-backed `Recorder` and a stub `Host`. Cover: filtering, naming and the duplicate error, path derivation, every participating report recorded including ones the caller ignores, `max_suspicion()` over scalar and dict suspicion, `strongest()` precedence, sibling cancellation on `terminate`, a raising child fails the call, `TypeError` on a bare `Context`.

Implemented: the decorators store the accepted payload classes on each instance and `step_types(instance)` (runner-facing, not exported) reads them; the runner filters with `isinstance`, so there is no stage-name vocabulary in the filter. Children are named by mapping key or by `registry_unqualified_name`, since a package-qualified registry name (`acme/suspicion`) would put the path separator inside a segment; two packages' monitors sharing a leaf name in one layer therefore collide, and the duplicate-name error tells the author to name them with a mapping. `run_monitor` and `run_protocol` check the child's registry type; `run_monitors`, `run_protocols` and `run_children` all validate their children once in `_named` and then fan out through one private `_run_named`, which owns the `terminate` cancellation. The task-group unwrap catches `ExceptionGroup` only, never `BaseExceptionGroup`, so a cancellation travelling with an error propagates intact, as in inspect_ai's `tg_collect`. `Observation` now rejects an empty structured suspicion, since `max_suspicion()` has no answer for it. Exceptions propagate unconditionally: the failure policy (warn and continue for a monitor nothing consumes, `fail="open"`) is not implemented here and is not PR 4's either; it needs a per-child hook in `_run_child`, because by the time an exception leaves the task group the siblings' results are gone, and it will be added with that hook in a later PR. A child cancelled before it reported, whether by a sibling's `terminate` or by cancellation from above, is recorded through `Recorder.cancelled()`; cancellation lands at a child's next await, so one that finishes without awaiting is recorded normally. When several children fail concurrently only the first exception surfaces, and a child raising its own `ExceptionGroup` passes through as a group; both match inspect_ai's `tg_collect` and the deferred failure-policy hook will want the sibling errors.

## PR 4: protocols, boundary check, compile, integration

`_protocols.py`, `_check.py`, `_compile.py`, `_integration.py`, `_registry.py`, and the `inspect_ai` entry point in `pyproject.toml`.

- `observe`: records, returns `None`.
- `concurrent`: `run_children`; `strongest()`; a `modify` with more than one deciding child becomes `reject` naming the modifier.
- `threshold(monitors, reject_at, terminate_at=None)`: as written in `sentinel.md`.
- `validate_decision(decision, stage, children)`: action legal for stage (`reject` and `modify` are `BeforeToolCall` only); `modified` set iff `action == "modify"`; not weaker than any authoritative child. Raises.
- `compile_sentinel(spec)`: monitor or monitors-only collection to `observe`; anything containing a protocol to `concurrent`; single protocol to `concurrent` of one.

Tests: each protocol's rules above; every row of the compile table; each of the three boundary-check conditions raising.

**Multi-function monitors (agreed 2026-09-25).** A `@monitor` or `@protocol` factory may return a sequence of functions instead of one. The group is one configured instance: one instance name, one `path`, one child context, and so one `store_as` namespace that every member shares; that is the point, since accumulating across stages is the trajectory-score case the design argues for and closure state is wrong for it (created once per configuration, shared across samples). Members are told apart by the inner function's `__name__`, recorded as a `function` field on `Reported` and `SentinelEvent`; the uniqueness claim becomes path plus function within a sample step, and read-mode validation keys on both. Duplicate function names within one factory are a configuration error, checked when the factory is called. The decorators validate each member and record the union of their payload types; the runner treats the group as one child and dispatches to every member that accepts the step, so one instance may contribute several reports at a step, which `Observations` and `Decisions` already accommodate as sequences. Each member still watches exactly one stage, so "One function, one point" stands; what changes is that a factory may emit several functions when they share state, which is the class-with-a-method-per-stage the design rejected, reached through functions so the static scan and per-function portability verdict still work. Two instances of one factory, and two samples, keep separate state as before. `sentinel.md` "One function, one point", "Instance names", and the `SentinelEvent` listing need updating.

## Design edits owed (on inspect_ai `design/monitor`)

Move `record` off `Host` into `Recorder` in `sentinel-deployment.md`; add `path` and `RunnerContext` to `sentinel-reference.md`'s `Context`; define `Stage`. (Done in the repo copy: `RunnerContext` named in the runner section; `named()` noted as the runner's private helper; unqualified instance names.)

## Out of scope

The failure policy (`design/sentinel.md`, "Failure semantics"): the runner propagates every exception until a later PR adds the per-child hook described under PR 3.

Generate stages, `chain`, `human`, `defer_to_trusted`, `resample`, views and rendering helpers, `store_as(scope=)`, `fetch`/`terminate` on `Host`, the `inspect_ai` dispatcher and `SentinelEvent`.
