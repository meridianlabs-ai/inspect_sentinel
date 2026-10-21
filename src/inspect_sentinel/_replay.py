from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import anyio
from inspect_ai._util.logger import warn_once
from inspect_ai._util.registry import (
    is_registry_object,
    registry_info,
    registry_unqualified_name,
)
from inspect_ai.model import ChatMessage, GenerateConfig, Model, ModelOutput, get_model
from inspect_ai.model._model import active_model, model_roles
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util import Store

from ._context import Context, EvalContext
from ._decorators import watched_stages
from ._host import HostContext, HumanAnswer
from ._protocols import observe_only
from ._report import Decision, Escalations, Failed, Report, Reported
from ._resolve import resolve_sentinel
from ._runner import run_sentinel
from ._step import AfterGenerate, AfterToolCall, BeforeGenerate, ReplayStep, Step
from ._types import Protocol, Sentinels
from ._unhandled import current_invocation
from ._validate import named_children

logger = logging.getLogger(__name__)

Split: TypeAlias = Literal["compaction", "conversation"] | None
"""How a replay groups a transcript's steps into units, which run concurrently, each in order with its own store: `"compaction"`, a conversation's steps between two of its compactions; `"conversation"`, a conversation's steps; `None`, the whole transcript."""

HumanStandIn: TypeAlias = Callable[
    [ReplayStep, Escalations, Sequence[str]], Awaitable[HumanAnswer]
]
"""A stand-in for the person `human()` asks, in a replay: handed the step, the escalations and the choices, as `Host.ask_human` is, and answering as the person would. An oracle built from labels, or a model."""

HumanPolicy: TypeAlias = Literal["recorded", "fail_closed"] | HumanStandIn
"""How a replay answers `human()`: `"recorded"`, the answer recorded at the step, else the fail-closed answer; `"fail_closed"`, the fail-closed answer always; or a `HumanStandIn`, asked where no answer is recorded."""

HumanSource: TypeAlias = Literal["recorded", "unanswered", "simulated"]
"""Where a replayed `human()` answer came from: the answer recorded at the step, the fail-closed answer given when no one can be asked, or a `HumanStandIn`."""

_STAGES: dict[type[Any], str] = {
    BeforeGenerate: "model_input",
    AfterGenerate: "model_output",
    AfterToolCall: "tool_result",
}
# where a step asked more than one human(), the least real answer says how the
# step was decided
_WEAKEST: tuple[HumanSource, ...] = ("unanswered", "simulated", "recorded")


@dataclass(frozen=True)
class ReplayReport:
    """What a sentinel reported at one replayed step."""

    step: ReplayStep
    """The step."""

    name: str
    """The replayed sentinel's instance name: the root protocol's registry name without its package prefix, or a lone monitor's."""

    path: str
    """The replayed sentinel's instance path: empty for a protocol, the monitor's name for a lone monitor, which the replay runs under `observe_only()`."""

    reports: tuple[Reported[Report], ...]
    """Every report recorded at the step, in the order recorded: each layer's children before the layer, so the root's decision comes last."""

    failed: tuple[Failed, ...]
    """The monitor functions that raised at the step instead of reporting."""

    superseded: tuple[Reported[Decision], ...]
    """The final decisions that did not take effect at the step, as `Recorder.superseded` records them."""

    decision: Decision | None
    """The step's outcome, as `run_sentinel` returns it live: the root's decision, with an escalate that reached the root as the `terminate` it becomes. `None` when the root abstained or nothing watched the stage."""

    human: HumanSource | None
    """Where the answers came from when the step reached `human()`: `"unanswered"` if any `human()` got the fail-closed answer, else `"simulated"` if a stand-in answered any, else `"recorded"`. `None` when no `human()` asked."""

    @property
    def report(self) -> Reported[Report] | None:
        """The replayed sentinel's own report at the step, at `path`: the root protocol's decision, or a lone monitor's observation (its last, if several of its functions watch the stage). `None` when it abstained or did not watch the stage."""
        for reported in reversed(self.reports):
            if reported.path == self.path:
                return reported
        return None


async def replay_monitor(
    sentinel: Sentinels,
    steps: Iterable[ReplayStep],
    *,
    context: EvalContext | None = None,
    select: Callable[[ReplayStep], bool] | None = None,
    human: HumanPolicy = "recorded",
    split: Split = "compaction",
    max_unit_steps: int | None = None,
    max_units: int = 10,
) -> list[ReplayReport]:
    """Run a sentinel over steps rebuilt from a transcript, and return what it reported at each.

    The same monitors and protocols run as in an eval, through the same runner, so a monitor runs the same code live and in replay. Steps come from `steps_from_messages()` for a message list, as in a monitor's unit tests, or from a log; `inspect_sentinel.scout.as_scanner()` runs this over a Scout scan.

    The steps are grouped into units by `split`, units run concurrently, at most `max_units` at once, and a unit runs its steps in order with its own `Store`, so `Context.store_as()` state carries from step to step within a unit and starts fresh at each. Only the steps at a stage the sentinel watches run, and each runs every stage the sentinel watches, as live.

    `human()` cannot ask a person in a replay. It is answered by the answer it gave at the step when the run was recorded, if any (`ReplayStep.recorded_answers`, by the asking instance's path), and otherwise by `human`: `"recorded"` gives the fail-closed answer a host gives when nobody can be asked, `reject` where it is offered and `terminate` otherwise; a stand-in is asked. `"fail_closed"` ignores recorded answers too. A recorded answer that is not among the choices the replayed `human()` offers counts as no answer. `ReplayReport.human` says which answered.

    A monitor that raises is recorded as failed, in `ReplayReport.failed`, as live; a protocol or monitor error that the runner lets through fails the replay.

    Args:
        sentinel: A protocol, or monitors and protocols as `Task(sentinel=)` takes them, which resolve to one root. Monitors alone run under `observe_only()`.
        steps: The steps, in the order they ran live.
        context: The sample the steps are from, as `Context.eval` gives it. None outside an eval.
        select: Which steps to run the sentinel at; all when None. The others are skipped, not run, so a monitor's state does not see them, and have no report. Bind it to the transcript the steps are from.
        human: How to answer `human()`; see above.
        split: How to group steps into units: by compaction segment (`ReplayStep.segment`), by conversation, or `None` for the whole transcript in one unit, which is the scope a store has live.
        max_unit_steps: Split each unit further into runs of at most this many of the steps that run, each with a fresh store. A stateful monitor sees less at each cut.
        max_units: How many units run at once.

    Returns:
        One report per step `select` chose, in the order given.

    Raises:
        ValueError: If a step's `stage` does not match its step's type, two steps of one stage share an id, or `max_unit_steps` or `max_units` is less than 1.
    """
    if max_unit_steps is not None and max_unit_steps < 1:
        raise ValueError(f"max_unit_steps must be at least 1; got {max_unit_steps}.")
    if max_units < 1:
        raise ValueError(f"max_units must be at least 1; got {max_units}.")
    root, name, path = _root(sentinel)
    watched = watched_stages(root)
    replayed = list(steps)
    reports: list[ReplayReport | None] = [None] * len(replayed)
    units: dict[object, list[int]] = {}
    seen: set[tuple[str, str]] = set()
    chosen: list[int] = []
    for index, replay_step in enumerate(replayed):
        _check(replay_step, seen)
        if select is not None and not select(replay_step):
            continue
        chosen.append(index)
        if type(replay_step.step) not in watched:
            reports[index] = ReplayReport(
                replay_step, name, path, (), (), (), None, None
            )
            continue
        units.setdefault(_unit(replay_step, split), []).append(index)

    async def run_unit(indexes: list[int]) -> None:
        async with limiter:
            store = Store()
            host = _ReplayHost(context, human)
            for index in indexes:
                replay_step = replayed[index]
                recorder = _Recorder()
                host.begin(replay_step)
                decision = await run_sentinel(
                    root,
                    HostContext(host=host, recorder=recorder, store=store),
                    replay_step.step,
                )
                reports[index] = ReplayReport(
                    step=replay_step,
                    name=name,
                    path=path,
                    reports=tuple(recorder.reports),
                    failed=tuple(recorder.failures),
                    superseded=tuple(recorder.supersessions),
                    decision=decision,
                    human=host.answered(),
                )

    limiter = anyio.CapacityLimiter(max_units)
    try:
        async with anyio.create_task_group() as tg:
            for unit in units.values():
                for run in _runs(unit, max_unit_steps):
                    tg.start_soon(run_unit, run)
    except BaseExceptionGroup as ex:
        raise _first(ex) from None
    return [cast(ReplayReport, reports[index]) for index in chosen]


def _root(sentinel: Sentinels) -> tuple[Protocol, str, str]:
    if is_registry_object(sentinel) and registry_info(sentinel).type == "monitor":
        name = registry_unqualified_name(registry_info(sentinel))
        return observe_only(cast(Any, sentinel)), name, name
    named = named_children(sentinel, None)
    if named and all(registry_info(child).type == "monitor" for _, child in named):
        root = observe_only(cast(Any, sentinel))
    else:
        root = resolve_sentinel(sentinel)
    return root, registry_unqualified_name(registry_info(root)), ""


def _check(replay_step: ReplayStep, seen: set[tuple[str, str]]) -> None:
    stage = _STAGES.get(type(replay_step.step), "tool_call")
    if replay_step.stage != stage:
        raise ValueError(
            f"Step {replay_step.id!r} has stage {replay_step.stage!r} but holds a {type(replay_step.step).__name__}, which is a {stage!r} step."
        )
    key = (replay_step.stage, replay_step.id)
    if key in seen:
        raise ValueError(
            f"Two {replay_step.stage!r} steps have the id {replay_step.id!r}; a step id must be unique within a transcript for its stage, so their reports could not be told apart."
        )
    seen.add(key)


def _unit(replay_step: ReplayStep, split: Split) -> object:
    if split == "compaction":
        return replay_step.segment
    if split == "conversation":
        return replay_step.segment[0]
    return None


def _runs(unit: list[int], max_unit_steps: int | None) -> list[list[int]]:
    if max_unit_steps is None:
        return [unit]
    return [unit[i : i + max_unit_steps] for i in range(0, len(unit), max_unit_steps)]


def _first(group: BaseExceptionGroup[BaseException]) -> BaseException:
    # a failed unit cancels the rest, so its error is the one to raise
    for ex in group.exceptions:
        if isinstance(ex, BaseExceptionGroup):
            return _first(cast(BaseExceptionGroup[BaseException], ex))
        if not isinstance(ex, anyio.get_cancelled_exc_class()):
            return ex
    return group


class _Recorder:
    def __init__(self) -> None:
        self.reports: list[Reported[Report]] = []
        self.failures: list[Failed] = []
        self.supersessions: list[Reported[Decision]] = []

    def record(
        self, context: Context, factory: str, step: Step, reported: Reported[Report]
    ) -> None:
        self.reports.append(reported)

    def failed(
        self, context: Context, factory: str, step: Step, failed: Failed
    ) -> None:
        self.failures.append(failed)

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        pass

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        pass

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Reported[Decision]
    ) -> None:
        self.supersessions.append(reported)


class _ReplayHost:
    # one per unit, whose steps run one at a time, so the step being replayed
    # is the one begin() was last given
    def __init__(self, eval: EvalContext | None, human: HumanPolicy) -> None:
        self._eval = eval
        self._human = human
        self._step: ReplayStep | None = None
        self._sources: list[HumanSource] = []

    @property
    def eval(self) -> EvalContext | None:
        return self._eval

    def begin(self, step: ReplayStep) -> None:
        self._step = step
        self._sources = []

    def answered(self) -> HumanSource | None:
        for source in _WEAKEST:
            if source in self._sources:
                return source
        return None

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: str | None = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        # models resolve as inspect_ai's host resolves them live, from the
        # scan's model and roles in place of the eval's
        if isinstance(cast(object, model), Model):
            raise TypeError(
                "Host.generate() takes a model name, not a Model. Configure the Model for a role with the scan's model_roles, and pass role='<role>'."
            )
        if model == "":
            raise ValueError(
                "Host.generate() model must be a model name, not an empty string. Pass None to use a role."
            )
        if role == "":
            raise ValueError(
                "Host.generate() role must be a model role, not an empty string. Pass None for the 'monitor' role."
            )
        if model is not None and role is None:
            resolved = copy(get_model(model))
            resolved._set_role("sentinel")  # pyright: ignore[reportPrivateUsage]
        else:
            role = role or "monitor"
            if model is None and role not in model_roles():
                warn_once(
                    logger,
                    f"No model is configured for the sentinel role '{role}', so monitor calls in this replay use the scan's model. Set one with the scan's model_roles={{'{role}': ...}}.",
                )
            resolved = get_model(model, role=role, default=active_model())
        return await resolved.generate(
            input,
            tools=tools or [],
            tool_choice=tool_choice,
            config=config or GenerateConfig(),
        )

    async def ask_human(
        self, step: Step, escalations: Escalations, choices: Sequence[str]
    ) -> HumanAnswer:
        replay_step = self._step
        if replay_step is None:
            raise RuntimeError("The replay host was asked about a step it never began.")
        node = current_invocation()
        path = node.path if node is not None else ""
        if self._human != "fail_closed":
            recorded = _recorded_answers(replay_step).get(path)
            if recorded is not None and (
                recorded.decision in choices or recorded.decision == "terminate"
            ):
                self._sources.append("recorded")
                return recorded
        if not isinstance(self._human, str):
            answer = await self._human(replay_step, escalations, tuple(choices))
            self._sources.append("simulated")
            return answer
        self._sources.append("unanswered")
        return HumanAnswer(
            decision="reject" if "reject" in choices else "terminate",
            reason="No person can be asked in a replay, and no answer was recorded here; failing closed.",
        )


def _recorded_answers(step: ReplayStep) -> Mapping[str, HumanAnswer]:
    # ReplayStep.recorded_answers lands with #133; until then a step has none
    return cast(Mapping[str, HumanAnswer], getattr(step, "recorded_answers", {}))
