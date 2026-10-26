from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable, Sequence
from copy import copy
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from inspect_ai._util.logger import warn_once
from inspect_ai._util.registry import (
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
from ._replay_step import ReplayStep
from ._report import Decision, Escalations, Failed, Report, Reported
from ._resolve import resolve_sentinel
from ._runner import run_sentinel
from ._step import Step
from ._types import Protocol, Sentinels
from ._unhandled import current_invocation
from ._validate import named_children

logger = logging.getLogger(__name__)

HumanStandIn: TypeAlias = Callable[
    [ReplayStep, Escalations, Sequence[str]], Awaitable[HumanAnswer]
]
"""A stand-in for the person `human()` asks, in a replay: handed the step, the escalations and the choices, as `Host.ask_human` is, and answering as the person would. An oracle built from labels, or a model."""

HumanFallback: TypeAlias = Literal["approve", "reject", "terminate"] | HumanStandIn
"""How a replay answers `human()` where no answer was recorded at the step: always with that answer, or by asking a `HumanStandIn`. A `human()` that does not offer the answer named gets `terminate`, which `human()` takes whatever it offers, as the answer a host gives when nobody can be asked."""

HumanSource: TypeAlias = Literal["recorded", "unanswered", "simulated"]
"""Where a replayed `human()` answer came from: the answer recorded at the step, the fallback's fixed answer, or a `HumanStandIn`."""

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
    """Where the answers came from when the step reached `human()`: `"unanswered"` if any `human()` got the fallback's fixed answer, else `"simulated"` if a stand-in answered any, else `"recorded"`. `None` when no `human()` asked."""

    human_answer: str | None = None
    """The fixed answer given where `human` is `"unanswered"`: the first a `human()` at the step got from the fallback. `None` otherwise."""

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
    human: HumanFallback = "reject",
) -> list[ReplayReport]:
    """Run a sentinel over steps rebuilt from a transcript, and return what it reported at each.

    The same monitors and protocols run as in an eval, through the same runner, so a monitor runs the same code live and in replay. The steps run in order, one at a time, with one `Store` and one recorder for the whole transcript, the scope they have live, so `Context.store_as()` state carries across compactions and conversations as it does in the eval. Only the steps at a stage the sentinel watches run, and each runs every stage the sentinel watches, as live.

    `human()` cannot ask a person in a replay. It is answered by the answer it gave at the step when the run was recorded, if any (`ReplayStep.recorded_answers`, by the asking instance's path), and otherwise by `human`: a fixed answer, or a stand-in, which is asked. A fixed answer the asking `human()` does not offer becomes `terminate`, as `reject` does after a tool runs or a model replies. A recorded answer that is not among the choices the replayed `human()` offers counts as no answer. `ReplayReport.human` says which answered.

    A monitor that raises is recorded as failed, in `ReplayReport.failed`, as live; a protocol or monitor error that the runner lets through fails the replay.

    Args:
        sentinel: A protocol, or monitors and protocols as `Task(sentinel=)` takes them, which resolve to one root. Monitors alone run under `observe_only()`; a lone monitor, bare or the only entry of a list or mapping, is the replayed instance, at its name or mapping key.
        steps: The steps, in the order they ran live.
        context: The sample the steps are from, as `Context.eval` gives it. None outside an eval.
        select: Which steps to run the sentinel at; all when None. The others are skipped, not run, so a monitor's state does not see them, and have no report. Bind it to the transcript the steps are from.
        human: How to answer `human()` where no answer was recorded; see above.

    Returns:
        One report per step `select` chose, in the order given.

    Raises:
        ValueError: If two steps of one stage share an id.
    """
    root, name, path = _root(sentinel)
    watched = watched_stages(root)
    store = Store()
    recorder = _Recorder()
    host = _ReplayHost(context, human)
    seen: set[tuple[str, str]] = set()
    reports: list[ReplayReport] = []
    for replay_step in steps:
        _check(replay_step, seen)
        if select is not None and not select(replay_step):
            continue
        if type(replay_step.step) not in watched:
            reports.append(
                ReplayReport(replay_step, name, path, (), (), (), None, None)
            )
            continue
        host.begin(replay_step)
        at = recorder.mark()
        decision = await run_sentinel(
            root,
            HostContext(host=host, recorder=recorder, store=store),
            replay_step.step,
        )
        recorded, failed, superseded = recorder.since(at)
        source, answer = host.answered()
        reports.append(
            ReplayReport(
                step=replay_step,
                name=name,
                path=path,
                reports=recorded,
                failed=failed,
                superseded=superseded,
                decision=decision,
                human=source,
                human_answer=answer,
            )
        )
    return reports


def _root(sentinel: Sentinels) -> tuple[Protocol, str, str]:
    # monitors alone run under observe_only(); a lone one, bare or as the only
    # entry of a list or mapping, reports at its own name, a mapping's key
    named = named_children(sentinel, None)
    if named and all(registry_info(child).type == "monitor" for _, child in named):
        root = observe_only(cast(Any, sentinel))
        if len(named) == 1:
            name = named[0][0]
            return root, name, name
    else:
        root = resolve_sentinel(sentinel)
    return root, registry_unqualified_name(registry_info(root)), ""


def _check(replay_step: ReplayStep, seen: set[tuple[str, str]]) -> None:
    key = (replay_step.stage, replay_step.id)
    if key in seen:
        raise ValueError(
            f"Two {replay_step.stage!r} steps have the id {replay_step.id!r}; a step id must be unique within a transcript for its stage, so their reports could not be told apart."
        )
    seen.add(key)


class _Recorder:
    # one for the whole transcript, as live; each step's records are the ones
    # made since its mark
    def __init__(self) -> None:
        self.reports: list[Reported[Report]] = []
        self.failures: list[Failed] = []
        self.supersessions: list[Reported[Decision]] = []

    def mark(self) -> tuple[int, int, int]:
        return len(self.reports), len(self.failures), len(self.supersessions)

    def since(
        self, mark: tuple[int, int, int]
    ) -> tuple[
        tuple[Reported[Report], ...], tuple[Failed, ...], tuple[Reported[Decision], ...]
    ]:
        reports, failures, supersessions = mark
        return (
            tuple(self.reports[reports:]),
            tuple(self.failures[failures:]),
            tuple(self.supersessions[supersessions:]),
        )

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
    # one per transcript, whose steps run one at a time, so the step being
    # replayed is the one begin() was last given
    def __init__(self, eval: EvalContext | None, human: HumanFallback) -> None:
        self._eval = eval
        self._human = human
        self._step: ReplayStep | None = None
        self._sources: list[tuple[HumanSource, str]] = []

    @property
    def eval(self) -> EvalContext | None:
        return self._eval

    def begin(self, step: ReplayStep) -> None:
        self._step = step
        self._sources = []

    def answered(self) -> tuple[HumanSource | None, str | None]:
        # the least real source, and the fixed answer where it is the fallback's
        for weakest in _WEAKEST:
            for source, decision in self._sources:
                if source == weakest:
                    return source, decision if source == "unanswered" else None
        return None, None

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
        recorded = replay_step.recorded_answers.get(path)
        if recorded is not None and (
            recorded.decision in choices or recorded.decision == "terminate"
        ):
            self._sources.append(("recorded", recorded.decision))
            return recorded
        if not isinstance(self._human, str):
            answer = await self._human(replay_step, escalations, tuple(choices))
            self._sources.append(("simulated", answer.decision))
            return answer
        # human() takes terminate whatever it offers, as the answer a host
        # gives when nobody can be asked
        decision = self._human if self._human in choices else "terminate"
        self._sources.append(("unanswered", decision))
        return HumanAnswer(
            decision=decision,
            reason=f"No person can be asked in a replay, and no answer was recorded here; answered {decision!r} for the replay.",
        )
