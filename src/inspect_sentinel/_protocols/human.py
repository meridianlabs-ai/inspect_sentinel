from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Sequence
from typing import Literal, cast

from .._context import Context
from .._decorators import protocol
from .._host import HumanAnswer, HumanQuestion, HumanSurface, answers_from_record
from .._report import Decision, Escalations
from .._step import AfterGenerate, AfterToolCall, BeforeGenerate, BeforeToolCall, Step
from .._types import Protocol, ProtocolGroup

_Stage = Literal["tool_call", "tool_result", "model_input", "model_output"]

_CHOICES = ("approve", "reject", "terminate", "modify")
_LEGAL = {
    "tool_call": _CHOICES,
    "tool_result": ("approve", "terminate"),
    "model_input": ("approve", "terminate"),
    "model_output": ("approve", "terminate"),
}
_DEFAULTS = {
    "tool_call": ("approve", "reject", "terminate"),
    "tool_result": ("approve", "terminate"),
    "model_input": ("approve", "terminate"),
    "model_output": ("approve", "terminate"),
}
# why a choice the person may pick before a call is not legal at the stage
_NOT_LEGAL = {
    "tool_result": "after a tool call, since the call has already run",
    "model_input": "before a generate, since reject and modify are not legal at the generate stages yet",
    "model_output": "after a generate, since reject and modify are not legal at the generate stages yet",
}


@protocol
def human(
    stages: Sequence[str],
    choices: Sequence[str] | None = None,
    surface: HumanSurface | str | None = None,
) -> Protocol | ProtocolGroup:
    """Ask a person to decide, every time the step reaches this protocol.

    The person is asked through `Host.ask_human`, which is handed the step and the escalations this protocol was handed and shows them together, so at the end of a `sequential()` the person sees who escalated and why. On its own, `human(stages=["tool_call"])` asks about every call. After a generate, the person sees the whole assistant turn, its text and every tool call it makes, before any of the calls run. Before a generate, the person sees the request the model is about to be sent. The answer maps to a decision: `approve` to `proceed` explained by the person's reason, so the log records why the step went ahead, `reject` to `reject` with the person's reason as both the `message` the agent reads and the `explanation`, `terminate` to `terminate` explained by the reason, and `modify` to a `modify` with the person's replacement call.

    With a `surface`, the person is asked through a team's own review tool instead of the host. `human()` hands the surface a `HumanQuestion` holding the stage, the step, each escalation's full report and the choices, and maps the `HumanAnswer` it returns exactly as it maps the host's, so the answer is recorded and seen at later steps as any other. A surface that raises, or answers a choice it was not offered other than `terminate`, fails the step rather than letting it proceed. In a replay nobody can be asked, so the replay answers in place of the surface, from the answer recorded at the step or from its fallback.

    The answer is an ordinary decision: it ends a `sequential()` as any link's decision does, and beside other protocols in a `concurrent()` it is one vote, so another protocol's `reject` outranks the person's `approve`.

    Args:
        stages: The stages to ask at, as `SentinelEvent.stage` records them: `"tool_call"` (before a tool call), `"tool_result"` (after it, with the result the model is about to receive), `"model_input"` (before a generate by the agent's model, with the request about to be sent) and `"model_output"` (after a generate by the agent's model, before its output reaches the agent).
        choices: What the person may pick, from `approve`, `reject`, `terminate` and `modify`. Defaults to `approve`, `reject` and `terminate` before a call, and `approve` and `terminate` after a call and at the generate stages, where `reject` and `modify` are not legal. A surface cannot be offered `modify` yet.
        surface: Where to ask the person instead of the host: an `async` function defined at module level that takes a `HumanQuestion` and returns a `HumanAnswer`, or its name as `"module:function"` (the module may be a `.py` file's path, e.g. `"tasks/review.py:ask_reviewers"`), so a configuration file can name it. The log records it by that name, so a retry finds it again. None, the default, asks through `Host.ask_human`.
    """
    valid = ", ".join(repr(stage) for stage in _LEGAL)
    if isinstance(stages, str) or not stages:
        raise ValueError(f"human needs a list of stages, from {valid}; got {stages!r}.")
    for stage in stages:
        if stage not in _LEGAL:
            raise ValueError(
                f"human cannot ask at stage {stage!r}; the stages it supports are {valid}."
            )
    if len(set(stages)) != len(stages):
        raise ValueError(f"human was given a stage twice: {list(stages)!r}.")
    offered = {stage: _choices(stage, choices) for stage in stages}
    ask_surface = _surface(surface) if surface is not None else None
    if ask_surface is not None and any("modify" in c for c in offered.values()):
        raise ValueError(
            f"human cannot offer 'modify' through the surface {_name(ask_surface)}, since a surface cannot answer with a replacement call yet."
        )

    async def ask(
        context: Context,
        step: Step,
        escalations: Escalations,
        stage: _Stage,
    ) -> Decision:
        if ask_surface is None or answers_from_record(context.host):
            answer = await context.host.ask_human(step, escalations, offered[stage])
            return _decision(answer, offered[stage], "The host")
        question = HumanQuestion(
            stage=stage,
            step=step,
            escalations=escalations,
            choices=offered[stage],
            eval=context.host.eval,
            proxy=context.proxy,
        )
        source = f"The surface {_name(ask_surface)}"
        try:
            answer = await ask_surface(question)
        except Exception as ex:
            ex.add_note(
                f"{source} raised while asking about a {stage} step, so the step does not proceed."
            )
            raise
        if not isinstance(cast(object, answer), HumanAnswer):
            raise TypeError(
                f"{source} returned {type(answer).__name__}, not a HumanAnswer, so the step does not proceed."
            )
        return _decision(answer, offered[stage], source)

    async def tool_call(
        context: Context,
        step: BeforeToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "tool_call")

    async def tool_result(
        context: Context,
        step: AfterToolCall,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "tool_result")

    async def model_input(
        context: Context,
        step: BeforeGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "model_input")

    async def model_output(
        context: Context,
        step: AfterGenerate,
        escalations: Escalations,
    ) -> Decision | None:
        return await ask(context, step, escalations, "model_output")

    functions: dict[str, Protocol] = {
        "tool_call": tool_call,
        "tool_result": tool_result,
        "model_input": model_input,
        "model_output": model_output,
    }
    if len(stages) == 1:
        return functions[stages[0]]
    return ProtocolGroup(*(functions[stage] for stage in stages))


def _choices(stage: str, choices: Sequence[str] | None) -> tuple[str, ...]:
    if choices is None:
        return _DEFAULTS[stage]
    if isinstance(choices, str) or not choices:
        raise ValueError(
            f"human needs at least one choice, from {', '.join(repr(c) for c in _CHOICES)}; got {choices!r}."
        )
    for choice in choices:
        if choice not in _CHOICES:
            raise ValueError(
                f"human cannot offer {choice!r}; the choices are {', '.join(repr(c) for c in _CHOICES)}."
            )
        if choice not in _LEGAL[stage]:
            raise ValueError(
                f"human cannot offer {choice!r} {_NOT_LEGAL[stage]}; the choices there are {', '.join(repr(c) for c in _LEGAL[stage])}."
            )
    return tuple(choices)


def _surface(surface: object) -> HumanSurface:
    # the surface a function or its `module:function` name gives; a function
    # must be defined at module level, since the log records it by that name
    # and a retry rebuilds human() from it
    found = _named(surface) if isinstance(surface, str) else surface
    if not inspect.isfunction(found) or "<" in found.__qualname__:
        raise TypeError(
            f"human takes a surface defined at module level, or its 'module:function' name, so the log can record it and a retry can find it again; got {found!r}."
        )
    if not inspect.iscoroutinefunction(found):
        raise TypeError(
            f"human takes an async surface, since the step waits for the person's answer; {_name(found)} is not async."
        )
    return cast(HumanSurface, found)


def _named(name: str) -> object:
    module, _, qualname = name.partition(":")
    if not module or not qualname:
        raise ValueError(
            f"A surface given by name is 'module:function', where the module may be a .py file's path; got {name!r}."
        )
    if module.endswith(".py"):
        # loaded under its path, the name inspect_ai gives a task file, so a
        # surface defined in a task file is recorded again by the same name
        spec = importlib.util.spec_from_file_location(module, module)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load the surface {name!r} from {module!r}.")
        loaded = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(loaded)
        except FileNotFoundError as ex:
            raise ValueError(
                f"Cannot load the surface {name!r}: there is no file {module!r}."
            ) from ex
    else:
        loaded = importlib.import_module(module)
    found: object = loaded
    for part in qualname.split("."):
        found = getattr(found, part, None)
    if found is None:
        raise ValueError(
            f"Cannot find the surface {name!r}: {module!r} does not define {qualname!r}."
        )
    return found


def _name(surface: object) -> str:
    return f"{getattr(surface, '__module__', '?')}:{getattr(surface, '__qualname__', repr(surface))}"


def _decision(answer: HumanAnswer, choices: tuple[str, ...], source: str) -> Decision:
    if answer.decision not in choices and answer.decision != "terminate":
        raise ValueError(
            f"{source} answered {answer.decision!r}, which is not one of the choices offered: {list(choices)!r}."
        )
    if answer.modified is not None and answer.decision != "modify":
        raise ValueError(
            f"{source} answered {answer.decision!r} with a modified call; only a 'modify' carries one."
        )
    reason = answer.reason or None
    match answer.decision:
        case "approve":
            return Decision.proceed(reason)
        case "reject":
            return Decision.reject(reason, message=reason)
        case "terminate":
            return Decision.terminate(reason)
        case _:
            return Decision(action="modify", modified=answer.modified)
