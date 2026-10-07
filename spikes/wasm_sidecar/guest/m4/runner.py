"""Thin guest-side runner.

A `Host` over the WIT imports, a `Recorder` that collects records for the host, and `run_sentinel` on a step from JSON.
"""

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from inspect_ai.core import ChatMessage, GenerateConfig, ModelOutput, Store, ToolInfo
from pydantic import TypeAdapter

from inspect_sentinel import BeforeToolCall, Context, HumanAnswer, Step
from inspect_sentinel._integration import HostContext, run_sentinel

STEP = TypeAdapter(BeforeToolCall)


class WitHost:
    """`inspect_sentinel.Host` forwarding to the component's `generate` import."""

    def __init__(self, generate: Callable[[str], Awaitable[str]]) -> None:
        self._generate = generate

    async def generate(
        self,
        input: str | list[ChatMessage],
        *,
        model: Any = None,
        role: str | None = None,
        tools: list[ToolInfo] | None = None,
        config: GenerateConfig | None = None,
    ) -> ModelOutput:
        request = {
            "input": input
            if isinstance(input, str)
            else [m.model_dump(exclude_none=True) for m in input],
            "model": model if isinstance(model, str) or model is None else str(model),
            "role": role if role is not None or model is not None else "monitor",
            "tools": [t.model_dump(exclude_none=True) for t in tools or []],
            "config": config.model_dump(exclude_none=True) if config else None,
        }
        return ModelOutput.model_validate_json(
            await self._generate(json.dumps(request))
        )

    async def ask_human(self, step: Step, choices: Sequence[str]) -> HumanAnswer:
        raise NotImplementedError("no audit queue in the spike host")


class ListRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def _add(self, status: str, context: Context, factory: str, **fields: Any) -> None:
        self.records.append(
            {"status": status, "path": context.path, "factory": factory, **fields}
        )

    def record(self, context: Context, factory: str, step: Step, reported: Any) -> None:
        report = reported.report
        self._add(
            "reported",
            context,
            factory,
            name=reported.name,
            report=type(report).__name__,
            suspicion=getattr(report, "suspicion", None),
            action=getattr(report, "action", None),
            explanation=report.explanation,
        )

    def failed(self, context: Context, factory: str, step: Step, failed: Any) -> None:
        self._add(
            "failed", context, factory, error=repr(getattr(failed, "error", failed))
        )

    def cancelled(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._add("cancelled", context, factory, name=name)

    def bypassed(self, context: Context, factory: str, step: Step, name: str) -> None:
        self._add("bypassed", context, factory, name=name)

    def superseded(
        self, context: Context, factory: str, step: Step, reported: Any
    ) -> None:
        self._add("superseded", context, factory, name=reported.name)


async def run_step(
    root: Any, step_json: str, generate: Callable[[str], Awaitable[str]]
) -> dict[str, Any]:
    step = STEP.validate_json(step_json)
    recorder = ListRecorder()
    host_context = HostContext(
        context=Context(path="", host=WitHost(generate), eval=None),
        recorder=recorder,
        store=Store(),
    )
    decision = await run_sentinel(root, host_context, step)
    return {
        "decision": None
        if decision is None
        else {
            "action": decision.action,
            "explanation": decision.explanation,
            "message": getattr(decision, "message", None),
        },
        "records": recorder.records,
    }
