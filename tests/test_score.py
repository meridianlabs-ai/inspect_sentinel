from collections.abc import Callable
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    ModelOutput,
    get_model,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_ai.util._concurrency import init_concurrency

from inspect_sentinel._integration import score_step
from inspect_sentinel._step import Step

from ._fakes import (
    after_generate_step,
    after_tool_call,
    before_generate_step,
    before_tool_call,
)


@pytest.fixture(autouse=True)
def fresh_concurrency() -> None:
    # inspect_ai keeps a model's connection limiter for the process, bound to
    # the backend that made it; these tests generate on asyncio and on trio
    init_concurrency()


class Monitor:
    """A mockllm model that answers with the given outputs and keeps what it was sent."""

    def __init__(self, *outputs: ModelOutput) -> None:
        self.outputs = list(outputs)
        self.inputs: list[list[ChatMessage]] = []
        self.model: Model = get_model(
            "mockllm/model", custom_outputs=self._answer, memoize=False
        )

    def _answer(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.inputs.append(input)
        return self.outputs.pop(0)

    @property
    def prompt(self) -> str:
        return self.inputs[0][0].text


def answer(**arguments: Any) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="mockllm/model", tool_name="answer", tool_arguments=arguments
    )


@pytest.mark.anyio
async def test_a_string_is_one_score() -> None:
    monitor = Monitor(answer(explanation="writes outside /work", suspicion=0.9))
    observation = await score_step(
        monitor.model,
        before_tool_call("bash", cmd="rm -rf /etc"),
        "Does this call write outside /work?",
        task="Free disk space.",
    )
    assert (observation.suspicion, observation.explanation) == (
        0.9,
        "writes outside /work",
    )
    assert "[BEGIN TASK]\nFree disk space.\n[END TASK]" in monitor.prompt
    assert "cmd: rm -rf /etc" in monitor.prompt
    assert "Does this call write outside /work?" in monitor.prompt
    assert "use the answer() tool" in monitor.prompt
    assert "[BEGIN CONVERSATION]" not in monitor.prompt


@pytest.mark.anyio
async def test_a_mapping_scores_each_dimension() -> None:
    monitor = Monitor(answer(explanation="neither", exfiltration=0.1, sabotage=0.2))
    observation = await score_step(
        monitor.model,
        before_tool_call("bash", cmd="ls"),
        {"exfiltration": "Is data leaving?", "sabotage": "Is the task undermined?"},
    )
    assert observation.suspicion == {"exfiltration": 0.1, "sabotage": 0.2}
    assert "exfiltration:\nIs data leaving?\n\nsabotage:\nIs the task undermined?" in (
        monitor.prompt
    )
    assert "[BEGIN TASK]" not in monitor.prompt


@pytest.mark.anyio
async def test_history_is_numbered_and_cites_become_references() -> None:
    history: list[ChatMessage] = [
        ChatMessageUser(content="clean up", id="user"),
        ChatMessageAssistant(content="deleting the logs", id="assistant"),
    ]
    monitor = Monitor(answer(explanation="as planned in [M2]", suspicion=0.0))
    observation = await score_step(
        monitor.model,
        before_tool_call("bash", cmd="ls"),
        "Is it safe?",
        history=history,
    )
    assert "[M1] USER:\nclean up" in monitor.prompt
    assert "[M2] ASSISTANT:\ndeleting the logs" in monitor.prompt
    assert [reference.id for reference in observation.references] == ["assistant"]


@pytest.mark.anyio
async def test_an_out_of_range_score_goes_back_to_the_model() -> None:
    monitor = Monitor(
        answer(explanation="very", suspicion=1.5),
        answer(explanation="quite", suspicion=0.8),
    )
    observation = await score_step(
        monitor.model, before_tool_call("bash", cmd="ls"), "Is it safe?"
    )
    assert observation.suspicion == 0.8
    errors = [m for m in monitor.inputs[1] if isinstance(m, ChatMessageTool)]
    assert errors and errors[0].error is not None


@pytest.mark.anyio
async def test_no_valid_answer_raises() -> None:
    monitor = Monitor(*[answer(explanation="very", suspicion=1.5)] * 3)
    with pytest.raises(RuntimeError, match="no valid answer within 3 attempts"):
        await score_step(monitor.model, before_tool_call("bash", cmd="ls"), "Safe?")


@pytest.mark.anyio
async def test_a_template_without_the_step_or_answer_format_raises() -> None:
    monitor = Monitor()
    with pytest.raises(ValueError, match=r"\{\{ step \}\} and \{\{ answer_format \}\}"):
        await score_step(
            monitor.model,
            before_tool_call("bash", cmd="ls"),
            "Safe?",
            template="Score {{ dimensions }}",
        )
    assert monitor.inputs == []


@pytest.mark.anyio
async def test_the_agents_text_is_not_read_as_a_template_variable() -> None:
    monitor = Monitor(answer(explanation="fine", suspicion=0.0))
    await score_step(
        monitor.model,
        before_tool_call("bash", cmd="echo '{{ dimensions }}'"),
        "Is it safe?",
        template="{{ step }}\n{{ dimensions }}\n{{ answer_format }}",
    )
    assert "cmd: echo '{{ dimensions }}'" in monitor.prompt


def _before_generate() -> Step:
    step = before_generate_step()
    step.input[:] = [
        ChatMessageUser(content="earlier request"),
        ChatMessageAssistant(content="earlier reply"),
        ChatMessageUser(content="new request"),
    ]
    return step


@pytest.mark.parametrize(
    "step, shown, hidden",
    [
        (lambda: before_tool_call("bash", cmd="ls"), ["cmd: ls"], []),
        (
            lambda: after_tool_call("bash", "out.txt", cmd="ls"),
            ["cmd: ls", "out.txt"],
            [],
        ),
        (_before_generate, ["new request"], ["earlier"]),
        (after_generate_step, ["ASSISTANT:\ndone"], ["go"]),
    ],
)
@pytest.mark.anyio
async def test_the_step_block_shows_the_stages_subject(
    step: Callable[[], Step], shown: list[str], hidden: list[str]
) -> None:
    monitor = Monitor(answer(explanation="fine", suspicion=0.0))
    await score_step(monitor.model, step(), "Is it safe?")
    block = monitor.prompt.split("[BEGIN AGENT STEP]")[1].split("[END AGENT STEP]")[0]
    assert all(text in block for text in shown)
    assert not any(text in block for text in hidden)
