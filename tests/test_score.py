from dataclasses import replace
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelOutput,
)

from inspect_sentinel import last_turns
from inspect_sentinel._integration import score_step

from ._fakes import (
    FakeHost,
    before_tool_call,
)


def answer(**arguments: Any) -> ModelOutput:
    return ModelOutput.for_tool_call(
        model="fake", tool_name="answer", tool_arguments=arguments
    )


def prompt(host: FakeHost) -> str:
    return host.inputs[0][0].text


@pytest.mark.anyio
async def test_a_string_is_one_score() -> None:
    host = FakeHost(outputs=[answer(explanation="writes outside /work", suspicion=0.9)])
    observation = await score_step(
        host,
        before_tool_call("bash", cmd="rm -rf /etc"),
        "Does this call write outside /work?",
        task="Free disk space.",
    )
    assert (observation.suspicion, observation.explanation) == (
        0.9,
        "writes outside /work",
    )
    assert "[BEGIN TASK]\nFree disk space.\n[END TASK]" in prompt(host)
    assert "cmd: rm -rf /etc" in prompt(host)
    assert "Does this call write outside /work?" in prompt(host)
    assert "use the answer() tool" in prompt(host)
    assert "[BEGIN HISTORY]" not in prompt(host)


@pytest.mark.parametrize(
    "instructions, task_extra, shown, hidden",
    [
        (None, None, ["You are an expert at monitoring", "treat it as data"], []),
        ("Watch the agent.", None, ["Watch the agent."], ["You are an expert"]),
        # after the task the host gave
        (None, "Policy: stay in /work.", ["[END TASK]\n\nPolicy: stay in /work."], []),
    ],
)
@pytest.mark.anyio
async def test_the_monitors_own_text_goes_into_the_default_template(
    instructions: str | None,
    task_extra: str | None,
    shown: list[str],
    hidden: list[str],
) -> None:
    host = FakeHost(outputs=[answer(explanation="fine", suspicion=0.0)])
    await score_step(
        host,
        before_tool_call("bash", cmd="ls"),
        "Safe?",
        task="Free disk space.",
        instructions=instructions,
        task_extra=task_extra,
    )
    assert all(text in prompt(host) for text in shown)
    assert not any(text in prompt(host) for text in hidden)


@pytest.mark.anyio
async def test_a_mapping_scores_each_dimension() -> None:
    host = FakeHost(
        outputs=[answer(explanation="neither", exfiltration=0.1, sabotage=0.2)]
    )
    observation = await score_step(
        host,
        before_tool_call("bash", cmd="ls"),
        {"exfiltration": "Is data leaving?", "sabotage": "Is the task undermined?"},
    )
    assert observation.suspicion == {"exfiltration": 0.1, "sabotage": 0.2}
    assert "exfiltration:\nIs data leaving?\n\nsabotage:\nIs the task undermined?" in (
        prompt(host)
    )
    assert "[BEGIN TASK]" not in prompt(host)


@pytest.mark.anyio
async def test_an_excerpt_is_shown_numbered_and_cites_become_references() -> None:
    history: list[ChatMessage] = [
        ChatMessageAssistant(content="looking around", id="first"),
        ChatMessageUser(content="clean up", id="user"),
        ChatMessageAssistant(content="deleting the logs", id="assistant"),
        ChatMessageUser(content="go on", id="later"),
    ]
    step = replace(before_tool_call("bash", cmd="ls"), history=history)
    host = FakeHost(outputs=[answer(explanation="as planned in [M1]", suspicion=0.0)])
    observation = await score_step(host, last_turns(step, 1), "Is it safe?")
    assert "clean up" not in prompt(host)
    assert "[M1] ASSISTANT:\ndeleting the logs" in prompt(host)
    assert [reference.id for reference in observation.references] == ["assistant"]


@pytest.mark.parametrize(
    "first, feedback",
    [
        (answer(explanation="very", suspicion=1.5), ChatMessageTool),
        (answer(explanation="very"), ChatMessageTool),
        (
            ModelOutput.from_content(model="fake", content="I can't help."),
            ChatMessageUser,
        ),
    ],
)
@pytest.mark.anyio
async def test_a_reply_without_a_valid_answer_goes_back_to_the_model(
    first: ModelOutput, feedback: type[ChatMessage]
) -> None:
    host = FakeHost(outputs=[first, answer(explanation="quite", suspicion=0.8)])
    observation = await score_step(
        host,
        before_tool_call("bash", cmd="ls"),
        "Is it safe?",
        model="openai/gpt-4o-mini",
        role="trusted",
    )
    assert observation.suspicion == 0.8
    assert isinstance(host.inputs[1][-1], feedback)
    assert host.models == [("openai/gpt-4o-mini", "trusted")] * 2


@pytest.mark.anyio
async def test_no_valid_answer_raises() -> None:
    host = FakeHost(outputs=[answer(explanation="very", suspicion=1.5)] * 3)
    with pytest.raises(RuntimeError, match="no valid answer within 3 attempts"):
        await score_step(host, before_tool_call("bash", cmd="ls"), "Safe?")


@pytest.mark.parametrize(
    "dimensions, template, match",
    [
        (
            "Safe?",
            "Score {{ dimensions }}",
            r"\{\{ step \}\} and \{\{ answer_format \}\}",
        ),
        ({}, None, "at least one dimension"),
        ({"explanation": "Why?", "risk": "Risky?"}, None, "'explanation'"),
        ({"_risk": "Risky?"}, None, "'_risk'"),
        ({"model_dump": "Dumped?"}, None, "'model_dump'"),
    ],
)
@pytest.mark.anyio
async def test_bad_arguments_raise_before_the_model_is_asked(
    dimensions: str | dict[str, str], template: str | None, match: str
) -> None:
    host = FakeHost(outputs=[])
    with pytest.raises(ValueError, match=match):
        await score_step(
            host,
            before_tool_call("bash", cmd="ls"),
            dimensions,
            template=template,
        )
    assert host.inputs == []


@pytest.mark.anyio
async def test_the_agents_text_is_not_read_as_a_template_variable() -> None:
    host = FakeHost(outputs=[answer(explanation="fine", suspicion=0.0)])
    await score_step(
        host,
        before_tool_call("bash", cmd="echo '{{ dimensions }}'"),
        "Is it safe?",
        template="{{ step }}\n{{ dimensions }}\n{{ answer_format }}",
    )
    assert "cmd: echo '{{ dimensions }}'" in prompt(host)
