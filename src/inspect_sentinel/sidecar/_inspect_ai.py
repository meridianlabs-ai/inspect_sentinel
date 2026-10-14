"""The parts of inspect_ai the sidecar needs that inspect_ai doesn't yet make public.

Every import of a private inspect_ai module in the sidecar is here and nowhere else, each beside the public function asked of inspect_ai in its place (design/sidecar.md, "Asked of inspect_ai"). When inspect_ai makes one public, it is swapped in here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from inspect_ai.model import ChatMessage, GenerateConfig
from inspect_ai.tool import Tool, ToolCall, ToolCallView, ToolChoice, ToolInfo

logger = logging.getLogger(__name__)

# A request offers the model every tool it declares. The agent bridge withholds
# the tools a provider runs unless its eval grants them, so here every one is
# granted: the sidecar reads a request, and leaves what it offers to the client.
_EVERY_PROVIDER: dict[str, Any] = {}

# the kinds of tool declaration a warning has been logged for
_warned: set[str] = set()


def tool_call_view(call: ToolCall) -> ToolCallView:
    """The view Inspect makes of a call to a tool with no viewer of its own.

    Asked of inspect_ai: a public way to make the default view of a tool call.
    """
    from inspect_ai.tool._tool_call import default_tool_call_viewer

    return default_tool_call_viewer(call)


async def anthropic_message(message: ChatMessage) -> dict[str, Any]:
    """A message written in Anthropic's Messages format, as inspect_ai's provider writes it.

    Asked of inspect_ai: a public function that writes a message in Anthropic's format, as there is for OpenAI's.
    """
    from inspect_ai.model._providers.anthropic import (  # pyright: ignore[reportMissingTypeStubs]
        message_param,
    )

    return cast("dict[str, Any]", await message_param(message))


def anthropic_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an Anthropic Messages request sends the model.

    Asked of inspect_ai: public functions that read these from a request in each provider's format.
    """
    from inspect_ai.agent._bridge.anthropic_api_impl import (
        generate_config_from_anthropic,
        tool_choice_from_anthropic_tool_choice,
        tools_from_anthropic_tools,
    )

    def read(declaration: Any) -> list[ToolInfo | Tool]:
        return tools_from_anthropic_tools(
            [declaration],
            None,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    def read_servers(servers: Any) -> list[ToolInfo | Tool]:
        return tools_from_anthropic_tools(
            None,
            servers,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    servers = request.get("mcp_servers")
    return (
        [
            *_readable("anthropic", request.get("tools"), read),
            *_readable("anthropic", [servers] if servers else [], read_servers),
        ],
        tool_choice_from_anthropic_tool_choice(request.get("tool_choice")),
        generate_config_from_anthropic(request),
    )


def chat_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an OpenAI Chat Completions request sends the model.

    Asked of inspect_ai: as for `anthropic_offered`.
    """
    from inspect_ai.agent._bridge.completions import (
        generate_config_from_openai_completions,
        tool_choice_from_openai_tool_choice,
        tools_from_openai_tools,
    )

    return (
        _readable(
            "openai-chat",
            request.get("tools"),
            lambda declaration: list(tools_from_openai_tools([declaration])),
        ),
        tool_choice_from_openai_tool_choice(request.get("tool_choice")),
        generate_config_from_openai_completions(request),
    )


def responses_offered(
    request: dict[str, Any],
) -> tuple[list[ToolInfo], ToolChoice | None, GenerateConfig]:
    """The tools, tool choice and configuration an OpenAI Responses request sends the model.

    Asked of inspect_ai: as for `anthropic_offered`.
    """
    from inspect_ai.agent._bridge.responses_impl import (
        generate_config_from_openai_responses,
        tool_choice_from_responses_tool_choice,
        tools_from_responses_tool,
    )

    def read(declaration: Any) -> list[ToolInfo | Tool]:
        # a kind inspect_ai doesn't know is left out here, with its own warning
        return tools_from_responses_tool(
            declaration,
            cast(Any, _EVERY_PROVIDER),
            cast(Any, _EVERY_PROVIDER),
            allow_remote_mcp=True,
        )

    return (
        _readable("openai-responses", request.get("tools"), read),
        tool_choice_from_responses_tool_choice(request.get("tool_choice")),
        generate_config_from_openai_responses(request),
    )


def _readable(
    api: str,
    declarations: object,
    read: Callable[[Any], Sequence[ToolInfo | Tool]],
) -> list[ToolInfo]:
    # A declaration that can't be read is left out of what a step is told the
    # model is offered, and the request is judged as usual: a call the model
    # makes to that tool is still in its reply, and is judged there.
    from inspect_ai.tool._tool_util import tool_to_tool_info

    tools: list[ToolInfo] = []
    for declaration in cast("list[Any]", declarations or []):
        try:
            tools.extend(
                tool if isinstance(tool, ToolInfo) else tool_to_tool_info(tool)
                for tool in read(declaration)
            )
        except Exception as ex:
            kind = f"{api} {_kind(declaration)}"
            if kind not in _warned:
                _warned.add(kind)
                logger.warning(
                    "A request offers a tool that can't be read (%s: %s), so a step is not told the model is offered it.",
                    kind,
                    ex,
                )
    return tools


def _kind(declaration: object) -> str:
    if isinstance(declaration, list):
        return "mcp_servers"
    if isinstance(declaration, dict):
        return str(cast("dict[str, Any]", declaration).get("type", "custom"))
    return type(declaration).__name__
