"""M3: pydantic models for the step and the report, validated in the guest."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    function: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]


class BeforeToolCall(BaseModel):
    stage: Literal["before_tool_call"]
    conversation: str
    message: str
    call: ToolCall
    input: list[Message]
    history: list[Message] = []


class Report(BaseModel):
    suspicion: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str
