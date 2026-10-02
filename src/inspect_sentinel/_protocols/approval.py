from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from inspect_ai._util.registry import is_registry_object, registry_info
from inspect_ai.approval import Approval, ApprovalPolicy, Approver
from inspect_ai.approval._call import call_approver
from inspect_ai.approval._policy import (
    approval_policies_from_config,
    policy_approver,
)

from .._context import Context
from .._decorators import protocol
from .._report import Decision
from .._step import BeforeToolCall
from .._types import Protocol

_POLICY_KEYS = {"approver", "tools"}


def as_protocol(approval: Approver | str | Sequence[ApprovalPolicy]) -> Protocol:
    """Run an inspect tool approver, or a list of approval policies, as a protocol before each tool call.

    An approver's decision maps to a decision: `approve` to `continue`, `reject` to `reject` with the approver's explanation as both the `message` the agent reads and the `explanation`, `terminate` to `terminate`, `escalate` to `escalate`, and `modify` to a `modify` with the approver's replacement call. As under `Task(approval=)`, an `approve` or `modify` carrying a replacement call runs that call and one without runs the call unchanged. A replacement call must keep the call's `id` and `function`. The approver's explanation and metadata are the decision's, and each approver call is recorded as an `ApprovalEvent`, as under `Task(approval=)`.

    Given policies, or a string as `Task(approval=)` takes (an approval config file or a registered approver's name), the protocol decides as `Task(approval=)` would: the approvers whose tools match the call are asked in order until one does not escalate, and a call that no approver covers, or that every one escalates, is rejected. Given one approver, its `escalate` is the protocol's decision, so in a `sequential()` the next link decides.

    Args:
        approval: A registered approver, approval policies, or an approval config file or registered approver name.

    Raises:
        TypeError: If an approver is not registered with `@approver`.
    """
    if isinstance(approval, str):
        return approval_protocol(
            _policy_params(approval_policies_from_config(approval))
        )
    if isinstance(approval, Sequence):
        return approval_protocol(_policy_params(approval))
    return approver_protocol(approval)


@protocol(name="approver")
def approver_protocol(approver: Approver) -> Protocol:
    _check_approver(approver)

    async def approve(context: Context, step: BeforeToolCall) -> Decision:
        approval = await call_approver(
            approver, step.message, step.call, step.view, step.history
        )
        return _decision(approval)

    return approve


@protocol(name="approval")
def approval_protocol(policies: str | Sequence[Mapping[str, Any]]) -> Protocol:
    if isinstance(policies, str):
        resolved = approval_policies_from_config(policies)
    else:
        resolved = [_policy(entry, i) for i, entry in enumerate(policies)]
    if not resolved:
        raise ValueError("approval needs at least one policy.")
    for policy in resolved:
        _check_approver(policy.approver)
    approver = policy_approver(resolved)

    async def approve(context: Context, step: BeforeToolCall) -> Decision:
        approval = await approver(step.message, step.call, step.view, step.history)
        return _decision(approval)

    return approve


def _policy_params(policies: Sequence[ApprovalPolicy]) -> list[dict[str, Any]]:
    return [{"approver": p.approver, "tools": p.tools} for p in policies]


def _policy(entry: object, index: int) -> ApprovalPolicy:
    if not isinstance(entry, Mapping) or set(cast(Mapping[str, Any], entry)) != (
        _POLICY_KEYS
    ):
        raise ValueError(
            f"approval policies[{index}] must be a mapping with exactly 'approver' and 'tools', not {entry!r}."
        )
    fields = cast(Mapping[str, Any], entry)
    return ApprovalPolicy(approver=fields["approver"], tools=fields["tools"])


def _check_approver(approver: object) -> None:
    if not is_registry_object(approver) or registry_info(approver).type != "approver":
        raise TypeError(
            f"{getattr(approver, '__name__', approver)!r} is not a configured approver; create it from a factory decorated with @approver."
        )


def _decision(approval: Approval) -> Decision:
    explanation = approval.explanation
    metadata = approval.metadata
    match approval.decision:
        case "approve" | "modify":
            if approval.modified is None:
                return Decision(
                    action="continue", explanation=explanation, metadata=metadata
                )
            return Decision(
                action="modify",
                modified=approval.modified,
                explanation=explanation,
                metadata=metadata,
            )
        case "reject":
            return Decision(
                action="reject",
                explanation=explanation,
                message=explanation,
                metadata=metadata,
            )
        case "terminate" | "escalate":
            return Decision(
                action=approval.decision, explanation=explanation, metadata=metadata
            )
