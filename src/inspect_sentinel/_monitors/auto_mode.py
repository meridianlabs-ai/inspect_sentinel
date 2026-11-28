from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inspect_ai.model import GenerateConfig

from .._context import Context
from .._decorators import monitor, recorded_always
from .._host import Host, role_configured
from .._preprocess import LastTurns, StepPreprocessor
from .._report import Observation
from .._step import BeforeToolCall
from .._types import Monitor
from .auto_mode_policy import (
    AutoModePolicy,
    category_of,
    deny_rule_names,
    policy_digest,
    policy_text,
    recorded_policy,
    resolve_policy,
)
from .auto_mode_prompt import RISK, instructions

_ROLE = "auto_mode"
_FALLBACK_ROLE = "monitor"


@monitor(version=2)
@recorded_always(policy=recorded_policy)
def auto_mode(
    policy: AutoModePolicy | Mapping[str, Any] | str | None = None,
    *,
    model: str | None = None,
    role: str | None = None,
    window: int = 20,
    include_assistant_text: bool = False,
    config: GenerateConfig | Mapping[str, Any] | None = None,
) -> Monitor:
    """Judge each tool call before it runs, as the auto modes of coding agents do: a model scores how much risk the call carries against a policy, so `threshold()` can let it run, ask a person, or refuse it.

    The judge model is shown the policy, the task, the agent's last `window` turns and the call, and answers with one score, `risk`, from 0.0 to 1.0, where 0.5 is the line between a call that may run without a person and one that should not. Its explanation begins with the deny rule that decided the score, which the observation records as `metadata["category"]` (`None` when no rule applies), beside `metadata["policy_digest"]`, which identifies the policy the score was made under. The configuration records the policy with every `"$defaults"` written out, so a replay judges with the policy the eval ran even after the presets change. The agent's own text and reasoning are left out by default, so the agent can't talk the judge into a low score; its tool calls and their results are shown, with every message marked by the sample's nonce.

    By default the judge runs on the `auto_mode` model role, or on the `monitor` role when `auto_mode` is not configured. When neither is configured and no `model` is given, the monitor raises rather than fall back to the agent's own model, which would judge its own calls; under `threshold()`, that fails the sample. A judge model that gives no valid answer, or that refuses, also raises, and content the judge can't be shown, such as an image, scores 1.0 without asking the model (`on_unrenderable="flag"`), so the monitor fails closed.

    The monitor only scores. Which scores refuse a call is `threshold()`'s to decide, as in `threshold(auto_mode(), reject_at=0.7)`; the agent is told only that the call was not approved, never the judge's explanation.

    Args:
        policy: What calls are judged against: `None` or `"inspect"` for the default policy, written for an agent in a sandbox that is discarded after its run; an `AutoModePolicy`; or a mapping of its fields, as a configuration file gives it.
        model: A model name for the judge, used when `role` (by default `auto_mode`) is not configured.
        role: The model role the judge runs on, in place of `auto_mode`. With `role` given, the `monitor` role is not tried.
        window: How many of the agent's most recent turns the judge is shown, as `LastTurns(window)`.
        include_assistant_text: Show the judge the agent's own text and reasoning, not only its calls and their results.
        config: Generation configuration for the judge's calls, such as a `timeout`; a call that times out fails the monitor.

    Raises:
        ValueError: If `policy` names no preset or is not a valid policy, `window` is less than 1, or `role` is empty.
    """
    resolved = resolve_policy(policy)
    if window < 1:
        raise ValueError(f"auto_mode() needs window >= 1, got {window}.")
    if role == "":
        raise ValueError(
            "auto_mode() role must be a model role, not an empty string. Pass None for the 'auto_mode' role."
        )
    # a configuration rebuilds the config from its recorded fields, as a mapping
    generate_config = (
        GenerateConfig.model_validate(dict(config))
        if isinstance(config, Mapping)
        else config
    )
    text = policy_text(resolved)
    names = deny_rule_names(resolved)
    digest = policy_digest(resolved)
    preprocessor = StepPreprocessor(
        input_window=LastTurns(window),
        omit_assistant_text=not include_assistant_text,
        omit_reasoning=not include_assistant_text,
    )

    async def judge(context: Context, step: BeforeToolCall) -> Observation:
        judge_model, judge_role = _judge_model(context.host, model, role)
        observation = await context.host.score(
            step,
            {"risk": RISK},
            preprocessor=preprocessor,
            instructions=instructions(in_eval=context.eval is not None),
            policy=text,
            model=judge_model,
            role=judge_role,
            config=generate_config,
            on_unrenderable="flag",
        )
        return _with_category(observation, names, digest)

    return judge


def _judge_model(
    host: Host, model: str | None, role: str | None
) -> tuple[str | None, str | None]:
    # the (model, role) the judge asks the host for: the role if configured,
    # then the model, then (when no role was named) the monitor role. Never
    # the host's own fallback, which in an eval is the agent's model
    asked = role or _ROLE
    if role_configured(host, asked):
        return None, asked
    if model is not None:
        return model, None
    if role is None and role_configured(host, _FALLBACK_ROLE):
        return None, _FALLBACK_ROLE
    roles = f"'{asked}'" if role is not None else f"'{_ROLE}' or '{_FALLBACK_ROLE}'"
    raise RuntimeError(
        f"auto_mode() has no model to judge with: no model is configured for the {roles} role, and no model= was given. "
        f"Set one with Task(model_roles={{'{asked}': ...}}) or --model-role {asked}=<model>. "
        "It does not fall back to the agent's own model, which would judge its own calls."
    )


def _with_category(
    observation: Observation, names: list[str], digest: str
) -> Observation:
    # the deny rule the explanation's first line names, as the policy names it
    # (None for "Rule: none", for a name the policy doesn't have, or for no
    # such line), and which policy the score was made under
    lines = (observation.explanation or "").strip().splitlines()
    first = lines[0].strip() if lines else ""
    label, _, named = first.partition(":")
    label = label.strip()
    is_rule = label.replace("*", "").replace("`", "").strip().casefold() == "rule"
    # a bold label, "**Rule:** name", leaves its closing "**" right after the
    # colon; it belongs to the label, so remove it before matching. Anything
    # else, including asterisks after a plain "Rule:", is part of the name
    bold_label = label.startswith("**") and not label.endswith("**")
    if bold_label:
        named = named.removeprefix("**")
    named = named.strip()
    category = category_of(named, names) if is_rule else None
    return observation.model_copy(
        update={
            "metadata": {
                **(observation.metadata or {}),
                "category": category,
                "policy_digest": digest,
            }
        }
    )
