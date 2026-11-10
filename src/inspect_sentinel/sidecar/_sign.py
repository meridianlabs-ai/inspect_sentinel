"""The HMACs an Inspect eval proves its model requests and its calls to the endpoint for evals with.

Each is an HMAC-SHA256, in hex, made with the eval's token over a JSON list of strings. Inspect makes the token, and sends it only in a `register_task` to a sidecar that holds no token for the eval. Every other call, and every model request, carries a `timestamp` and a `mac` in its place.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Sequence

from .._context import InspectSample

TIMESTAMP_WINDOW = 300.0
"""How far, in seconds, a request's or a call's `timestamp` may be from the sidecar's clock, by default."""

MIN_TOKEN_LENGTH = 32
"""The fewest characters an eval's token has."""


def sign(token: str, values: Sequence[object]) -> str:
    """The HMAC of `values`, each as a string, made with the eval's token.

    Args:
        token: The eval's token.
        values: What is signed, in order. Each is written as `str` writes it, as Inspect takes sample ids to be unique by their string form.
    """
    message = json.dumps([str(value) for value in values], separators=(",", ":"))
    return hmac.new(token.encode(), message.encode(), hashlib.sha256).hexdigest()


def request_values(
    sample: InspectSample, request_id: str, timestamp: int
) -> list[object]:
    """What a model request's `mac` signs: the values of its `x-inspect-context` that name its run, its `x-irid`, and its `timestamp`.

    The `x-irid` is the eval's id for the request, under which the sidecar keeps what it decided of the request for the eval. Signing it means a `mac` made for one request can't put what was decided of its sample under another request's id. The eval's id isn't among the values, since it picks the token, and neither is the model, which names no run.
    """
    return [
        "model-request",
        sample.task,
        sample.sample_id,
        sample.epoch,
        sample.sample_uuid,
        sample.attempt,
        sample.attempt_number,
        request_id,
        timestamp,
    ]


def call_values(method: str, eval_id: str, timestamp: int) -> list[object]:
    """What a call's `mac` signs: the method, the eval and the `timestamp`. No method is named `model-request`, so one can't be taken for the other."""
    return [method, eval_id, timestamp]


def unproven(
    token: str,
    values: Sequence[object],
    timestamp: int,
    mac: str,
    *,
    window: float,
    what: str,
) -> str | None:
    """Why a `timestamp` and `mac` don't prove what they were sent with, or None if they do.

    Args:
        token: The eval's token, as the sidecar holds it.
        values: What the `mac` signs.
        timestamp: When the sender says it signed, in seconds since the epoch.
        mac: The HMAC the sender made.
        window: How far the `timestamp` may be from the sidecar's clock, in seconds.
        what: What was sent, to begin the reason with, such as `The call's`.

    Returns:
        The reason, which never holds the `mac` the sidecar expected.
    """
    # How far behind the sidecar's clock, so a clock problem shows as one.
    # Whole seconds, as a timestamp is, because a sender can send an integer
    # too large to become a float, and that must be refused, not raise.
    behind = int(time.time()) - timestamp
    if abs(behind) > window:
        way = "behind" if behind > 0 else "ahead of"
        far = f"{abs(behind)} seconds" if abs(behind) < 10**12 else "far"
        return f"{what} timestamp is {far} {way} the sidecar's clock, more than the {window:g} seconds allowed. If the eval sent it just now, the eval's clock and the sidecar's differ."

    # A mac is hex, so anything else is refused before it is compared. The
    # comparison fails closed: if the sent values can't be signed at all, the
    # request is refused, never let through as the sentinel's failure.
    wrong = f"{what} mac isn't one made with the eval's token from the values sent."
    if not mac.isascii():
        return wrong
    try:
        expected = sign(token, values)
    except Exception:
        return wrong
    if not hmac.compare_digest(expected.encode(), mac.encode()):
        return wrong
    return None
