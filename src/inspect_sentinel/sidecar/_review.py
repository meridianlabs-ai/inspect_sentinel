"""Questions a sentinel in the sidecar asks a person, held until an Inspect eval answers them.

A sentinel asks a person with `human()`, through `Host.ask_human`. For a run an Inspect eval names, the sidecar keeps the question here while the step waits, the eval finds it with `pending_questions`, shows it in Inspect's approval UI, and sends the person's answer with `answer_question`.

This first way of asking has two temporary exceptions to the sidecar's rules, both in `design/sidecar-plan.md`, step 13:

- The answer comes on the eval's key, the token it signs its calls with, and is taken only for a question from one of that eval's own runs. Reviewers are to answer on credentials of their own, which the sidecar checks.
- The question hands the eval's process the call's arguments, even where the person then rejects the call. A reviewer surface outside the eval's process is to show them instead.

A question belongs to one HTTP try of one model request. What became of it is kept in a `Review`, with that try, whether or not the step went on to a decision.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import anyio
from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallView

from .._host import HumanAnswer
from .._report import Escalations
from ._host import Stage

logger = logging.getLogger(__name__)

REVIEW_TIMEOUT = 600.0
"""How long a step waits for a person's answer, in seconds, by default, before it is answered with caution."""

MAX_QUESTIONS_WAIT = 30.0
"""The longest `pending_questions` waits for a question, in seconds."""

ANSWERABLE = ("approve", "reject", "terminate")
"""The choices an answer through the endpoint for evals may pick. `modify` needs a changed call, which Inspect's approval UI can't make yet."""

_KEPT_RECEIPTS = 10_000
"""How many answers' receipts are kept, so that an answer sent again is told it was taken."""

Cause: TypeAlias = Literal[
    "superseded",
    "replaced_attempt",
    "sample_ended",
    "run_ended",
    "time_limit",
    "stopped",
]
"""Why a question's wait was cancelled: a later try of its request began, a later attempt at its sample began, the eval ended its sample, a sentinel ended its run, judging ran out of time, or the sidecar stopped judging for another reason."""


class UnknownQuestionError(Exception):
    """No open question, and no receipt of an answer, has the id for the eval asking: it never existed, it was another eval's, it closed unanswered, or its sample has ended."""


class AnsweredError(Exception):
    """The question was already answered, with another choice or reason."""


class ChoiceError(Exception):
    """The answer picks a choice the question didn't offer, or one the endpoint for evals can't take."""


@dataclass(frozen=True)
class Escalated:
    """An escalation the asking protocol was handed, as a person is shown it."""

    name: str
    """The escalating instance's name."""

    explanation: str | None
    """Why it escalated, if it said."""


@dataclass
class Review:
    """What became of one question, kept with the HTTP try that asked it.

    It holds no arguments, view or result of the call, only the call's id. An answer is recorded when it is taken, apart from how the wait ended: a try stopped just after its answer was taken keeps the answer, which then took no effect.
    """

    id: str
    """The question's id."""

    stage: Stage
    """Where the step was judged."""

    call: str
    """The id of the call the question was about."""

    asked: float
    """When it was asked, in seconds since the epoch."""

    decision: str | None = None
    """What the eval's answer picked, if an answer was taken. The eval's signature shows that the eval sent it, not that a person chose it."""

    reason: str | None = None
    """The reason the eval's answer gave, if any."""

    answered: float | None = None
    """When the answer was taken."""

    resolution: Literal["answered", "timed_out", "cancelled"] | None = None
    """How the wait ended, or None while the step still waits: on the answer, on nobody answering in time, when the step was answered with caution, or cancelled, when the step came to no decision."""

    cause: Cause | None = None
    """Why the wait was cancelled, when it was."""

    resolved: float | None = None
    """When the wait ended."""

    def resolve(
        self, resolution: Literal["answered", "timed_out", "cancelled"]
    ) -> None:
        """Mark how the wait ended, once."""
        if self.resolution is None:
            self.resolution, self.resolved = resolution, time.time()
            if resolution == "cancelled" and self.cause is None:
                self.cause = "stopped"


@dataclass
class Question:
    """What a sentinel asked a person, for a run an Inspect eval names. Held only while the step waits."""

    eval_id: str
    """The eval whose run asked, the only one that may see and answer the question."""

    sample_uuid: str
    """The sample the run is, as the eval names it."""

    request_id: str
    """The eval's id for the model request the step was found in, its `x-irid`."""

    stage: Stage
    """Where the step was judged: `tool_call` or `tool_result`."""

    call: ToolCall
    """The call the person decides about, as it will run."""

    message: str
    """The assistant's text alongside the call."""

    view: ToolCallView
    """How the call is shown."""

    result: ChatMessageTool | None
    """The call's result, at `tool_result`."""

    escalations: tuple[Escalated, ...]
    """The escalations the asking protocol was handed, in order."""

    choices: tuple[str, ...]
    """What the person may pick."""

    closes: float
    """When the step stops waiting, in seconds since the epoch: when the review timeout passes, or sooner if judging runs out of time first."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """The question's id, which an answer names."""

    asked: float = field(default_factory=time.time)
    """When it was asked, in seconds since the epoch."""

    answer: HumanAnswer | None = None
    """The eval's answer, once taken."""

    answered: anyio.Event = field(default_factory=anyio.Event)
    """Set once the question is answered."""

    review: Review = field(init=False)
    """What becomes of the question, which outlives it."""

    def __post_init__(self) -> None:
        self.review = Review(self.id, self.stage, self.call.id, self.asked)


@dataclass(frozen=True)
class _Receipt:
    """That a question was answered, and how: all that is kept of it once its step stops waiting."""

    eval_id: str
    sample_uuid: str
    decision: str
    reason: str | None


def escalated(escalations: Escalations) -> tuple[Escalated, ...]:
    """The escalations as a person is shown them."""
    return tuple(
        Escalated(reported.name, reported.report.explanation)
        for reported in escalations
    )


def cautious(choices: Sequence[str], reason: str) -> HumanAnswer:
    """The answer for a person who didn't answer: `reject` if it was offered, otherwise `terminate`, so the step fails closed."""
    return HumanAnswer(
        decision="reject" if "reject" in choices else "terminate", reason=reason
    )


class Reviews:
    """The questions the sidecar holds for Inspect evals to answer. Kept in this process, as a step waits on its question here."""

    def __init__(self, timeout: float = REVIEW_TIMEOUT) -> None:
        """Hold no questions yet.

        Args:
            timeout: How long a step waits for an answer, in seconds, before it is answered with caution.

        Raises:
            ValueError: If `timeout` isn't a positive number of seconds.
        """
        if not timeout > 0:
            raise ValueError(f"The review timeout is some seconds, not {timeout}.")
        self._timeout = timeout
        self._open: dict[str, Question] = {}
        self._receipts: OrderedDict[str, _Receipt] = OrderedDict()
        # set and replaced whenever a question is asked, so that a
        # `pending` that waits wakes at once
        self._asked = anyio.Event()

    @property
    def timeout(self) -> float:
        """How long a step waits for an answer, in seconds."""
        return self._timeout

    async def ask(self, question: Question) -> HumanAnswer:
        """Hold the question until an eval answers it, or the timeout passes.

        The wait is cancelled from outside, by cancelling the judging it is part of, when the question no longer has anyone to answer to: `close` takes the question away at once, and the step then comes to no decision.

        Returns:
            The eval's answer, or, if nobody answered in time, the cautious one: `reject` if offered, otherwise `terminate`.
        """
        review = question.review
        self._open[question.id] = question
        self._asked.set()
        self._asked = anyio.Event()
        logger.info(
            "Asked a person about the call to %s at %s, as question %s of the eval %s.",
            question.call.function,
            question.stage,
            question.id,
            question.eval_id,
        )
        try:
            with anyio.move_on_after(self._timeout):
                await question.answered.wait()
        except anyio.get_cancelled_exc_class():
            # an answer taken just before stays on the record, and takes no effect
            review.resolve("cancelled")
            logger.info("Question %s was cancelled (%s).", question.id, review.cause)
            raise
        finally:
            # a step that stops waiting takes its question away, so no later
            # answer is taken for a step no longer held
            self._open.pop(question.id, None)
        if question.answer is not None:
            review.resolve("answered")
            return question.answer
        review.resolve("timed_out")
        logger.warning(
            "Nobody answered question %s within %g seconds, so the %s stage of the call to %s was answered with caution.",
            question.id,
            self._timeout,
            question.stage,
            question.call.function,
        )
        return cautious(
            question.choices,
            f"Nobody answered within {self._timeout:g} seconds, so the sidecar answered with caution.",
        )

    async def pending(
        self, eval_id: str, sample_uuid: str | None = None, wait: float = 0
    ) -> list[Question]:
        """The eval's unanswered questions, oldest first.

        Args:
            eval_id: The eval asking. Another eval's questions are never shown.
            sample_uuid: Only this sample's, if given.
            wait: How long to wait, in seconds, while there are none, at most `MAX_QUESTIONS_WAIT`. The answer comes as soon as there is one.
        """
        deadline = time.monotonic() + min(max(wait, 0), MAX_QUESTIONS_WAIT)
        while True:
            found = sorted(
                (
                    question
                    for question in self._open.values()
                    if question.eval_id == eval_id
                    and question.answer is None
                    and sample_uuid in (None, question.sample_uuid)
                ),
                key=lambda question: question.asked,
            )
            left = deadline - time.monotonic()
            if found or left <= 0:
                return found
            asked = self._asked
            with anyio.move_on_after(left):
                await asked.wait()

    def answer(self, eval_id: str, id: str, decision: str, reason: str | None) -> None:
        """Take an eval's answer to its question, once. An answer sent again, with the same choice and reason, is taken as already given.

        The answer is put on the question's record before this returns, so the record holds every answer that was acknowledged, including one whose step is stopped before the answer takes effect.

        Args:
            eval_id: The eval answering. Another eval's question is unknown to it.
            id: The question's id.
            decision: The choice the answer picks.
            reason: The reason given, if any.

        Raises:
            UnknownQuestionError: If the eval has no open question with the id, and no receipt of an answer to it.
            AnsweredError: If the question was answered already, otherwise.
            ChoiceError: If `decision` wasn't offered, or is `modify`.
        """
        receipt = self._receipts.get(id)
        if receipt is not None and receipt.eval_id == eval_id:
            if (receipt.decision, receipt.reason) == (decision, reason):
                return
            raise AnsweredError(
                f"The question {id!r} was already answered {receipt.decision!r}."
            )
        question = self._open.get(id)
        if receipt is not None or question is None or question.eval_id != eval_id:
            raise UnknownQuestionError(
                f"The sidecar holds no question {id!r} for the eval {eval_id!r}. It may have closed unanswered, or its sample may have ended."
            )
        if question.answer is not None:
            # taken already, and its receipt no longer kept: the first stands
            if (question.answer.decision, question.answer.reason) == (decision, reason):
                return
            raise AnsweredError(
                f"The question {id!r} was already answered {question.answer.decision!r}."
            )
        if decision not in question.choices:
            raise ChoiceError(
                f"The question {id!r} offers {', '.join(question.choices)}, not {decision!r}."
            )
        if decision not in ANSWERABLE:
            raise ChoiceError(
                f"An answer through the endpoint for evals is one of {', '.join(ANSWERABLE)}, not {decision!r}."
            )
        review = question.review
        review.decision, review.reason, review.answered = decision, reason, time.time()
        question.answer = HumanAnswer(decision=decision, reason=reason)
        question.answered.set()
        self._receipts[id] = _Receipt(eval_id, question.sample_uuid, decision, reason)
        while len(self._receipts) > _KEPT_RECEIPTS:
            self._receipts.popitem(last=False)
        logger.info("The eval %s answered question %s: %s.", eval_id, id, decision)

    def close(self, reviews: Sequence[Review], cause: Cause) -> None:
        """Close the questions of these reviews now, as cancelled for `cause`. From here on none is listed, and an answer to one is for an unknown question, whether or not the step that waits on it has stopped waiting yet."""
        for review in reviews:
            self._open.pop(review.id, None)
            if review.resolution is None:
                review.cause = cause
                review.resolve("cancelled")

    def forget_sample(self, eval_id: str, sample_uuid: str) -> None:
        """Drop the receipts of a sample's answers, once the eval has ended the sample. An answer to one of its questions sent again after this is for an unknown question."""
        for id in [
            id
            for id, receipt in self._receipts.items()
            if (receipt.eval_id, receipt.sample_uuid) == (eval_id, sample_uuid)
        ]:
            del self._receipts[id]
