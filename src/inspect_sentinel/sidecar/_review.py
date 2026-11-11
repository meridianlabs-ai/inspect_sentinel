"""Questions a sentinel in the sidecar asks a person, held until an Inspect eval answers them.

A sentinel asks a person with `human()`, through `Host.ask_human`. For a run an Inspect eval names, the sidecar keeps the question here while the step waits, the eval finds it with `pending_questions`, shows it in Inspect's approval UI, and sends the person's answer with `answer_question`.

For now the answer comes on the eval's key, the token it signs its calls with, and is taken only for a question from one of that eval's own runs. That is a temporary exception to the rule that an eval's credential never answers: reviewers are to answer on credentials of their own, which the sidecar checks (`design/sidecar-plan.md`, step 13).
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

import anyio
from inspect_ai.model import ChatMessageTool
from inspect_ai.tool import ToolCall, ToolCallView

from .._host import HumanAnswer
from .._report import Escalations
from ._host import Stage

REVIEW_TIMEOUT = 600.0
"""How long a step waits for a person's answer, in seconds, by default, before it is answered with caution."""

MAX_QUESTIONS_WAIT = 30.0
"""The longest `pending_questions` waits for a question, in seconds."""

ANSWERABLE = ("approve", "reject", "terminate")
"""The choices an answer through the endpoint for evals may pick. `modify` needs a changed call, which Inspect's approval UI can't make yet."""

_KEPT_ANSWERED = 10_000
"""How many answered questions are kept, so that an answer sent again is told it was taken."""


class UnknownQuestionError(Exception):
    """No open or recently answered question has the id, for the eval asking: it never existed, it was another eval's, or nobody answered it in time and it closed."""


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
    """Why it escalated, or None where the eval isn't sent explanations."""


@dataclass
class Question:
    """What a sentinel asked a person, for a run an Inspect eval names."""

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

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    """The question's id, which an answer names."""

    asked: float = field(default_factory=time.time)
    """When it was asked, in seconds since the epoch, for showing how long it has waited."""

    answer: HumanAnswer | None = None
    """The person's answer, once given."""

    answered: anyio.Event = field(default_factory=anyio.Event)
    """Set once the question is answered."""


def escalated(escalations: Escalations, explanations: bool) -> tuple[Escalated, ...]:
    """The escalations as a person is shown them, with their explanations only where the eval is sent them."""
    return tuple(
        Escalated(reported.name, reported.report.explanation if explanations else None)
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
        self._answered: OrderedDict[str, Question] = OrderedDict()
        # set and replaced whenever a question is asked, so that a
        # `pending` that waits wakes at once
        self._asked = anyio.Event()

    @property
    def timeout(self) -> float:
        """How long a step waits for an answer, in seconds."""
        return self._timeout

    async def ask(self, question: Question) -> HumanAnswer:
        """Hold the question until an eval answers it, or the timeout passes.

        Returns:
            The person's answer, or, if nobody answered in time, the cautious one: `reject` if offered, otherwise `terminate`.
        """
        self._open[question.id] = question
        self._asked.set()
        self._asked = anyio.Event()
        try:
            with anyio.move_on_after(self._timeout):
                await question.answered.wait()
        finally:
            # a step that stops waiting, answered or not, takes its question
            # away, so no later answer is taken for a step no longer held
            self._open.pop(question.id, None)
        if question.answer is not None:
            return question.answer
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

    def answer(
        self, eval_id: str, id: str, decision: str, reason: str | None
    ) -> Question:
        """Answer an eval's question, once. An answer sent again, with the same choice and reason, is taken as already given.

        Args:
            eval_id: The eval answering. Another eval's question is unknown to it.
            id: The question's id.
            decision: The choice the person picked.
            reason: The person's reason, if they gave one.

        Returns:
            The question, answered.

        Raises:
            UnknownQuestionError: If no open or recently answered question of the eval has the id.
            AnsweredError: If the question was answered already, otherwise.
            ChoiceError: If `decision` wasn't offered, or is `modify`.
        """
        question = self._open.get(id) or self._answered.get(id)
        if question is None or question.eval_id != eval_id:
            raise UnknownQuestionError(
                f"The sidecar holds no question {id!r} for the eval {eval_id!r}. It may have closed when nobody answered within {self._timeout:g} seconds."
            )
        if question.answer is not None:
            if (question.answer.decision, question.answer.reason) == (decision, reason):
                return question
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
        question.answer = HumanAnswer(decision=decision, reason=reason)
        question.answered.set()
        self._answered[id] = question
        while len(self._answered) > _KEPT_ANSWERED:
            self._answered.popitem(last=False)
        return question
