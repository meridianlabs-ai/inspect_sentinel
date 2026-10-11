# pyright: reportUnnecessaryTypeIgnoreComment=error
# Checked by pyright, not run: each `# pyright: ignore[...]` marks an error the
# types must report, and an unnecessary one is itself an error, so a check that
# stops firing fails the type check.
from typing_extensions import assert_type

from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Escalate,
    Monitor,
    NotEscalate,
    Observation,
    Proceed,
    Protocol,
    ProtocolGroup,
    Reject,
    RootSentinels,
    Terminate,
    concurrent,
    handle_escalation,
    human,
    monitor,
    observe_only,
    protocol,
    sequential,
    threshold,
)


@monitor
def score() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def rule() -> Protocol[Reject | Proceed | None]:
    async def decide(context: Context, step: BeforeToolCall) -> Reject | Proceed | None:
        return None

    return decide


@protocol
def triage() -> Protocol[Reject | Escalate | Proceed]:
    async def decide(
        context: Context, step: BeforeToolCall
    ) -> Reject | Escalate | Proceed:
        return Decision.escalate("unsure")

    return decide


@protocol
def anything() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return None

    return decide


# a factory's instances take the type of its return annotation
assert_type(rule(), Protocol[Reject | Proceed | None])
assert_type(anything(), Protocol[Decision | None])

# the shipped protocols
assert_type(
    threshold(score(), reject_at=0.5), Protocol[Terminate | Reject | Proceed | None]
)
assert_type(observe_only(score()), Protocol[None])
assert_type(handle_escalation("terminate"), ProtocolGroup[Terminate | Proceed])
assert_type(
    human(stages=["tool_call"]),
    Protocol[NotEscalate] | ProtocolGroup[NotEscalate],
)

# compositions take the union of their children
assert_type(
    concurrent([rule(), triage(), score()]),
    Protocol[Reject | Proceed | Escalate | None],
)
assert_type(
    concurrent({"rule": rule(), "llm": threshold(score(), reject_at=0.5)}),
    Protocol[Reject | Proceed | Terminate | None],
)
assert_type(sequential([triage()]), Protocol[Reject | Escalate | Proceed | None])
assert_type(concurrent([score()]), Protocol[None])

# then= ends a chain with a protocol that always decides
assert_type(
    sequential([triage()], then=handle_escalation("terminate")),
    Protocol[NotEscalate],
)
assert_type(
    sequential([triage()], then=human(stages=["tool_call"])), Protocol[NotEscalate]
)
sequential([triage()], then=rule())  # pyright: ignore[reportArgumentType]
sequential([triage()], then=anything())  # pyright: ignore[reportArgumentType]

# a root that cannot escalate is a RootSentinels; one that can is flagged
handled: RootSentinels = sequential([triage()], then=human(stages=["tool_call"]))
quiet: RootSentinels = {"rule": rule(), "llm": threshold(score(), reject_at=0.5)}
watched: RootSentinels = observe_only(score())
lone: RootSentinels = triage()  # pyright: ignore[reportAssignmentType]
chained: RootSentinels = sequential([triage()])  # pyright: ignore[reportAssignmentType]
beside: RootSentinels = [rule(), triage()]  # pyright: ignore[reportAssignmentType]
untyped: RootSentinels = anything()  # pyright: ignore[reportAssignmentType]


def build() -> RootSentinels:
    return concurrent(
        [rule(), sequential([triage()], then=handle_escalation("continue"))]
    )
