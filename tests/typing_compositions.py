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
    Modify,
    Monitor,
    Observation,
    Proceed,
    Protocol,
    ProtocolGroup,
    Reject,
    Sentinels,
    Terminate,
    Verdict,
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
    Protocol[Verdict] | ProtocolGroup[Verdict],
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


@protocol
def edits() -> Protocol[Modify | Proceed]:
    async def decide(context: Context, step: BeforeToolCall) -> Modify | Proceed:
        return Decision.proceed()

    return decide


# a contested modify becomes a reject, and a concurrent abstains where none of
# its protocols decides, so both are in its type
assert_type(concurrent([edits()]), Protocol[Modify | Proceed | Reject | None])
contested: Protocol[Modify | Proceed] = concurrent([edits(), edits()])  # pyright: ignore[reportAssignmentType]

# then= ends a chain with a protocol that always decides; a chain ending in one
# can abstain where nothing in it runs, so it is not a then itself
assert_type(
    sequential([triage()], then=handle_escalation("terminate")),
    Protocol[Verdict | None],
)
assert_type(
    sequential([triage()], then=human(stages=["tool_call"])),
    Protocol[Verdict | None],
)
sequential([rule()], then=sequential([triage()], then=handle_escalation("continue")))  # pyright: ignore[reportArgumentType]
sequential([triage()], then=rule())  # pyright: ignore[reportArgumentType]
sequential([triage()], then=anything())  # pyright: ignore[reportArgumentType]

# a configuration that cannot escalate is a Sentinels[Verdict | None]; one that
# can is flagged, whether built by a factory or by a plain helper
handled: Sentinels[Verdict | None] = sequential(
    [triage()], then=human(stages=["tool_call"])
)
quiet: Sentinels[Verdict | None] = {
    "rule": rule(),
    "llm": threshold(score(), reject_at=0.5),
}
watched: Sentinels[Verdict | None] = observe_only(score())
lone: Sentinels[Verdict | None] = triage()  # pyright: ignore[reportAssignmentType]
chained: Sentinels[Verdict | None] = sequential([triage()])  # pyright: ignore[reportAssignmentType]
beside: Sentinels[Verdict | None] = [rule(), triage()]  # pyright: ignore[reportAssignmentType]
untyped: Sentinels[Verdict | None] = anything()  # pyright: ignore[reportAssignmentType]


@protocol
def guarded() -> Protocol[Verdict | None]:
    return sequential([triage()], then=handle_escalation("continue"))


@protocol
def unguarded() -> Protocol[Verdict | None]:
    return sequential([triage()])  # pyright: ignore[reportReturnType]


assert_type(guarded(), Protocol[Verdict | None])


def build() -> Sentinels[Verdict | None]:
    return concurrent(
        [rule(), sequential([triage()], then=handle_escalation("continue"))]
    )


def build_unhandled() -> Sentinels[Verdict | None]:
    return [rule(), triage()]  # pyright: ignore[reportReturnType]
