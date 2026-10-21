from __future__ import annotations

from typing import Literal, TypeAlias

from inspect_scout import Transcript

from inspect_sentinel import ReplayStep, steps_from_messages

Turns: TypeAlias = Literal["recorded", "inferred", "stored"]


def transcript_steps(transcript: Transcript) -> tuple[list[ReplayStep], Turns]:
    # The seam where inspect_ai's log driver plugs in (R7 PR 3):
    # steps_from_events(transcript.messages, transcript.events,
    # transcript.events_data["messages"], sample_uuid=..., eval_info=...,
    # timeline=..., human_approvers=...), over the condensed events the
    # scanner asks for, with "recorded" turns where model events carry an
    # origin. Until it lands every transcript replays from its messages, whose
    # turns are inferred: each assistant message is taken for an agent turn.
    steps = steps_from_messages(
        transcript.messages, conversation=transcript.transcript_id
    )
    return list(steps), "inferred"
