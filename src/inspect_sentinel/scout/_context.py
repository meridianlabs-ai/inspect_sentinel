from __future__ import annotations

from inspect_ai.log import EvalLog, read_eval_log_async, read_eval_log_sample_async
from inspect_ai.model import ChatMessageUser
from inspect_scout import Transcript

from inspect_sentinel import EvalContext

# the sample record's fields the context reads, and what it reads them with;
# the rest of the sample, its events and messages, Scout has already read
_SKIPPED = {"events", "messages", "store", "events_data", "timelines", "scores"}


class Contexts:
    # the context of each transcript a scanner is handed, reading each log's
    # header once
    def __init__(self) -> None:
        self._headers: dict[str, EvalLog] = {}

    async def eval_context(self, transcript: Transcript) -> EvalContext | None:
        if transcript.source_type == "eval_log" and transcript.source_uri:
            return await self._from_log(transcript, transcript.source_uri)
        return _from_info(transcript)

    async def _from_log(self, transcript: Transcript, log: str) -> EvalContext:
        header = self._headers.get(log)
        if header is None:
            header = await read_eval_log_async(log, header_only=True)
            self._headers[log] = header
        # the sample record's own values: Scout's columns hold its id as a
        # string, which would replay an id of 7 as "7"
        sample = await read_eval_log_sample_async(
            log,
            uuid=transcript.transcript_id,
            resolve_attachments=True,
            exclude_fields=_SKIPPED,
        )
        return EvalContext(
            task=header.eval.task,
            task_description=header.eval.task_description,
            sample_id=sample.id,
            epoch=sample.epoch,
            sample_description=sample.description,
            sample_input=sample.input,
            metadata={**(header.eval.metadata or {}), **sample.metadata},
        )


def _from_info(transcript: Transcript) -> EvalContext | None:
    if transcript.task_set is None or transcript.task_id is None:
        return None
    first = next(
        (m for m in transcript.messages if isinstance(m, ChatMessageUser)), None
    )
    return EvalContext(
        task=transcript.task_set,
        task_description=None,
        sample_id=transcript.task_id,
        epoch=transcript.task_repeat or 1,
        sample_description=None,
        sample_input=first.text if first is not None else "",
        metadata=dict(transcript.metadata),
    )
