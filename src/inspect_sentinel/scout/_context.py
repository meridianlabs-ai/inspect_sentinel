from __future__ import annotations

from typing import TypeAlias

import anyio

# inspect_ai's designated boundary for sentinel, not inspect_ai's public API
from inspect_ai._sentinel._integration import EvalInfo
from inspect_ai.log import (
    EvalLog,
    read_eval_log_async,
    read_eval_log_sample_async,
    read_eval_log_sample_summaries_async,
)
from inspect_ai.model import ChatMessageUser
from inspect_scout import Transcript

from inspect_sentinel import EvalContext

# the sample record's fields the context reads, and what it reads them with;
# the rest of the sample, its events and messages, Scout has already read
_SKIPPED = {"events", "messages", "store", "events_data", "timelines", "scores"}


# a sample's uuid, or its id as a string and its epoch
_SampleKey: TypeAlias = str | tuple[str | None, int | None]


class Contexts:
    # the context of each transcript a scanner is handed, reading each log's
    # header and sample summaries once
    def __init__(self) -> None:
        self._headers: dict[str, EvalLog] = {}
        self._samples: dict[str, dict[_SampleKey, tuple[int | str, int]]] = {}
        # one read of a log at a time, so transcripts from one log that miss
        # the cache together share the read the first makes; a read that
        # fails or is cancelled caches nothing, and the next tries again
        self._reading: dict[str, anyio.Lock] = {}

    async def eval_context(self, transcript: Transcript) -> EvalContext | None:
        if transcript.source_type == "eval_log" and transcript.source_uri:
            return await self._from_log(transcript, transcript.source_uri)
        return _from_info(transcript)

    async def eval_info(self, transcript: Transcript) -> EvalInfo:
        # what the log driver reads from the log's header
        if transcript.source_type == "eval_log" and transcript.source_uri:
            return EvalInfo.from_eval((await self._header(transcript.source_uri)).eval)
        return EvalInfo(inspect_ai_version=None, sentinel=False)

    async def _header(self, log: str) -> EvalLog:
        header = self._headers.get(log)
        if header is not None:
            return header
        async with self._lock(log):
            header = self._headers.get(log)
            if header is None:
                header = await read_eval_log_async(log, header_only=True)
                self._headers[log] = header
        return header

    def _lock(self, log: str) -> anyio.Lock:
        return self._reading.setdefault(log, anyio.Lock())

    async def _from_log(self, transcript: Transcript, log: str) -> EvalContext:
        header = await self._header(log)
        # the sample record's own values: Scout's columns hold its id as a
        # string, which would replay an id of 7 as "7"
        sample_id, epoch = await self._identity(log, transcript)
        sample = await read_eval_log_sample_async(
            log,
            id=sample_id,
            epoch=epoch,
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

    async def _identity(
        self, log: str, transcript: Transcript
    ) -> tuple[int | str, int]:
        # Scout's transcript id is the sample's uuid; a sample without one is
        # found by the id and epoch Scout records as its task id and repeat,
        # the id as a string, which names one sample since inspect_ai refuses
        # ids that share a string
        samples = self._samples.get(log)
        if samples is None:
            async with self._lock(log):
                samples = self._samples.get(log)
                if samples is None:
                    samples = dict[_SampleKey, tuple[int | str, int]](
                        (s.uuid or (str(s.id), s.epoch), (s.id, s.epoch))
                        for s in await read_eval_log_sample_summaries_async(log)
                    )
                    self._samples[log] = samples
        found = samples.get(transcript.transcript_id) or samples.get(
            (transcript.task_id, transcript.task_repeat)
        )
        if found is None:
            raise ValueError(
                f"Transcript {transcript.transcript_id!r} is not a sample of {log}: no sample there has that uuid, or its id and epoch."
            )
        return found


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
