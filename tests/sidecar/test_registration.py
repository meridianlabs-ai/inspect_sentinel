import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import protocol
from inspect_sentinel._report import Decision, Escalations
from inspect_sentinel._step import BeforeGenerate
from inspect_sentinel._types import Protocol
from inspect_sentinel.sidecar import Handler, Refuse, Request, _state
from inspect_sentinel.sidecar._registration import (
    FileTask,
    RegistrationError,
    RegistrationsFullError,
    SampleFacts,
    TaskFacts,
    TaskRegistration,
    UnknownTaskError,
    UnregisteredSampleError,
    read_tasks_file,
    read_tasks_folder,
)
from inspect_sentinel.sidecar._state import MemoryState
from tests.sidecar._traffic import (
    SAMPLE,
    holding,
    inspect_headers,
    message,
    reply,
    request,
    weather,
    weather_rule,
)

_TASK = TaskFacts(
    task="philadelphia",
    task_description="File the weather report.",
    metadata={"desk": "weather", "shared": "task"},
    samples=[
        SampleFacts(
            sample_id=1,
            sample_description="A rainy day.",
            sample_input="File today's report for Philadelphia.",
            metadata={"shared": "sample", "city": "Philadelphia"},
        )
    ],
)


@protocol
def reports_the_eval() -> Protocol:
    async def decide(
        context: Context, step: BeforeGenerate, escalations: Escalations
    ) -> Decision | None:
        given = context.eval
        if given is None:
            return Decision.proceed("None")
        return Decision.proceed(
            json.dumps(
                {
                    "task": given.task,
                    "task_description": given.task_description,
                    "sample_id": given.sample_id,
                    "epoch": given.epoch,
                    "sample_description": given.sample_description,
                    "sample_input": given.sample_input_text,
                    "metadata": given.metadata,
                }
            )
        )

    return decide


def _registration(task: TaskFacts, epochs: int = 2) -> TaskRegistration:
    # what an eval registers of the task at task init: its samples' ids only
    return TaskRegistration(
        task=task.task,
        task_description=task.task_description,
        metadata=task.metadata,
        sample_ids=[sample.sample_id for sample in task.samples],
        epochs=epochs,
    )


async def _register(
    handler: Handler, task: TaskFacts = _TASK, expiry: float | None = None
) -> float | None:
    # an eval registers the task, then completes each sample's registration
    granted = await handler.register_task("eval_789", _registration(task), expiry)
    for sample in task.samples:
        await handler.complete_registration("eval_789", task.task, sample)
    return granted


def _asked(id: str, **sample: Any) -> Request:
    # a request from the Inspect sample the header names
    return replace(request(id), headers=inspect_headers(**sample))


async def _eval_seen(handler: Handler, id: str) -> Any:
    records = await handler.records(id)
    assert records is not None
    explanation = cast(Any, records[-1].detail).report.explanation
    return None if explanation == "None" else json.loads(explanation)


@pytest.mark.anyio
async def test_a_run_of_a_registered_task_is_told_its_sample() -> None:
    handler = await holding(Handler(reports_the_eval()))
    await _register(handler)

    await handler.request(_asked("a"))

    assert await _eval_seen(handler, "a") == {
        "task": "philadelphia",
        "task_description": "File the weather report.",
        "sample_id": 1,
        # the epoch is the request's, as each epoch is a run of its own
        "epoch": SAMPLE["epoch"],
        "sample_description": "A rainy day.",
        "sample_input": "File today's report for Philadelphia.",
        # the sample's own metadata wins, as in Inspect's own process
        "metadata": {"desk": "weather", "shared": "sample", "city": "Philadelphia"},
    }


@pytest.mark.anyio
async def test_a_sample_input_of_chat_messages_is_kept_as_messages() -> None:
    handler = await holding(Handler(reports_the_eval()))
    sample = SampleFacts.model_validate(
        {
            "sample_id": 1,
            "sample_input": [
                {"role": "system", "content": "You report the weather."},
                {"role": "user", "content": "File today's report."},
            ],
        }
    )
    await _register(handler, _TASK.model_copy(update={"samples": [sample]}))

    await handler.request(_asked("a"))

    seen = await _eval_seen(handler, "a")
    assert seen["sample_input"] == "You report the weather.\nFile today's report."


@pytest.mark.anyio
@pytest.mark.parametrize(
    "sample",
    [
        {"task": "another_task"},
        {"eval_id": "another_eval"},
        {"sample_id": 2},
    ],
    ids=["another task", "another eval", "a sample the task doesn't hold"],
)
async def test_a_run_of_a_task_not_registered_is_told_no_sample(
    sample: dict[str, Any],
) -> None:
    handler = await holding(Handler(reports_the_eval()))
    # another eval that registered, though not this task
    await holding(handler, "another_eval")
    await _register(handler)

    await handler.request(_asked("a", **sample))

    assert await _eval_seen(handler, "a") is None


@pytest.mark.anyio
async def test_a_run_keeps_the_sample_it_first_found() -> None:
    handler = await holding(Handler(reports_the_eval()))
    await _register(handler)
    await handler.request(_asked("a"))

    # the task is registered again, changed, while the run goes on
    await _register(handler, _TASK.model_copy(update={"task_description": "Changed."}))
    await handler.request(_asked("b"))
    # a run that starts now is told the new registration
    await handler.request(_asked("c", sample_uuid="sample_124"))

    assert (await _eval_seen(handler, "b"))["task_description"] == (
        "File the weather report."
    )
    assert (await _eval_seen(handler, "c"))["task_description"] == "Changed."


@pytest.mark.anyio
async def test_a_registration_that_arrives_late_reaches_the_run() -> None:
    handler = await holding(Handler(reports_the_eval()))
    await handler.request(_asked("a"))

    await _register(handler)
    await handler.request(_asked("b"))

    assert await _eval_seen(handler, "a") is None
    assert (await _eval_seen(handler, "b"))["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_client_without_the_header_is_told_no_sample() -> None:
    handler = Handler(reports_the_eval())
    await _register(handler)

    await handler.request(request("a"))

    assert await _eval_seen(handler, "a") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("asked", "granted"),
    [
        (None, 24 * 60 * 60),
        (7 * 24 * 60 * 60, 7 * 24 * 60 * 60),
        (365 * 24 * 60 * 60, 30 * 24 * 60 * 60),
    ],
    ids=["a day unless asked", "as asked", "no longer than the handler allows"],
)
async def test_a_registration_is_kept_as_long_as_asked_within_a_limit(
    asked: float | None, granted: float
) -> None:
    assert await _register(Handler(reports_the_eval()), expiry=asked) == granted


@pytest.mark.anyio
async def test_the_limit_on_keeping_a_registration_can_be_set() -> None:
    handler = Handler(reports_the_eval(), task_expiry=60)

    assert await _register(handler, expiry=3600) == 60


class _Clock:
    now = 0.0

    @classmethod
    def monotonic(cls) -> float:
        return cls.now


@pytest.mark.anyio
async def test_a_registration_is_kept_while_requests_name_its_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the state's own clock only: the event loop keeps the real one
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = await holding(
        Handler(reports_the_eval(), state=MemoryState(expiry=10**6))
    )
    await _register(handler, expiry=100)

    # each request names the task before the registration expires
    for id, now in [("a", 90.0), ("b", 180.0), ("c", 270.0)]:
        _Clock.now = now
        await handler.request(_asked(id, sample_uuid=f"run_{id}"))
    # and then nothing names it for longer than it is kept
    _Clock.now = 371.0
    await handler.request(_asked("d", sample_uuid="run_d"))

    assert (await _eval_seen(handler, "c"))["task"] == "philadelphia"
    assert await _eval_seen(handler, "d") is None


_FILE_TASK = FileTask.model_validate(
    {**_TASK.model_dump(), "tools": {"bash": "weather_tools/bash"}}
)


@pytest.mark.anyio
async def test_a_task_registered_beforehand_is_found_for_any_eval() -> None:
    handler = await holding(
        Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})
    )

    await holding(handler, "any_eval")
    await handler.request(_asked("a", eval_id="any_eval"))

    assert (await _eval_seen(handler, "a"))["task"] == "philadelphia"


@pytest.mark.anyio
async def test_a_task_registered_beforehand_cant_be_replaced() -> None:
    handler = await holding(
        Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})
    )

    kept = await _register(
        handler, _TASK.model_copy(update={"task_description": "Changed."})
    )
    await handler.request(_asked("a"))

    assert kept is None

    assert (await _eval_seen(handler, "a"))["task_description"] == (
        "File the weather report."
    )


@pytest.mark.anyio
async def test_a_sample_a_task_registered_beforehand_doesnt_hold_is_refused() -> None:
    handler = await holding(
        Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})
    )

    with pytest.raises(UnregisteredSampleError, match="sample 2"):
        await handler.request(_asked("a", sample_id=2))


@pytest.mark.anyio
async def test_a_sidecar_can_take_no_registrations_from_evals() -> None:
    handler = Handler(reports_the_eval(), runtime_task_registration=False)

    with pytest.raises(RegistrationError, match="takes no tasks"):
        await _register(handler)


@pytest.mark.anyio
async def test_a_sidecar_that_takes_no_registrations_still_answers_a_task_registered_beforehand() -> (
    None
):
    handler = Handler(
        reports_the_eval(),
        tasks={"philadelphia": _FILE_TASK},
        runtime_task_registration=False,
    )

    assert await _register(handler) is None


@pytest.mark.anyio
async def test_a_sample_is_told_its_facts_once_its_registration_is_complete() -> None:
    handler = await holding(Handler(reports_the_eval()))
    await handler.register_task("eval_789", _registration(_TASK))
    await handler.request(_asked("a"))

    await handler.complete_registration("eval_789", "philadelphia", _TASK.samples[0])
    await handler.request(_asked("b", sample_uuid="sample_124"))

    assert await _eval_seen(handler, "a") is None
    assert (await _eval_seen(handler, "b"))["sample_input"] == (
        "File today's report for Philadelphia."
    )


@pytest.mark.anyio
async def test_registering_a_task_again_keeps_the_samples_it_still_names() -> None:
    two = SampleFacts(sample_id=2, sample_input="File tomorrow's report.")
    handler = await holding(Handler(reports_the_eval()))
    await _register(
        handler, _TASK.model_copy(update={"samples": [*_TASK.samples, two]})
    )

    # an eval that retries its registration, now without sample 2
    await handler.register_task("eval_789", _registration(_TASK))
    await handler.request(_asked("a"))
    await handler.request(_asked("b", sample_id=2, sample_uuid="sample_124"))

    assert (await _eval_seen(handler, "a"))["sample_id"] == 1
    assert await _eval_seen(handler, "b") is None


@pytest.mark.anyio
async def test_registering_a_large_task_again_takes_time_in_proportion() -> None:
    # 100,000 samples took 14 seconds when each was looked up in a list
    count = 100_000
    registration = _registration(_TASK).model_copy(
        update={"sample_ids": list(range(count))}
    )
    handler = Handler(reports_the_eval())
    await handler.register_task("eval_789", registration)
    kept = await handler._state.registration("eval_789", "philadelphia")
    assert kept is not None
    kept.samples = {
        id: SampleFacts.model_construct(sample_id=id, sample_input="Hi.", metadata={})
        for id in range(count)
    }

    started = time.perf_counter()
    await handler.register_task("eval_789", registration)

    assert time.perf_counter() - started < 1.0


@pytest.mark.anyio
async def test_a_sample_the_registration_doesnt_name_extends_it() -> None:
    added = SampleFacts(sample_id=2, sample_input="File tomorrow's report.")
    handler = await holding(Handler(reports_the_eval()))
    await _register(handler)

    # as for a sample a task adds while it runs
    await handler.complete_registration("eval_789", "philadelphia", added)
    await handler.request(_asked("a", sample_id=2))

    assert (await _eval_seen(handler, "a"))["sample_input"] == (
        "File tomorrow's report."
    )


def _sample(sample_id: int, size: int = 1_000) -> SampleFacts:
    return SampleFacts(sample_id=sample_id, sample_input="x" * size)


async def _held(handler: Handler) -> int:
    kept = await handler._state.registration("eval_789", "philadelphia")
    assert kept is not None
    return kept.held


@pytest.mark.anyio
async def test_a_registration_holds_no_more_than_the_sidecar_takes_for_a_task() -> None:
    # anything that can reach the endpoint can register a task, so what one
    # registration holds is bounded, its samples' facts included
    handler = Handler(reports_the_eval(), max_registration_bytes=5_000)
    await handler.register_task(
        "eval_789",
        _registration(_TASK).model_copy(update={"sample_ids": [1, 2, 3, 4, 5]}),
    )
    for sample_id in [1, 2, 3]:
        await handler.complete_registration(
            "eval_789", "philadelphia", _sample(sample_id)
        )
    held = await _held(handler)

    with pytest.raises(RegistrationsFullError, match="no more than 5000"):
        for sample_id, size in [(4, 1_000), (5, 2_000)]:
            await handler.complete_registration(
                "eval_789", "philadelphia", _sample(sample_id, size)
            )

    # what fails isn't kept, and what was kept still is
    kept = await handler._state.registration("eval_789", "philadelphia")
    assert kept is not None
    assert sorted(kept.samples) == [1, 2, 3, 4]
    assert held < kept.held <= 5_000


@pytest.mark.anyio
async def test_a_first_registration_holds_no_more_than_the_sidecar_takes() -> None:
    handler = Handler(reports_the_eval(), max_registration_bytes=300)
    large = _registration(_TASK).model_copy(update={"sample_ids": list(range(100))})

    with pytest.raises(RegistrationsFullError, match="no more than 300"):
        await handler.register_task("eval_789", large)

    assert await handler._state.registration("eval_789", "philadelphia") is None


@pytest.mark.anyio
async def test_completing_a_samples_registration_again_holds_it_once() -> None:
    # an eval completes a sample's registration for each epoch and retry
    handler = Handler(reports_the_eval(), max_registration_bytes=5_000)
    await handler.register_task("eval_789", _registration(_TASK))
    await handler.complete_registration("eval_789", "philadelphia", _sample(1))
    once = await _held(handler)

    for _ in range(10):
        await handler.complete_registration("eval_789", "philadelphia", _sample(1))

    assert await _held(handler) == once


@pytest.mark.anyio
async def test_the_samples_a_registration_adds_count_toward_what_it_holds() -> None:
    # an open sidecar takes samples the registration doesn't name; their ids
    # are held too, so new ids can't grow a registration without bound
    handler = Handler(reports_the_eval(), max_registration_bytes=2_000)
    await handler.register_task("eval_789", _registration(_TASK))

    with pytest.raises(RegistrationsFullError):
        for sample_id in range(10_000):
            await handler.complete_registration(
                "eval_789", "philadelphia", _sample(sample_id, size=0)
            )

    kept = await handler._state.registration("eval_789", "philadelphia")
    assert kept is not None
    assert len(kept.sample_ids) < 100
    assert kept.held <= 2_000


@pytest.mark.anyio
async def test_registering_a_task_again_holds_only_the_samples_it_keeps() -> None:
    handler = Handler(reports_the_eval())
    registration = _registration(_TASK).model_copy(update={"sample_ids": [1, 2]})
    await handler.register_task("eval_789", registration)
    await handler.register_task("eval_789", _registration(_TASK))
    fresh = await _held(handler)
    await handler.register_task("eval_789", registration)
    for sample_id in [1, 2]:
        await handler.complete_registration(
            "eval_789", "philadelphia", _sample(sample_id)
        )

    # registering again without sample 2 holds sample 1's facts only
    await handler.register_task("eval_789", _registration(_TASK))

    assert await _held(handler) == fresh + len(_sample(1).model_dump_json())


@pytest.mark.anyio
async def test_a_samples_registration_needs_its_task_registered() -> None:
    handler = Handler(reports_the_eval())

    with pytest.raises(UnknownTaskError, match="no registration of the task"):
        await handler.complete_registration(
            "eval_789", "philadelphia", _TASK.samples[0]
        )


@pytest.mark.anyio
async def test_a_samples_registration_for_a_task_registered_beforehand_is_not_kept() -> (
    None
):
    handler = await holding(
        Handler(reports_the_eval(), tasks={"philadelphia": _FILE_TASK})
    )
    changed = _TASK.samples[0].model_copy(update={"sample_input": "Changed."})

    assert await handler.complete_registration("eval_789", "philadelphia", changed)
    await handler.request(_asked("a"))

    assert (await _eval_seen(handler, "a"))["sample_input"] == (
        "File today's report for Philadelphia."
    )


@pytest.mark.anyio
async def test_an_open_sidecar_judges_a_request_it_cant_place() -> None:
    handler = await holding(Handler(reports_the_eval()))

    await handler.request(_asked("a"))
    await handler.request(request("b"))

    assert await _eval_seen(handler, "a") is None
    assert await _eval_seen(handler, "b") is None


@pytest.mark.anyio
async def test_a_sidecar_that_judges_only_known_tasks_refuses_another() -> None:
    handler = await holding(Handler(reports_the_eval(), strictness="known_tasks"))
    await handler.register_task("eval_789", _registration(_TASK))

    with pytest.raises(UnregisteredSampleError, match="holds no task 'another'"):
        await handler.request(_asked("a", task="another"))
    # a known task whose sample isn't complete yet, and a request no eval names
    await handler.request(_asked("b"))
    await handler.request(request("c"))

    assert await _eval_seen(handler, "b") is None
    assert await _eval_seen(handler, "c") is None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("sample", "says"),
    [
        (None, "has no x-inspect-context header"),
        ({"task": "another"}, "holds no task 'another'"),
        ({"sample_id": 2}, "isn't complete for the sample 2"),
        ({"epoch": 3}, "runs epochs 1 to 2, not epoch 3"),
        ({"epoch": 0}, "runs epochs 1 to 2, not epoch 0"),
        ({"epoch": -1}, "runs epochs 1 to 2, not epoch -1"),
    ],
    ids=[
        "no header",
        "another task",
        "a sample not completed",
        "an epoch past the last",
        "epoch 0",
        "a negative epoch",
    ],
)
async def test_a_sidecar_that_judges_only_registered_samples_refuses_another(
    sample: dict[str, Any] | None, says: str
) -> None:
    handler = await holding(Handler(reports_the_eval(), strictness="registered_only"))
    await _register(handler)
    # signed as the test runs, within the window
    asked = request("a") if sample is None else _asked("a", **sample)

    with pytest.raises(UnregisteredSampleError, match=says):
        await handler.request(asked)


@pytest.mark.anyio
async def test_a_sidecar_that_judges_only_registered_samples_judges_one() -> None:
    handler = await holding(Handler(reports_the_eval(), strictness="registered_only"))
    await _register(handler)

    await handler.request(_asked("a"))

    assert (await _eval_seen(handler, "a"))["sample_id"] == 1


@pytest.mark.anyio
async def test_a_sidecar_that_judges_only_registered_samples_takes_no_other() -> None:
    added = SampleFacts(sample_id=2, sample_input="File tomorrow's report.")
    handler = Handler(reports_the_eval(), strictness="registered_only")
    await _register(handler)

    with pytest.raises(RegistrationError, match="doesn't name the sample 2"):
        await handler.complete_registration("eval_789", "philadelphia", added)


_YAML = """
tasks:
  - task: philadelphia
    task_description: File the weather report.
    metadata:
      desk: weather
    samples:
      - sample_id: 1
        sample_input: File today's report for Philadelphia.
      - sample_id: two
        sample_input:
          - role: system
            content: You report the weather.
          - role: user
            content: File today's report.
    tools:
      bash: weather_tools/bash
"""


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_tasks_file_is_read_as_yaml_or_json(tmp_path: Path, suffix: str) -> None:
    import yaml

    path = tmp_path / f"tasks{suffix}"
    parsed = yaml.safe_load(_YAML)
    path.write_text(_YAML if suffix == ".yaml" else json.dumps(parsed))

    tasks = read_tasks_file(path)

    task = tasks["philadelphia"]
    assert task.tools == {"bash": "weather_tools/bash"}
    assert [sample.sample_id for sample in task.samples] == [1, "two"]
    messages = task.samples[1].sample_input
    assert not isinstance(messages, str)
    assert [message.role for message in messages] == ["system", "user"]


@pytest.mark.parametrize(
    ("text", "says"),
    [
        ("tasks: [{task: a, samples: []}, {task: a, samples: []}]", "named twice"),
        ("tasks: [{task: a}]", "not a tasks file"),
        ("[oops", "could not parse"),
        ("tasks: [{task: a, samples: [{sample_id: 1}]}]", "not a tasks file"),
        (
            "tasks: [{task: a, samples: [{sample_id: 1, sample_input: Hi, "
            "metadata: {desk: news, desk: weather}}]}]",
            "duplicate key 'desk'",
        ),
    ],
    ids=[
        "a task twice",
        "no samples",
        "not YAML",
        "a sample without input",
        "a key twice",
    ],
)
def test_a_tasks_file_that_cant_be_read(tmp_path: Path, text: str, says: str) -> None:
    path = tmp_path / "tasks.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=says):
        read_tasks_file(path)


def test_a_tasks_folder_holds_every_tasks_file_in_it(tmp_path: Path) -> None:
    (tmp_path / "weather.yaml").write_text(_YAML)
    (tmp_path / "travel.json").write_text(
        json.dumps({"tasks": [{"task": "itinerary", "samples": []}]})
    )
    (tmp_path / "README.md").write_text("Not a tasks file.")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "ignored.yaml").write_text("not: a tasks file")

    tasks = read_tasks_folder(tmp_path)

    assert sorted(tasks) == ["itinerary", "philadelphia"]


def test_a_task_named_in_two_files_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text(_YAML)
    (tmp_path / "b.yml").write_text(_YAML)

    with pytest.raises(ValueError, match="named in a.yaml too"):
        read_tasks_folder(tmp_path)


def test_a_tasks_folder_that_isnt_one_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a folder"):
        read_tasks_folder(tmp_path / "missing")


@pytest.mark.anyio
async def test_a_request_stripped_of_its_header_doesnt_reach_a_terminated_run() -> None:
    # where an eval can't be trusted, a request without the header is refused
    # unjudged, so a client can't send a terminated run's conversation as a
    # fresh one by dropping the header
    handler = await holding(
        Handler(weather_rule("terminate", "Tokyo"), strictness="registered_only")
    )
    await _register(handler)
    await handler.request(_asked("a"))
    await handler.reply(reply(message(weather("Tokyo")), id="a"))
    stripped = replace(request("b"), headers={"x-irid": "irid-b"})

    with pytest.raises(UnregisteredSampleError):
        await handler.request(stripped)

    # nothing was kept or judged, and the run stays ended
    assert await handler._state.attempts("irid-b", "eval_789") is None  # pyright: ignore[reportPrivateUsage]
    assert await handler.records("b") is None
    ended = await handler.sample_result("sample_123", "eval_789")
    assert ended is not None and ended.outcome == "terminate"


# A sample's facts are held while a run of the sample is in progress.


async def _two_samples(strictness: str = "open") -> Handler:
    # a handler that holds the registration of a task with samples 1 and 2
    handler = await holding(
        Handler(
            reports_the_eval(),
            strictness=cast(Any, strictness),
            state=MemoryState(expiry=10**6),
        )
    )
    await handler.register_task("eval_789", _two())
    return handler


def _two() -> TaskRegistration:
    return _registration(_TASK).model_copy(update={"sample_ids": [1, 2]})


async def _complete(handler: Handler, run: str | None, sample_id: int = 1) -> None:
    await handler.complete_registration(
        "eval_789", "philadelphia", _sample(sample_id), run
    )


async def _end(handler: Handler, run: str, task: str | None = "philadelphia") -> str:
    return await handler.end_sample(run, "eval_789", task)


async def _facts_held(handler: Handler) -> list[str | int]:
    kept = await handler._state.registration("eval_789", "philadelphia", renew=False)
    assert kept is not None
    return sorted(kept.samples, key=str)


@pytest.mark.anyio
async def test_a_task_with_more_facts_than_a_registration_holds_runs_as_its_samples_end() -> (
    None
):
    # room for two samples' facts at once, and no more
    handler = await holding(Handler(reports_the_eval(), max_registration_bytes=3_000))
    ids = list(range(1, 11))
    await handler.register_task(
        "eval_789", _registration(_TASK).model_copy(update={"sample_ids": ids})
    )

    for sample_id in ids:
        await _complete(handler, f"run_{sample_id}", sample_id)
        assert await _end(handler, f"run_{sample_id}") == "ended"

    assert await _facts_held(handler) == []
    assert await _held(handler) <= 3_000


@pytest.mark.anyio
async def test_a_samples_facts_are_held_until_its_last_run_ends() -> None:
    handler = await _two_samples()
    fresh = await _held(handler)
    # two epochs at once: one has made a request, and the other has only
    # completed its registration
    await _complete(handler, "run_a")
    await _complete(handler, "run_b")
    await handler.request(_asked("a", sample_uuid="run_a"))

    await _end(handler, "run_a")
    # ending it again changes nothing
    assert await _end(handler, "run_a") == "ended"

    assert await _facts_held(handler) == [1]
    await handler.request(_asked("b", sample_uuid="run_b"))
    assert (await _eval_seen(handler, "b"))["sample_id"] == 1

    await _end(handler, "run_b")

    assert await _facts_held(handler) == []
    # what is left is the mark that the sample's facts were freed
    assert await _held(handler) == fresh + len(json.dumps(1))


@pytest.mark.anyio
async def test_a_run_counts_once_however_often_it_completes_its_registration() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_a")
    held = await _held(handler)

    # as each retry of the sample does
    await _complete(handler, "run_a")

    assert await _held(handler) == held
    await _end(handler, "run_a")
    assert await _facts_held(handler) == []


@pytest.mark.anyio
async def test_a_sample_registered_again_after_its_facts_were_freed_is_told_them() -> (
    None
):
    handler = await _two_samples(strictness="registered_only")
    await _complete(handler, "run_a")
    await _end(handler, "run_a")

    # the next epoch
    await _complete(handler, "run_b")
    await handler.request(_asked("b", sample_uuid="run_b"))

    assert (await _eval_seen(handler, "b"))["sample_id"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("unnamed_first", [True, False])
async def test_a_registration_that_names_no_run_keeps_the_facts_until_it_expires(
    unnamed_first: bool,
) -> None:
    handler = await _two_samples()
    # an Inspect that sends no sample uuid: its run may be yet to call
    for run in [None, "run_b"] if unnamed_first else ["run_b", None]:
        await _complete(handler, run)
    await handler.request(_asked("b", sample_uuid="run_b"))

    assert await _end(handler, "run_b") == "ended"

    assert await _facts_held(handler) == [1]


@pytest.mark.anyio
@pytest.mark.parametrize("unnamed_first", [True, False])
async def test_a_run_named_beside_one_that_wasnt_is_still_ended(
    unnamed_first: bool,
) -> None:
    handler = await _two_samples()
    for run in [None, "run_b"] if unnamed_first else ["run_b", None]:
        await _complete(handler, run)

    # the named run ends before its first request: it is remembered as
    # ended, though the facts stay for the run that wasn't named
    assert await _end(handler, "run_b") == "ended"
    late = await handler.request(_asked("b", sample_uuid="run_b"))

    assert isinstance(late, Refuse) and "has ended" in late.message
    assert await _facts_held(handler) == [1]


@pytest.mark.anyio
async def test_a_registration_completed_after_its_sample_ended_doesnt_count() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_a")
    await _complete(handler, "run_b")
    await _end(handler, "run_a")
    held = await _held(handler)

    # a completion of the ended run that arrives late
    await _complete(handler, "run_a")

    assert await _held(handler) == held
    await _end(handler, "run_b")
    assert await _facts_held(handler) == []


@pytest.mark.anyio
async def test_a_run_the_registration_has_no_room_for_isnt_counted() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_b")
    held = await _held(handler)
    cast(Any, handler)._max_registration_bytes = held

    # another run of the sample makes a request, as one whose task was
    # registered again after it completed its registration
    await handler.request(_asked("a", sample_uuid="run_a"))

    assert await _held(handler) == held
    # it goes on with the facts it found, once the sample's are freed
    await _end(handler, "run_b")
    await handler.request(_asked("a2", sample_uuid="run_a"))
    assert (await _eval_seen(handler, "a2"))["sample_id"] == 1


@pytest.mark.anyio
async def test_freeing_a_samples_facts_never_holds_more() -> None:
    handler = await holding(Handler(reports_the_eval()))
    # an id that takes more bytes escaped than as it is written
    sample_id = "\N{SNOWMAN}" * 40
    registration = _registration(_TASK).model_copy(update={"sample_ids": []})
    await handler.register_task("eval_789", registration)
    await handler.complete_registration(
        "eval_789",
        "philadelphia",
        SampleFacts(sample_id=sample_id, sample_input=""),
        "run_a",
    )
    held = await _held(handler)

    await _end(handler, "run_a")

    assert await _facts_held(handler) == []
    assert await _held(handler) < held


@pytest.mark.anyio
async def test_a_sample_ended_without_its_task_named_keeps_its_facts() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_a")
    await _complete(handler, "run_b")
    await handler.request(_asked("b", sample_uuid="run_b"))

    # as an Inspect that names no task ends a sample: a run that made a
    # request is ended, and one that didn't is unknown, as before
    assert await _end(handler, "run_b", task=None) == "ended"
    assert await _end(handler, "run_a", task=None) == "unknown"
    # and a task the eval didn't register finds nothing to free
    assert await _end(handler, "run_a", task="another") == "unknown"

    assert await _facts_held(handler) == [1]


@pytest.mark.anyio
async def test_a_sample_that_ends_before_its_first_request_is_ended() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_a")
    await _complete(handler, "run_b")

    assert await _end(handler, "run_a") == "ended"
    # its first request arrives late: it is refused as an ended sample's is,
    # and doesn't count as a run in progress again
    late = await handler.request(_asked("a", sample_uuid="run_a"))
    await _end(handler, "run_b")

    assert isinstance(late, Refuse) and "has ended" in late.message
    assert await _facts_held(handler) == []


@pytest.mark.anyio
async def test_a_run_that_makes_a_request_keeps_the_facts_another_run_registered() -> (
    None
):
    handler = await _two_samples(strictness="registered_only")
    # run_a never completed the registration here, as when its task was
    # registered again after it had
    await _complete(handler, "run_b")
    await handler.request(_asked("a", sample_uuid="run_a"))

    await _end(handler, "run_b")
    await handler.request(_asked("a2", sample_uuid="run_a"))

    assert await _facts_held(handler) == [1]
    assert (await _eval_seen(handler, "a2"))["sample_id"] == 1


@pytest.mark.anyio
@pytest.mark.parametrize("strictness", ["open", "known_tasks", "registered_only"])
async def test_a_run_with_no_facts_of_its_own_is_refused_once_its_samples_were_freed(
    strictness: str,
) -> None:
    handler = await _two_samples(strictness)
    await _complete(handler, "run_b")
    await _end(handler, "run_b")

    # never judged without the facts the sample's runs had
    with pytest.raises(UnregisteredSampleError, match="freed the facts"):
        await handler.request(_asked("a", sample_uuid="run_a"))

    # a sample whose facts never arrived is as it was
    if strictness != "registered_only":
        await handler.request(_asked("c", sample_uuid="run_c", sample_id=2))
        assert await _eval_seen(handler, "c") is None


@pytest.mark.anyio
@pytest.mark.parametrize("strictness", ["open", "known_tasks", "registered_only"])
async def test_a_run_keeps_the_facts_it_found_when_its_samples_are_freed(
    monkeypatch: pytest.MonkeyPatch, strictness: str
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = await _two_samples(strictness)
    await _complete(handler, "run_a")
    await handler.request(_asked("a", sample_uuid="run_a"))
    # the task's registration expires, and another run registers it again
    _Clock.now = 10**5
    await handler.register_task("eval_789", _two())
    await _complete(handler, "run_b")
    await _end(handler, "run_b")
    assert await _facts_held(handler) == []

    # judged with its own, where the sidecar takes only registered samples too
    await handler.request(_asked("a2", sample_uuid="run_a"))
    assert (await _eval_seen(handler, "a2"))["sample_id"] == 1
    # and still held to the epochs the registration runs
    if strictness == "registered_only":
        with pytest.raises(UnregisteredSampleError, match="epochs 1 to 2"):
            await handler.request(_asked("a3", sample_uuid="run_a", epoch=3))


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("strictness", "judged"),
    [("open", True), ("known_tasks", True), ("registered_only", False)],
)
async def test_a_task_that_expired_and_was_registered_again_is_as_it_was(
    monkeypatch: pytest.MonkeyPatch, strictness: str, judged: bool
) -> None:
    monkeypatch.setattr(_state, "time", _Clock)
    _Clock.now = 0.0
    handler = await _two_samples(strictness)
    # sample 1 completes its registration, and waits to make its first request
    await _complete(handler, "run_a")
    # the task expires meanwhile, and sample 2 registers it again
    _Clock.now = 10**5
    await handler.register_task("eval_789", _two())
    await _complete(handler, "run_b", sample_id=2)

    # nothing was freed, so sample 1 is a sample whose facts never arrived
    if judged:
        await handler.request(_asked("a", sample_uuid="run_a"))
        assert await _eval_seen(handler, "a") is None
    else:
        with pytest.raises(UnregisteredSampleError, match="isn't complete"):
            await handler.request(_asked("a", sample_uuid="run_a"))


@pytest.mark.anyio
async def test_registering_a_task_again_keeps_what_was_freed_and_what_is_kept() -> None:
    handler = await _two_samples()
    # sample 1's facts are freed, and sample 2's are kept until expiry
    await _complete(handler, "run_a")
    await _end(handler, "run_a")
    await _complete(handler, None, sample_id=2)
    await _complete(handler, "run_c", sample_id=2)
    held = await _held(handler)

    await handler.register_task("eval_789", _two())

    assert await _held(handler) == held
    with pytest.raises(UnregisteredSampleError, match="freed the facts"):
        await handler.request(_asked("b", sample_uuid="run_b"))
    await _end(handler, "run_c")
    assert await _facts_held(handler) == [2]


@pytest.mark.anyio
async def test_registering_a_task_again_keeps_the_runs_in_progress() -> None:
    handler = await _two_samples()
    await _complete(handler, "run_a")
    await _complete(handler, "run_b")

    await handler.register_task("eval_789", _two())
    await _end(handler, "run_a")

    assert await _facts_held(handler) == [1]
    await _end(handler, "run_b")
    assert await _facts_held(handler) == []
