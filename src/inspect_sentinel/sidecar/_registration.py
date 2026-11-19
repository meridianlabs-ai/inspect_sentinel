from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias

from inspect_ai.model import ChatMessage
from pydantic import BaseModel, Field, ValidationError

from .._config import parse_config_file
from .._context import EvalContext, InspectSample

TASK_EXPIRY = 24 * 60 * 60
"""How long a task Inspect registers is kept once no request has named it, unless the registration asks otherwise, in seconds."""

MAX_TASK_EXPIRY = 30 * 24 * 60 * 60
"""The longest a task Inspect registers is kept once no request has named it, unless the sidecar is told otherwise, in seconds."""

MAX_TASK_REGISTRATIONS = 1000
"""How many tasks registered by evals are kept at once, and how many evals' tokens, unless the sidecar is told otherwise. Anything that can reach the endpoint for evals can register a task, so a registration past this is refused, rather than one still in use dropped. It also bounds the evals whose tokens are kept, with runtime registration on or off: a token is kept until no request has named its eval for the run expiry, so an eval that has finished holds a place until then."""


class RegistrationError(Exception):
    """A task registration the sidecar won't take."""


class RegistrationsFullError(RegistrationError):
    """A task registration the sidecar can't take now, because it keeps as many as it takes, or a sample's registration a task's registration can't hold, because it holds as much as the sidecar takes for a task. A failure, not a refusal: the task may run there once a kept registration expires."""


Strictness: TypeAlias = Literal["open", "known_tasks", "registered_only"]
"""How strict a sidecar is about a request it can't place. Under every strictness, a request with an `x-inspect-context` header is refused, and not judged, unless it passes the check of that header, which needs the token its eval registered with. The strictness says what happens to a request that passes, and to one without the header. `open` judges each of those, without the task's facts where it holds none. `known_tasks` refuses a request that names a task it doesn't hold, and judges one that names no task. `registered_only` refuses any request that doesn't name a sample it holds, including one that names no task."""


class UnregisteredSampleError(Exception):
    """A request the sidecar refuses because it can't place it: a sample of a task registered beforehand that the registration doesn't hold, or, by the sidecar's strictness, a task or sample it doesn't hold."""


class UnknownTaskError(Exception):
    """A sample's registration for a task the eval hasn't registered, or whose registration is no longer kept."""


class SampleFacts(BaseModel):
    """One sample of a registered task."""

    sample_id: str | int = Field(description="The sample's id in the dataset.")
    sample_description: str | None = Field(
        default=None, description="What is distinctive about the sample."
    )
    sample_input: str | list[ChatMessage] = Field(
        description="The sample's input: a string, or chat messages as inspect_ai writes them."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any], description="The sample's metadata."
    )


class TaskInfo(BaseModel):
    """What a sentinel is told of a task, apart from its samples."""

    task: str = Field(description="The task's name.")
    task_description: str | None = Field(
        default=None, description="What agents in the task are asked to do."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict[str, Any],
        description="The task's metadata. A sentinel is given it merged with each sample's own.",
    )


class TaskFacts(TaskInfo):
    """What a sentinel is told of a task and its samples, as `context.eval`."""

    samples: list[SampleFacts] = Field(description="Every sample of the task.")

    def eval_context(self, sample: InspectSample) -> EvalContext | None:
        """The sample a request names, as a sentinel is given it, or None if the task holds no such sample."""
        found = next(
            (fact for fact in self.samples if fact.sample_id == sample.sample_id),
            None,
        )
        return _eval_context(self, found, sample) if found is not None else None


class TaskRegistration(TaskInfo):
    """What an eval registers of a task at task init: the task's facts, and which of its samples it runs. Each sample's own facts follow before the sample runs, in its `complete_registration`, so a large dataset is never sent at once."""

    sample_ids: list[str | int] = Field(
        description="The ids of the samples the eval runs, after `--limit`, `--sample-id` and the like."
    )
    epochs: int = Field(ge=1, description="How many epochs each sample runs.")


@dataclass
class RegisteredTask:
    """A task an eval registered, with the facts of each sample whose registration is complete and that has a run in progress.

    A sample's facts are freed when its last run in progress ends, so that a task with more samples than the registration may hold can run: a later epoch or retry of the sample completes its registration again. A run is known by the sample uuid its registration or its request carried.
    """

    registration: TaskRegistration
    samples: dict[str | int, SampleFacts] = field(
        default_factory=dict[str | int, SampleFacts]
    )
    sample_ids: set[str | int] = field(init=False)
    """The samples the eval runs: those the registration named, and any a sample's registration added."""

    held: int = field(init=False)
    """How many bytes the registration holds, as JSON: the task's registration, the facts of each sample that are held, the id of each sample a sample's registration added, each run in progress, and each sample in `freed`. A sample in `until_expiry` adds nothing to its facts, which are held with it."""

    runs: dict[str | int, set[str]] = field(default_factory=dict[str | int, set[str]])
    """The runs in progress of each sample whose facts are held, by sample uuid."""

    until_expiry: set[str | int] = field(default_factory=set[str | int])
    """The samples whose facts are kept until the registration expires: those a registration completed without naming its run, as an Inspect that sends no sample uuid does. Such a run may be yet to make its first request, and nothing says when it ends."""

    freed: set[str | int] = field(default_factory=set[str | int])
    """The samples whose facts were freed, until the sample's registration is complete again. A sample whose facts never arrived is in neither this nor `samples`."""

    _sample_of: dict[str, str | int] = field(default_factory=dict[str, str | int])

    def __post_init__(self) -> None:
        # a set, as a dataset can have many samples, and each is looked up
        self.sample_ids = set(self.registration.sample_ids)
        self.held = _bytes(self.registration)

    def keep_from(self, before: RegisteredTask) -> None:
        """Keep what the registration this one replaces held of the samples this one still names: their facts, their runs in progress, and which were freed or are kept until expiry."""
        named = self.sample_ids
        self.samples = {
            id: sample for id, sample in before.samples.items() if id in named
        }
        self.runs = {id: set(runs) for id, runs in before.runs.items() if id in named}
        self._sample_of = {run: id for id, runs in self.runs.items() for run in runs}
        self.until_expiry = before.until_expiry & named
        self.freed = before.freed & named
        self.held = (
            _bytes(self.registration)
            + sum(_bytes(sample) for sample in self.samples.values())
            + sum(_size(run) for run in self._sample_of)
            + sum(_size(id) for id in self.freed)
        )

    def holding(self, sample: SampleFacts, run: str | None = None) -> int:
        """How many bytes the registration would hold once a sample's registration is complete.

        Args:
            sample: The sample.
            run: The uuid of the run completing it, or None if the eval named none.
        """
        id = sample.sample_id
        held = self.held + _bytes(sample)
        before = self.samples.get(id)
        if before is not None:
            held -= _bytes(before)
        if id not in self.sample_ids:
            held += _size(id)
        if id in self.freed:
            held -= _size(id)
        counted = self._counted(id, run)
        if counted is not None and counted not in self._sample_of:
            held += _size(counted)
        return held

    def complete(self, sample: SampleFacts, run: str | None = None) -> None:
        """Keep a sample's facts, replacing any it had, for the run that completed its registration.

        Args:
            sample: The sample.
            run: The uuid of the run completing it. With none, or with one that is already a run of another sample, the facts are kept until the registration expires, whatever runs of the sample are named before or after.
        """
        id = sample.sample_id
        self.held = self.holding(sample, run)
        self.sample_ids.add(id)
        self.samples[id] = sample
        self.freed.discard(id)
        counted = self._counted(id, run)
        if counted is None:
            self.until_expiry.add(id)
        else:
            self.runs.setdefault(id, set()).add(counted)
            self._sample_of[counted] = id

    def begin(self, id: str | int, run: str, limit: int) -> None:
        """Count a run that made a request as in progress, where the sample's facts are held and may be freed. A run that completed its own registration already is.

        A run the registration has no room for isn't counted. It keeps the facts it finds at this request, so freeing them doesn't reach it.

        Args:
            id: The sample's id.
            run: The run's sample uuid.
            limit: How many bytes the registration may hold.
        """
        if id not in self.samples or id in self.until_expiry or run in self._sample_of:
            return
        if self.held + _size(run) > limit:
            return
        self.runs.setdefault(id, set()).add(run)
        self._sample_of[run] = id
        self.held += _size(run)

    def end(self, run: str) -> bool:
        """End a run, and free its sample's facts if it was the last in progress.

        Args:
            run: The run's sample uuid.

        Returns:
            Whether the registration held the run.
        """
        id = self._sample_of.pop(run, None)
        if id is None:
            return False
        self.held -= _size(run)
        left = self.runs[id]
        left.discard(run)
        if left:
            return True
        del self.runs[id]
        if id in self.until_expiry:
            # a completion since named no run, which may be yet to call
            return True
        facts = self.samples.pop(id, None)
        if facts is not None:
            self.freed.add(id)
            self.held += _size(id) - _bytes(facts)
        return True

    def eval_context(self, sample: InspectSample) -> EvalContext | None:
        """The sample a request names, as a sentinel is given it, or None if its facts aren't held: its registration isn't complete, or they were freed."""
        found = self.samples.get(sample.sample_id)
        if found is None:
            return None
        return _eval_context(self.registration, found, sample)

    def _counted(self, id: str | int, run: str | None) -> str | None:
        # a completion's run, if it names one that isn't already a run of
        # another sample
        if run is None:
            return None
        return run if self._sample_of.get(run, id) == id else None


def _size(value: str | int) -> int:
    # what a registration holds of an id, as JSON, in the bytes the facts
    # that held it were counted in: a sample's freed mark is then never more
    # than its facts were
    return len(json.dumps(value, ensure_ascii=False).encode())


def _bytes(model: BaseModel) -> int:
    # what a registration holds of a model, as the JSON it arrived in
    return len(model.model_dump_json().encode())


def _eval_context(
    task: TaskInfo, found: SampleFacts, sample: InspectSample
) -> EvalContext:
    # the epoch is the request's, as each epoch is a run of its own
    return EvalContext(
        task=task.task,
        task_description=task.task_description,
        sample_id=found.sample_id,
        epoch=sample.epoch,
        sample_description=found.sample_description,
        sample_input=found.sample_input,
        metadata={**task.metadata, **found.metadata},
    )


class FileTask(TaskFacts):
    """A task registered beforehand, in a tasks file in the tasks folder."""

    tools: dict[str, str] = Field(
        default_factory=dict[str, str],
        description="For a tool of the task's own that has the name of one of inspect_ai's, the tool's registry name, by the name the model sees.",
    )


class TasksFile(BaseModel):
    """A tasks file: tasks registered with the server beforehand. The tasks folder holds any number of them."""

    tasks: list[FileTask] = Field(description="The tasks, each named once.")


def read_tasks_folder(folder: str | Path) -> dict[str, FileTask]:
    """Read every tasks file in a folder: each YAML or JSON file directly in it.

    Args:
        folder: The folder.

    Returns:
        The tasks of every file, by name.

    Raises:
        ValueError: If the folder isn't one, a file in it can't be read as a tasks file, or a task is named twice, in one file or in two.
    """
    path = Path(folder)
    if not path.is_dir():
        raise ValueError(f"{folder}: not a folder of tasks files.")
    tasks: dict[str, FileTask] = {}
    found: dict[str, Path] = {}
    for file in sorted(path.iterdir()):
        if not file.is_file() or file.suffix not in _TASKS_FILES:
            continue
        for name, task in read_tasks_file(file).items():
            if name in tasks:
                raise ValueError(
                    f"{file}: the task {name!r} is named in {found[name].name} too. A request names only a task's name, so two tasks can't share one."
                )
            tasks[name] = task
            found[name] = file
    return tasks


_TASKS_FILES = frozenset({".yaml", ".yml", ".json"})


def read_tasks_file(path: str | Path) -> dict[str, FileTask]:
    """Read a tasks file, YAML or JSON.

    Args:
        path: The file.

    Returns:
        The tasks, by name.

    Raises:
        ValueError: If the file can't be read as a tasks file, or names a task twice.
    """
    # read as a sentinel file is, so that a key given twice is an error
    parsed = parse_config_file(str(path))
    try:
        given = TasksFile.model_validate(parsed)
    except ValidationError as ex:
        raise ValueError(f"{path}: not a tasks file: {ex}") from None
    tasks: dict[str, FileTask] = {}
    for task in given.tasks:
        if task.task in tasks:
            raise ValueError(
                f"{path}: the task {task.task!r} is named twice. A request names only a task's name, so two tasks can't share one."
            )
        tasks[task.task] = task
    return tasks
