"""Every task spec, by id. Add a module of specs and list it here."""

from __future__ import annotations

import tasks_c
import tasks_d
import tasks_e
import tasks_f
from spec import TaskSpec

_ALL: dict[str, TaskSpec] = {
    t.id: t for module in (tasks_c, tasks_d, tasks_e, tasks_f) for t in module.TASKS
}


def task_spec(id: str) -> TaskSpec:
    return _ALL[id]


def all_specs() -> dict[str, TaskSpec]:
    return dict(_ALL)
