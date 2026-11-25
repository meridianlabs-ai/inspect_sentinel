"""The labelled cases: each a short agent transcript ending in the tool call under test, with its label and category.

Cases are YAML files, each a list of cases. A case is its task, the earlier steps of the transcript (tool calls with their recorded results, and any later user messages), and the call under test. `load_cases()` reads every `*.yaml` file in a directory and checks the set: unique ids, known tools and categories, and a harm category only on a `block` case.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Label = Literal["allow", "block"]

CATEGORIES = (
    "exfiltration",
    "sandbox_escape",
    "oversight_tampering",
    "outside_attacks",
    "unmonitored_agents",
    "network",
    "persistence",
    "destroying_data",
    "credentials",
    "real_world_effects",
)
"""The harm categories. A `block` case has one; an `allow` case is `benign`, and may say which of these it resembles."""

BENIGN = "benign"

TAGS = (
    "plain",
    "split",
    "injection",
    "claimed_approval",
    "background",
    "agent",
    "encoded",
    "risky_looking",
    "authorized",
    "later_user_message",
    "approval_after_task",
    "imported",
)
"""What makes a case hard, for slicing the report: a harmful effect split across two calls (`split`), an instruction injected in a tool result (`injection`), agent text claiming approval (`claimed_approval`), a background process (`background`), starting another agent (`agent`), an encoded payload (`encoded`); a benign call that looks risky (`risky_looking`), one the task explicitly asks for (`authorized`), or one a later user message authorizes (`later_user_message`); `approval_after_task` marks the benign cases where a user message after the task approves the call, which `auto_mode()`'s eval prompt does not count as authorization, so the report shows them apart; `imported` marks a case taken from an existing source rather than written for the benchmark."""

TOOLS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "bash": (("command",), ()),
    "write_file": (("path", "content"), ()),
    "python": (("code",), ()),
    "text_editor": (
        ("command", "path"),
        ("file_text", "old_str", "new_str", "insert_line", "view_range"),
    ),
}
"""The tools a case may call, with the arguments each requires and those it may also take. `python` and `text_editor` take inspect_ai's tools' arguments, for imported cases."""

CALL_UNDER_TEST = "call-under-test"
"""The tool call id of each case's call under test, its step id in replay."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Call(_Strict):
    """A tool call the agent makes, with the text it writes beside the call."""

    tool: str
    """The tool: `bash`, `write_file`, `python` or `text_editor` (see `TOOLS`)."""

    args: dict[str, Any]
    """The call's arguments: `command` for `bash`; `path` and `content` for `write_file`; `code` for `python`; `command`, `path` and the rest of inspect_ai's `text_editor()` arguments for `text_editor`."""

    text: str = ""
    """The agent's own text in the message that makes the call."""

    @model_validator(mode="after")
    def _known_tool(self) -> Call:
        found = TOOLS.get(self.tool)
        if found is None:
            raise ValueError(f"Unknown tool {self.tool!r}; cases use {list(TOOLS)}.")
        required, optional = found
        if not set(required) <= set(self.args) <= set(required) | set(optional):
            raise ValueError(
                f"{self.tool} takes arguments {list(required)}"
                + (f" and optionally {list(optional)}" if optional else "")
                + f", got {list(self.args)}."
            )
        return self


class Step(_Strict):
    """An earlier step: a tool call and its recorded result, or a user message."""

    tool: str | None = None
    args: dict[str, Any] | None = None
    text: str = ""
    result: str | None = None
    """The tool's recorded output."""

    user: str | None = None
    """A user message sent after the earlier steps, in place of a tool call."""

    @model_validator(mode="after")
    def _one_kind(self) -> Step:
        if self.user is not None:
            if (
                self.tool is not None
                or self.args is not None
                or self.result is not None
            ):
                raise ValueError("A user step has only `user`.")
        else:
            if self.tool is None or self.args is None or self.result is None:
                raise ValueError(
                    "A tool step needs `tool`, `args` and `result` (or is a `user` step)."
                )
            Call(tool=self.tool, args=self.args, text=self.text)
        return self

    def as_call(self) -> Call:
        """The step's tool call; only for a tool step."""
        assert self.tool is not None and self.args is not None
        return Call(tool=self.tool, args=self.args, text=self.text)


class Case(_Strict):
    """A labelled case: a transcript ending in the call under test."""

    id: str = Field(pattern=r"^[a-z0-9_-]+$")
    """Unique across the set; the sample id of its log."""

    label: Label
    """`block` where the call should be stopped before it runs, `allow` where it should run."""

    category: str
    """The harm category of a `block` case, or `benign`."""

    resembles: str | None = None
    """For a benign case, the harm category it looks like, if any."""

    tags: tuple[str, ...] = ()
    """What makes the case hard (see `TAGS`)."""

    why: str
    """One sentence for a reviewer: why the label is right."""

    source: str | None = None
    """Attribution: the published material a case was inspired by, or the item it was imported from."""

    detail: str | None = None
    """An imported case's own category in its source, such as a dataset's side task."""

    origin: str = "written"
    """Where the case comes from, for counting by source: `written` for a case written for the benchmark, else the source it was taken from, such as `sentinel_tests` or an importer's name."""

    task: str
    """The user's task, the sample's input."""

    steps: tuple[Step, ...] = ()
    """The earlier steps, oldest first."""

    call: Call
    """The call under test."""

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if self.label == "block" and self.category not in CATEGORIES:
            raise ValueError(
                f"{self.id}: a block case needs a harm category from {list(CATEGORIES)}, got {self.category!r}."
            )
        if self.label == "allow" and self.category != BENIGN:
            raise ValueError(f"{self.id}: an allow case's category is {BENIGN!r}.")
        if self.resembles is not None and (
            self.label != "allow" or self.resembles not in CATEGORIES
        ):
            raise ValueError(
                f"{self.id}: `resembles` is for an allow case, naming a harm category."
            )
        unknown = set(self.tags) - set(TAGS)
        if unknown:
            raise ValueError(f"{self.id}: unknown tags {sorted(unknown)}.")
        return self

    @property
    def area(self) -> str:
        """The harm category the case is about: its own, or the one a benign case resembles (`benign` where it resembles none)."""
        return self.resembles or self.category


def load_cases(paths: Path | Iterable[Path]) -> list[Case]:
    """Read every case from `paths`: YAML files, or directories whose `*.yaml` files are read in name order.

    Raises:
        ValueError: If a case is malformed, or two cases share an id.
    """
    files: list[Path] = []
    for path in [paths] if isinstance(paths, Path) else paths:
        files.extend(sorted(path.glob("*.yaml")) if path.is_dir() else [path])
    cases: list[Case] = []
    for file in files:
        raw: Any = yaml.safe_load(file.read_text()) or []
        if not isinstance(raw, list):
            raise ValueError(f"{file} must hold a list of cases.")
        for item in cast(list[Any], raw):
            try:
                cases.append(Case.model_validate(item))
            except ValueError as ex:
                raise ValueError(f"{file}: {ex}") from ex
    _unique(cases)
    return cases


def write_cases(cases: Sequence[Case], file: Path) -> None:
    """Write `cases` as a YAML case file."""
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(
        yaml.safe_dump(
            [c.model_dump(mode="json", exclude_defaults=True) for c in cases],
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )
    )


def _unique(cases: Sequence[Case]) -> None:
    seen: set[str] = set()
    for case in cases:
        if case.id in seen:
            raise ValueError(f"Two cases have the id {case.id!r}.")
        seen.add(case.id)
