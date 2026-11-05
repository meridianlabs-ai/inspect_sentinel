# The adapter copies two of Scout's private names, and inspect_ai's log
# driver's version and layout checks, rather than import them; these pin each
# copy to its original, so a change there fails here.
import pytest
from inspect_ai._sentinel._replay import (  # pyright: ignore[reportPrivateUsage]
    _before_phase_spans as driver_before_phase_spans,  # pyright: ignore[reportPrivateUsage]
)
from inspect_ai._sentinel._replay import (  # pyright: ignore[reportPrivateUsage]
    _has_earlier_layout as driver_has_earlier_layout,  # pyright: ignore[reportPrivateUsage]
)
from inspect_ai.event import (
    Event,
    InfoEvent,
    ModelEvent,
    SpanBeginEvent,
    TimelineEvent,
    TimelineSpan,
    ToolEvent,
)
from inspect_ai.model import ChatMessageUser, GenerateConfig, ModelOutput
from inspect_scout._scanner.scanner import (  # pyright: ignore[reportPrivateUsage]
    SCANNER_TIMELINE_NEEDED_ATTR,
)
from inspect_scout._transcript.timeline import (  # pyright: ignore[reportPrivateUsage]
    _walk_spans as scout_walk_spans,  # pyright: ignore[reportPrivateUsage]
)

from inspect_sentinel.scout._scanner import TIMELINE_NEEDED_ATTR
from inspect_sentinel.scout._steps import (
    _before_phase_spans,  # pyright: ignore[reportPrivateUsage]
    _has_earlier_layout,  # pyright: ignore[reportPrivateUsage]
    _walk_spans,  # pyright: ignore[reportPrivateUsage]
)


def model() -> TimelineEvent:
    return TimelineEvent(
        event=ModelEvent(
            model="mockllm/model",
            input=[ChatMessageUser(content="hi")],
            tools=[],
            tool_choice="auto",
            config=GenerateConfig(),
            output=ModelOutput.from_content("mockllm/model", "out"),
        )
    )


def span(
    id: str, *children: TimelineSpan, calls: bool = False, utility: bool = False
) -> TimelineSpan:
    return TimelineSpan(
        id=id,
        name=id,
        content=[*([model()] if calls else []), *children],
        utility=utility,
    )


TREES = {
    # the synthetic root timeline_build makes: a container of agents
    "agents": span(
        "root",
        span("helper", span("inside_helper", calls=True), calls=True, utility=True),
        span(
            "container",
            span(
                "outer",
                span("middle", span("inner", calls=True), calls=True),
                calls=True,
            ),
        ),
        span(
            "worker",
            span("tooling", span("deep", calls=True), calls=True, utility=True),
            calls=True,
        ),
        span("empty"),
    ),
    # a root that makes model calls of its own
    "scannable root": span(
        "root", span("child", span("grandchild", calls=True), calls=True), calls=True
    ),
    # a utility root, walked through as a container is
    "utility root": span("root", span("child", calls=True), calls=True, utility=True),
}


@pytest.mark.parametrize("depth", [None, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("tree", TREES)
def test_the_span_walk_is_scouts(tree: str, depth: int | None) -> None:
    root = TREES[tree]
    ours = [s.id for s, _ in _walk_spans(root, depth)]
    assert ours == [s.id for s in scout_walk_spans(root, depth=depth)]


@pytest.mark.parametrize("tree", TREES)
def test_a_spans_level_is_the_least_depth_scout_reaches_it_at(tree: str) -> None:
    root = TREES[tree]
    reached = {
        s.id: next(d for d in range(1, 5) if s in scout_walk_spans(root, depth=d))
        for s in scout_walk_spans(root, depth=None)
    }
    assert {s.id: level for s, level in _walk_spans(root, None)} == reached


def test_the_spans_walked_skip_utility_and_container_spans() -> None:
    # what the walk is pinned to: only spans that make model calls count
    assert [s.id for s, _ in _walk_spans(TREES["agents"], 1)] == [
        "inside_helper",
        "outer",
        "worker",
    ]


def test_the_timeline_attribute_is_scouts() -> None:
    assert TIMELINE_NEEDED_ATTR == SCANNER_TIMELINE_NEEDED_ATTR


@pytest.mark.parametrize(
    "version",
    [
        None,
        "",
        "unknown",
        "0.3.43",
        "0.3.99",
        "0.3.99.post1",
        "0.3.100",
        "0.3.100.0",
        "v0.3.100",
        "0.3.100rc1",
        "0.3.100.dev3+g0123abc",
        "0.3.100.post1.dev2",
        "0.3.100-1",
        "0.3.101.dev1",
        "0.3.288.dev255+gb0a3f9368",
        "0.4",
        "1!0.3.43",
        "0!0.3.200",
        "0.3.99xyz",
    ],
)
def test_the_version_check_is_the_log_drivers(version: str | None) -> None:
    assert _before_phase_spans(version) == driver_before_phase_spans(version)


def begin(id: str, type: str | None = None, parent: str | None = None) -> Event:
    return SpanBeginEvent(id=id, name=id, type=type, parent_id=parent)


def tool(*inside: Event) -> Event:
    return ToolEvent(id="call", function="add", arguments={}, events=list(inside))


# The driver also counts a solver or scorer `StepEvent` as the earlier layout.
# A scan never loads step events (`EVENTS` in `scout/_scanner.py` has no
# "step"), so the adapter's copy leaves that check out and no case here has one.
LAYOUTS: dict[str, tuple[list[Event], bool]] = {
    "current solver and scorer spans": (
        [
            begin("init"),
            begin("solvers"),
            begin("generate", "solver", "solvers"),
            begin("scorers"),
            begin("match", "scorer", "scorers"),
        ],
        False,
    ),
    # Task.cleanup, or a hook, awaits a chain outside the `solvers` span
    "a cleanup chain beside the solvers span": (
        [
            begin("solvers"),
            begin("generate", "solver", "solvers"),
            begin("cleanup", "solver"),
        ],
        False,
    ),
    "a flat tool event": ([begin("solvers"), tool()], False),
    "no spans": ([InfoEvent(data="hi")], False),
    "a top-level scorer span": (
        [
            begin("solvers"),
            begin("generate", "solver", "solvers"),
            begin("match", "scorer"),
        ],
        True,
    ),
    "an unphased solver sample": (
        [begin("init"), begin("generate", "solver")],
        True,
    ),
    # rescoring adds a `scorers` span to a sample that has no `solvers` span
    "a rescored unphased solver sample": (
        [
            begin("init"),
            begin("generate", "solver"),
            begin("scorers"),
            begin("match", "scorer", "scorers"),
        ],
        True,
    ),
    # only a top-level `solvers` span is the solvers phase
    "an unphased solver sample with a nested solvers span": (
        [begin("generate", "solver"), begin("solvers", parent="generate")],
        True,
    ),
    "a tool event that nests events": (
        [begin("solvers"), tool(InfoEvent(data="inside the tool"))],
        True,
    ),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_the_layout_check_is_the_log_drivers(layout: str) -> None:
    events, earlier = LAYOUTS[layout]
    assert _has_earlier_layout(events) == driver_has_earlier_layout(events) == earlier
