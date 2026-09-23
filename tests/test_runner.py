import pytest

from inspect_sentinel._report import Action, Decision, Observation, Reported
from inspect_sentinel._runner import Decisions, Observations, Reports


def _obs(name: str, suspicion: float | dict[str, float]) -> Reported[Observation]:
    return Reported(name=name, path=name, report=Observation.score(suspicion))


def _dec(name: str, action: Action) -> Reported[Decision]:
    return Reported(name=name, path=name, report=Decision(action=action))


def test_observations_is_a_sequence() -> None:
    observations = Observations([_obs("a", 0.1), _obs("b", 0.2)])
    assert len(observations) == 2
    assert observations[1].name == "b"
    assert [o.name for o in observations] == ["a", "b"]


def test_max_suspicion_over_scalars_and_dicts() -> None:
    assert Observations([]).max_suspicion() is None
    assert Observations([_obs("a", 0.3), _obs("b", 0.7)]).max_suspicion() == 0.7
    assert (
        Observations([_obs("a", {"x": 0.2, "y": 0.9}), _obs("b", 0.5)]).max_suspicion()
        == 0.9
    )


@pytest.mark.parametrize(
    ("actions", "expected"),
    [
        (["continue", "reject", "modify"], "reject"),
        (["modify", "terminate", "reject"], "terminate"),
        (["continue", "modify"], "modify"),
        (["continue"], "continue"),
        (["escalate", "escalate"], None),
        ([], None),
    ],
)
def test_strongest_by_precedence(actions: list[Action], expected: str | None) -> None:
    decisions = Decisions([_dec(f"d{i}", a) for i, a in enumerate(actions)])
    strongest = decisions.strongest()
    assert (strongest.report.action if strongest else None) == expected


def test_strongest_prefers_the_first_on_ties() -> None:
    decisions = Decisions([_dec("first", "reject"), _dec("second", "reject")])
    strongest = decisions.strongest()
    assert strongest is not None and strongest.name == "first"


def test_reports_holds_both_families() -> None:
    reports = Reports(
        Observations([_obs("a", 0.1)]), Decisions([_dec("d", "continue")])
    )
    assert reports.observations.max_suspicion() == 0.1
    assert reports.decisions.strongest() is not None
