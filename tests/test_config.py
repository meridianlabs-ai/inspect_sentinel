from collections.abc import Mapping
from typing import Any, cast

import pytest
from inspect_ai.core import SentinelConfig
from inspect_ai.core import _registry as registry
from inspect_ai.core._registry import registry_info, registry_params
from pydantic import ValidationError

from inspect_sentinel._context import Context
from inspect_sentinel._decorators import (
    monitor,
    protocol,
)
from inspect_sentinel._integration import (
    config_from_sentinel,
    resolve_sentinel,
    run_sentinel,
    sentinel_from_config,
)
from inspect_sentinel._protocols import concurrent, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from inspect_sentinel._types import (
    Monitor,
    MonitorGroup,
    Monitors,
    Protocol,
    Sentinel,
    Sentinels,
)
from tests._fakes import ListRecorder, after_step, before_step, host_context


def _listed(built: Sentinels) -> list[Sentinel]:
    assert isinstance(built, list)
    return cast(list[Sentinel], built)


def _mapped(built: Sentinels) -> dict[str, Sentinel]:
    assert isinstance(built, dict)
    return cast(dict[str, Sentinel], built)


def _only(built: Sentinels) -> Sentinel:
    [child] = _listed(built)
    return child


@monitor
def cfg_suspicion(model: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.9)

    return check


@monitor
def cfg_pair() -> MonitorGroup:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.2)

    return MonitorGroup(before, after)


@protocol
def cfg_bundle(**groups: Monitors) -> Protocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


@protocol
def cfg_rule(reason: str = "no") -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject(reason)

    return decide


def test_a_list_builds_each_entry_through_the_registry() -> None:
    built = _listed(
        sentinel_from_config(
            [
                {"name": "cfg_suspicion", "params": {"model": "openai/gpt-4o-mini"}},
                {"name": "cfg_rule"},
            ]
        )
    )
    assert [registry_info(child).name for child in built] == [
        "cfg_suspicion",
        "cfg_rule",
    ]
    assert registry_params(built[0]) == {"model": "openai/gpt-4o-mini"}


def test_a_mapping_keeps_its_instance_names() -> None:
    built = _mapped(sentinel_from_config({"first": {"name": "cfg_rule"}}))
    assert list(built) == ["first"]
    assert registry_info(built["first"]).type == "protocol"


def test_bare_names_find_the_shipped_protocols() -> None:
    built = _only(
        sentinel_from_config(
            [
                {
                    "name": "threshold",
                    "params": {"reject_at": 0.8},
                    "monitors": [{"name": "cfg_suspicion"}],
                }
            ]
        )
    )
    assert registry_info(built).name == "inspect_sentinel/threshold"
    assert registry_params(built)["reject_at"] == 0.8


def test_a_group_instance_is_built_like_any_other() -> None:
    built = _only(sentinel_from_config([{"name": "cfg_pair"}]))
    assert registry_info(built).name == "cfg_pair"


@pytest.mark.anyio
async def test_nested_entries_become_the_parameter_and_compose_paths() -> None:
    built = sentinel_from_config(
        {
            "attempt": {
                "name": "concurrent",
                "children": {
                    "watch": {"name": "cfg_suspicion"},
                    "block": {"name": "cfg_rule", "params": {"reason": "stop"}},
                },
            },
            "escape": {"name": "cfg_rule"},
        }
    )
    recorder = ListRecorder()
    await run_sentinel(
        resolve_sentinel(built), host_context(recorder=recorder), before_step()
    )
    assert sorted(r.reported.path for r in recorder.records) == [
        "",
        "attempt",
        "attempt/block",
        "attempt/watch",
        "escape",
    ]


def test_a_parsed_config_model_is_accepted() -> None:
    config = SentinelConfig.model_validate(
        [
            {
                "name": "threshold",
                "params": {"reject_at": 0.5},
                "monitors": [{"name": "cfg_suspicion"}],
            }
        ]
    )
    built = _only(sentinel_from_config(config))
    assert registry_info(built).name == "inspect_sentinel/threshold"


def test_a_bare_registered_name_is_a_lone_instance() -> None:
    built = sentinel_from_config("cfg_suspicion")
    assert registry_info(built).name == "cfg_suspicion"


@pytest.mark.parametrize(
    "raw",
    [
        {"name": "cfg_rule", "params": {"reason": "stop"}},
        SentinelConfig.model_validate(
            {"name": "cfg_rule", "params": {"reason": "stop"}}
        ),
    ],
)
def test_an_entry_with_a_string_name_is_a_lone_instance(raw: Any) -> None:
    built = sentinel_from_config(raw)
    assert registry_info(built).name == "cfg_rule"
    assert registry_params(built) == {"reason": "stop"}


@pytest.mark.parametrize(
    "raw",
    [
        {"name": {"name": "cfg_rule"}},
        {"name": {"name": "cfg_rule"}, "b": {"name": "cfg_rule"}},
    ],
)
def test_an_instance_named_name_is_still_a_mapping(raw: Any) -> None:
    assert isinstance(SentinelConfig.model_validate(raw).root, dict)
    built = _mapped(sentinel_from_config(raw))
    assert set(built) == set(raw)


@pytest.mark.parametrize(
    "raw, where",
    [
        ({"a": "cfg_rule"}, ()),
        ({"name": 3}, ()),
        ([{"name": "concurrent", "children": {"a": 1}}], ("list", 0, "children")),
    ],
)
def test_a_layer_that_is_neither_an_entry_nor_a_mapping_of_entries_is_an_error(
    raw: Any, where: tuple[str | int, ...]
) -> None:
    with pytest.raises(ValidationError, match="string 'name'") as info:
        SentinelConfig.model_validate(raw)
    assert info.value.errors()[0]["loc"] == where


INVALID: list[tuple[Any, type[Exception], str]] = [
    (
        [{"name": "cfg_nope"}],
        ValueError,
        r"sentinel\[0\]: 'cfg_nope' is not a registered monitor or protocol",
    ),
    (
        [{"name": "cfg_rule", "monitors": [{"name": "cfg_suspicion"}]}],
        ValueError,
        r"sentinel\[0\]: 'monitors' is not a parameter of cfg_rule",
    ),
    (
        {"m": {"name": "cfg_suspicion", "monitors": [{"name": "cfg_suspicion"}]}},
        ValueError,
        r"sentinel\.m: 'monitors' is not a parameter of cfg_suspicion",
    ),
    (
        [{"name": "cfg_rule", "tools": ["bash"]}],
        ValueError,
        r"sentinel\[0\]: 'tools' is not a parameter of cfg_rule",
    ),
    (
        {
            "attempt": {
                "name": "concurrent",
                "children": [{"name": "cfg_rule"}, {"name": "cfg_nope"}],
            }
        },
        ValueError,
        r"sentinel\.attempt\.children\[1\]: 'cfg_nope' is not a registered",
    ),
    ([{"params": {}}], ValueError, r"sentinel\[0\]: an entry needs a 'name'"),
    ([{"name": 3}], ValueError, r"sentinel\[0\]: an entry needs a 'name'"),
    (
        [{"name": "cfg_rule", "params": ["no"]}],
        ValueError,
        r"sentinel\[0\]\.params must be a mapping",
    ),
    (
        [
            {
                "name": "threshold",
                "params": {"reject_at": 0.5, "monitors": []},
                "monitors": [{"name": "cfg_suspicion"}],
            }
        ],
        ValueError,
        r"sentinel\[0\]: 'monitors' is given both nested and in params",
    ),
    (
        [
            {
                "name": "threshold",
                "params": {"reject_at": 0.5},
                "monitors": "cfg_suspicion",
            }
        ],
        ValueError,
        r"sentinel\[0\]\.monitors must be a list or a mapping of entries",
    ),
    (["cfg_rule"], ValueError, r"sentinel\[0\] must be a mapping with a 'name'"),
    (3, ValueError, r"sentinel must be a list or a mapping of entries"),
    (
        [{"name": "threshold", "monitors": [{"name": "cfg_suspicion"}]}],
        TypeError,
        r"sentinel\[0\]: .*reject_at",
    ),
    (
        [
            {
                "name": "threshold",
                "params": {"reject_at": 0.8, "terminate_at": 0.5},
                "monitors": [{"name": "cfg_suspicion"}],
            }
        ],
        ValueError,
        r"sentinel\[0\]: threshold's terminate_at must be above reject_at",
    ),
    (
        {
            "a": {
                "name": "concurrent",
                "children": [{"name": "cfg_rule"}, {"name": "cfg_rule"}],
            }
        },
        ValueError,
        r"sentinel\.a: Duplicate instance name 'cfg_rule'",
    ),
    (
        "cfg_no_such_thing",
        ValueError,
        r"'cfg_no_such_thing' is neither a config file nor a registered",
    ),
]


@pytest.mark.parametrize("config, error, message", INVALID)
def test_invalid_config_names_the_entry(
    config: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        sentinel_from_config(config)


ROUND_TRIPS: list[Any] = [
    {"name": "cfg_rule", "params": {"reason": "stop"}},
    {
        "name": "threshold",
        "params": {"reject_at": 0.5},
        "monitors": [{"name": "cfg_suspicion", "params": {}}],
    },
    [{"name": "cfg_suspicion", "params": {"model": "openai/gpt-4o-mini"}}],
    [{"name": "cfg_pair"}, {"name": "cfg_rule", "params": {"reason": "stop"}}],
    [
        {
            "name": "threshold",
            "params": {"reject_at": 0.8, "terminate_at": 0.95},
            "monitors": {
                "a": {"name": "cfg_suspicion"},
                "b": {"name": "cfg_suspicion"},
            },
        }
    ],
    [
        {
            "name": "threshold",
            "params": {
                "reject_at": {"a": 0.8, "b.sabotage": 0.9, "*": 0.95},
                "terminate_at": 0.99,
            },
            "monitors": {
                "a": {"name": "cfg_suspicion"},
                "b": {"name": "cfg_suspicion"},
            },
        }
    ],
    {
        "attempt": {
            "name": "concurrent",
            "children": {
                "watch": {"name": "cfg_suspicion"},
                "inner": {
                    "name": "concurrent",
                    "children": [
                        {"name": "cfg_rule"},
                        {
                            "name": "observe_only",
                            "monitors": {"m": {"name": "cfg_pair"}},
                        },
                    ],
                },
            },
        },
        "escape": {"name": "cfg_rule"},
    },
    [
        {
            "name": "cfg_bundle",
            "fast": [{"name": "cfg_suspicion"}],
            "slow": {"s": {"name": "cfg_suspicion", "params": {"model": "m"}}},
        }
    ],
]


@pytest.mark.parametrize("raw", ROUND_TRIPS)
def test_config_to_sentinel_to_config_is_equal(raw: Any) -> None:
    config = SentinelConfig.model_validate(raw)
    recorded = config_from_sentinel(sentinel_from_config(raw))
    assert recorded == config
    assert SentinelConfig.model_validate_json(recorded.model_dump_json()) == config


async def _paths(sentinels: Sentinels) -> list[tuple[str, str]]:
    recorder = ListRecorder()
    context = host_context(recorder=recorder)
    await run_sentinel(resolve_sentinel(sentinels), context, before_step())
    await run_sentinel(resolve_sentinel(sentinels), context, after_step())
    return sorted((r.reported.path, r.reported.function) for r in recorder.records)


def _identity(sentinels: object) -> Any:
    if isinstance(sentinels, Mapping):
        mapping = cast(Mapping[str, object], sentinels)
        return {k: _identity(v) for k, v in mapping.items()}
    if isinstance(sentinels, list):
        return [_identity(v) for v in cast(list[object], sentinels)]
    return (registry_info(sentinels).name, registry_params(sentinels))


@pytest.mark.anyio
async def test_sentinel_to_config_to_sentinel_is_equivalent() -> None:
    original: Sentinels = {
        "attempt": concurrent(
            {"watch": cfg_suspicion("openai/gpt-4o-mini"), "block": cfg_rule("stop")}
        ),
        "gate": threshold(
            {"a": cfg_suspicion(), "b": cfg_suspicion("m")}, reject_at=0.8
        ),
        "dims": threshold(
            {"a": cfg_suspicion()},
            reject_at={"a": 0.5, "sabotage": 0.7},
            terminate_at={"*": 0.95},
        ),
        "pair": cfg_pair(),
        "bundle": cfg_bundle(fast=[cfg_suspicion()]),
    }
    rebuilt = sentinel_from_config(config_from_sentinel(original))
    assert _identity(rebuilt) == _identity(original)
    assert await _paths(rebuilt) == await _paths(original)


def test_package_names_are_recorded_bare_and_others_in_full() -> None:
    config = config_from_sentinel([threshold([cfg_suspicion()], reject_at=0.5)])
    assert config.model_dump() == [
        {
            "name": "threshold",
            "params": {"reject_at": 0.5},
            "monitors": [{"name": "cfg_suspicion", "params": {}}],
        }
    ]


def test_a_lone_instance_is_recorded_as_a_lone_entry() -> None:
    assert config_from_sentinel(cfg_rule("stop")) == SentinelConfig.model_validate(
        {"name": "cfg_rule", "params": {"reason": "stop"}}
    )


@pytest.mark.anyio
async def test_a_lone_protocol_records_the_same_paths_once_rebuilt() -> None:
    original = threshold(cfg_suspicion(), reject_at=0.5)
    rebuilt = sentinel_from_config(config_from_sentinel(original))
    assert (
        await _paths(rebuilt)
        == await _paths(original)
        == [
            ("", "decide"),
            ("cfg_suspicion", "check"),
        ]
    )


def test_a_non_sentinel_is_not_recorded() -> None:
    with pytest.raises(TypeError, match="monitor or protocol"):
        config_from_sentinel(cast(Any, [cfg_rule]))


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))


@pytest.mark.usefixtures("scratch_registry")
def test_an_exact_name_wins_over_the_package_fallback() -> None:
    @protocol
    def observe_only(monitors: Monitors) -> Protocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            return None

        return decide

    config: Any = [
        {"name": "observe_only", "monitors": [{"name": "cfg_suspicion"}]},
        {
            "name": "inspect_sentinel/observe_only",
            "monitors": [{"name": "cfg_suspicion"}],
        },
    ]
    built = _listed(sentinel_from_config(config))
    assert [registry_info(child).name for child in built] == [
        "observe_only",
        "inspect_sentinel/observe_only",
    ]
    assert config_from_sentinel(built) == SentinelConfig.model_validate(config)


@monitor(version=2)
def cfg_versioned() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.3)

    return check


@protocol(version=5)
def cfg_versioned_rule() -> Protocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject("no")

    return decide


def test_a_version_round_trips() -> None:
    config = config_from_sentinel([cfg_versioned()])
    assert config.model_dump() == [
        {"name": "cfg_versioned", "params": {}, "version": 2}
    ]
    assert config_from_sentinel(sentinel_from_config(config)) == config
    assert SentinelConfig.model_validate_json(config.model_dump_json()) == config


def test_version_0_is_omitted() -> None:
    config = config_from_sentinel(cfg_rule("stop"))
    assert config.model_dump() == {"name": "cfg_rule", "params": {"reason": "stop"}}
    assert "version" not in config.model_dump_json()


def test_an_entry_without_a_version_builds_without_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    built = _only(sentinel_from_config([{"name": "cfg_versioned"}]))
    assert registry_info(built).name == "cfg_versioned"
    assert not caplog.records


def test_a_version_mismatch_warns_once_and_still_builds(
    caplog: pytest.LogCaptureFixture,
) -> None:
    raw = [{"name": "cfg_versioned_rule", "version": 4}]
    built = _only(sentinel_from_config(raw))
    assert registry_info(built).name == "cfg_versioned_rule"
    sentinel_from_config(raw)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert warnings == [
        "cfg_versioned_rule was recorded at version 4 but version 5 is installed; building it anyway."
    ]


def test_a_matching_version_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    sentinel_from_config([{"name": "cfg_versioned", "version": 2}])
    assert not caplog.records


def test_a_version_must_be_an_integer() -> None:
    with pytest.raises(ValueError, match=r"sentinel\[0\]\.version must be an integer"):
        sentinel_from_config(cast(Any, [{"name": "cfg_versioned", "version": "2"}]))


def test_meta_is_not_written() -> None:
    config = config_from_sentinel([cfg_versioned(), cfg_bundle(fast=[cfg_suspicion()])])
    assert "meta" not in config.model_dump_json()


def test_meta_is_ignored_when_building() -> None:
    raw = [
        {
            "name": "cfg_bundle",
            "meta": {"added": ["later"]},
            "fast": [{"name": "cfg_versioned", "version": 2, "meta": {"x": 1}}],
        }
    ]
    expected = SentinelConfig.model_validate(
        [
            {
                "name": "cfg_bundle",
                "params": {},
                "fast": [{"name": "cfg_versioned", "params": {}, "version": 2}],
            }
        ]
    )
    assert config_from_sentinel(sentinel_from_config(raw)) == expected
    validated = SentinelConfig.model_validate(raw)
    assert config_from_sentinel(sentinel_from_config(validated)) == expected


def test_nested_entries_carry_their_own_versions() -> None:
    original = cfg_bundle(
        fast=[cfg_versioned()],
        slow={"plain": cfg_suspicion()},
    )
    gate = threshold({"v": cfg_versioned()}, reject_at=0.5)
    config = config_from_sentinel({"bundle": original, "gate": gate})
    assert config.model_dump() == {
        "bundle": {
            "name": "cfg_bundle",
            "params": {},
            "fast": [{"name": "cfg_versioned", "params": {}, "version": 2}],
            "slow": {"plain": {"name": "cfg_suspicion", "params": {}}},
        },
        "gate": {
            "name": "threshold",
            "params": {"reject_at": 0.5},
            "monitors": {"v": {"name": "cfg_versioned", "params": {}, "version": 2}},
        },
    }
    assert config_from_sentinel(sentinel_from_config(config)) == config
