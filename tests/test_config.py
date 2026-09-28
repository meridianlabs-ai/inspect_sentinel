import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
import yaml
from inspect_ai._util import registry
from inspect_ai._util.registry import registry_info, registry_params

from inspect_sentinel._context import Context
from inspect_sentinel._integration import (
    SentinelConfig,
    Sentinels,
    config_from_sentinel,
    resolve_sentinel,
    run_root,
    sentinel_from_config,
)
from inspect_sentinel._monitor import (
    ControlProtocol,
    Monitor,
    Monitors,
    monitor,
    protocol,
)
from inspect_sentinel._protocols import concurrent, threshold
from inspect_sentinel._report import Decision, Observation
from inspect_sentinel._step import AfterToolCall, BeforeToolCall, Step
from tests._fakes import ListRecorder, after_step, before_step, runner_context


@monitor
def cfg_suspicion(model: str | None = None) -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.9)

    return check


@monitor
def cfg_pair() -> list[Monitor]:
    async def before(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    async def after(context: Context, step: AfterToolCall) -> Observation | None:
        return Observation.score(0.2)

    return [before, after]


@protocol
def cfg_bundle(**groups: Monitors) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


@protocol
def cfg_fields(name: Monitors, params: Monitors | None = None) -> ControlProtocol:
    async def decide(context: Context, step: Step) -> Decision | None:
        return None

    return decide


@protocol
def cfg_rule(reason: str = "no") -> ControlProtocol:
    async def decide(context: Context, step: BeforeToolCall) -> Decision | None:
        return Decision.reject(reason)

    return decide


def test_a_list_builds_each_entry_through_the_registry() -> None:
    built = sentinel_from_config(
        [
            {"name": "cfg_suspicion", "params": {"model": "openai/gpt-4o-mini"}},
            {"name": "cfg_rule"},
        ]
    )
    assert isinstance(built, list)
    assert [registry_info(child).name for child in built] == [
        "cfg_suspicion",
        "cfg_rule",
    ]
    assert registry_params(built[0]) == {"model": "openai/gpt-4o-mini"}


def test_a_mapping_keeps_its_instance_names() -> None:
    built = sentinel_from_config({"first": {"name": "cfg_rule"}})
    assert isinstance(built, dict)
    assert list(built) == ["first"]
    assert registry_info(built["first"]).type == "protocol"


def test_bare_names_find_the_shipped_protocols() -> None:
    [built] = sentinel_from_config(
        [
            {
                "name": "threshold",
                "params": {"reject_at": 0.8},
                "monitors": [{"name": "cfg_suspicion"}],
            }
        ]
    )
    assert registry_info(built).name == "inspect_sentinel/threshold"
    assert registry_params(built)["reject_at"] == 0.8


def test_a_group_instance_is_built_like_any_other() -> None:
    [built] = sentinel_from_config([{"name": "cfg_pair"}])
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
    await run_root(
        resolve_sentinel(built), runner_context(recorder=recorder), before_step()
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
    [built] = sentinel_from_config(config)
    assert registry_info(built).name == "inspect_sentinel/threshold"


@pytest.mark.parametrize("suffix", [".yaml", ".json"])
def test_a_file_holds_the_config_under_sentinel(tmp_path: Path, suffix: str) -> None:
    content = {"sentinel": {"escape": {"name": "cfg_rule"}}}
    file = tmp_path / f"sentinel{suffix}"
    file.write_text(json.dumps(content) if suffix == ".json" else yaml.dump(content))
    built = sentinel_from_config(str(file))
    assert isinstance(built, dict)
    assert registry_info(built["escape"]).name == "cfg_rule"


def test_a_bare_registered_name_is_a_list_of_one() -> None:
    [built] = sentinel_from_config("cfg_suspicion")
    assert registry_info(built).name == "cfg_suspicion"


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


@pytest.mark.parametrize(
    "content", [{"approvers": []}, {"sentinel": [], "extra": 1}, [{"name": "cfg_rule"}]]
)
def test_a_file_must_hold_only_a_sentinel_key(tmp_path: Path, content: Any) -> None:
    file = tmp_path / "sentinel.yaml"
    file.write_text(yaml.dump(content))
    with pytest.raises(ValueError, match="sentinel"):
        sentinel_from_config(str(file))


ROUND_TRIPS: list[Any] = [
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
    {
        "attempt": {
            "name": "concurrent",
            "children": {
                "watch": {"name": "cfg_suspicion"},
                "inner": {
                    "name": "concurrent",
                    "children": [
                        {"name": "cfg_rule"},
                        {"name": "observe", "monitors": {"m": {"name": "cfg_pair"}}},
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
    [
        {
            "name": "cfg_fields",
            "params": {
                "name": [{"type": "monitor", "name": "cfg_suspicion", "params": {}}],
                "params": {"p": {"type": "monitor", "name": "cfg_pair", "params": {}}},
            },
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
    context = runner_context(recorder=recorder)
    await run_root(resolve_sentinel(sentinels), context, before_step())
    await run_root(resolve_sentinel(sentinels), context, after_step())
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
        "pair": cfg_pair(),
        "bundle": cfg_bundle(fast=[cfg_suspicion()]),
        "fields": cfg_fields([cfg_suspicion()], params={"p": cfg_pair()}),
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


def test_a_lone_instance_is_recorded_as_a_list_of_one() -> None:
    assert config_from_sentinel(cfg_rule()) == SentinelConfig.model_validate(
        [{"name": "cfg_rule"}]
    )


def test_a_non_sentinel_is_not_recorded() -> None:
    with pytest.raises(TypeError, match="monitor or protocol"):
        config_from_sentinel(cast(Any, [cfg_rule]))


@pytest.mark.parametrize(
    "text, where, key",
    [
        (
            "sentinel:\n  attempt:\n    name: cfg_rule\n  attempt:\n    name: cfg_rule\n",
            "sentinel",
            "attempt",
        ),
        (
            "sentinel:\n  - name: cfg_rule\n    params:\n      reason: a\n      reason: b\n",
            r"sentinel\[0\]\.params",
            "reason",
        ),
        ("sentinel: []\nsentinel: []\n", "the top level", "sentinel"),
        (
            '{"sentinel": {"a": {"name": "cfg_rule", "name": "cfg_suspicion"}}}',
            r"sentinel\.a",
            "name",
        ),
    ],
)
def test_a_repeated_key_in_a_file_is_an_error(
    tmp_path: Path, text: str, where: str, key: str
) -> None:
    file = tmp_path / "sentinel.yaml"
    file.write_text(text)
    with pytest.raises(
        ValueError, match=rf"sentinel\.yaml: {where}: duplicate key '{key}'"
    ):
        sentinel_from_config(str(file))


def test_tab_indented_json_is_read(tmp_path: Path) -> None:
    file = tmp_path / "sentinel.json"
    file.write_text('{\n\t"sentinel": [\n\t\t{"name": "cfg_rule"}\n\t]\n}\n')
    [built] = sentinel_from_config(str(file))
    assert registry_info(built).name == "cfg_rule"


def test_a_malformed_file_names_the_file(tmp_path: Path) -> None:
    file = tmp_path / "broken.yaml"
    file.write_text("sentinel: [\n")
    with pytest.raises(ValueError, match=r"broken\.yaml: could not parse"):
        sentinel_from_config(str(file))


@pytest.fixture
def scratch_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "_registry", dict(registry._registry))


@pytest.mark.usefixtures("scratch_registry")
def test_an_exact_name_wins_over_the_package_fallback() -> None:
    @protocol
    def observe(monitors: Monitors) -> ControlProtocol:
        async def decide(context: Context, step: Step) -> Decision | None:
            return None

        return decide

    config: Any = [
        {"name": "observe", "monitors": [{"name": "cfg_suspicion"}]},
        {"name": "inspect_sentinel/observe", "monitors": [{"name": "cfg_suspicion"}]},
    ]
    built = sentinel_from_config(config)
    assert [registry_info(child).name for child in built] == [
        "observe",
        "inspect_sentinel/observe",
    ]
    assert config_from_sentinel(built) == SentinelConfig.model_validate(config)
