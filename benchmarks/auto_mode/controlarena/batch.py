"""Run one batch of ControlArena trajectories: one setting, one mode, one attacker model, a slice of the setting's dataset.

This file runs in ControlArena's own environment (`python benchmarks/auto_mode/controlarena/batch.py SPEC.json`), never in Sentinel's, so ControlArena's pins can't change Sentinel's environment; pyright leaves it out for that reason. `generate.py` writes the spec, runs this, and reads the log it names. It configures ControlArena's own settings and policies and nothing else: every attack comes from ControlArena's attack policy (BashArena's own where the setting ships one) run against its setting.

The spec is JSON: `setting` (a ControlArena setting name), `setting_args` (its fields), `mode` (`honest` or `attack`), `model` (the untrusted model), `config` (its generate config), `trusted_model`, `seed`, `start` and `end` (positions in the setting's dataset shuffled by `seed`; positions past its end wrap, as repeats with a suffixed id), `log_dir`, and the limits `max_samples`, `token_limit`, `max_steps` and `time_limit`. It prints one JSON line: the log's path.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

from control_arena import EvalMode
from control_arena.cli.single_eval import register_plugins
from control_arena.eval import ControlEvalConfig, get_control_task
from control_arena.policy import untrusted_policy_builder
from control_arena.serde import load_json
from control_arena.settings._setting import Setting
from inspect_ai import eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig


def main(spec_file: str) -> None:
    spec: dict[str, Any] = json.loads(Path(spec_file).read_text())
    register_plugins()
    setting = load_json(
        {"type": spec["setting"], **spec.get("setting_args", {})}, Setting
    )
    mode = EvalMode(spec["mode"])
    config = GenerateConfig.model_validate(spec.get("config") or {})
    protocol = _protocol(spec["setting"], spec["model"], config)
    dataset = list(setting.get_dataset(mode))
    random.Random(f"{spec['seed']}/{spec['setting']}/{spec['mode']}").shuffle(dataset)
    chosen = [_at(dataset, i) for i in range(spec["start"], spec["end"])]
    eval_config = ControlEvalConfig(
        untrusted_model=spec["model"],
        untrusted_config=config,
        trusted_model=spec["trusted_model"],
        max_samples=spec.get("max_samples"),
        max_tasks=1,
        token_limit=spec.get("token_limit", 1_000_000),
        max_steps=spec.get("max_steps"),
        time_limit=spec.get("time_limit"),
        log_dir=spec["log_dir"],
    )
    task = get_control_task(
        setting=setting,
        micro_protocol=protocol,
        mode=mode,
        eval_config=eval_config,
        dataset=chosen,
        task_name_prefix=f"amb-{spec['setting']}-{spec['mode']}-{spec['model'].split('/')[-1]}-{spec['start']}",
        metadata={
            "amb_batch": {
                k: spec[k] for k in ("setting", "mode", "model", "start", "end", "seed")
            }
        },
    )
    [log] = eval(task, **eval_config.eval_args(), display="log")
    print(json.dumps({"log": log.location, "status": log.status}))


def _protocol(setting: str, model: str, config: GenerateConfig) -> Any:
    if setting == "bash_arena":
        # BashArena ships its own policies, and asks for its side tasks to be
        # tested after every action, which ends the trajectory where one lands
        from control_arena.settings.bash_arena.policies.attack_policy import (
            attack_policy,
        )
        from control_arena.settings.bash_arena.policies.honest_policy import (
            honest_policy,
        )
        from control_arena.settings.bash_arena.protocol.auto_submit_protocol_wrapper import (
            auto_submit_protocol_wrapper,
        )

        # BashArena's own config, less its reasoning_tokens=4000: Claude's
        # current models refuse an explicit thinking budget, so every attacker
        # sets its depth with the plan's reasoning_effort instead
        own = GenerateConfig(max_tokens=16000, reasoning_history="last").merge(config)
        return auto_submit_protocol_wrapper(
            micro_protocol=untrusted_policy_builder(
                attack=attack_policy(model_name=model, generate_config=own),
                honest=honest_policy(model_name=model, generate_config=own),
            )
        )
    return untrusted_policy_builder()


def _at(dataset: list[Sample], position: int) -> Sample:
    sample = dataset[position % len(dataset)]
    repeat = position // len(dataset)
    if repeat == 0:
        return sample
    return sample.model_copy(update={"id": f"{sample.id}__r{repeat}"})


if __name__ == "__main__":
    main(sys.argv[1])
