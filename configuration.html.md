# Configuration – Inspect Sentinel

## Overview

A task has one sentinel. You can give it one in three ways:

1.  In Python, with `Task(sentinel=...)`, when the sentinel is part of the task’s design.
2.  With `eval(sentinel=...)`, `eval_set(sentinel=...)` or `inspect eval --sentinel`, to replace the task’s sentinel for one run, for example to try a different monitor on the same task.
3.  In a YAML or JSON configuration file, so that you can change thresholds and swap monitors without editing code.

The eval log records the sentinel as a configuration, and [eval_retry()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval_retry) rebuilds the same sentinel from it.

The examples on this page use the rules and monitors from the repository’s `examples/` directory, the `no_network()` and `triage()` rules from the [Tutorial](./tutorial.html.md), `protected()` from [Final Decisions](./protocols.html.md#final-decisions), and the `suspicion()` LLM monitor.

## Task Sentinel

`Task(sentinel=...)` takes one of three forms. The first is one protocol:

``` python
sentinel=threshold(suspicion(), reject_at=0.8)
```

The second is a list of monitors and protocols, with at least one protocol:

``` python
sentinel=[no_network(), protected()]
```

The third is a mapping of instance names to monitors and protocols, again with at least one protocol:

``` python
sentinel={
    "guard": concurrent({"network": no_network(), "protected": protected()}),
    "audit": suspicion(),
}
```

A list or a mapping runs under a `concurrent()` protocol that you don’t need to write: every protocol in it runs, and the strongest decision wins. Monitors in a list or mapping have their suspicion scores recorded, but no protocol acts on them. A mapping lets you choose the instance names that appear in the log. See [Protocols](./protocols.html.md) for how `concurrent()` combines decisions.

A sentinel made only of monitors is an error, because nothing decides what happens at each step. Put the monitors under `threshold()` to act on their scores, or under `observe_only()` to record them without acting.

## Overriding the Sentinel

To run a task with a different sentinel, pass `sentinel` to [eval()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval). It replaces the task’s sentinel for that run:

``` python
from inspect_ai import eval

from inspect_sentinel import threshold
from llm_suspicion import llm_suspicion, suspicion

eval(
    llm_suspicion(),
    model="openai/gpt-5-mini",
    sentinel=threshold(suspicion(), reject_at=0.5),
)
```

`eval_set(sentinel=...)` works the same way. On the command line, pass `--sentinel` a configuration file or a registered name, both described below:

``` bash
inspect eval examples/llm_suspicion.py --model openai/gpt-5-mini \
  --sentinel sentinel.yaml
```

## Configuration Files

A configuration file is YAML or JSON with one top-level key, `sentinel`. Each monitor or protocol is an *entry*: a mapping with a `name` and optional `params`. Here is the sentinel from `examples/llm_suspicion.py`, `threshold(suspicion(), reject_at=0.8)`, as a file:

    sentinel.yaml

``` yaml
sentinel:
  name: threshold
  params:
    reject_at: 0.8
  monitors:
    - name: suspicion
```

The keys of an entry are:

| Key | Value |
|----|----|
| `name` | The registered name of the monitor or protocol factory, such as `threshold` or `suspicion`. |
| `params` | The factory’s arguments, other than the monitors and protocols under it. |
| The name of a factory parameter that takes monitors or protocols | The entries under it, written the same way. This is `monitors` for `threshold()` and `observe_only()`, and `children` for `concurrent()` and `sequential()`. |

An entry is built by calling its factory with these arguments, so a misspelled parameter fails as it would in Python. To run the task with this file:

``` bash
inspect eval examples/llm_suspicion.py --model openai/gpt-5-mini \
  --sentinel sentinel.yaml
```

You can also pass the file’s path to `Task(sentinel=...)`, `eval(sentinel=...)` or `eval_set(sentinel=...)`.

### Lists and Mappings

The value of `sentinel`, and of each key that holds monitors or protocols, can be one entry, a list of entries, or a mapping of instance names to entries, as in Python. This file is the mapping from `examples/nested.py`, with a `concurrent()` that has a mapping of its own:

    nested.yaml

``` yaml
sentinel:
  guard:
    name: concurrent
    children:
      network:
        name: no_network
      protected:
        name: protected
  audit:
    name: suspicion
```

This file is the `sequential()` from `examples/escalate_to_human.py`. Its children are a list: the `triage()` rule, then `human()`, which asks a person about the tool calls that `triage()` escalates. A parameter whose value is a list, such as `stages` for `human()`, is written as a YAML list:

    human.yaml

``` yaml
sentinel:
  name: sequential
  children:
    - name: triage
    - name: human
      params:
        stages: [tool_call]
```

The same format works in JSON. This file is a list of two rules:

    rules.json

``` json
{
  "sentinel": [
    {"name": "no_network"},
    {"name": "protected"}
  ]
}
```

In Python, `Task(sentinel=...)` and `eval(sentinel=...)` also accept these structures directly, as a list or a mapping of entry dictionaries.

## Registered Names

To use one monitor or protocol with no file, pass its registered name to `--sentinel`. Pass the factory’s arguments with `--sentinel-arg key=value`, once for each argument:

``` bash
inspect eval examples/no_network.py --model openai/gpt-5-mini \
  --sentinel human --sentinel-arg 'stages=[tool_call]'
```

This replaces the task’s `no_network()` rule with `human(stages=["tool_call"])`, which asks a person about every tool call. Values are parsed as YAML, as `-T` task arguments are, so `stages=[tool_call]` is a list and `0.8` is a number.

A registered name works only for a factory whose arguments are all plain values. A factory that takes monitors or protocols, such as `threshold()`, needs a configuration file. `--sentinel-arg` can’t be used with a configuration file: put the arguments under `params` in the file instead.

The `INSPECT_EVAL_SENTINEL` and `INSPECT_EVAL_SENTINEL_ARGS` environment variables set `--sentinel` and `--sentinel-arg`.

## How Names Are Found

A name in a configuration file or on the command line is looked up among the registered monitor and protocol factories:

- The built-in protocols (`threshold`, `concurrent`, `sequential`, `observe_only`, `human` and `handle_escalation`) are registered as `inspect_sentinel/threshold` and so on. The bare name `threshold` finds the same factory.

- The `@monitor` and `@protocol` decorators register a factory in your own code under its function name when its module is imported. Running a task file imports it, so a configuration for that task can name the factories the task file defines or imports. `examples/nested.py` imports `suspicion` and `no_network`, so both names work with it.

- A factory in an installed package is registered as `package/name`, such as `my_rules/no_network`. Inspect finds it when the package declares an `inspect_ai` entry point in its `pyproject.toml`:

  ``` toml
  [project.entry-points.inspect_ai]
  my_rules = "my_rules"
  ```

Use `@monitor(name=...)` or `@protocol(name=...)` to register a factory under a different name.

## Configuration Errors

Inspect validates a configuration when the eval starts, before any sample runs. An error names the entry by its position in the configuration. For example, a misspelled parameter in the monitor under `threshold`:

``` text
TypeError: sentinel.monitors[0]: suspicion() got an unexpected keyword argument 'modle'. Did you mean 'model'?
```

And an unknown name in the `concurrent()` under the instance named `guard`:

``` text
ValueError: sentinel.guard.children[0]: 'nope' is not a registered monitor or protocol.
```

## Model Roles

An LLM monitor gets its model from a model role, `monitor` by default. Set the role’s model on the command line with `--model-role`:

``` bash
inspect eval examples/llm_suspicion.py --model openai/gpt-5-mini \
  --model-role monitor=anthropic/claude-haiku-4-5
```

In Python, pass `model_roles={"monitor": "anthropic/claude-haiku-4-5"}` to [eval()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval) or `Task()`. Because the model is set outside the configuration, one configuration file works with any monitor model. See [LLM Monitors](./llm-monitors.html.md) for choosing the monitor model.

## The Sentinel in the Log

The log records the sentinel that ran in `eval.config.sentinel`, whether it came from the task, from [eval()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval) or from the command line. The record uses the configuration file format:

``` python
from inspect_ai.log import read_eval_log

log = read_eval_log("logs/<file>.eval")
print(log.eval.config.sentinel.model_dump_json(indent=2))
```

[eval_retry()](https://inspect.aisi.org.uk/reference/inspect_ai.html#eval_retry) and `inspect eval-retry` rebuild the sentinel from this record, so a retried eval runs the same monitors and protocols under the same instance paths. The names in the record must still be registered when you retry, which running the task file again does for the factories it defines or imports.

If a factory declares a version with `@monitor(version=...)` or `@protocol(version=...)`, the record includes it. When the recorded version differs from the installed factory’s, Inspect rebuilds the entry anyway and logs a warning that names both versions.
