# Configuration – Inspect Sentinel

Tools that run a sentinel somewhere other than an eval, such as a scan that replays it over logs, use these to carry it between processes as its configuration and to tell which steps it reads. See [Configuration](../configuration.html.md) for the configuration’s shape.

### sentinel_config

Record constructed monitors and protocols as the configuration that rebuilds them.

The inverse of [sentinel_from_config()](../reference/config.html.md#sentinel_from_config), for the eval log and retry: each instance becomes an entry with its registry name, the params it was created with, and its factory’s version unless that is 0, and a param holding monitors or protocols becomes nested entries. A package monitor or protocol is recorded by its bare name when that finds it unambiguously. A lone instance is recorded as a lone entry, so it rebuilds as the root it was.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/5e7b6470495de1113088a4e5b0fc813b1aab35b9/src/inspect_sentinel/_config.py#L262)

``` python
def sentinel_config(sentinels: Sentinels) -> SentinelConfig
```

`sentinels` Sentinels  
One monitor or protocol, or a sequence or mapping of instance names to them, as `Task(sentinel=)` accepts.

### sentinel_from_config

Build the monitors and protocols a configuration describes.

Each entry is constructed through the registry with its `params` and its nested entries, which are built first. An entry whose recorded `version` differs from the installed factory’s is built anyway, with a warning naming both versions; its `meta` is ignored. One entry, or a bare registered name, builds one instance, so it resolves as the root itself; a list or mapping builds a list or mapping. The result is not resolved into its root: it is what `Task(sentinel=)` takes.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/5e7b6470495de1113088a4e5b0fc813b1aab35b9/src/inspect_sentinel/_config.py#L44)

``` python
def sentinel_from_config(
    config: str | SentinelConfig | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> Sentinels
```

`config` str \| SentinelConfig \| Sequence\[Mapping\[str, Any\]\] \| Mapping\[str, Any\]  
A YAML or JSON file whose only key is `sentinel`, a registered monitor or protocol name, or the configuration itself: one entry, a list of entries, or a mapping of instance names to entries.

### watched_stages

The step payload types a configured monitor or protocol, or any monitor or protocol given to its factory, watches.

A composition that only forwards the step to its children, such as `concurrent`, `sequential` or `observe_only`, watches what its children watch, and `handle_escalation`, which only answers escalations from earlier links, watches nothing of its own. Any other function watches the stages it accepts, so a protocol annotated `Step` watches every stage.

[Source](https://github.com/meridianlabs-ai/inspect_sentinel/blob/5e7b6470495de1113088a4e5b0fc813b1aab35b9/src/inspect_sentinel/_decorators.py#L250)

``` python
def watched_stages(sentinel: Sentinel) -> frozenset[type[Any]]
```

`sentinel` Sentinel  
An instance returned by a `@monitor` or `@protocol` factory.
