# Inspect Sentinel

Monitors and control protocols that watch agents and intervene in their execution.

A **monitor** observes a step of an agent's execution and reports a suspicion score; it never acts. A **protocol** decides what happens at that step, optionally after consulting monitors, and a protocol with no monitors is a rule. Both run before a tool call executes or after its result comes back; stages around model generation are planned. The same monitor runs inside an Inspect eval and offline over recorded transcripts with Inspect Scout for validation and calibration; running it in a network proxy in front of a model API is planned.

```bash
pip install inspect-sentinel
```

It requires `inspect-ai` 0.3.278 or later.

The package is under active design. The design documents are in [`design/`](design/); start with [`sentinel-overview.md`](design/sentinel-overview.md).

Runnable examples are in [`examples/`](examples/).

Documentation: <https://meridianlabs-ai.github.io/inspect_sentinel>.
