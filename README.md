# Inspect Sentinel

Monitors and control protocols that watch agents and intervene in their execution.

A **monitor** observes a step of an agent's execution and reports a suspicion score; it never acts. A **protocol** decides what happens at that step, optionally after consulting monitors, and a protocol with no monitors is a rule. Both run at one of four points in the agent loop: before or after a model generate, and before or after a tool call. The same monitor runs inside an Inspect eval, in a network proxy in front of a model API, and offline over recorded transcripts for validation and calibration.

```bash
pip install git+https://github.com/meridianlabs-ai/inspect_sentinel
```

This also installs the development version of `inspect_ai` from its `main` branch, which Sentinel tracks.

The package is under active design. The design documents are in [`design/`](design/); start with [`sentinel-overview.md`](design/sentinel-overview.md).

Documentation: <https://meridianlabs-ai.github.io/inspect_sentinel>.
