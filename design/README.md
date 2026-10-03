# Design

Eight documents. **Start with [sentinel-overview.md](sentinel-overview.md)**, which is the short form written to solicit feedback on the concepts and Python API. [sentinel.md](sentinel.md) is the full design and the place arguments are made; [sentinel-reference.md](sentinel-reference.md) restates its conclusions as a reference, without the reasoning. The remaining three each take one concern the core design depends on.

| | covers | status |
|---|---|---|
| [sentinel-overview.md](sentinel-overview.md) | **The short form.** Monitors and protocols, the four stages, actions, context, the shipped protocols, and one paragraph each on deployment and development. | Feedback draft |
| [sentinel.md](sentinel.md) | **The full design.** Prior art, the payloads and context, what monitors and protocols return, the protocol layer (runner, compositions, boundary check, humans), registration, state, configuration, transcript, views, and the relationship to approval and review. Ends with the open questions. | Partly built; the Python for unbuilt parts is illustrative |
| [sentinel-reference.md](sentinel-reference.md) | **The reference form** of sentinel.md: types, rules, and rationale sections, without the argument. | Tracks sentinel.md |
| [sentinel-deployment.md](sentinel-deployment.md) | Running a sentinel **outside the eval process**, in a proxy on the wire: what a proxy can see, the host ABI, portability, and the sidecar and WASM execution modes. | Measured where marked, reasoned elsewhere |
| [sentinel-development.md](sentinel-development.md) | **Measuring and calibrating** a monitor before it acts: replaying it over transcripts as an Inspect Scout scanner, step ids, validation, and threshold calibration. | Sketch; Scout facts measured 2026-09-09 |
| [pr-series.md](pr-series.md) | **How the design becomes code**: the first sentinel PRs and the inspect_ai registry PR, with the decisions taken along the way (`Recorder`, `HostContext`, tool stages only, and the later decision records). | Kept current; ends with the Deferred list |
| [workstreams.md](workstreams.md) | **Who can work on what**: the areas that can be owned separately in priority order (high: inspect_core, generate stages, LLM affordances for monitors, a host in a proxy; next: shipped protocols and helpers, auto-mode approvers, bridged agents and deployment, building and validating monitors, a portability linter; lower: monitoring the monitors, remote human surfaces; done: `sequential()` and `human()`), with owners where assigned, each with its design doc and what it touches. | Current as of 2026-10-02 |
| [inspect-core.md](inspect-core.md) | Extracting Inspect's **wire types** into a leaf package light enough for this one, or another language, to depend on. | Measured where marked, reasoned elsewhere |

These documents are canonical. They were drafted on the `design/monitor` branch of [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai), which is no longer updated. Issue numbers (`#5423`, `#5355`) and file paths in them refer to that repository.
