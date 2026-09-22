# Design

Seven documents. **Start with [sentinel-overview.md](sentinel-overview.md)**, which is the short form written to solicit feedback on the concepts and Python API. [sentinel.md](sentinel.md) is the full design and the place arguments are made; [sentinel-reference.md](sentinel-reference.md) restates its conclusions as a reference, without the reasoning. The remaining three each take one concern the core design depends on.

| | covers | status |
|---|---|---|
| [sentinel-overview.md](sentinel-overview.md) | **The short form.** Monitors and protocols, the four steps, actions, context, the shipped protocols, and one paragraph each on deployment and development. | Feedback draft |
| [sentinel.md](sentinel.md) | **The full design.** Prior art, the payloads and context, what monitors and protocols return, the protocol layer (runner, compositions, boundary check, humans), registration, state, configuration, transcript, views, and the relationship to approval and review. Ends with the open questions. | Sketch; the Python is illustrative |
| [sentinel-reference.md](sentinel-reference.md) | **The reference form** of sentinel.md: types, rules, and rationale sections, without the argument. | Tracks sentinel.md |
| [sentinel-deployment.md](sentinel-deployment.md) | Running a sentinel **outside the eval process**, in a proxy on the wire: what a proxy can see, the host ABI, portability, and the sidecar and WASM execution modes. | Measured where marked, reasoned elsewhere |
| [sentinel-development.md](sentinel-development.md) | **Measuring and calibrating** a monitor before it acts: replaying it over transcripts as an Inspect Scout scanner, step ids, validation, and threshold calibration. | Sketch; Scout facts measured 2026-09-09 |
| [pr-series.md](pr-series.md) | **How the design becomes code**: the first four sentinel PRs and the inspect_ai registry PR, with the decisions taken along the way (`Recorder`, `RunnerContext`, tool stages only). | PR 0 open; PR 1 next |
| [inspect-core.md](inspect-core.md) | Extracting Inspect's **wire types** into a leaf package light enough for this one, or another language, to depend on. | Measured where marked, reasoned elsewhere |

These documents were drafted on the `design/monitor` branch of [inspect_ai](https://github.com/UKGovernmentBEIS/inspect_ai) and moved here when the package was created. Issue numbers (`#5423`, `#5355`) and file paths in them refer to that repository.
