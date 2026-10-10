# inspect_judge: related design

Monitor-facing model and prompt work designed in other documents, and how it relates to [inspect-judge.md](inspect-judge.md).

| Item | Where | Status there | Bearing on this plan |
|---|---|---|---|
| `messages_as_str`, `message_as_str`, `MessageFormatOptions`, a public tool-call renderer | `pr-series.md`, "Shared message rendering" (decided 2026-10-06); `sentinel.md`, "Views" | Not started. Moves from Scout into inspect_ai (`inspect_ai.model`, later `inspect_ai.core`), output exactly Scout's. Supersedes `sentinel.md`'s "lift into `inspect_sentinel`". Lessons from closed sentinel #48 recorded there. | Rendering does not go in `inspect_judge`. |
| `MessagesPreprocessor` transform, `message_numbering`, `extract_refs` | `pr-series.md` (same item); `sentinel.md`, "The helpers"; `sentinel-development.md` | `pr-series.md` keeps them in Scout; `sentinel.md` has monitors use them. | They move down into `inspect_judge` (decided 2026-10-08), since Scout sits above sentinel; Scout re-exports them. |
| Windowing: `last_turn`, `last_n`, `new_since_last_report` | `sentinel.md`, "The helpers"; `sentinel-reference.md` | Designed, not built. | The author passes a window as `score`'s `history`. `pr-series.md` has `last_turns(messages, n)` instead, with turn rules; names to reconcile. Not `inspect_judge`'s. |
| `step_as_str`, `briefing_as_str`, `call_as_str`, `result_as_str` | `sentinel.md`; `sentinel-reference.md`; `pr-series.md` (`step_as_str` spec) | Designed, not built. Sentinel keeps `step_as_str`. | Sentinel's, not `inspect_judge`'s. `score` renders `{{ step }}` with `step_as_str`; `briefing_as_str` is close to `{{ task }}`. |
| `monitor_prompt(context, step, *, question, answer=..., history=...)` and the evidence envelope | `sentinel.md`, "The helpers" and feature direction 2; `sentinel-reference.md`; open question 17 (default history window) | Designed, not built. | Overlaps `score`, whose template does the same job: instructions, the agent's step inside a delimited block, the answer format. Its example's `parse_score` is what `score` replaces. |
| A helper for escaping untrusted agent text | `pr-series.md`, "Views and prompt helpers" | Raised, not designed. | Belongs in `step_as_str`, so that `{{ step }}` cannot close its block. |
| A structured-verdict helper | `pr-series.md`, "Views and prompt helpers" | Raised, not designed. | `Host.score` is this helper. |
| Chunking long inputs and reducing | `workstreams.md` §3 | Scope only. | Rare for monitors (incremental by default, `sentinel.md`). |
| Prompt caching: a stable prefix across steps | `workstreams.md` §3; `pr-series.md` (`step_as_str` renders oldest first); `sentinel-development.md`, "Cost and parallelism" | Scope and a rendering rule. | Deferred; see inspect-judge.md, Open. |
| Monitor inference exempt from the agent's limits; `monitor` default role; usage separable from the agent's | `sentinel.md`, "Inference, budget, and injection" | Decided. | Applies to `score` as to `generate`. |
| Retries and timeouts belong to the host | `sentinel-deployment.md`, "Other constraints on `fetch`" | Decided. | `score`'s format retries are the host's too. |
| Messages as model input, for screenshots | `sentinel.md`, feature direction 6 | Done for `generate`. | See inspect-judge.md, Open (images). |
| References from cites | `sentinel.md`; `sentinel-reference.md` (`extract_refs(explanation)` into `references`) | Designed. | `score` numbers the history it renders and fills `references` itself. |
