# Implementation plan: the sidecar

How [sidecar.md](sidecar.md) becomes code. Each pull request can be reviewed on its own, has its own tests, and leaves `main` importable and passing. Proposed 2026-10-14.

Earlier prototype code exists for much of this. It is reused where it fits the design in `sidecar.md`, and rewritten where it does not.

## Repositories

| Repository | Owner | What lands there |
|---|---|---|
| `inspect_sentinel` | Meridian | the core package, the sidecar (`inspect_sentinel.sidecar`), and the proxy hooks, each a package of its own under `src/proxies` |
| `inspect_ai` | Meridian | `RemoteSentinel`, and the asks in `sidecar.md`, "Asked of inspect_ai" |
| `hawk` | METR | Middleman's passthrough hook, and routing `/sentinel/*` at the load balancer in front of Middleman |
| LiteLLM | the LiteLLM project | nothing. The guardrail depends only on LiteLLM's behaviour, and is tested against each LiteLLM version it supports |

## Order

```
inspect_sentinel                                              inspect_ai

1 reading traffic
        |
2 handler
        |
3 endpoint for proxies
        |
        +--------------------+
        |                    |
4 LiteLLM guardrail    5 Inspect clients in the sidecar
        |                    |
        |              +-----+------------+
        |              |                  |
        |        6 endpoint for evals   7 per-call reject and modify
        |              |                  |
        |              +--------+---------+
        |                       |
        |                       +-------------------------->  8 RemoteSentinel

10 Middleman: inspect_sentinel and hawk, after 3 and 5          9 smaller asks, at any time
```

PRs 1 to 3 come first, one after another, and land as one pull request with a commit for each, because the endpoint is the first point at which anything runs. After 3, the guardrail (4) and the sidecar's support for Inspect clients (5) can go in parallel. PR 4 gains the `sentinel` field once 5 has landed. PR 8 needs 5 and 6, and needs 7 for per-call rejection and `modify`. PR 9 needs nothing and can start at any time. PR 10 waits on METR accepting Middleman's passthrough hook.

## PR 1: reading provider traffic

Repository: `inspect_sentinel`. Depends on: nothing.

- Read Anthropic Messages, OpenAI Chat Completions and OpenAI Responses requests and replies, streamed and not streamed, into Inspect's types, with inspect_ai's converters and each SDK's accumulator (`sidecar.md` section 2).
- The `format` parameter, and working the API out from the provider and the body when it is absent.
- The rules for an unreadable request or reply: unknown content, a Beta type, more than one choice, a conversation stored at the provider, a background request (sections 2 and 6).
- Tests: a replay corpus of recorded replies, with the cases that must be unreadable beside them.

## PR 2: the handler

Repository: `inspect_sentinel`. Depends on: PR 1.

- Runs, keyed by a name the proxy passes on or by how a conversation opens, and conversations within a run (section 7).
- The four steps, found in the traffic: `AfterToolCall` and `BeforeGenerate` on a request, `AfterGenerate` and `BeforeToolCall` on a reply, each tool result judged once (section 3).
- Carrying out decisions on the request or reply held: `proceed`, refusing a whole turn, a `terminate` that keeps the run ended, rejected turns put back on later requests, and three rejections in a row ending the run (section 4).
- No `history` at any step, as the core defines it behind a proxy (section 8).
- The state behind one interface, with an in-memory implementation (section 10).
- Tests: the handler driven directly, with no proxy and no HTTP.

## PR 3: the endpoint for proxies

Repository: `inspect_sentinel`. Depends on: PR 2.

- JSON-RPC 2.0 at `POST /proxy/v1`: `judge_request` and `judge_reply`, answered with `pass`, `replace` or `refuse` (section 1).
- The server command: one port by default, `GET /health`, `--max-body-bytes`, `--time-limit` and `--reject-status`.
- The schema file `proxy_endpoint.schema.json`, generated from the endpoint's models, with a test that fails when it is stale.
- Tests: the endpoint through its methods, held to its schema file.

## PR 4: the LiteLLM guardrail

Repository: `inspect_sentinel`, package `inspect_sentinel_litellm` under `src/proxies/litellm`. Depends on: PR 3, and PR 5 for the `sentinel` field.

- `LiteLLMGuardrail`: `judge_request` and `judge_reply` on LiteLLM's three routes, holding a streamed reply whole, and LiteLLM's own setting for a sidecar that gives no answer (section 1).
- Passing `x-inspect-context` and `x-irid` to the sidecar without reading them, and removing both before the provider (section 7).
- Writing a refusal and a replaced reply in each of LiteLLM's forms, including the three cases where the guardrail writes the event itself so that the `sentinel` field arrives (section 1).
- Serving `/sentinel/*` through its own route in LiteLLM's app, behind LiteLLM's key check (section 1).
- The repair of a Claude `reasoning` item written through `/responses` (section 1).
- A README with its settings, and an example stack: LiteLLM, the sidecar, and an Inspect task.
- Tests: with LiteLLM's own classes; and live, with a real LiteLLM, a running sidecar and a stand-in provider, on every route, streamed and not, passed, edited and refused, for each supported LiteLLM version.

## PR 5: Inspect clients in the sidecar

Repository: `inspect_sentinel`, the sidecar and the core package. Depends on: PR 3.

- Reading `x-inspect-context`, and taking the run from its `sample_uuid` (section 7).
- Building the `sentinel` field from every step judged during an exchange: the stage, the call, the original call, the decision and the records; with what the exchange came to, and the turn withheld from the client (section 9).
- Answering an Inspect client's reply with `replace`, with the field added, and putting the field on `refuse` (sections 1 and 9). Other clients get the provider's bytes untouched.
- Trimming the field under `--no-eval-results` (section 9).
- In the core package: `Context.proxy`, with `inspect`, what `x-inspect-context` says ("Proposed for the core package").
- Tests: the field's contents for each outcome in section 9's table, for each provider format, streamed and not.

## PR 6: the endpoint for evals

Repository: `inspect_sentinel`. Depends on: PR 5.

- `GET /sentinel/info`, and JSON-RPC 2.0 at `POST /sentinel/v1`: `register_task`, `request_result`, `run_result` and `end_run` (section 9).
- Task registration: on unless `--no-runtime-task-registration`, replacing the last unless the task is in the tasks file, `expiry` capped by `--task-expiry`, a run taking its sample's facts at its first request, and `context.eval` from them.
- The tasks file, read with `--tasks FILE` (section 9). Producing it is asked of the workstream that builds `RemoteSentinel`.
- `--run-expiry`, dropping registrations Inspect made once their `expiry` passes, and `request_result` answering `dropped` for a while after a request's records go.
- The `sidecar` stanza in the sentinel config file, if the core team approves it ("Proposed for the core package").
- Tool views from a task's registration: inspect_ai's own viewers, the default view, and a task's own viewers where its package is installed (section 9).
- The second port and its own token, as an option.
- The schema file `eval_endpoint.schema.json`.
- Tests: each method, held to its schema file; a registration that changes, an ended run, and a run that expires.

## PR 7: per-call reject and modify

Repository: `inspect_sentinel`. Depends on: PRs 2 and 5.

- Taking a rejected call out of a reply, and changing a modified call, block by block in the provider's own document, for each provider format, streamed and not (section 4).
- Putting the call back, or the model's original call, on every later request of the conversation, and adding nothing when the request already holds it.
- Refusing the whole turn when every call in it is rejected, and ending the run after three such turns in a row. A turn in which some calls ran starts the count again (section 4).
- Tests: edited replies read back by each SDK; the put-back request equal to what Inspect sends in process.

## PR 8: `RemoteSentinel`

Repository: `inspect_ai`. Depends on: PRs 5 and 6, and PR 7 for per-call rejection and `modify`.

- The probe, `register_task` and the `required` or `optional` setting at task init (section 9).
- `x-inspect-context` and `x-irid` on each model request, only after the probe found a sentinel.
- Reading the `sentinel` field in each provider: on a reply, in the last event of a stream, and on an error, including where LiteLLM nests it (section 1).
- Acting on the field as Inspect does in process: a call taken out goes back into Inspect's conversation with an approval error, a changed call keeps the model's original, a `terminate` ends the sample, and every step is written into the transcript.
- `request_result` when the field is missing, and recording a request as not judged when the sidecar cannot say.
- `run_result` before scoring, and `end_run` after.
- Tests: in inspect_ai's own suite, with a stand-in sidecar; and the conformance suite in `sidecar.md` section 11, comparing Inspect's transcript in process and through a sidecar.

## PR 9: smaller asks of inspect_ai

Repository: `inspect_ai`. Depends on: nothing. Each can be its own pull request, and each needs an accepted issue in inspect_ai first.

- Public OpenAI converters that read a dict as leniently as the SDK does.
- An Anthropic converter that says so when it meets a block it does not recognise, or a Beta type.
- Chat Completions converters that read a custom tool call, as the Responses converters do.
- An Anthropic converter that reads every block of a document whose source is a list of blocks.
- A public function that writes a message in Anthropic's format.
- Public functions that read the tools, the tool choice and the generation settings from a request, in each provider's format.
- A public function that describes a tool as the model is offered it, as a `ToolInfo`.
- A public function for each provider that turns stream events into a `ModelOutput`.
- A public way to obtain the running sample's uuid and attempt id before its first model call.
- A public way to make the default view of a tool call.

## PR 10: Middleman

Repositories: `inspect_sentinel` and `hawk`. Depends on: PRs 3 and 5, and METR accepting Middleman's passthrough hook.

- In `inspect_sentinel`: `MiddlemanHook`, a package under `src/proxies/middleman`, calling `judge_request` and `judge_reply`, passing on and removing Inspect's two headers, and putting the `sentinel` field into Middleman's errors.
- In `hawk`: the passthrough hook in Middleman, which is METR's to accept; Middleman letting the `sentinel` field through on its errors; and routing `/sentinel/*` to the sidecar at the load balancer in front of Middleman.
- A README for the hook, including the load balancer's routing.

## Not planned yet

These need design first. Each names the repositories it would touch.

- Asking a person from the sidecar: the sidecar holding the turn, `RemoteSentinel` showing the question in Inspect's approval UI, and the answer coming back on the person's own credential. `inspect_sentinel` and `inspect_ai`.
- How Inspect says that a run is new or resumed. `inspect_sentinel` and `inspect_ai`.
- Model calls that Inspect answers from its cache, which never reach the proxy. `inspect_ai`, and possibly `inspect_sentinel`.
- Filling `step.sent` from each conversation the sidecar keeps, with the turns before a compaction, once the core has `step.sent`. `inspect_sentinel`.
- Tallies kept at the task level. `inspect_sentinel`.
- A credential tied to the evals it may name. `inspect_sentinel`.
- The Envoy shim. `inspect_sentinel`.
- A state implementation that processes share and that outlasts a restart. `inspect_sentinel`.
