# Implementation plan: the sidecar

How [sidecar.md](sidecar.md) becomes code. Each pull request can be reviewed on its own, has its own tests, and leaves `main` importable and passing. Proposed 2026-10-14; Envoy, incremental streaming and human review added 2026-10-17; the work after step 7 reordered and renumbered, with streaming split into proxy and Inspect client phases, on 2026-10-19; TLS on the sidecar's listeners added as step 16, before Middleman, on 2026-10-20. Measuring what a sentinel did added as steps 18 to 20 on 2026-11-16, and the record log as step 21 on the same day; reordered as steps 18 to 24 on 2026-11-18, at Meridian's request, so typed log fields and the viewer come first.

Earlier prototype code exists for much of this. It is reused where it fits the design in `sidecar.md`, and rewritten where it does not.

## Repositories

| Repository | Owner | What lands there |
| --- | --- | --- |
| `inspect_sentinel` | Meridian | the core package, the sidecar (`inspect_sentinel.sidecar`) with its Envoy shim, and the proxy hooks, each a package of its own under `src/proxies` |
| `inspect_ai` | Meridian | Inspect's sidecar client, and the asks in `sidecar.md`, "Asked of inspect_ai"; the typed log fields of step 18, a sentinel inside Inspect filling them in step 20, the sidecar client filling them in step 22, and step 23's columns in `inspect_ai.analysis` |
| ts-mono | Meridian | the log viewer: types generated from inspect_ai's schema, and how it shows the fields of step 18 |
| `inspect_scout` | Meridian | its schema, regenerated when inspect_ai's `SentinelEvent` changes |
| `hawk` | METR | Middleman's passthrough hook, and routing `/sentinel/*` at the load balancer in front of Middleman |
| LiteLLM | the LiteLLM project | nothing. The guardrail depends only on LiteLLM's behaviour, and is tested against each LiteLLM version it supports |

## Order

After steps 1–7, the main delivery path is **8 model-facing history → 9 Envoy → 10 proxy streaming → 11 Inspect sidecar client → 13 human review**. Step 12, Inspect incremental streaming, was dropped ([below](#step-12-dropped)). This builds and validates Envoy and incremental streaming in `inspect_sentinel` before implementing the client in `inspect_ai`. Discuss the Inspect client contract with Meridian while the proxy work proceeds. Smaller upstream asks (14) can proceed independently. Trusted run identity (15) and TLS on the sidecar's listeners (16) come before Middleman (17), whose implementation waits on Hawk/METR accepting the gateway change; identity work can proceed through Envoy and LiteLLM while that approval is pending.

```text
inspect_sentinel

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
        |                       v
        |                  8 model-facing history
        |                       |
        |                       v
        |                  9 Envoy shim
        |                       |
        +-----------------------+
                                v
                   10 proxy incremental streaming
                                |
                                v
                   11 Inspect sidecar client                 inspect_ai
                                |
                                v
                   13 human review                  both repositories

14 smaller asks: inspect_ai, independently, after accepting each issue
15 trusted run identity: after 6 and 11, before Middleman; no Hawk dependency
16 TLS on the sidecar's listeners: after 4, 6 and 9; before Middleman
17 Middleman: inspect_sentinel and hawk, after 3 and 5; scheduled after 15 and 16
   Implementation also needs Hawk/METR accepting the passthrough hook
   Middleman streaming follows 10 and 17 and METR accepting chunk delivery
18 typed fields in the log: inspect_ai, ts-mono types, inspect_scout schema
19 the log viewer shows them: ts-mono, after 18
20 a sentinel inside Inspect fills them: inspect_ai, after 18
21 times and counts in results: inspect_sentinel, after 6 and 13
22 the sidecar client fills them: inspect_ai, after 11, 18 and 21
23 reading the measures from logs: inspect_ai columns and inspect_sentinel, after 18, 20 and 22
24 the record log: inspect_sentinel, after 21
shared state: inspect_sentinel, before more than one sidecar worker or restart continuity
```

Steps 1 to 3 come first, one after another, and land as one pull request with a commit for each, because the endpoint is the first point at which anything runs. After 3, the sidecar's support for Inspect clients (5) and the guardrail (4) land together as one pull request, with a commit for each: the guardrail's tests run a real sidecar. The endpoint for evals (6) and per-call rejection and `modify` (7) land together after that as one pull request, with commits for each, so that one review covers both.

Step 8 comes next: the sidecar fills `step.history` from the traffic it keeps, so that step 9's model-history conformance checks run through Envoy against it. It needs nothing beyond step 7. Step 9 then builds the Envoy shim with whole-response holding. Step 10 adds incremental streaming to the shared handler, Envoy and LiteLLM, tested with ordinary provider SDK clients and a small client of the eval endpoint. Neither needs Inspect's sidecar client or changes to `inspect_ai` source. Whole-response holding remains available for policies that need the full turn.

Step 11's technical prerequisites remain 5, 6 and 7; placing it after 10 is a delivery choice, so the proxy behavior and recovery contract are exercised before a client depends on them. From 7 it needs per-call rejection and `modify`, and also that a rejection Inspect puts back into its conversation is not judged as a tool result: before 7, the sidecar would judge each such rejection as a call that ran, and could end the run for a call that never ran. Step 11 establishes the Inspect client with whole-response holding. The client keeps whole-response holding, since step 12, which would have connected it to replies released early, was dropped. Step 13 follows and adds human review through the sidecar's host, Inspect's approval UI and external review consumers.

[Shared and durable state](#follow-up-shared-and-durable-state) is delivered before the sidecar runs as more than one worker or promises that a run outlasts a restart; until then a deployment is one process. Step 14 needs nothing and can start at any time, with each upstream issue accepted first. Step 15 implements trusted run identity after registration and the Inspect client, using Envoy and LiteLLM first, without waiting for a Hawk change. Step 16 adds TLS, and client certificates, on every port the sidecar listens on, for Envoy and LiteLLM first. Step 17 is scheduled after steps 15 and 16 and waits on Hawk/METR accepting Middleman's passthrough hook. Extending Middleman to hand over chunks also needs step 10 and METR accepting that change; it does not block Envoy, LiteLLM, identity or the Inspect client work.

Steps 18 to 23 let an operator report what a sentinel did: coverage, review latency and escalation rate ([sidecar.md, section 12](sidecar.md#12-measuring-what-a-sentinel-did)). They hold up nothing else. Step 18 comes first: it adds typed fields to inspect_ai's log format, a mechanical change, so that a sentinel inside Inspect and a sidecar fill the same fields, and the viewer and the schema know them. Step 19 shows them in the log viewer. Then the fields are filled: step 20 for a sentinel inside Inspect, and, through a sidecar, step 21 adds times and totals to the sidecar's results and step 22 writes them in. Step 21 builds on step 13's record of each question, and doesn't need step 18, so it can proceed in parallel. Step 23 reads the fields from many logs.

Step 24 writes what the sidecar records to a log outside the sidecar. It follows step 21, because each entry in the log holds the times and the stage outcomes that step 21 adds. It holds up nothing else. An agent with no eval has no eval log, so step 24 is what gives it a record of what a sentinel decided, and the totals that step 21 leaves out.

## Step 1: reading provider traffic

Repository: `inspect_sentinel`. Depends on: nothing.

- Read Anthropic Messages, OpenAI Chat Completions and OpenAI Responses requests and replies, streamed and not streamed, into Inspect's types, with inspect_ai's converters and each SDK's accumulator (`sidecar.md` section 2).
- The `format` parameter, and working the API out from the provider and the body when it is absent.
- The rules for an unreadable request or reply: unknown content, a Beta type, more than one choice, a conversation stored at the provider, a background request (sections 2 and 6).
- Tests: a replay corpus of recorded replies, with the cases that must be unreadable beside them.

## Step 2: the handler

Repository: `inspect_sentinel`. Depends on: step 1.

- Runs, keyed by a name the proxy passes on or by how a conversation opens, and conversations within a run (section 7).
- The four steps, found in the traffic: `AfterToolCall` and `BeforeGenerate` on a request, `AfterGenerate` and `BeforeToolCall` on a reply, each tool result judged once (section 3).
- Carrying out decisions on the request or reply held: `proceed`, refusing a whole turn, a `terminate` that keeps the run ended, rejected turns put back on later requests, and three rejections in a row ending the run (section 4).
- No `agent_messages` at any step, as the core defines it behind a proxy (section 8).
- The state behind one interface, with an in-memory implementation (section 10).
- Tests: the handler driven directly, with no proxy and no HTTP.

## Step 3: the endpoint for proxies

Repository: `inspect_sentinel`. Depends on: step 2.

- JSON-RPC 2.0 at `POST /proxy/v1`: `judge_request` and `judge_reply`, answered with `pass`, `replace` or `refuse` (section 1).
- The server command: one port by default, `GET /health`, `--max-body-bytes`, `--judging-timeout` and `--reject-status`.
- The schema file `proxy_endpoint.schema.json`, generated from the endpoint's models, with a test that fails when it is stale.
- Tests: the endpoint through its methods, held to its schema file.

## Step 4: the LiteLLM guardrail

Repository: `inspect_sentinel`, package `inspect_sentinel_litellm` under `src/proxies/litellm`. Depends on: step 3. Lands with step 5, in one pull request.

- `LiteLLMGuardrail`: `judge_request` and `judge_reply` on LiteLLM's three routes, holding a streamed reply whole, and LiteLLM's own setting for a sidecar that gives no answer (section 1).
- Passing `x-inspect-context` and `x-irid` to the sidecar without reading them, and removing both before the provider (section 7).
- Writing a refusal and a replaced reply in each of LiteLLM's forms, including the cases where the guardrail writes the error event itself, so that the provider SDK raises with the refusal's status (section 1).
- Serving `/sentinel/*` through its own route in LiteLLM's app, behind LiteLLM's key check (section 1).
- The repair of a Claude `reasoning` item written through `/responses` (section 1).
- A README with its settings, and an example stack: LiteLLM, the sidecar, and an Inspect task.
- Tests: with LiteLLM's own classes; and live, with a real LiteLLM, a running sidecar and a stand-in provider, on every route, streamed and not, passed, edited and refused, for each supported LiteLLM version.

## Step 5: Inspect clients in the sidecar

Repository: `inspect_sentinel`, the sidecar and the core package. Depends on: step 3.

- Reading `x-inspect-context`, and taking the run from its `sample_uuid` (section 7).
- Building the judgment of an exchange from every step judged during it: the stage, the call, the decision and the records; with what the exchange came to, and the turn withheld from the client or proposed by the model, with each call a decision acted on redacted (section 9).
- Every client, an Inspect client included, gets the provider's reply or a provider-format error, with nothing added (sections 1 and 9).
- Trimming the judgment under `--no-eval-results` (section 9).
- In the core package: `Context.proxy`, with `inspect`, what `x-inspect-context` says ("Proposed for the core package").
- Tests: the judgment's contents for each outcome in section 9's table, for each provider format, streamed and not.

## Step 6: the endpoint for evals

Repository: `inspect_sentinel`. Depends on: step 5.

- `GET /sentinel/info`, and JSON-RPC 2.0 at `POST /sentinel/v1`: `register_task`, `request_result`, `sample_result` and `end_sample` (section 9).
- Task registration: on unless `--no-runtime-task-registration`, replacing the last unless the task is in the tasks folder, where it is answered as already registered, with `fixed` true, and nothing is stored, `expiry` capped by `--task-expiry`, a run taking its sample's facts at its first request, and `context.eval` from them.
- The tasks folder, of tasks files each with a `tasks` list, named by `tasks_folder` or `--tasks-folder` (section 9). Producing the files is asked of the workstream that builds Inspect's sidecar client.
- `--run-expiry`, dropping registrations Inspect made once their `expiry` passes, and `request_result` answering `dropped` for a while after a request's records go.
- The configuration file: the `sentinel` stanza, a `server` stanza of the sidecar's settings, and `tasks_folder` (section 10).
- Tool views from a task's registration: inspect_ai's own viewers, the default view, and a task's own viewers where its package is installed (section 9).
- The second port and its own token, as an option.
- The schema file `eval_endpoint.schema.json`.
- Tests: each method, held to its schema file; a registration that changes, an ended run, and a run that expires.

## Step 7: per-call reject and modify

Repository: `inspect_sentinel`. Depends on: steps 2 and 5.

- Taking a rejected call out of a reply, and changing a modified call, block by block in the provider's own document, for each provider format, streamed and not (section 4).
- Putting the call back, or the model's original call, on every later request of the conversation, and adding nothing when the request already holds it.
- Refusing the whole turn when every call in it is rejected, and ending the run after three such turns in a row. A turn in which some calls ran starts the count again (section 4).
- Tests: edited replies read back by each SDK; the put-back request equal to what Inspect sends in process.

## Step 8: model-facing history

Repository: `inspect_sentinel`. Depends on: step 7. This is the next implementation step after steps 6 and 7, before step 9's transcript and model-history conformance checks run through Envoy. The core's as-sent `step.history` landed in sentinel #98, so this waits on no core change.

- Fill `step.history` from the traffic seen for each conversation, keeping the messages from before a compaction, and following Inspect's as-sent history semantics. Keep different runs, sample attempts and conversations separate.
- Keep `agent_messages` as `None`: the proxy cannot know the scaffold's own list. Do not invent history the sidecar never saw; use `None` where it cannot establish it.
- Tests: compare the history given to a monitor in process and through the sidecar before and after compaction, on retries within an attempt, and on a fresh sample attempt. A new attempt must not inherit the discarded attempt's history.

## Step 9: the Envoy shim

Repository: `inspect_sentinel`. Depends on: steps 3, 6, 7 and 8. Does not depend on step 11, Middleman or WASM.

- A Python gRPC `ext_proc` shim inside the sidecar, with dependencies in an optional extra. It translates Envoy's exchanges into the shared handler's `Request` and `Reply`, and applies `Pass`, `Replace` and `Refuse`; provider reading, sentinel execution and run state stay in the shared implementation.
- Anthropic Messages, OpenAI Chat Completions and OpenAI Responses, including request amendments, edited replies, and refusals. Response bodies use `FULL_DUPLEX_STREAMED` from the first version, but the shim initially holds each reply whole (section 5). Incremental release follows in step 10.
- Envoy 1.39.1 or later. `FULL_DUPLEX_STREAMED` was added in 1.34, and releases after it fixed `ext_proc` bugs that truncated or duplicated bodies, which would corrupt a reply the shim holds and re-emits. The example and the tests pinned `envoyproxy/envoy:v1.39.4`, the latest 1.39 release when the PR was built, and now pin `v1.39.6`, which no longer trusts the compromised eMudhra CAs, whose `ext_proc` protos are those of 1.39.1, where the measurements in `sidecar.md` were taken. Release 1.40.0, a week old then, was not taken.
- Generated Envoy configuration and a runnable example: provider routes, request and response processing, message and body limits, timeouts long enough for sentinel decisions, and requesting uncompressed provider replies. Sidecar failures refuse traffic by default, and an unexpected content encoding is refused when it cannot be read and rewritten correctly.
- Inspect's request context reaches the handler, `request_result` answers for each exchange, replies and refusals alike, and `/sentinel/*` reaches the endpoint for evals under the deployment's access controls. Exercise these with a small test client of the existing contract, without implementing Inspect's sidecar client or changing `inspect_ai` source.
- Tests with a real Envoy, the sidecar and a controlled provider, on all three formats, streamed and not, passed, edited and refused. Cover bodies above Envoy's ordinary buffer and gRPC defaults, split events, response trailers, provider errors, sidecar failures, hold limits and client disconnects. Run the model-history conformance cases through Envoy as well.

Done when the generated configuration runs the example and the same sentinel decisions, records and model-facing conversation survive an exchange through Envoy as through the existing handler.

## Step 10: proxy incremental streaming

Repository: `inspect_sentinel`. Depends on: steps 4 and 9. Does not depend on step 11 or require changes to `inspect_ai` source. This phase builds and validates the streaming machinery with ordinary provider SDK clients before Inspect implements a remote client.

- An incremental handler interface and a corresponding transport for proxy hooks, alongside the existing whole-exchange methods. Envoy's shim feeds it chunks directly; LiteLLM's guardrail sends chunks as they arrive. Provider decoding continues to use inspect_ai and the provider SDKs, and sentinels still judge complete steps.
- Settle and document what a sentinel sees under early release before implementing the incremental path. A monitor must not have to know that a reply was streamed: every step keeps the whole-turn meaning it has in Inspect's own process. Retain the existing whole-response behavior by default, and make early release an explicit setting. Settled in [`sidecar.md`, "What a sentinel sees under early release"](sidecar.md#what-a-sentinel-sees-under-early-release): `release: early` in the `server` stanza, and a turn's calls judged once it has ended, with the whole turn, so no sentinel ever sees part of one and none needs whole holding to be judged correctly. Judging a call before its turn ends is not built; a workload that needs it would be a monitor's own opt-in.
- Stream text and reasoning as they arrive where early release is allowed. Hold every byte of a tool call, and everything after it, until the turn has been judged, then release, modify or remove each call. Preserve event order, and establish for each of the three provider formats which of its events carry text and reasoning, rather than assuming their stream lifecycles are interchangeable. Settled in [`sidecar.md`, "Releasing a reply as it arrives"](sidecar.md#releasing-a-reply-as-it-arrives): the text and reasoning before the first call go out as they arrive, and everything from the first call on is held until the turn is judged, with the error event each SDK raises on.
- Preserve step 7's rejection and modification semantics, the original calls restored in later model requests, and termination remembered for the run. Define what happens when every call is rejected, or the run is ended, after preceding text has gone out. Released content cannot be recalled. After the response status is committed, refusals and failures use an error event the client SDK recognises, with no successful stream ending in its place.
- Keep request and run results correct on incremental replies and failures, including when results are trimmed. A small test client exercises the recovery contract: for a cut stream, `request_result` must still expose the available decisions and turns, or explicitly report a pending judgment, a failure to judge, dropped records or an unknown request. This validates the protocol; writing Inspect's transcript and acting on the recovered outcome were left to step 12, which was dropped. Checked by `tests/sidecar/test_stream_recovery.py`: a reply released early is recovered as decided with its outcome, the turn and each decision, as pending while its turn is judged after its client went and then decided, as cut off when its client went before the turn ended, as not judged when the sentinel failed, and as dropped or unknown.
- Keep connections alive while tool calls wait, with provider-compatible keepalives. Bound queues and held bytes, apply backpressure and hold deadlines, and define cancellation and record retention when the client disconnects or a proxy stops waiting. Keepalives settled in [`sidecar.md`, "Releasing a reply as it arrives"](sidecar.md#releasing-a-reply-as-it-arrives): an SSE comment after `keepalive` seconds of silence, under `release: early` only.
- Deliver and test the incremental path through both Envoy and LiteLLM. A proxy or provider format without a validated incremental path continues to use whole-response holding. Middleman's streaming extension is a separate follow-up in `inspect_sentinel` and `hawk`, after steps 10 and 17 and METR accepting chunk delivery. It uses the same interface and does not block this phase; until it lands, Middleman holds replies whole.

Acceptance tests use a provider and a sentinel whose progress is controlled: preceding text must reach the client before the provider finishes or the sentinel answers, and no part of a tool call may reach it before approval. Cover multiple calls, mixed decisions, modification, rejection of every call, termination after earlier release, split and combined chunks, malformed or interrupted streams, keepalives, backpressure and disconnects. Each provider SDK must read successful edited streams and raise on refused ones. Replies released early must retain the model-facing history of the whole-response path, including the original calls and rejection results on later requests, and give each call the same step it gets when the reply is held whole; `release: whole` must demonstrably keep the reply held. Check endpoint recovery with the test client.

## Step 11: Inspect client for the sidecar

Repositories: `inspect_ai`, with the sidecar's side in `inspect_sentinel`. Depends on: steps 5, 6 and 7 (above). Design: [sidecar.md, section 9](sidecar.md#9-a-connection-between-an-eval-and-its-sentinel).

Scheduled after step 10. The client uses whole-response holding, and refuses a sidecar that releases replies early. While steps 9 and 10 are built, agree the client contract with Meridian: activation and discovery failure, registration requirements, model-cache behavior, request and message identity, retries, missing results, scoring and cleanup. Review the trusted-host scope alongside #117 before implementing the client. Keep ordinary evals and in-process sentinels unchanged, and split the implementation into separately reviewable changes.

Keep the integration transparent to eval authors and preserve the existing eval and task interfaces. "Inspect client for the sidecar" describes its responsibility, without prescribing a public type or a choice users must make between local and remote execution. Its internal names, structure and any necessary deployment configuration are for Meridian to agree.

- The probe, `register_task` and the client's one setting, `off` (the default), `auto` or `required`, at task init ([sidecar.md, section 9](sidecar.md#turning-the-client-on)). An answer with `fixed` true, for a task in the tasks folder, is a success.
- `complete_registration` before each sample's first epoch, with the sample's input and metadata. The sidecar's side, with its `strictness` setting, is built ahead of the client.
- Not in step 11: the exemption for model calls that aren't the agent's ([sidecar.md, "Deferred"](sidecar.md#deferred)). It follows as its own change, after the client, once what Inspect signs and how the key reaches both sides are settled. It is off by default, so nothing in step 11 depends on it.
- Any subset of a registered task's samples is valid. An eval that runs only some of the samples in the tasks folder, chosen with `--sample-id`, `--limit` or any other selection, runs them as usual.
- `x-inspect-context` and `x-irid` on each model request, only after the probe found a sentinel.
- `request_result` after every agent call that went to a sidecar, a reply or an error alike, by the call's `x-irid`, with `wait`, asking again while the answer is `judging`. A request the sidecar says it couldn't judge is recorded as not judged. One whose judgment is lost, unknown or can't be looked up is recorded as unavailable, and never as allowed.
- Acting on the judgment as Inspect does in process: a call taken out goes back into Inspect's conversation redacted, with an approval error, a changed call is held as it ran, a `terminate` ends the sample, and every step is written into the transcript.
- The eval's token: kept from `register_task`, presented in every later call for the eval, and never put into a model request, a sandbox or the log.
- The sidecar's side, in `inspect_sentinel`: no field in any reply, refusal or stream event, for any client, and none placed by any proxy hook; `request_result`'s `judgment`, without the top-level `withheld` and `proposed`; steps without `original`, and each call a decision acted on redacted in `withheld` and `proposed`, with the model's original call put back in place of each redacted call on later requests; `request_result`'s `wait`; and the eval's token, returned by `register_task` and required by every other method, with the error `unauthorized` (section 9).
- `sample_result` before scoring each sample, and `end_sample` after.
- Prevent Inspect's completion-response cache from bypassing the sidecar while the client is active. Initially disable that cache, or arrange fresh sidecar judging of a cached completion before the agent can use it. A decision made for a cached completion in an earlier eval is not a decision for the current sample. This is separate from provider prompt caching, which still sends the model request through the proxy.
- Tests: in inspect_ai's own suite, with a stand-in sidecar; and the conformance suite in `sidecar.md` section 11, comparing Inspect's transcript in process and through a sidecar with whole-response holding. Include a pre-populated completion cache: a cached tool call cannot execute without the sidecar judging it for the current sample, and a decision cached by an earlier eval cannot authorize it.

## Step 12: dropped

Inspect incremental streaming is not being done. An eval's agent acts only on a whole turn, so Inspect goes on refusing a sidecar set to `release: early` ([sidecar.md, "Streams"](sidecar.md#streams)).

## Step 13: human review

Repositories: `inspect_sentinel` and `inspect_ai`. Depends on: steps 10 and 11, including their Envoy and LiteLLM prerequisites. This is the sidecar part of [remote human channels](workstreams.md#13-remote-human-channels).

Step 13 is built in two parts:

1. **First, the feature, without reviewer credentials.** Inspect sends the person's answer on the eval's key, the secret it signs its calls to the sidecar with ("the eval's token" in [sidecar.md, section 9](sidecar.md#the-evals-token)), and the sidecar accepts it only for a question from one of that eval's own runs. This gets human review working end to end, so that its experience in Inspect's approval UI can be developed and tried.
2. **Then, reviewer credentials.** This is the direction, and a TODO. The answer is signed by the reviewer, not by Inspect's process, and an eval's key can no longer answer. It needs a reviewer identity the sidecar can check, which comes with [step 16](#step-16-tls-on-the-sidecars-listeners)'s client certificates and [#189](https://github.com/meridianlabs-ai/inspect_sentinel/issues/189)'s control of who may call the endpoint for evals.

The first part is a temporary exception to the rule that an eval's credential never answers a review question ([sidecar.md, "A client that can't be trusted"](sidecar.md#a-client-that-cant-be-trusted)). It gives a client that has taken over an eval's process nothing it can't already do: that client can run the eval's tools directly, and answer an in-process sentinel's questions itself. It can't answer another eval's questions, or those of an agent without an eval.

The first part has a second temporary exception, and three limits ([sidecar.md, "Questions for a person"](sidecar.md#questions-for-a-person)):

- **A question hands Inspect the call's arguments** before anything is decided, an exception to the rule that a rejected call's arguments are never returned to anyone ([sidecar.md, "Redacted calls"](sidecar.md#redacted-calls)). It ends when a reviewer is shown questions on a channel outside the eval's process. Reviewer credentials alone don't end it.
- **An eval that isn't sent results is not asked.** With `--no-eval-results`, no question is kept and the step fails closed. Reading results and answering reviews are to become separate permissions.
- **A review isn't kept across a sidecar restart**, and an agent without an eval is not asked.
- **A turn's calls are asked about one at a time.** Showing them together waits on the grouping questions below.

- Settle the [human-review questions](sidecar.md#human-review-questions-for-step-13) before implementing the queue schema and channel integration: reviewer identity, context and automated assessments, call grouping and decision scope, pending age and sorting, approval validity, ownership and lifetime, reason disclosure, and audit completeness and durability.
- A sidecar implementation of `Host.ask_human` that publishes a question to a review queue and waits for the person's `HumanAnswer`, while the shared handler holds the step. Each question names its run, conversation, exchange and step, and carries the permitted choices, tool view, relevant context and escalations. Built for all four stages: Inspect shows a whole turn as one view after a generate, and the request before one.
- Define the queue's discovery and answer protocol, with a schema, access controls and timestamps for showing pending age. Specify how several questions are listed, grouped and sorted, and how scores, evidence and commentary from preceding monitors and protocols are shown. External review tools consume that interface, including for agents running without an Inspect eval; they answer once reviewers have credentials, in the second part. Reuse existing review channels rather than building a new review application as part of the sidecar.
- Inspect's sidecar client discovers pending questions while its model call is paused, shows them through Inspect's existing approval UI, and delivers the person's answer. In the first part, the answer goes on the eval's key, and is accepted only for that eval's own questions. In the second, an answer requires the reviewer's own credential, the agent's or eval's credential cannot approve its calls, and the sidecar enforces which runs and questions a reviewer may see and answer.
- Validate an answer against the choices offered for that question and the step's legal actions. Present `terminate` wherever it is offered, distinct from rejecting one call or an explicit operation on a group. Before a call, the default choices are approve, reject and terminate; rejecting one call does not reject its siblings. Resolve each question once: retries of the same answer are idempotent, and conflicting, expired or cancelled answers cannot release a call or revive an ended run. The answer returns to the existing sentinel runner and compositions as an ordinary `HumanAnswer`.
- No connection waits for a person, since a review may take days. The sidecar answers the model call at once with "pending review", keeps the paused judging and, at `tool_call`, the provider's reply, and goes on when the answer comes. The call is an operation, bound to its request, the API format included: a request sent again reattaches to it, is never reviewed afresh, and never causes a call to the provider by reattaching. The provider is called again only after an exchange with it failed. At `tool_call`, Inspect gets the judged reply by sending the call's request again, under the same `x-irid`, to `/sentinel/v1/judged-reply`, and reads it as the provider's own. At `tool_result`, it sends the request again once the request stage is finished. The sidecar keeps the completed reply of every such call, so that one lost on its way to Inspect can still be read. The eval can cancel the call at any state before its end, and nothing that arrives later opens it again ([sidecar.md, "A review pauses the model call, not a connection"](sidecar.md#a-review-pauses-the-model-call-not-a-connection)). An agent without an eval, which needs its resent request answered from the judged reply by its proxy, is in the second part.
- While a person is deciding, that sample waits, inside its model call, as it does for Inspect's own approval, and the eval's other samples carry on. Specify question expiry, unanswered questions, run termination, client disconnects and sidecar restart, including how pending questions are cancelled or recovered. No missing answer implicitly approves a call; where nobody can be asked, retain the host's cautious fallback.
- Record the question, who answered, the answer and any reason (in the first part, that the answer came from the eval; in the second, the reviewer's identity), with the final sentinel decision, so Inspect's transcript and the sidecar's request and run results retain what happened. A reviewer-only explanation stays separate from a rejection's agent-facing message, as in the existing contract.
- A runnable example and integration tests through Envoy and LiteLLM, with Inspect's approval channel for an eval, and, in the second part, an external review consumer for an agent without an eval. Keep the queue and host independent of the proxy; the Middleman integration uses the same review path when available.

Done when a tool call stays held while its review question is visible, approval releases it once if the final sentinel decision permits it, rejection or termination is enforced with the expected model history and records, and unrelated streams can still progress. Acceptance tests cover multiple waiting calls, per-call answers within a group, terminate as a distinct option, pending age and queue ordering, assessment provenance, retries, conflicting and stale answers, reviewer access, unanswered questions, disconnects, restart and a run ending during review. A review must outlast an idle timeout between the client and the proxy at both tool stages, streamed and not: an `AfterToolCall` question on a held request, and a `BeforeToolCall` question on a held reply, where a retry must receive the reply that was held. Test the audit guarantees chosen in the design questions, including recording failures. In the first part, an eval's key must demonstrably be unable to answer another eval's question. In the second, an eval credential must demonstrably be unable to answer any review question.

## Step 14: smaller asks of inspect_ai

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

## Step 15: trusted run identity and termination scopes

Repositories: `inspect_sentinel` and `inspect_ai`. Depends on: steps 6 and 11. Tracked in [#117](https://github.com/meridianlabs-ai/inspect_sentinel/issues/117). Built and tested through Envoy and LiteLLM, with no dependency on Hawk or Middleman.

A sentinel can end a run, and the sidecar then refuses every later model request of that run. The sidecar learns a request's run from `x-inspect-context`. An agent that sends its own model requests through the proxy could otherwise change that header: take a new `sample_uuid` or a higher `attempt_number` to escape the `terminate`, or name another sample to add rejections to its run. This PR makes Inspect prove the header's values with the eval's token ([sidecar.md, section 7](sidecar.md#7-which-run-a-request-belongs-to), and [section 9, "The eval's token"](sidecar.md#the-evals-token)).

- The eval's token: Inspect makes it with `secrets.token_urlsafe(32)` for each task and sidecar, and it never changes. It travels only in a `register_task` to a sidecar that holds no token for the eval. The sidecar never sends it back and never makes one, and refuses a token shorter than 32 characters.
- Calls to the endpoint for evals carry `eval_id`, `timestamp` and an HMAC-SHA256 `mac` over `[method, eval_id, timestamp]`, in place of the token. The `mac` covers no other params, so a captured one is a short-lived bearer credential for that method and eval. `register_task` carries exactly one of `token`, or `timestamp` and `mac`. A `mac` registration to a sidecar that holds no token is answered `unknown_task`, and Inspect sends it again with the token.
- `x-inspect-context` gains `timestamp` and a `mac` over `["model-request", task, sample_id, epoch, sample_uuid, attempt, attempt_number, x_irid, timestamp]`, where `x_irid` is the value of the request's `x-irid` header. The sidecar checks it before the request can change any run, and refuses a request with `x-inspect-context` and no `x-irid` as unreadable. Nothing is kept under a request's `x-irid` until its header has passed the check, so a request sent with another request's `x-irid` doesn't change what `request_result` answers for it. A request is kept under its eval and its `x-irid` together, so another eval's request with the same `x-irid` never joins its history. A run belongs to the eval that started it, so a request signed by another eval, such as one a client registered itself, is refused. Inspect's httpx request hook makes a fresh `timestamp` and `mac` for each HTTP try, so retries after a long backoff are accepted.
- The sidecar's window for `timestamp`: 5 minutes, set with `--timestamp-window` (`timestamp_window` in the `server` stanza). Outside it, an eval call is refused `unauthorized`, and a model request is refused and not judged.
- A model request naming an eval whose token the sidecar doesn't hold, as after a restart, is refused and not judged under every `strictness`, and `request_result` answers `unknown` for it. Under `required`, Inspect fails the attempt, a sandbox-bridged agent's too, and marks its monitor state as lost. Under `auto`, the call's error stands, so a bridged agent may go on with monitor state that began after the restart. Recovery is Inspect's existing retry of a sample, which needs `retry_on_error` to allow another attempt. A request without `x-inspect-context` is judged as before, so `registered_only` is how a deployment stops a client that strips the header.
- Keeping the token out of logs: a filter on the Anthropic SDK's `anthropic._base_client` logger in Inspect, the token taken out of any error a call to the endpoint raises, and operator guidance not to capture request bodies on `/sentinel/` in the proxy.
- Tests: changing any signed value after signing is refused, and changes nothing about the run; another eval's token or sample is refused, including a sample named from an eval the client registered itself; a header without `timestamp` or `mac` is unreadable, and so is one without `x-irid`; a `mac` made for one `x-irid` is refused with another, of the same sample or another; a request refused unchecked, unreadable, or without `x-inspect-context`, sent with another request's `x-irid`, leaves that request's `request_result` as it was; the window's edges, with a settable clock; a stripped header under `registered_only`; a restart under each `strictness`, then a registration with the same token; every registration case; each other method with a wrong `mac`, a stale `timestamp`, or no token held; the window setting from the command line and the configuration file. In inspect_ai: the token sent only when the sidecar holds none; no model request carrying the token; a model request's `mac` covering its `x-irid`; fresh `mac`s on Inspect's retries and the SDK's; a restart's `unknown` failing a `required` attempt, its own call's and a bridged agent's, before and after another sample registers the eval again, and leaving its monitor state uncopied; a bridged agent under `auto` handed the error; no token in any debug log record; and no change for a model with no sidecar.

Not in this PR:

- **Who may call the endpoint for evals through the proxy** ([#189](https://github.com/meridianlabs-ai/inspect_sentinel/issues/189)). A caller that can reach `/sentinel/` can register an eval of its own, with a token it chose, and resend a conversation a sentinel ended under that eval.
- **Encrypting the token in a registration** ([#190](https://github.com/meridianlabs-ai/inspect_sentinel/issues/190)), with a public key the probe offers. It needs a dependency on `cryptography` in both repositories.
- **Keeping termination decisions and monitor state across a restart.** That is [shared and durable state](#follow-up-shared-and-durable-state).

## Step 16: TLS on the sidecar's listeners

Repository: `inspect_sentinel`. Depends on: steps 3, 4, 6 and 9. Scheduled before step 17, so that `MiddlemanHook` supports it from its first version. It does not depend on step 15 or on Hawk.

A proxy reaches the sidecar over plain HTTP, and Envoy reaches the Envoy shim over plain gRPC. Each is protected by where the sidecar listens: an operator makes it reachable by the proxy alone. This PR lets the sidecar serve TLS on every port it listens on, and require a client certificate from the proxies that call it. Plain listeners stay the default, for a sidecar that runs beside its proxy.

- The sidecar: a `tls` stanza inside `server`, with `cert`, `key` and `client_ca`, and the options `--tls-cert`, `--tls-key` and `--tls-client-ca`. The certificate is served on every port: the endpoints' port, the Envoy shim's port, and the endpoint for evals' own port where it has one. `client_ca` requires a client certificate signed by that CA on the ports proxies call. The endpoint for evals' own port keeps its token, since Inspect presents no client certificate.
- Envoy: the configuration generator takes the CA that signed the sidecar's certificate, and Envoy's client certificate and key, as files in Envoy's file system. It puts TLS on both of Envoy's clusters for the sidecar, checking the sidecar's host name, with `h2` for the shim's gRPC.
- LiteLLM's guardrail: an `https` URL for the sidecar, the CA to trust, and a client certificate and key.
- Certificates are read at startup, so a new certificate needs a restart. Reloading them without one, such as through Envoy's SDS, is not planned.
- Tests: a CA and certificates the tests make; Envoy and the LiteLLM guardrail calling the sidecar with mutual TLS; a client with no certificate, or one signed by another CA, refused; the endpoint for evals reached with its token over TLS.

## Step 17: Middleman

Repositories: `inspect_sentinel` and `hawk`. Technical prerequisites: steps 3 and 5, and Hawk/METR accepting Middleman's passthrough hook. Scheduled after steps 15 and 16; it does not hold up trusted run identity or the other proxy integrations.

- In `inspect_sentinel`: `MiddlemanHook`, a package under `src/proxies/middleman`, calling `judge_request` and `judge_reply`, over TLS with a client certificate where step 16's settings are given, passing on and removing Inspect's two headers, and putting the sidecar's refusals into Middleman's errors.
- In `hawk`: the passthrough hook in Middleman, which is METR's to accept; and routing `/sentinel/*` to the sidecar at the load balancer in front of Middleman.
- A README for the hook, including the load balancer's routing.

## Step 18: typed fields for what a sentinel did

Repository: `inspect_ai`, with ts-mono's generated types and then `inspect_scout`'s schema. Depends on: nothing. This is the first step towards the measures in [sidecar.md, section 12](sidecar.md#12-measuring-what-a-sentinel-did): coverage, review latency and escalation rate. It takes in [#167](https://github.com/meridianlabs-ai/inspect_sentinel/issues/167).

The measures need times, outcomes and totals that the log has no field for. This PR adds the fields, as one data model, and fills none of them but #167's. The model is described once, in [sidecar.md, "One data model, filled from two places"](sidecar.md#one-data-model-filled-from-two-places). The steps after this one are the code that fills it or reads it: [step 20](#step-20-a-sentinel-inside-inspect-fills-the-fields) fills it inside Inspect, steps [21](#step-21-times-and-counts-in-the-sidecars-results) and [22](#step-22-inspects-sidecar-client-fills-the-fields) through a sidecar, and steps 19 and 23 read only the shared shape.

- `SentinelEvent` gains `completed`, `working_time`, the `unjudged` status with its `reason`, and `sidecar`, a `SentinelSidecar`.
- `EvalSample` gains `sentinel_stats`, a `SentinelStats` for each sentinel by name, whose `sidecar` is a `SentinelSidecarStats`. `EvalSampleSummary` gains it too, as it has the sample's `model_usage`, `total_time` and `limit`, so the list of samples and `samples_df` read it without loading the sample. `EvalStats` gains the same, added up.
- `EvalConfig` gains `sentinel_proxy`, from #167.
- The names are Meridian's suggestions, and this PR settles them. #167's provenance moves into `SentinelSidecar`: nothing but its writer read `metadata.sentinel_sidecar`, so older logs need no second reader.
- Every field is optional and absent unless a sentinel ran. An older log reads them as `None`, and an older reader ignores them. Each is marked experimental, as `SentinelEvent` is.
- Landing, as #167 says, follows inspect_ai's `land-ts-mono` skill: regenerate inspect_ai's OpenAPI schema and ts-mono's generated types, land ts-mono and then inspect_ai, then regenerate `inspect_scout`'s schema, whose `SentinelEvent` changes, and land that.
- Tests: each new field round-trips; a sample's summary has the sample's `sentinel_stats`; a log written before this step reads with them `None`; an `unjudged` event needs a `reason`, is at the empty path, and records no report, an explanation and an audit included; a sidecar's question ids are on the step's root event alone; a trimmed report keeps an empty `function`; #167's tests, as that issue lists them.

## Step 19: the log viewer shows the fields

Repository: ts-mono. Depends on: step 18. It can land before or alongside step 23.

Step 18 puts the fields into the log, but without this step they show only as raw data. This PR shows them in Inspect's log viewer, so they can be seen without step 23's reader. It reads only the shared shape, so one view serves a sentinel inside Inspect and a sidecar.

- **In the transcript**, the viewer already groups a step's sentinel events into one check (`buildSentinelSteps` in `transcript/transform/toolSentinels.ts`, shown by `SentinelInset` and `SentinelEventView`). That grouping learns the `unjudged` root event, as it reads `bypassed` and `error` events today, and shows the step as not judged, with the reason. Each check shows its `working_time` in its title, as `ToolEventView` does, and its `completed`. A sidecar's report shows its call's `x-irid`, and a trimmed one says so.
- **For a sample**, its `sentinel_stats`, in the sample's Info tab, under its usage, as the eval's are under the eval's usage. Not in the Activity tab, which a large sample, whose events are loaded in chunks, doesn't have. For each sentinel it shows the steps that reached it, judged and unjudged, the decisions by action, and the questions by how they ended. It shows that the stats are incomplete and why, where they are. For an entry with a `sidecar` part, it shows the proxy, the model calls and unknown calls, and whether the sidecar withheld its results. The stats come from the sample's summary, so the list of samples can show a column for them without loading each sample.
- **For the eval**, `EvalStats`' `sentinel_stats`, with the log's other stats, and the number of samples it left out.
- Tests, as ts-mono tests its other event cards and panels.

## Step 20: a sentinel inside Inspect fills the fields

Repository: `inspect_ai`. Depends on: step 18.

A sentinel inside Inspect knows every step it judged, and when, as it judges. This PR is the code that fills step 18's data model inside Inspect. It fills the shared core, and leaves both `sidecar` sub-objects `None`.

- Inspect's dispatcher (`_sentinel/_dispatch.py`) times each monitor and protocol on a step, and writes `timestamp`, `completed` and `working_time` on its event. Monitors run at once by default, so each is timed around its own function, not from the gaps between events. `working_time` is clock time less the sample's waiting time, as on a `ToolEvent`. The step's judging as a whole already has a start and an end: it runs in a span of type `sentinel`.
- Decide, with the PR, whether an event is written `pending` when its function starts and updated when it reports, as a `ModelEvent` is. That would show a monitor still running, and a `human()` question still open, in a live log. Otherwise the event is written once, when the function reports, with `timestamp` set back to when it started, and the viewer's grouping is unchanged.
- A step whose sentinel raised has the root's `error` event, as today, and is counted as unjudged, with the reason `error`. A step cancelled before its sentinel decided gets an `unjudged` root event, with the reason `cancelled`. A stage no sentinel watches is skipped, and doesn't count as a step that reached the sentinel.
- The dispatcher counts the sample's steps, decisions and `human()` questions as it goes. Inspect writes them into the sample's `sentinel_stats`, under the sentinel's name, in the sample's teardown, before the sample is logged: the block in `_eval/task/run.py` that every final attempt reaches, cancelled samples included ([step 22](#step-22-inspects-sidecar-client-fills-the-fields) names it). It fills `steps`, `actions` and `human_reviews`. The counts are Inspect's own, so they are `complete` wherever the dispatcher counted every step. Where it couldn't, such as a sample cancelled in the middle of a step, `complete` is false and `incomplete_reason` says why.
- How long a person took to answer `human()` is then in the log: `human()`'s event has when it began, which is when the person was asked, and when it reported. Nothing more is needed for it ([workstreams.md, 12](workstreams.md#12-monitoring-the-monitors)).
- Tests: a monitor's times lie inside its step's span, with two monitors run at once given durations of their own; a monitor's `working_time` leaves out the sample's waiting time, as a tool event's does; a `human()` that waits for a person has the wait between `timestamp` and `completed`; a sentinel that raises counts its step as unjudged, and a cancelled step gets an `unjudged` event; a sample cancelled by a time limit has its summary; the summary of a sample with a rejected call, a changed call and a question for a person.

## Step 21: times and counts in the sidecar's results

Repository: `inspect_sentinel`. Depends on: steps 6 and 13.

The sidecar records what each monitor and protocol reported, and not when. It answers about one call at a time, and gives no totals. This PR adds both to what the endpoint for evals returns. With step 22 it is the code that fills step 18's data model through a sidecar: this step sends the values, and [step 22](#step-22-inspects-sidecar-client-fills-the-fields) writes them into the fields. Each addition is a new optional member, so an Inspect that doesn't know it ignores it. It doesn't need step 18 to land first.

- Settle the definitions first, in `sidecar.md`, section 12: the unit (a model call or a tool call, at one stage, where a model call is one `x-irid`), what counts as judged, and what each time is measured from. The members below follow from them.
- Each record gains the time it was made, in seconds since the epoch as the times of step 13's questions are, and how long its monitor or protocol took.
- `request_result` gains when the sidecar was handed the call's request, and when it was handed its reply.
- `request_result` gains the outcome of each stage the call reached: judged, or not judged with the reason. The call's `state` stays as it is, so a call whose request was judged before its provider failed is still `not_judged`, and now says that `model_input` was judged.
- `sample_result` gains totals for the sample, by stage: steps seen, steps judged, and steps not judged, by reason. It gains the sample's decisions, by action, counted over every call the sidecar judged, the attempts Inspect retried included. It also gains the sample's questions for a person: how many were asked, counted as the distinct question `id`s in the calls' `reviews`, and how many of those ended at each `resolution`, which is `answered`, `timed_out` or `cancelled`, with `cancelled` by its `cause`. A question is counted by its `resolution`, and not by whether it has an `answered` time, since an answer taken just before its call closed took no effect. A question that is still open has no `resolution`, and is counted only as asked.
- With `--no-eval-results`, none of this PR's additions are sent. `sample_result` is off and records aren't sent, so neither the totals nor the records' times are. Each stage's outcome and the two hand-over times are left out of `request_result`, because the flag sends an eval what Inspect acts on and no more ([sidecar.md, "A client that can't be trusted"](sidecar.md#a-client-that-cant-be-trusted)). `judging_time` and `judging_usage` are still sent, as they are today, with the call's `state`, `reason` and `records`. Separately, no question is kept under the flag, so a sample has none, whoever counts them.
- The probe's answer, `GET /sentinel/info`, gains `sentinel`: the name of the sentinel the sidecar was started with, the instance name of its root. Inspect keys the sample's stats by it. With `--no-eval-results` it is left out, and Inspect keys by the proxy's address.
- The probe's answer also gains `results_withheld`: `true` under `--no-eval-results`, and `false` without it ([#242](https://github.com/meridianlabs-ai/inspect_sentinel/issues/242) names both). It states the mode, so Inspect doesn't infer it from what is missing. `review` can't stand in for it, since a sidecar from before step 13 sends no `review` either ([sidecar.md, "Finding the sidecar"](sidecar.md#finding-the-sidecar)).
- Each quantity, the field it is counted from and what the flag does to it are in [sidecar.md, "What each quantity is counted from"](sidecar.md#what-each-quantity-is-counted-from).
- The schema file, and the reference in `sidecar.md`, "The methods".
- Tests: a record's time lies between the hand-over of its exchange and the call's decision; a record made before a person was asked is earlier than the time the question was asked; the totals of a sample with a judged call, a call that ran out of time and a call a person was asked about; the decisions by action of a sample with a rejected call and a changed call, an attempt Inspect retried included; a request judged before its provider failed counts as judged at `model_input`; a call the SDK sent again counts once, and a call Inspect retried under a new `x-irid` counts as two; a call a person was asked about, sent again under its `x-irid`, counts once, with one question; a sample's questions with one answered, one that timed out and one the eval cancelled are counted by `resolution` and `cause`; with `--no-eval-results`, `request_result` has none of the new members, and `judging_time` as before; the probe says `results_withheld` `true` with the flag and `false` without it, and gives the sentinel's name only without it.

Not in this PR:

- **Totals for an agent with no eval.** It has no sample to ask about. They wait for the record log ([step 24](#step-24-the-record-log)).
- **Anything kept across a restart.** That is [shared and durable state](#follow-up-shared-and-durable-state).

## Step 22: Inspect's sidecar client fills the fields

Repository: `inspect_ai`. Depends on: steps 11, 18 and 21. Fixes [#229](https://github.com/meridianlabs-ai/inspect_sentinel/issues/229).

Inspect's sidecar client writes each of the sidecar's records into the eval's log as a `SentinelEvent`, with the time Inspect wrote it. That is after the call's judging ended, and after a person's answer where step 13 asked one. This PR writes step 21's times, outcomes and totals into step 18's data model, and adds the counts that only Inspect has. The shared core holds the same things a sentinel inside Inspect writes in step 20, and what only a sidecar has goes in the two `sidecar` sub-objects.

- Each event's `timestamp`, `completed` and `working_time` are its record's, from step 21. Its `sidecar` sub-object has the call's `x-irid`, its two hand-over times, and whether the report was trimmed, as the judgment says, which moves there from `metadata.sentinel_sidecar`. The event's `working_start` stays the sample's working time when Inspect wrote it, since the sidecar's clock knows nothing of it.
- The step's root event has a typed record of each question the step asked a person, in `sidecar.reviews`: its `id`, when it was `asked` and `answered`, the `decision`, its `resolution` and `cause`. A call that ended not judged has them on its `unjudged` event. They take the place of the info events `log_reviews` writes today, which carry the same record as a dict.
- A stage the call reached and the sidecar didn't judge, from step 21's outcomes, gets an `unjudged` root event with the sidecar's reason. So a call whose request was judged before its provider failed gets an `unjudged` event at `model_output`. Its `model_input` reports are in the log only if the decision below writes the records of a call that ended not judged, since such a call has no judgment, and its records aren't written today. Either way the sample's `steps` count `model_input` as judged, from the sidecar's totals.
- A sidecar that sends no times or outcomes, because it is older, leaves the events as they are today.
- The sample's `sentinel_stats` has an entry for each sidecar the sample's final attempt called, keyed by the sentinel's name from the probe, or by the proxy's address where the probe gives none. Its `steps`, `actions` and `human_reviews` come from the totals of the last `sample_result` that answered. Inspect asks for it at each copy of monitor state: before each scoring, which `react` also does during the sample, and when it saves a failed sample ([sidecar.md, "The end of a sample"](sidecar.md#the-end-of-a-sample)). So the last answer can predate the sample's last calls, and `complete` says whether the counts are the sample's whole count:
  - `complete` is true where the latest copy answered, and no call was sent through the sidecar, or was still waiting on its result, after that copy began.
  - It is false, with the counts kept up to that copy, where the latest copy failed after an earlier one answered, or calls went through the sidecar after the copy that answered. That happens when `react` scores, the agent carries on, and the sample is then cancelled by a time limit. `incomplete_reason` says which.
  - It is false, with `steps` `None`, where no copy answered. The sample was cancelled before any scoring, and a cancelled sample's teardown asks no sidecar for its results (`end_sidecar_sample` in `inspect_ai._sentinel._sidecar._call`); every copy failed; or the sidecar withheld its results. `incomplete_reason` says which.
- Inspect's client has to start tracking calls for this. Today `called_through` in `inspect_ai._sentinel._sidecar._task` keeps one entry for each sample and proxy, which each later call overwrites: no count, no times, and nothing about a call still waiting. This PR adds, for each sample and sidecar, a count of the calls sent, which of them still wait on a result, and when each copy began. `complete` and `model_calls` both come from it.
- The entry's `sidecar` part has the proxy's address, `model_calls`, `unknown_calls` and `results_withheld`. `model_calls` is Inspect's own count of the calls it sent through the sidecar, so a cancelled sample has it too. `unknown_calls` counts the calls the sidecar answered `unknown` or `dropped` for. Such a call is never counted as one that went round the sidecar, because the sidecar gives the same answer for a call it refused before keeping anything, and for one it has forgotten.
- `results_withheld` is as the probe said, and `None` for a sidecar whose probe doesn't say, because it is older. Inspect doesn't infer it from `sample_result`, which a cancelled sample's teardown never asks for, nor from the trimmed events, since a call that ended not judged has no judgment, and so no event.
- The stats are written in the sample's teardown in `_eval/task/run.py`, once `_attempt_outcome` says the attempt won't be retried, and before `log_sample` writes the sample. That is the `if not isinstance(outcome, _AttemptRetry)` block, which every final attempt reaches, whether it completed, failed or was cancelled. The call to `save_sidecar_monitor_state` inside it is not the place: it runs only for a sample that wasn't scored. Nor is the `end_sample` call, since a cancelled sample is not ended ([sidecar.md, "The end of a sample"](sidecar.md#the-end-of-a-sample)), and `end_sidecar_sample` runs after the sample is logged.
- Decide, with the PR, whether the records of a call that ended not judged, and of an attempt Inspect retried, are written into the log. Today only their judge models' usage is recorded. So a count taken from the log misses steps that were judged, and a monitor's report on an attempt that failed is not in the log. The sidecar returns such records in `recorded`, outside any judgment.
- An eval with no sidecar is unchanged.
- Tests: with a person who takes a while, an event's times are its record's and not the time of the lookup; a sidecar that sends no times; a call whose request was judged before its provider failed has an `unjudged` event at `model_output`, counts as judged at `model_input` in the sample's `steps`, and has its `model_input` reports where the decision above writes them; a sample's stats with a call the sidecar never saw and a generate Inspect retried, counted in `model_calls`; a scored sample has its stats, though it never reaches `save_sidecar_monitor_state`; a sample's `actions` include a decision on an attempt Inspect retried; a trimmed report has `trimmed` set; a step that asked a person has the question's typed record on its root event, an answered one and a timed-out one, and a call that ended not judged has it on its `unjudged` event; a sample scored twice has the counts of its last scoring, `complete`; a sample `react` scored, which then made more calls and whose final copy failed, has the earlier counts, not `complete`; a sample `react` scored, which then made more calls and was cancelled by a time limit, has them, not `complete`; a sample cancelled by a time limit before any scoring has `steps` `None`, not `complete`, as cancelled, and still has its `model_calls`; a sample through a sidecar that withholds results has `results_withheld` true, where its only call ended not judged as well, and where the sample was cancelled by a time limit before scoring; a sidecar whose probe doesn't say has `results_withheld` `None`; a sidecar whose probe gives no sentinel name is keyed by its proxy's address, and one whose name a sentinel inside Inspect shares has the address added.

## Step 23: reading the measures from logs

Repository: `inspect_ai` for the columns, and `inspect_sentinel` for the summary. Depends on: steps 18, 20 and 22. This is the sidecar's part of [monitoring the monitors](workstreams.md#12-monitoring-the-monitors). It reads only step 18's shared shape, so it serves a sentinel inside Inspect and a sidecar alike.

Steps 20 and 22 put the times, outcomes and totals into each eval's log, and step 19 shows them for one log. Nothing adds them up across logs. This PR does, over as many logs as it is given. It builds on Inspect's data frames, `events_df`, `samples_df` and `evals_df` in `inspect_ai.analysis`, and doesn't read logs its own way.

- **Columns, in `inspect_ai.analysis`.** `SentinelEventColumns` for `events_df`, beside `ModelEventColumns` and `ToolEventColumns`: an event's factory, path, function, step, stage, kind, status, action, suspicion and reason, and its `sidecar` fields. `EventTiming` already reads its times. A column group for `samples_df` reads `sentinel_stats` from the sample's summary, and one for `evals_df` reads `EvalStats`' roll-up.
- **A table of steps, in `inspect_sentinel`**, made from `events_df`: a row for each step at each stage, with the eval, sample and call it belongs to, the monitors and protocols that reported, the decision, whether it was judged, its times, and any question for a person with how it ended. It groups a step's events as the viewer does (`buildSentinelSteps` in ts-mono). It reads step 18's fields, the same for both sources.
- At the model stages a row is one generate, and at the tool stages one tool call, for both sources. That is the step both write: a sentinel inside Inspect judges once for each generate, whatever was retried, and Inspect writes a sidecar's reports under the generate's id. A sidecar's row names the `x-irid` whose judgment it holds, from the event's `sidecar`.
- A summary of that table, for a period, grouped by protocol, stage or eval: coverage; the share of steps a sentinel rejected, changed, ended the run at, or asked a person about; and the median and the 95th percentile of the time a monitor took and of the time a person took.
- The figures follow the definitions in `sidecar.md`, section 12, and each is counted from the field its table names ([sidecar.md, "What each quantity is counted from"](sidecar.md#what-each-quantity-is-counted-from)). The summary says what it counted: the number of steps, and how many calls were not verifiable.
- Each sample's totals are read from its `sentinel_stats`. A sidecar's count is by `x-irid`, so the summary reports a sidecar's totals beside the table's figures, and never adds the two into one denominator.
- A sidecar's entry that isn't `complete` is reported with its `incomplete_reason`, such as a sample cancelled before scoring, or after `react` scored it and the agent carried on. Its rows are still in the table. The summary leaves it out of the sidecar's totals, says how many samples it left out and why, and doesn't make the totals up from the rows, which count generates and not `x-irid`s. Only `complete` stats are added up, as in `EvalStats`.
- A record the log doesn't hold is reported as missing, and never as a step that wasn't judged. Until step 22 decides whether the records of a retried or not-judged attempt are logged, the reader says where a sample's `sidecar.model_calls` is more than the calls the log has records for.
- A log written through a sidecar that withholds results, with `--no-eval-results`, has trimmed decisions and no totals. The reader knows it from the sample's stats, `sidecar.results_withheld`, and not from the trimmed events alone. For such a log it reports coverage and the monitors' durations as withheld. It doesn't work them out from the trimmed decisions, and it doesn't count the missing steps as not judged. The trimmed decisions are still rows of the table, marked as trimmed from the event's `sidecar.trimmed`.
- Results that were withheld are reported apart from data that is missing for another reason, such as a log from before step 18, a record step 22 left out, or a sidecar whose `results_withheld` is `None`. For such a sidecar the reader reports coverage as unknown: it doesn't work it out from the events, which may be trimmed, and doesn't call the results withheld.
- A page in the docs, with one worked example.
- Tests: `events_df` with `SentinelEventColumns` and `EventTiming` reads a sentinel event's fields and times, and `samples_df` reads a sample's `sentinel_stats` without loading the sample; for a run with no retries, logs written by a sentinel inside Inspect and by a sidecar give the same table, the times apart; for a generate Inspect retried, each gives one row, and the sample's `sidecar.model_calls` counts the retry; a log without the records of a retried attempt is reported as incomplete, with the sample's totals as the sidecar gave them; a sample cancelled before scoring gives its rows, and its totals as missing because it was cancelled; a sample whose stats aren't `complete` gives its rows, and is left out of the sidecar's totals with its reason; a log written through a sidecar that withholds results gives its decisions as trimmed rows, and its coverage and its monitors' durations as withheld, not as zero, a sample whose only call ended not judged included, and a sample cancelled by a time limit; a log through a sidecar whose `results_withheld` is `None` gives its coverage as unknown, apart from withheld; a log from before step 18 is read with its times and totals empty, not wrong.

Not in this PR:

- **A dashboard, or a store of results over time.** The table is a pandas data frame, as `events_df` returns, for whatever an operator already uses.
- **A person's later review of a decision**, which nothing records yet ([sidecar.md, section 12](sidecar.md#what-the-sidecar-doesnt-measure)).

## Step 24: the record log

Repository: `inspect_sentinel`. Depends on: step 21.

The sidecar keeps what a sentinel reported in memory, for a limited window, so that an eval can read it with `request_result`. An eval then writes it into the eval's own log. An agent with no eval has no log, so what a sentinel decided about its calls is kept nowhere. This PR adds the record log: the sidecar writes an entry to a log outside the sidecar as each thing happens ([sidecar.md, section 10](sidecar.md#10-operations)). The log is then the record. What the sidecar keeps in memory is a copy, for the endpoint for evals.

- Settle the design first, in `sidecar.md`, section 10. The next seven bullets are what the design has to answer. The options and the format of an entry follow from the answers.
- **What an entry holds.** Each record a monitor or protocol made, with its time and duration, under the names a `SentinelEvent` uses, so that step 23 reads an entry and an event with the same columns. Each stage's outcome for a call. Each question for a person, with how it ended. These come from steps 13 and 21. Each entry names its run. Where an eval made the call, the entry also names the eval, the sample and the call's `x-irid`.
- **Whether the log says which sentinel the sidecar runs, and holds the judge models' calls.** An eval's log has both for a sentinel inside Inspect, and neither for a sidecar ([sidecar.md, section 9](sidecar.md#the-sidecars-reports-in-inspects-transcript)).
- **Where entries are written.** The operator chooses the destination. So the sidecar writes through one interface, and each destination is an implementation of it. The first destination is chosen with the design. An entry leaves the sidecar when it is made. It doesn't wait for the call or the run to end.
- **How the log is kept from being rewritten.** The operator secures this: the sidecar is given a way to add entries, and no way to change or delete them. The docs say how to set that up for each destination this PR supports.
- **What happens to a call whose entry can't be written.** A turn that can't be judged is refused by default (section 10). The design says whether a turn that was judged, and can't be logged, is refused as well, and whether a deployment may choose otherwise.
- **Whether entries are numbered**, so that a reader of the log can tell that an entry is missing.
- **How an entry is matched to the proxy's own log.** A proxy can keep its own log of every request and reply. The record log doesn't copy that traffic. Each entry holds the proxy's id for the exchange, so that a reader can find the request and the reply in the proxy's log. LiteLLM's guardrail makes that id itself, and doesn't use LiteLLM's call id, because a caller can set LiteLLM's call id ([sidecar.md, section 1](sidecar.md#litellm)). So the design says what a reader matches on behind LiteLLM.
- The option that names the destination, in the sidecar's configuration file, and a page in the docs.
- Reading the record log into step 23's table, so that an agent with no eval gets the same measures as an eval, and so does a deployment that withholds results from its evals with `--no-eval-results`. Whichever of steps 23 and 24 lands second adds this. An agent with no eval is counted less exactly, because its requests carry no `x-irid` ([sidecar.md, section 12](sidecar.md#what-is-counted)).
- Tests: the entries for a call match what `request_result` answers for that call, with the same times; an agent with no eval has its entries in the log; a destination that can't be written to gives the behaviour the design chose; the sidecar's memory can drop a call's records and the log still holds them.

Not in this PR:

- **The proxy's own log of requests and replies.** It is the proxy's feature, and the operator turns it on. The docs page says to keep it, and how an entry is matched to it.
- **Reading the log back into the sidecar after a restart.** The log is a record. The sidecar's working state is [shared and durable state](#follow-up-shared-and-durable-state).

## Follow-up: shared and durable state

Repository: `inspect_sentinel`, with deployment configuration and Inspect lifecycle handling where needed. Deliver before running more than one sidecar worker or promising that a run's state survives a restart. The initial deployment remains one process, with no promise of restart continuity.

- Define and implement shared state and the coordination it needs: run termination, the sentinel's store and tallies, conversation amendments and judged results, registered facts, held exchanges, and request results must remain consistent when successive calls reach different workers.
- Define persistence, recovery and unavailable-state behavior before allowing a run to continue after restart. Losing state must not silently turn a previously ended run into a fresh one. Coordinate terminal-state retention and credential expiry with [step 15](#step-15-trusted-run-identity-and-termination-scopes).
- Tests: alternate a run's requests and replies between workers; check that termination, tallies and amendments survive the handoff and that overlapping duplicate results are judged once. Restart between exchanges and verify the chosen recovery or refusal behavior, including a run that was already terminated.

## Not planned yet

These need design first. Each names the repositories it would touch.

- A full policy for model calls that Inspect answers from its cache, which never reach the proxy. The initial remote-mode cache behavior must be agreed before step 11; supporting cached calls with equivalent remote judgment needs further design in `inspect_ai`, and possibly `inspect_sentinel`.
- Tallies kept at the task level. `inspect_sentinel`.
- A record that a person looked at a sentinel's decision after it was made: who, when, and what they concluded. `inspect_sentinel` and `inspect_ai`, and `inspect_scout` for a scanner's results.
- An identity for an agent that outlasts a run, so that its measures can be followed from run to run. `inspect_sentinel`, after step 15.
- A name for one generate that each of its model calls carries, so that the sidecar counts a generate Inspect retried as one. `inspect_ai` and `inspect_sentinel`.
