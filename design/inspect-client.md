# Inspect working with a sentinel sidecar

This is the design for [PR 11](sidecar-plan.md#pr-11-inspect-client-for-the-sidecar). It describes what Inspect does when its model calls go through a proxy that has a sentinel sidecar behind it.

It is written to be read from the top. Each section starts with the short version.

Each part is marked:

- **In the plan**: already agreed in [sidecar-plan.md](sidecar-plan.md) or [sidecar.md](sidecar.md).
- **Proposed**: suggested here, not yet agreed.
- **To decide**: an open question, listed at the end.

## The short version

Up to this point in the implementation, nothing in Inspect knows the sidecar is there. An eval that sends its model calls through one gets none of the in-process behaviour. A rejected turn reaches Inspect as an HTTP error, which fails the sample. The sentinel gets no facts about the task or the sample. And nothing it decides reaches the transcript.

When the sentinel runs inside Inspect, a rejection goes into the conversation and the agent carries on. Through a sidecar, Inspect can't do that yet.

This PR teaches Inspect to notice the sidecar and work with it. An eval author changes nothing. Their task, their agent and their command line stay the same.

## The pieces

![Inspect sends model traffic and eval-control traffic through one proxy. Model traffic goes to the provider and is checked by the sentinel sidecar. Eval-control traffic under /sentinel/ goes to the sidecar's eval API.](images/inspect-proxy-traffic.png)

- **Inspect** runs the eval: the task, the samples, the agent and its tools.
- **The proxy** sits between Inspect and the model provider. Envoy, LiteLLM and Middleman are examples.
- **The sidecar** runs the sentinel. The proxy hands it each request and each reply, and the sidecar says what may go on.
- **The provider** is the model API, such as Anthropic or OpenAI.

Inspect talks to the sidecar in two ways:

1. **Inside its model calls.** Inspect adds a header to the agent's model calls, saying which sample each is for. The sidecar adds a `sentinel` field to each reply.
2. **Through a small endpoint.** Inspect calls the sidecar directly under `/sentinel/`, through the proxy's own address. It uses this at the start of a task, at the end of a sample, and when a reply arrives without the `sentinel` field.

## 1. Finding the sidecar

**Short version:** at the start of each task, Inspect asks the proxy once whether a sidecar is there. If it isn't, nothing else in this document happens.

**In the plan.** Inspect sends `GET /sentinel/info`. A sidecar answers:

```json
{"service": "inspect_sentinel", "versions": [1], "release": "whole"}
```

Inspect goes on only if `versions` holds a version it knows.

**Proposed: one setting for the sidecar client.** It has three modes. Its name is for Meridian to choose.

| Mode | What Inspect does |
| --- | --- |
| `auto`, the default | Probes quietly, and uses a sidecar it finds. If the probe can't tell, Inspect carries on as today. |
| `required` | Fails the task at the start unless a sidecar Inspect supports answers. |
| `off` | Doesn't probe, and runs as today. |

What each probe answer means:

| Probe answer | Under `auto` | Under `required` |
| --- | --- | --- |
| a sidecar, with a version Inspect knows | use it | use it |
| a sidecar, with no version Inspect knows | fail the task | fail the task |
| a 404, or an answer that isn't a sidecar | no sidecar: run as today | fail the task |
| a 401 or 403, a 5xx, or a timeout | can't tell: run as today, with a debug line | fail the task |

`auto` keeps an ordinary eval exactly as it is today, but it can't guarantee that the sidecar is used. A deployment that relies on its sidecar sets `required`.

**Proposed: credentials.** The probe and every call under `/sentinel/` reuse the authentication configured for the model's endpoint, in whatever form Inspect uses for that model. The proxy checks them as it checks a model call.

**Not in PR 11: a separate port for the endpoint.** A sidecar can serve its endpoint for evals on a port of its own, with its own token. Inspect can't find that from the model's address, so PR 11 doesn't support it.

**Proposed: where the probe goes.** The probe goes to the root of the model's base URL, not on the end of it.

| Model base URL | Probe |
| --- | --- |
| `https://proxy.example/v1` | `https://proxy.example/sentinel/info` |
| `https://proxy.example` | `https://proxy.example/sentinel/info` |

Envoy and LiteLLM both forward `/sentinel/` at the root. An OpenAI base URL usually ends in `/v1`, so adding `/sentinel/info` to the end would miss the sidecar.

**Proposed: which endpoint, and how often.**

- Inspect probes the endpoint of the task's assigned model: the model the eval runs with, such as `--model`.
- It probes once per task.
- Other models are not probed and get none of Inspect's headers. Examples are a model given to a role, or a grader model a scorer calls.
- Nothing is shared between tasks in this PR.

More than one sidecar in a task is question J.

Models are shared between tasks inside Inspect. So the sidecar's state for a task lives with the task, not with the model.

## 2. Registering the task

**Short version:** the sentinel is given each task's facts: the task's name and description, plus each sample's input and metadata. A task gets registered in one of two ways:

1. **Dynamic registration, by Inspect.** Inspect sends the task's facts at the start of the task. This is the usual way.
2. **Pre-registration, by an admin.** The admin writes the task's facts into the sidecar's configuration before the eval runs. This is an advanced setup, described under "Pre-registration" below.

Inspect calls `register_task` either way. A pre-registered task runs only samples predefined by the admin, though the client can choose to run only a subset of them.

**In the plan.** Inspect calls `register_task` with:

- the eval's id
- the task's name, description and metadata
- each sample selected for this run, after `--limit`, `--sample-id` and the like: its id, input, description and metadata

The sidecar gives one of four answers.

| Answer | What it means | What Inspect does |
| --- | --- | --- |
| Registered, `fixed: false` | The sidecar stored the facts. | Starts the task. |
| Registered, `fixed: true` | An admin registered this task beforehand, in the sidecar's tasks folder. | Starts the task. |
| `registration_refused` | The sidecar runs only the tasks its admin registered, and this isn't one. | Fails the task before any sample starts, with an error that names the task and says the sidecar runs only the tasks its admin registered. The error is shown as the eval runs and written to the task's log. |
| `registrations_full` | The sidecar already keeps as many registrations as it takes. | Treats it as a failure, like an endpoint that can't be reached: the task fails, as below. |

**When registration fails.** The endpoint may be down, may not answer, or may answer `registrations_full`.

- **Once a supported sidecar has answered the probe, registration is required, in both `auto` and `required`.** The task fails before any sample starts, with an error saying the sidecar couldn't register it, and why.
- **A refusal fails the task too.** The deployment has said the task may not run there.

A policy that lets a task run without registering can be added later, if one is needed.

### Pre-registration

An admin can register tasks beforehand, in the sidecar's configuration. A task registered this way is called predefined. Inspect still calls `register_task` for it. The sidecar matches the call to the predefined task by the task's name, and keeps the admin's facts.

It is fine for Inspect to register a predefined task when:

- **it runs the admin's samples, or any subset of them.** If the admin registered 50 samples and the researcher runs samples 1 and 2, that works. The sidecar answers with `fixed: true`, which says the task was already registered, for information only. Inspect starts the task as usual.

It is not fine when:

- **it selects a sample the admin didn't register.** Today the sidecar refuses that sample's first model request. Question A below proposes failing such a sample before it starts.
- **it changes a sample's input or metadata, under a sample id the admin registered.** The sidecar doesn't detect this. The sentinel is given the admin's facts, not Inspect's. Catching it is not yet designed.

This is an advanced setup, described in sidecar.md section 9:

- [The tasks folder](sidecar.md#the-tasks-folder) shows how an admin writes the tasks.
- [Advanced: a sidecar that runs only the tasks in its folder](sidecar.md#advanced-a-sidecar-that-runs-only-the-tasks-in-its-folder) shows a sidecar that runs only predefined tasks.

**To decide:** what happens to a selected sample that isn't in the admin's file (question A below).

## 3. Each model call

**Short version:** after a successful probe, Inspect adds a header to the agent's model calls. The sidecar uses it to tell which sample, and which attempt, a call belongs to.

**Proposed: which calls get it.** `x-inspect-context` goes only on calls that are:

- **agent turns**, the calls a sentinel inside Inspect would judge. A scorer's, an approver's or a compaction summary's calls are not.
- **to the endpoint the probe found a sidecar at**, the assigned model's (section 1).

`x-irid` is unchanged: Inspect sends it on every model call, as it does today.

**In the plan.**

| Header | What it holds |
| --- | --- |
| `x-irid` | Inspect's id for this model call. Inspect already sends it today. |
| `x-inspect-context` | Which sample this call is for, as JSON. |

`x-inspect-context` looks like this:

```json
{
  "version": 1,
  "eval_id": "eval_789",
  "task": "data_exfiltration",
  "sample_id": 1,
  "epoch": 2,
  "sample_uuid": "sample_123",
  "attempt": "attempt_1"
}
```

- `sample_uuid` stays the same when Inspect retries a failed sample.
- `attempt` is new for each retry of the sample.

The proxy removes both headers before the request reaches the provider.

**Proposed: Inspect's completion cache is off.** While a sidecar is active, Inspect doesn't reuse cached model replies. A cached reply was judged for an earlier run, so it says nothing about this one. The plan also allows having the sidecar judge a cached reply afresh. Turning the cache off is simpler. The provider's own prompt caching is unaffected.

## 4. What comes back

**Short version:** every reply Inspect gets falls into one of four cases. Inspect handles each as it would if the sentinel ran inside Inspect.

**Case A: the reply goes on.** The reply carries `outcome: proceed`.

Some calls may have been taken out or changed. The field's `proposed` holds the model's turn as the model wrote it. Inspect then:

- keeps the model's own turn in the conversation
- answers each call that was taken out with an approval error, as in process
- runs each changed call with its new arguments, while the model's turn keeps the original

**Case B: the whole turn is rejected.** Every call in the turn was rejected, so the proxy sends an error, with the field and `outcome: reject`. The field's `withheld` holds the model's turn.

Inspect turns that error into an ordinary model output. The output holds the model's turn, and each call gets an approval error. The agent then goes on, as it would in process.

This is a change. Today, Inspect only sees an HTTP error. It retries only when the sidecar is set to answer with 503.

**Case C: the run is ended.** The field says `outcome: terminate`. Inspect ends the sample, as in process: the sample's limit is recorded as `operator`, and the sample is still scored.

**Case D: no field.** See the next section.

**In the plan: where the field is.** It depends on the provider format, and on whether the reply is streamed.

| Format | Not streamed | Streamed | On an error |
| --- | --- | --- | --- |
| Anthropic Messages | top level of the reply | the last `message_delta` event | either place, below |
| OpenAI Chat | top level of the reply | the last chunk | either place, below |
| OpenAI Responses | top level of the reply | `response.completed`, or `response.incomplete` when the reply was cut short | either place, below |

**On an error, Inspect checks two places**, because the proxy decides where the field goes:

- the top level, beside `error`, as Envoy sends it;
- `error.provider_specific_fields.sentinel`, as LiteLLM sends it.

Inspect has to read the field while it reads the stream. The provider SDKs drop it when they put the stream together.

**Proposed: the transcript.** Each monitor and protocol report in the field becomes a `SentinelEvent`, as it does in process. How each report is tied to Inspect's messages and calls is question E below.

## 5. When the field is missing

**Short version:** a reply without the field doesn't mean the call was allowed. Inspect asks the sidecar what happened.

Some ways the field goes missing:

- the proxy gave up waiting for the sidecar and sent its own error
- the sidecar couldn't be reached, and the proxy let the reply through unjudged
- the connection dropped before the reply ended

**In the plan.** Inspect calls `request_result` with the request's `x-irid`.

**Proposed: what Inspect does with each answer.** The model call came back either as a reply or as an error.

| Answer | What it means | With a reply | With an error |
| --- | --- | --- | --- |
| `decided` | the sidecar judged the call | acts on the decision, as in section 4 | a rejection or a `terminate`: acts on it, as in section 4. A `proceed`: the reply itself was lost on the way, so the call stays a failed call, and Inspect retries it as it does today |
| `judging` or `awaiting_reply` | it's still being judged | waits, then asks again (question D) | waits, then asks again (question D) |
| `not_judged`, with a reason | the sidecar saw the call but couldn't judge it | uses the reply, since the proxy chose to let it through, and records why it wasn't judged | fails the sample with that reason |
| `dropped` | the sidecar once held the call's records, and no longer keeps them. Judging may not have finished | `auto`: uses the reply, and records the judgment as unavailable. `required`: fails the sample | fails the sample |
| `unknown` | the sidecar never saw the call | as for `dropped` | fails the sample |
| no answer | `request_result` itself failed | as for `dropped` | fails the sample |

- **Ordinary retries stay.** Inspect still retries the provider and network failures it retries today. Once `request_result` confirms that the sidecar couldn't judge a call that came back as an error, the sample fails, with no extra retry on top.
- **None of these is permission to go on.** `dropped`, `unknown` and a failed lookup say nothing about what the sentinel decided. Under `auto`, the reply is used as it would be with no sidecar, and the transcript says the judgment is unavailable.

**Proposed: a fuller answer.** Today `request_result` gives the outcome and the records, but not the field's step-by-step decisions. It should give the same information as the field, so Inspect can act exactly as it would on the field. This is a change on the sidecar's side.

## 6. The end of a sample

**Short version:** before any scoring, Inspect copies the sentinel's store into the sample. After the final scoring, it tells the sidecar the sample is over.

**Scoring can happen during a sample, not only at its end.** For example, `react` with more than one attempt scores each answer the model submits, to decide whether to stop (`agent/_react.py`). In process, that scorer reads the sentinel's live store. Through a sidecar, the store has to be fetched first.

**Proposed.**

1. Before any scoring, Inspect calls `sample_result` and copies the store into the sample's own store, so scorers read the latest. Scoring happens in two places in Inspect, and both need it:
   - `score()` (`scorer/_score.py`), which an agent such as `react` calls during the sample;
   - the final scoring, where the task runner calls each scorer itself (`_eval/task/run.py`).
2. Inspect scores the sample.
3. After the final scoring, `end_sample`: the sidecar drops the sample's state.

Inspect skips `end_sample` when it will retry the sample, so that the retry keeps the sentinel's state.

**To decide:** what happens when `sample_result` fails (question D), and samples that have no result to fetch (question I). A failed fetch must never look like a sample in which the sentinel had nothing to say.

## 7. Streams

**Short version:** PR 11 works with a sidecar that holds whole replies. Early release waits for PR 12.

**Decided.** PR 11 is built and tested for `release: whole`, where the proxy holds the whole reply until it is judged. A whole reply can still travel as a stream: Inspect reads the stream, and the `sentinel` field is in its last event (section 4).

`release: early`, where text arrives while the rest is held, is [PR 12](sidecar-plan.md#pr-12-inspect-incremental-streaming). That covers a refusal arriving as an error event after some text, streams that are cut off, cancellation, and partial output shown while streaming.

**Proposed: a sidecar set to `early` before PR 12.** The probe answer says `"release": "early"`. Inspect stops the task with an error at the start, saying the sidecar releases replies early, which Inspect doesn't support yet.

Running as if there were no sidecar would be unsafe. Inspect's completion cache would come back on, and a cached tool call could run without the sidecar ever seeing it.

## Not built

**A response header.** If some proxy can't pass the `sentinel` field through, the sidecar could add a header saying the request was judged. No proxy we know of needs one, so none is designed.

## Questions to decide

The aim throughout is that a sentinel in a sidecar behaves as one inside Inspect does. Each question says what's going on and why it matters, then gives the options, with what each one gains and costs, and what we propose.

### A. What if a researcher picks a sample the admin didn't register?

- *What's going on:* an admin can pre-register a task by writing its samples into the sidecar's configuration (section 2). Say the admin registered samples 1 to 50, and a researcher runs sample 51.

- *Why it matters:* today the task starts as normal. Sample 51 fails only at its first model call, which the sidecar refuses because the sample isn't in the admin's file. By then the sample's setup has run, such as starting its sandbox, and the error looks like a failed model call rather than a sample that isn't allowed.

- *Option 1, leave it as it is:* nothing to build. But the researcher waits through the setup and gets a confusing error.

- *Option 2, refuse the whole registration:* a small change on the sidecar's side, and the error comes when the task starts. But one unknown sample stops every sample in the task, including the 50 that are allowed.

- *Option 3, list the unknown samples in the answer to `register_task`:* Inspect fails just those samples, before their setup, with a message saying the admin didn't register them, and the rest run. It needs a change on both sides: the sidecar reports the unknown ids, and Inspect acts on them.

- *Proposed:* option 3.

### B. Where should Inspect act on the sentinel's decisions?

- *What's going on:* inside Inspect, the sentinel judges each tool call just before it runs, after any approver, such as a human asked to allow each call. Through a sidecar, the whole turn is judged in the proxy, which takes out rejected calls and rewrites changed ones before Inspect sees the reply. The field's `proposed` holds the turn as the model wrote it.

- *Why it matters:* to look as it does inside Inspect, Inspect has to put the model's turn back, answer each rejected call with an approval error, and run each changed call with its new arguments. Where Inspect does this decides how close the result is to inside Inspect, and whether a rejected call can still run.

- *Option 1, act when the reply arrives:* the turn Inspect keeps never holds a rejected call that could run, so the proxy's protection holds whatever the agent does next. But new code writes the approval errors and transcript events, so they can drift from the ones inside Inspect. And an approver never sees a rejected call, where inside Inspect it sees every call first.

- *Option 2, carry each decision to the moment its tool runs:* the same code as inside Inspect does the work, at the same point, after the approver, so errors, events and limits come out the same. But the restored turn holds the rejected calls again, and only Inspect's tool loop, where Inspect runs each call, stops them. An agent that runs its own tools would run them. That includes a bridged agent, one that runs outside Inspect and only borrows its model calls, and a custom agent that calls its tools directly.

- *Option 3, option 2 with a guard:* Inspect restores the model's turn only for agents that run their tools through its tool loop, and hands every other agent the reply as the proxy changed it. Inspect can tell a bridged agent apart, because its calls take their own path. It can't tell a custom agent that calls its tools directly.

- *To settle:* option 2 matches inside Inspect most closely. Option 3 adds the proxy's protection for bridged agents, and leaves custom agents that call tools directly as the one gap.

### C. What if a task has a sentinel inside Inspect and a sidecar too?

- *What's going on:* a researcher can run a sentinel inside Inspect, set with `--sentinel`, while the deployment runs another in a sidecar behind the proxy.

- *Why it matters:* both would judge the same calls. They could disagree, and the transcript would hold two sets of reports for each call.

- *Option 1, fail the task at the start:* simple, and nobody is surprised by two sentinels. But it rules out a researcher adding their own monitor on top of the deployment's.

- *Option 2, run both:* the sidecar judges first, in the proxy, and the sentinel inside Inspect then judges each call before it runs, so either can stop it. But every call is judged twice, at twice the cost, and the transcript needs a way to show which sentinel said what.

- *Proposed:* option 1, until someone needs both.

### D. How long should Inspect wait for the sidecar's answer?

- *What's going on:* when a reply arrives without the field, Inspect asks the sidecar what happened, with `request_result` (section 5). The sidecar can answer that it is still judging, or `awaiting_reply`: it has judged the request, and the provider's reply hasn't reached it yet.

- *Why it matters:* inside Inspect, Inspect waits for its sentinel as it waits for anything in the sample, within the sample's limits. Through a sidecar, waiting too long stalls the sample, and giving up too soon loses the decision, which may have been a rejection.

- *Option 1, a fixed deadline in Inspect:* predictable. But a slow monitor's decision is lost, and under `auto` the reply is then used as if no sidecar had judged it.

- *Option 2, keep asking until the sidecar answers something else:* closest to inside Inspect. The sidecar can be started with a time limit for judging, after which it answers `not_judged`, but the limit is off by default. Without one, only the sample's own limits end the wait.

- *Also to settle:* how often to ask, since each ask is a call through the proxy, and that a sample that errors or is cancelled stops asking at once.

- *Proposed:* option 2, with the sample's limits as the backstop.

### E. How should the sentinel's decisions show in the transcript?

- *What's going on:* inside Inspect, each monitor's report becomes a `SentinelEvent` in the transcript, the step-by-step record in the log viewer. Each event is tied to the message or tool call it judged.

- *Why it matters:* through a sidecar, people should read the transcript the same way, and anything that reads those events should work unchanged.

- *Option 1, rebuild the same events from the field:* the transcript matches inside Inspect. But the sidecar never sees Inspect's message ids, because the provider request doesn't carry them, so Inspect has to tie each report to its message by the tool call's id and its place in the turn.

- *Option 2, one event per model call, holding the field as it came:* simple, and nothing is lost. But the log viewer, and anything else that reads sentinel events, would need to learn a second kind.

- *Also to settle:* a trimmed report. A sidecar started with `--no-eval-results` leaves monitors' explanations out of the field, so an agent can't learn what the monitors look for. Inspect can't fetch the rest, so the event should say the report was trimmed, rather than look as if the monitor had nothing to say.

- *Proposed:* option 1, to match inside Inspect.

### F. What if a task has a very large dataset?

- *What's going on:* registering a task sends every selected sample's input and metadata to the sidecar when the task starts. Inside Inspect, the sentinel reads each sample directly, so nothing is sent.

- *Why it matters:* a large dataset makes that one big request, and proxies usually cap a request's size. Samples added while the task runs aren't in it.

- *Option 1, one request:* simple, and the sidecar knows the whole task from the start. But a large task can go over the proxy's limit and fail to start, and the upload delays the first sample.

- *Option 2, batches:* stays under the limit. But a task can end up half registered if one batch fails, and the design needs a rule for that.

- *Option 3, register each sample just before it starts:* each request is small, and samples added later are covered. But it adds a call per sample, and the sidecar never holds the whole task, which a monitor that compares samples would need.

- *Proposed:* option 1 in PR 11, measured on a large dataset, with option 3 as the fallback.

### G. Should a locked-down sidecar refuse calls from tasks it doesn't know?

- *What's going on:* a sidecar can be locked down to run only the tasks an admin registered, for a deployment where the eval's client can't be trusted. It refuses to register any other task. But it still judges calls that name a task it doesn't hold, such as from a client that skipped registration.

- *Why it matters:* the lockdown is aimed at a client that can't be trusted, and that client is the one most likely to skip registration. Its calls are then judged without the task's facts, which a monitor may rely on.

- *Option 1, refuse calls that name a task it doesn't hold:* the lockdown then covers every call, not only registration. But a client with a bug in its registration fails at its first model call rather than at the start.

- *Option 2, judge them, as today:* nothing changes. But the lockdown only binds clients that register honestly.

- *Also to settle:* calls that name no task at all, such as a naked agent's, which the sidecar also judges today. That is a separate choice.

- *Proposed:* option 1. This is a sidecar change, not an Inspect one.

### H. Is fetching the store before every scoring too slow?

- *What's going on:* the sentinel's store holds what its monitors note about a sample, such as counts, and scorers can read it. Some agents score their own answers as they go, to decide whether to try again (section 6). Inside Inspect, such a scorer reads the live store.

- *Why it matters:* through a sidecar, the store lives in the sidecar, so each scoring needs a fresh copy, which is one more call through the proxy.

- *Option 1, fetch before every scoring:* always the latest store, as inside Inspect. But it is a call per scoring, and an agent that scores often pays it each time.

- *Option 2, fetch only before the final scoring:* one call per sample. But a scorer during the sample reads an out-of-date store, which differs from inside Inspect.

- *Option 3, fetch only if a model call has gone through since the last fetch:* the store changes only when the sidecar judges a call, so this is always the latest, with fewer calls. But Inspect has to track which calls went through since.

- *Proposed:* option 1 in PR 11, and option 3 if the cost shows up.

### I. What if there's nothing to fetch?

- *What's going on:* two cases get an error from `sample_result`, never an empty store. A sample the sidecar never saw gets `unknown_sample`. A sidecar started with `--no-eval-results`, to keep monitors' findings from an untrusted client, doesn't send results at all.

- *Why it matters:* a scorer that reads the sentinel's store must never be handed empty values. It would score the sample as one in which the sentinel found nothing.

- *Option 1, fail the sample's scoring:* safe. But scorers that never read the store can't score the sample either.

- *Option 2, mark the store as unavailable:* scorers that don't read it score as normal, and one that reads it gets an error saying which case happened. But Inspect needs a way to mark a store as unavailable, which it doesn't have today.

- *Also:* a sample that never called the model can skip the fetch, since the sidecar has nothing for it.

- *Proposed:* option 2.

### J. What if a task's models sit behind more than one sidecar?

- *What's going on:* a task can use several models. For example, an agent can hand part of its work to a model given to a role, and each model could go through a different proxy.

- *Why it matters:* inside Inspect, the sentinel judges every agent call, whatever the model. Through sidecars, each proxy's sidecar would need its own probe and registration, and Inspect would need to combine their stores.

- *Option 1, probe only the task's assigned model:* simple, and a task has at most one sidecar. But an agent call to another model gets none of Inspect's headers, so no sidecar can tie it to its sample, which differs from inside Inspect.

- *Option 2, probe every model the task uses:* every agent call is tied to its sample, as inside Inspect. But it needs a probe and a registration per sidecar, and a rule for combining their stores.

- *Proposed:* option 1 in PR 11, with the gap stated, and option 2 when a deployment needs it.

### K. What about model calls that aren't the agent's?

- *What's going on:* scorers often ask a "grader" model to judge an answer, and those calls can go through the same proxy. Inspect sends them without `x-inspect-context` (section 3), so the sidecar can't tell whose they are.

- *Why it matters:* inside Inspect, the sentinel never judges a grader. A sidecar judges a grader's calls as it judges any client's.

- *Option 1, leave them unmarked:* nothing to build. But the sidecar spends judging on calls the sentinel inside Inspect would ignore, and a monitor could block a grader.

- *Option 2, mark them, and the sidecar lets marked calls through:* matches inside Inspect. But the mark is a header the client sets and the sidecar can't check, so any client could mark an agent's call to get it past the sentinel. A deployment that can't trust its client must ignore the mark.

- *Proposed:* option 2, with the sidecar honouring the mark only where the deployment trusts the client.
