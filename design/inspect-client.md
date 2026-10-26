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

## Words this document uses

Three words mean more than one thing around Inspect and the sentinel. This document uses each in one sense only, and says where in the stack it happens. **Client side** is Inspect, where the eval runs. **Server side** is the proxy and its sidecar.

**Judging a step.** The sentinel looks at one step, such as a model call, its reply, a tool call or a tool result, and reaches a decision on it. Judging has two parts:

- a **suspicion rating**: what a monitor reports about the step, from 0.0 to 1.0. It is evidence, and decides nothing.
- a **protocol decision**: what a protocol does with the ratings: proceed, modify, reject, terminate or escalate.

Judging happens wherever the sentinel runs: client side when the sentinel runs inside Inspect, server side when it runs in the sidecar. A monitor can ask a model for its rating. This document calls that model the **judge model**, and it runs with the sentinel.

**Scoring.** Inspect grading a sample: a scorer gives the sample a **score**, such as correct or incorrect. Scoring is always client side, in Inspect, and the sentinel never scores. It usually happens after the agent finishes, and sometimes during a sample (section 6). A scorer can ask a model to grade an answer. This document calls that model the **grader**. Its calls can pass through the proxy, but they are not the agent's.

**The stores.**

- **The sample store**: Inspect's key-value store for one sample in one epoch. It is client side, always. It exists only while that sample runs, and its final contents are saved in the eval log. Tools, solvers and scorers use it.
- **Monitor state**: what a monitor or protocol keeps between steps, with `context.store_as()`, such as a count of failed calls. When the sentinel runs inside Inspect, it is a slice of the sample store, client side, named by the monitor's place in the sentinel. Through a sidecar, it is server side, kept for each run: one sample in one epoch, named by the `sample_uuid` in `x-inspect-context`. The sidecar drops it at `end_sample`, or a day after the run's last request.
- **The sidecar's records**: server side. What the sentinel reported for each model call, by the call's `x-irid`. They reach Inspect in the `sentinel` field, or through `request_result`. When the sentinel runs inside Inspect, the same history is the transcript's `SentinelEvent`s, client side.

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

- **At task start, Inspect probes the endpoint of the task's assigned model:** the model the eval runs with, such as `--model`. So `required` fails the task before any sample runs.
- **Any other endpoint is probed the first time an agent call goes to it,** such as a model given to a role that the agent hands work to. Inspect can't know these up front, because an agent can pick a model at any point.
- **Each endpoint is probed once per task,** per endpoint and credentials, and the answer is kept for the rest of the task.
- **An endpoint only graders, approvers, compaction or monitors use is never probed,** because no agent call reaches it.
- **An endpoint with no sidecar behind it gets plain model calls,** with none of Inspect's headers, as today.
- Nothing is shared between tasks in this PR.

Why it works this way, and what it costs, is question J.

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

**Decided:** a selected sample that isn't in the admin's file fails with a sample error that isn't retried (question A below).

## 3. Each model call

**Short version:** after a successful probe, Inspect adds a header to the agent's model calls. The sidecar uses it to tell which sample, and which attempt, a call belongs to.

**Proposed: which calls get it.** `x-inspect-context` goes only on calls that are:

- **agent turns**, the calls the sentinel would judge if it ran inside Inspect. Calls made by a scorer's grader, an approver or a compaction summary are not.
- **to an endpoint where a probe found a sidecar**, whatever the model (section 1).

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

**Proposed: Inspect's completion cache is off.** While a sidecar is active, Inspect doesn't reuse cached model replies. The sidecar judged a cached reply when it was first made, in an earlier eval, so its protocol decision says nothing about this one. The plan also allows having the sidecar judge a cached reply afresh. Turning the cache off is simpler. The provider's own prompt caching is unaffected.

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

**Case C: the run is ended.** The field says `outcome: terminate`. Inspect ends the sample, as in process: the sample's limit is recorded as `operator`, and Inspect's scorers still grade the sample.

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
| `dropped` | the sidecar once held the call's records, and no longer keeps them. The sidecar's judging may not have finished | `auto`: uses the reply, and records the protocol decision as unavailable. `required`: fails the sample | fails the sample |
| `unknown` | the sidecar never saw the call | as for `dropped` | fails the sample |
| no answer | `request_result` itself failed | as for `dropped` | fails the sample |

- **Ordinary retries stay.** Inspect still retries the provider and network failures it retries today. Once `request_result` confirms that the sidecar couldn't judge a call that came back as an error, the sample fails, with no extra retry on top.
- **None of these is permission to go on.** `dropped`, `unknown` and a failed lookup say nothing about what the sentinel decided. Under `auto`, the reply is used as it would be with no sidecar, and the transcript says the sidecar's protocol decision is unavailable.

**Proposed: a fuller answer.** Today `request_result` gives the outcome and the records, but not the field's step-by-step decisions. It should give the same information as the field, so Inspect can act exactly as it would on the field. This is a change on the sidecar's side.

## 6. The end of a sample

**Short version:** before Inspect scores a sample, it copies the monitor state from the sidecar (server side) into the sample store (client side). After the final scoring, it tells the sidecar the sample is over.

**Scoring can happen during a sample, not only at its end.** For example, `react` with more than one attempt scores each answer the model submits, to decide whether to stop (`agent/_react.py`). When the sentinel runs inside Inspect, the monitor state is already in the sample store, where that scorer can read it. Through a sidecar, the monitor state is server side, so it has to be fetched first.

**Proposed.**

1. Before any scoring, Inspect calls `sample_result` and copies the monitor state into the sample store, so Inspect's scorers read the latest. Scoring happens in three places in Inspect, and all of them need it:
   - `score()` (`scorer/_score.py`), which an agent such as `react` calls during the sample;
   - the final scoring, where the task runner calls each scorer itself (`_eval/task/run.py`);
   - interim scoring, where `inspect ctl task score` scores a running eval's samples on request, from outside them (question H).

   A fourth place needs no copy: `inspect score` re-scores a finished eval from its log (`_eval/score.py`), after the sidecar may have dropped the run. It reads the monitor state saved in the log with the sample store, which is why the copy has to happen before the final scoring.
2. Inspect scores the sample.
3. After the final scoring, `end_sample`: the sidecar drops the run's monitor state, and the rest of what it keeps for the sample.

Inspect skips `end_sample` when it will retry the sample, so that the retry keeps its monitor state in the sidecar.

**To decide:** what Inspect's scorers see when `sample_result` fails or has no monitor state for the sample (question I), and whether this copy is wanted at all (also question I). A failed copy must never look like a sample in which the sentinel had nothing to say.

## 7. Streams

**Short version:** PR 11 works with a sidecar that holds whole replies. Early release waits for PR 12.

**Decided.** PR 11 is built and tested for `release: whole`, where the proxy holds the whole reply until the sidecar has judged it. A whole reply can still travel as a stream: Inspect reads the stream, and the `sentinel` field is in its last event (section 4).

`release: early`, where text arrives while the rest is held, is [PR 12](sidecar-plan.md#pr-12-inspect-incremental-streaming). That covers a refusal arriving as an error event after some text, streams that are cut off, cancellation, and partial output shown while streaming.

**Proposed: a sidecar set to `early` before PR 12.** The probe answer says `"release": "early"`. Inspect stops the task with an error at the start, saying the sidecar releases replies early, which Inspect doesn't support yet.

Running as if there were no sidecar would be unsafe. Inspect's completion cache would come back on, and a cached tool call could run without the sidecar ever seeing it.

## Not built

**A response header.** If some proxy can't pass the `sentinel` field through, the sidecar could add a header saying it judged the request. No proxy we know of needs one, so none is designed.

## Questions to decide

The aim throughout is that a sentinel in a sidecar behaves as one inside Inspect does. Each question says what's going on and why it matters, and what the existing design docs already say. Then it gives the options, with what each one gains and costs, and what we propose.

### A. What if a researcher picks a sample the admin didn't register?

- *What's going on:* an admin can pre-register a task by writing its samples into the sidecar's tasks folder (section 2). Say the admin registered samples 1 to 50, and a researcher runs sample 51.

- *Already in the design:* the sidecar refuses a model request for a sample that isn't in the folder, with a 400 ([sidecar.md, the tasks folder](sidecar.md#the-tasks-folder)).

- *Why it matters:* the refusal comes at the sample's first model call. By then its setup has run, such as starting its sandbox, and the researcher sees what looks like a failed model call, not a sample that isn't allowed.

- *Option 1, keep the refusal where it is, and have Inspect name the cause:* no change to when it fails, and it also covers samples a task adds while it runs, with `enqueue_sample` (inspect_ai's `design/sample-source.md`). But the setup still runs first, and the sidecar's refusal has to say why, so Inspect can tell the researcher.

- *Option 2, refuse the whole registration:* the error comes when the task starts. But one unknown sample stops every sample in the task, including the 50 that are allowed.

- *Option 3, list the unknown samples in the answer to `register_task`:* Inspect fails just those samples, before their setup. But it needs a change on both sides, and it can't cover samples the task adds later, because they don't exist at registration.

- *Decided:* option 1. Inspect turns the sidecar's refusal into a sample error, which says the admin didn't register the sample. The eval's own error settings decide what it tolerates, as for any other sample error: `fail_on_error` fails the eval on the first error by default, or allows a number or proportion of failed samples.

- *Decided: the error is not retried.* A retry would be refused the same way, after running the sample's setup again. Inspect knows the cause for certain: the model call fails with a 400 and no `sentinel` field, so Inspect asks `request_result` (section 5), and the sidecar answers `not_judged` with the reason `unregistered`. Only that reason is never retried. The other reasons, such as `time_limit` or `provider_error`, can come out differently on another attempt, so they stay retryable.

- *Needed in inspect_ai:* a sample error that `retry_on_error` doesn't retry. Today Inspect retries every sample error while retries remain, except a cancellation (`SampleAttempt` in `_eval/task/run.py`).

### B. Where should Inspect act on the sentinel's decisions?

- *What's going on:* inside Inspect, the sentinel judges each tool call just before it runs, after any approver, such as a human asked to allow each call. Through a sidecar, the whole turn is judged in the proxy, which takes out rejected calls and rewrites changed ones before Inspect sees the reply. The field's `proposed` holds the turn as the model wrote it.

- *Already in the design:* Inspect acts on the field when the reply arrives. It puts each call that was taken out back into its own copy of the conversation, from `proposed`, with an approval error, and records the arguments a changed call ran with ([sidecar.md, the `sentinel` field](sidecar.md#the-sentinel-field)). Inspect's own approval works the same way for bridged agents, agents that run outside Inspect and only borrow its model calls: decisions apply to each model response "before that response is handed back to the agent" (inspect_ai's `docs/approval.qmd`).

- *Why it's still a question:* acting when the reply arrives differs from inside Inspect in one way. An approver never sees a call the sentinel took out, where inside Inspect the approver sees every call first.

- *Option 1, act when the reply arrives, as designed:* one rule for every agent, including bridged agents, which run their own tools. A rejected call never reaches any agent, so the proxy's protection holds. But this is new code beside the in-process code, and the approval errors and events it writes have to be kept in step with those.

- *Option 2, carry each decision to the moment its tool runs:* the same code as inside Inspect, at the same point, after the approver. But putting the model's turn back puts the rejected calls back too, and only Inspect's tool loop would stop them. A bridged agent, or a custom agent that calls its tools directly, would run them.

- *Decided:* option 1, as designed. This is deliberately different from inside Inspect, and the differences are to be stated plainly, here and in the user docs:
  - **An approver never sees a call the sentinel took out.** Inside Inspect, an approver sees every call first, and the sentinel judges after it. Through a sidecar, the sentinel has judged the whole turn before any approver runs, and an approver sees only the calls left.
  - **The whole turn is judged at once.** Inside Inspect, the sentinel judges each call just before it runs. Through a sidecar, it judges every call of a turn together, before any of them runs, so a `terminate` ends the sample before any call in that turn runs.
  - **What stays the same:** a rejected call is answered with an approval error, a changed call runs with its new arguments, and the model's own turn stays in the conversation. The new code builds those errors and events the way the in-process code does, so the two read the same.

### C. What if a task has a sentinel inside Inspect and a sidecar too?

- *What's going on:* a task can carry its own sentinel: its author sets `sentinel=` on the `Task`. Someone can then run that task in a deployment whose proxy has a sidecar. The person running the eval can also add one with `--sentinel`, which is hidden from `--help` while it's experimental.

- *Why it matters:* the task's author and the proxy's admin are usually different people, and neither knows about the other's sentinel. Both would judge the same calls. They could disagree, and the transcript would hold two sets of reports for each call.

- *Already in the design:* Inspect can't switch the sidecar off. "Whoever runs the proxy chooses the sentinel" ([sidecar.md](sidecar.md#two-kinds-of-client)), and no method on the endpoint for evals can loosen a decision or change the sentinel ([sidecar.md, a client that can't be trusted](sidecar.md#a-client-that-cant-be-trusted)). Leaving the sidecar out doesn't work either: the proxy hands every call to the sidecar, so it would judge Inspect's calls anyway, without the task's facts.

- *Option 1, fail the task at the start:* simple, and nobody is surprised by two sentinels. But a task with its own sentinel can never run in a deployment that has a sidecar.

- *Option 2, run both:* the sidecar judges first, in the proxy, and the sentinel inside Inspect then judges each call before it runs, so either can stop it. Each owner's sentinel runs, as each expects. But every call is judged twice, the transcript has to show which sentinel said what, and a monitor inside Inspect that calls a model through the same proxy has its own calls judged too, until question K is settled ([pr-series.md](pr-series.md#then-hosts-in-process-bridge-sidecar)).

- *Decided:* option 2, run both, with the sentinel inside Inspect first wherever it can be.
  - On a request, the sentinel inside Inspect judges it before Inspect sends it. It can end the sample there, so the request never reaches the proxy. Otherwise the request goes on, and the sidecar judges it in the proxy.
  - On a reply, the order is fixed the other way: the reply passes through the proxy before Inspect gets it, so the sidecar has judged it first. Inspect acts on the sidecar's decisions (question B), then the sentinel inside Inspect judges the reply and each remaining tool call as usual.
  - The transcript says which sentinel made each report.

- *Still to settle:* a monitor inside Inspect that calls a model through the same proxy has its own calls judged by the sidecar, until question K is settled.

- *To confirm with Meridian:* that "inside Inspect first" means wherever it can be. Going first on a reply too would need the proxy to hold every reply until Inspect's sentinel had judged it, which is a much larger design.

### D. How long should Inspect wait for the sidecar's answer?

- *What's going on:* when a reply arrives without the field, Inspect asks the sidecar what happened, with `request_result` (section 5). The sidecar can answer that it is still judging, or `awaiting_reply`: it has judged the request, and the provider's reply hasn't reached it yet.

- *Why it matters:* waiting too long stalls the sample. Giving up too soon loses the decision, which may have been a rejection.

- *Already decided in inspect_ai:* time a sample spends on its sentinel's monitors counts as waiting, not as working time, as it does for approvers (inspect_ai's `design/working-time-concurrency.md`). Waiting on the sidecar should count the same way. The plan leaves the wait itself to agree with Meridian ([sidecar-plan.md](sidecar-plan.md#pr-11-inspect-client-for-the-sidecar)).

- *Option 1, a fixed deadline in Inspect:* predictable. But a slow monitor's decision is lost, and under `auto` the reply is then used as if no sidecar had judged it.

- *Option 2, keep asking until the sidecar answers something else:* closest to inside Inspect, where Inspect waits for its sentinel within the sample's limits. The sidecar can be started with a time limit for judging a step, after which it answers `not_judged`, but that limit is off by default. Without one, only the sample's own limits end the wait.

- *Decided: how often to ask.* Inspect backs off: it asks again after half a second, then doubles the wait each time, up to a cap such as 10 seconds. A short judgment is caught quickly, and a long one costs only a few calls through the proxy. inspect_ai already backs off this way when it retries model calls (`_util/retry.py`). `request_result` can't wait for a judgment today. If the lag after a long wait ever matters, the sidecar could take an optional wait, at the cost of a request that stays quiet, which a proxy or load balancer may cut off.

- *Also:* a sample that errors or is cancelled stops asking at once, and `end_sample` belongs where Inspect already records every way a sample can end, exactly once (inspect_ai's `design/sample-lifecycle.md`).

- *Decided:* option 2, with the same settings Inspect has today for a sentinel inside it, and no new ones until users ask for them.
  - Inside Inspect, nothing limits how long a sentinel may take. While it runs, the sample's token, turn and working limits are paused, and only the sample's time limit, its clock time, still applies. Waiting on the sidecar works the same way.
  - The one limit on judging itself is the sidecar's `--time-limit`, set by whoever runs the proxy. It is off by default, as it is inside Inspect.
  - How often Inspect asks is Inspect's own choice, not a setting.

### E. How should the sentinel's decisions show in the transcript?

- *What's going on:* inside Inspect, each monitor's report becomes a `SentinelEvent` in the transcript, the step-by-step record in the log viewer. Each event is tied to the message or tool call it judged.

- *Already in the design:* "Inspect's transcript is the same either way" ([sidecar.md, testing](sidecar.md#11-testing)). The sidecar's records carry the step id and conversation, as `SentinelEvent` does, so the two can be shown in one viewer ([pr-series.md](pr-series.md#then-hosts-in-process-bridge-sidecar)).

- *Why it's still a question:* the sidecar never sees Inspect's message ids, because the provider request doesn't carry them. Inspect has to tie each report to its message another way.

- *Proposed:* tie each report by the tool call's id, and by its place in the turn for a report that isn't about one call.

- *Also to settle:* a trimmed report. A sidecar started with `--no-eval-results` leaves monitors' explanations out of the field, so an agent can't learn what the monitors look for. Inspect can't fetch the rest, so the event should say the report was trimmed, rather than look as if the monitor had nothing to say.

- *Also needed: where each report came from.* A `SentinelEvent` doesn't record where its sentinel ran, because until now it always ran inside Inspect. Each event should say whether it came from inside Inspect or from a sidecar. Then the log viewer, and any portal that shows results, can make a remote report plain, and the two sentinels of question C can be told apart. For a sidecar's report, Inspect also knows details a reader may want:
  - the proxy address where Inspect found the sidecar;
  - the endpoint version the sidecar speaks, and whether it holds whole replies, both from the probe (section 1);
  - the model call's `x-irid`, which matches the report to the sidecar's own records;
  - whether the report was trimmed.

  The event should carry these. Which of them a viewer shows can follow from what users ask for.

- *Gaps compared with inside Inspect, even with the events right:*
  - **Which sentinel ran.** Inside Inspect, the eval log records the sentinel's configuration: each monitor's and protocol's name, parameters and version. Through a sidecar, Inspect doesn't learn which sentinel the sidecar runs, so the log can't record it unless the sidecar says, for example in its answer to the probe.
  - **The judge models' calls.** Inside Inspect, a monitor's calls to its judge model appear in the transcript as model events, inside the sentinel's span. Through a sidecar, the sidecar makes those calls, server side, and they are recorded nowhere an eval or an operator can read later. They belong in the sidecar's record log, which is not yet designed ([sidecar.md](sidecar.md#10-operations)).

### F. What if a task has a very large dataset?

- *What's going on:* registering a task sends every selected sample's input and metadata to the sidecar before the first sample starts ([sidecar.md, registering a task](sidecar.md#registering-a-task)). The proxy's size limits are the operator's to raise. Inside Inspect, the sentinel reads each sample directly, so nothing is sent.

- *Why it matters:* Inspect is built not to hold a large dataset in memory: big datasets are read lazily, as samples run (inspect_ai's `design/hf-lazy-dataset.md`). A task can also add samples while it runs, with `enqueue_sample`, and those aren't in the registration.

- *Option 1, one request, as designed:* simple, and the sidecar knows the whole task from the start. But Inspect has to read the whole dataset at the start to send it, and samples added later aren't registered.

- *Option 2, batches:* stays under size limits. But it still reads the whole dataset at the start, and a task can be left half registered if one batch fails.

- *Option 3, register each sample just before it starts:* small requests, nothing read early, and samples added later are covered. But it adds a call per sample, and the sidecar never holds the whole task, which a monitor that compares samples would need.

- *Option 4, register the task with its sample ids, and complete each sample's registration before it runs:* `register_task` sends the task's facts, the ids of the samples selected for this run, and the number of epochs. Before a sample's first epoch, Inspect sends that sample's input and metadata, which completes its registration. A sample's input is the same in every epoch, so it is sent once. The first request stays small, the sidecar knows from the start which samples to expect, and Inspect reads each sample only when it runs, as it does anyway. But it adds a call per sample, and a new method on the sidecar's endpoint. A monitor that compares samples sees each one's input only once it has started.

- *Edges of option 4:*
  - A sample a task adds while it runs, with `enqueue_sample`, isn't in the first list. Adding it extends the registration rather than completing it, which a locked-down sidecar should refuse.
  - Listing the ids is free when Inspect assigns them, since they are numbers. A dataset with its own ids has to be read once, record by record, without keeping anything, as inspect_ai's lazy datasets already do when they are built.

- *Proposed:* option 4. A fixed task is unaffected, since the admin's file already holds every sample.

### G. How strict should a sidecar be about calls it can't place?

- *What's going on:* a sidecar can be locked down to run only the tasks an admin registered, for a deployment where the eval's client can't be trusted. It refuses to register any other task. But it still judges every call the proxy carries, and those come in three kinds:
  - calls that name a task the sidecar holds, which it judges with the task's facts;
  - calls that name a task it doesn't hold, such as from a client that skipped registration;
  - calls that name no task at all: non-eval traffic, such as a naked agent's, and today also a grader's calls (question K).

- *Already in the design:* what to do with the second and third kinds is stated as not yet designed ([sidecar.md](sidecar.md#advanced-a-sidecar-that-runs-only-the-tasks-in-its-folder)). The plan for PR 15 points one way: "A protected model route must not treat missing identity as permission to create an anonymous run" ([sidecar-plan.md](sidecar-plan.md#pr-15-trusted-run-identity-and-termination-scopes)).

- *Why it matters:* the lockdown is aimed at a client that can't be trusted, and that client is the one most likely to skip registration. Its calls are then judged without the task's facts, which a monitor may rely on.

- *The catch with telling eval traffic from other traffic:* the only thing that marks a call as an eval's is the `x-inspect-context` header, which the client sets. A client that wants to dodge a refusal can leave the header off, and its calls then look like non-eval traffic. So a rule that blocks unknown tasks but lets non-eval traffic through only binds an honest client, unless the proxy tells the two apart some other way, such as by the caller's credential.

- *Proposed: one setting on the sidecar, with levels of strictness.*
  - `open`, today's behaviour and the default: every call is judged, and a call that names a task the sidecar doesn't hold is judged without its facts.
  - `known_tasks`: a call that names a task the sidecar doesn't hold is refused. A call that names no task is still judged, so non-eval traffic keeps working. This catches a client with a bug in its registration, but not one that leaves the header off.
  - `registered_only`: a call that doesn't name a task the sidecar holds is refused, including non-eval traffic. This binds a client that can't be trusted, but nothing other than registered evals can use the proxy, and a grader's calls are refused until question K gives them a way through.

- *Also to settle:* whether a deployment that wants non-eval traffic and a client it can't trust should tell them apart by credential, as PR 15's trusted run identity would. The setting's name and its levels are Meridian's to choose. This is a sidecar change, not an Inspect one.

### H. Is copying the monitor state before every scoring too slow?

- *What's going on:* monitor state holds what monitors note about a sample, such as counts, and Inspect's scorers can read it. When the sentinel runs inside Inspect, the monitor state is in the sample store, client side, so a scorer reads it live. Through a sidecar, the monitor state is server side, so each scoring needs a fresh copy in the sample store, which is one more call through the proxy.

- *Why it matters:* Inspect scores at more moments than the end of a sample. Some agents have their answers scored as they go, to decide whether to try again (section 6). And `inspect ctl task score` scores a running eval's samples on request, from outside the samples (inspect_ai's `design/ctl/interim-scoring.md`), so a fetch tied to a sample's own scoring would miss it.

- *Option 1, copy before every scoring, wherever it runs:* the scorer always reads the latest monitor state, as inside Inspect. But it is a call per scoring, and an agent that scores often pays it each time.

- *Option 2, copy only before the final scoring:* one call per sample. But a scorer during the sample, or an interim scoring, reads out-of-date monitor state, which differs from inside Inspect.

- *Option 3, copy only if a model call has gone through since the last copy:* the monitor state changes only when the sidecar judges a step, so this is always the latest, with fewer calls. But Inspect has to track which calls went through since.

- *Deliberately different, whichever option:* inside Inspect, every change to monitor state is recorded in the transcript as a `StoreEvent`, and the state survives checkpoint and resume. Through a sidecar, the copy gives the sample store only the monitor state at the moment of scoring, with no record of each change.

- *Depends on question I:* if scorers aren't to read monitor state through a sidecar at all, there is nothing to copy, and this question goes away.

- *Proposed:* option 1 in PR 11, and option 3 if the cost shows up.

### I. What do Inspect's scorers see when the sidecar has no monitor state for a sample?

- *Background:* while a sample runs, monitors keep monitor state, such as how many calls they flagged. An Inspect scorer can read it from the sample store, for example to mark a sample unsafe if the sentinel flagged anything. Through a sidecar, the monitor state is server side, so Inspect copies it into the sample store with `sample_result` before scoring (section 6).

- *Where this comes from:* Meridian's own design docs say a scorer can read what a monitor accumulated: "a scorer holding a `TaskState` reads it with the existing `store_as(TrajectoryState, instance="attempt/internet_attempt")`" ([sentinel.md, State](sentinel.md#state), and the same in sentinel-reference.md and sentinel-deployment.md). It is incidental, not designed: it works because, inside Inspect, monitor state is kept in the sample store. Nothing in inspect_sentinel uses it, in an example or a test. The sidecar's design took it as a capability to keep, and built `sample_result` and the copy for it.

- *First, to settle with Meridian:* whether scorers should be able to read monitor state through a sidecar at all. If not, `sample_result`'s copy isn't needed for scorers, section 6 shrinks to `end_sample`, and this question and question H go away.

- *What's going on:* sometimes `sample_result` answers with an error instead of the monitor state. A sample the sidecar never saw gets `unknown_sample`, for example one that failed before its first model call. A sidecar started with `--no-eval-results` doesn't send monitors' findings to the eval at all, so that an untrusted agent can't learn what the monitors look for.

- *Already in the design:* missing monitor state is never read as clean. Without the copy, "a scorer would find the default values in the store and would read the sample as clean, with no error to show that anything was missing" ([sidecar.md](sidecar.md#what-comes-back-to-the-eval)).

- *Option 1, fail the sample's scoring:* safe. But scorers that never read the monitor state can't score the sample either.

- *Option 2, mark the monitor state as unavailable in the sample store:* scorers that don't read it score as normal, and one that reads it gets an error saying which case happened. But Inspect needs a way to mark part of the sample store as unavailable, which it doesn't have today. Interim scoring already works this way for scorer failures: they're "recorded on the row and don't fail the pass".

- *Also:* a sample that never called the model can skip the fetch, since the sidecar has nothing for it.

- *Proposed, if the copy is kept:* option 2.

### J. What if a task's models sit behind more than one sidecar?

- *What's going on:* a task can use several models. For example, an agent can hand part of its work to a model given to a role, and each model could go through a different proxy, or straight to its provider.

- *Already in inspect_ai's design:* inside Inspect, the sentinel judges every agent call: "a call is the agent's whatever model it uses" ([generate-stages.md](generate-stages.md#the-rule)).

- *Why it matters:* an agent call that gets none of Inspect's headers can't be tied to its sample by any sidecar. Probing only the assigned model would leave that gap for every other model the agent uses.

- *Decided: probe each endpoint the agent uses, and send headers only where a sidecar answers* (section 1). The assigned model's endpoint is probed at task start. Any other endpoint is probed on the agent's first call to it, and the answer is kept for the task, per endpoint and credentials. An endpoint with no sidecar gets plain model calls. Endpoints that only graders, approvers, compaction or monitors use are never probed, since no agent call reaches them.

- *What it adds:*
  - Each sidecar found gets its own `register_task`, and its own `end_sample` when the sample ends. Inspect sends `request_result` to the sidecar the call went to.
  - Registering can happen mid-run, on the agent's first call to a new endpoint. A failed or refused registration there fails that sample, not the task. This fits the registration that question F proposes, which comes in pieces anyway.

- *Stated plainly:* `required` covers only the assigned model's endpoint. An agent call to another endpoint with no sidecar behind it goes through unjudged, even under `required`.

- *Still to settle:* monitor state from more than one sidecar. If Inspect's scorers are to read monitor state (question I), Inspect would copy it from each sidecar, and two sentinels could use the same names in it. If they aren't, this goes away.

### K. What about model calls that aren't the agent's?

- *What's going on:* Inspect's scorers often ask a grader to grade an answer, and the grader's calls can go through the same proxy. Inspect sends them without `x-inspect-context` (section 3). The sidecar judges them as agent turns.

- *Why it matters:* inside Inspect, the sentinel never judges a grader's calls. A sidecar spends judging on them, and a protocol decision could block a grader.

- *Already in the design:* Meridian's plan is that "the proxy marks calls that aren't the agent's turns … so the sidecar can skip them as `is_agent_generate()` does in-process" ([pr-series.md](pr-series.md#then-hosts-in-process-bridge-sidecar)). Where the sidecar's own calls are exempted, it is "by the sidecar's signed-in identity, and not by a header" ([sidecar.md](sidecar.md#8-what-a-sentinel-is-given)), because anything in a header is the caller's claim. And no method on the endpoint for evals can loosen a decision ([sidecar.md](sidecar.md#a-client-that-cant-be-trusted)), so Inspect can't be the one to say which calls go unjudged.

- *Direction:* the operator preconfigures which models serve roles other than the agent's, such as the grader, in the sidecar's configuration or the admin's task file. It is fixed for the run, and nothing the client sends can change it. The sidecar doesn't judge calls to those models.

- *The catch, and how to close it:* that exempts a model, not a caller. An agent allowed to call the grader's model could do its work there, unjudged. So the proxy side honours the exemption only for a call that carries proof it came from Inspect's grader. The proof can't be reusable, or one leak would exempt anything forever. Each proof is:
  - tied to its request, as a signature over the request's content, so it can't be moved to another call;
  - single use, with a nonce the proxy side refuses to see twice, so a captured call can't be replayed;
  - short-lived, with a timestamp in a small window;
  - scoped to one sample and one role.

  AWS's request signing works the same way: every request is signed over its own content and time, with a key that never travels.

- *Not this:* `Host.score` already puts a nonce on the markers in the prompt a monitor sends its judge model. That guards the prompt against injection. It is kept in the monitor state, which can leak to the agent, and no proxy checks it.

- *Still to settle:*
  - **Where the signing key lives, out of the agent's reach.** In a Docker sandbox, or with a sandboxed agent behind the bridge, the agent's actions run away from the host, where Inspect can keep the key. With the `local` sandbox, the agent's commands run on the host, so no key there is safe. In a deployment that runs a sidecar because it can't trust the client, the host is the untrusted part, and the operator should exempt nothing.
  - **How Inspect keeps the key out of anything it passes into a sandbox,** such as environment variables.
  - **Who issues the key:** the operator, on both sides, or the sidecar at registration, as a short-lived key for that task. Either way it belongs with PR 15's trusted run identity ([sidecar-plan.md](sidecar-plan.md#pr-15-trusted-run-identity-and-termination-scopes)).
