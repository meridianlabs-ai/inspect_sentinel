# Inspect Sentinel guardrail for LiteLLM

A guardrail for the LiteLLM proxy. LiteLLM runs the guardrail, and the guardrail calls a remote [Inspect Sentinel](https://github.com/meridianlabs-ai/inspect_sentinel) sidecar over HTTPS and applies its answer: pass, replace or refuse.

Inspect Sentinel does not run in LiteLLM. Only this guardrail does. It needs only LiteLLM and what LiteLLM's proxy already has.

## Using it

Install it into LiteLLM's image:

```bash
pip install inspect_sentinel_litellm
```

Then add it to LiteLLM's config:

```yaml
guardrails:
  - guardrail_name: inspect-sentinel
    litellm_params:
      guardrail: inspect_sentinel_litellm.LiteLLMGuardrail
      mode: [pre_call, post_call]
      default_on: true
      api_base: https://<sentinel host>:8900
```

| Setting | Meaning |
| --- | --- |
| `api_base` | the sidecar's address. The sidecar listens on port 8900 unless it is started with another |
| `mode` | both `pre_call` and `post_call`. A sentinel judges the tool calls in a reply, so the guardrail has to be shown replies as well as requests |
| `default_on` | `true` to judge every request. Without it, LiteLLM judges only a request or a key that names the guardrail |
| `unreachable_fallback` | optional: `fail_closed` refuses a call when the sidecar gives no usable answer, and `fail_open` lets it through. A replaced request that LiteLLM can't take counts as no usable answer, and with `fail_open` its reply goes on unjudged. `fail_closed` unless set |
| `timeout` | optional: how many seconds the sidecar may take to answer. For a streamed reply, it counts from when the sidecar has the whole reply, so neither a slow provider nor a caller slow to read is held to it. No limit unless set, because a sentinel may ask a person |
| `sidecar_headers` | optional: request headers to show the sidecar besides Inspect's own, as a list of names, or prefixes ending in `*` |
| `max_reply_bytes` | optional: the largest reply that is held to be judged. 32 MiB unless set |

An older LiteLLM loads a guardrail only from a file beside its config, and stops with `Could not find module file`. There, put this one line in a file named `sentinel_guardrail.py` beside the config, and name the class as `sentinel_guardrail.LiteLLMGuardrail`:

```python
from inspect_sentinel_litellm import LiteLLMGuardrail as LiteLLMGuardrail
```

## What it reads

Requests to LiteLLM's `/v1/messages`, `/chat/completions` and `/responses` routes, and their replies, streamed or not. A request to another route goes on unread.

A request on those routes is refused, whatever `unreachable_fallback` says, when it sets a field that LiteLLM rewrites the request from after the sidecar has judged it, since the provider would be sent something the sidecar never saw. These are `extra_body`, which the provider's SDK merges over the request; `ensure_alternating_roles` and the continue messages it inserts; the fields that load a stored prompt, such as `prompt_id`; `litellm_system_prompt`; the custom prompt template fields, such as `roles` and `final_prompt_value`; and `supports_system_message`. A client that calls through a provider's SDK sends none of them: the SDK merges its own `extra_body` into the request before sending it. A deployment that relies on LiteLLM's prompt management can't use it through the guardrail.

A request with an MCP tool that LiteLLM's own MCP gateway would run is refused too, whatever `unreachable_fallback` says. LiteLLM runs the model's calls to such a tool itself, and sends the provider follow-up requests that no guardrail is shown. The guardrail asks LiteLLM's own test, so it refuses what the installed LiteLLM would hand to its gateway: a `server_url` of `litellm_proxy`, or one ending in `/mcp/<name>`. An MCP tool that the provider calls itself goes to the sidecar as usual.

A streamed reply is handed to the sidecar as LiteLLM yields it, and the caller is sent what the sidecar releases. By default the sidecar holds the whole reply until it is judged, so a reply that is refused is refused before any of it has gone out. A sidecar started with `--release early` releases the text and reasoning before the turn's first tool call as they arrive, and the rest of the turn once it is judged; a reply refused after some of it has gone out ends with an error event that the provider's SDK raises on. `unreachable_fallback` decides what happens when the sidecar gives no usable answer, or doesn't judge the reply, only while none of the reply has gone out, keepalives aside; after that, the reply ends with an error event. While the turn is held, the sidecar sends a keepalive the caller's SDK skips, so that a quiet connection isn't closed. LiteLLM 1.83 can't pass one on for `/chat/completions` or `/responses`, so with that version only `/v1/messages` replies get them.

## What it sends

Each request and each reply that isn't streamed is one JSON-RPC 2.0 call to the sidecar, `judge_request` or `judge_reply`. A streamed reply is a reply stream: `begin_reply`, a `reply_chunk` call for what has arrived each time the last one is answered, and `end_reply`, while a `next_release` call waits beside them for what to send the caller. A call carries the route's format, the model's name in LiteLLM, the user's id, the request's or the reply's body, and some of the request's headers:

- An Inspect eval's own two, `x-inspect-context` and `x-irid`, which say which sample a request is for. The guardrail passes them on without reading them, and removes them from what LiteLLM sends the provider.
- Any listed in `sidecar_headers`.

A call never carries a credential, and it leaves out the notes LiteLLM keeps on the caller.

A reply also carries the headers the provider sent with it, where LiteLLM kept them. On `/chat/completions` and `/responses` it keeps them with the reply or the stream, and on `/v1/messages` with its log of the call when Anthropic's own API served it. A stream that another callback has wrapped first carries none on the OpenAI routes. They never include a cookie, a credential or one of LiteLLM's own headers. The sidecar keeps them with the reply of a model call that waited for a person, which an Inspect eval then reads as the provider's own.

A request that the sidecar let through and that then came to an error is told to the sidecar as well, as a `judge_reply` with the error, so that the sidecar doesn't go on waiting for a reply:

- The status is the one LiteLLM answers its caller with. For a provider it could not reach, that is the status LiteLLM gives such an error, such as 500 or 408.
- The headers are the provider's, where it replied and LiteLLM kept them. They can say whether the call may be tried again, as `retry-after` does.
- The body is what LiteLLM kept of the provider's error. On `/v1/messages` that is the provider's own body. On the other routes it is LiteLLM's account of the error as text, which holds the provider's.

A streamed request counts as one that came to an error when its stream fails before its first chunk. For some providers, such as Gemini and Vertex, LiteLLM makes the call only when the stream is first read, so that is where their error comes. A stream that fails after its first chunk is told to the sidecar as a reply that was cut off.

The caller is answered with LiteLLM's own error whatever the sidecar says. A caller that goes away before the provider replies is not told to the sidecar, as LiteLLM calls no guardrail then.

The body is what LiteLLM is about to send, with one thing put right. LiteLLM writes a Claude model's thinking, asked for through `/responses`, as a `reasoning` item that OpenAI's own types reject. A client that sends that item back in a later request sends one the sidecar can't read, so the guardrail corrects the item in what it sends the sidecar. LiteLLM is still sent the item as the client sent it.

The calls and their results are described by [`proxy_endpoint.schema.json`](https://github.com/meridianlabs-ai/inspect_sentinel/blob/main/src/inspect_sentinel/sidecar/proxy_endpoint.schema.json).

## How a refusal reaches the caller

Every client, an Inspect eval included, is sent the reply as the sidecar left it, or the refusal, in the provider's own format. An Inspect eval asks the sidecar what was decided of each of its requests, through the route below.

- A refused reply that isn't streamed is answered with LiteLLM's own error, with the sentinel's message.
- A refused stream on `/v1/messages` or `/chat/completions` is answered with an error event in place of the reply, which LiteLLM sends the caller as an error, as JSON under the refusal's status.
- A refused Responses stream is answered with a `response.failed` event, which the guardrail sends in place of LiteLLM's own. The event has an error at its top as well as the response's own, holding the same message and code: the OpenAI SDK raises an `APIError` on that error, where without it a client of the SDK would read the event as one more of the reply's. Its error code follows the refusal's status as LiteLLM's own would, so a client tries again only after a server error.

## The sidecar's routes for evals

An Inspect eval knows only the base URL of the model API it calls, which here is LiteLLM's. So the guardrail adds a route to LiteLLM's app: everything under `/sentinel/` is passed on to the sidecar at `api_base`, which is where the eval talks to the sidecar. Nothing else needs setting up.

LiteLLM checks the caller's key on these routes as it does on its model routes, and a key that may call LiteLLM's OpenAI routes may call these too. The sidecar is never sent the key.

- **What the sidecar is sent:** the request's body, its `content-type`, and Inspect's own `x-irid` and `x-inspect-context` headers. No other header is passed on.
- **What the eval is sent back:** the sidecar's status, its headers and its body. The headers of one hop, such as `content-length` and `connection`, are LiteLLM's own. An event stream is passed on as it arrives.

An eval also gets a held reply this way. When a sentinel asks a person about a tool call, the sidecar holds the provider's reply until the person decides. Once the turn may go on, Inspect sends the model call's request to `/sentinel/v1/held-reply`, under the call's `x-irid`. The sidecar answers with the provider's reply as it held it, so LiteLLM doesn't call the provider again.

## What LiteLLM decides

The guardrail is tested with LiteLLM 1.83 and 1.105.

LiteLLM writes the error a caller is sent when the guardrail refuses. With LiteLLM 1.83 and 1.105, a refusal has the sentinel's message and its status, as JSON, streamed or not. The one exception is a streamed `/responses` reply, which is answered with status 200 and a `response.failed` event that holds the message, and which the OpenAI SDK raises on.

With LiteLLM 1.83, more of the replies that LiteLLM converts from another provider's format can't be read. On its `/v1/messages` route, a refused reply that isn't streamed is answered with an error whose message starts with the status.
