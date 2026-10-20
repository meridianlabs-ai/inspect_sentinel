# Always sunny, behind Envoy

A sentinel that runs beside Envoy, not inside the eval.

The task is a weather reporter. It looks up Philadelphia's weather, which is heavy rain, and files a report. The sentinel has one rule: it's always sunny in Philadelphia. It rejects the rainy report, the model is told why, and it files a sunny one.

The task itself knows nothing about the sentinel. There is no `sentinel=` in `task.py`. Every model call goes through Envoy, and Envoy hands each request and reply to the sidecar before passing it on. No code of ours runs in Envoy: its `ext_proc` filter calls the sidecar's Envoy shim over gRPC.

[`../always_sunny`](../always_sunny) is the same example behind LiteLLM.

| File | What it is |
|------|------------|
| `rule/` | the rule, `always_sunny`, in a small package, `weather_rules`, that the sidecar's image installs |
| `sentinel.yaml` | the sidecar's file: the sentinel, which names that rule, and how the sidecar serves |
| `envoy.yaml` | Envoy's configuration, written by `python -m inspect_sentinel.sidecar.envoy` |
| `task.py` | the Inspect task, `philadelphia` |
| `compose.yaml` | a stack to run it all: Envoy and the sidecar |

## Running the stack

You need Docker and a provider API key.

```bash
docker compose up --build
```

Envoy listens on `localhost:8080`. Run the task with the model's base URL set to Envoy. Envoy holds no key of its own: the task sends its provider key, and Envoy passes it on to the provider. The sidecar is never shown it.

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 \
  inspect eval task.py@philadelphia --model anthropic/claude-haiku-4-5
```

For an OpenAI model:

```bash
OPENAI_BASE_URL=http://localhost:8080/v1 \
  inspect eval task.py@philadelphia --model openai/gpt-5-mini
```

To try a rule of your own, edit `rule/weather_rules/__init__.py` and `sentinel.yaml`, and rebuild the sidecar with `docker compose up --build sidecar`.

## What to look for

The sample passes, and Inspect reports one HTTP retry. That retry is the rejection.

`reject_status: 503` in `sentinel.yaml` is what lets the run carry on. The sidecar refuses the rainy report with a status Inspect retries by itself, and adds the rejection to the retried request, so the model sees its own report and the editor's note. With the default, 400, the refusal is final and the sample fails with the sentinel's message.

## What an operator sets

The sidecar's settings are in `sentinel.yaml`, under `server:`. `port` serves the sidecar's endpoints, and `envoy_port` serves the shim that Envoy's `ext_proc` filter calls. Each setting is also a command-line option, such as `--envoy-port`.

Envoy's settings are in `envoy.yaml`. Write it with the generator rather than by hand, since many of Envoy's defaults are wrong for holding a reply: a request has 200 ms to be judged, a reply's body can't be held whole, and a provider's certificate isn't checked. Tell the generator where the sidecar is, with ports that match `sentinel.yaml`:

```bash
python -m inspect_sentinel.sidecar.envoy --sidecar-host sidecar --sidecar-port 8900 --shim-port 8901 > envoy.yaml
```

Between each pair of parts:

| From | To | How it is kept safe |
|------|----|---------------------|
| Envoy | the provider | TLS, with the provider's certificate checked against the system's trusted certificates |
| the agent | Envoy | plain HTTP in this stack, where only your own machine reaches Envoy. Elsewhere, put Envoy where only the agents it serves can reach it, or add Envoy's own TLS and authentication |
| Envoy | the sidecar | plain HTTP and gRPC inside the stack, where nothing but Envoy reaches the sidecar: no port of the sidecar's is published |
