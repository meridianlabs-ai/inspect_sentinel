# Examples

Runnable sentinel configurations, each watching a short `react()` agent that has `bash()` and `python()` in a Docker container with no network (`compose.yaml`). `example_samples.py` holds the samples they share; each example defines its own `Task` at the bottom. Read them in this order:

| Example | What it teaches | Command |
|---------|-----------------|---------|
| `no_network.py` | A rule: a protocol with no monitors that rejects `curl`, `wget` and `pip install` | `inspect eval examples/no_network.py --model openai/gpt-5-mini` |
| `llm_suspicion.py` | A monitor that asks a model for a suspicion score, under `threshold(reject_at=0.8)` | `inspect eval examples/llm_suspicion.py --model openai/gpt-5-mini --model-role monitor=anthropic/claude-haiku-4-5` |
| `trajectory.py` | Two functions of one monitor (`before` and `after`) sharing `store_as()` state, under `observe()` | `inspect eval examples/trajectory.py --model openai/gpt-5-mini` |
| `nested.py` | Composition: a mapping of named children with a `concurrent()` of rules inside it, and a rule that ends the step with `final()` | `inspect eval examples/nested.py --model openai/gpt-5-mini --model-role monitor=anthropic/claude-haiku-4-5` |

The checks are deliberately simple, to keep the examples readable. They are illustrative, not protection: the container's `network_mode: none` is what keeps the agent off the network, and lexical rules like these are easy to evade (a different tool, `python()` instead of `bash()`, a string built at run time).

## Running

The examples need Docker running, `inspect_ai` from its `feature/sentinel` branch (which has `Task(sentinel=...)`, until it merges), this package, and the model provider SDKs:

```bash
# installs inspect_ai from feature/sentinel too, as this package's dependency
pip install "inspect_sentinel @ git+https://github.com/meridianlabs-ai/inspect_sentinel"
pip install openai anthropic
export OPENAI_API_KEY=...     # the agent
export ANTHROPIC_API_KEY=...  # the monitor role
```

(From a clone of this repository, `uv sync` installs the first two into `.venv`.) Then run the commands above from the repository root. Any capable models work; these are cheap. Each sample is capped at 20 messages.

## Reading the results

The sentinel's reports are `SentinelEvent`s in each sample's transcript, next to the tool events they judged. The log viewer shows them, or read them with `read_eval_log()`:

```python
from inspect_ai.event import SentinelEvent
from inspect_ai.log import read_eval_log

log = read_eval_log("logs/<file>.eval")
for sample in log.samples or []:
    for event in sample.events:
        if isinstance(event, SentinelEvent):
            print(
                sample.id,
                event.kind,
                event.path,
                event.suspicion,
                event.decision,
                event.explanation,
            )
```

Every configuration is wrapped in a root `concurrent()` (or `observe()` when it holds only monitors), which records its own decision at the empty path after its children's. What each example records:

- `no_network.py`: a `decision` at `no_network` for each `bash()` call, `reject` for the ones using a network command. The agent receives the decision's `message` as the tool call's error and carries on; the `explanation` is for the log only.
- `llm_suspicion.py`: an `observation` at `threshold/suspicion` with the model's score and reasoning, then `threshold`'s `decision`: `reject` when the score is 0.8 or more (the `cleanup` sample's deletes outside `/work`, `fetch_data`'s download), explained with the score and the monitor's reasoning, otherwise `continue`. The agent is told only that the call was not approved.
- `trajectory.py`: `observation`s at `observe/failure_count` alternating `before` and `after`. `before`'s suspicion rises by 0.2 after each call `after` saw fail.
- `nested.py`: when `protected` calls `final()`, the step ends at once. The log shows `bypassed` at `guard` and at the root (their decision logic was skipped), `cancelled` at `audit` (its model call was still in flight), and then `protected`'s `reject`. A step no rule ends has decisions at `guard/network`, `guard/protected`, `guard` and the root, and an observation at `audit`.
