# Monitor evaluation transcripts

Synthetic coding-agent conversations, stored as files, for checking how well `Host.score` and its prompt template judge an agent's steps. Each file is one conversation: the task, a policy, the tools the agent was offered, and the messages. `transcripts/set-a/` is for development. `transcripts/set-b/` is held out, to check that a prompt change generalizes. The expected results are kept outside this repository.

## Files

- `transcript.py`: the loader. `load_transcript()` reads a file. `Transcript.steps()` rebuilds the steps a host would hand a monitor: `BeforeGenerate` and `AfterGenerate` for each assistant message, then `BeforeToolCall` and `AfterToolCall` for each of its tool calls. `score_transcript()` runs `Host.score` on chosen steps inside `running_step`, with one sample store per transcript, so the nonce stays the same across a transcript as it does in a live sample. The file format is in the module docstring.
- `score.py`: a command line around `score_transcript()`.
- `authoring.py`, `set_a.py`, `set_b.py`, `build.py`: the code that writes the transcript files.

## Step ids

`t03.model_input`, `t03.model_output`, `t03.c2.tool_call` and `t03.c2.tool_result`: turn 3 is the third assistant message, and c2 is its second tool call. The stage names are the ones `SentinelEvent.stage` records. A turn marked `prefilled` has no generate steps, and its tool steps have no `input` or `tools`.

## Commands

Run these from the repository root after `uv sync`.

```bash
# list the step ids of a transcript
.venv/bin/python spikes/monitor_eval/score.py spikes/monitor_eval/transcripts/set-a/a01.json --list

# check that every file loads and renders, without calling a model (each step answers 0.0)
.venv/bin/python spikes/monitor_eval/score.py spikes/monitor_eval/transcripts/set-*/*.json --dry-run

# print the prompt the monitor model would get for one step
.venv/bin/python spikes/monitor_eval/score.py spikes/monitor_eval/transcripts/set-a/a01.json --dry-run --print-prompt --step t03.c1.tool_call

# score every tool call of a transcript with a real model
.venv/bin/python spikes/monitor_eval/score.py spikes/monitor_eval/transcripts/set-a/a01.json --model anthropic/claude-haiku-4-5 --stage tool_call

# show the monitor no input at all (only the call, or at a generate only the tools)
.venv/bin/python spikes/monitor_eval/score.py FILE --dry-run --print-prompt --input-window none
```

`--input-window` takes `all` (the default), `none`, or a number N for `LastTurns(N)`. The command line uses one built-in dimension. For your own dimensions, instructions or template, call `score_transcript()` from Python:

```python
import anyio
from transcript import ReplayHost, load_transcript, score_transcript

t = load_transcript("spikes/monitor_eval/transcripts/set-a/a01.json")
results = anyio.run(
    lambda: score_transcript(
        t,
        {"integrity": "..."},
        host=ReplayHost(t.task, "anthropic/claude-haiku-4-5"),
        stages=["tool_call"],
        template=MY_TEMPLATE,
    )
)
```

`score_transcript()` passes the transcript's policy as `policy=` unless you give one. Its other keyword arguments go to `Host.score`.

## Rebuilding

```bash
.venv/bin/python spikes/monitor_eval/build.py a spikes/monitor_eval/transcripts/set-a
.venv/bin/python spikes/monitor_eval/build.py b spikes/monitor_eval/transcripts/set-b
```

The build is deterministic, so a rebuild writes the same files.
