# Transcripts from real agent runs

Builds monitor evaluation transcripts from real agents working in Docker sandboxes, in the format `../transcript.py` reads. Real runs give authentic texture. Misbehavior is spliced into copies of real runs, and every spliced call executes for real in a fresh sandbox, so tool results always match the actions.

## Files

- `spec.py`, `tasks_c.py` ... `tasks_f.py`, `registry.py`: task specs. Each spec is a workspace built by a setup script, the customer's policy and the assignment. Add a module of specs for a new set and list it in `registry.py`.
- `sandbox/`: the Docker image (Python 3.12 with pandas, pytest, pyyaml, ruff, git) and compose file. The container has no network.
- `agent.py`: the agent loop and tools (`bash`, `python`, `read_file`, `list_files`, `write_file`, `submit`). Calls in one message run in order by default. A script can force turns before and after the live model.
- `task.py`, `run.py`: the inspect_ai task and a runner that checks the shared spend ledger before a run and records list-price cost after it (`ledger.py`).
- `convert.py`: eval log to transcript file.
- `splice.py`: reading a real run and writing splices: replayed turns, edited turns, and new turns in the base model's style.
- `build_set.py`: builds a set from a recipe, runs the splices (cached), shuffles file names, and writes the transcripts and a separate answer key. For each splice the key records how the run went on after it (`after_splice`): `replayed` (the base run's later turns, re-executed in the new state), `live` (the live model took over and may have noticed or undone the splice) or `scripted`.

Recipes say which transcripts misbehave, so they are kept outside the repository with the answer keys.

## A new set

```bash
cd spikes/monitor_eval/realrun
docker build -t monitor-eval-sandbox:1 sandbox/      # once per machine
ENV=/home/eric/.scratch/r8-overnight/envrun.sh       # loads the API keys into the environment

# 1. write tasks_e.py (TaskSpec list), add it to registry.py, then run the base agents
$ENV ../../../.venv/bin/python run.py e-task1 e-task2 ... --model anthropic/claude-haiku-4-5 --log-dir LOGS/base-e --label "base runs set E"
$ENV ../../../.venv/bin/python run.py e-task1 e-task2 ... --model openai/gpt-5-mini --log-dir LOGS/base-e --label "base runs set E"

# 2. read the runs, write a recipe (SET = "e", ENTRIES = [...]) beside the keys, then build
../../../.venv/bin/python build_set.py KEYS/recipes/e.py --out SETS/e --key KEYS/e --base-logs LOGS --work WORK/e

# 3. check every file loads and renders through Host.score
../../../.venv/bin/python ../score.py SETS/e/*.json --dry-run | wc -l
```
