# A rule

A protocol with no monitors decides each tool call by itself. `no_network` (from `examples/no_network.py`) rejects any `bash()` command that uses `curl`, `wget` or `pip install`. A rejected call does not run, and the agent gets the rule's `message` as the call's error.

## Sub-features

- `rule-proceed`: a call the rule allows runs, and the root records `continue`.
- `rule-reject`: a call the rule rejects does not run, and the root records `reject` with `message` and `explanation`.
- `rule-agent-message`: the agent sees the `message`, not the `explanation`, as the tool error.
- `rule-config-recorded`: the log header records the sentinel as config (`no_network`), so a retry rebuilds it.

## How to get to it (user POV)

- `Task(sentinel=no_network())` in a task file, then `inspect eval task.py`.

## Driving it with verify

Preconditions:

- `verify doctor` passes on the inspect_sentinel and inspect_ai lines.

- **Run.** `verify run rule`. The task `rule` runs one sample whose agent runs `echo allowed > $VERIFY_WORK/allowed.txt`, then `curl -s http://127.0.0.1:9 ; touch $VERIFY_WORK/network-ran.txt`.
- **Allowed call.** `report.txt` shows `decision reported path='' factory=no_network action=continue`, then the echo tool call with `error=None`.
- **Rejected call.** Next comes `action=reject explanation='command uses curl'`, then the curl tool call with `error='`curl` needs the network, which this task does not allow.'`.
- **Side effect.** `work files: ['allowed.txt']`. `network-ran.txt` is absent.
- **Config.** `sentinel config: SentinelConfig(root=SentinelEntry(name='no_network', ...))`.
- **Proof.** All checks PASS. Keep the run's `evidence/` path.

## Gotchas

- `no_network` matches the text of the call's arguments, whatever the argument names. A wrongly named argument still gets rejected, but then inspect_ai's tool validation also reports an error. If an allowed call shows a validation error, the scripted tool arguments are wrong (bash takes `command`).
- The rule only sees `bash` calls; a `python()` call that fetches a URL passes. That is the example's documented limit, not a bug.
