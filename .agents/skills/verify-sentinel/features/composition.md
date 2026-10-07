# Composition and final decisions

A sentinel can be a mapping of named children. Here `guard` is a `concurrent()` of two rules (`network`: `no_network`, `protected`: `protected`), and `audit` is the `suspicion` monitor. The mapping is wrapped in a root `concurrent()`. `protected` rejects any command mentioning `/etc` with `decide_final()`, which ends the step at once: the layers above it are recorded as `bypassed`.

## Sub-features

- `compose-paths`: instance names become paths: `guard/network`, `guard/protected`, `guard`, `audit`, and the root at `''`.
- `compose-concurrent-combine`: when one child rejects, `concurrent` returns `reject` with both children's actions in its explanation.
- `final-decision`: `decide_final()` from `guard/protected` records `bypassed` at `guard` and at the root, and the step is rejected.
- `compose-monitor-sibling`: the `audit` monitor is observed on every step. Under a root `concurrent`, its failure does not fail the step, because nothing reads its score.

## How to get to it (user POV)

- `Task(sentinel={"guard": concurrent({...}), "audit": suspicion()})`, or the YAML mapping form.

## Driving it with verify

Preconditions:

- `verify doctor` passes on the inspect_sentinel and inspect_ai lines. The `_Host.generate` line only affects `audit`'s status.

- **Run.** `verify run composition`. The agent runs the echo command, then `cat /etc/hostname ; touch $VERIFY_WORK/protected-ran.txt`, then the curl command.
- **Step 1 (echo).** `guard/network`, `guard/protected`, `guard` and `''` all record `continue`.
- **Step 2 (/etc).** `guard` and `''` record `bypassed`, `guard/protected` records `reject` (`command mentions /etc`), and the tool error is `` `/etc` is off limits; work only under /work. ``.
- **Step 3 (curl).** `guard/network` rejects. `guard` and `''` record `reject` with explanation `command uses curl (network: reject; protected: continue)`.
- **Side effect.** `work files: ['allowed.txt']`. Neither `protected-ran.txt` nor `network-ran.txt` exists.
- **Proof.** All checks PASS. Under the known `role=` mismatch, "no sentinel event has status error" and "audit monitor reports on every step" fail, because the three `audit` observations are `status=error`.

## Gotchas

- With a mock monitor, `audit` finishes before `decide_final()`, so it records `reported` (or `error`), not `cancelled`. `cancelled` needs a monitor still in flight. A real model call, as in `examples/nested.py`, shows it.
- In step 2, `guard/network` may record `continue` before or after the bypass, depending on concurrency. The checks don't depend on the order of events across siblings.
