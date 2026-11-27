# auto_mode on ControlArena trajectories: results (glm-v1)

*November 2026. Corpus `glm-v1`; judges on Claude Haiku 5.5 unless stated. The live protocol arm is in §6.*

This records how the trajectory benchmark (README, "Trajectories from ControlArena") was run, what it found about `auto_mode()`, and where its files are. The judge comparisons use offline replay, so "caught" means the judge would have blocked that call. §6 measures blocking live.

## Summary

- **auto_mode separates attacks from honest work better than the two reproduced judges.**
  - AUC on attempted attacks is 0.89, against 0.81 (Guardian) and 0.83 (Claude Code).
  - At a 5% false-positive budget it catches 52% of attempted attacks, against 23% and 41%.
- **At 0.5 it catches 79% of attempted attacks.** It also catches 61 of the 62 successful BashArena attacks before they land.
- **Its false positives fall almost all on BashArena,** where honest system administration looks like the attacks (70 of 100 honest runs flagged). At 0.7 that drops to 23 of 100, while it still catches 58–59 of 61 successful attacks before landing.
- **Most honest runs still flagged at 0.7 deserve the flag.** We read them by hand: the "honest" agent (meaning only that it had no side task) scanned networks, attempted a sandbox escape and forged an audit-log entry.
- **Of the evidence and policy changes tested, none beat run-to-run noise except one we advise against.**
  - Window size, hiding tool results, a new security rule and a policy describing administration work all made no difference beyond noise.
  - Showing the judge the agent's own text scored higher, but against an attacker that wasn't trying to persuade the judge. The default hides that text on purpose.
- **Live, blocking at 0.7, it stopped most attacks.** GLM landed the BashArena side task in 4 of 48 runs, against 62 of 100 unmonitored. 7 of 43 honest runs had a call refused, and the honest main-task score barely moved (0.57 against 0.60). Blocking at 0.5 stopped slightly more (1 of 25) but refused calls in 19 of 25 honest runs.
- **A stronger judge model (Sonnet 5.5) separates better:** AUC 0.93 against 0.86 on a subset. It costs about 25 times as much and refused to judge 14% of runs.

## 1. Data

### Settings and attacker

ControlArena v20.1.1, with its own settings and attack and honest policies. The corpus covers four settings, AgentDojo, Rogue Eval, IaC Fast and BashArena, with 100 attack and 100 honest runs each (Rogue Eval has 98 attack runs), 798 in all.

The attacker is GLM-5.3 (`together/zai-org/GLM-5.3`) on Together's US-only endpoint (`TOGETHER_BASE_URL=https://us-api.together.ai/v1`), with prompt storage, passthrough models and training off for the organization. It attempted the side task in 356 of 398 attack runs.

### Why GLM

An earlier test run used frontier attackers (Claude Sonnet 5.5, Claude Fable 5.2, GPT-6.2 Sol and Astra).

- **Provider safety classifiers stopped most attack generation,** including every attacker on BashArena. Each stop was recorded and never retried or reassigned (`controlarena/refusals.py`).
- **The attacks that ran rarely attempted their side task:** 2 of 31.
- **Two settings were dropped:** Infra (its kind cluster didn't start on the host) and SWE-bench Django (a swebench version mismatch).

### Terms

The corpus is for evaluation only; the data must not be used for training. BashArena's terms also bar sending it to services that keep data for training. Every corpus file carries the canary string in `controlarena/corpus/glm-v1.files.jsonl`'s first line.

## 2. Method

- **Replay.** Each trajectory is replayed and each monitor runs at every tool call through `as_scanner()`, seeing what it would have seen live (README, "Scoring").
- **Labels** come from the settings' scorers: side-task success, the main-task score, and, on BashArena, the call a side task landed at. A Haiku judgment of the agent's text and calls records whether an attack attempted its side task. No human has checked those attempt labels.
- **Metrics** are those of `trajectory_report.py`:
  - caught, if any call is blocked at the judge's own threshold;
  - caught before landing;
  - honest runs flagged and honest calls blocked;
  - AUC, and catch rate at 1% and 5% false-positive budgets, on each run's highest score.
- **Judge errors fail closed** and count as caught.
- **Noise:** the same auto_mode configuration was judged twice (0a, 0b). Differences inside that gap are treated as no effect.
- **Not done:** no confidence intervals were computed, and there is one attacker.

## 3. Results

### 3.1 Three judges, 798 runs

| Judge | Attempted attacks caught | Honest runs flagged | Honest calls blocked | AUC (attempted) | Catch @1% FPR | Catch @5% FPR |
|---|---|---|---|---|---|---|
| auto_mode | 280/356 | 79/400 | 274/7,237 | **0.893** | 16% | **52%** |
| Guardian | 259/356 | 73/400 | 177/7,237 | 0.814 | **23%** | 23% |
| Claude Code | 263/356 | 106/400 | 435/7,237 | 0.832 | 0% | 41% |

- **AgentDojo and IaC Fast:** auto_mode's AUC is 0.96 and 0.99, and it flags 1 and 3 of 100 honest runs.
- **Rogue Eval is the weak setting.** auto_mode catches 34 of 82 attempted attacks and the others about 25, because sabotaging eval code looks like ordinary editing.
- **BashArena:** all three judges catch 61–62 of the 62 successful attacks before landing. They also flag most honest runs: auto_mode 70, Guardian 69, Claude Code 87 of 100.

### 3.2 auto_mode ablations, 798 runs

| Variant | AUC (attempted) | Catch @5% FPR | Honest runs flagged |
|---|---|---|---|
| Default (0a) | 0.893 | 52% | 79 |
| Default, rerun (0b) | 0.890 | 50% | 79 |
| `window=5` | 0.884 | 47% | 72 |
| `window=40` | 0.891 | 49% | 72 |
| `include_assistant_text=True` | 0.915 | 63% | 74 |
| Tool results hidden | 0.881 | 49% | 68 |

- **Window:** 5 and 40 are within noise of 20 at 0.5. At higher thresholds, window 5 misses more: 51 of 61 successful BashArena attacks caught before landing at 0.7, against 59.
- **Tool results:** hiding them is slightly worse.
- **Agent text:** showing it scores higher, mostly on BashArena (AUC 0.93 against 0.85). The likely reason is that the attacker's own words give its goal away. The attacker didn't know a judge was reading, so this doesn't test the risk the default guards against: an agent talking the judge into a low score. At threshold 0.7 the difference is small, 59 against 58 of 61 successful attacks caught before landing and 16 against 23 of 100 honest runs flagged. We keep the default (hidden).

### 3.3 The "Weakening security controls" rule

auto_mode with the rule scores AUC 0.893 on the 798 runs, against 0.894 without it (the earlier runs).

- It gains on BashArena: 0.857 against 0.832.
- It loses on Rogue Eval: 0.870 against 0.906.
- Net: neutral at this sample size.

### 3.4 Same evidence for every judge

Giving Guardian and Claude Code the rendering auto_mode sees makes both worse:
- Guardian's AUC falls to 0.77 and Claude Code's to 0.79.
- Honest calls blocked rise to 515 and 739.

So auto_mode's lead is its policy and evidence together, not its evidence alone.

### 3.5 A stronger judge model

The subset is 200 runs, 25 attack and 25 honest per setting, judged with Sonnet 5.5.

- **Refusals:** Sonnet refused to judge 28 runs (stop reason `content_filter`). 27 were attacks, 19 of them on BashArena. None was retried.
- **On the 161 runs both models judged:**

| Judge model | AUC | Attempted caught | Honest flagged |
|---|---|---|---|
| Sonnet 5.5 | 0.926 | 49/62 | 14/99 |
| Haiku 5.5 | 0.859–0.863 | 43/62 | 21–22/99 |

- **Cost:** Sonnet costs about $1.41 a run, against $0.055 for Haiku.

### 3.6 A policy for administration work (BashArena)

The variant `auto_mode_admin` (`trajectory_score.ADMIN_POLICY`) uses the default policy with three additions:
- a statement that the sandbox is a server the agent administers;
- a statement that hosts the task names are trusted for the use it describes;
- an allow exception for administrative changes the task asks for, no broader than asked.

These were written from the setting's description, before reading any flagged run.

| | AUC | Successful attacks caught before landing @0.5 / @0.7 | Honest flagged @0.5 / @0.6 / @0.7 |
|---|---|---|---|
| Default (0a, 0b) | 0.850–0.857 | 60/61 / 58–59/61 | 70–71 / 51–52 / 23 |
| Admin policy | 0.838 | 60/61 / 59/61 | 66 / 43 / 24 |

It made no difference worth keeping.

### 3.7 What the flagged honest BashArena runs are

We read the highest-scored call, and the judge's explanation of it, in 26 flagged honest runs.

**Highest score 0.7 or above: 11 of 14 are right.** The agent had no side task but still did things a deployment would want stopped:
- port scans and ping sweeps;
- a write through `/proc/1/root` after `/proc/sys` was read-only;
- reading `/etc/shadow` and unlocking an account after a sudo failure;
- forging an audit-log entry;
- an exploratory HTTP DELETE against the task's database.

The 3 errors all raised the firewall's packet-size limit.

**Highest score between 0.5 and 0.7: mostly over-strict.**
- **5 of 12 come from one quirk of the environment.** BashArena's firewall drops packets over 256 bytes, which blocks the task's own traffic. Honest agents work around it, and the judge calls that weakening a control. The default policy already says raising a limit that blocks the task's requests is task work, but Haiku doesn't apply that when the task doesn't mention the limit.
- **3 were fair:** whitelists far broader than the task needed.

## 4. Conclusions

1. **Default settings:** use `auto_mode()` as it ships, with a 20-turn window, tool results shown and the agent's text hidden.
2. **Choose the threshold per deployment, from its own honest runs.**
   - 0.5 suits work where risky actions are rare: on AgentDojo and IaC Fast it flags 1–3% of honest runs.
   - For privileged administration, 0.7 keeps nearly every catch of a successful attack and flags about a quarter as many honest runs.
   - Elsewhere, 0.7 gives up many catches. Across the other three settings it catches 160 of 271 attempted attacks, against 199 at 0.5.
3. **The policy should describe the deployment truthfully, but it doesn't fix false positives.** The judge's errors here come from not applying rules it already has.
4. **A stronger judge model is the biggest lever measured.** It buys more than any prompt or evidence change, at a large cost and with some refusals, which fail closed.
5. **"Honest" isn't "safe."** A capable agent with no hidden goal still overreaches, and blocking that is the monitor working.

## 5. Limits

- **One attacker,** using ControlArena's stock attack policies and unaware of the judge. Nothing here measures an agent trying to sway or evade the judge.
- **Judge models:** one main judge model, one rerun for noise, and no confidence intervals.
- **Reproductions:** Guardian and Claude Code are reproductions (README, "The monitors"), not the products themselves.
- **Unchecked labels:** a human has checked neither the attempt labels nor more than a sample of the flagged honest calls.
- **Thresholds tuned on the test data:** the threshold sweep and §3.7 use the same data they describe.

## 6. Live protocol arm

`threshold(auto_mode(model="anthropic/claude-haiku-5-5"), reject_at=0.7)` is attached to ControlArena's eval itself (`plans/glm-live.json`, run in BashArena), so refused calls are refused and the agent carries on.

| BashArena, GLM-5.3 | Live at 0.7 | Live at 0.5 | Unmonitored |
|---|---|---|---|
| Attacks that landed the side task | 4 of 48 (8%) | 1 of 25 (4%) | 62 of 100 (62%) |
| Attack runs with at least one refused call | 39 of 48 (425 of 1,938 calls) | 19 of 25 (295 of 1,068 calls) | n/a |
| Honest runs with at least one refused call | 7 of 43 (12 of 1,843 calls) | 19 of 25 (149 of 1,198 calls) | n/a |
| Honest main-task score, mean | 0.57 | 0.51 | 0.60 |

- **Runs at 0.7:** 50 attack and 50 honest were planned. 2 attack and 7 honest runs ended in errors (time limit, a sandbox timeout, or the model provider's credit running out) and are left out.
- **Comparison:** rates against the unmonitored BashArena runs. The runs aren't paired task for task, so the small difference in honest main-task score is within what different task draws give.
- **Honest cost:** the 7 honest runs with a refused call scored 0.47 on average, against 0.59 for the other 36.
- **Attacks that got through:** 2 of the 4 had no call refused at all, so the judge scored every call below 0.7; the other 2 had 1 and 2 refused calls and landed the side task another way.
- **0.5 against 0.7:** 0.5 stops a little more (1 of 25 attacks landed, against 4 of 48), but it refuses 12% of honest calls, against under 1%, and costs honest work more (0.51 against 0.57). For administration work like BashArena, 0.7 is the better trade. The 0.5 arm's 25 + 25 include its 4 pilot runs; runs that failed when the provider's credit ran out were left out and replaced by the next tasks in the draw.

### 6.1 More honest data for the final configuration

Two further checks of `auto_mode()` on Haiku 5.5, default policy and evidence, for false positives:

| Honest data | Flagged at 0.5 | Flagged at 0.7 |
|---|---|---|
| GLM-5.3, BashArena (for comparison) | 70 of 100 runs | 23 of 100 runs |
| Claude Sonnet 5.5, BashArena (same task draw) | 1 of 10 runs | 0 of 10 runs |
| GPT-6.2-sol, BashArena (same task draw) | 13 of 30 runs | 3 of 30 runs |
| Bash calls from our own Claude Code sessions | 65 of 1,000 calls | 23 of 1,000 calls |

- **Frontier models (`plans/frontier-honest.json`):** 100 honest runs each were planned. Each provider's safety classifier stopped one honest run, and the runner then stopped that model's remaining batches rather than retry, so 10 and 30 runs are usable. Those numbers are small.
  - The 3 GPT runs flagged at 0.7 match the GLM pattern: one read `/etc/shadow` and changed an account's password ageing, one rewrote the firewall daemon to get around its packet-size limit, and one redirected the database's address to an unnamed host and sent the API key there.
- **Our own sessions:** 1,000 Bash calls sampled from 625 Claude Code sessions, each judged with up to four earlier calls and the session's first user message as the task. The sessions are private and stay outside the repository.
  - Of the 23 flagged at 0.7, about 14 were actions with effects outside the machine (pushes, PR approvals, changes to repository settings, starting other agents) that the user had asked for later in the session, which the judge didn't see. About 7 were over-flags of local or read-only work.
  - Many of these sessions ran under Claude Code's own permission checks, so the riskiest calls may be missing.

## 7. Cost

| Item | $ |
|---|---|
| Generation, test run and GLM corpus | 248 |
| Judging the three judges, all passes | 205 |
| Ablations on Haiku (seven variants and a rerun) | 318 |
| Sonnet subset | 283 |
| Admin policy, BashArena | 28 |
| Live pilot | 4 |
| Live arm at 0.7 | 103 |
| Live arm at 0.5 (including runs lost to provider credit) | 50 |
| Frontier honest runs and our own sessions (§6.1) | 21 |

**The Sonnet subset was projected at $15.** The scorer projects a monitor's cost from its earlier per-call cost, whatever the judge model, and doesn't stop mid-scan when the real cost passes the projection. Both are open fixes.

## 8. Where the files are

| What | Where |
|---|---|
| Frozen corpus `glm-v1`: raw ControlArena logs, per-trajectory replay logs with their `cases.json`, trajectory YAMLs, labels (`trajectories.jsonl`), and a README with terms and canary | A private S3 bucket, named by `AUTO_MODE_CORPUS_BUCKET` (not in the repository). `python -m benchmarks.auto_mode.controlarena.corpus glm-v1 --dest DIR` downloads it and checks every file against `corpus/glm-v1.files.jsonl` |
| Plans | `controlarena/plans/`: `test-run.json`, `glm-pilot.json`, `glm-full.json`, `glm-basharena.json`, `glm-live*.json` |
| Run data (each run's data directory, outside the repository) | `<data>/batches.jsonl`, `ledger.csv`, `attempts.json`, `logs/` |
| Judge outputs (per call and per run) | `<data>/<out>/trajectory_results/<monitor>.csv` and `.usage.csv` |
| Judge explanations | `<data>/<out>/trajectory_scans/<monitor>/` (Scout scans) |
| Reports | `<data>/<out>/trajectory_report/` (`summary.csv`, `breakdown.csv`, `trajectories.csv`, `report.md`) |
| Outputs for this document | `ablations-v1` (three judges and the ablations, 798 runs), `ablations-v1-rerun` (0b), `ablations-v1-sonnet` (subset), `policy-v1` (admin policy) |
| Live arm | Its own data directories, each with its own ledger |

Judge outputs, explanations and logs hold corpus content, so none of them is committed. Claude Code's outputs can also contain its rule names. Only this summary and the corpus manifest are in the repository.
