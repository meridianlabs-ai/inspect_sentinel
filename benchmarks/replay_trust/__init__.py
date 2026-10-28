"""Showing replay can be trusted: production-shaped monitors run live twice and replayed twice over scripted haystacks with planted needles, comparing their prompts, verdicts and needles found.

Internal: our own validation of replay, not part of the package. Run it with `python -m benchmarks.replay_trust haystacks` from the repository root; see `README.md` here. It is not run in CI, since with a real monitor model it spends money; `tests/test_replay_trust_benchmark.py` runs the whole pipeline on deterministic models instead.
"""
