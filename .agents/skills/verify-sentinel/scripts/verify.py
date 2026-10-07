"""Drive inspect_sentinel through `inspect eval` and keep the evidence.

  verify doctor
  verify run <feature>|all
  verify cleanup <run-dir>

`VERIFY_BASE=<checkout>` runs another checkout's `src/` and `examples/` (e.g. origin/main, for a baseline) with this checkout's .venv and harness.

Features: rule, yaml-config, llm-monitor, trajectory, composition. Evidence goes to `$VERIFY_OUT` (default `$TMPDIR/verify-sentinel`)/<run-id>/evidence; the sandbox scratch dir is <run-id>/work. `run` exits 1 if any check fails.
"""

import importlib.metadata
import inspect as pyinspect
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
VENV = REPO / ".venv" / "bin"
BASE = Path(os.environ.get("VERIFY_BASE", REPO)).resolve()
OUT = Path(
    os.environ.get("VERIFY_OUT", Path(tempfile.gettempdir()) / "verify-sentinel")
)

YAML_RULE = "sentinel:\n  name: no_network\n"


@dataclass(frozen=True)
class Ev:
    """One SentinelEvent or ToolEvent, flattened for checks."""

    type: str
    path: str = ""
    kind: str = ""
    status: str = ""
    factory: str = ""
    function: str | None = None
    suspicion: object = None
    action: str | None = None
    explanation: str | None = None
    error: str | None = None
    command: str = ""


@dataclass(frozen=True)
class Result:
    status: str
    config: object
    events: list[Ev]
    work: set[str]

    def at(self, path: str, kind: str = "decision") -> list[Ev]:
        return [
            e
            for e in self.events
            if e.type == "sentinel" and e.path == path and e.kind == kind
        ]

    def tool_errors(self) -> list[str | None]:
        return [e.error for e in self.events if e.type == "tool"]


Check = tuple[str, Callable[[Result], bool]]

COMMON: list[Check] = [
    ("eval log status is success", lambda r: r.status == "success"),
    (
        "no sentinel event has status error",
        lambda r: all(e.status != "error" for e in r.events if e.type == "sentinel"),
    ),
]

RULE: list[Check] = [
    (
        "root no_network decides continue, then reject",
        lambda r: [e.action for e in r.at("")] == ["continue", "reject"],
    ),
    (
        "agent sees the rule's message as the curl call's error",
        lambda r: "needs the network" in str(r.tool_errors()[-1]),
    ),
    (
        "allowed call ran, rejected call did not (work dir)",
        lambda r: r.work == {"allowed.txt"},
    ),
]

FEATURES: dict[str, tuple[str, list[Check]]] = {
    "rule": (
        "rule",
        [
            *RULE,
            (
                "log records sentinel config no_network",
                lambda r: "name='no_network'" in repr(r.config),
            ),
        ],
    ),
    "yaml-config": (
        "unwatched",
        [
            *RULE,
            (
                "log records the YAML config as no_network",
                lambda r: "name='no_network'" in repr(r.config),
            ),
        ],
    ),
    "llm-monitor": (
        "llm_monitor",
        [
            (
                "monitor observes 0.1, then 0.9",
                lambda r: (
                    [e.suspicion for e in r.at("suspicion", "observation")]
                    == [0.1, 0.9]
                ),
            ),
            (
                "threshold decides continue, then reject",
                lambda r: [e.action for e in r.at("")] == ["continue", "reject"],
            ),
            (
                "allowed call ran, rejected call did not (work dir)",
                lambda r: r.work == {"allowed.txt"},
            ),
        ],
    ),
    "trajectory": (
        "trajectory",
        [
            (
                "before() suspicion rises with stored failure count",
                lambda r: (
                    [
                        e.suspicion
                        for e in r.at("failure_count", "observation")
                        if e.function == "before"
                    ]
                    == [0.0, 0.2, 0.4]
                ),
            ),
            (
                "after() sees failed, failed, succeeded",
                lambda r: (
                    [
                        e.explanation
                        for e in r.at("failure_count", "observation")
                        if e.function == "after"
                    ]
                    == ["call failed", "call failed", "call succeeded"]
                ),
            ),
            (
                "observe_only acts on nothing",
                lambda r: not any(e.action == "reject" for e in r.events),
            ),
            ("all calls ran (work dir)", lambda r: r.work == {"allowed.txt"}),
        ],
    ),
    "composition": (
        "composition",
        [
            (
                "guard/protected rejects the /etc call with decide_final",
                lambda r: (
                    [e.action for e in r.at("guard/protected")]
                    == ["continue", "reject", "continue"]
                ),
            ),
            (
                "decide_final bypasses guard and root on that step",
                lambda r: (
                    [e.status for e in r.at("guard")][1] == "bypassed"
                    and [e.status for e in r.at("")][1] == "bypassed"
                ),
            ),
            (
                "guard/network rejects curl, and guard and root return it",
                lambda r: (
                    [e.action for e in r.at("guard")][-1] == "reject"
                    and [e.action for e in r.at("")][-1] == "reject"
                ),
            ),
            (
                "audit monitor reports on every step",
                lambda r: (
                    [e.status for e in r.at("audit", "observation")] == ["reported"] * 3
                ),
            ),
            (
                "only the allowed call ran (work dir)",
                lambda r: r.work == {"allowed.txt"},
            ),
        ],
    ),
}


def doctor() -> int:
    ok = True

    def line(good: bool, text: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"{'ok  ' if good else 'FAIL'} {text}")

    def git(repo: Path, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True
        ).stdout.strip()

    import inspect_ai

    import inspect_sentinel

    ia_src = Path(inspect_ai.__file__).resolve().parents[2]
    is_src = Path(inspect_sentinel.__file__).resolve().parents[2]
    line(Path(sys.executable).parent == VENV, f"python {sys.executable}")
    line(is_src == BASE, f"inspect_sentinel imported from {is_src}")
    print(
        f"     inspect_sentinel at {git(BASE, 'log', '-1', '--format=%h %D') or BASE}"
    )
    print(
        f"     inspect_ai {importlib.metadata.version('inspect_ai')} from {ia_src} ({git(ia_src, 'log', '-1', '--format=%h %D')})"
    )
    dirty = git(BASE, "status", "--porcelain", "--", "src", "examples")
    line(
        not dirty,
        "src/ and examples/ have no local changes" + (f":\n{dirty}" if dirty else ""),
    )

    from inspect_ai.core._imports import check_imports

    violations = check_imports("inspect_sentinel", allowed=["anyio", "exceptiongroup"])
    line(
        not violations,
        f"inspect_sentinel imports only inspect_ai.core {sorted({v.module for v in violations}) or ''}",
    )

    from inspect_ai._sentinel._dispatch import _Host

    from inspect_sentinel import Host

    want = set(pyinspect.signature(Host.generate).parameters)
    have = set(pyinspect.signature(_Host.generate).parameters)
    line(
        want == have,
        f"inspect_ai _Host.generate takes sentinel Host.generate's params (missing {sorted(want - have)}, extra {sorted(have - want)})",
    )
    line((VENV / "inspect").exists(), "inspect CLI in .venv")
    return 0 if ok else 1


def _env(work: Path | None = None) -> dict[str, str]:
    base = {"PYTHONPATH": str(BASE / "src")} if BASE != REPO else {}
    return {
        **os.environ,
        **base,
        "VERIFY_EXAMPLES": str(BASE / "examples"),
        **({"VERIFY_WORK": str(work)} if work else {}),
    }


def _read(log_path: Path, work: Path) -> Result:
    from inspect_ai.event import SentinelEvent, ToolEvent
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path), resolve_attachments=True)
    events = [
        Ev(
            "sentinel",
            e.path,
            e.kind,
            e.status,
            e.factory,
            e.function,
            e.suspicion,
            e.action,
            e.explanation,
            e.error,
        )
        if isinstance(e, SentinelEvent)
        else Ev(
            "tool",
            error=e.error.message if e.error else None,
            command=str(e.arguments.get("command", "")).replace(
                str(work), "$VERIFY_WORK"
            ),
        )
        for sample in log.samples or []
        for e in sample.events
        if isinstance(e, SentinelEvent | ToolEvent)
    ]
    status = log.status if not log.error else f"{log.status}: {log.error.message}"
    return Result(
        status, log.eval.config.sentinel, events, {p.name for p in work.iterdir()}
    )


def _row(e: Ev) -> str:
    if e.type == "tool":
        return f"tool     {e.command!r} -> error={e.error!r}"
    return f"sentinel {e.kind:<11} {e.status:<9} path={e.path!r} factory={e.factory} fn={e.function} suspicion={e.suspicion} action={e.action} explanation={e.explanation!r} error={e.error!r}"


def run(feature: str) -> int:
    assert feature in FEATURES, f"unknown feature {feature}; one of {list(FEATURES)}"
    task, checks = FEATURES[feature]
    label = "" if BASE == REPO else f"-base-{BASE.name}"
    run_dir = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}-{feature}{label}"
    evidence, work = run_dir / "evidence", run_dir / "work"
    evidence.mkdir(parents=True)
    work.mkdir()
    args = [
        str(VENV / "inspect"),
        "eval",
        f"{HERE / 'tasks.py'}@{task}",
        "--display",
        "plain",
        "--log-dir",
        str(evidence / "logs"),
        "--log-format",
        "json",
    ]
    if feature == "yaml-config":
        (evidence / "sentinel.yaml").write_text(YAML_RULE)
        args += ["--sentinel", str(evidence / "sentinel.yaml")]
    proc = subprocess.run(
        args, capture_output=True, text=True, env=_env(work), cwd=REPO
    )
    (evidence / "command.txt").write_text(
        f"sentinel src: {BASE / 'src'}\n"
        + " ".join(args)
        + f"\nexit={proc.returncode}\n"
    )
    (evidence / "stdout.txt").write_text(proc.stdout)
    (evidence / "stderr.txt").write_text(proc.stderr)
    logs = sorted((evidence / "logs").glob("*.json"))
    if not logs:
        # usually the task file failed to import (e.g. VERIFY_BASE's sentinel against this inspect_ai)
        print(f"FAIL inspect eval wrote no log (exit {proc.returncode}); stderr:")
        print("\n".join(proc.stderr.strip().splitlines()[-15:]))
        print(f"\nevidence: {evidence}")
        return 1
    result = _read(logs[-1], work)

    def passes(check: Check) -> bool:
        try:
            return check[1](result)
        except IndexError:
            return False

    verdicts = [(passes(c), c[0]) for c in [*COMMON, *checks]]
    report = [
        f"feature {feature}  task {task}  exit={proc.returncode}",
        f"log status: {result.status}",
        f"sentinel config: {result.config!r}",
        *map(_row, result.events),
        f"work files: {sorted(result.work)}",
        "",
        *(f"{'PASS' if ok else 'FAIL'} {name}" for ok, name in verdicts),
    ]
    (evidence / "report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\nevidence: {evidence}\nrun dir:  {run_dir}")
    return 0 if all(ok for ok, _ in verdicts) else 1


def cleanup(run_dir: str) -> int:
    work = Path(run_dir).resolve() / "work"
    assert work.parent.parent == OUT.resolve(), f"{run_dir} is not a run under {OUT}"
    shutil.rmtree(work, ignore_errors=True)
    print(f"removed {work}; kept {work.parent / 'evidence'}")
    return 0


COMMANDS: dict[str, Callable[[list[str]], int]] = {
    "doctor": lambda a: (
        subprocess.run([sys.executable, __file__, "_doctor"], env=_env()).returncode
    ),
    "_doctor": lambda a: doctor(),
    "run": lambda a: max(run(f) for f in FEATURES) if a[0] == "all" else run(a[0]),
    "cleanup": lambda a: cleanup(a[0]),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2:]))
