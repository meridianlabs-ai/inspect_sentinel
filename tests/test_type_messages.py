import json
import subprocess
import sys
from pathlib import Path

# a type checker run on this source must say why each marked line is wrong:
# the protocol's type, the type it needed, and that Escalate is not a Verdict
_SOURCE = """
from inspect_sentinel import (
    BeforeToolCall,
    Context,
    Decision,
    Escalate,
    Monitor,
    Observation,
    Proceed,
    Protocol,
    Sentinels,
    Verdict,
    handle_escalation,
    monitor,
    protocol,
    sequential,
)


@monitor
def score() -> Monitor:
    async def check(context: Context, step: BeforeToolCall) -> Observation | None:
        return Observation.score(0.1)

    return check


@protocol
def unsure() -> Protocol[Escalate | Proceed]:
    async def decide(context: Context, step: BeforeToolCall) -> Escalate | Proceed:
        return Decision.escalate("unsure")

    return decide


def task(sentinel: Sentinels[Verdict | None]) -> None: ...


task(unsure())  # lone
task(sequential([unsure()]))  # chain
sequential([unsure()], then=score())  # monitor
task(sequential([unsure()], then=handle_escalation("terminate")))
task([score(), sequential([unsure()], then=handle_escalation("continue"))])
"""

_EXPECTED = {
    "lone": [
        '"Protocol[Escalate | Proceed]" is not assignable to "Protocol[Verdict | None]"',
        'Type "Escalate" is not assignable to type "Verdict | None"',
    ],
    "chain": [
        '"Protocol[Escalate | Proceed | None]" is not assignable to "Protocol[Verdict | None]"',
        'Type "Escalate" is not assignable to type "Verdict | None"',
    ],
    "monitor": [
        'Argument of type "Monitor" cannot be assigned to parameter "then"',
        '"Awaitable[Observation | None]" is not assignable to "Awaitable[Verdict]"',
    ],
}


def test_a_type_error_names_the_protocol_type_and_escalate(tmp_path: Path) -> None:
    source = tmp_path / "typed.py"
    source.write_text(_SOURCE)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--outputjson",
            "--pythonpath",
            sys.executable,
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmp_path,
    )
    report = json.loads(result.stdout)
    errors: dict[int, str] = {}
    for diagnostic in report["generalDiagnostics"]:
        if diagnostic["severity"] == "error":
            line = diagnostic["range"]["start"]["line"] + 1
            errors[line] = errors.get(line, "") + diagnostic["message"]
    marked = {
        number: line.rsplit("# ", 1)[1]
        for number, line in enumerate(_SOURCE.splitlines(), start=1)
        if line.endswith(tuple(f"# {mark}" for mark in _EXPECTED))
    }
    assert set(errors) == set(marked), errors
    for number, mark in marked.items():
        for text in _EXPECTED[mark]:
            assert text in errors[number], (mark, errors[number])
