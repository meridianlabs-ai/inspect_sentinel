import subprocess
import sys


def _modules_after(statement: str) -> set[str]:
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys\n{statement}\nprint('\\n'.join(sys.modules))",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return set(done.stdout.split())


def test_core_does_not_load_the_sidecar() -> None:
    loaded = _modules_after("import inspect_sentinel")

    assert not {name for name in loaded if name.startswith("inspect_sentinel.sidecar")}


def test_sidecar_loads_nothing_the_core_does_not() -> None:
    core = _modules_after("import inspect_sentinel")
    added = _modules_after("import inspect_sentinel.sidecar") - core

    assert added
    # only this package's own modules. A provider's SDK, and inspect_ai's
    # private modules, are loaded when they are first needed.
    assert not {name for name in added if not name.startswith("inspect_sentinel.")}
