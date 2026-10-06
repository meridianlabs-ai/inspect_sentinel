"""Sandbox probes (M5): what a guest can and cannot reach without the host."""

import json
from typing import Any

# Imported at module level so componentize-py bundles them: a module first
# imported at run time is not in the snapshot (ModuleNotFoundError), which
# would hide what WASI itself permits.
_IMPORTS: dict[str, Any] = {}
for _name in (
    "socket",
    "subprocess",
    "threading",
    "urllib.request",
    "ssl",
    "asyncio",
    "multiprocessing",
):
    try:
        _IMPORTS[_name] = __import__(_name, fromlist=["_"])
    except BaseException as _ex:
        _IMPORTS[_name] = f"{type(_ex).__name__}: {_ex}"


def _try(name: str, fn: Any) -> dict[str, Any]:
    try:
        return {"probe": name, "ok": True, "result": repr(fn())[:200]}
    except BaseException as ex:
        return {"probe": name, "ok": False, "error": f"{type(ex).__name__}: {ex}"[:300]}


def _socket() -> Any:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("93.184.216.34", 80))
    return s


def _subprocess() -> Any:
    import subprocess

    return subprocess.run(["/bin/sh", "-c", "echo hi"], capture_output=True)


def _urllib() -> Any:
    import urllib.request

    return urllib.request.urlopen("http://example.com", timeout=2).status


def _thread() -> Any:
    import threading

    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return "started"


def _open_write() -> Any:
    with open("/tmp/sentinel-probe.txt", "w") as f:
        f.write("x")
    return "written"


def run(host: Any) -> list[dict[str, Any]]:
    import os

    return [
        {
            "probe": "build-time imports",
            "ok": True,
            "result": {
                k: (v if isinstance(v, str) else "imported")
                for k, v in _IMPORTS.items()
            },
        },
        _try("open('/etc/passwd')", lambda: open("/etc/passwd").read(64)),
        _try("open('/tmp/x', 'w')", _open_write),
        _try("os.listdir('/')", lambda: os.listdir("/")),
        _try("os.getcwd()", os.getcwd),
        _try("os.environ", lambda: dict(os.environ)),
        _try("os.environ['OPENAI_API_KEY']", lambda: os.environ["OPENAI_API_KEY"]),
        _try("socket.connect", _socket),
        _try("urllib.urlopen", _urllib),
        _try("subprocess.run", _subprocess),
        _try("threading.Thread", _thread),
        _try("os.system", lambda: os.system("echo hi")),
        _try(
            "host.generate (mediated)",
            lambda: json.loads(host.generate(json.dumps({"input": "ping"})))["choices"][
                0
            ]["message"]["content"],
        ),
        _try(
            "host.fetch('allowlist')",
            lambda: host.fetch("allowlist", json.dumps({"domain": "example.com"})),
        ),
        _try(
            "host.fetch('http://evil.example')",
            lambda: host.fetch("http://evil.example", "{}"),
        ),
    ]
