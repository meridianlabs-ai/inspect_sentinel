"""Print the inspect-ai and inspect-scout releases the release PR is tested against.

The release PR's CI resolves without `[tool.uv.sources]`, so it installs the
newest inspect-ai and inspect-scout releases that clear their cooldown
(.github/actions/uv-sync). This locks them the same way, prints each as
`<package>=<version>`, and puts uv.lock back. release-pin-deps.yml runs this in
a job that holds no write token, and raises the `>=` floors to the printed
versions.

Usage:
    python3 -I .github/scripts/release_floors.py
"""

import subprocess
import tomllib
from pathlib import Path

LOCK = Path("uv.lock")
PACKAGES = ("inspect-ai", "inspect-scout")


def locked_version(lock: str, name: str) -> str:
    (package,) = [p for p in tomllib.loads(lock)["package"] if p["name"] == name]
    if "registry" not in package["source"]:
        raise ValueError(f"{name} is not locked from PyPI: {package['source']}")
    version: str = package["version"]
    return version


def main() -> None:
    lock = LOCK.read_text()
    upgrades = [arg for name in PACKAGES for arg in ("--upgrade-package", name)]
    try:
        subprocess.run(
            ["uv", "lock", "--no-sources", *upgrades],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        locked = LOCK.read_text()
        versions = [(name, locked_version(locked, name)) for name in PACKAGES]
    finally:
        LOCK.write_text(lock)
    for name, version in versions:
        print(f"{name}={version}")


if __name__ == "__main__":
    main()
