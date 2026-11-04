# pyproject.toml holds back releases younger than seven days, or six hours for
# a list of model provider SDKs and Inspect packages. Dependabot applies the
# seven days to its uv.lock bumps and leaves that list to uv. Every CI job
# installs from uv.lock, except that inspect-ai and inspect-scout move to their
# main's latest commit first. Testing against the newest releases happens in
# meridianlabs-ai/canaries, outside this repository.

import importlib.util
import pathlib
import re
import tomllib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import yaml

REPO = pathlib.Path(__file__).parents[1]
GITHUB = REPO / ".github"
WORKFLOWS = GITHUB / "workflows"
UV_SYNC_ACTION = "./.github/actions/uv-sync"
SHARED_ACTIONS = "meridianlabs-ai/actions/"
RELEASE_BRANCH = "release-please--branches--main"
MATRIX_PYTHON = "${{ matrix.python-version }}"
LOCKED_JOBS = {
    "py-pyright": {"python-version": MATRIX_PYTHON, "extras": "dev"},
    "py-test": {"python-version": MATRIX_PYTHON, "extras": "dev"},
    "litellm-guardrail": {"python-version": MATRIX_PYTHON, "extras": "dev"},
    "py-build": {"python-version": MATRIX_PYTHON, "project-install": "skip"},
}
EXEMPT_COOLDOWN = "6 hours"
# CI moves these to their main, so Dependabot leaves them alone.
MAIN_TRACKED = ["inspect-ai", "inspect-scout"]
# The LiteLLM guardrail is checked in an environment of its own, with the wheel
# just built and a pinned LiteLLM.
GUARDRAIL_IMPORT = (
    "build.yaml:py-build",
    'uv run --no-project --with dist-litellm/*.whl --with "litellm[proxy]==1.105.0"'
    " python -c \"import inspect_sentinel_litellm; print('Guardrail imported successfully')\"",
)

spec = importlib.util.spec_from_file_location(
    "release_floors", GITHUB / "scripts" / "release_floors.py"
)
assert spec and spec.loader
release_floors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_floors)


def pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def uv_config() -> dict[str, Any]:
    config: dict[str, Any] = pyproject()["tool"]["uv"]
    return config


def normalize(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package).lower()


def direct_dependencies() -> set[str]:
    project = pyproject()["project"]
    requirements = [
        *project["dependencies"],
        *(r for extra in project["optional-dependencies"].values() for r in extra),
    ]
    return {
        normalize(re.split(r"[^A-Za-z0-9._-]", requirement, maxsplit=1)[0])
        for requirement in requirements
    }


def cooldown_days(exclude_newer: str) -> int:
    days = re.fullmatch(r"(\d+) days", exclude_newer)
    assert days, exclude_newer
    return int(days[1])


def dependabot_uv() -> dict[str, Any]:
    config = yaml.safe_load((GITHUB / "dependabot.yml").read_text())
    root = [
        u
        for u in config["updates"]
        if u["package-ecosystem"] == "uv" and u["directory"] == "/"
    ]
    assert len(root) == 1, "expected one uv entry for directory / in dependabot.yml"
    entry: dict[str, Any] = root[0]
    return entry


def workflows() -> list[pathlib.Path]:
    return sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])


def local_actions() -> list[pathlib.Path]:
    return sorted((GITHUB / "actions").glob("*/action.y*ml"))


def run_lines(job: dict[str, Any]) -> list[str]:
    # Joins backslash continuations so a command's flags stay on one line.
    return [
        line.strip()
        for step in job.get("steps", [])
        for line in step.get("run", "").replace("\\\n", " ").splitlines()
        if line.strip()
    ]


def all_steps() -> list[tuple[str, dict[str, Any]]]:
    return [
        *(
            (f"{path.name}:{name}", step)
            for path in workflows()
            for name, job in yaml.safe_load(path.read_text())["jobs"].items()
            for step in job.get("steps", [])
        ),
        *(
            (path.parent.name, step)
            for path in local_actions()
            for step in yaml.safe_load(path.read_text())["runs"].get("steps", [])
        ),
    ]


def workflow_run_lines() -> list[tuple[str, str]]:
    return [
        *(
            (f"{path.name}:{name}", line)
            for path in workflows()
            for name, job in yaml.safe_load(path.read_text())["jobs"].items()
            for line in run_lines(job)
        ),
        *(
            (path.parent.name, line)
            for path in local_actions()
            for line in run_lines(yaml.safe_load(path.read_text())["runs"])
        ),
    ]


def verify_exclude_newer_package(
    config: dict[str, Any], direct: set[str], now: datetime
) -> None:
    # Each entry is "6 hours" or a dated entry younger than the cooldown.
    max_age = timedelta(days=cooldown_days(config["exclude-newer"]))
    for package, cooldown in config["exclude-newer-package"].items():
        assert normalize(package) in direct, f"{package} is not a direct dependency"
        if cooldown != EXEMPT_COOLDOWN:
            age = now - datetime.fromisoformat(cooldown)
            assert timedelta(0) <= age < max_age, (
                f'dated entry {package} = "{cooldown}" is in the future or stale'
            )


def verify_dependabot_cooldown(
    config: dict[str, Any], dependabot: dict[str, Any]
) -> None:
    cooldown = dependabot["cooldown"]
    assert cooldown["default-days"] == cooldown_days(config["exclude-newer"])
    assert dependabot["ignore"] == [{"dependency-name": p} for p in MAIN_TRACKED]
    # Dependabot's cooldown counts whole days, so it leaves the "6 hours"
    # entries to uv. Dated entries are locked by hand.
    exempt = sorted(
        package
        for package, cooldown in config["exclude-newer-package"].items()
        if cooldown == EXEMPT_COOLDOWN and package not in MAIN_TRACKED
    )
    assert sorted(cooldown["exclude"]) == exempt
    # The exempt packages have their own group, listed before the catch-all.
    groups = dependabot["groups"]
    assert list(groups) == ["six-hour-cooldown", "python-packages"]
    assert sorted(groups["six-hour-cooldown"]["patterns"]) == exempt
    assert groups["python-packages"]["patterns"] == ["*"]


def test_exclude_newer_package_entries() -> None:
    verify_exclude_newer_package(uv_config(), direct_dependencies(), datetime.now(UTC))


def test_dependabot_cooldown_matches_pyproject() -> None:
    verify_dependabot_cooldown(uv_config(), dependabot_uv())


def test_fresh_dated_entry_passes() -> None:
    config = uv_config()
    now = datetime.now(UTC)
    config["exclude-newer-package"]["jinja2"] = (now - timedelta(days=6)).isoformat()
    verify_exclude_newer_package(config, direct_dependencies(), now)
    verify_dependabot_cooldown(config, dependabot_uv())


@pytest.mark.parametrize(
    "package,cooldown,error",
    [
        ("certifi", EXEMPT_COOLDOWN, "not a direct dependency"),
        ("jinja2", timedelta(days=-7), "dated entry"),
        ("jinja2", timedelta(days=1), "dated entry"),
    ],
)
def test_bad_exclude_newer_package_entry_is_caught(
    package: str, cooldown: str | timedelta, error: str
) -> None:
    config = uv_config()
    now = datetime.now(UTC)
    config["exclude-newer-package"][package] = (
        (now + cooldown).isoformat() if isinstance(cooldown, timedelta) else cooldown
    )
    with pytest.raises(AssertionError, match=error):
        verify_exclude_newer_package(config, direct_dependencies(), now)


def test_other_cooldown_value_is_caught() -> None:
    config = uv_config()
    config["exclude-newer-package"]["jinja2"] = "1 day"
    with pytest.raises(ValueError):
        verify_exclude_newer_package(config, direct_dependencies(), datetime.now(UTC))


def test_uv_comes_from_the_shared_setup_uv() -> None:
    # meridianlabs-ai/actions/setup-uv holds the uv pin. One commit for every
    # shared action means one uv.
    uses = [(where, step["uses"]) for where, step in all_steps() if "uses" in step]
    assert [u for u in uses if u[1].startswith("astral-sh/setup-uv@")] == []
    refs = {
        ref.split("@")[1].split()[0]
        for _, ref in uses
        if ref.startswith(
            (
                SHARED_ACTIONS + "setup-uv@",
                SHARED_ACTIONS + "install-python-dependencies@",
            )
        )
    }
    assert len(refs) == 1, refs
    assert re.fullmatch(r"[0-9a-f]{40}", refs.pop())


def test_ruff_action_uses_the_locked_ruff() -> None:
    # Without a version, ruff-action installs the newest ruff release.
    steps = [
        step
        for _, step in all_steps()
        if step.get("uses", "").startswith("astral-sh/ruff-action@")
    ]
    assert steps
    for step in steps:
        assert step["with"]["version-file"] == "uv.lock", step


@pytest.mark.parametrize("name,inputs", LOCKED_JOBS.items())
def test_locked_jobs_install_from_the_lockfile(
    name: str, inputs: dict[str, str]
) -> None:
    job = yaml.safe_load((WORKFLOWS / "build.yaml").read_text())["jobs"][name]
    installs = [
        step.get("with", {})
        for step in job["steps"]
        if step.get("uses") == UV_SYNC_ACTION
    ]
    assert installs == [inputs]
    assert not [line for line in run_lines(job) if line.startswith("uv sync")]


def test_uv_sync_action_moves_only_the_main_tracked_packages() -> None:
    # Moving them re-locks, which would also absorb a pyproject.toml change
    # nobody locked, so the committed lock is checked first. The release PR
    # resolves without sources, and later steps see the committed uv.lock.
    action = yaml.safe_load((GITHUB / "actions" / "uv-sync" / "action.yml").read_text())
    setup, move, install, restore = action["runs"]["steps"]
    assert setup["uses"].startswith(SHARED_ACTIONS + "setup-uv@")
    assert setup["with"] == {"python-version": "${{ inputs.python-version }}"}
    assert move["env"]["HEAD_REF"] == "${{ github.head_ref }}"
    assert run_lines({"steps": [move]}) == [
        "uv lock --check",
        f'if [ "$HEAD_REF" = "{RELEASE_BRANCH}" ]; then',
        "export UV_NO_SOURCES=1",
        'echo "UV_NO_SOURCES=1" >> "$GITHUB_ENV"',
        "fi",
        "uv lock --upgrade-package inspect-ai --upgrade-package inspect-scout",
    ]
    assert install["uses"].startswith(SHARED_ACTIONS + "install-python-dependencies@")
    assert install["with"] == {
        "versions": "locked",
        "python-version": "${{ inputs.python-version }}",
        "extras": "${{ inputs.extras }}",
        "project-install": "${{ inputs.project-install }}",
    }
    assert run_lines({"steps": [restore]}) == ["git checkout -- uv.lock"]


def test_release_branch_name_matches_everywhere() -> None:
    # A misspelt name silently runs the release PR as a dev PR.
    names = {
        where: set(re.findall(r"release-please--branches--[\w-]+", text))
        for where, text in [
            *[(path.name, path.read_text()) for path in workflows()],
            ("uv-sync", (GITHUB / "actions" / "uv-sync" / "action.yml").read_text()),
        ]
    }
    assert set().union(*names.values()) == {RELEASE_BRANCH}
    assert names["uv-sync"] and names["release-pin-deps.yml"]


# A bare `uv run` or `uv sync` re-resolves an out-of-date uv.lock and installs
# the result. `uv run --no-project` ignores the project and uv.lock. `--with`
# installs unlocked packages whatever the other flags.
def test_uv_never_relocks() -> None:
    unlocked = [
        (where, line)
        for where, line in workflow_run_lines()
        if re.search(r"\buv (run|sync)\b", line)
        and (
            "--with" in line
            or not {"--locked", "--no-sync", "--no-project"} & set(line.split())
        )
    ]
    assert unlocked == [GUARDRAIL_IMPORT]


# pip installs skip uv.lock, so they may only add local packages without their
# dependencies, or install a hashed lock (`-r` or `uv pip sync`).
def test_pip_installs_are_dependency_free_or_hashed() -> None:
    unlocked = [
        (where, line)
        for where, line in workflow_run_lines()
        if (re.search(r"\bpip3? install\b", line) and "--no-deps" not in line.split())
        or (
            re.search(r"\bpip sync\b|\bpip3? install\b.* -r\b", line)
            and "--require-hashes" not in line.split()
        )
    ]
    assert unlocked == []


def test_no_workflow_opts_out_of_the_cooldown() -> None:
    floating = [
        (where, line)
        for where, line in workflow_run_lines()
        if re.search(r"--exclude-newer(=|\s+)false", line)
    ]
    assert floating == []


def test_floor_script_raises_the_main_tracked_floors() -> None:
    # push-floor rewrites each "<package>>=<version>" in pyproject.toml.
    assert list(release_floors.PACKAGES) == MAIN_TRACKED
    dependencies = pyproject()["project"]["dependencies"]
    for package in MAIN_TRACKED:
        assert len([d for d in dependencies if d.startswith(f"{package}>=")]) == 1


def test_floor_script_requires_a_released_package() -> None:
    lock = """
[[package]]
name = "inspect-scout"
version = "0.6.3.dev20"
source = { git = "https://github.com/meridianlabs-ai/inspect_scout?branch=main#abc" }
"""
    with pytest.raises(ValueError, match="not locked from PyPI"):
        release_floors.locked_version(lock, "inspect-scout")
    released = lock.replace('version = "0.6.3.dev20"', 'version = "0.6.2"').replace(
        'git = "https://github.com/meridianlabs-ai/inspect_scout?branch=main#abc"',
        'registry = "https://pypi.org/simple"',
    )
    assert release_floors.locked_version(released, "inspect-scout") == "0.6.2"
