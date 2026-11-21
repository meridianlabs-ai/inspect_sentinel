"""The examples that come in two copies, one behind LiteLLM and one behind Envoy, keep their rule and their task the same."""

from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).parents[1] / "examples"


def _twins() -> list[tuple[Path, Path]]:
    twins: list[tuple[Path, Path]] = []
    for envoy in sorted(_EXAMPLES.glob("*_envoy")):
        name = envoy.name.removesuffix("_envoy")
        litellm = next(
            (
                _EXAMPLES / candidate
                for candidate in (f"{name}_litellm", name)
                if (_EXAMPLES / candidate).is_dir()
            ),
            None,
        )
        if litellm is not None:
            twins.append((litellm, envoy))
    return twins


def _shared_files(example: Path) -> dict[Path, str]:
    # the rule's code and the task; each copy's pyproject.toml names its own example
    files = [example / "task.py", *(example / "rule").rglob("*.py")]
    return {path.relative_to(example): path.read_text() for path in files}


def test_examples_have_twins() -> None:
    assert _twins()


@pytest.mark.parametrize(("litellm", "envoy"), _twins(), ids=lambda path: path.name)
def test_twin_examples_share_their_rule_and_task(litellm: Path, envoy: Path) -> None:
    assert _shared_files(litellm) == _shared_files(envoy)
