"""Offline subprocess checks for the public Hydra script interface."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    "script", ["src/scripts/generate_persona.py", "src/scripts/build_dataset.py"]
)
def test_hydra_scripts_expose_help(script: str) -> None:
    """Hydra scripts parse help without preparing data or contacting a provider."""
    result = subprocess.run(
        ["uv", "run", script, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "powered by Hydra" in result.stdout


@pytest.mark.parametrize(
    ("script", "override", "setting"),
    [
        (
            "src/scripts/generate_persona.py",
            "generate_persona.output_dir=null",
            "generate_persona.output_dir",
        ),
        (
            "src/scripts/build_dataset.py",
            "build_dataset.rows=null",
            "build_dataset.rows",
        ),
    ],
)
def test_hydra_scripts_reject_missing_required_settings(
    script: str, override: str, setting: str
) -> None:
    """Invalid required settings fail before preparation or provider calls."""
    result = subprocess.run(
        ["uv", "run", script, override],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert setting.rsplit(".", maxsplit=1)[-1] in f"{result.stdout}\n{result.stderr}"
