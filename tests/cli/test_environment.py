"""Contracts for executable-script environment bootstrapping."""

import ast
import os
from pathlib import Path

import pytest

from danish_personas.environment import load_repository_environment

ROOT = Path(__file__).parents[2]
SCRIPTS = tuple(sorted((ROOT / "src/scripts").glob("*.py")))


@pytest.mark.parametrize("script_path", SCRIPTS, ids=lambda path: path.name)
def test_every_script_bootstraps_repository_environment(script_path: Path) -> None:
    """Each public script loads the environment before its CLI setup."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    bootstrap_indices = [
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load_repository_environment"
    ]

    assert len(bootstrap_indices) == 1
    bootstrap_index = bootstrap_indices[0]
    if script_path.name == "fix_dot_env_file.py":
        assert bootstrap_index < next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.FunctionDef)
        )
    else:
        setup_index = next(
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "enable_hydra_cli"
        )
        assert bootstrap_index < setup_index


def test_load_repository_environment_ignores_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing dotenv file does not affect the process environment."""
    monkeypatch.setenv("EXISTING_VALUE", "unchanged")

    load_repository_environment(env_path=tmp_path / ".env")

    assert os.environ["EXISTING_VALUE"] == "unchanged"


def test_load_repository_environment_loads_all_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every assignment in the dotenv file is loaded into the process."""
    env_path = tmp_path / ".env"
    env_path.write_text("FIRST_VALUE=one\nSECOND_VALUE=two\n", encoding="utf-8")
    monkeypatch.delenv("FIRST_VALUE", raising=False)
    monkeypatch.delenv("SECOND_VALUE", raising=False)

    load_repository_environment(env_path=env_path)

    assert os.environ["FIRST_VALUE"] == "one"
    assert os.environ["SECOND_VALUE"] == "two"


def test_load_repository_environment_preserves_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Values already present in the process are never overridden."""
    env_path = tmp_path / ".env"
    env_path.write_text("EXISTING_VALUE=from-file\n", encoding="utf-8")
    monkeypatch.setenv("EXISTING_VALUE", "from-process")

    load_repository_environment(env_path=env_path)

    assert os.environ["EXISTING_VALUE"] == "from-process"
