"""Contracts for executable-script environment bootstrapping."""

import ast
import os
import subprocess
from pathlib import Path

import pytest

from danish_personas.environment import load_repository_environment

ROOT = Path(__file__).parents[2]
SCRIPTS = tuple(sorted((ROOT / "src/scripts").glob("*.py")))


@pytest.mark.parametrize("script_path", SCRIPTS, ids=lambda path: path.name)
def test_every_script_bootstraps_repository_environment(script_path: Path) -> None:
    """Each public script loads the environment immediately before its CLI."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    main_guards = [
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "__main__"
    ]

    assert len(main_guards) == 1
    guard_body = main_guards[0].body
    assert len(guard_body) == 2
    bootstrap, entry_point = guard_body
    assert isinstance(bootstrap, ast.Expr)
    assert isinstance(bootstrap.value, ast.Call)
    assert isinstance(bootstrap.value.func, ast.Name)
    assert bootstrap.value.func.id == "load_repository_environment"
    assert isinstance(entry_point, ast.Expr)
    assert isinstance(entry_point.value, ast.Call)
    assert isinstance(entry_point.value.func, ast.Name)
    assert entry_point.value.func.id in {"fix_dot_env_file", "main"}


def test_importing_scripts_does_not_load_repository_environment() -> None:
    """Importing a script does not read dotenv values into the process."""
    marker = "DANISH_PERSONAS_IMPORT_SENTINEL"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import importlib
import os

import danish_personas.environment as environment

marker = "DANISH_PERSONAS_IMPORT_SENTINEL"
os.environ[marker] = "from-process"


def unexpected_load(*args: object, **kwargs: object) -> None:
    os.environ[marker] = "from-dotenv"
    raise AssertionError("script import loaded the repository environment")


environment.load_dotenv = unexpected_load
for module in (
    "scripts.fix_dot_env_file",
    "scripts.generate_persona",
    "scripts.build_dataset",
    "scripts.build_persona_dashboard",
):
    importlib.import_module(module)
assert os.environ[marker] == "from-process"
""",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert marker not in result.stderr


@pytest.mark.parametrize("script_path", SCRIPTS, ids=lambda path: path.name)
def test_direct_script_execution_bootstraps_environment(
    script_path: Path, tmp_path: Path
) -> None:
    """Direct CLI execution loads dotenv values before parsing command options."""
    marker = "DANISH_PERSONAS_DIRECT_BOOTSTRAP"
    env_path = tmp_path / ".env"
    env_path.write_text(f"{marker}=from-dotenv\n", encoding="utf-8")
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            """
import os
import runpy
import sys
from pathlib import Path

import danish_personas.environment as environment

env_path = Path(sys.argv[1])
script_path = sys.argv[2]
load_environment = environment.load_repository_environment


def bootstrap() -> None:
    load_environment(env_path=env_path)


environment.load_repository_environment = bootstrap
sys.argv = [script_path, "--help"]
try:
    runpy.run_path(script_path, run_name="__main__")
except SystemExit as error:
    if error.code not in (None, 0):
        raise
assert os.environ["DANISH_PERSONAS_DIRECT_BOOTSTRAP"] == "from-dotenv"
""",
            str(env_path),
            str(script_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={key: value for key, value in os.environ.items() if key != marker},
    )

    assert result.returncode == 0, result.stderr


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
