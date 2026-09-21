"""Ensure that the local ``.env`` file contains Git identity values."""

import logging
from pathlib import Path

import click

from danish_personas.cli_logging import configure_cli_logging

LOGGER = logging.getLogger(__name__)
DESIRED_ENVIRONMENT_VARIABLES = {
    "GIT_NAME": "Enter your full name, to be shown in Git commits:\n> ",
    "GIT_EMAIL": "Enter your email, as registered on your Github account:\n> ",
}


@click.command()
@click.option(
    "--non-interactive",
    is_flag=True,
    default=False,
    help="If set, the script will not ask for user input.",
)
def fix_dot_env_file(non_interactive: bool) -> None:
    """Ensure that ``.env`` exists and contains the desired variables.

    Args:
        non_interactive:
            If set, leave missing values blank rather than asking for input.
    """
    configure_cli_logging()
    LOGGER.info("Preparing local environment files")
    env_path = Path(".env")
    name_and_email_path = Path(".name_and_email")
    env_path.touch(exist_ok=True)
    name_and_email_path.touch(exist_ok=True)

    env_file_lines = [
        line
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    name_and_email_file_lines = name_and_email_path.read_text(
        encoding="utf-8"
    ).splitlines()
    env_vars = _parse_environment_lines(env_file_lines)
    name_and_email_vars = _parse_environment_lines(name_and_email_file_lines)

    missing = [
        variable
        for variable in DESIRED_ENVIRONMENT_VARIABLES
        if variable not in env_vars
    ]
    with env_path.open("a", encoding="utf-8") as env_file:
        for variable in missing:
            value = name_and_email_vars.get(variable, "")
            if value == "" and not non_interactive:
                value = input(DESIRED_ENVIRONMENT_VARIABLES[variable])
            env_file.write(f"{variable}={value}\n")

    name_and_email_path.unlink()
    LOGGER.info("Local environment files are ready")


def _parse_environment_lines(lines: list[str]) -> dict[str, str]:
    """Parse non-comment environment assignments without truncating values."""
    return {
        key: value
        for line in lines
        if "=" in line
        for key, value in [line.split("=", maxsplit=1)]
    }


if __name__ == "__main__":
    fix_dot_env_file()
