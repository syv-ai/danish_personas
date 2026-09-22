"""Load repository-local environment variables for executable scripts."""

from pathlib import Path

from dotenv import load_dotenv

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ENV_PATH = REPOSITORY_ROOT / ".env"


def load_repository_environment(env_path: Path = REPOSITORY_ENV_PATH) -> None:
    """Load all variables from a repository-root ``.env`` file.

    Existing process environment variables take precedence. A missing file is ignored
    by :func:`dotenv.load_dotenv`.

    Args:
        env_path (optional):
            Environment file to load. Defaults to the repository-root ``.env`` file.
    """
    load_dotenv(dotenv_path=env_path, override=False)
