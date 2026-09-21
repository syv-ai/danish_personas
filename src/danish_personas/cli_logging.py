"""Logging configuration shared by the standalone command scripts."""

import logging
import sys

_SUPPRESSED_LOGGERS = ("httpx", "httpcore", "huggingface_hub")


def configure_cli_logging() -> None:
    """Configure concise command diagnostics on stderr.

    Standalone commands may be invoked repeatedly by an embedding process or test
    runner, so the handler is refreshed on every invocation to target the current
    stderr stream. Third-party HTTP and upload clients remain quiet unless they
    emit warnings or errors.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    for logger_name in _SUPPRESSED_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
