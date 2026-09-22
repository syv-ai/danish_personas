"""Compatibility setup for Hydra's command-line parser."""

import argparse
import collections.abc as c
import sys
import typing as t

_PATCH_MARKER = "_danish_personas_hydra_help_patch"


def enable_hydra_cli() -> None:
    """Allow Hydra's lazy completion help object on Python 3.14.

    Python 3.14 validates argument help eagerly, while Hydra deliberately supplies a
    lazy non-string object for shell-completion help. Convert only such objects to the
    representation Hydra already intends to render.
    """
    if sys.version_info < (3, 14) or getattr(
        argparse.ArgumentParser, _PATCH_MARKER, False
    ):
        return
    original_check_help = t.cast(
        c.Callable[[argparse.ArgumentParser, argparse.Action], None],
        getattr(argparse.ArgumentParser, "_check_help"),
    )

    def check_help(parser: argparse.ArgumentParser, action: argparse.Action) -> None:
        if action.help is not None and not isinstance(action.help, str):
            action.help = repr(action.help)
        original_check_help(parser, action)

    setattr(argparse.ArgumentParser, "_check_help", check_help)
    setattr(argparse.ArgumentParser, _PATCH_MARKER, True)
