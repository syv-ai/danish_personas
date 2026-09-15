"""Rebuild the committed raw source archive deterministically."""

import logging
import tarfile
from compression.zstd import CompressionParameter
from pathlib import Path

import click

from danish_personas.io import sha256_file
from danish_personas.sources.restore import DEFAULT_ARCHIVE, RAW_DIRECTORY

ARCHIVE_LEVEL = 19
ARCHIVE_MODE = 0o644


@click.command()
@click.option(
    "--raw-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_ARCHIVE.parent / RAW_DIRECTORY,
    show_default=True,
)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def main(raw_dir: Path, output: Path | None) -> None:
    """Pack restored snapshots into a byte-stable Zstandard archive.

    Every member is stored with a sorted name, fixed mode, zero ownership, and zero
    timestamp, so repacking unchanged snapshots reproduces the same archive.

    Raises:
        click.ClickException:
            If the raw directory holds no snapshot files.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = output if output is not None else raw_dir.with_suffix(".tar.zst")
    files = sorted(path for path in raw_dir.rglob("*") if path.is_file())
    if not files:
        message = f"No snapshot files under {raw_dir}"
        raise click.ClickException(message)
    options: dict[int, int] = {CompressionParameter.compression_level: ARCHIVE_LEVEL}
    with tarfile.open(output, "w:zst", options=options) as archive:
        for path in files:
            info = archive.gettarinfo(
                path, arcname=str(Path(raw_dir.name) / path.relative_to(raw_dir))
            )
            info.mode = ARCHIVE_MODE
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            with path.open("rb") as handle:
                archive.addfile(info, handle)
    logging.info("Packed %s files into %s", len(files), output)
    logging.info("Archive SHA-256: %s", sha256_file(output))


if __name__ == "__main__":
    main()
