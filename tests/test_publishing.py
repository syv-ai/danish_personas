"""Tests for direct Hugging Face dataset publication."""

from pathlib import Path

import pytest

from danish_personas import publishing


def test_upload_dataset_rejects_missing_input(tmp_path: Path) -> None:
    """A missing generated dataset fails before a network client is created."""
    with pytest.raises(ValueError, match="regular file"):
        publishing.upload_dataset(
            repo_id="owner/dataset", parquet_path=tmp_path / "missing.parquet"
        )


def test_upload_dataset_uses_standard_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The uploader accepts no token and uploads only the generated Parquet file."""
    parquet_path = tmp_path / "generated-personas.parquet"
    parquet_path.write_bytes(b"parquet")
    calls: list[dict[str, object]] = []

    class FakeApi:
        def upload_file(self, **kwargs: object) -> None:
            calls.append(kwargs)

    monkeypatch.setattr(publishing, "HfApi", FakeApi)

    publishing.upload_dataset(repo_id="owner/dataset", parquet_path=parquet_path)

    assert calls == [
        {
            "path_or_fileobj": str(parquet_path),
            "path_in_repo": "data/train-00000-of-00001.parquet",
            "repo_id": "owner/dataset",
            "repo_type": "dataset",
            "create_pr": True,
        }
    ]
