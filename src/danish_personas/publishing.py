"""Direct publication of generated persona datasets."""

from pathlib import Path

from huggingface_hub import HfApi


def upload_dataset(*, repo_id: str, parquet_path: Path) -> None:
    """Upload generated Parquet data to a Hugging Face dataset repository.

    Authentication is delegated to ``huggingface_hub``. Tokens are never accepted as
    arguments.

    Args:
        repo_id:
            Hugging Face dataset repository identifier.
        parquet_path:
            Generated dataset file.

    Raises:
        ValueError:
            If the repository identifier is blank or the dataset file is missing.
    """
    if not repo_id.strip():
        raise ValueError("Hugging Face repository identifier must not be blank")
    if not parquet_path.is_file() or parquet_path.is_symlink():
        raise ValueError("Generated dataset must be a regular file")
    HfApi().upload_file(
        path_or_fileobj=str(parquet_path),
        path_in_repo="data/train-00000-of-00001.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        create_pr=True,
    )
