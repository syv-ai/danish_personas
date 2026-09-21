"""Verified release upload service for Hugging Face datasets."""

from pathlib import Path

from huggingface_hub import HfApi


def upload_release(*, repo_id: str, release_dir: Path) -> None:
    """Upload one already verified release directory to a dataset repository.

    Authentication is deliberately delegated to ``huggingface_hub``. Tokens are not
    accepted as arguments and are never included in logs or return values.

    Args:
        repo_id:
            Hugging Face dataset repository identifier.
        release_dir:
            Release directory that has passed local verification.

    Raises:
        ValueError:
            If the repository identifier is blank or the release directory is not a
            regular directory.
    """
    if not repo_id.strip():
        raise ValueError("Hugging Face repository identifier must not be blank")
    if not release_dir.is_dir() or release_dir.is_symlink():
        raise ValueError("Verified release directory must be a regular directory")
    HfApi().upload_folder(
        folder_path=str(release_dir),
        repo_id=repo_id,
        repo_type="dataset",
        create_pr=True,
    )
