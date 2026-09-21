"""Helpers for binding tests to the committed origin-label contract."""

import typing as t
from pathlib import Path

from danish_personas.origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_CONTRACT_SHA256,
    load_origin_label_contract,
)


class OriginContractFields(t.TypedDict):
    """Current manifest origin-label binding fields."""

    origin_labels_contract_path: str
    origin_labels_contract_version: int
    origin_labels_contract_sha256: str
    origin_labels_contract_content: str


PROJECT_ROOT = Path(__file__).parents[2]
_ORIGIN_LABEL_CONTRACT_CONTENT = (
    PROJECT_ROOT / DEFAULT_ORIGIN_LABEL_CONTRACT_PATH
).read_text(encoding="utf-8")


def origin_contract_fields() -> OriginContractFields:
    """Return the exact binding for the committed origin-label contract."""
    contract = load_origin_label_contract()
    return {
        "origin_labels_contract_path": ORIGIN_LABEL_CONTRACT_PATH,
        "origin_labels_contract_version": contract.version,
        "origin_labels_contract_sha256": ORIGIN_LABEL_CONTRACT_SHA256,
        "origin_labels_contract_content": _ORIGIN_LABEL_CONTRACT_CONTENT,
    }
