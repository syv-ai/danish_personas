"""Tests for the immutable official FOLK2 IELAND label contract."""

import collections.abc as c
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from danish_personas.origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256,
    ORIGIN_LABEL_SOURCE_METADATA_EN_SHA256,
    OriginLabelContract,
    _UniqueKeyLoader,
    bind_origin_labels,
    load_bound_origin_labels,
    load_origin_label_contract,
    origin_label_contract_sha256,
    source_metadata_sha256,
)
from danish_personas.sources.archive import restore_raw_sources

PROJECT_ROOT = Path(__file__).parents[2]
ARCHIVE_PATH = PROJECT_ROOT / "data" / "raw-hardened-20260919.tar.zst"


def test_binding_rejects_metadata_checksum_and_label_drift() -> None:
    """Bindings reject independent metadata checksum and partition tampering."""
    contract = load_origin_label_contract()
    metadata = {
        "id": "FOLK2",
        "variables": [
            {
                "id": "IELAND",
                "values": list(
                    {"id": code, "text": label}
                    for code, label in contract.ordered_labels
                ),
            }
        ],
    }

    with pytest.raises(ValueError, match="checksum"):
        bind_origin_labels(contract, metadata, metadata_sha256="0" * 64)
    metadata["variables"][0]["values"][0]["text"] = "Tampered"
    with pytest.raises(ValueError, match="match contract"):
        bind_origin_labels(contract, metadata)


def test_contract_checksum_is_stable_and_representative_labels_are_official() -> None:
    """The checked-in bytes and representative Danish labels remain stable."""
    first = origin_label_contract_sha256(DEFAULT_ORIGIN_LABEL_CONTRACT_PATH)
    second = origin_label_contract_sha256(DEFAULT_ORIGIN_LABEL_CONTRACT_PATH)
    contract = load_origin_label_contract()

    assert first == second
    assert len(contract.labels) == 241
    assert contract.ordered_labels[:3] == (
        ("5100", "Danmark"),
        ("5122", "Albanien"),
        ("5124", "Andorra"),
    )
    assert contract.labels["5103"] == "Statsløs"
    assert contract.labels["5999"] == "Uoplyst"


def test_contract_reconstructs_archived_folk2_metadata(tmp_path: Path) -> None:
    """Every committed label and its order must match the archived metadata."""
    output_dir = tmp_path / "data"
    restore_raw_sources(archive_path=ARCHIVE_PATH, output_dir=output_dir)
    metadata_path = next(
        (output_dir / "raw-hardened-20260919" / "folk2").glob("*/metadata-da.json")
    )

    contract = load_origin_label_contract()
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    variable = next(item for item in metadata["variables"] if item["id"] == "IELAND")
    expected = {item["id"]: item["text"] for item in variable["values"]}

    assert contract.labels == expected
    assert load_bound_origin_labels(metadata_path) == expected
    assert (
        source_metadata_sha256(metadata_path) == ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256
    )


def test_contract_rejects_duplicate_yaml_codes(tmp_path: Path) -> None:
    """Duplicate YAML keys cannot be collapsed before validation."""
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "version: 1\n"
        "table_id: FOLK2\n"
        "dimension: IELAND\n"
        "language: da\n"
        f"source_metadata_en_sha256: {ORIGIN_LABEL_SOURCE_METADATA_EN_SHA256}\n"
        f"source_metadata_da_sha256: {ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256}\n"
        "labels_en:\n  '5100': Denmark\n  '5100': Denmark\n"
        "labels_da:\n  '5100': Danmark\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate YAML key"):
        yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)


@pytest.mark.parametrize(
    "change",
    [
        lambda labels: (
            labels["labels_en"].pop("5100"),
            labels["labels_da"].pop("5100"),
        ),
        lambda labels: (
            labels["labels_en"].__setitem__("6000", "New country"),
            labels["labels_da"].__setitem__("6000", "Nyt land"),
        ),
        lambda labels: (
            labels["labels_en"].__setitem__("bad", "Malformed code"),
            labels["labels_da"].__setitem__("bad", "Misdannet kode"),
        ),
        lambda labels: labels["labels_da"].__setitem__("5100", ""),
        lambda labels: labels["labels_da"].__setitem__("5100", " Danmark"),
        lambda labels: labels["labels_da"].__setitem__("5100", "Danmark "),
        lambda labels: labels["labels_da"].__setitem__("5100", "e\u0301land"),
        lambda labels: labels["labels_da"].__setitem__("5122", "danmark"),
        lambda labels: labels["labels_da"].__setitem__("5100", None),
    ],
    ids=[
        "missing-code",
        "extra-code",
        "malformed-code",
        "blank-label",
        "leading-whitespace",
        "trailing-whitespace",
        "non-nfc-label",
        "duplicate-label",
        "null-label",
    ],
)
def test_contract_rejects_malformed_or_duplicate_labels(
    change: c.Callable[[dict[str, object]], object],
) -> None:
    """Cardinality, codes, whitespace, NFC, types, and uniqueness are guarded."""
    payload = load_origin_label_contract().model_dump()
    change(payload)

    with pytest.raises((ValidationError, ValueError)):
        OriginLabelContract.model_validate(payload)


def test_contract_rejects_missing_and_extra_fields() -> None:
    """The contract schema cannot silently evolve or lose identity fields."""
    payload = load_origin_label_contract().model_dump()
    payload.pop("labels_da")
    with pytest.raises(ValidationError):
        OriginLabelContract.model_validate(payload)

    payload = load_origin_label_contract().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        OriginLabelContract.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 2),
        ("table_id", "OTHER"),
        ("dimension", "OTHER"),
        ("language", "en"),
        ("source_metadata_da_sha256", "0" * 64),
    ],
    ids=["version", "table-id", "dimension", "language", "metadata-checksum"],
)
def test_contract_rejects_wrong_identity_fields(field: str, value: object) -> None:
    """Every immutable contract identity field is checked."""
    payload = load_origin_label_contract().model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        OriginLabelContract.model_validate(payload)
