"""Tests for the independently reviewed LONS20 contract."""

from pathlib import Path

import pytest

from danish_personas.io import load_yaml_model, sha256_file
from danish_personas.models import (
    SourceLock,
    StatBankMetadata,
    StatBankValue,
    StatBankVariable,
)
from danish_personas.sources.lons20 import (
    _validate_lons20_contract,
    load_lons20_contract,
    lons20_expectations,
)
from danish_personas.sources.prepare import _bundle_id, _validate_lons20_source
from danish_personas.sources.statbank import _validate_metadata_expectations

CONTRACT_PATH = Path("config/lons20-contract.yaml")
LOCK_PATH = Path("config/sources.lock.yaml")


def test_contract_checksum_changes_bundle_identity(tmp_path: Path) -> None:
    """Changing the reviewed contract produces a different bundle identity."""
    lock = tmp_path / "lock.yaml"
    categories = tmp_path / "categories.yaml"
    contract = tmp_path / "contract.yaml"
    lock.write_text("lock", encoding="utf-8")
    categories.write_text("categories", encoding="utf-8")
    contract.write_text("contract", encoding="utf-8")
    first = _bundle_id(
        lock_path=lock,
        categories_path=categories,
        contract_path=contract,
        contract_version=1,
    )
    contract.write_text("changed contract", encoding="utf-8")
    second = _bundle_id(
        lock_path=lock,
        categories_path=categories,
        contract_path=contract,
        contract_version=1,
    )

    assert sha256_file(contract) != sha256_file(tmp_path / "categories.yaml")
    assert first != second


def test_contract_rejects_missing_and_extra_arbf_codes() -> None:
    """The canonical partition cannot silently shrink or grow."""
    contract = load_lons20_contract(path=CONTRACT_PATH)
    missing = contract.model_copy(update={"arbf": {"01": contract.arbf["01"]}})
    extra = contract.model_copy(update={"arbf": {**contract.arbf, "97": "Unreviewed"}})

    with pytest.raises(ValueError, match="ARBF codes changed"):
        _validate_lons20_contract(contract=missing)
    with pytest.raises(ValueError, match="ARBF codes changed"):
        _validate_lons20_contract(contract=extra)


def test_coordinated_lock_and_snapshot_drift_still_fails() -> None:
    """Changing both inputs cannot replace the canonical contract."""
    contract = load_lons20_contract(path=CONTRACT_PATH)
    lock = load_yaml_model(path=LOCK_PATH, model=SourceLock)
    source = next(source for source in lock.sources if source.table_id == "LONS20")
    assert source.metadata_expectations is not None
    drift = source.metadata_expectations.model_copy(
        update={"table_text": "Drifted earnings"}
    )
    tampered_source = source.model_copy(update={"metadata_expectations": drift})
    tampered_lock = lock.model_copy(
        update={
            "sources": [
                tampered_source if item is source else item for item in lock.sources
            ]
        }
    )

    with pytest.raises(ValueError, match="canonical contract"):
        _validate_lons20_source(
            lock=tampered_lock,
            contract_expectations=lons20_expectations(contract=contract),
        )


def test_lock_metadata_matches_canonical_contract() -> None:
    """The resolved lock cannot redefine reviewed LONS20 semantics."""
    contract = load_lons20_contract(path=CONTRACT_PATH)
    lock = load_yaml_model(path=LOCK_PATH, model=SourceLock)
    _validate_lons20_source(
        lock=lock, contract_expectations=lons20_expectations(contract=contract)
    )


def test_tampered_lock_only_fails_against_contract() -> None:
    """A label changed only in the lock is rejected."""
    contract = load_lons20_contract(path=CONTRACT_PATH)
    lock = load_yaml_model(path=LOCK_PATH, model=SourceLock)
    source = next(source for source in lock.sources if source.table_id == "LONS20")
    assert source.metadata_expectations is not None
    drifted_labels = {
        **source.metadata_expectations.values["ARBF"],
        "01": "Changed official label",
    }
    expectations = source.metadata_expectations.model_copy(
        update={
            "values": {**source.metadata_expectations.values, "ARBF": drifted_labels}
        }
    )
    tampered_source = source.model_copy(update={"metadata_expectations": expectations})
    tampered_lock = lock.model_copy(
        update={
            "sources": [
                tampered_source if item is source else item for item in lock.sources
            ]
        }
    )

    with pytest.raises(ValueError, match="canonical contract"):
        _validate_lons20_source(
            lock=tampered_lock,
            contract_expectations=lons20_expectations(contract=contract),
        )


def test_tampered_snapshot_only_fails_against_contract() -> None:
    """Snapshot metadata is checked independently of lock metadata."""
    contract = load_lons20_contract(path=CONTRACT_PATH)
    expectations = lons20_expectations(contract=contract)
    variables = [
        StatBankVariable(
            id=dimension,
            text=label,
            values=[
                StatBankValue(id=code, text=value_label)
                for code, value_label in expectations.values[dimension].items()
            ],
        )
        for dimension, label in expectations.dimensions.items()
    ]
    metadata = StatBankMetadata(
        id="LONS20",
        text=expectations.table_text,
        description=expectations.description,
        unit=expectations.unit,
        updated="2025-09-29T08:00:00",
        variables=variables,
    )
    tampered = metadata.model_copy(
        update={
            "variables": [
                variables[0].model_copy(
                    update={
                        "values": [
                            StatBankValue(id="01", text="Changed"),
                            *variables[0].values[1:],
                        ]
                    }
                ),
                *variables[1:],
            ]
        }
    )

    with pytest.raises(ValueError, match="metadata label changed"):
        _validate_metadata_expectations(
            table_id="LONS20",
            metadata=tampered,
            expectations=expectations,
            dimensions={
                dimension: list(values)
                for dimension, values in expectations.values.items()
            },
        )
