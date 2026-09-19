"""Canonical, reviewed semantics for the LONS20 source."""

from pathlib import Path

from ..io import load_yaml_model
from ..models import DISCO_TWO_DIGIT_CODES, Lons20Contract, SourceMetadataExpectations

DEFAULT_LONS20_CONTRACT_PATH = Path("config/lons20-contract.yaml")
LONS20_SELECTOR_CODES: dict[str, tuple[str, ...]] = {
    "SEKTOR": ("1000",),
    "AFLOEN": ("TIFA",),
    "LONGRP": ("LTOT",),
    "LØNMÅL": ("ANTAL",),
    "KØN": ("M", "K"),
    "Tid": ("2024",),
}


def load_lons20_contract(path: Path = DEFAULT_LONS20_CONTRACT_PATH) -> Lons20Contract:
    """Load the checked-in LONS20 contract.

    Args:
        path:
            Path to the separately reviewed contract.

    Returns:
        Validated canonical contract.
    """
    contract = load_yaml_model(path=path, model=Lons20Contract)
    _validate_lons20_contract(contract=contract)
    return contract


def _validate_lons20_contract(contract: Lons20Contract) -> None:
    if contract.version < 1 or contract.table_id != "LONS20":
        raise ValueError("LONS20 contract has an invalid version or table id")
    expected_dimension_ids = {"ARBF", *LONS20_SELECTOR_CODES}
    if set(contract.dimensions) != expected_dimension_ids:
        missing = sorted(expected_dimension_ids - set(contract.dimensions))
        extra = sorted(set(contract.dimensions) - expected_dimension_ids)
        raise ValueError(
            f"LONS20 contract dimensions changed: missing={missing}, extra={extra}"
        )
    expected_codes = set(DISCO_TWO_DIGIT_CODES)
    if set(contract.arbf) != expected_codes:
        missing = sorted(expected_codes - set(contract.arbf))
        extra = sorted(set(contract.arbf) - expected_codes)
        raise ValueError(
            f"LONS20 contract ARBF codes changed: missing={missing}, extra={extra}"
        )
    if any(not label.strip() for label in contract.arbf.values()):
        raise ValueError("LONS20 contract contains a blank ARBF label")
    if set(contract.selectors) != set(LONS20_SELECTOR_CODES):
        raise ValueError("LONS20 contract selectors do not cover fixed dimensions")
    for dimension, codes in LONS20_SELECTOR_CODES.items():
        values = contract.selectors[dimension]
        if set(values) != set(codes):
            missing = sorted(set(codes) - set(values))
            extra = sorted(set(values) - set(codes))
            raise ValueError(
                f"LONS20 contract {dimension} codes changed: "
                f"missing={missing}, extra={extra}"
            )
        if any(not label.strip() for label in values.values()):
            raise ValueError(f"LONS20 contract contains a blank {dimension} label")
    if any(not label.strip() for label in contract.dimensions.values()):
        raise ValueError("LONS20 contract contains a blank dimension label")


def lons20_dimensions(contract: Lons20Contract) -> dict[str, list[str]]:
    """Return the fixed LONS20 selection from the canonical contract."""
    _validate_lons20_contract(contract=contract)
    return {
        "ARBF": list(DISCO_TWO_DIGIT_CODES),
        **{
            dimension: list(codes) for dimension, codes in LONS20_SELECTOR_CODES.items()
        },
    }


def lons20_expectations(contract: Lons20Contract) -> SourceMetadataExpectations:
    """Convert the canonical contract to metadata expectations.

    Returns:
        The shared source metadata expectation model.
    """
    _validate_lons20_contract(contract=contract)
    values = dict(contract.selectors)
    values["ARBF"] = dict(contract.arbf)
    return SourceMetadataExpectations(
        table_text=contract.table_text,
        description=contract.description,
        unit=contract.unit,
        dimensions=dict(contract.dimensions),
        values=values,
    )


__all__ = [
    "DEFAULT_LONS20_CONTRACT_PATH",
    "LONS20_SELECTOR_CODES",
    "load_lons20_contract",
    "lons20_dimensions",
    "lons20_expectations",
]
