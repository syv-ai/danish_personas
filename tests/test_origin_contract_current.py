"""Adversarial tests for the current origin-label contract boundary."""

import typing as t
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from danish_personas.generation.models import GenerationConfig
from danish_personas.io import sha256_file
from danish_personas.models import BundleManifest, FrozenSampleManifest, RunManifest
from danish_personas.origin_labels import (
    DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
    ORIGIN_LABEL_CONTRACT_SHA256,
    load_origin_label_contract,
    validate_origin_contract_reference,
)
from danish_personas.sampling.generator import _origin_quota_sample
from tests.generation.manifest_helpers import origin_contract_fields

CONTRACT_TEXT = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH.read_text(encoding="utf-8")


class OriginBinding(t.TypedDict):
    """Current manifest origin-label binding fields."""

    origin_labels_contract_path: str
    origin_labels_contract_version: int
    origin_labels_contract_sha256: str
    origin_labels_contract_content: str


def test_current_manifests_require_every_origin_binding_field() -> None:
    """Omitting any current-schema binding field fails closed."""
    bundle = BundleManifest(
        bundle_id="bundle",
        prepared_bundle_schema_version=6,
        created_at="2026-01-01T00:00:00+00:00",
        source_lock_sha256="0" * 64,
        categories_sha256="1" * 64,
        source_snapshots=[],
        classification_snapshots=[],
        files={},
        reference_periods={},
        assumptions=[],
        lons20_contract_version=1,
        lons20_contract_sha256="2" * 64,
        **_binding(),
    )
    frozen = FrozenSampleManifest(
        sample_schema_version=3,
        source_run_id="run",
        rows=1,
        strata=[],
        method="test",
        data_file=Path("sample.parquet"),
        sha256="3" * 64,
        llm_calls=0,
        **_binding(),
    )
    run = RunManifest(
        run_id="run",
        sampler_schema_version=6,
        created_at="2026-01-01T00:00:00+00:00",
        bundle_id="bundle",
        bundle_manifest_sha256="4" * 64,
        sampling_config_sha256="5" * 64,
        rows=1,
        seed=1,
        data_file=Path("data.parquet"),
        data_sha256="6" * 64,
        logical_content_sha256="7" * 64,
        llm_calls=0,
        **_binding(),
    )
    for model in (bundle, frozen, run):
        payload = model.model_dump(mode="json")
        for field in _binding():
            omitted = payload.copy()
            omitted.pop(field)
            with pytest.raises(ValidationError):
                type(model).model_validate(omitted)


def _binding() -> OriginBinding:
    """Return the current embedded contract binding."""
    return origin_contract_fields()


def test_custom_contract_and_recalculated_outer_digest_do_not_bypass() -> None:
    """A custom path or embedded content remains invalid despite a valid digest."""
    with pytest.raises(ValueError):
        validate_origin_contract_reference(
            path=Path("config/custom.yaml"),
            version=1,
            sha256=ORIGIN_LABEL_CONTRACT_SHA256,
            content=CONTRACT_TEXT,
        )
    altered = CONTRACT_TEXT.replace("'5100': Danmark", "'5100': Sverige")
    with pytest.raises(ValueError):
        validate_origin_contract_reference(
            path=DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
            version=1,
            sha256=sha256_file(DEFAULT_ORIGIN_LABEL_CONTRACT_PATH),
            content=altered,
        )


def test_english_label_drift_is_not_just_a_nonblank_value() -> None:
    """The reviewed English half of each triple is immutable."""
    contract = load_origin_label_contract()
    payload = contract.model_dump()
    payload["labels_en"]["5100"] = "A different nonblank label"
    with pytest.raises(ValidationError):
        type(contract).model_validate(payload)


def test_generation_config_rejects_custom_contract_path() -> None:
    """Generation context creation accepts only the repository-relative path."""
    payload = {
        "version": 3,
        "llm_generation_enabled": False,
        "base_url": None,
        "model": None,
        "api_key_env": None,
        "timeout_seconds": 1.0,
        "maximum_http_attempts": 1,
        "maximum_validation_attempts": 1,
        "maximum_total_requests": 1,
        "retry_backoff_seconds": 0.0,
        "maximum_smoke_rows": 1,
        "response_format": "json_object",
        "attributes_prompt": "config/prompts/attributes-da.md",
        "personas_prompt": "config/prompts/personas-da.md",
        "origin_label_contract": "config/custom.yaml",
    }
    with pytest.raises(ValidationError):
        GenerationConfig.model_validate(payload)


def test_legacy_three_column_origin_table_is_rejected() -> None:
    """Sampling cannot synthesise Danish labels for a legacy table."""
    frame = pl.DataFrame(
        {"origin_country_code": ["5100"], "origin_country": ["Denmark"], "count": [1]}
    )
    with pytest.raises(ValueError, match="origin_country_da"):
        _origin_quota_sample(
            frame=frame, rows=1, rng=np.random.default_rng(1), require_danish=True
        )
