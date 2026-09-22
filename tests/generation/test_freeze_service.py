"""Public frozen-sample service and filesystem-safety tests."""

import typing as t
from pathlib import Path

import polars as pl
import pytest
from generation_test_helpers import write_generation_inputs

from danish_personas.io import sha256_file
from danish_personas.sampling.freeze import (
    SampleSizeError,
    _select_sample,
    freeze_sample,
)


def test_freeze_allocates_exact_origin_largest_remainder_quotas() -> None:
    """Origin quotas sum exactly to the request for finite weights."""
    sample = _select_sample(
        frame=_source_frame(("5100", 5), ("5456", 3), ("5999", 2)),
        rows=7,
        mode="population_proportional",
    )

    assert sample.get_column("origin_country_code").value_counts().sort(
        "origin_country_code"
    ).to_dicts() == [
        {"origin_country_code": "5100", "count": 4},
        {"origin_country_code": "5456", "count": 2},
        {"origin_country_code": "5999", "count": 1},
    ]
    assert sample.height == 7


def _source_frame(*groups: tuple[str, int]) -> pl.DataFrame:
    rows = [
        {
            "persona_id": f"{origin}-{index}",
            "origin_country_code": origin,
            "municipality_code": "101",
            "education_level": "higher_education",
            "labour_market_status": "employed",
        }
        for origin, count in groups
        for index in range(count)
    ]
    return pl.DataFrame(rows)


@pytest.mark.parametrize(
    ("mode", "expected_municipalities"),
    [
        ("population_proportional", {"101": 4}),
        ("stratified_round_robin", {"101": 2, "265": 2}),
    ],
)
def test_freeze_applies_mode_within_each_origin(
    mode: t.Literal["population_proportional", "stratified_round_robin"],
    expected_municipalities: dict[str, int],
) -> None:
    """Both modes preserve origin quotas before applying inner strata semantics."""
    frame = pl.DataFrame(
        [
            {
                "persona_id": f"{origin}-{municipality}-{index}",
                "origin_country_code": origin,
                "municipality_code": municipality,
                "education_level": "higher_education",
                "labour_market_status": "employed",
            }
            for origin in ("5100", "5456")
            for municipality, count in (("101", 3), ("265", 1))
            for index in range(count)
        ]
    )

    sample = _select_sample(frame=frame, rows=4, mode=mode)

    assert sample.height == 4
    assert sorted(
        sample.get_column("origin_country_code").value_counts().to_dicts(),
        key=lambda item: item["origin_country_code"],
    ) == [
        {"origin_country_code": "5100", "count": 2},
        {"origin_country_code": "5456", "count": 2},
    ]
    assert sorted(
        sample.get_column("municipality_code").value_counts().to_dicts(),
        key=lambda item: item["municipality_code"],
    ) == [
        {"municipality_code": code, "count": count}
        for code, count in sorted(expected_municipalities.items())
    ]


def test_freeze_origin_ties_use_sorted_codes_not_input_order() -> None:
    """Equal origin remainders select the lowest sorted codes first."""
    frame = _source_frame(("5999", 2), ("5100", 2), ("5456", 2))
    reversed_frame = frame.reverse()

    first = _select_sample(frame=frame, rows=2, mode="population_proportional")
    second = _select_sample(
        frame=reversed_frame, rows=2, mode="population_proportional"
    )

    assert first.get_column("origin_country_code").to_list() == ["5100", "5456"]
    assert (
        first.get_column("persona_id").to_list()
        == second.get_column("persona_id").to_list()
    )


def test_freeze_service_default_is_population_proportional(tmp_path: Path) -> None:
    """The default freeze preserves joint proportions with deterministic quotas."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    source = pl.read_parquet(run_dir / "structured-records.parquet")
    rows = pl.concat(
        [
            pl.DataFrame(
                [source.row(0, named=True) | {"persona_id": f"persona-a-{index}"}]
            )
            for index in range(8)
        ]
        + [
            pl.DataFrame(
                [source.row(1, named=True) | {"persona_id": f"persona-b-{index}"}]
            )
            for index in range(2)
        ],
        how="vertical",
    )
    rows.write_parquet(run_dir / "structured-records.parquet")
    output = run_dir / "proportional.parquet"

    freeze_sample(run_dir=run_dir, rows=5, output=output)

    sample = pl.read_parquet(output)
    assert sample.height == 5
    assert sorted(
        sample.get_column("municipality_code").value_counts().to_dicts(),
        key=lambda item: item["municipality_code"],
    ) == [
        {"municipality_code": "101", "count": 4},
        {"municipality_code": "265", "count": 1},
    ]
    manifest = output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    assert '"mode": "population_proportional"' in manifest
    assert '"origin_country_code"' in manifest
    assert "largest-remainder quotas" in manifest


def test_freeze_service_is_deterministic_and_bounds_size(tmp_path: Path) -> None:
    """The public freezer preserves round-robin output and rejects oversized input."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    first = run_dir / "first.parquet"
    second = run_dir / "second.parquet"

    freeze_sample(run_dir=run_dir, rows=2, output=first)
    freeze_sample(run_dir=run_dir, rows=2, output=second)

    assert sha256_file(first) == sha256_file(second)
    assert pl.read_parquet(first).get_column("persona_id").to_list() == [
        "persona-1",
        "persona-2",
    ]
    assert first.parent == run_dir
    assert first.with_suffix(".manifest.json").parent == run_dir
    with pytest.raises(SampleSizeError, match="exceeds the run row count"):
        freeze_sample(run_dir=run_dir, rows=3, output=first)


def test_freeze_service_rejects_dangling_output_symlink(tmp_path: Path) -> None:
    """A dangling output link is rejected before sample generation."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "dangling.parquet"
    _make_symlink(path=output, target=tmp_path / "missing.parquet")

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert output.is_symlink()


def _make_symlink(*, path: Path, target: Path, directory: bool = False) -> None:
    """Create a symlink or skip where the platform disallows symlinks."""
    try:
        path.symlink_to(target, target_is_directory=directory)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_freeze_service_rejects_manifest_symlink(tmp_path: Path) -> None:
    """A manifest link is rejected before the corresponding sample is written."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "manifest-link.parquet"
    external = tmp_path / "external-manifest.json"
    external.write_text("protected", encoding="utf-8")
    _make_symlink(path=output.with_suffix(".manifest.json"), target=external)

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert not output.exists()
    assert external.read_text(encoding="utf-8") == "protected"


def test_freeze_service_rejects_output_symlink_to_external_file(tmp_path: Path) -> None:
    """An output link cannot redirect a freeze to an external file."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    external = tmp_path / "external.parquet"
    external.write_bytes(b"protected")
    output = run_dir / "external-output.parquet"
    _make_symlink(path=output, target=external)

    with pytest.raises(ValueError, match="symlink"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)

    assert external.read_bytes() == b"protected"
    assert output.is_symlink()


def test_freeze_service_rejects_relocation_from_run_manifest(tmp_path: Path) -> None:
    """A frozen sample cannot be detached from its validated upstream evidence."""
    paths = write_generation_inputs(root=tmp_path)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(
            run_dir=paths["sample"].parent,
            rows=1,
            output=tmp_path / "relocated.parquet",
        )


def test_freeze_service_rejects_symlinked_parent(tmp_path: Path) -> None:
    """A path through a linked parent is not a direct run child."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    real_parent = run_dir / "real-parent"
    real_parent.mkdir()
    linked_parent = run_dir / "linked-parent"
    _make_symlink(path=linked_parent, target=real_parent, directory=True)

    with pytest.raises(ValueError, match="inside its validated run directory"):
        freeze_sample(run_dir=run_dir, rows=1, output=linked_parent / "sample.parquet")


def test_freeze_service_rejects_traversal_alias(tmp_path: Path) -> None:
    """Lexical traversal aliases are rejected even when they normalise in-run."""
    paths = write_generation_inputs(root=tmp_path)
    run_dir = paths["sample"].parent
    output = run_dir / "nested" / ".." / "traversal.parquet"

    with pytest.raises(ValueError, match="traversal"):
        freeze_sample(run_dir=run_dir, rows=1, output=output)


def test_freeze_service_supports_explicit_round_robin_mode(tmp_path: Path) -> None:
    """Round-robin remains available as a named non-default mode."""
    paths = write_generation_inputs(root=tmp_path)
    output = paths["sample"].parent / "round-robin.parquet"

    freeze_sample(
        run_dir=paths["sample"].parent,
        rows=2,
        output=output,
        mode="stratified_round_robin",
    )

    manifest = output.with_suffix(".manifest.json").read_text(encoding="utf-8")
    assert '"mode": "stratified_round_robin"' in manifest


def test_freeze_small_sample_may_drop_origins_but_keeps_exact_size() -> None:
    """Fewer requested rows than origins cannot represent every origin."""
    sample = _select_sample(
        frame=_source_frame(("5999", 2), ("5100", 2), ("5456", 2)),
        rows=1,
        mode="stratified_round_robin",
    )

    assert sample.height == 1
    assert sample.get_column("origin_country_code").to_list() == ["5100"]
