"""Prepared-bundle filesystem and manifest verification tests."""

import os
from pathlib import Path, PureWindowsPath

import pytest

import danish_personas.sources.bundle as bundle_module
from danish_personas.io import sha256_file, write_json
from danish_personas.models import BundleManifest
from danish_personas.sources.bundle import (
    _regular_file_inventory,
    verify_prepared_bundle,
)
from tests.support.bundles import _write_bundle


def test_prepared_bundle_inventory_accepts_normal_files(tmp_path: Path) -> None:
    """The inventory contract accepts ordinary files on every platform."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)

    inventory = _regular_file_inventory(bundle_dir=bundle_dir)

    assert inventory["source-preparation-report.json"] == (
        bundle_dir / "source-preparation-report.json"
    )


def test_prepared_bundle_manifest_excludes_itself(tmp_path: Path) -> None:
    """The manifest is the sole regular file excluded from its own file map."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    files["bundle-manifest.json"] = sha256_file(manifest_path)
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="exclude"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize(
    "replacement_kind",
    ["same-size", "symlink", "hardlink"],
    ids=["same-size-replacement", "symlink-replacement", "hardlink-replacement"],
)
def test_prepared_bundle_rechecks_path_after_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement_kind: str
) -> None:
    """A path swap after capture cannot alter or evade verification."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "source-preparation-report.json"
    replacement = tmp_path / "replacement.json"
    target_content = target.read_bytes()
    replacement.write_bytes(target_content)
    replaced = False
    original_hash = bundle_module._sha256_bytes

    def replace_during_hash(content: bytes) -> str:
        nonlocal replaced
        if not replaced and content == target_content:
            replaced = True
            target.unlink()
            if replacement_kind == "symlink":
                target.symlink_to(replacement)
            elif replacement_kind == "hardlink":
                os.link(replacement, target)
            else:
                replacement.replace(target)
        return original_hash(content)

    monkeypatch.setattr(bundle_module, "_sha256_bytes", replace_during_hash)
    with pytest.raises(ValueError, match="changed|safely"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_prepared_bundle_verifier_accepts_windows_manifest_keys(tmp_path: Path) -> None:
    """Windows relative keys are canonicalised before bundle verification."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    windows_files = {
        "\\".join(PureWindowsPath(relative_path).parts): checksum
        for relative_path, checksum in manifest.files.items()
    }
    write_json(
        path=manifest_path, payload=manifest.model_copy(update={"files": windows_files})
    )

    verified = verify_prepared_bundle(bundle_dir=bundle_dir)

    assert verified.files == windows_files


@pytest.mark.parametrize(
    "kind",
    ["symlink", "hardlink", "case", "mixed"],
    ids=["symlink", "hardlink", "case-collision", "separator-alias"],
)
def test_prepared_bundle_verifier_rejects_inventory_attacks(
    tmp_path: Path, kind: str
) -> None:
    """Symlinks, hard links, case collisions, and separator aliases fail."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "source-preparation-report.json"
    if kind == "symlink":
        target.unlink()
        try:
            target.symlink_to(tmp_path / "outside")
        except OSError:
            pytest.skip("symbolic links are unavailable")
    elif kind == "hardlink":
        replacement = bundle_dir / "hardlink-source.txt"
        replacement.write_text("hard link", encoding="utf-8")
        target.unlink()
        try:
            os.link(replacement, target)
        except OSError:
            pytest.skip("hard links are unavailable")
    elif kind == "case":
        case_path = bundle_dir / "SOURCE-PREPARATION-REPORT.JSON"
        case_path.write_text("collision", encoding="utf-8")
        if os.path.samefile(case_path, target):
            pytest.skip("case-insensitive filesystem cannot contain this collision")
    else:
        if os.name == "nt":
            pytest.skip("mixed POSIX and Windows separators need POSIX filenames")
        (bundle_dir / "normalized").joinpath("alias\\name").write_text(
            "alias", encoding="utf-8"
        )
        (bundle_dir / "normalized" / "alias").mkdir()
        (bundle_dir / "normalized" / "alias" / "name").write_text(
            "alias", encoding="utf-8"
        )

    with pytest.raises(
        ValueError, match="symlink|reparse point|hard link|case-colliding"
    ):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_prepared_bundle_verifier_rejects_inventory_extras_and_missing_files(
    tmp_path: Path,
) -> None:
    """The manifest and regular-file inventory must match exactly."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    extra = bundle_dir / "extra.txt"
    extra.write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="extra"):
        verify_prepared_bundle(bundle_dir=bundle_dir)

    extra.unlink()
    missing = bundle_dir / "source-preparation-report.json"
    missing.unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize(
    "alias_kind", ["separator", "case"], ids=["separator-alias", "case-alias"]
)
def test_prepared_bundle_verifier_rejects_manifest_aliases(
    tmp_path: Path, alias_kind: str
) -> None:
    """Canonical duplicate and case-folded manifest keys are rejected."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    path, checksum = next(iter(files.items()))
    if alias_kind == "separator":
        alias = path.replace("/", "\\") if "/" in path else path.replace("\\", "/")
        assert alias != path
        assert bundle_module._canonical_relative_key(alias) == (
            bundle_module._canonical_relative_key(path)
        )
    else:
        alias = path.swapcase()
    files[alias] = checksum
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="duplicate|colliding"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


def test_prepared_bundle_verifier_rejects_nonregular_inventory_entry(
    tmp_path: Path,
) -> None:
    """Special files are not valid prepared bundle contents."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    os.mkfifo(bundle_dir / "special.fifo")

    with pytest.raises(ValueError, match="non-regular"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        ".",
        "normalized/.",
        "normalized//file.parquet",
        "normalized/../file.parquet",
        "../escape.txt",
        "/absolute.txt",
        "\\rooted.txt",
        "C:relative.txt",
        "C:/absolute.txt",
        "\\\\server\\share\\file.txt",
        "\\\\?\\C:\\device.txt",
        "normal\\..\\escape.txt",
    ],
    ids=[
        "empty",
        "dot",
        "dot-component",
        "duplicate-separator",
        "parent-component",
        "parent-escape",
        "absolute-posix",
        "rooted-windows",
        "drive-relative",
        "drive-absolute",
        "unc-path",
        "device-path",
        "normalised-parent-escape",
    ],
)
def test_prepared_bundle_verifier_rejects_unsafe_manifest_paths(
    tmp_path: Path, unsafe_path: str
) -> None:
    """Manifest keys cannot use ambiguous or escaping path syntax."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    manifest_path = bundle_dir / "bundle-manifest.json"
    manifest = BundleManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    files = dict(manifest.files)
    _, checksum = files.popitem()
    files[unsafe_path] = checksum
    write_json(path=manifest_path, payload=manifest.model_copy(update={"files": files}))

    with pytest.raises(ValueError, match="path|paths"):
        verify_prepared_bundle(bundle_dir=bundle_dir)


@pytest.mark.parametrize(
    "kind",
    ["normal", "symlink", "hardlink"],
    ids=["regular-file", "symlink", "hardlink"],
)
def test_windows_native_inventory_contract(tmp_path: Path, kind: str) -> None:
    """Windows inventory uses native identity for files and links."""
    bundle_dir, _, _, _ = _write_bundle(root=tmp_path)
    target = bundle_dir / "source-preparation-report.json"
    if kind == "normal":
        inventory = _regular_file_inventory(bundle_dir=bundle_dir)
        assert inventory["source-preparation-report.json"] == target
        return
    if kind == "symlink":
        target.unlink()
        try:
            target.symlink_to(tmp_path / "outside")
        except OSError:
            pytest.skip("symbolic links are unavailable")
    else:
        replacement = bundle_dir / "hardlink-source.txt"
        replacement.write_text("hard link", encoding="utf-8")
        target.unlink()
        try:
            os.link(replacement, target)
        except OSError:
            pytest.skip("hard links are unavailable")

    with pytest.raises(ValueError, match="symlink|reparse point|hard link"):
        _regular_file_inventory(bundle_dir=bundle_dir)
