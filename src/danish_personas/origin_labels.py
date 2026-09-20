"""Immutable official Danish FOLK2 origin-country labels."""

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr, model_validator

DEFAULT_ORIGIN_LABEL_CONTRACT_PATH = Path("config/folk2-ieland-labels-da.yaml")
ORIGIN_LABEL_CONTRACT_VERSION = 1
ORIGIN_LABEL_COUNT = 241
ORIGIN_LABEL_TABLE_ID = "FOLK2"
ORIGIN_LABEL_DIMENSION = "IELAND"
ORIGIN_LABEL_LANGUAGE = "da"
ORIGIN_LABEL_SOURCE_METADATA_EN_SHA256 = (
    "cb2558d35bee7b3eed451984f457cc4dc7b683f71e67022d6767f3415d04884a"
)
ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256 = (
    "f5c1f0a20f29372d6b222ce7a23cdc4ef0481d9e23fa6bd9b66b116e7adcb213"
)
ORIGIN_LABEL_SOURCE_METADATA_SHA256 = ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256
ORIGIN_LABEL_CONTRACT_SHA256 = (
    "87a296b5c67b5763b952ca314be1b737c8238e5281461419e05005cab9539b61"
)
_CODE_PATTERN = re.compile(r"[0-9]{4}\Z")


def _validate_labels(
    labels: Mapping[str, str], *, allow_official_padding: bool = False
) -> None:
    normalised_labels: set[str] = set()
    for code, label in labels.items():
        if _CODE_PATTERN.fullmatch(code) is None:
            raise ValueError(f"Malformed FOLK2 IELAND code: {code!r}")
        if not label or (not allow_official_padding and label.strip() != label):
            raise ValueError(f"FOLK2 IELAND label is blank or padded: {code!r}")
        if unicodedata.normalize("NFC", label) != label:
            raise ValueError(f"FOLK2 IELAND label is not NFC-normalised: {code!r}")
        normalised = unicodedata.normalize("NFC", label).casefold()
        if normalised in normalised_labels:
            raise ValueError("FOLK2 IELAND labels must be unique")
        normalised_labels.add(normalised)


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that preserves the contract's one-entry-per-code rule."""

    def construct_mapping(
        self, node: yaml.nodes.MappingNode, deep: bool = False
    ) -> dict[object, object]:
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate YAML key: {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def load_bound_origin_labels(
    metadata_path: Path, contract_path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH
) -> dict[str, str]:
    """Load a contract and bind it to a metadata JSON file.

    Args:
        metadata_path:
            Exact source metadata JSON path.
        contract_path:
            YAML contract path. Defaults to the repository contract.

    Returns:
        The bound code-to-label mapping.
    """
    with metadata_path.open(encoding="utf-8") as file:
        metadata = json.load(file)
    contract = load_origin_label_contract(path=contract_path)
    return bind_origin_labels(
        contract=contract,
        metadata=metadata,
        metadata_sha256=source_metadata_sha256(metadata_path),
    )


def _as_mapping(value: object, name: str) -> Mapping[str, object]:
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def source_metadata_sha256(path: Path) -> str:
    """Return the byte checksum of a Danish StatBank metadata file."""
    return _sha256_file(path)


class OriginLabelContract(BaseModel):
    """Versioned, ordered code-to-English-Danish FOLK2 triples."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: StrictInt
    table_id: StrictStr
    dimension: StrictStr
    language: StrictStr
    source_metadata_en_sha256: StrictStr
    source_metadata_da_sha256: StrictStr
    labels_en: dict[StrictStr, StrictStr]
    labels_da: dict[StrictStr, StrictStr]

    @property
    def source_metadata_sha256(self) -> str:
        """Danish source checksum for existing callers."""
        return self.source_metadata_da_sha256

    @property
    def labels(self) -> dict[str, str]:
        """Danish labels for existing table-writing callers."""
        return self.labels_da

    @property
    def ordered_labels(self) -> tuple[tuple[str, str], ...]:
        """Danish labels in their immutable official order."""
        return tuple(self.labels_da.items())

    @property
    def ordered_triples(self) -> tuple[tuple[str, str, str], ...]:
        """Code, English label, and Danish label in official order."""
        return tuple(
            (code, self.labels_en[code], self.labels_da[code])
            for code in self.labels_en
        )

    @model_validator(mode="after")
    def validate_contract(self) -> "OriginLabelContract":
        """Reject changes to the reviewed contract identity or label partition.

        Returns:
            The validated contract.

        Raises:
            ValueError: If identity or labels differ from the reviewed contract.
        """
        _validate_contract_identity(self)
        _validate_labels(self.labels_en, allow_official_padding=True)
        _validate_labels(self.labels_da)
        if tuple(self.labels_en) != tuple(self.labels_da):
            raise ValueError("Origin-label English and Danish code order differs")
        if len(set(self.labels_en.values())) != ORIGIN_LABEL_COUNT:
            raise ValueError("Origin-label English labels must be unique")
        return self


def _validate_contract_identity(contract: OriginLabelContract) -> None:
    if contract.version != ORIGIN_LABEL_CONTRACT_VERSION:
        raise ValueError("Unsupported origin-label contract version")
    if contract.table_id != ORIGIN_LABEL_TABLE_ID:
        raise ValueError("Origin-label contract table_id must be FOLK2")
    if contract.dimension != ORIGIN_LABEL_DIMENSION:
        raise ValueError("Origin-label contract dimension must be IELAND")
    if contract.language != ORIGIN_LABEL_LANGUAGE:
        raise ValueError("Origin-label contract language must be da")
    if contract.source_metadata_en_sha256 != ORIGIN_LABEL_SOURCE_METADATA_EN_SHA256:
        raise ValueError("Origin-label English metadata checksum does not match")
    if contract.source_metadata_da_sha256 != ORIGIN_LABEL_SOURCE_METADATA_DA_SHA256:
        raise ValueError("Origin-label Danish metadata checksum does not match")
    if (
        len(contract.labels_en) != ORIGIN_LABEL_COUNT
        or len(contract.labels_da) != ORIGIN_LABEL_COUNT
    ):
        raise ValueError(
            (
                f"Origin-label contract must contain {ORIGIN_LABEL_COUNT} labels "
                "per language"
            )
        )


def bind_origin_labels(
    contract: OriginLabelContract,
    metadata: Mapping[str, object] | BaseModel,
    *,
    metadata_sha256: str | None = None,
    source_metadata_sha256_value: str | None = None,
    expected_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Bind the contract to the official metadata and return its label mapping.

    Args:
        contract:
            Validated origin-label contract.
        metadata:
            Parsed StatBank metadata mapping or Pydantic metadata model.
        metadata_sha256:
            Optional checksum of the exact metadata bytes being bound.
        source_metadata_sha256_value:
            Backwards-compatible explicit spelling for ``metadata_sha256``.
        expected_labels:
            Expected code-to-label mapping for this metadata language.

    Returns:
        A new code-to-label mapping in official metadata order.

    Raises:
        ValueError:
            If metadata identity, checksum, dimension, order, codes, or labels do
            not exactly match the reviewed contract.
    """
    if metadata_sha256 is not None and source_metadata_sha256_value is not None:
        raise ValueError("Provide only one metadata checksum")
    observed_sha256 = metadata_sha256 or source_metadata_sha256_value
    if (
        observed_sha256 is not None
        and observed_sha256 != contract.source_metadata_da_sha256
    ):
        raise ValueError("FOLK2 IELAND metadata checksum does not match contract")

    metadata_mapping = _as_mapping(metadata, "metadata")
    if metadata_mapping.get("id") != contract.table_id:
        raise ValueError("FOLK2 IELAND metadata table does not match contract")
    variables = metadata_mapping.get("variables")
    if not isinstance(variables, list):
        raise ValueError("FOLK2 metadata variables must be a list")
    matching_variables = [
        _as_mapping(variable, "variable")
        for variable in variables
        if _as_mapping(variable, "variable").get("id") == contract.dimension
    ]
    if len(matching_variables) != 1:
        raise ValueError("FOLK2 metadata must contain exactly one IELAND dimension")
    values = matching_variables[0].get("values")
    if not isinstance(values, list) or len(values) != ORIGIN_LABEL_COUNT:
        raise ValueError(
            f"FOLK2 IELAND metadata must contain {ORIGIN_LABEL_COUNT} values"
        )

    bound: dict[str, str] = {}
    for value in values:
        value_mapping = _as_mapping(value, "IELAND value")
        code = value_mapping.get("id")
        label = value_mapping.get("text")
        if not isinstance(code, str) or not isinstance(label, str):
            raise ValueError("FOLK2 IELAND metadata values must contain string id/text")
        bound[code] = label
    expected = expected_labels or contract.labels_da
    if tuple(bound.items()) != tuple(expected.items()):
        raise ValueError(
            "FOLK2 IELAND metadata does not match contract order or labels"
        )
    return dict(bound)


def bind_origin_label_triples(
    contract: OriginLabelContract,
    english_metadata: Mapping[str, object] | BaseModel,
    danish_metadata: Mapping[str, object] | BaseModel,
    *,
    english_metadata_sha256: str,
    danish_metadata_sha256: str,
) -> tuple[dict[str, str], dict[str, str]]:
    """Validate and return the exact English/Danish metadata triple maps.

    Returns:
        English and Danish code-to-label mappings.

    Raises:
        ValueError: If either metadata response differs from the reviewed contract.
    """
    if english_metadata_sha256 != contract.source_metadata_en_sha256:
        raise ValueError("FOLK2 English metadata checksum does not match contract")
    if danish_metadata_sha256 != contract.source_metadata_da_sha256:
        raise ValueError("FOLK2 Danish metadata checksum does not match contract")
    english = bind_origin_labels(
        contract=contract, metadata=english_metadata, expected_labels=contract.labels_en
    )
    danish = bind_origin_labels(
        contract=contract,
        metadata=danish_metadata,
        metadata_sha256=danish_metadata_sha256,
        expected_labels=contract.labels_da,
    )
    if tuple(english) != tuple(danish):
        raise ValueError("FOLK2 English and Danish metadata code order differs")
    return english, danish


def validate_origin_contract_reference(
    *, path: str | Path, version: int, sha256: str, content: str | OriginLabelContract
) -> None:
    """Require an exact canonical contract identity in a provenance binding.

    Raises:
        ValueError: If any binding component is not the reviewed contract.
    """
    if Path(path) != DEFAULT_ORIGIN_LABEL_CONTRACT_PATH or str(path) != str(
        DEFAULT_ORIGIN_LABEL_CONTRACT_PATH
    ):
        raise ValueError("Origin-label contract path must be canonical")
    if version != ORIGIN_LABEL_CONTRACT_VERSION:
        raise ValueError("Origin-label contract version is not current")
    if sha256 != ORIGIN_LABEL_CONTRACT_SHA256:
        raise ValueError("Origin-label contract checksum is not reviewed")
    if isinstance(content, str):
        observed = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if observed != ORIGIN_LABEL_CONTRACT_SHA256:
            raise ValueError("Embedded origin-label contract content changed")
        payload = yaml.load(content, Loader=_UniqueKeyLoader)
        embedded = OriginLabelContract.model_validate(payload)
    else:
        embedded = content
    if embedded != load_origin_label_contract():
        raise ValueError("Embedded origin-label contract is not canonical")


bind_origin_triples = bind_origin_label_triples


def _is_canonical_contract_path(path: Path) -> bool:
    return path == DEFAULT_ORIGIN_LABEL_CONTRACT_PATH or (
        path.is_absolute()
        and path.name == DEFAULT_ORIGIN_LABEL_CONTRACT_PATH.name
        and path.parent.name == DEFAULT_ORIGIN_LABEL_CONTRACT_PATH.parent.name
    )


def load_origin_label_contract(
    path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
) -> OriginLabelContract:
    """Load and validate the one canonical origin-label contract.

    Args:
        path:
            Must be exactly the repository-relative canonical contract path.

    Returns:
        A strictly validated origin-label contract.

    Raises:
        ValueError: If the path is not canonical or the contract is invalid.
    """
    if not _is_canonical_contract_path(path):
        raise ValueError("Only the canonical origin-label contract is permitted")
    contract_bytes = path.read_bytes()
    if hashlib.sha256(contract_bytes).hexdigest() != ORIGIN_LABEL_CONTRACT_SHA256:
        raise ValueError("Canonical origin-label contract bytes are not reviewed")
    payload = yaml.load(contract_bytes.decode("utf-8"), Loader=_UniqueKeyLoader)
    return OriginLabelContract.model_validate(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def origin_label_contract_sha256(
    path: Path = DEFAULT_ORIGIN_LABEL_CONTRACT_PATH,
) -> str:
    """Return the byte checksum of the canonical origin-label contract.

    Raises:
        ValueError: If the path is not canonical.
    """
    if not _is_canonical_contract_path(path):
        raise ValueError("Only the canonical origin-label contract is permitted")
    return _sha256_file(path)
