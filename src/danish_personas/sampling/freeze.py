"""Deterministic Phase-3 sample freezing service."""

import json
import logging
import os
import stat
import tempfile
import typing as t
from pathlib import Path

import polars as pl

from ..checksum import ChecksumValidationPolicy
from ..io import sha256_file
from ..models import (
    FROZEN_SAMPLE_SCHEMA_VERSION,
    SAMPLER_SCHEMA_VERSION,
    FrozenSampleManifest,
    RunManifest,
)

LOGGER = logging.getLogger(__name__)


ORIGIN_STRATUM = "origin_country_code"
ORIGIN_MARGINAL_METHOD = (
    "deterministic largest-remainder quotas for the origin_country_code marginal"
)
STRATA = ("municipality_code", "education_level", "labour_market_status")
FREEZE_MODES = ("population_proportional", "stratified_round_robin")


def freeze_sample(
    *,
    run_dir: Path,
    rows: int,
    output: Path,
    mode: t.Literal["population_proportional", "stratified_round_robin"] = (
        "population_proportional"
    ),
    checksum_policy: ChecksumValidationPolicy = ChecksumValidationPolicy.STRICT,
) -> Path:
    """Select and persist a deterministic development sample.

    Both modes first create the globally balanced selection using the requested
    municipality/education/status strategy. Deterministic largest-remainder
    origin quotas are then imposed with same-stratum swaps, retaining the global
    stratum counts exactly whenever the margins permit it. If those margins are
    infeasible, the deterministic fallback keeps the largest possible overlap
    with the global selection while still enforcing the origin quotas.

    Args:
        run_dir:
            Validated deterministic demographic run directory.
        rows:
            Number of records to select.
        output:
            Destination Parquet path for the frozen sample.
        mode:
            Selection strategy. Defaults to population-proportional allocation.
        checksum_policy:
            Whether persisted run bindings must match. Defaults to strict validation.

    Returns:
        Path to the written frozen sample.

    Raises:
        SampleSizeError:
            If the requested sample is larger than the source run.
        ValueError:
            If rows is less than one, mode is unknown, or either output path is
            not a direct, non-linked child of the validated run directory.
    """
    if rows < 1:
        raise ValueError("Requested sample must contain at least one row")
    if mode not in FREEZE_MODES:
        raise ValueError(f"Unknown freeze mode: {mode}")
    canonical_run_dir = _canonical_run_directory(run_dir=run_dir)
    output_path, manifest_path = _validate_destinations(
        run_dir=canonical_run_dir, output=output
    )
    manifest = RunManifest.model_validate_json(
        (canonical_run_dir / "run-manifest.json").read_text(encoding="utf-8"),
        context={"checksum_policy": checksum_policy},
    )
    if manifest.sampler_schema_version != SAMPLER_SCHEMA_VERSION:
        raise ValueError("Cannot freeze a legacy demographic run")
    frame = pl.read_parquet(canonical_run_dir / manifest.data_file)
    if rows > frame.height:
        raise SampleSizeError("Requested sample exceeds the run row count")
    sample = _select_sample(frame=frame, rows=rows, mode=mode)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=canonical_run_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        sample.write_parquet(temporary, compression="zstd")
        _validate_destinations(run_dir=canonical_run_dir, output=output_path)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    sample_manifest = FrozenSampleManifest.model_validate(
        {
            "sample_schema_version": FROZEN_SAMPLE_SCHEMA_VERSION,
            "source_run_id": manifest.run_id,
            "rows": sample.height,
            "strata": [ORIGIN_STRATUM, *STRATA],
            "method": (
                f"global {mode} STRATA selection, then {ORIGIN_MARGINAL_METHOD}; "
                "deterministic same-STRATA swaps preserve global STRATA counts "
                "when feasible, otherwise deterministic maximum-overlap fallback"
            ),
            "mode": mode,
            "data_file": output_path.name,
            "sha256": sha256_file(output_path),
            "llm_calls": 0,
            "origin_labels_contract_path": manifest.origin_labels_contract_path,
            "origin_labels_contract_version": manifest.origin_labels_contract_version,
            "origin_labels_contract_sha256": manifest.origin_labels_contract_sha256,
            "origin_labels_contract_content": manifest.origin_labels_contract_content,
        },
        context={"checksum_policy": checksum_policy},
    )
    _write_manifest(
        path=manifest_path,
        run_dir=canonical_run_dir,
        payload=sample_manifest.model_dump(mode="json"),
    )
    LOGGER.info("Frozen %s development records at %s", sample.height, output_path)
    return output


class SampleSizeError(ValueError):
    """Raised when a requested sample is larger than its source run."""


def _canonical_run_directory(*, run_dir: Path) -> Path:
    """Resolve and validate the source run directory.

    Args:
        run_dir:
            Source run directory.

    Returns:
        Canonical source run directory.

    Raises:
        ValueError:
            If the source run cannot be resolved or is not a directory.
    """
    try:
        canonical = run_dir.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("Validated source run directory cannot be resolved") from error
    if not canonical.is_dir():
        raise ValueError("Validated source run directory must be a directory")
    return canonical


def _select_sample(
    *,
    frame: pl.DataFrame,
    rows: int,
    mode: t.Literal["population_proportional", "stratified_round_robin"],
) -> pl.DataFrame:
    """Select rows with an origin marginal and the requested inner strategy.

    The unconstrained selection is made before applying the origin marginal. A
    A deterministic flow fallback finds a selection with the same STRATA margins,
    when one exists. This is equivalent to deterministic swaps within a STRATA
    cell and avoids making the origin marginal distort the global allocation.

    Returns:
        The selected rows sorted by persona identifier.

    Raises:
        ValueError:
            If persona identifiers are not unique.
    """
    if frame.get_column("persona_id").n_unique() != frame.height:
        raise ValueError("Cannot freeze rows with duplicate persona_id values")

    sorted_frame = frame.sort([*STRATA, "persona_id"])
    baseline = _select_within_origin(group=sorted_frame, rows=rows, mode=mode)
    baseline_ids = set(baseline.get_column("persona_id").to_list())
    origin_frame = frame.sort([ORIGIN_STRATUM, *STRATA, "persona_id"])
    origin_groups = origin_frame.partition_by([ORIGIN_STRATUM], maintain_order=True)
    origin_quotas = _largest_remainder_quotas(
        groups=origin_groups, rows=rows, group_key=ORIGIN_STRATUM
    )
    origin_targets = {
        group.item(0, ORIGIN_STRATUM): quota
        for group, quota in zip(origin_groups, origin_quotas)
    }
    cell_rows, cell_capacity, cell_baseline = _build_cells(
        frame=sorted_frame, baseline_ids=baseline_ids
    )
    cell_counts = _select_cell_counts(
        cell_capacity=cell_capacity,
        cell_baseline=cell_baseline,
        cell_rows=rows,
        origin_targets=origin_targets,
    )
    baseline_positions = {
        index
        for index, persona_id in enumerate(sorted_frame.get_column("persona_id"))
        if persona_id in baseline_ids
    }
    selected_indices = [
        index
        for cell in _ordered_cells(cell_rows=cell_rows)
        for index in _choose_cell_rows(
            positions=cell_rows[cell],
            count=cell_counts[cell],
            baseline_positions=baseline_positions,
        )
    ]
    if len(selected_indices) != rows or len(set(selected_indices)) != rows:
        raise ValueError("Freeze selection did not produce the requested unique rows")
    return sorted_frame.gather(selected_indices).sort("persona_id")


def _build_cells(
    *, frame: pl.DataFrame, baseline_ids: set[object]
) -> tuple[
    dict[tuple[tuple[object, ...], str], list[int]],
    dict[tuple[tuple[object, ...], str], int],
    dict[tuple[tuple[object, ...], str], int],
]:
    """Index rows by STRATA and origin in deterministic frame order.

    Returns:
        Row positions, cell capacities, and baseline counts by cell.
    """
    cell_rows: dict[tuple[tuple[object, ...], str], list[int]] = {}
    cell_baseline: dict[tuple[tuple[object, ...], str], int] = {}
    values = frame.select([*STRATA, ORIGIN_STRATUM, "persona_id"]).iter_rows()
    for index, row in enumerate(values):
        stratum = tuple(row[: len(STRATA)])
        origin = row[-2]
        cell = (stratum, origin)
        cell_rows.setdefault(cell, []).append(index)
        if row[-1] in baseline_ids:
            cell_baseline[cell] = cell_baseline.get(cell, 0) + 1
    return (
        cell_rows,
        {cell: len(indices) for cell, indices in cell_rows.items()},
        cell_baseline,
    )


def _ordered_cells(
    *, cell_rows: dict[tuple[tuple[object, ...], str], list[int]]
) -> list[tuple[tuple[object, ...], str]]:
    """Return cells in sorted STRATA and origin order."""
    return sorted(cell_rows, key=lambda cell: (cell[0], cell[1]))


def _choose_cell_rows(
    *, positions: list[int], count: int, baseline_positions: set[int]
) -> list[int]:
    """Choose retained rows before replacement rows within one cell.

    Returns:
        Selected row positions for the cell.
    """
    retained = [position for position in positions if position in baseline_positions]
    replacements = [
        position for position in positions if position not in baseline_positions
    ]
    return (retained + replacements)[:count]


def _select_cell_counts(
    *,
    cell_capacity: dict[tuple[tuple[object, ...], str], int],
    cell_baseline: dict[tuple[tuple[object, ...], str], int],
    cell_rows: int,
    origin_targets: dict[str, int],
) -> dict[tuple[tuple[object, ...], str], int]:
    """Find an origin-quota selection, preserving STRATA where possible.

    Returns:
        Selected row counts by STRATA and origin cell.

    Raises:
        ValueError:
            If even the origin quotas cannot be satisfied.
    """
    strata = {cell[0] for cell in cell_capacity}
    stratum_targets = {
        stratum: sum(
            count
            for (cell_stratum, _), count in cell_baseline.items()
            if cell_stratum == stratum
        )
        for stratum in strata
    }
    exact = _greedy_strata_swaps(
        cell_capacity=cell_capacity,
        cell_baseline=cell_baseline,
        origin_targets=origin_targets,
    )
    if exact is None:
        exact = _run_cell_flow(
            cell_capacity=cell_capacity,
            cell_baseline=cell_baseline,
            cell_rows=cell_rows,
            origin_targets=origin_targets,
            stratum_targets=stratum_targets,
            preserve_strata=True,
        )
    if exact is not None:
        return exact
    fallback = _run_cell_flow(
        cell_capacity=cell_capacity,
        cell_baseline=cell_baseline,
        cell_rows=cell_rows,
        origin_targets=origin_targets,
        stratum_targets=stratum_targets,
        preserve_strata=False,
    )
    if fallback is None:
        raise ValueError("Origin quotas cannot be satisfied by the source rows")
    return fallback


def _greedy_strata_swaps(
    *,
    cell_capacity: dict[tuple[tuple[object, ...], str], int],
    cell_baseline: dict[tuple[tuple[object, ...], str], int],
    origin_targets: dict[str, int],
) -> dict[tuple[tuple[object, ...], str], int] | None:
    """Apply cheap deterministic same-STRATA swaps before general flow.

    Returns:
        Cell counts with exact origin margins, or ``None`` when the greedy pass
        cannot find all required swaps.
    """
    selected = dict(cell_baseline)
    baseline_origins = {
        origin: sum(
            count
            for (stratum, cell_origin), count in cell_baseline.items()
            if cell_origin == origin
        )
        for origin in origin_targets
    }
    surplus = {
        origin: baseline_origins[origin] - origin_targets[origin]
        for origin in origin_targets
        if baseline_origins[origin] > origin_targets[origin]
    }
    deficits = {
        origin: origin_targets[origin] - baseline_origins[origin]
        for origin in origin_targets
        if baseline_origins[origin] < origin_targets[origin]
    }
    strata = sorted({cell[0] for cell in cell_capacity})
    for stratum in strata:
        for surplus_origin in sorted(surplus):
            available = min(
                surplus[surplus_origin], cell_baseline.get((stratum, surplus_origin), 0)
            )
            for deficit_origin in sorted(deficits):
                capacity = cell_capacity.get((stratum, deficit_origin), 0)
                free = capacity - cell_baseline.get((stratum, deficit_origin), 0)
                moved = min(available, deficits[deficit_origin], free)
                if moved:
                    selected[(stratum, surplus_origin)] -= moved
                    selected[(stratum, deficit_origin)] = (
                        selected.get((stratum, deficit_origin), 0) + moved
                    )
                    surplus[surplus_origin] -= moved
                    deficits[deficit_origin] -= moved
                    available -= moved
                if not available:
                    break
    if any(surplus.values()) or any(deficits.values()):
        return None
    return {cell: selected.get(cell, 0) for cell in cell_capacity}


class _FlowEdge:
    """Mutable residual edge for the deterministic min-cost flow solver."""

    def __init__(self, *, target: int, reverse: int, capacity: int, cost: int) -> None:
        self.target = target
        self.reverse = reverse
        self.capacity = capacity
        self.cost = cost
        self.initial_capacity = capacity


def _run_cell_flow(
    *,
    cell_capacity: dict[tuple[tuple[object, ...], str], int],
    cell_baseline: dict[tuple[tuple[object, ...], str], int],
    cell_rows: int,
    origin_targets: dict[str, int],
    stratum_targets: dict[tuple[object, ...], int],
    preserve_strata: bool,
) -> dict[tuple[tuple[object, ...], str], int] | None:
    """Solve the quota assignment and return selected rows per cell.

    Returns:
        Selected row counts by cell, or ``None`` when the margins are infeasible.
    """
    cells = _ordered_cells(cell_rows={cell: [] for cell in cell_capacity})
    strata = sorted({cell[0] for cell in cells})
    origins = sorted(origin_targets)
    source, sink, node_count, stratum_nodes, origin_nodes = _flow_layout(
        strata=strata, origins=origins, preserve_strata=preserve_strata
    )
    graph: list[list[_FlowEdge]] = [[] for _ in range(node_count)]
    _add_flow_boundaries(
        graph=graph,
        source=source,
        sink=sink,
        strata=strata,
        origins=origins,
        stratum_nodes=stratum_nodes,
        origin_nodes=origin_nodes,
        origin_targets=origin_targets,
        stratum_targets=stratum_targets,
        cell_capacity=cell_capacity,
        preserve_strata=preserve_strata,
    )
    references: dict[tuple[tuple[object, ...], str], list[tuple[int, int]]] = {}
    for cell in cells:
        capacity = cell_capacity[cell]
        baseline = min(cell_baseline.get(cell, 0), capacity)
        start, end = _cell_endpoints(
            cell=cell,
            stratum_nodes=stratum_nodes,
            origin_nodes=origin_nodes,
            preserve_strata=preserve_strata,
        )
        references[cell] = [
            _add_flow_edge(graph, start, end, baseline, -1),
            _add_flow_edge(graph, start, end, capacity - baseline, 0),
        ]

    if preserve_strata:
        flow = _max_flow(graph=graph, source=source, sink=sink, required=cell_rows)
    else:
        for _, extra_reference in references.values():
            extra_edge = graph[extra_reference[0]][extra_reference[1]]
            extra_edge.capacity = 0
        flow = _max_flow(graph=graph, source=source, sink=sink, required=cell_rows)
        baseline_flow = flow
        for _, extra_reference in references.values():
            extra_edge = graph[extra_reference[0]][extra_reference[1]]
            extra_edge.capacity = extra_edge.initial_capacity
        flow += _max_flow(
            graph=graph, source=source, sink=sink, required=cell_rows - baseline_flow
        )
    if flow < cell_rows:
        return None
    return {
        cell: sum(
            graph[node][edge_index].initial_capacity - graph[node][edge_index].capacity
            for node, edge_index in edge_refs
        )
        for cell, edge_refs in references.items()
    }


def _flow_layout(
    *, strata: list[tuple[object, ...]], origins: list[str], preserve_strata: bool
) -> tuple[int, int, int, dict[tuple[object, ...], int], dict[str, int]]:
    """Allocate deterministic node identifiers for a flow network.

    Returns:
        Source, sink, node count, and node maps for the network.
    """
    if preserve_strata:
        stratum_nodes = {stratum: index + 1 for index, stratum in enumerate(strata)}
        origin_start = len(strata) + 1
        origin_nodes = {
            origin: origin_start + index for index, origin in enumerate(origins)
        }
        sink = origin_start + len(origins)
    else:
        origin_nodes = {origin: index + 1 for index, origin in enumerate(origins)}
        stratum_start = len(origins) + 1
        stratum_nodes = {
            stratum: stratum_start + index for index, stratum in enumerate(strata)
        }
        sink = stratum_start + len(strata)
    return 0, sink, sink + 1, stratum_nodes, origin_nodes


def _add_flow_boundaries(
    *,
    graph: list[list[_FlowEdge]],
    source: int,
    sink: int,
    strata: list[tuple[object, ...]],
    origins: list[str],
    stratum_nodes: dict[tuple[object, ...], int],
    origin_nodes: dict[str, int],
    origin_targets: dict[str, int],
    stratum_targets: dict[tuple[object, ...], int],
    cell_capacity: dict[tuple[tuple[object, ...], str], int],
    preserve_strata: bool,
) -> None:
    """Add source and sink margins to a cell flow network."""
    if preserve_strata:
        for stratum in strata:
            _add_flow_edge(
                graph, source, stratum_nodes[stratum], stratum_targets[stratum], 0
            )
        for origin in origins:
            _add_flow_edge(graph, origin_nodes[origin], sink, origin_targets[origin], 0)
        return
    for origin in origins:
        _add_flow_edge(graph, source, origin_nodes[origin], origin_targets[origin], 0)
    for stratum in strata:
        capacity = sum(cell_capacity.get((stratum, origin), 0) for origin in origins)
        _add_flow_edge(graph, stratum_nodes[stratum], sink, capacity, 0)


def _cell_endpoints(
    *,
    cell: tuple[tuple[object, ...], str],
    stratum_nodes: dict[tuple[object, ...], int],
    origin_nodes: dict[str, int],
    preserve_strata: bool,
) -> tuple[int, int]:
    """Return the network endpoints for one cell.

    Returns:
        Source-side and sink-side node identifiers for the cell edge.
    """
    if preserve_strata:
        return stratum_nodes[cell[0]], origin_nodes[cell[1]]
    return origin_nodes[cell[1]], stratum_nodes[cell[0]]


def _add_flow_edge(
    graph: list[list[_FlowEdge]], source: int, target: int, capacity: int, cost: int
) -> tuple[int, int]:
    """Add a forward and residual edge and return the forward reference.

    Returns:
        The graph list index identifying the forward edge.
    """
    source_index = len(graph[source])
    target_index = len(graph[target])
    graph[source].append(
        _FlowEdge(target=target, reverse=target_index, capacity=capacity, cost=cost)
    )
    graph[target].append(
        _FlowEdge(target=source, reverse=source_index, capacity=0, cost=-cost)
    )
    return source, source_index


def _max_flow(
    *, graph: list[list[_FlowEdge]], source: int, sink: int, required: int
) -> int:
    """Send flow along deterministic level-graph paths.

    Returns:
        Number of flow units sent.
    """
    flow = 0
    while flow < required:
        levels = _flow_levels(graph=graph, source=source)
        if levels[sink] < 0:
            break
        cursors = [0] * len(graph)
        while flow < required:
            amount = _send_blocking_flow(
                graph=graph,
                node=source,
                sink=sink,
                amount=required - flow,
                levels=levels,
                cursors=cursors,
            )
            if not amount:
                break
            flow += amount
    return flow


def _flow_levels(*, graph: list[list[_FlowEdge]], source: int) -> list[int]:
    """Build a residual BFS level graph.

    Returns:
        Level for every graph node; ``-1`` marks an unreachable node.
    """
    levels = [-1] * len(graph)
    levels[source] = 0
    queue = [source]
    for node in queue:
        for edge in graph[node]:
            if edge.capacity > 0 and levels[edge.target] < 0:
                levels[edge.target] = levels[node] + 1
                queue.append(edge.target)
    return levels


def _send_blocking_flow(
    *,
    graph: list[list[_FlowEdge]],
    node: int,
    sink: int,
    amount: int,
    levels: list[int],
    cursors: list[int],
) -> int:
    """Send one DFS path through the current level graph.

    Returns:
        Number of flow units sent.
    """
    if node == sink:
        return amount
    while cursors[node] < len(graph[node]):
        edge_index = cursors[node]
        edge = graph[node][edge_index]
        if edge.capacity > 0 and levels[edge.target] == levels[node] + 1:
            sent = _send_blocking_flow(
                graph=graph,
                node=edge.target,
                sink=sink,
                amount=min(amount, edge.capacity),
                levels=levels,
                cursors=cursors,
            )
            if sent:
                edge.capacity -= sent
                graph[edge.target][edge.reverse].capacity += sent
                return sent
        cursors[node] += 1
    return 0


def _largest_remainder_quotas(
    *, groups: list[pl.DataFrame], rows: int, group_key: str | None
) -> list[int]:
    """Allocate rows proportionally, resolving ties by sorted group code.

    Returns:
        One non-negative quota for each input group.
    """
    total = sum(group.height for group in groups)
    numerators = [group.height * rows for group in groups]
    quotas = [numerator // total for numerator in numerators]
    remaining = rows - sum(quotas)
    if group_key is None:
        tie_keys = list(range(len(groups)))
    else:
        tie_keys = [group.item(0, group_key) for group in groups]
    remainders = [numerator % total for numerator in numerators]
    remainder_order = sorted(
        range(len(groups)), key=lambda index: (-remainders[index], tie_keys[index])
    )
    for index in remainder_order[:remaining]:
        quotas[index] += 1
    return quotas


def _select_within_origin(
    *,
    group: pl.DataFrame,
    rows: int,
    mode: t.Literal["population_proportional", "stratified_round_robin"],
) -> pl.DataFrame:
    """Select rows from one origin using the existing freeze strategy.

    Returns:
        The selected rows in deterministic order within this origin.
    """
    strata_groups = group.partition_by(list(STRATA), maintain_order=True)
    if mode == "stratified_round_robin":
        selected: list[pl.DataFrame] = []
        depth = 0
        while len(selected) < rows:
            added = False
            for stratum in strata_groups:
                if depth < stratum.height:
                    selected.append(stratum.slice(depth, 1))
                    added = True
                    if len(selected) == rows:
                        break
            if not added:
                break
            depth += 1
        return pl.concat(selected)

    quotas = _largest_remainder_quotas(groups=strata_groups, rows=rows, group_key=None)
    return pl.concat(
        [stratum.head(quota) for stratum, quota in zip(strata_groups, quotas) if quota]
    )


def _validate_destinations(*, run_dir: Path, output: Path) -> tuple[Path, Path]:
    """Validate both direct output paths before either can be written.

    Args:
        run_dir:
            Canonical source run directory.
        output:
            Requested sample output path.

    Returns:
        Canonical lexical paths for the sample and adjacent manifest.

    Raises:
        ValueError:
            If either destination is outside the run or linked.
    """
    if ".." in output.parts:
        message = "Frozen sample output must not contain traversal components"
        raise ValueError(message)
    output_path = _lexical_absolute(path=output)
    manifest_path = output_path.with_suffix(".manifest.json")
    _validate_destination(run_dir=run_dir, path=output_path, label="sample output")
    _validate_destination(run_dir=run_dir, path=manifest_path, label="sample manifest")
    return output_path, manifest_path


def _lexical_absolute(*, path: Path) -> Path:
    """Make an absolute path without resolving symlink targets.

    Args:
        path:
            Path to normalise lexically.

    Returns:
        Absolute, lexically normalised path.
    """
    return Path(os.path.abspath(os.fspath(path)))


def _validate_destination(*, run_dir: Path, path: Path, label: str) -> None:
    """Reject aliases, links, and non-files at a direct output destination.

    Args:
        run_dir:
            Canonical source run directory.
        path:
            Destination path to inspect.
        label:
            Human-readable destination name for errors.

    Raises:
        ValueError:
            If the destination is not a direct, non-linked file path.
    """
    if ".." in path.parts or path.parent != run_dir:
        message = (
            f"Frozen {label} must remain inside its validated run directory "
            "as a direct path"
        )
        raise ValueError(message)
    current = Path(path.anchor)
    try:
        for component in path.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"Frozen {label} must not contain a symlink")
        entry = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError(f"Frozen {label} cannot be inspected") from error
    if path.is_symlink() or not stat.S_ISREG(entry.st_mode):
        raise ValueError(f"Frozen {label} must be a regular non-linked file")


def _write_manifest(*, path: Path, run_dir: Path, payload: dict[str, object]) -> None:
    """Atomically write a manifest through a private temporary file."""
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=run_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(content, encoding="utf-8", newline="\n")
        _validate_destination(run_dir=run_dir, path=path, label="sample manifest")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
