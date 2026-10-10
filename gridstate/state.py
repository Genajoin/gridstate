"""SE state vector.

In the basic WLS mode the vector E has dimension ``2·n_bus − 1``:

- ``δ`` — angles of all nodes *except* slack (``n_bus − 1`` components, radians);
- ``V`` — voltage magnitudes of all nodes (``n_bus`` components, p.u.).

In IPM mode the state additionally contains box variables of load and
generation:

- ``Pgen_i`` for nodes with ``exist_gen=1`` — length ``len(pgen_node_pos)``;
- ``Qgen_i`` — length ``len(qgen_node_pos)``;
- ``Pnag_i`` for nodes with ``exist_load=1`` — length ``len(pnag_node_pos)``;
- ``Qnag_i`` — length ``len(qnag_node_pos)``.

The layout is fixed: ``[δ, V, Pgen, Qgen, Pnag, Qnag]``. If all four
``*_node_pos`` are empty, the layout is identical to the old one (`size = 2·n_bus − 1`)
and WLS works unchanged.

The slack angle is fixed (usually 0 rad) and is not part of the state vector.
The layout to/from a flat array is handled by ``pack`` / ``unpack``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _empty_int64() -> np.ndarray:
    return np.array([], dtype=np.int64)


@dataclass(frozen=True)
class StateLayout:
    """Description of the state vector layout for a given network.

    Attributes:
        n_bus: number of nodes.
        slack_idx: positional index of the slack node (``0..n_bus−1``).
        non_slack_idx: indices of the remaining nodes (length ``n_bus−1``); the order
            corresponds to the δ part of the vector E.
        pgen_node_pos: positions (in bus ordering) of nodes with the variable
            ``Pgen``. Empty in WLS mode.
        qgen_node_pos: positions of nodes with the variable ``Qgen``.
        pnag_node_pos: positions of nodes with the variable ``Pnag``.
        qnag_node_pos: positions of nodes with the variable ``Qnag``.
    """

    n_bus: int
    slack_idx: int
    non_slack_idx: np.ndarray
    pgen_node_pos: np.ndarray = field(default_factory=_empty_int64)
    qgen_node_pos: np.ndarray = field(default_factory=_empty_int64)
    pnag_node_pos: np.ndarray = field(default_factory=_empty_int64)
    qnag_node_pos: np.ndarray = field(default_factory=_empty_int64)

    def __post_init__(self) -> None:
        if self.n_bus < 1:
            raise ValueError(f"n_bus must be ≥1, got {self.n_bus}")
        if not (0 <= self.slack_idx < self.n_bus):
            raise ValueError(f"slack_idx={self.slack_idx} is out of range [0, {self.n_bus})")
        if self.non_slack_idx.shape != (self.n_bus - 1,):
            raise ValueError(
                f"non_slack_idx must have shape ({self.n_bus - 1},), got {self.non_slack_idx.shape}"
            )
        if self.slack_idx in self.non_slack_idx.tolist():
            raise ValueError(f"slack_idx={self.slack_idx} must not be present in non_slack_idx")
        for name in ("pgen_node_pos", "qgen_node_pos", "pnag_node_pos", "qnag_node_pos"):
            arr = getattr(self, name)
            if arr.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape={arr.shape}")
            if arr.size and (arr.min() < 0 or arr.max() >= self.n_bus):
                raise ValueError(
                    f"{name} contains an index outside [0, {self.n_bus}): min={arr.min()} max={arr.max()}"
                )

    @property
    def size(self) -> int:
        """Dimension of the state vector.

        In WLS mode (all box fields empty) — ``2·n_bus − 1``.
        In IPM mode the total length of the four box sections is added.
        """
        return 2 * self.n_bus - 1 + self.n_box

    @property
    def n_box(self) -> int:
        """Total number of box variables (``Pgen + Qgen + Pnag + Qnag``)."""
        return (
            int(self.pgen_node_pos.size)
            + int(self.qgen_node_pos.size)
            + int(self.pnag_node_pos.size)
            + int(self.qnag_node_pos.size)
        )

    @property
    def has_box(self) -> bool:
        """``True`` if the layout includes box variables (IPM mode)."""
        return self.n_box > 0

    @property
    def offset_delta(self) -> int:
        """Offset of the δ section in the flat E (always 0)."""
        return 0

    @property
    def offset_v(self) -> int:
        """Offset of the V section in the flat E."""
        return self.n_bus - 1

    @property
    def offset_pgen(self) -> int:
        """Offset of the Pgen section in the flat E."""
        return 2 * self.n_bus - 1

    @property
    def offset_qgen(self) -> int:
        """Offset of the Qgen section in the flat E."""
        return self.offset_pgen + int(self.pgen_node_pos.size)

    @property
    def offset_pnag(self) -> int:
        """Offset of the Pnag section in the flat E."""
        return self.offset_qgen + int(self.qgen_node_pos.size)

    @property
    def offset_qnag(self) -> int:
        """Offset of the Qnag section in the flat E."""
        return self.offset_pnag + int(self.pnag_node_pos.size)

    @classmethod
    def from_slack(cls, n_bus: int, slack_idx: int) -> StateLayout:
        """Build a WLS layout (without box variables).

        ``non_slack_idx`` is filled in ascending order (``0, 1, ...``)
        skipping ``slack_idx``. The box sections are empty — the behaviour and size
        of the state vector are identical to the pre-IPM version.
        """
        non_slack = np.array([i for i in range(n_bus) if i != slack_idx], dtype=np.int64)
        return cls(n_bus=n_bus, slack_idx=slack_idx, non_slack_idx=non_slack)


def pack(
    delta: np.ndarray,
    v: np.ndarray,
    layout: StateLayout,
    *,
    pgen_estimated: np.ndarray | None = None,
    qgen_estimated: np.ndarray | None = None,
    pnag_estimated: np.ndarray | None = None,
    qnag_estimated: np.ndarray | None = None,
) -> np.ndarray:
    """Assemble E from the full state arrays.

    Basic WLS mode (layout without box-vars):
        ``E = [δ_non_slack, V]`` of length ``2·n_bus−1``.

    IPM mode (layout with box-vars):
        ``E = [δ_non_slack, V, Pgen, Qgen, Pnag, Qnag]``.

    Args:
        delta: (n_bus,) — voltage angles (radians), including slack.
        v: (n_bus,) — voltage magnitudes (p.u.), including slack.
        layout: layout.
        pgen_estimated: (len(pgen_node_pos),) — ``Pgen`` for box-vars.
            Required if ``layout.pgen_node_pos`` is non-empty.
        qgen_estimated, pnag_estimated, qnag_estimated: likewise.

    Returns:
        ``E`` of length ``layout.size``.
    """
    if delta.shape != (layout.n_bus,):
        raise ValueError(f"delta must have length {layout.n_bus}, got {delta.shape}")
    if v.shape != (layout.n_bus,):
        raise ValueError(f"v must have length {layout.n_bus}, got {v.shape}")

    e = np.empty(layout.size, dtype=np.float64)
    e[layout.offset_delta : layout.offset_delta + layout.n_bus - 1] = delta[layout.non_slack_idx]
    e[layout.offset_v : layout.offset_v + layout.n_bus] = v

    box_sections = (
        ("pgen_estimated", pgen_estimated, layout.pgen_node_pos, layout.offset_pgen),
        ("qgen_estimated", qgen_estimated, layout.qgen_node_pos, layout.offset_qgen),
        ("pnag_estimated", pnag_estimated, layout.pnag_node_pos, layout.offset_pnag),
        ("qnag_estimated", qnag_estimated, layout.qnag_node_pos, layout.offset_qnag),
    )
    for name, arr, pos, offset in box_sections:
        sz = pos.size
        if sz == 0:
            if arr is not None and arr.size != 0:
                raise ValueError(
                    f"{name} must be None or empty, the layout has no corresponding box-vars"
                )
            continue
        if arr is None:
            raise ValueError(
                f"{name} is required in IPM mode (the layout contains "
                f"{sz} box-vars of the corresponding type)"
            )
        if arr.shape != (sz,):
            raise ValueError(f"{name} must have length {sz}, got {arr.shape}")
        e[offset : offset + sz] = arr

    return e


def unpack(
    e: np.ndarray,
    layout: StateLayout,
    slack_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Unpack ``E`` back into the full ``(δ, V)`` of length ``n_bus``.

    Args:
        e: (2·n_bus−1,) — state vector.
        layout: layout.
        slack_delta: the angle substituted at the slack node position. Default
            0 rad (the classic SE normalization).

    Returns:
        (delta, v) — two arrays of length ``n_bus`` in the original node indexing.
    """
    if e.shape != (layout.size,):
        raise ValueError(f"e must have length {layout.size}, got {e.shape}")

    delta = np.empty(layout.n_bus, dtype=np.float64)
    delta[layout.slack_idx] = slack_delta
    delta[layout.non_slack_idx] = e[: layout.n_bus - 1]

    # In IPM mode the box sections follow V — the slice is limited to n_bus.
    v = e[layout.offset_v : layout.offset_v + layout.n_bus].astype(
        np.float64,
        copy=True,
    )
    return delta, v


def flat_start(layout: StateLayout) -> np.ndarray:
    """Flat-start.

    Base part: ``δ = 0`` (for all non-slack nodes), ``V = 1.0`` p.u.
    for all.

    In IPM mode (layout with box-vars) the box sections stay zeros. For
    feasibility they should be initialized with middle-of-box values
    via ``flat_start_with_box`` or corrected before calling the
    solver (``_project_to_interior`` in the IPM does this automatically).
    """
    e = np.zeros(layout.size, dtype=np.float64)
    e[layout.offset_v : layout.offset_v + layout.n_bus] = 1.0
    return e


def unpack_full(
    e: np.ndarray,
    layout: StateLayout,
    slack_delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unpack E with box sections.

    Returns the six-tuple ``(delta, v, pgen, qgen, pnag, qnag)`` where
    the box sections are arrays of length ``len(*_node_pos)`` (may be empty).
    For the basic WLS mode the last three are empty arrays.

    Used by the IPM solver to extract the values of the load/generation
    variables after optimization.
    """
    if e.shape != (layout.size,):
        raise ValueError(f"e must have length {layout.size}, got {e.shape}")
    delta, v = unpack(e, layout, slack_delta=slack_delta)

    pgen = e[layout.offset_pgen : layout.offset_qgen].astype(np.float64, copy=True)
    qgen = e[layout.offset_qgen : layout.offset_pnag].astype(np.float64, copy=True)
    pnag = e[layout.offset_pnag : layout.offset_qnag].astype(np.float64, copy=True)
    qnag = e[layout.offset_qnag : layout.offset_qnag + layout.qnag_node_pos.size].astype(
        np.float64, copy=True
    )
    return delta, v, pgen, qgen, pnag, qnag


def flat_start_with_box(
    layout: StateLayout,
    *,
    pgen_init: np.ndarray | None = None,
    qgen_init: np.ndarray | None = None,
    pnag_init: np.ndarray | None = None,
    qnag_init: np.ndarray | None = None,
) -> np.ndarray:
    """Flat start + explicit initialization of the box sections.

    The base part is like ``flat_start`` (δ=0, V=1). The box sections are filled with
    the passed ``*_init`` arrays — these are the values for
    ``Pgen/Qgen/Pnag/Qnag`` at the start of the IPM iterations. If the corresponding
    box section is empty in the layout, ``*_init`` must be None or empty.

    For feasibility the caller is responsible for making sure the ``*_init`` lie within
    the box bounds (or the solver projects them into the interior itself via
    ``_project_to_interior``).
    """
    return pack(
        delta=np.zeros(layout.n_bus, dtype=np.float64),
        v=np.ones(layout.n_bus, dtype=np.float64),
        layout=layout,
        pgen_estimated=pgen_init,
        qgen_estimated=qgen_init,
        pnag_estimated=pnag_init,
        qnag_estimated=qnag_init,
    )


@dataclass(frozen=True)
class InitialPoint:
    """Explicit solver initial point ``x0``: node V/δ keyed by node id.

    The initial point of :func:`gridstate.api.estimate` is otherwise read from
    the node table (``init="results"``), where ``voltage_magnitude`` also
    feeds the pseudo-V priors. An ``InitialPoint`` carries the start
    separately, so a warm start can change where the solver begins without
    changing the estimation problem.

    Attributes:
        node_ids: (k,) node ids.
        voltage_magnitude: (k,) V magnitude, kV (converted to p.u. with the
            node's ``voltage_nominal`` at solve time; a non-positive value
            starts the node at 1.0 p.u.).
        voltage_angle: (k,) V angle, rad.

    A repeated id keeps its last entry.
    """

    node_ids: np.ndarray
    voltage_magnitude: np.ndarray
    voltage_angle: np.ndarray

    def __post_init__(self) -> None:
        n = np.shape(self.node_ids)
        if np.shape(self.voltage_magnitude) != n or np.shape(self.voltage_angle) != n:
            raise ValueError(
                "node_ids, voltage_magnitude and voltage_angle must have the same shape, got "
                f"{np.shape(self.node_ids)}, {np.shape(self.voltage_magnitude)}, "
                f"{np.shape(self.voltage_angle)}"
            )

    @classmethod
    def from_result(cls, result: Any) -> InitialPoint:
        """V/δ of a previous solution (an ``SEResult`` or a compatible object).

        The source is ``result.outputs.nodes`` (the Output table), falling back
        to ``result.model.nodes``.
        """
        out_nodes = getattr(getattr(result, "outputs", None), "nodes", None)
        if (
            out_nodes is not None
            and getattr(out_nodes, "size", 0) > 0
            and "voltage_magnitude" in (out_nodes.dtype.names or ())
        ):
            src = out_nodes
        else:
            src = result.model.nodes.to_numpy()
        return cls(
            node_ids=np.asarray(src["id"], dtype=np.int64).copy(),
            voltage_magnitude=np.asarray(src["voltage_magnitude"], dtype=np.float64).copy(),
            voltage_angle=np.asarray(src["voltage_angle"], dtype=np.float64).copy(),
        )

    def as_dict(self) -> dict[int, tuple[float, float]]:
        """``{node id: (V kV, δ rad)}``; a repeated id keeps its last entry."""
        return {
            int(i): (float(vm), float(va))
            for i, vm, va in zip(
                self.node_ids.tolist(),
                self.voltage_magnitude.tolist(),
                self.voltage_angle.tolist(),
                strict=True,
            )
        }
