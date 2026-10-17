"""Assembly of the measurement vector ``z``, the covariance ``R`` and the index ``MeasurementIndex``.

Reads the measurement collection (``Working.measurements``) and turns it into the
inputs expected by ``gridstate.algebra.base.BaseAlgebra``:

- ``z`` — vector of measurement values in p.u. (with conversion from the named
  units MW/MVAr/kV/A);
- ``R`` — diagonal sparse matrix with the variances ``σ²`` (in p.u.²);
- ``MeasurementIndex`` — a structure describing the link of each measurement to
  the function ``h(x)``: type, object (node/branch), positional index, side.

Measurements with ``status=False`` or ``quality=BAD`` are skipped.
Measurements with a non-finite value or variance, or variance ≤ 0, are
skipped with a warning.

**The branch side** is determined as follows:

1. if ``MEASUREMENT_DTYPE`` has the field ``branch_side`` (a contract
   field) — it is taken directly;
2. otherwise — a reverse lookup: by the measurement ``id`` the references
   ``ti_p_from / ti_q_from / ti_p_to / ti_q_to`` in the ``BRANCH_DTYPE`` row
   of the corresponding branch are checked.

If the side cannot be determined, the measurement is skipped with a warning.
"""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import csr_matrix, diags

from gridstate._arrays import meas_mask
from gridstate.bounds import resolve_bounds
from gridstate.constants import MeasurementObjectType, MeasurementType
from gridstate.utils import id_to_pos_map
from gridstate.working import table_view


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection


logger = logging.getLogger(__name__)

# Ids of measurements already reported as non-finite, per collection: z is
# rebuilt on every (re-)solve, and the warning is logged once per collection.
_NONFINITE_REPORTED: weakref.WeakKeyDictionary[object, set[int]] = weakref.WeakKeyDictionary()


# The measurement type encoding in ``MeasurementIndex.kind`` matches
# ``gridstate.constants.MeasurementType`` (plain ints for hot comparison loops).
KIND_POWER_P = int(MeasurementType.POWER_P)
KIND_POWER_Q = int(MeasurementType.POWER_Q)
KIND_VOLTAGE = int(MeasurementType.VOLTAGE)
KIND_CURRENT = int(MeasurementType.CURRENT)
KIND_POWER_INJECTION_P = int(MeasurementType.POWER_INJECTION_P)
KIND_POWER_INJECTION_Q = int(MeasurementType.POWER_INJECTION_Q)

# IPM mode: the nodal balance links Sbus(V, δ) with the variables
# Pgen/Qgen/Pnag/Qnag from the state-vector. ``z=0``, ``σ²=tiny`` (hard
# equality):
#     Sbus[i].real - (Pgen_est[i] - Pnag_est[i]) = 0
#     Sbus[i].imag - (Qgen_est[i] - Qnag_est[i]) = 0
# Nodes without a corresponding box-var enter the brackets as ``0`` —
# this is equivalent to the old ``zero_injection`` for transit nodes.
KIND_NODE_BALANCE_P = 6
KIND_NODE_BALANCE_Q = 7

# IPM mode: soft prior towards 0 for box-vars (penalizes the deviation of
# pn,qn,pg,qg from 0). Without the
# prior, the IPM on nodes with very wide boxes (BUS equivalents,
# e.g. box=[-500, 85000] MW) spread the residual over them and produced
# multi-gigawatt ΔPgen errors. h(x)=box-var, z=0; affects only the
# column of the corresponding box-var in the Jacobian.
KIND_BOX_PRIOR_PGEN = 8
KIND_BOX_PRIOR_QGEN = 9
KIND_BOX_PRIOR_PNAG = 10
KIND_BOX_PRIOR_QNAG = 11

# Measured object type (``gridstate.constants.MeasurementObjectType``).
OBJ_NODE = int(MeasurementObjectType.NODE)
OBJ_BRANCH = int(MeasurementObjectType.BRANCH)
OBJ_GENERATOR = int(MeasurementObjectType.GENERATOR)

SIDE_FROM = 0
SIDE_TO = 1
SIDE_NONE = -1


@dataclass
class MeasurementIndex:
    """Description of each row of the vector ``z``.

    The length of all ``m`` arrays matches the length of ``z``. All ``object_pos`` are
    *positional* indices into ``NetworkPU.bus_ids``/``branch_ids`` (not ``id``).

    Attributes:
        kind: measurement type (``MeasurementType``).
        object_kind: 0=Node / 1=Branch / 2=Generator.
        object_pos: positional index of the object.
        branch_side: branch side (0=from, 1=to, -1=not a branch).
        meas_id: the original ``Measurement.id`` — for writing ``estimated_si``
            and ``residual`` back after convergence.
    """

    kind: np.ndarray
    object_kind: np.ndarray
    object_pos: np.ndarray
    branch_side: np.ndarray
    meas_id: np.ndarray

    def __len__(self) -> int:
        return int(self.kind.shape[0])


def build_z_and_r(
    model: Working,
    measurements: _ArrayCollection,
    network_pu: NetworkPU,
) -> tuple[np.ndarray, csr_matrix, MeasurementIndex]:
    """Assemble ``(z, R, meas_index)`` from the active measurements.

    Args:
        model: the model — needed to resolve ``object_id`` into a positional index
            and for the base voltage when converting kV/A → p.u.
        measurements: measurement collection; only those with ``status=True`` and
            ``quality != BAD``, with a finite value and a finite positive
            variance.
        network_pu: internal p.u. representation of the network.

    Returns:
        z: (m,) f8 — values in p.u.;
        R: (m × m) sparse — diagonal ``σ² = variance`` (in p.u.²);
        meas_index: metadata for the subsequent h(x).
    """
    arr = table_view(measurements)
    base_mva = float(network_pu.base_mva)
    ids = arr["id"].astype(np.int64)
    kind = arr["measurement_type"].astype(np.int64)
    obj_kind = arr["object_type"].astype(np.int64)
    obj_id = arr["object_id"].astype(np.int64)
    value = arr["value"].astype(np.float64)
    variance = arr["variance"].astype(np.float64)

    keep = meas_mask(arr, exclude_bad=True)
    nonfinite = keep & ~(np.isfinite(value) & np.isfinite(variance))
    if nonfinite.any():
        _warn_nonfinite(measurements, ids[nonfinite], value[nonfinite], variance[nonfinite])
        keep &= ~nonfinite
    for i in np.where(keep & (variance <= 0))[0]:
        logger.warning("Measurement id=%d has variance=%g ≤ 0 — skipped", int(ids[i]), variance[i])
    keep &= ~(variance <= 0)

    pos = np.full(arr.size, -1, dtype=np.int64)
    side = np.full(arr.size, SIDE_NONE, dtype=np.int64)
    v_base = np.full(arr.size, np.nan, dtype=np.float64)

    # ----- Nodal measurements (and generator measurements — via the generator's node) -----
    bus_id_to_pos = id_to_pos_map(network_pu.bus_ids)
    is_node = keep & (obj_kind == OBJ_NODE)
    pos[is_node] = _lookup(bus_id_to_pos, obj_id[is_node])
    for i in np.where(is_node & (pos < 0))[0]:
        logger.warning(
            "Measurement id=%d refers to a missing node id=%d — skipped",
            int(ids[i]),
            int(obj_id[i]),
        )
    is_gen = keep & (obj_kind == OBJ_GENERATOR)
    for i in np.where(is_gen)[0]:
        gen = model.generators.get_by_id(int(obj_id[i]))
        if gen is None or int(gen.node_id) not in bus_id_to_pos:
            logger.warning(
                "Measurement id=%d on generator %d: generator/node not found — skipped",
                int(ids[i]),
                int(obj_id[i]),
            )
            continue
        pos[i] = bus_id_to_pos[int(gen.node_id)]
    at_node = (is_node | is_gen) & (pos >= 0)

    # ----- Branch measurements -----
    is_branch = keep & (obj_kind == OBJ_BRANCH)
    b_idx = np.where(is_branch)[0]
    if b_idx.size:
        branch_pos = _lookup(id_to_pos_map(network_pu.branch_ids), obj_id[b_idx])
        for i in b_idx[branch_pos < 0]:
            logger.warning(
                "Measurement id=%d refers to a missing branch id=%d — skipped",
                int(ids[i]),
                int(obj_id[i]),
            )
        found = branch_pos >= 0
        b_idx, branch_pos = b_idx[found], branch_pos[found]
        branches_arr = table_view(model.branches)
        rows = branches_arr[_lookup(id_to_pos_map(branches_arr["id"]), obj_id[b_idx])]
        b_side = _branch_sides(arr[b_idx], rows, kind[b_idx], ids[b_idx])
        for i in b_idx[b_side == SIDE_NONE]:
            logger.warning(
                "Could not determine the side (from/to) of measurement id=%d on branch %d — skipped",
                int(ids[i]),
                int(obj_id[i]),
            )
        ok = b_side != SIDE_NONE
        b_idx, branch_pos, b_side = b_idx[ok], branch_pos[ok], b_side[ok]
        pos[b_idx] = branch_pos
        side[b_idx] = b_side
        end = np.where(
            b_side == SIDE_FROM, network_pu.from_idx[branch_pos], network_pu.to_idx[branch_pos]
        )
        v_base[b_idx] = network_pu.bus_vn_kv[end]
    at_branch = np.zeros(arr.size, dtype=bool)
    at_branch[b_idx] = True

    for i in np.where(keep & ~np.isin(obj_kind, (OBJ_NODE, OBJ_BRANCH, OBJ_GENERATOR)))[0]:
        logger.warning(
            "Measurement id=%d has an unknown object_type=%d — skipped",
            int(ids[i]),
            int(obj_kind[i]),
        )

    # ----- Conversion to p.u. -----
    z = np.full(arr.size, np.nan, dtype=np.float64)
    var_pu = np.full(arr.size, np.nan, dtype=np.float64)
    power = np.isin(
        kind, (KIND_POWER_P, KIND_POWER_Q, KIND_POWER_INJECTION_P, KIND_POWER_INJECTION_Q)
    )
    node_power = at_node & power
    node_v = at_node & (kind == KIND_VOLTAGE)
    if np.any(at_node & ~node_power & ~node_v):
        bad = int(kind[np.where(at_node & ~node_power & ~node_v)[0][0]])
        raise ValueError(f"Measurement type {bad} is not supported on a node")
    branch_power = at_branch & np.isin(kind, (KIND_POWER_P, KIND_POWER_Q))
    branch_i = at_branch & (kind == KIND_CURRENT)
    if np.any(at_branch & ~branch_power & ~branch_i):
        bad = int(kind[np.where(at_branch & ~branch_power & ~branch_i)[0][0]])
        raise ValueError(f"Measurement type {bad} is not supported on a branch")
    for sel in (node_power, branch_power):
        z[sel] = value[sel] / base_mva
        var_pu[sel] = variance[sel] / (base_mva * base_mva)
    vb = network_pu.bus_vn_kv[pos[node_v]].astype(np.float64)
    z[node_v] = value[node_v] / vb
    var_pu[node_v] = variance[node_v] / (vb * vb)
    # Base current on the branch side: I_base = base_mva·1000 / (√3·V_base_kV) A.
    i_base = base_mva * 1000.0 / (np.sqrt(3.0) * v_base[branch_i])
    z[branch_i] = value[branch_i] / i_base
    var_pu[branch_i] = variance[branch_i] / (i_base * i_base)

    used = at_node | at_branch
    if not used.any():
        logger.warning("_ArrayCollection contains no valid measurement at all")
    r_matrix = cast("csr_matrix", diags(var_pu[used], format="csr"))
    meas_index = MeasurementIndex(
        kind=kind[used].astype(np.int8),
        object_kind=np.where(obj_kind[used] == OBJ_GENERATOR, OBJ_NODE, obj_kind[used]).astype(
            np.int8
        ),
        object_pos=pos[used],
        branch_side=side[used].astype(np.int8),
        meas_id=ids[used],
    )
    return z[used], r_matrix, meas_index


def measurement_bounds_pu(
    measurements: _ArrayCollection,
    meas_index: MeasurementIndex,
    network_pu: NetworkPU,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds ``[lo, hi]`` (p.u.) of the measured values of the rows of ``meas_index``.

    Read from ``min_value`` / ``max_value`` and converted like the values in
    :func:`build_z_and_r`. Unset sides (see :func:`gridstate.bounds.resolve_bounds`:
    a both-zero or both-sentinel pair, an inverted pair) and rows without the
    columns or without a measurement row (balance / prior rows) are ``∓inf``.
    """
    m = len(meas_index)
    lo = np.full(m, -np.inf)
    hi = np.full(m, np.inf)
    arr = table_view(measurements)
    names = arr.dtype.names or ()
    if "min_value" not in names or "max_value" not in names:
        return lo, hi
    row_of = id_to_pos_map(arr["id"].astype(np.int64))
    base_mva = float(network_pu.base_mva)
    for k in range(m):
        i = row_of.get(int(meas_index.meas_id[k]))
        if i is None:
            continue
        lo_k, hi_k = resolve_bounds(arr["min_value"][i], arr["max_value"][i])
        if not (np.isfinite(lo_k) or np.isfinite(hi_k)):
            continue
        kind = int(meas_index.kind[k])
        pos = int(meas_index.object_pos[k])
        if kind == KIND_VOLTAGE:
            scale = float(network_pu.bus_vn_kv[pos])
        elif kind == KIND_CURRENT:
            end = (
                network_pu.from_idx[pos]
                if int(meas_index.branch_side[k]) == SIDE_FROM
                else network_pu.to_idx[pos]
            )
            scale = base_mva * 1000.0 / (np.sqrt(3.0) * float(network_pu.bus_vn_kv[end]))
        else:
            scale = base_mva
        lo[k] = lo_k / scale
        hi[k] = hi_k / scale
    return lo, hi


def _warn_nonfinite(
    measurements: object, ids: np.ndarray, value: np.ndarray, variance: np.ndarray
) -> None:
    """Log skipped non-finite measurements (WARNING once per collection, then DEBUG)."""
    try:
        reported = _NONFINITE_REPORTED.setdefault(measurements, set())
    except TypeError:  # not weak-referenceable: no de-duplication
        reported = set()
    for mid, val, var in zip(ids.tolist(), value.tolist(), variance.tolist(), strict=True):
        level = logging.DEBUG if mid in reported else logging.WARNING
        logger.log(
            level,
            "Measurement id=%d has non-finite value=%g / variance=%g — skipped",
            mid,
            val,
            var,
        )
        reported.add(mid)


def _lookup(id_to_pos: dict[int, int], ids: np.ndarray) -> np.ndarray:
    """Positions of objects by ``id`` (no such ``id`` → ``−1``)."""
    return np.array([id_to_pos.get(i, -1) for i in ids.tolist()], dtype=np.int64)


def _branch_sides(
    meas: np.ndarray, rows: np.ndarray, kind: np.ndarray, ids: np.ndarray
) -> np.ndarray:
    """Branch side for each measurement (``SIDE_NONE`` — not determined).

    First the measurement's ``branch_side`` field (if it is in the dtype), otherwise
    a reverse lookup by the references ``ti_p_from/ti_q_from/ti_p_to/ti_q_to`` of the
    branch row. For current there are no separate ti fields — all four are checked.
    """
    side = np.full(ids.size, SIDE_NONE, dtype=np.int64)
    ref = {
        f: rows[f].astype(np.int64) == ids for f in ("ti_p_from", "ti_q_from", "ti_p_to", "ti_q_to")
    }
    p, q, cur = kind == KIND_POWER_P, kind == KIND_POWER_Q, kind == KIND_CURRENT
    # Order of assignments: the last one wins, so "to" comes before "from".
    side[p & ref["ti_p_to"]] = SIDE_TO
    side[p & ref["ti_p_from"]] = SIDE_FROM
    side[q & ref["ti_q_to"]] = SIDE_TO
    side[q & ref["ti_q_from"]] = SIDE_FROM
    side[cur & (ref["ti_p_to"] | ref["ti_q_to"])] = SIDE_TO
    side[cur & (ref["ti_p_from"] | ref["ti_q_from"])] = SIDE_FROM
    if meas.dtype.names is not None and "branch_side" in meas.dtype.names:
        direct = meas["branch_side"].astype(np.int64)
        explicit = (direct == SIDE_FROM) | (direct == SIDE_TO)
        side[explicit] = direct[explicit]
    return side
