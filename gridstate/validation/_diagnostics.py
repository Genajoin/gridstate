"""Internal validation utilities: rebuild ``r``, ``H``, ``R⁻¹`` from the current
``Working`` state (after ``estimate()``).

Used by ``chi2_test`` and ``bad_data`` — both need the same
summary quantities, so they are collected in one file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.algebra.base import BaseAlgebra
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.utils import floored_sigma2, id_to_pos_map, sparse_diag
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


class Diagnostics(NamedTuple):
    """Summary data for the validation tests."""

    r: np.ndarray  # (m,) — residual z − h(state)
    H: csr_matrix  # (m × (2n−1))
    R_inv: csr_matrix  # (m × m) diagonal 1/σ²
    sigma2: np.ndarray  # (m,) σ² with regularization
    meas_index: MeasurementIndex
    layout: StateLayout
    network_pu: NetworkPU


def state_from_model(model: Working, network_pu: NetworkPU) -> tuple[np.ndarray, np.ndarray]:
    """Read ``(v_pu, delta_rad)`` of length ``n_bus`` from the current state of ``model``.

    Nodes without a ``voltage_magnitude > 0`` entry are initialized as ``V=1.0`` p.u.,
    angles default to 0.
    """
    nodes_arr = model.nodes.to_numpy()
    id_to_pos = id_to_pos_map(network_pu.bus_ids)
    v_pu = np.ones(network_pu.n_bus, dtype=np.float64)
    delta_rad = np.zeros(network_pu.n_bus, dtype=np.float64)
    for row in nodes_arr:
        if not row["status"]:
            continue
        pos = id_to_pos.get(int(row["id"]))
        if pos is None:
            continue
        vm = float(row["voltage_magnitude"])
        vn = float(row["voltage_nominal"])
        if vm > 0 and vn > 0:
            v_pu[pos] = vm / vn
        delta_rad[pos] = float(row["voltage_angle"])
    return v_pu, delta_rad


def compute_diagnostics(
    model: Working,
    measurements: _ArrayCollection,
) -> Diagnostics:
    """Rebuild ``r``, ``H``, ``R⁻¹`` for the current state of ``model``."""
    network_pu = model_to_pu(model)
    ybus, yf, yt = build_ybus(network_pu)
    z, R_matrix, meas_index = build_z_and_r(model, measurements, network_pu)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    v_pu, delta_rad = state_from_model(model, network_pu)
    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    h = algebra.evaluate_h(v_pu, delta_rad)
    H = algebra.evaluate_jacobian(v_pu, delta_rad)
    r = z - h

    sigma2 = floored_sigma2(R_matrix.diagonal())
    R_inv = cast("csr_matrix", sparse_diag(1.0 / sigma2))

    return Diagnostics(
        r=r,
        H=H,
        R_inv=R_inv,
        sigma2=sigma2,
        meas_index=meas_index,
        layout=layout,
        network_pu=network_pu,
    )
