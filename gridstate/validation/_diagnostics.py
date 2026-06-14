"""Внутренние утилиты валидации: пересборка ``r``, ``H``, ``R⁻¹`` из текущего
состояния ``Working`` (после ``estimate()``).

Используется ``chi2_test`` и ``bad_data`` — оба нуждаются в одних и тех же
сводных величинах, поэтому собрано в один файл.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.algebra.base import BaseAlgebra
from gridstate.constants import SIGMA2_FLOOR
from gridstate.state import StateLayout
from gridstate.units import model_to_pu
from gridstate.ybus import build_ybus
from gridstate.z_vector import build_z_and_r


if TYPE_CHECKING:
    from gridstate.units import NetworkPU
    from gridstate.working import Working, _ArrayCollection
    from gridstate.z_vector import MeasurementIndex


class Diagnostics(NamedTuple):
    """Сводные данные для валидационных тестов."""

    r: np.ndarray  # (m,) — невязка z − h(state)
    H: csr_matrix  # (m × (2n−1))
    R_inv: csr_matrix  # (m × m) диагональ 1/σ²
    sigma2: np.ndarray  # (m,) σ² с регуляризацией
    meas_index: MeasurementIndex
    layout: StateLayout
    network_pu: NetworkPU


def state_from_model(model: Working, network_pu: NetworkPU) -> tuple[np.ndarray, np.ndarray]:
    """Прочитать ``(v_pu, delta_rad)`` длины ``n_bus`` из текущего состояния ``model``.

    Узлы без записи ``voltage_magnitude > 0`` инициализируются как ``V=1.0`` p.u.,
    углы по умолчанию 0.
    """
    nodes_arr = model.nodes.to_numpy()
    id_to_pos: dict[int, int] = {
        int(nid): pos for pos, nid in enumerate(network_pu.bus_ids.tolist())
    }
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
    """Пересобрать ``r``, ``H``, ``R⁻¹`` для текущего состояния ``model``."""
    network_pu = model_to_pu(model)
    ybus, yf, yt = build_ybus(network_pu)
    z, R_matrix, meas_index = build_z_and_r(model, measurements, network_pu)
    layout = StateLayout.from_slack(network_pu.n_bus, network_pu.slack_idx)

    v_pu, delta_rad = state_from_model(model, network_pu)
    algebra = BaseAlgebra(ybus, yf, yt, meas_index, layout, network_pu)
    h = algebra.evaluate_h(v_pu, delta_rad)
    H = algebra.evaluate_jacobian(v_pu, delta_rad)
    r = z - h

    sigma2 = R_matrix.diagonal().astype(np.float64).copy()
    sigma2[sigma2 < SIGMA2_FLOOR] = SIGMA2_FLOOR
    n_meas = sigma2.shape[0]
    R_inv = cast(
        "csr_matrix",
        csr_matrix(
            (1.0 / sigma2, (np.arange(n_meas), np.arange(n_meas))),
            shape=(n_meas, n_meas),
        ),
    )

    return Diagnostics(
        r=r,
        H=H,
        R_inv=R_inv,
        sigma2=sigma2,
        meas_index=meas_index,
        layout=layout,
        network_pu=network_pu,
    )
