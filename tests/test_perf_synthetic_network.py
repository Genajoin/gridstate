"""Performance / regression guard on a synthetic network of a few thousand buses.

The largest committed case has 118 buses; this guard builds a deterministic
meshed 110 kV network in memory (no data files), synthesises a fully redundant
measurement set from a known state with seeded noise, and runs the full IPM
pipeline through the contract facade.

What is asserted:

* convergence, the number of IPM iterations and the accuracy against the known
  state (stable counters, independent of the machine);
* a loose wall-time ratio against a calibration run on a 4x smaller network
  of the same structure, made in the same process. Absolute times are only
  reported (``record_property``), never compared, so a slow or loaded machine
  does not fail the test; a change of the asymptotic cost (e.g. a dense
  operation sneaking into the hot path) does.

Run explicitly with ``pytest -m slow tests/test_perf_synthetic_network.py -s``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from gridstate.constants import BranchType, NodeType
from gridstate.contract import SE_INPUT, SE_OUTPUT, SEInput, io_dtype
from gridstate.contract import run as run_se
from gridstate.pipeline import PipelineConfig
from gridstate.units import BASE_MVA, model_to_pu
from gridstate.working import Working
from gridstate.ybus import build_ybus
from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


pytestmark = pytest.mark.slow

_VN_KV = 110.0
_SIGMA_V_KV = 0.005 * _VN_KV
_SIGMA_PQ = 2.0  # MW / Mvar

# Guard thresholds (observed: 9 IPM iterations on both sizes, max |dV| ~ 4e-3
# p.u., max |d delta| ~ 5e-3 rad, time ratio ~ 11 for 4x the buses).
_MAX_ITERATIONS = 15
_TOL_V_PU = 0.01
_TOL_DELTA_RAD = 0.01
_MAX_TIME_RATIO = 40.0


def _synthetic_grid(rows: int, cols: int, seed: int = 0) -> tuple[Working, np.ndarray, np.ndarray]:
    """Meshed ``rows x cols`` grid: every row is a chain, the first column is a
    chain, and ~35 % of the remaining vertical links exist. Returns the model with
    measurements and the true ``(v_pu, delta_rad)`` in node-id order."""
    rng = np.random.default_rng(seed)
    n = rows * cols

    nodes = np.zeros(n, dtype=io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes))
    nodes["id"] = np.arange(1, n + 1)
    nodes["voltage_nominal"] = _VN_KV
    nodes["voltage_magnitude"] = _VN_KV
    nodes["status"] = True
    nodes["node_type"] = int(NodeType.PQ)
    nodes["node_type"][0] = int(NodeType.SLACK)

    edges = []
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if c + 1 < cols:
                edges.append((k, k + 1))
            if r + 1 < rows and (c == 0 or rng.random() < 0.35):
                edges.append((k, k + cols))
    e = np.asarray(edges)
    n_br = len(e)
    branches = np.zeros(n_br, dtype=io_dtype(SE_INPUT.branches, SE_OUTPUT.branches))
    branches["id"] = np.arange(1, n_br + 1)
    branches["from_node"] = e[:, 0] + 1
    branches["to_node"] = e[:, 1] + 1
    branches["parallel_id"] = 1
    branches["branch_type"] = int(BranchType.LINE)
    branches["status"] = True
    length_km = rng.uniform(5.0, 30.0, n_br)
    branches["resistance"] = 0.12 * length_km
    branches["reactance"] = 0.40 * length_km
    branches["susceptance"] = 2.8e-6 * length_km
    branches["tap_ratio"] = 1.0

    generators = np.zeros(1, dtype=SE_INPUT.generators.input_dtype())
    generators["id"] = 1
    generators["node_id"] = 1
    generators["status"] = True
    no_meas = np.zeros(0, dtype=io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements))
    network = Working.from_arrays(
        nodes=nodes, branches=branches, measurements=no_meas, generators=generators
    )

    # Known state: |V| within +-2 %, angles falling smoothly away from the slack.
    pu = model_to_pu(network)
    ybus, yf, yt = build_ybus(pu)
    bus_pos = {int(b): i for i, b in enumerate(pu.bus_ids)}
    order = np.array([bus_pos[int(i)] for i in nodes["id"]])
    v = 1.0 + rng.uniform(-0.02, 0.02, n)
    delta = -0.2 * np.cumsum(rng.uniform(0.0, 0.002, n))
    v[order[0]] = 1.0
    delta[order[0]] = 0.0
    volts = v * np.exp(1j * delta)
    s_bus = (volts * np.conj(ybus @ volts) * BASE_MVA)[order]
    br_pos = {int(b): i for i, b in enumerate(pu.branch_ids)}
    br_order = np.array([br_pos[int(i)] for i in branches["id"]])
    s_from = (volts[pu.from_idx] * np.conj(yf @ volts) * BASE_MVA)[br_order]
    s_to = (volts[pu.to_idx] * np.conj(yt @ volts) * BASE_MVA)[br_order]

    m = 3 * n + 4 * n_br
    meas = np.zeros(m, dtype=io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements))
    meas["id"] = np.arange(1, m + 1)
    meas["status"] = True
    k = 0
    for kind, values, sigma in (
        (KIND_VOLTAGE, v[order] * _VN_KV, _SIGMA_V_KV),
        (KIND_POWER_INJECTION_P, s_bus.real, _SIGMA_PQ),
        (KIND_POWER_INJECTION_Q, s_bus.imag, _SIGMA_PQ),
    ):
        sl = slice(k, k + n)
        meas["object_type"][sl] = OBJ_NODE
        meas["object_id"][sl] = nodes["id"]
        meas["measurement_type"][sl] = kind
        meas["value"][sl] = values + rng.normal(0.0, sigma, n)
        meas["variance"][sl] = sigma * sigma
        meas["branch_side"][sl] = -1
        k += n
    for kind, values, side in (
        (KIND_POWER_P, s_from.real, 0),
        (KIND_POWER_Q, s_from.imag, 0),
        (KIND_POWER_P, s_to.real, 1),
        (KIND_POWER_Q, s_to.imag, 1),
    ):
        sl = slice(k, k + n_br)
        meas["object_type"][sl] = OBJ_BRANCH
        meas["object_id"][sl] = branches["id"]
        meas["measurement_type"][sl] = kind
        meas["value"][sl] = values + rng.normal(0.0, _SIGMA_PQ, n_br)
        meas["variance"][sl] = _SIGMA_PQ * _SIGMA_PQ
        meas["branch_side"][sl] = side
        k += n_br

    # Operating point of the node table (generation / load split of the injection).
    p, q = s_bus.real, s_bus.imag
    nodes["generation_p"] = np.where(p > 0, p, 0.0)
    nodes["load_p"] = np.where(p < 0, -p, 0.0)
    nodes["generation_q"] = np.where(q > 0, q, 0.0)
    nodes["load_q"] = np.where(q < 0, -q, 0.0)
    nodes["exist_gen"] = (p > 0).astype(np.int8)
    nodes["exist_load"] = (p <= 0).astype(np.int8)
    nodes["exist_gen"][0] = 1
    model = Working.from_arrays(
        nodes=nodes, branches=branches, measurements=meas, generators=generators
    )
    return model, v[order], delta[order]


def _timed_run(rows: int, cols: int) -> dict:
    model, v_true, delta_true = _synthetic_grid(rows, cols)
    se_in = SEInput.from_model(model)
    t0 = time.perf_counter()
    out = run_se(se_in, config=PipelineConfig(algorithm="ipm"))
    elapsed = time.perf_counter() - t0
    v_se = out.nodes["voltage_magnitude"] / _VN_KV
    delta_se = out.nodes["voltage_angle"] - out.nodes["voltage_angle"][0]
    return {
        "n_bus": rows * cols,
        "n_branch": len(model.branches),
        "n_meas": len(model.measurements),
        "success": bool(out.success),
        "status": out.convergence_status,
        "iterations": int(out.iterations),
        "seconds": elapsed,
        "max_dv_pu": float(np.max(np.abs(v_se - v_true))),
        "max_ddelta_rad": float(np.max(np.abs(delta_se - delta_true))),
    }


def test_synthetic_network_ipm_scales(record_property: pytest.RecordProperty) -> None:
    # Calibration: the smaller network twice (the first run also warms up
    # imports and JIT caches); the faster of the two is the reference.
    calib = min((_timed_run(25, 30) for _ in range(2)), key=lambda r: r["seconds"])
    large = _timed_run(50, 60)
    ratio = large["seconds"] / calib["seconds"]

    record_property("calibration", calib)
    record_property("large", large)
    record_property("time_ratio", ratio)
    print(
        f"\nsynthetic IPM: {calib['n_bus']} buses {calib['seconds']:.2f}s "
        f"({calib['iterations']} it), {large['n_bus']} buses {large['seconds']:.2f}s "
        f"({large['iterations']} it), ratio {ratio:.1f}"
    )

    for run in (calib, large):
        assert run["success"], run
        assert run["iterations"] <= _MAX_ITERATIONS, run
        assert run["max_dv_pu"] <= _TOL_V_PU, run
        assert run["max_ddelta_rad"] <= _TOL_DELTA_RAD, run
    assert large["n_bus"] >= 3000
    assert ratio <= _MAX_TIME_RATIO, (calib, large, ratio)
