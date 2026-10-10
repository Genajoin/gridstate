"""The pandapower adapter against pandapower itself (IEEE case14 / case30 / case118).

Skipped unless pandapower is installed (``pip install -e ".[test,test-models]"``)::

    pytest tests/test_from_pandapower.py -rs

What is compared (everything in p.u. on ``BASE_MVA = 100`` and the bus nominal
voltages, the bases pandapower uses as well):

* the bus admittance matrix built by gridstate from the adapter output against
  pandapower's internal ``Ybus`` after a power flow (``~1e-12``);
* branch flows ``S_from = V_f * conj(Yf V)`` / ``S_to`` against the pandapower
  power-flow results (orientation, pi-model charging split, tap side);
* nodal P/Q against the power flow at buses with a specified injection.

The adapter leaves out the transformer magnetising branch (``i0_percent`` /
``pfe_kw``). case14 and case30 have none; on case118 the comparison is done on a
copy with it removed, and a separate test pins the size and place of the
difference on the unmodified network.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest


pp = pytest.importorskip("pandapower")
nw = pytest.importorskip("pandapower.networks")

from gridstate.adapters import from_pandapower, measurement_array  # noqa: E402
from gridstate.contract import SE_INPUT  # noqa: E402
from gridstate.contract import run as run_se  # noqa: E402
from gridstate.pipeline import PipelineConfig  # noqa: E402
from gridstate.units import model_to_pu  # noqa: E402
from gridstate.ybus import build_ybus  # noqa: E402
from gridstate.z_vector import (  # noqa: E402
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
)


_TOL_Y = 1e-12


def _solved(name: str, *, drop_magnetising: bool = False) -> Any:
    net = getattr(nw, name)()
    if drop_magnetising and len(net.trafo):
        net.trafo["i0_percent"] = 0.0
        net.trafo["pfe_kw"] = 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pp.runpp(net, calculate_voltage_angles=True, numba=False)
    assert net.converged
    return net


def _ybus_pair(net: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    """(gridstate Ybus, pandapower Ybus in the same bus order, NetworkPU)."""
    pu = model_to_pu(from_pandapower(net).model)
    y_gs = build_ybus(pu)[0].toarray()
    y_pp = net._ppc["internal"]["Ybus"].toarray()
    pos = net._pd2ppc_lookups["bus"][np.asarray(pu.bus_ids, dtype=np.int64)]
    return y_gs, y_pp[np.ix_(pos, pos)], pu


def _pf_voltage(net: Any, pu: Any) -> np.ndarray:
    res = net.res_bus.loc[np.asarray(pu.bus_ids, dtype=np.int64)]
    return res.vm_pu.to_numpy() * np.exp(1j * np.radians(res.va_degree.to_numpy()))


@pytest.mark.parametrize(
    ("name", "drop_magnetising"),
    [("case14", False), ("case30", False), ("case118", True)],
)
def test_ybus_matches_pandapower(name: str, drop_magnetising: bool) -> None:
    net = _solved(name, drop_magnetising=drop_magnetising)
    y_gs, y_pp, _ = _ybus_pair(net)
    assert y_gs.shape == y_pp.shape == (len(net.bus), len(net.bus))
    np.testing.assert_allclose(y_gs, y_pp, rtol=0.0, atol=_TOL_Y)


def test_case118_difference_is_only_the_magnetising_admittance() -> None:
    """Current behaviour: on the unmodified case118 the adapter output differs from
    pandapower only in the 2x2 blocks of transformers with a non-zero ``i0_percent`` (the
    omitted magnetising branch; pandapower's T model spreads it over the diagonal
    and the off-diagonal entries of the transformer); every other entry agrees."""
    net = _solved("case118")
    y_gs, y_pp, pu = _ybus_pair(net)
    diff = np.abs(y_gs - y_pp)
    pos_of = {int(b): i for i, b in enumerate(pu.bus_ids)}
    allowed = np.zeros(diff.shape, dtype=bool)
    for hv, lv in zip(
        net.trafo.hv_bus[net.trafo.i0_percent != 0],
        net.trafo.lv_bus[net.trafo.i0_percent != 0],
        strict=True,
    ):
        block = [pos_of[int(hv)], pos_of[int(lv)]]
        allowed[np.ix_(block, block)] = True
    assert allowed.any()
    assert diff[~allowed].max() <= _TOL_Y
    # Small but not negligible against the diagonal (max |Y_ii| ~ 390 p.u.).
    assert 0.1 < diff[allowed].max() < 1.0


@pytest.mark.parametrize("name", ["case14", "case30"])
def test_branch_flows_match_the_power_flow(name: str) -> None:
    net = _solved(name)
    pu = model_to_pu(from_pandapower(net).model)
    _, yf, yt = build_ybus(pu)
    v = _pf_voltage(net, pu)
    s_from = v[pu.from_idx] * np.conj(yf @ v) * 100.0
    s_to = v[pu.to_idx] * np.conj(yt @ v) * 100.0
    n_line = len(net.line)
    # Adapter branch ids: lines 1..n_line in table order, then transformers.
    order = np.argsort(pu.branch_ids)
    s_from, s_to = s_from[order], s_to[order]
    np.testing.assert_allclose(s_from[:n_line].real, net.res_line.p_from_mw, atol=1e-8)
    np.testing.assert_allclose(s_from[:n_line].imag, net.res_line.q_from_mvar, atol=1e-8)
    np.testing.assert_allclose(s_to[:n_line].real, net.res_line.p_to_mw, atol=1e-8)
    np.testing.assert_allclose(s_to[:n_line].imag, net.res_line.q_to_mvar, atol=1e-8)
    if len(net.trafo):
        np.testing.assert_allclose(s_from[n_line:].real, net.res_trafo.p_hv_mw, atol=1e-8)
        np.testing.assert_allclose(s_from[n_line:].imag, net.res_trafo.q_hv_mvar, atol=1e-8)
        np.testing.assert_allclose(s_to[n_line:].real, net.res_trafo.p_lv_mw, atol=1e-8)
        np.testing.assert_allclose(s_to[n_line:].imag, net.res_trafo.q_lv_mvar, atol=1e-8)


@pytest.mark.parametrize("name", ["case14", "case30"])
def test_nodal_injections_match_the_power_flow(name: str) -> None:
    """P at every bus except the slack, Q at buses without a voltage-controlled
    generator: ``generation - load`` equals the power-flow network injection
    (``res_bus`` minus the bus shunt, which gridstate keeps in the Y-bus)."""
    net = _solved(name)
    nodes = from_pandapower(net).model.nodes.to_numpy()
    by_id = {int(r["id"]): r for r in nodes}
    slack = int(net.ext_grid.bus.iloc[0])
    pv = set(net.gen.bus[net.gen.in_service])
    p_inj, q_inj = _network_injection(net)
    for bus in net.bus.index:
        row = by_id[int(bus)]
        if bus != slack:
            assert row["generation_p"] - row["load_p"] == pytest.approx(p_inj[bus], abs=1e-9)
        if bus != slack and bus not in pv:
            assert row["generation_q"] - row["load_q"] == pytest.approx(q_inj[bus], abs=1e-9)
    assert by_id[slack]["node_type"] != by_id[next(iter(pv))]["node_type"]


def test_exact_measurements_reproduce_the_power_flow_state() -> None:
    """WLS on the adapter output with noise-free measurements from the power flow
    recovers the power-flow voltages."""
    net = _solved("case14")
    meas = _exact_measurements(net)
    out = run_se(from_pandapower(net, measurements=meas), config=PipelineConfig(algorithm="wls"))
    assert out.success
    ids = [int(x) for x in out.nodes["id"]]
    vn = net.bus.vn_kv.loc[ids].to_numpy()
    np.testing.assert_allclose(
        out.nodes["voltage_magnitude"] / vn, net.res_bus.vm_pu.loc[ids], atol=1e-6
    )
    slack = ids.index(int(net.ext_grid.bus.iloc[0]))
    va = np.degrees(out.nodes["voltage_angle"] - out.nodes["voltage_angle"][slack])
    va_pf = net.res_bus.va_degree.loc[ids].to_numpy()
    np.testing.assert_allclose(va, va_pf - va_pf[slack], atol=1e-5)


def _network_injection(net: Any) -> tuple[Any, Any]:
    """Nodal network injection (generation - load, MW / Mvar) of the power flow.

    pandapower's ``res_bus`` is the sum over all bus elements *including* the
    shunts; gridstate models the bus shunt inside the Y-bus, so its injection
    ``V * conj(Ybus V)`` excludes the shunt power.
    """
    p = -net.res_bus.p_mw.copy()
    q = -net.res_bus.q_mvar.copy()
    if len(net.shunt):
        p = p.add(net.res_shunt.p_mw.groupby(net.shunt.bus).sum(), fill_value=0.0)
        q = q.add(net.res_shunt.q_mvar.groupby(net.shunt.bus).sum(), fill_value=0.0)
    return p, q


def _exact_measurements(net: Any) -> np.ndarray:
    p_inj, q_inj = _network_injection(net)
    rows: list[tuple[int, int, int, float, float, int]] = []
    for bus in net.bus.index:
        vn = float(net.bus.vn_kv[bus])
        rows.append((OBJ_NODE, int(bus), KIND_VOLTAGE, net.res_bus.vm_pu[bus] * vn, 1e-2, -1))
        rows.append((OBJ_NODE, int(bus), KIND_POWER_INJECTION_P, p_inj[bus], 1.0, -1))
        rows.append((OBJ_NODE, int(bus), KIND_POWER_INJECTION_Q, q_inj[bus], 1.0, -1))
    for k, li in enumerate(net.line.index):
        rows.append((OBJ_BRANCH, k + 1, KIND_POWER_P, net.res_line.p_from_mw[li], 1.0, 0))
        rows.append((OBJ_BRANCH, k + 1, KIND_POWER_Q, net.res_line.q_from_mvar[li], 1.0, 0))
    arr = measurement_array(len(rows))
    for i, (ot, oid, mt, val, var, side) in enumerate(rows):
        arr[i]["id"] = i + 1
        arr[i]["object_type"] = ot
        arr[i]["object_id"] = oid
        arr[i]["measurement_type"] = mt
        arr[i]["value"] = val
        arr[i]["variance"] = var
        arr[i]["branch_side"] = side
        arr[i]["status"] = True
    return arr


def test_measurements_of_the_input_dtype_are_carried_over() -> None:
    net = _solved("case14")
    src = np.zeros(2, dtype=SE_INPUT.measurements.input_dtype())
    src["id"] = [7, 8]
    src["object_type"] = OBJ_NODE
    src["object_id"] = [0, 1]
    src["measurement_type"] = KIND_VOLTAGE
    src["value"] = [140.0, 139.0]
    src["variance"] = 1.0
    src["status"] = True
    model = from_pandapower(net, measurements=src).model
    got = model.measurements.to_numpy()
    assert got["id"].tolist() == [7, 8]
    assert got["value"].tolist() == [140.0, 139.0]
    assert len(from_pandapower(net).model.measurements) == 0


def test_line_ratings_and_branch_metadata() -> None:
    net = _solved("case30")
    branches = from_pandapower(net).model.branches.to_numpy()
    n_line = len(net.line)
    np.testing.assert_allclose(
        branches["current_limit_normal"][:n_line], net.line.max_i_ka.to_numpy() * 1000.0
    )
    assert (branches["parallel_id"] == 1).all()
    assert branches["id"].tolist() == list(range(1, len(branches) + 1))


def test_input_checks() -> None:
    net = nw.case14()
    net.sn_mva = 1.0
    with pytest.raises(ValueError, match="sn_mva"):
        from_pandapower(net)

    with pytest.raises(ValueError):
        from_pandapower(pp.create_empty_network(sn_mva=100.0))

    net = nw.case14()
    pp.create_ext_grid(net, bus=int(net.bus.index[3]), vm_pu=1.0)
    with pytest.raises(ValueError, match="ext_grid"):
        from_pandapower(net)
