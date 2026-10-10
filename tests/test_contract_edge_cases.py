"""Edge cases of the contract containers and the ``.npz`` round trip.

Complements ``tests/test_contract_runtime.py`` and ``tests/test_serialize_npz.py``
(main paths) with the corners: default/empty outputs, row accessors on missing
ids, files without the optional keys, older derived blobs, legacy extra
columns, carriers without domain tables, and the compatibility shims.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from gridstate.contract import SE_INPUT, SE_OUTPUT, SEInput, SEOutput, load_se_input
from gridstate.contract import run as contract_run
from gridstate.contract.derived import DerivedInputs
from gridstate.contract.serialize import (
    _DERIVED_KEY,
    _META_KEY,
    load_se_input_npz,
    save_se_input,
)
from gridstate.contract.version import CONTRACT_VERSION
from gridstate.pipeline import PipelineConfig
from gridstate.working import Working


_IEEE_DIR = Path(__file__).parent / "test_data" / "ieee"


def _case14() -> SEInput:
    return load_se_input_npz(_IEEE_DIR / "case14.npz")


def _rewrite(path: Path, *, drop: tuple[str, ...] = (), replace: dict | None = None) -> None:
    """Rewrite an ``.npz`` without some keys and/or with replaced arrays."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files if k not in drop}
    arrays.update(replace or {})
    np.savez_compressed(path, **arrays)


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------


def test_default_output_is_empty_and_typed() -> None:
    out = SEOutput()
    for name in ("nodes", "branches", "measurements"):
        table = getattr(out, name)
        assert table.size == 0
        assert table.dtype == getattr(SE_OUTPUT, name).output_dtype()
    assert out.node(1) is None
    assert out.branch(1) is None
    assert out.measurement(1) is None
    assert not out.success
    assert np.isnan(out.objective_value)
    assert out.contract_version == CONTRACT_VERSION


def test_row_accessors_return_dicts_and_none_for_unknown_ids() -> None:
    out = contract_run(_case14(), config=PipelineConfig(algorithm="wls"), validate=False)
    assert out.success
    node_id = int(out.nodes["id"][0])
    branch_id = int(out.branches["id"][0])
    meas_id = int(out.measurements["id"][0])

    node = out.node(node_id)
    assert node is not None
    assert node["id"] == node_id
    assert set(node) == set(out.nodes.dtype.names or ())
    assert node["voltage_magnitude"] == float(out.nodes["voltage_magnitude"][0])

    branch = out.branch(branch_id)
    assert branch is not None
    assert branch["id"] == branch_id
    meas = out.measurement(meas_id)
    assert meas is not None
    assert meas["id"] == meas_id

    missing = 10**9
    assert out.node(missing) is None
    assert out.branch(missing) is None
    assert out.measurement(missing) is None


def test_seinput_constructors_carry_version_and_plans() -> None:
    model = _case14().model
    plans = DerivedInputs(voltage_nominal={1: 110.0})
    se_in = SEInput.from_model(model, derived=plans, contract_version="1.0.0")
    assert se_in.model is model
    assert se_in.derived is plans
    assert se_in.contract_version == "1.0.0"
    wrapped = load_se_input(model)
    assert wrapped.derived is None
    assert wrapped.contract_version == CONTRACT_VERSION
    assert wrapped.validate().ok


# ---------------------------------------------------------------------------
# .npz round trip
# ---------------------------------------------------------------------------


def test_save_without_suffix_returns_the_real_npz_path(tmp_path: Path) -> None:
    se_in = _case14()
    written = save_se_input(se_in, tmp_path / "case")
    assert written == tmp_path / "case.npz"
    assert written.exists()
    back = load_se_input_npz(written)
    assert len(back.model.nodes) == len(se_in.model.nodes)


def test_round_trip_without_plans_keeps_derived_none(tmp_path: Path) -> None:
    se_in = SEInput.from_model(_case14().model, contract_version=CONTRACT_VERSION)
    back = load_se_input_npz(save_se_input(se_in, tmp_path / "x.npz"))
    assert back.derived is None
    assert back.contract_version == CONTRACT_VERSION


def test_file_without_optional_keys_loads_with_defaults(tmp_path: Path) -> None:
    """No contract version and no derived blob: current version, ``derived=None``."""
    path = save_se_input(_case14(), tmp_path / "x.npz")
    _rewrite(path, drop=(_DERIVED_KEY, _META_KEY))
    back = load_se_input_npz(path)
    assert back.derived is None
    assert back.contract_version == CONTRACT_VERSION


def test_older_derived_blob_without_sigma_scales(tmp_path: Path) -> None:
    """Blobs written before the sigma-scale plans existed load them as ``None``."""
    path = save_se_input(_case14(), tmp_path / "x.npz")
    old_blob = {
        "topology_resolved": [1, 2],
        "telemetry_resolved": None,
        "telemetry_arg_keys": None,
        "telemetry_total_args": 3,
        "materialize_obs": None,
        "voltage_nominal": {1: 110.0},
    }
    blob = np.frombuffer(pickle.dumps(old_blob), dtype=np.uint8)
    _rewrite(path, replace={_DERIVED_KEY: blob})
    back = load_se_input_npz(path)
    assert back.derived is not None
    assert back.derived.topology_resolved == [1, 2]
    assert back.derived.telemetry_total_args == 3
    assert back.derived.voltage_nominal == {1: 110.0}
    assert back.derived.v_sigma2_scale is None
    assert back.derived.flow_sigma2_scale is None


def test_legacy_columns_are_copied_or_dropped(tmp_path: Path) -> None:
    """An older file may carry OUTPUT columns (copied as-is) and columns unknown to
    the current schema (dropped)."""
    path = save_se_input(_case14(), tmp_path / "x.npz")
    with np.load(path, allow_pickle=False) as npz:
        nodes = npz["nodes"]
    legacy_dt = np.dtype([*nodes.dtype.descr, ("p_inj_calc", "f8"), ("legacy_col", "f8")])
    legacy = np.zeros(len(nodes), dtype=legacy_dt)
    for name in nodes.dtype.names or ():
        legacy[name] = nodes[name]
    legacy["p_inj_calc"] = np.arange(len(nodes), dtype=np.float64)
    legacy["legacy_col"] = 42.0
    _rewrite(path, replace={"nodes": legacy})

    back = load_se_input_npz(path).model.nodes.to_numpy()
    assert "legacy_col" not in (back.dtype.names or ())
    np.testing.assert_array_equal(back["p_inj_calc"], np.arange(len(nodes), dtype=np.float64))
    np.testing.assert_array_equal(back["id"], nodes["id"])


def test_carrier_without_domain_tables(tmp_path: Path) -> None:
    """A carrier object exposing only the four contract tables saves and loads; the
    domain tables come back empty."""
    model = _case14().model

    class _Carrier:
        nodes = model.nodes
        branches = model.branches
        measurements = model.measurements
        generators = model.generators
        shunts = object()  # no to_numpy(): ignored

    path = save_se_input(SEInput.from_model(_Carrier()), tmp_path / "x.npz")
    with np.load(path, allow_pickle=False) as npz:
        assert not {"tap_steps", "load_characteristics", "shunts"} & set(npz.files)
    back = load_se_input_npz(path).model
    assert len(back.tap_steps) == 0
    assert len(back.shunts) == 0
    assert len(back.nodes) == len(model.nodes)


def test_saved_tables_use_the_input_dtype(tmp_path: Path) -> None:
    """Contract tables are written with ``input_dtype`` (no OUTPUT columns)."""
    path = save_se_input(_case14(), tmp_path / "x.npz")
    with np.load(path, allow_pickle=False) as npz:
        for name in ("nodes", "branches", "measurements"):
            assert npz[name].dtype == getattr(SE_INPUT, name).input_dtype()


# ---------------------------------------------------------------------------
# Facades and compatibility shims
# ---------------------------------------------------------------------------


def test_prepare_network_facade_validates_and_returns_a_copy() -> None:
    from gridstate.runtime import prepare_network

    se_in = _case14()
    before = se_in.model.nodes.to_numpy().copy()
    net = prepare_network(se_in)
    assert isinstance(net, Working)
    assert net is not se_in.model
    assert len(net.nodes) == len(se_in.model.nodes)
    np.testing.assert_array_equal(se_in.model.nodes.to_numpy(), before)
    unchecked = prepare_network(se_in, validate=False)
    np.testing.assert_array_equal(unchecked.nodes.to_numpy(), net.nodes.to_numpy())


def test_former_shunt_sanity_entry_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    from gridstate.contract import runtime as former
    from gridstate.pipeline import orchestration

    seen: dict = {}
    sentinel = object()

    def fake(**kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(orchestration, "shunt_sanity_rerun", fake)
    cfg = PipelineConfig()
    out = former._shunt_sanity_rerun(
        base_result="base",  # type: ignore[arg-type]
        pristine="model",
        config=cfg,
        derived=None,
        on_event=None,
    )
    assert out is sentinel
    assert seen == {
        "base_result": "base",
        "model": "model",
        "config": cfg,
        "derived": None,
        "on_event": None,
    }
