"""Serialization SEInput → .npz, the input boundary of gridstate.

A regression for :mod:`gridstate.contract.serialize`: a bit-identical round trip
of the contract tables and ``DerivedInputs``, and end-to-end equivalence of
``run(SEInput)`` from the npz and from the original model (on a small synthetic
model, so it is cheap and needs no external data).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT, SEInput
from gridstate.contract import run as contract_run
from gridstate.contract.serialize import load_se_input_npz, save_se_input
from gridstate.pipeline import PipelineConfig
from tests.test_contract_runtime import _arrays_bit_identical, _make_model


def _input_columns_bit_identical(a: np.ndarray, b: np.ndarray, table: str) -> bool:
    """Compare only the INPUT columns (KEY + INPUT + WORKING) of two arrays."""
    in_dt = getattr(SE_INPUT, table).input_dtype()
    for col in in_dt.names:
        nan_aware = a.dtype[col].kind in "fc"
        if not np.array_equal(a[col], b[col], equal_nan=nan_aware):
            return False
    return True


def test_roundtrip_contract_tables_bit_identical(tmp_path):
    """Contract tables (nodes/branches/measurements/generators) are bit-for-bit on the INPUT columns."""
    se_in = SEInput.from_model(_make_model())
    p = save_se_input(se_in, tmp_path / "m.npz")
    se_out = load_se_input_npz(p)

    for table in ("nodes", "branches", "measurements", "generators"):
        a = getattr(se_in.model, table).to_numpy()
        b = getattr(se_out.model, table).to_numpy()
        # Generators have no OUTPUT - the full arrays are identical
        if table == "generators":
            assert _arrays_bit_identical(a, b), f"{table}: round-trip is not bit-for-bit"
        else:
            # Other tables: the OUTPUT columns in se_out are zeros (added by expand_to_io),
            # so we compare only the contract INPUT columns
            assert _input_columns_bit_identical(a, b, table), (
                f"{table}: INPUT columns are not bit-for-bit"
            )
    assert se_out.contract_version == se_in.contract_version
    assert se_out.derived is None  # model without plans


def test_npz_has_no_output_columns(tmp_path):
    """The saved NPZ contains only INPUT columns (no OUTPUT)."""
    from gridstate.contract.tables import SE_INPUT as _SE_INPUT

    se_in = SEInput.from_model(_make_model())
    p = save_se_input(se_in, tmp_path / "clean.npz")

    with np.load(p, allow_pickle=False) as npz:
        for table in ("nodes", "branches", "measurements"):
            arr = np.asarray(npz[table])
            in_dt = getattr(_SE_INPUT, table).input_dtype()
            # The array must have exactly input_dtype - no more, no fewer
            assert set(arr.dtype.names or ()) == set(in_dt.names or ()), (
                f"{table}: NPZ contains extra columns: "
                f"{set(arr.dtype.names or ()) - set(in_dt.names or ())}"
            )


def test_roundtrip_domain_tables(tmp_path):
    """Domain input tables tap_steps/load_characteristics/shunts survive
    save/load without loss."""
    from gridstate.contract import SE_INPUT
    from gridstate.working import Working

    taps = np.zeros(1, dtype=SE_INPUT.tap_steps.input_dtype())
    taps["id"] = [0]
    taps["branch_id"] = [100]
    taps["tap_ratio"] = [0.95]
    taps["shunt_factor"] = [1.0]

    lc = np.zeros(1, dtype=SE_INPUT.load_characteristics.input_dtype())
    lc["id"] = [0]
    lc["coeff_p_a2"] = [1.0]

    sh = np.zeros(2, dtype=SE_INPUT.shunts.input_dtype())
    sh["id"] = [0, 1]
    sh["node_id"] = [1, 3]
    sh["susceptance"] = [-150.0, -200.0]
    sh["status"] = [True, False]

    base = SEInput.from_model(_make_model()).model
    working = Working.from_arrays(
        nodes=base.nodes.to_numpy(),
        branches=base.branches.to_numpy(),
        measurements=base.measurements.to_numpy(),
        generators=base.generators.to_numpy(),
        tap_steps=taps,
        load_characteristics=lc,
        shunts=sh,
    )
    se_in = SEInput(model=working)
    p = save_se_input(se_in, tmp_path / "aux.npz")
    se_out = load_se_input_npz(p)

    for name, ref in (("tap_steps", taps), ("load_characteristics", lc), ("shunts", sh)):
        got = getattr(se_out.model, name).to_numpy()
        assert _arrays_bit_identical(got, ref), f"{name}: round-trip is not bit-for-bit"


def test_domain_tables_absent_loads_empty(tmp_path):
    """An npz without domain tables loads: the collections are empty, the run does not fail."""
    se_in = SEInput.from_model(_make_model())  # model without domain tables
    p = save_se_input(se_in, tmp_path / "noaux.npz")
    se_out = load_se_input_npz(p)
    for name in ("tap_steps", "load_characteristics", "shunts"):
        assert len(getattr(se_out.model, name)) == 0


def test_roundtrip_derived_plans(tmp_path):
    """``DerivedInputs`` (numeric step plans) are restored by the round trip."""
    from gridstate.contract.derived import DerivedInputs

    derived = DerivedInputs(
        topology_resolved=[("LINE", 100, False, None), ("NODE", 2, True, None)],
        telemetry_resolved={(1, "PN"): (12.5, 2, "guid-1", 0), (100, "QBEG"): (-3.0, 1, "g2", 1)},
        telemetry_arg_keys=[(1, "PN"), (100, "QBEG")],
        telemetry_total_args=42,
        materialize_obs={1: 10.0, 2: -5.0},
        voltage_nominal={1: 110.0, 3: 220.0},
    )
    se_in = SEInput.from_model(_make_model(), derived=derived)
    p = save_se_input(se_in, tmp_path / "d.npz")
    d = load_se_input_npz(p).derived

    assert d is not None
    assert d.topology_resolved == derived.topology_resolved
    assert d.telemetry_resolved == derived.telemetry_resolved
    assert d.telemetry_arg_keys == derived.telemetry_arg_keys
    assert d.telemetry_total_args == derived.telemetry_total_args
    assert d.materialize_obs == derived.materialize_obs
    assert d.voltage_nominal == derived.voltage_nominal


def _rewrite_contract_version(path, new_version: str):
    """Overwrite ``__contract_version__`` in the npz (everything else unchanged)."""
    with np.load(path, allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    arrays["__contract_version__"] = np.asarray(str(new_version))
    np.savez_compressed(path, **arrays)


def test_load_npz_raises_on_incompatible_version(tmp_path):
    """Loading an npz with an incompatible (different major) contract_version fails explicitly.

    The input boundary must reject data of an old schema (for example, under
    contract 1.0.0 after the 2.0.0 freeze) instead of silently loading an incompatible
    structure that would fail deeper with an obscure error.
    """
    from gridstate.contract.version import current_version

    se_in = SEInput.from_model(_make_model())
    p = save_se_input(se_in, tmp_path / "stale.npz")

    incompatible = f"{current_version().major - 1 if current_version().major > 0 else current_version().major + 1}.0.0"
    _rewrite_contract_version(p, incompatible)

    with pytest.raises(ValueError) as exc:
        load_se_input_npz(p)
    msg = str(exc.value)
    # The message carries both versions and a hint to regenerate.
    assert incompatible in msg
    assert "regenerate" in msg.lower()


def test_load_npz_accepts_current_version(tmp_path):
    """Sanity: an npz built by the current contract loads without a version error."""
    se_in = SEInput.from_model(_make_model())
    p = save_se_input(se_in, tmp_path / "fresh.npz")
    # Must not raise - the version matches the built-in one.
    se_out = load_se_input_npz(p)
    assert se_out.contract_version == se_in.contract_version


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_run_bit_identical_via_npz(tmp_path, algorithm):
    """End-to-end: ``run`` from the npz == ``run`` from the source model (bit for bit)."""
    cfg = PipelineConfig(algorithm=algorithm)
    model = _make_model()
    out_direct = contract_run(SEInput.from_model(model), config=cfg, validate=False)

    p = save_se_input(SEInput.from_model(_make_model()), tmp_path / "run.npz")
    out_npz = contract_run(load_se_input_npz(p), config=cfg, validate=False)

    assert out_npz.success == out_direct.success
    assert out_npz.iterations == out_direct.iterations
    assert np.max(np.abs(np.asarray(out_npz.v_pu) - np.asarray(out_direct.v_pu))) == 0.0
    assert np.max(np.abs(np.asarray(out_npz.delta_rad) - np.asarray(out_direct.delta_rad))) == 0.0
