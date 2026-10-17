"""Typed derived-input tables of the ``.npz`` contract and the legacy reader.

Covers :mod:`gridstate.contract.derived_tables` (vocabulary, round trip of every
``DerivedInputs`` field through the file, validation of malformed tables) and
the trust boundary of :func:`gridstate.contract.serialize.load_se_input_npz`:
the default path reads plain arrays only, a legacy pickled blob is refused
without ``trusted=True``, and the contract version gates old data.
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gridstate.contract import SEInput
from gridstate.contract import run as contract_run
from gridstate.contract._legacy import LEGACY_DERIVED_KEY, legacy_kind
from gridstate.contract.derived import DerivedInputs
from gridstate.contract.derived_tables import (
    DERIVED_META_KEY,
    DERIVED_TABLE_PREFIX,
    DERIVED_TABLES,
    FIELD_TABLES,
    FLOW_KINDS,
    INJECTION_KINDS,
    SOURCE_GUID_WIDTH,
    TELEMETRY_KINDS,
    derived_from_tables,
    derived_to_tables,
    split_kind,
)
from gridstate.contract.serialize import (
    LegacyContractWarning,
    UntrustedLegacyFormatError,
    load_se_input_npz,
    save_se_input,
)
from gridstate.contract.version import (
    CONTRACT_VERSION,
    LEGACY_MAJOR,
    current_version,
    is_data_compatible,
)
from gridstate.pipeline import PipelineConfig


_IEEE_DIR = Path(__file__).parent / "test_data" / "ieee"
_VERSION_KEY = "__contract_version__"


def _case14() -> SEInput:
    return load_se_input_npz(_IEEE_DIR / "case14.npz")


def _full_derived() -> DerivedInputs:
    """Every field set, with each corner of the vocabulary."""
    return DerivedInputs(
        topology_resolved=[
            ("branch", 7, False, None),
            ("node", 2, True, None),
            ("generator", 3, None, "no_value"),
            ("node", 5, None, "partial_inputs"),
            ("branch", 8, None, "evaluation_error"),
        ],
        telemetry_resolved={
            (1, "v"): (225.0, 1, "{0f8fad5b-d9cb-469f-a165-70867728950e}", 0),
            (7, "p_from"): (150.0, 2, "g-p", 1),
            (7, "p_to"): (-148.5, 1, "g-p2", 0),
            (7, "q_from"): (12.0, 1, "", 2),
            (7, "q_to"): (None, 0, "", 0),
            (5, "p_gen"): (300.0, 1, "g-pg", 0),
            (5, "p_gen:0"): (10.0, 1, "g-u0", 0),
            (5, "q_gen:12"): (4.0, 1, "g-u12", 0),
            (5, "p_load"): (120.0, 1, "g-pn", 1),
            (5, "q_load"): (None, 3, "g-qn", 0),
        },
        telemetry_arg_keys=[(5, "p_load"), (1, "v"), (7, "p_from"), (5, "p_gen:0")],
        telemetry_total_args=17,
        materialize_obs={"gen_p": {5: 300.0}, "load_p": {2: 40.0, 3: -1.5}, "load_q": {}},
        voltage_nominal={1: 110.0, 3: 220.0},
        v_sigma2_scale={1: 0.25, 4: 4.0},
        flow_sigma2_scale={(7, "q_from"): 100.0, (8, "p_to"): 0.5},
        telemetry_bounds={
            (1, "v"): (200.0, 250.0),
            (7, "p_from"): (-np.inf, 300.0),
            (5, "p_gen:0"): (0.0, 50.0),
        },
    )


def _save(tmp_path: Path, derived: DerivedInputs | None, name: str = "x.npz") -> Path:
    return save_se_input(SEInput.from_model(_case14().model, derived=derived), tmp_path / name)


def _arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {k: npz[k] for k in npz.files}


def _write(path: Path, arrays: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", [*FIELD_TABLES, "telemetry_total_args"])
def test_every_field_round_trips_alone(tmp_path: Path, field: str) -> None:
    """Each field, set on its own, comes back equal (others stay ``None``)."""
    full = _full_derived()
    derived = DerivedInputs(**{field: getattr(full, field)})
    back = load_se_input_npz(_save(tmp_path, derived)).derived
    assert back == derived


def test_all_fields_round_trip_with_order(tmp_path: Path) -> None:
    derived = _full_derived()
    back = load_se_input_npz(_save(tmp_path, derived)).derived
    assert back == derived
    assert back is not None
    # Order is part of the plans: application order, mapping order, quantity order.
    assert back.topology_resolved == derived.topology_resolved
    assert list(back.telemetry_resolved or {}) == list(derived.telemetry_resolved or {})
    assert back.telemetry_arg_keys == derived.telemetry_arg_keys
    assert list(back.materialize_obs or {}) == ["gen_p", "load_p", "load_q"]
    # Python scalars, not numpy ones.
    value = (back.telemetry_resolved or {})[(7, "p_from")]
    assert [type(x) for x in value] == [float, int, str, int]
    assert type((back.topology_resolved or [])[0][2]) is bool


def test_empty_plans_are_not_none(tmp_path: Path) -> None:
    derived = DerivedInputs(
        topology_resolved=[],
        telemetry_resolved={},
        telemetry_arg_keys=[],
        materialize_obs={},
        voltage_nominal={},
        v_sigma2_scale={},
        flow_sigma2_scale={},
        telemetry_bounds={},
    )
    assert load_se_input_npz(_save(tmp_path, derived)).derived == derived


def test_default_derived_round_trips(tmp_path: Path) -> None:
    """``DerivedInputs()`` (all plans ``None``) is not the same as ``derived=None``."""
    assert load_se_input_npz(_save(tmp_path, DerivedInputs())).derived == DerivedInputs()
    assert load_se_input_npz(_save(tmp_path, None, "none.npz")).derived is None


def test_file_layout_is_plain_typed_arrays(tmp_path: Path) -> None:
    """Every derived table has its declared dtype, no object arrays, JSON metadata."""
    arrays = _arrays(_save(tmp_path, _full_derived()))
    assert all(arr.dtype != object for arr in arrays.values())
    assert LEGACY_DERIVED_KEY not in arrays
    for schema in DERIVED_TABLES:
        arr = arrays[DERIVED_TABLE_PREFIX + schema.name]
        assert arr.dtype == schema.numpy_dtype()
        assert arr.ndim == 1
    meta = json.loads(str(arrays[DERIVED_META_KEY]))
    assert meta == {
        "fields": list(FIELD_TABLES),
        "telemetry_total_args": 17,
        "materialize_quantities": ["gen_p", "load_p", "load_q"],
    }
    assert str(arrays[_VERSION_KEY]) == CONTRACT_VERSION
    tele = arrays[DERIVED_TABLE_PREFIX + "telemetry"]
    row = tele[(tele["object_id"] == 5) & (tele["component"] == 12)][0]
    assert row["kind"] == TELEMETRY_KINDS["q_gen"]


def _telemetry_plan(se_in: SEInput) -> DerivedInputs:
    """A telemetry plan that re-derives the measurements of an input (plus other plans)."""
    kinds = {(0, 0): "p_from", (0, 1): "p_to", (1, 0): "q_from", (1, 1): "q_to"}
    values: dict[tuple[int, str], tuple[float | None, int, str, int]] = {}
    for row in se_in.model.measurements.to_numpy():
        mt, side, oid = int(row["measurement_type"]), int(row["branch_side"]), int(row["object_id"])
        kind = {2: "v", 4: "p_gen", 5: "q_gen"}.get(mt) or kinds[(mt, side)]
        values[(oid, kind)] = (float(row["value"]), 1, f"m{int(row['id'])}", 0)
    nodes = se_in.model.nodes.to_numpy()
    return DerivedInputs(
        telemetry_resolved=values,
        telemetry_arg_keys=list(values),
        telemetry_total_args=len(values),
        topology_resolved=[("node", int(nodes["id"][0]), True, None)],
        v_sigma2_scale={int(nodes["id"][1]): 0.5},
        flow_sigma2_scale={
            (k[0], k[1]): 2.0 for k in list(values)[:3] if k[1].startswith(("p_", "q_"))
        },
    )


def test_run_from_file_equals_run_in_memory(tmp_path: Path) -> None:
    """Applying the plans read back from the file gives the same solution."""
    derived = _telemetry_plan(_case14())
    cfg = PipelineConfig()
    direct = contract_run(
        SEInput.from_model(_case14().model, derived=derived), config=cfg, validate=False
    )
    back = contract_run(load_se_input_npz(_save(tmp_path, derived)), config=cfg, validate=False)
    assert direct.success
    assert back.iterations == direct.iterations
    np.testing.assert_array_equal(back.v_pu, direct.v_pu)
    np.testing.assert_array_equal(back.delta_rad, direct.delta_rad)
    for col in direct.measurements.dtype.names or ():
        np.testing.assert_array_equal(back.measurements[col], direct.measurements[col])


# ---------------------------------------------------------------------------
# Vocabulary and validation
# ---------------------------------------------------------------------------


def test_split_kind() -> None:
    assert split_kind("p_from") == ("p_from", -1)
    assert split_kind("p_gen:3") == ("p_gen", 3)
    for bad in ("PBEG", "p_from:1", "p_gen:", "p_gen:-1", "p_gen:x", "v:0"):
        with pytest.raises(ValueError):
            split_kind(bad)


@pytest.mark.parametrize(
    "derived",
    [
        DerivedInputs(topology_resolved=[("line", 1, True, None)]),
        DerivedInputs(topology_resolved=[("node", 1, None, None)]),
        DerivedInputs(topology_resolved=[("node", 1, None, "skipped")]),
        DerivedInputs(telemetry_resolved={(1, "U"): (1.0, 1, "", 0)}),
        DerivedInputs(telemetry_resolved={(1, "v"): (1.0, 1, "x" * (SOURCE_GUID_WIDTH + 1), 0)}),
        DerivedInputs(telemetry_arg_keys=[(1, "q_to:2")]),
        DerivedInputs(materialize_obs={"load": {1: 1.0}}),
        DerivedInputs(flow_sigma2_scale={(1, "v"): 2.0}),
    ],
)
def test_writer_rejects_keys_outside_the_vocabulary(tmp_path: Path, derived: DerivedInputs) -> None:
    with pytest.raises(ValueError):
        _save(tmp_path, derived)


def _tables(derived: DerivedInputs) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return derived_to_tables(derived)


def test_reader_rejects_malformed_tables() -> None:
    tables, meta = _tables(_full_derived())
    assert derived_from_tables(tables, meta) == _full_derived()

    with pytest.raises(ValueError, match="missing"):
        derived_from_tables({k: v for k, v in tables.items() if k != "topology"}, meta)
    with pytest.raises(ValueError, match="unknown fields"):
        derived_from_tables(tables, {**meta, "fields": [*meta["fields"], "extra"]})
    with pytest.raises(ValueError, match="telemetry_total_args"):
        derived_from_tables(tables, {**meta, "telemetry_total_args": "3"})

    bad_kind = tables["telemetry"].copy()
    bad_kind["kind"][0] = 99
    with pytest.raises(ValueError, match="code 99"):
        derived_from_tables({**tables, "telemetry": bad_kind}, meta)

    dup = np.concatenate([tables["telemetry_order"], tables["telemetry_order"][:1]])
    with pytest.raises(ValueError, match="duplicate"):
        derived_from_tables({**tables, "telemetry_order": dup}, meta)

    narrow = tables["voltage_nominal"].astype([("node_id", "i8"), ("value", "f8")])
    with pytest.raises(ValueError, match="dtype"):
        derived_from_tables({**tables, "voltage_nominal": narrow}, meta)

    wrong_cols = tables["v_sigma2_scale"].astype([("node_id", "i4"), ("value", "f8")])
    with pytest.raises(ValueError, match="columns"):
        derived_from_tables({**tables, "v_sigma2_scale": wrong_cols}, meta)

    # Safe casts are accepted (a narrower integer / string column).
    small = tables["voltage_nominal"].astype([("node_id", "i2"), ("value", "f8")])
    assert derived_from_tables({**tables, "voltage_nominal": small}, meta) == _full_derived()


def test_reader_rejects_malformed_metadata(tmp_path: Path) -> None:
    path = _save(tmp_path, _full_derived())
    arrays = _arrays(path)
    arrays[DERIVED_META_KEY] = np.asarray("{not json")
    _write(path, arrays)
    with pytest.raises(ValueError, match="JSON"):
        load_se_input_npz(path)


# ---------------------------------------------------------------------------
# Trust boundary: no pickle on the default path
# ---------------------------------------------------------------------------


_EXECUTED: list[str] = []


def _record_execution(tag: str) -> None:
    _EXECUTED.append(tag)


class _Payload:
    """Unpickling it calls :func:`_record_execution` — stands for arbitrary code."""

    def __reduce__(self) -> tuple:
        return (_record_execution, ("payload ran",))


def _legacy_file(tmp_path: Path, blob: Any, *, version: str | None = "2.0.0") -> Path:
    path = _save(tmp_path, None, "legacy.npz")
    arrays = _arrays(path)
    arrays[LEGACY_DERIVED_KEY] = np.frombuffer(pickle.dumps(blob), dtype=np.uint8)
    if version is None:
        arrays.pop(_VERSION_KEY)
    else:
        arrays[_VERSION_KEY] = np.asarray(version)
    _write(path, arrays)
    return path


def _forbid_pickle(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("pickle used on the default load path")

    monkeypatch.setattr(pickle, "loads", boom)
    monkeypatch.setattr(pickle, "load", boom)
    monkeypatch.setattr(pickle, "Unpickler", boom)
    real_load = np.load

    def checked_load(*args: Any, **kwargs: Any) -> Any:
        assert kwargs.get("allow_pickle") is False, "np.load without allow_pickle=False"
        return real_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)


def test_default_load_never_unpickles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _save(tmp_path, _full_derived())
    _forbid_pickle(monkeypatch)
    assert load_se_input_npz(path).derived == _full_derived()


@pytest.mark.parametrize("version", ["2.0.0", None])
def test_crafted_legacy_file_is_refused_without_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    """A pickled blob is never unpickled by default, with or without a version key."""
    path = _legacy_file(tmp_path, {"telemetry_total_args": _Payload()}, version=version)
    _EXECUTED.clear()
    _forbid_pickle(monkeypatch)
    with pytest.raises(UntrustedLegacyFormatError, match="trusted=True"):
        load_se_input_npz(path)
    assert _EXECUTED == []


def test_crafted_blob_next_to_typed_tables_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A current file with an extra pickled blob reads the typed tables only."""
    path = _save(tmp_path, _full_derived())
    arrays = _arrays(path)
    arrays[LEGACY_DERIVED_KEY] = np.frombuffer(pickle.dumps(_Payload()), dtype=np.uint8)
    _write(path, arrays)
    _EXECUTED.clear()
    _forbid_pickle(monkeypatch)
    assert load_se_input_npz(path).derived == _full_derived()
    assert _EXECUTED == []


def test_trusted_legacy_file_maps_the_old_vocabulary(tmp_path: Path) -> None:
    blob = {
        "topology_resolved": [
            ("LINE", 7, False, None),
            ("NODE", 2, None, "skipped_no_value"),
            ("GENERATOR", 3, None, "skipped_partial_args"),
            ("NODE", 4, None, "skipped_formula_error"),
            ("REACTOR", 9, True, None),  # never applied by the core: dropped
        ],
        "telemetry_resolved": {
            (1, "U"): (225.0, 1, "g-v", 0),
            (7, "PBEG"): (150.0, 1, "g", 0),
            (7, "PEND"): (-149.0, 1, "g", 0),
            (7, "QBEG"): (10.0, 1, "g", 0),
            (7, "QEND"): (-9.0, 1, "g", 0),
            (5, "PG"): (300.0, 1, "g", 0),
            (5, "PN"): (100.0, 1, "g", 0),
            (5, "QG"): (30.0, 1, "g", 0),
            (5, "QN"): (None, 0, "", 0),
            (5, "PG_G3"): (12.0, 1, "g", 0),
            (5, "QG_G3"): (2.0, 1, "g", 0),
            (5, "XX"): (1.0, 1, "g", 0),  # no counterpart: dropped
        },
        "telemetry_arg_keys": [(5, "XX"), (1, "U"), (5, "PG_G3")],
        "telemetry_total_args": 11,
        "materialize_obs": {"load_p": {2: 1.0}, "load_q": {}, "gen_p": {}, "gen_q": {}},
        "voltage_nominal": None,
        "v_sigma2_scale": {1: 0.5},
        "flow_sigma2_scale": {(7, "QBEG"): 100.0},
    }
    path = _legacy_file(tmp_path, blob)
    with pytest.warns(LegacyContractWarning, match="conductance_from"):
        se_in = load_se_input_npz(path, trusted=True)
    d = se_in.derived
    assert d is not None
    assert se_in.contract_version == CONTRACT_VERSION
    assert d.topology_resolved == [
        ("branch", 7, False, None),
        ("node", 2, None, "no_value"),
        ("generator", 3, None, "partial_inputs"),
        ("node", 4, None, "evaluation_error"),
    ]
    assert list(d.telemetry_resolved or {}) == [
        (1, "v"),
        (7, "p_from"),
        (7, "p_to"),
        (7, "q_from"),
        (7, "q_to"),
        (5, "p_gen"),
        (5, "p_load"),
        (5, "q_gen"),
        (5, "q_load"),
        (5, "p_gen:3"),
        (5, "q_gen:3"),
    ]
    assert d.telemetry_arg_keys == [(1, "v"), (5, "p_gen:3")]
    assert d.telemetry_total_args == 11
    assert d.flow_sigma2_scale == {(7, "q_from"): 100.0}
    assert d.v_sigma2_scale == {1: 0.5}
    # Re-saving writes the current format, readable without trust.
    resaved = save_se_input(se_in, tmp_path / "resaved.npz")
    assert LEGACY_DERIVED_KEY not in _arrays(resaved)
    assert load_se_input_npz(resaved).derived == d


def test_telemetry_kind_tables_use_the_contract_vocabulary() -> None:
    """``gridstate.telemetry.kinds`` is keyed by exactly the derived-table kinds."""
    from gridstate.telemetry import kinds

    assert set(kinds.KIND_MAP) == {"v", *FLOW_KINDS}
    assert set(kinds.NODE_INJ_MAP) == set(INJECTION_KINDS)
    assert set(kinds.KIND_MAP) | set(kinds.NODE_INJ_MAP) == set(TELEMETRY_KINDS)


def test_legacy_kind_mapping() -> None:
    assert legacy_kind("PG_G12") == "p_gen:12"
    assert legacy_kind("QG_G0") == "q_gen:0"
    assert legacy_kind("PN_G1") is None
    assert legacy_kind(3) is None
    for kind in ("U", "PBEG", "PEND", "QBEG", "QEND", "PG", "PN", "QG", "QN"):
        mapped = legacy_kind(kind)
        assert mapped is not None
        split_kind(mapped)


# ---------------------------------------------------------------------------
# Contract version
# ---------------------------------------------------------------------------


def test_version_gates_legacy_major() -> None:
    assert current_version().major == LEGACY_MAJOR + 1
    assert is_data_compatible(CONTRACT_VERSION)
    assert not is_data_compatible(f"{LEGACY_MAJOR}.0.0")
    assert not is_data_compatible(f"{LEGACY_MAJOR}.9.9")


def test_legacy_version_without_blob_needs_trust(tmp_path: Path) -> None:
    """Major-2 tables alone (no derived inputs) are still refused by default."""
    path = _save(tmp_path, None)
    arrays = _arrays(path)
    arrays[_VERSION_KEY] = np.asarray(f"{LEGACY_MAJOR}.0.0")
    _write(path, arrays)
    with pytest.raises(UntrustedLegacyFormatError, match=f"{LEGACY_MAJOR}.0.0"):
        load_se_input_npz(path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se_in = load_se_input_npz(path, trusted=True)
    assert any(issubclass(w.category, LegacyContractWarning) for w in caught)
    assert se_in.derived is None
    assert se_in.contract_version == CONTRACT_VERSION


@pytest.mark.parametrize("trusted", [False, True])
def test_other_incompatible_versions_are_refused_even_if_trusted(
    tmp_path: Path, trusted: bool
) -> None:
    cur = current_version()
    for version in (
        f"{LEGACY_MAJOR - 1}.0.0",
        f"{cur.major}.{cur.minor + 1}.0",
        f"{cur.major + 1}.0.0",
    ):
        path = _save(tmp_path, _full_derived())
        arrays = _arrays(path)
        arrays[_VERSION_KEY] = np.asarray(version)
        _write(path, arrays)
        with pytest.raises(ValueError, match="Incompatible contract version") as exc:
            load_se_input_npz(path, trusted=trusted)
        assert not isinstance(exc.value, UntrustedLegacyFormatError)


def test_current_file_carries_current_version_and_loads_quietly(tmp_path: Path) -> None:
    path = _save(tmp_path, _full_derived())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        se_in = load_se_input_npz(path)
    assert se_in.contract_version == CONTRACT_VERSION
