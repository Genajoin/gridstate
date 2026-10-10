"""Tests of the SE data contract (``gridstate.contract``).

The contract is only a schema DECLARATION; pipeline behaviour is unchanged. The tests
check that:

1. the contract version is correct and the SemVer compatibility logic works;
2. the schema is self-consistent (valid dtypes, keys present, no duplicate columns);
3. the output layer of the contract matches :mod:`gridstate.result` (drift guard);
4. :func:`validate_input` accepts a correct input and catches problems.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import (
    CONTRACT_VERSION,
    SE_INPUT,
    SE_OUTPUT,
    ContractVersion,
    Role,
    current_version,
    is_data_compatible,
    validate_input,
)
from gridstate.contract.tables import TableSchema
from gridstate.result import BRANCH_OUTPUT_FIELDS, MEAS_OUTPUT_FIELDS, NODE_OUTPUT_FIELDS


# ---------------------------------------------------------------------------
# 1. Version
# ---------------------------------------------------------------------------


def test_contract_version_is_semver():
    v = current_version()
    assert str(v) == CONTRACT_VERSION
    assert (v.major, v.minor, v.patch) == tuple(int(p) for p in CONTRACT_VERSION.split("."))


def test_version_parse_roundtrip():
    assert str(ContractVersion.parse("2.5.13")) == "2.5.13"


@pytest.mark.parametrize("bad", ["1.2", "1.2.3.4", "a.b.c", "1.-2.3", ""])
def test_version_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        ContractVersion.parse(bad)


def test_version_compatibility_rules():
    schema = ContractVersion(1, 3, 0)
    # same major, data minor <= schema minor -> compatible
    assert ContractVersion(1, 0, 0).is_compatible_with(schema)
    assert ContractVersion(1, 3, 9).is_compatible_with(schema)
    # data minor > schema minor -> the data may carry unknown fields
    assert not ContractVersion(1, 4, 0).is_compatible_with(schema)
    # different major -> incompatible
    assert not ContractVersion(2, 0, 0).is_compatible_with(schema)
    assert not ContractVersion(0, 9, 0).is_compatible_with(schema)


def test_is_data_compatible_against_current():
    assert is_data_compatible(CONTRACT_VERSION)
    cur = current_version()
    assert not is_data_compatible(f"{cur.major + 1}.0.0")


# ---------------------------------------------------------------------------
# 2. Schema self-consistency
# ---------------------------------------------------------------------------

_ALL_TABLES = [SE_INPUT.nodes, SE_INPUT.branches, SE_INPUT.measurements, SE_INPUT.generators]


@pytest.mark.parametrize("table", _ALL_TABLES, ids=lambda t: t.name)
def test_table_no_duplicate_columns(table: TableSchema):
    names = [c.name for c in table.columns]
    assert len(names) == len(set(names)), f"duplicate column in {table.name}"


@pytest.mark.parametrize("table", _ALL_TABLES, ids=lambda t: t.name)
def test_table_keys_present_as_key_role(table: TableSchema):
    for k in table.key:
        col = table.column(k)
        assert col is not None, f"key {k} is missing in {table.name}"
        assert col.role is Role.KEY


@pytest.mark.parametrize("table", _ALL_TABLES, ids=lambda t: t.name)
def test_table_dtypes_build(table: TableSchema):
    # Both the input and the output dtype build without errors and contain the key.
    in_dt = table.input_dtype()
    assert in_dt.names is not None
    for k in table.key:
        assert k in in_dt.names
    # the output layer builds (may be empty for generators - no OUTPUT columns)
    table.output_dtype()


def test_input_tables_have_no_output_columns():
    # The input is a separate space: no OUTPUT columns at all.
    for table in SE_INPUT.tables():
        assert table.column_names(Role.OUTPUT) == (), f"{table.name}: OUTPUT column in the input"


def test_output_tables_are_key_plus_output_only():
    for table in SE_OUTPUT.tables():
        for c in table.columns:
            assert c.role in (Role.KEY, Role.OUTPUT), (
                f"{table.name}.{c.name}: not KEY/OUTPUT in the output"
            )


def test_nodes_share_only_vd_across_layers():
    # The only columns shared by the node input and output are id + V/delta
    # (in the input: initial guess/WORKING; in the output: solution/OUTPUT).
    shared = set(SE_INPUT.nodes.column_names()) & set(SE_OUTPUT.nodes.column_names())
    assert shared == {"id", "voltage_magnitude", "voltage_angle"}


# ---------------------------------------------------------------------------
# 4. Output layer == gridstate.result (drift guard)
# ---------------------------------------------------------------------------


def test_output_layer_matches_result_module():
    assert SE_OUTPUT.nodes.column_names(Role.OUTPUT) == NODE_OUTPUT_FIELDS
    assert SE_OUTPUT.branches.column_names(Role.OUTPUT) == BRANCH_OUTPUT_FIELDS
    assert SE_OUTPUT.measurements.column_names(Role.OUTPUT) == MEAS_OUTPUT_FIELDS


# ---------------------------------------------------------------------------
# 5. validate_input
# ---------------------------------------------------------------------------


def _make_min_model():
    """Minimal valid model (2 nodes + branch + measurement + generator)."""
    from gridstate.working import Working

    m = Working.empty()
    m.nodes.add({"id": 1, "voltage_nominal": 110.0, "node_type": 2, "balance_priority": 1})
    m.nodes.add({"id": 2, "voltage_nominal": 110.0})
    m.branches.add({"id": 10, "from_node": 1, "to_node": 2, "resistance": 1.0, "reactance": 10.0})
    m.generators.add({"id": 100, "node_id": 1, "power_output": 50.0})
    m.measurements.add(
        {"id": 1000, "object_type": 0, "object_id": 1, "measurement_type": 2, "value": 110.0}
    )
    return m


def test_validate_input_accepts_full_model():
    model = _make_min_model()
    report = validate_input(model)
    assert report.ok, [str(i) for i in report.issues]


def test_validate_input_flags_missing_table_in_mapping():
    # Mapping without a branches table -> missing_table.
    nodes = np.zeros(1, dtype=SE_INPUT.nodes.input_dtype())
    report = validate_input({"nodes": nodes})
    assert not report.ok
    kinds = {(i.table, i.kind) for i in report.issues}
    assert ("branches", "missing_table") in kinds


def test_validate_input_flags_missing_required_column():
    # nodes without voltage_nominal (required INPUT) -> missing_column.
    full = SE_INPUT.nodes.input_dtype()
    fields_wo_vn = [(n, full.fields[n][0]) for n in full.names if n != "voltage_nominal"]
    nodes = np.zeros(1, dtype=np.dtype(fields_wo_vn))
    branches = np.zeros(0, dtype=SE_INPUT.branches.input_dtype())
    measurements = np.zeros(0, dtype=SE_INPUT.measurements.input_dtype())
    generators = np.zeros(0, dtype=SE_INPUT.generators.input_dtype())
    report = validate_input(
        {
            "nodes": nodes,
            "branches": branches,
            "measurements": measurements,
            "generators": generators,
        }
    )
    assert not report.ok
    assert any(
        i.table == "nodes" and i.column == "voltage_nominal" and i.kind == "missing_column"
        for i in report.issues
    )


def test_validate_input_version_incompatible():
    model = _make_min_model()
    cur = current_version()
    report = validate_input(model, data_version=f"{cur.major + 1}.0.0")
    assert not report.ok
    assert any(i.kind == "version_incompatible" for i in report.issues)


def test_validate_input_strict_raises():
    from gridstate.contract import ContractValidationError

    with pytest.raises(ContractValidationError):
        validate_input({}, strict=True)
