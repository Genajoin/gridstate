"""Working-layer boundary guard: INPUT/KEY columns of the contract are NOT mutated
by the pipeline - only WORKING/OUTPUT ones are.

This is the **invariant that justifies the "narrow clone"** (copy only WORKING+KEY,
keep INPUT shared/read-only): if even one production step writes to a column
marked ``Role.INPUT``/``Role.KEY`` in the contract (gridstate/contract/tables.py),
then either the contract markup is wrong, or the step will start mutating the
Input once the clone is removed. The guard catches both cases.

Method: run a full ``run`` (Input read-only -> working copy in ``result.model``),
compare all INPUT+KEY columns of the 4 collections by ``id`` between the Input and the working
final state.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT, Role
from gridstate.pipeline import PipelineConfig, run

# Reuse the synthetic model from the idempotency suite.
from tests.test_pipeline_idempotent import _make_model_with_reactor


def _assert_input_key_columns_unchanged(input_model, working_model) -> int:
    """Check that every INPUT/KEY column of the contract is bit-for-bit equal between Input and working.

    Matching is by ``id`` (measurements in working may have ADDED
    pseudo rows - those are not checked; the original ids must be preserved). Returns
    the number of (collection, column) pairs checked - for a liveness check.
    """
    pairs = [
        (SE_INPUT.nodes, input_model.nodes, working_model.nodes),
        (SE_INPUT.branches, input_model.branches, working_model.branches),
        (SE_INPUT.measurements, input_model.measurements, working_model.measurements),
        (SE_INPUT.generators, input_model.generators, working_model.generators),
    ]
    checked = 0
    for schema, in_coll, wk_coll in pairs:
        in_arr = in_coll.to_numpy()
        if len(in_arr) == 0:
            continue
        wk_arr = wk_coll.to_numpy()
        present = set(in_arr.dtype.names or ()) & set(wk_arr.dtype.names or ())
        cols = [c for c in schema.column_names(Role.KEY, Role.INPUT) if c in present]
        wk_by_id = {int(row["id"]): row for row in wk_arr}
        for row in in_arr:
            rid = int(row["id"])
            assert rid in wk_by_id, f"{schema.name} id={rid} vanished from the working copy"
            wrow = wk_by_id[rid]
            for c in cols:
                assert np.array_equal(row[c], wrow[c]), (
                    f"{schema.name}.{c} (role INPUT/KEY) changed at id={rid}: "
                    f"{row[c]!r} -> {wrow[c]!r} - either the contract markup is wrong, "
                    f"or a step mutates the Input"
                )
            checked += len(cols)
    return checked


# ---------------------------------------------------------------------------
# Synthetic model (always runs): covers the steps that need no derived inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_input_key_columns_immutable_synthetic(algorithm):
    """Full run on the synthetic model: INPUT/KEY columns of the 4 collections untouched."""
    m = _make_model_with_reactor()
    n_meas_in = len(m.measurements.to_numpy())

    r = run(m, config=PipelineConfig(algorithm=algorithm))

    checked = _assert_input_key_columns_unchanged(m, r.model)
    assert checked > 0, "the guard checked nothing - is the INPUT/KEY column list empty?"
    # Liveness: the pipeline really did something in the WORKING layer (added pseudo-measurements
    # and/or wrote a shunt) - otherwise the guard would pass vacuously.
    assert len(r.model.measurements.to_numpy()) >= n_meas_in


def test_working_columns_do_change_synthetic():
    """Control: WORKING columns (reactor shunt_b) do change -
    confirms the guard above is not vacuous (Input and working are distinguishable)."""
    m = _make_model_with_reactor()
    r = run(m, config=PipelineConfig(algorithm="wls"))
    # reactor at node 1 -> working shunt_b != 0, Input shunt_b == 0.
    assert float(m.nodes.get_by_id(1).shunt_b) == pytest.approx(0.0)
    assert float(r.model.nodes.get_by_id(1).shunt_b) != pytest.approx(0.0)
