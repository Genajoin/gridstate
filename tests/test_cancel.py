"""Cooperative cancellation (:mod:`gridstate.cancel`): a stop inside the solver step."""

from __future__ import annotations

from pathlib import Path

import pytest

from gridstate import Cancelled, cancel_scope
from gridstate.cancel import checkpoint
from gridstate.contract import run as run_se
from gridstate.contract.serialize import load_se_input_npz
from gridstate.pipeline import PipelineConfig


_CASE = Path(__file__).parent / "test_data" / "ieee" / "case118.npz"


@pytest.mark.parametrize("algorithm", ["wls", "ipm"])
def test_cancel_stops_inside_the_estimate_step(algorithm: str) -> None:
    """The check is polled on solver iterations: the run stops inside ``estimate``."""
    se_in = load_se_input_npz(_CASE)
    events: list[dict] = []
    polls_in_estimate = 0

    def check() -> bool:
        nonlocal polls_in_estimate
        last = events[-1] if events else {}
        inside = last.get("type") == "step_start" and last.get("name") == "estimate"
        if inside:
            polls_in_estimate += 1
        return inside and polls_in_estimate >= 2  # second iteration of the solver

    with cancel_scope(check), pytest.raises(Cancelled):
        run_se(
            se_in,
            config=PipelineConfig(algorithm=algorithm),
            on_event=events.append,
            validate=False,
        )
    assert polls_in_estimate == 2
    assert not any(e.get("type") == "step_done" and e.get("name") == "estimate" for e in events)
    assert not any(e.get("type") == "final" for e in events)


def test_cancel_scope_is_restored_and_run_completes() -> None:
    se_in = load_se_input_npz(_CASE)
    with cancel_scope(lambda: True), pytest.raises(Cancelled):
        checkpoint()
    checkpoint()  # outside the scope: no check
    out = run_se(se_in, config=PipelineConfig(algorithm="wls"), validate=False)
    assert out.success


def test_pipeline_run_cancel_argument() -> None:
    from gridstate.pipeline import run

    se_in = load_se_input_npz(_CASE)
    with pytest.raises(Cancelled):
        run(se_in.model, cancel=lambda: True)
