"""Declared order invariants of the pipeline step registry (``Step.after``)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from gridstate.pipeline import STEPS, Step, check_step_order
from gridstate.pipeline.runner import _REQUIRED_DERIVED_PLANS


def _by_name() -> dict[str, Step]:
    return {s.name: s for s in STEPS}


def _closure(name: str) -> set[str]:
    """All steps ``name`` relies on, directly or through other steps."""
    by_name = _by_name()
    seen: set[str] = set()
    todo = list(by_name[name].after)
    while todo:
        dep = todo.pop()
        if dep not in seen:
            seen.add(dep)
            todo.extend(by_name[dep].after)
    return seen


def test_registry_satisfies_declared_order() -> None:
    check_step_order(STEPS)


@pytest.mark.parametrize(
    ("step", "dep"),
    [
        # z-vector application reads the branch susceptance the tap step scales
        ("telemetry", "rpn"),
        # reactor shunts are summed over the reactors active after ON_LINE
        ("reactors", "topology"),
        # the second orphan-branch pass follows the first one
        ("disable_disconnected", "disable_orphan_branches"),
        # node off => generator off, after every node cascade
        ("generator_status", "disable_disconnected"),
        ("generator_status", "disable_isolated"),
        # the generation chain: aggregate -> materialize -> add_pseudo -> sigma floor
        ("materialize", "aggregate_generators"),
        ("add_pseudo", "materialize"),
        ("flow_sigma_floor", "add_pseudo"),
        # add_pseudo reads the nodes whose injections were released
        ("add_pseudo", "release_load_only_injections"),
        ("release_load_only_injections", "aggregate_generators"),
        # post-solve passes
        ("bad_data_repass", "estimate"),
        ("v_refine", "bad_data_repass"),
        ("v_mirror", "v_refine"),
        ("v_mirror_chain", "v_mirror"),
        ("refine_merged", "v_mirror_chain"),
        ("anti_overshoot", "refine_merged"),
    ],
)
def test_documented_invariant_is_declared(step: str, dep: str) -> None:
    assert dep in _closure(step)


def test_every_step_after_estimate_relies_on_it() -> None:
    names = [s.name for s in STEPS]
    for name in names[names.index("estimate") + 1 :]:
        assert "estimate" in _closure(name), name


def test_estimate_follows_the_measurement_preparation() -> None:
    assert {"telemetry", "aggregate_generators", "add_pseudo", "flow_sigma_floor"} <= _closure(
        "estimate"
    )


def test_required_derived_plans_belong_to_needs_derived_steps() -> None:
    by_name = _by_name()
    for name in _REQUIRED_DERIVED_PLANS:
        assert name in by_name, name
        assert by_name[name].needs_derived, name


def _swap(steps: list[Step], a: str, b: str) -> list[Step]:
    out = list(steps)
    i = next(k for k, s in enumerate(out) if s.name == a)
    j = next(k for k, s in enumerate(out) if s.name == b)
    out[i], out[j] = out[j], out[i]
    return out


def test_reordering_a_dependency_is_rejected() -> None:
    with pytest.raises(ValueError, match="'telemetry' must run after 'rpn'"):
        check_step_order(_swap(STEPS, "rpn", "telemetry"))


def test_reordering_the_generation_chain_is_rejected() -> None:
    with pytest.raises(ValueError, match="'add_pseudo' must run after 'materialize'"):
        check_step_order(_swap(STEPS, "materialize", "add_pseudo"))


def test_unknown_dependency_is_rejected() -> None:
    steps = [replace(s, after=("no_such_step",)) if s.name == "rpn" else s for s in STEPS]
    with pytest.raises(ValueError, match="unknown step 'no_such_step'"):
        check_step_order(steps)


def test_network_step_cannot_rely_on_a_measurement_step() -> None:
    steps = [replace(s, after=("telemetry",)) if s.name == "refine_slack" else s for s in STEPS]
    with pytest.raises(ValueError, match="network step 'refine_slack'"):
        check_step_order(steps)


def test_duplicate_step_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="registered twice"):
        check_step_order([*STEPS, STEPS[0]])


def test_unknown_toggle_is_rejected() -> None:
    steps = [replace(s, toggle="no_such_field") if s.name == "rpn" else s for s in STEPS]
    with pytest.raises(ValueError, match="'no_such_field' is not a config field"):
        check_step_order(steps)
