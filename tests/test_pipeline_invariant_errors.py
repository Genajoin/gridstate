"""Pipeline invariants raise explicit errors (they hold under ``python -O`` too)."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from gridstate.contract.derived import DerivedInputs
from gridstate.pipeline import PipelineConfig
from gridstate.pipeline.passes import _refine_two_pass, _s_anti_overshoot, _s_refine_merged
from gridstate.pipeline.preparation import (
    _s_release_load_only_injections,
    _s_telemetry,
    _s_topology,
    _s_voltage_nominal,
)


def _ctx(**kw: object) -> SimpleNamespace:
    base: dict[str, object] = {
        "model": None,
        "cfg": PipelineConfig(),
        "derived": None,
        "result": None,
        "released_injections": {},
    }
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize(
    ("fn", "plan"),
    [
        (_s_voltage_nominal, "derived.voltage_nominal"),
        (_s_topology, "derived.topology_resolved"),
        (_s_telemetry, "derived.telemetry_resolved"),
        (_s_release_load_only_injections, "derived.telemetry_arg_keys"),
    ],
)
def test_missing_derived_plan_raises_value_error(fn, plan) -> None:
    with pytest.raises(ValueError, match=plan):
        fn(_ctx())
    with pytest.raises(ValueError, match=plan):
        fn(_ctx(derived=DerivedInputs()))


def test_post_solve_steps_without_a_solution_raise() -> None:
    with pytest.raises(RuntimeError, match="'anti_overshoot' needs a solution"):
        _s_anti_overshoot(_ctx())
    with pytest.raises(RuntimeError, match="'refine_merged' needs a solution"):
        _s_refine_merged(_ctx(cfg=PipelineConfig(refine_merge=True)))
    with pytest.raises(RuntimeError, match="needs a solution"):
        _refine_two_pass(
            _ctx(),
            classify=lambda c: None,
            apply=lambda c, p: {},
            empty_stats=lambda p: {},
            unusable_reason="",
        )


def test_invariants_hold_under_optimized_python() -> None:
    code = (
        "from types import SimpleNamespace\n"
        "from gridstate.pipeline import PipelineConfig\n"
        "from gridstate.pipeline.passes import _s_anti_overshoot\n"
        "from gridstate.pipeline.preparation import _s_topology\n"
        "ctx = SimpleNamespace(model=None, cfg=PipelineConfig(), derived=None, result=None)\n"
        "for fn, exc in ((_s_topology, ValueError), (_s_anti_overshoot, RuntimeError)):\n"
        "    try:\n"
        "        fn(ctx)\n"
        "    except exc:\n"
        "        continue\n"
        "    raise SystemExit(f'{fn.__name__} did not raise')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-O", "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
