"""Юнит-тесты перф-флага ``refine_merge`` (слитый refine-re-solve)."""

from __future__ import annotations

from types import SimpleNamespace

from gridstate.pipeline import (
    STEPS,
    PipelineConfig,
    _s_v_mirror,
    _s_v_mirror_chain,
    _s_v_refine,
)


def test_default_off():
    assert PipelineConfig().refine_merge is False


def test_merged_step_registered_before_anti_overshoot():
    names = [s.name for s in STEPS]
    assert "refine_merged" in names
    assert names.index("refine_merged") == names.index("anti_overshoot") - 1
    assert names.index("refine_merged") > names.index("v_mirror_chain")


def test_individual_steps_skip_when_merged():
    ctx = SimpleNamespace(cfg=PipelineConfig(refine_merge=True))
    for step_fn in (_s_v_refine, _s_v_mirror, _s_v_mirror_chain):
        stats = step_fn(ctx)  # type: ignore[arg-type]
        assert "skipped" in stats and "refine_merged" in stats["skipped"]
