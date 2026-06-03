"""Тёплый старт ``gridstate.pipeline.run`` через явный ``init_state``.

Input read-only ⇒ ``run`` каждый раз клонирует Input (V/δ плоские), поэтому
межпрогонный warm-start задаётся явно: прошлый ``SEResult`` передаётся как
``init_state``, его V/δ засеваются в рабочую копию до препроцессинга.

* unit: ``_seed_warm_start`` пишет V/δ из результата в клон;
* e2e synthetic: ``run(wls)`` → ``run(ipm, init_state=res)`` без крэша + событие;
* specs-gated: цепочка wls→ipm на реальной региональной модели.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from gridstate.pipeline import PipelineConfig, _build_working, _seed_warm_start, run


# Переиспользуем синтетическую модель из соседнего модуля.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline_idempotent import _make_model_with_reactor


def test_seed_warm_start_writes_vd():
    """``_seed_warm_start`` переносит V/δ из прошлого результата в рабочую копию."""
    m = _make_model_with_reactor()
    r = run(m, config=PipelineConfig(algorithm="wls"))

    # Подменим V/δ в Output-таблице прошлого результата на распознаваемые значения.
    out = r.outputs.nodes
    out["voltage_magnitude"] = 115.5
    out["voltage_angle"] = 0.037

    w = _build_working(m)
    seeded = _seed_warm_start(w, r)
    assert seeded == len(w.nodes.to_numpy())
    for n in w.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(115.5)
        assert float(n["voltage_angle"]) == pytest.approx(0.037)


def test_run_with_init_state_no_crash_and_event():
    """``run(wls)`` → ``run(ipm, init_state=res)``: без крэша + событие warm_start."""
    m = _make_model_with_reactor()
    r_wls = run(m, config=PipelineConfig(algorithm="wls"))

    events: list[dict] = []
    r_ipm = run(
        m,
        config=PipelineConfig(algorithm="ipm"),
        init_state=r_wls,
        on_event=events.append,
    )
    assert r_ipm is not None
    warm = [e for e in events if e.get("type") == "warm_start"]
    assert warm and warm[0]["seeded_nodes"] == len(m.nodes.to_numpy())
    # Input по-прежнему не тронут.
    for n in m.nodes.to_numpy():
        assert float(n["voltage_magnitude"]) == pytest.approx(110.0)
