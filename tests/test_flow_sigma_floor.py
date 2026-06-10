"""Тесты σ-floor real-flow мер от шкалы канала (``apply_flow_sigma_floor``).

Мини-модель: 2 узла 500 кВ + ветвь. Floor при kv_frac=0.010:
σ_min = 0.010·√3·500 ≈ 8.66 МВт → floor² ≈ 75.0 МВт².
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.constants import MeasurementObjectType, MeasurementType, NodeType
from gridstate.pipeline import PipelineConfig, _Ctx, _s_flow_sigma_floor
from gridstate.telemetry import apply_flow_sigma_floor
from gridstate.working import Working


FLOOR2_500 = (0.010 * np.sqrt(3.0) * 500.0) ** 2  # ≈ 75.0 МВт²


def _add_meas(model: Working, mid: int, mt: int, *, variance: float, is_pseudo: bool) -> None:
    model.measurements.add(
        {
            "id": mid,
            "object_type": int(MeasurementObjectType.BRANCH),
            "object_id": 10,
            "measurement_type": mt,
            "value": 20.0,
            "variance": variance,
            "weight": 1.0 / variance,
            "status": True,
            "quality": 0,
            "is_pseudo": is_pseudo,
        }
    )


def _build() -> Working:
    """2 узла 500 кВ + ветвь 10; меры добавляют тесты."""
    m = Working.empty()
    for nid in (1, 2):
        m.nodes.add(
            {
                "id": nid,
                "voltage_nominal": 500.0,
                "status": True,
                "node_type": int(NodeType.PQ),
            }
        )
    m.branches.add({"id": 10, "from_node": 1, "to_node": 2, "status": True})
    return m


def test_floor_raises_real_flow_not_pseudo() -> None:
    """Real P/Q-flow с крошечной variance поднимаются до floor²; pseudo — нет."""
    m = _build()
    _add_meas(m, 1, int(MeasurementType.POWER_P), variance=4.0, is_pseudo=False)
    _add_meas(m, 2, int(MeasurementType.POWER_Q), variance=4.0, is_pseudo=False)
    _add_meas(m, 3, int(MeasurementType.POWER_P), variance=4.0, is_pseudo=True)

    stats = apply_flow_sigma_floor(m, kv_frac=0.010)

    me = m.measurements.to_numpy()
    assert stats == {"checked": 2, "floored": 2}
    for i in (0, 1):  # real P и Q — подняты
        assert me[i]["variance"] == pytest.approx(FLOOR2_500)
        assert me[i]["weight"] == pytest.approx(1.0 / FLOOR2_500)
    # pseudo — не тронута
    assert me[2]["variance"] == pytest.approx(4.0)
    assert me[2]["weight"] == pytest.approx(1.0 / 4.0)


def test_floor_keeps_variance_above_floor() -> None:
    """Variance выше floor² не понижается (floor — нижняя граница, не override)."""
    m = _build()
    _add_meas(m, 1, int(MeasurementType.POWER_P), variance=400.0, is_pseudo=False)

    stats = apply_flow_sigma_floor(m, kv_frac=0.010)

    me = m.measurements.to_numpy()
    assert stats == {"checked": 1, "floored": 0}
    assert me[0]["variance"] == pytest.approx(400.0)


def test_floor_none_is_noop_in_pipeline_step() -> None:
    """flow_sigma_floor_kv_frac=None (default) → шаг пайплайна не меняет variance."""
    m = _build()
    _add_meas(m, 1, int(MeasurementType.POWER_P), variance=4.0, is_pseudo=False)
    before = m.measurements.to_numpy().copy()

    cfg = PipelineConfig()
    assert cfg.flow_sigma_floor_kv_frac is None
    stats = _s_flow_sigma_floor(_Ctx(model=m, cfg=cfg))

    assert "skipped" in stats
    after = m.measurements.to_numpy()
    np.testing.assert_array_equal(after["variance"], before["variance"])
    np.testing.assert_array_equal(after["weight"], before["weight"])


def test_floor_enabled_in_pipeline_step() -> None:
    """flow_sigma_floor_kv_frac=0.010 → шаг пайплайна применяет floor к real-мере."""
    m = _build()
    _add_meas(m, 1, int(MeasurementType.POWER_P), variance=4.0, is_pseudo=False)

    cfg = PipelineConfig(flow_sigma_floor_kv_frac=0.010)
    stats = _s_flow_sigma_floor(_Ctx(model=m, cfg=cfg))

    assert stats == {"checked": 1, "floored": 1}
    me = m.measurements.to_numpy()
    assert me[0]["variance"] == pytest.approx(FLOOR2_500)
