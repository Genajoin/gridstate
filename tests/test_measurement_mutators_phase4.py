"""Ф4.1 (слайс 6): float-ядра measurement-мутаторов над контрактными массивами.

4 функции (`apply_voltage_range_filter`, `apply_voltage_meas_calibration_for_gen_nodes`,
`resolve_merged_measurement_conflicts`, `deactivate_orphan_measurements`) расщеплены на
тонкий адаптер (резолв PSC-енумов object/measurement-type → готовые int) + ЯДРО
`_*_on_arrays`, читающее/мутирующее только контрактные колонки `SE_INPUT.measurements`
(+ nodes/branches/generators), БЕЗ PSC/XML. Здесь — корректность ядер на голых массивах;
бит-в-бит публичного API — canon transitively + end-to-end дифф OLD-vs-NEW (4 региональные модели).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.contract import SE_INPUT
from gridstate.telemetry.measurements import (
    _deactivate_orphan_on_arrays,
    _resolve_merged_on_arrays,
)
from gridstate.telemetry.voltage_filter import (
    _voltage_meas_calibration_on_arrays,
    _voltage_range_filter_on_arrays,
)


OT_NODE, OT_BRANCH, OT_GEN = 0, 1, 2
MT_V, MT_PINJ, MT_QINJ = 2, 4, 5


def _meas(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.measurements.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _nodes(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.nodes.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _branches(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.branches.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


def _gens(rows: list[dict]) -> np.ndarray:
    arr = np.zeros(len(rows), dtype=SE_INPUT.generators.input_dtype())
    arr["status"] = True
    for i, row in enumerate(rows):
        for k, v in row.items():
            arr[i][k] = v
    return arr


VR_KW = {
    "ot_node": OT_NODE,
    "mt_v": MT_V,
    "upper_margin_pct": 10.0,
    "min_voltage_nominal_kv": 110.0,
    "upper_fallback_factor": 1.4,
    "action": "downweight",
    "detect_nominal_substitution": False,
    "nominal_substitution_eps": 0.001,
    "questionable_sigma2_multiplier": 100.0,
}


# ---------------------------------------------------------------- voltage_range_filter


def test_vrf_in_range_untouched():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 115.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["checked"] == 1 and stats["out_of_range"] == 0
    assert float(meas[0]["variance"]) == 1.0  # не тронут
    assert int(meas[0]["quality"]) == 0


def test_vrf_above_hi_downweighted():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 200.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["out_of_range"] == 1
    assert float(meas[0]["variance"]) == 100.0  # ×multiplier
    assert float(meas[0]["weight"]) == pytest.approx(0.01)
    assert int(meas[0]["quality"]) == 1
    assert bool(meas[0]["status"]) is True  # downweight, не деактивация


def test_vrf_below_lo_downweighted():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 50.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["out_of_range"] == 1  # 50 < lo=60


def test_vrf_deactivate_action():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 200.0,
                "variance": 1.0,
            }
        ]
    )
    kw = {**VR_KW, "action": "deactivate"}
    _voltage_range_filter_on_arrays(meas, nodes, **kw)
    assert bool(meas[0]["status"]) is False


def test_vrf_v_le_zero_guard():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 0.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["out_of_range"] == 1
    assert int(meas[0]["quality"]) == 1  # V≤0 guard, не зависит от lo/hi


def test_vrf_skip_low_vnom():
    # vnom=10 < min_voltage_nominal_kv=110 → пропускается (ген-шина).
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 10.0, "voltage_critical": 5.0, "voltage_max": 11.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 999.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["checked"] == 0 and stats["out_of_range"] == 0


def test_vrf_skip_non_v_meas():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_PINJ,
                "value": 999.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["checked"] == 0


def test_vrf_nominal_substitution():
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 121.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 110.0,
                "variance": 1.0,
            }
        ]
    )
    kw = {**VR_KW, "detect_nominal_substitution": True}
    stats = _voltage_range_filter_on_arrays(meas, nodes, **kw)
    assert stats["downweighted_nominal_substitution"] == 1
    assert int(meas[0]["quality"]) == 1


def test_vrf_lo_fallback_half_nom_rejects_sentinel():
    # U_KRIT=1.0 на 500-кВ узле — заглушка; lo=V_ном/2=250. V=100 < 250 → out_of_range.
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 500.0, "voltage_critical": 1.0, "voltage_max": 525.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 100.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["out_of_range"] == 1


def test_vrf_hi_fallback_when_vmax_zero():
    # vmax=0 → hi=vnom·1.4=154. V=160 > 154 → out_of_range.
    nodes = _nodes(
        [{"id": 1, "voltage_nominal": 110.0, "voltage_critical": 60.0, "voltage_max": 0.0}]
    )
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 160.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _voltage_range_filter_on_arrays(meas, nodes, **VR_KW)
    assert stats["out_of_range"] == 1


# ---------------------------------------------------------- voltage_meas_calibration

VC_KW = {"slack_type": 3, "ot_node": OT_NODE, "mt_v": MT_V, "sigma2": 0.1}


def test_vc_gen_node_target():
    nodes = _nodes([{"id": 1, "generation_p_max": 50.0}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 110.0,
                "variance": 9.0,
            }
        ]
    )
    stats = _voltage_meas_calibration_on_arrays(meas, nodes, **VC_KW)
    assert stats["updated_meas"] == 1 and stats["target_nodes"] == 1
    assert float(meas[0]["variance"]) == pytest.approx(0.1)
    assert float(meas[0]["weight"]) == pytest.approx(10.0)


def test_vc_slack_target():
    nodes = _nodes([{"id": 1, "node_type": 3}])  # slack_type=3
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 110.0,
                "variance": 9.0,
            }
        ]
    )
    stats = _voltage_meas_calibration_on_arrays(meas, nodes, **VC_KW)
    assert stats["target_nodes"] == 1 and stats["updated_meas"] == 1


def test_vc_skip_non_target():
    nodes = _nodes([{"id": 1, "node_type": 1, "generation_p_max": 0.0}])  # не slack, нет ген.
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 110.0,
                "variance": 9.0,
            }
        ]
    )
    stats = _voltage_meas_calibration_on_arrays(meas, nodes, **VC_KW)
    assert stats["updated_meas"] == 0
    assert float(meas[0]["variance"]) == 9.0  # не тронут


def test_vc_skip_off_node():
    nodes = _nodes([{"id": 1, "status": False, "generation_p_max": 50.0}])
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 1,
                "measurement_type": MT_V,
                "value": 110.0,
                "variance": 9.0,
            }
        ]
    )
    stats = _voltage_meas_calibration_on_arrays(meas, nodes, **VC_KW)
    assert stats["target_nodes"] == 0 and stats["updated_meas"] == 0


# ---------------------------------------------------------- resolve_merged


def test_rm_v_weighted_average():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_V,
                "value": 10.0,
                "variance": 1.0,
            },
            {
                "id": 11,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_V,
                "value": 20.0,
                "variance": 1.0,
            },
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats["resolved_v"] == 1 and stats["deactivated"] == 1
    assert float(meas[0]["value"]) == pytest.approx(15.0)  # (10·1+20·1)/2
    assert float(meas[0]["variance"]) == pytest.approx(0.5)  # 1/(1+1)
    assert float(meas[0]["weight"]) == pytest.approx(2.0)
    assert bool(meas[1]["status"]) is False


def test_rm_p_inj_sum():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_PINJ,
                "value": 10.0,
                "variance": 2.0,
            },
            {
                "id": 11,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_PINJ,
                "value": 30.0,
                "variance": 3.0,
            },
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats["resolved_p_inj"] == 1
    assert float(meas[0]["value"]) == pytest.approx(40.0)
    assert float(meas[0]["variance"]) == pytest.approx(5.0)
    assert bool(meas[1]["status"]) is False


def test_rm_q_inj_sum():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_QINJ,
                "value": 10.0,
                "variance": 2.0,
            },
            {
                "id": 11,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_QINJ,
                "value": 30.0,
                "variance": 3.0,
            },
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats["resolved_q_inj"] == 1


def test_rm_single_no_op():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_V,
                "value": 10.0,
                "variance": 1.0,
            }
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats == {"resolved_v": 0, "resolved_p_inj": 0, "resolved_q_inj": 0, "deactivated": 0}
    assert float(meas[0]["value"]) == 10.0


def test_rm_branch_meas_not_merged():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_BRANCH,
                "object_id": 5,
                "measurement_type": 0,
                "value": 10.0,
                "variance": 1.0,
            },
            {
                "id": 11,
                "object_type": OT_BRANCH,
                "object_id": 5,
                "measurement_type": 0,
                "value": 20.0,
                "variance": 1.0,
            },
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats["deactivated"] == 0  # ot=branch не группируется
    assert bool(meas[1]["status"]) is True


def test_rm_different_side_not_merged():
    meas = _meas(
        [
            {
                "id": 10,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_V,
                "value": 10.0,
                "variance": 1.0,
                "branch_side": 0,
            },
            {
                "id": 11,
                "object_type": OT_NODE,
                "object_id": 5,
                "measurement_type": MT_V,
                "value": 20.0,
                "variance": 1.0,
                "branch_side": 1,
            },
        ]
    )
    stats = _resolve_merged_on_arrays(meas)
    assert stats["resolved_v"] == 0 and stats["deactivated"] == 0  # разные ключи


# ---------------------------------------------------------- deactivate_orphan


def test_do_node_orphan_deactivated():
    nodes = _nodes([{"id": 1, "status": False}])  # узел off
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    stats = _deactivate_orphan_on_arrays(meas, nodes, _branches([]), _gens([]))
    assert bool(meas[0]["status"]) is False and stats["node_meas"] == 1


def test_do_branch_orphan_deactivated():
    branches = _branches([{"id": 7, "status": False}])
    meas = _meas([{"id": 10, "object_type": OT_BRANCH, "object_id": 7, "measurement_type": 0}])
    stats = _deactivate_orphan_on_arrays(meas, _nodes([]), branches, _gens([]))
    assert bool(meas[0]["status"]) is False and stats["branch_meas"] == 1


def test_do_gen_orphan_deactivated():
    gens = _gens([{"id": 3, "status": False}])
    meas = _meas([{"id": 10, "object_type": OT_GEN, "object_id": 3, "measurement_type": 0}])
    stats = _deactivate_orphan_on_arrays(meas, _nodes([]), _branches([]), gens)
    assert bool(meas[0]["status"]) is False and stats["gen_meas"] == 1


def test_do_active_object_kept():
    nodes = _nodes([{"id": 1}])  # активен
    meas = _meas([{"id": 10, "object_type": OT_NODE, "object_id": 1, "measurement_type": MT_V}])
    stats = _deactivate_orphan_on_arrays(meas, nodes, _branches([]), _gens([]))
    assert bool(meas[0]["status"]) is True and stats["node_meas"] == 0


def test_do_unknown_object_type_counted():
    meas = _meas([{"id": 10, "object_type": 9, "object_id": 1, "measurement_type": MT_V}])
    stats = _deactivate_orphan_on_arrays(meas, _nodes([]), _branches([]), _gens([]))
    assert stats["orphan_object_id"] == 1
    assert bool(meas[0]["status"]) is True  # неизвестный ot не трогается
