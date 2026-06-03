"""Синтез инжекции узла из branch-flow замеров (процедура эталонной SE).

Эталонная SE после применения TI создаёт виртуальную инжекцию узла как сумму
известных потоков по инцидентным ветвям: ``P_inj_node = -Σ(P_flow на
стороне узла)`` (по 1-му закону Кирхгофа для узлов без локальной
нагрузки/генерации). У нас этого шага нет — узлы с известными
branch-flow'ами, но без P_inj/Q_inj-замера, попадают в
``add_pseudo_measurements`` с ``value=pg-pn=0`` и слабым σ, и в WLS
``V`` уезжает.

На терминалах магистральных 750 кВ ВЛ ``pg/qg`` появляются именно на
этом шаге — здесь мы повторяем его в нашем pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gridstate.z_vector import (
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    OBJ_BRANCH,
    OBJ_NODE,
)


__all__ = [
    "synthesize_block_bus_injection_from_branch_xml",
    "synthesize_node_injection_from_branch_flows",
]


def _synthesize_node_injection_on_arrays(
    nodes_arr: Any,
    branches_arr: Any,
    meas_arr: Any,
    *,
    sigma_frac: float = 0.05,
    sigma_min_mw: float = 5.0,
    require_all_sides_known: bool = True,
    mid_start: int = 290_000_000,
) -> tuple[list[dict], dict[str, int]]:
    """Построить pseudo P_inj/Q_inj-строки из branch-flow на контрактных массивах.

    vendor-free ядро (CLASS-1, append-паттерн): читает ТОЛЬКО контрактные колонки
    ``node.{id,status}``, ``branch.{status,from_node,to_node,ti_p_from,ti_q_from,
    ti_p_to,ti_q_to}``, ``measurement.{id,status,object_type,measurement_type,value,
    is_pseudo,object_id}``; НЕ мутирует входы, ВОЗВРАЩАЕТ ``(new_rows, stats)``.

    **Должно оставаться последовательным Python-циклом** в исходном порядке обхода
    (``nodes_arr`` и incident-список построены детерминированно через ``enumerate`` по
    ``branches_arr``); суммы ``p_sum/q_sum`` накапливаются в том же порядке →
    **строгий бит-в-бит 1e-9**. Без collision-skip (id монотонно с ``mid_start``).
    """
    # 1) Карты id → измерение (только real, не pseudo).
    p_branch_by_id: dict[int, dict] = {}
    q_branch_by_id: dict[int, dict] = {}
    have_pinj_real: set[int] = set()
    for r in meas_arr:
        if not r["status"]:
            continue
        mid = int(r["id"])
        ot = int(r["object_type"])
        kind = int(r["measurement_type"])
        is_pseudo = bool(r["is_pseudo"]) if "is_pseudo" in meas_arr.dtype.names else False
        if ot == OBJ_BRANCH and not is_pseudo:
            if kind == KIND_POWER_P:
                p_branch_by_id[mid] = {"value": float(r["value"])}
            elif kind == KIND_POWER_Q:
                q_branch_by_id[mid] = {"value": float(r["value"])}
        elif ot == OBJ_NODE and not is_pseudo and kind == KIND_POWER_INJECTION_P:
            have_pinj_real.add(int(r["object_id"]))

    # 2) node_id → список инцидентных активных ветвей с их ti_* IDs.
    incident: dict[int, list[tuple[int, str]]] = {}  # node_id → [(branch_idx, side)]
    for bi, b in enumerate(branches_arr):
        if not bool(b["status"]):
            continue
        f = int(b["from_node"])
        t = int(b["to_node"])
        incident.setdefault(f, []).append((bi, "from"))
        incident.setdefault(t, []).append((bi, "to"))

    # 3) Проходимся по активным узлам.
    nodes_synth = 0
    skipped_has_inj = 0
    skipped_no_cov = 0
    skipped_orphan = 0
    mid_counter = mid_start

    new_rows: list[dict] = []
    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        nid = int(row["id"])
        if nid in have_pinj_real:
            skipped_has_inj += 1
            continue
        adj = incident.get(nid)
        if not adj:
            skipped_orphan += 1
            continue

        p_sum = 0.0
        q_sum = 0.0
        full = True
        for bi, side in adj:
            br = branches_arr[bi]
            if side == "from":
                pid = int(br["ti_p_from"])
                qid = int(br["ti_q_from"])
            else:
                pid = int(br["ti_p_to"])
                qid = int(br["ti_q_to"])
            pm = p_branch_by_id.get(pid) if pid > 0 else None
            qm = q_branch_by_id.get(qid) if qid > 0 else None
            if pm is None or qm is None:
                full = False
                if require_all_sides_known:
                    break
                continue
            p_sum += pm["value"]
            q_sum += qm["value"]

        if require_all_sides_known and not full:
            skipped_no_cov += 1
            continue
        if not full and not require_all_sides_known:
            # Хотя бы один side был — продолжаем
            pass

        # Знак: p_side_of_node = расход из узла в ветвь (конвенция входного формата).
        # Σ расходов = -P_inj_node (т.к. P_inj = втекание; расход = -втекание).
        synth_p = -p_sum
        synth_q = -q_sum

        # σ — фракция от модуля либо минимум.
        s_magn = float(np.hypot(synth_p, synth_q))
        sigma = max(sigma_frac * s_magn, sigma_min_mw)
        variance = float(sigma * sigma)

        new_rows.append(
            {
                "id": mid_counter,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": synth_p,
                "variance": variance,
                "status": True,
                "quality": 0,
                "is_pseudo": True,
            }
        )
        mid_counter += 1
        new_rows.append(
            {
                "id": mid_counter,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": synth_q,
                "variance": variance,
                "status": True,
                "quality": 0,
                "is_pseudo": True,
            }
        )
        mid_counter += 1
        nodes_synth += 1

    stats = {
        "nodes_synthesized": nodes_synth,
        "skipped_has_inj": skipped_has_inj,
        "skipped_no_full_coverage": skipped_no_cov,
        "skipped_orphan": skipped_orphan,
    }
    return new_rows, stats


def synthesize_node_injection_from_branch_flows(
    model,
    *,
    sigma_frac: float = 0.05,
    sigma_min_mw: float = 5.0,
    require_all_sides_known: bool = True,
    mid_start: int = 290_000_000,
) -> dict[str, int]:
    """Синтезировать pseudo `P_inj/Q_inj` из branch-flow на узлах без замера инжекции.

    Для каждого активного узла без real `P_inj`-замера:

    1. Собрать инцидентные активные ветви.
    2. Для каждой ветви взять real (non-pseudo) `P/Q` замер на стороне
       этого узла (через ``branch.ti_p_from / ti_q_from / ti_p_to /
       ti_q_to``).
    3. Если все стороны известны (или ``require_all_sides_known=False`` —
       суммировать частично), вычислить
       ``P_inj = -Σ(p_side_of_node)``, ``Q_inj = -Σ(q_side_of_node)``.
       Знак: ``p_from`` (конвенция входного формата) = поток узла **в** ветвь,
       тогда сумма расходов в инцидентные ветви равна инжекции узла;
       ``-`` компенсирует, что мы суммируем «исходящие», а инжекция —
       «втекающая».
    4. Добавить pseudo `P_inj/Q_inj` с σ = ``max(sigma_frac·|S_inj|,
       sigma_min_mw)``.

    Args:
        model: ``Working`` (in-place).
        sigma_frac: фракция от |S| для σ pseudo-замера (default 5 %).
        sigma_min_mw: минимальный σ в МВт (для узлов с малой
            инжекцией, чтобы не получить σ→0).
        require_all_sides_known: если True, синтезировать только если
            **все** инцидентные ветви имеют real замер на стороне узла.
            Если False — суммировать частично (рискованно: skips приведут
            к смещению).
        mid_start: начальный ID для новых measurements. Должен не
            пересекаться с ``add_pseudo_measurements.mid_start`` (default
            300_000_000) — здесь 290e6.

    Returns:
        ``{"nodes_synthesized": N, "skipped_has_inj": N,
        "skipped_no_full_coverage": N, "skipped_orphan": N}``.
    """
    new_rows, stats = _synthesize_node_injection_on_arrays(
        model.nodes.to_numpy(),
        model.branches.to_numpy(),
        model.measurements.to_numpy(),
        sigma_frac=sigma_frac,
        sigma_min_mw=sigma_min_mw,
        require_all_sides_known=require_all_sides_known,
        mid_start=mid_start,
    )
    for r in new_rows:
        model.measurements.add(r)
    return stats


def synthesize_block_bus_injection_from_branch_xml(
    model,
    *,
    vn_threshold_kv: float = 25.0,
    sigma_frac: float = 0.05,
    sigma_min_mw: float = 2.0,
    mid_start: int = 295_000_000,
) -> dict[str, int]:
    """Pseudo ``P_inj/Q_inj`` для блочных ген-шин из ``branch.power_<side>_*``.

    Целевая категория: блочные шины с ``vn ≤ vn_threshold_kv`` (обычно
    6–25 кВ), ``exist_gen=True``, без реального V/P_inj-замера, с
    **единственной** активной ветвью к host'у.

    На таких узлах в TI отсутствуют как ``ti_p_<side>`` блока, так и
    ``ti_p_<other_side>`` host'а — TM на трансформаторы блок↔host не
    приходит. Из-за этого ``synthesize_node_injection_from_branch_flows``
    их пропускает. Но в XML остаётся ``branch.power_<side>_*`` как
    result-snapshot последнего OC прогона (initial поток через блочный
    трансформатор). Этим значением эталонная SE «затравливает» свою
    синтезированную инжекцию узла; мы повторяем тот же шаг.

    Конвенция знака: ``branch.power_<side>_p`` — поток **с узла в branch**
    (с положительным знаком при отдаче узла). Тогда
    ``P_inj_node = +power_<side>_p`` (плюс — то, что узел отдаёт во
    внешнюю сеть равно его внешней инжекции). Аналогично для Q.

    σ — фракция от ``|S|`` либо ``sigma_min_mw``. На больших ген-шинах
    (P~50 МВт) даёт σ≈2.5 МВт — tight (=якорь). На мелких (P<5 МВт)
    σ доминирует ``sigma_min_mw`` — loose, не давит pseudo V=Vnom.

    Args:
        model: ``Working`` (in-place).
        vn_threshold_kv: верхний предел Vnom для отнесения к блочной шине
            (default 25 кВ — покрывает 6/10/14/16/18/24/25 кВ блоки
            ТЭЦ/АЭС/ГЭС/ГРЭС).
        sigma_frac: фракция от |S| для σ pseudo-замера (default 5 %).
        sigma_min_mw: минимальный σ в МВт (для узлов с малой инжекцией).
        mid_start: начальный ID для новых measurements; default 295e6 —
            после ``synthesize_node_injection_from_branch_flows`` (290e6)
            и до ``add_pseudo_measurements`` (300e6).

    Returns:
        ``{"added": N, "skipped_not_block": N, "skipped_has_real": N,
        "skipped_multi_branch": N, "skipped_no_branch": N}``.
    """
    nodes_arr = model.nodes.to_numpy()
    branches_arr = model.branches.to_numpy()
    meas_arr = model.measurements.to_numpy()

    real_v: set[int] = set()
    real_p: set[int] = set()
    real_q: set[int] = set()
    for r in meas_arr:
        if not bool(r["status"]):
            continue
        if bool(r["is_pseudo"]) if "is_pseudo" in meas_arr.dtype.names else False:
            continue
        if int(r["object_type"]) != OBJ_NODE:
            continue
        kind = int(r["measurement_type"])
        nid = int(r["object_id"])
        if kind == 2:  # KIND_VOLTAGE
            real_v.add(nid)
        elif kind == KIND_POWER_INJECTION_P:
            real_p.add(nid)
        elif kind == KIND_POWER_INJECTION_Q:
            real_q.add(nid)

    incident: dict[int, list[tuple[int, str]]] = {}
    for bi, b in enumerate(branches_arr):
        if not bool(b["status"]):
            continue
        incident.setdefault(int(b["from_node"]), []).append((bi, "from"))
        incident.setdefault(int(b["to_node"]), []).append((bi, "to"))

    added = 0
    skipped_not_block = 0
    skipped_has_real = 0
    skipped_multi_branch = 0
    skipped_no_branch = 0
    mid_counter = mid_start
    new_rows: list[dict] = []

    for row in nodes_arr:
        if not bool(row["status"]):
            continue
        vn = float(row["voltage_nominal"])
        if vn <= 0 or vn > vn_threshold_kv or not bool(row["exist_gen"]):
            skipped_not_block += 1
            continue
        nid = int(row["id"])
        if nid in real_v or nid in real_p or nid in real_q:
            skipped_has_real += 1
            continue
        adj = incident.get(nid)
        if not adj:
            skipped_no_branch += 1
            continue
        if len(adj) != 1:
            skipped_multi_branch += 1
            continue
        bi, side = adj[0]
        b = branches_arr[bi]
        p_inj = float(b[f"power_{side}_p"])
        q_inj = float(b[f"power_{side}_q"])

        s_magn = float(np.hypot(p_inj, q_inj))
        sigma = max(sigma_frac * s_magn, sigma_min_mw)
        variance = float(sigma * sigma)

        new_rows.append(
            {
                "id": mid_counter,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_P,
                "value": p_inj,
                "variance": variance,
                "status": True,
                "quality": 0,
                "is_pseudo": True,
            }
        )
        mid_counter += 1
        new_rows.append(
            {
                "id": mid_counter,
                "object_type": OBJ_NODE,
                "object_id": nid,
                "measurement_type": KIND_POWER_INJECTION_Q,
                "value": q_inj,
                "variance": variance,
                "status": True,
                "quality": 0,
                "is_pseudo": True,
            }
        )
        mid_counter += 1
        added += 1

    for r in new_rows:
        model.measurements.add(r)

    return {
        "added": added,
        "skipped_not_block": skipped_not_block,
        "skipped_has_real": skipped_has_real,
        "skipped_multi_branch": skipped_multi_branch,
        "skipped_no_branch": skipped_no_branch,
    }
