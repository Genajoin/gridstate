"""Конвертация ``Working`` (именованные единицы) ↔ внутреннее
p.u.-представление SE.

Внутри SE всё считается в системе p.u. с ``base_mva = 100`` и базой
напряжения, равной ``NodeCollection.voltage_nominal`` для соответствующей
шины. В именованных единицах работают только входные данные и итоговые записи
обратно в ``Working``.

Соглашения:

- Базовая мощность ``S_base = 100 МВА`` (3-phase basis).
- Базовое напряжение каждой шины ``V_base = voltage_nominal`` (LL, кВ).
- Базовый импеданс ``Z_base = V_base² / S_base`` (Ом).
- Базовый ток ``I_base = S_base · 1000 / (√3 · V_base)`` (А).
- Импедансы и проводимости ветвей трактуются как заданные на стороне «от»
  (``from_node``), что соответствует pandapower-конвенции.

Для алгоритмов SE используются *позиционные* индексы шин (``0..n_bus−1``);
оригинальные ``id`` из ``NODE_DTYPE`` сохраняются в ``NetworkPU.bus_ids`` —
по ним результаты пишутся обратно в коллекции ``Working``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from gridstate.working import Working


BASE_MVA: float = 100.0
"""Базовая мощность SE (фиксированная, 3-phase)."""


@dataclass
class NetworkPU:
    """Внутреннее p.u.-представление сети для численных алгоритмов SE.

    Все индексы (``slack_idx``, ``from_idx``, ``to_idx``) — *позиционные* в
    массиве шин (``0..n_bus−1``); сами ``id`` сохраняются в ``bus_ids`` для
    write-back.
    """

    # Топология
    n_bus: int
    n_branch: int
    bus_ids: np.ndarray  # (n_bus,) i8   — NODE_DTYPE.id
    bus_vn_kv: np.ndarray  # (n_bus,) f8   — voltage_nominal, кВ
    bus_type: np.ndarray  # (n_bus,) i1   — 0=PQ, 1=PV, 2=SLACK
    slack_idx: int  # позиционный индекс slack-узла

    branch_ids: np.ndarray  # (n_branch,) i8 — BRANCH_DTYPE.id
    from_idx: np.ndarray  # (n_branch,) i8 — позиционный индекс «от»
    to_idx: np.ndarray  # (n_branch,) i8 — позиционный индекс «до»

    # Параметры ветвей в p.u.
    branch_r: np.ndarray  # последовательное R
    branch_x: np.ndarray  # последовательное X
    branch_g: np.ndarray  # суммарный шунт G ветви (Π-схема)
    branch_b: np.ndarray  # суммарный шунт B ветви (Π-схема)
    branch_g_from: np.ndarray  # шунт со стороны «от»
    branch_b_from: np.ndarray
    branch_g_to: np.ndarray  # шунт со стороны «до»
    branch_b_to: np.ndarray
    tap_ratio: np.ndarray  # модуль прямого Ktr (безразмерное)
    phase_shift: np.ndarray  # arg(Ktr), радианы

    # Шунты узлов (p.u.)
    bus_g_shunt: np.ndarray
    bus_b_shunt: np.ndarray

    # Инъекции для классификации zero-injection (P_gen − P_load, p.u.)
    bus_p_injection: np.ndarray
    bus_q_injection: np.ndarray

    base_mva: float = BASE_MVA


# ---------------------------------------------------------------------------
# Внешний → внутренний (Working → NetworkPU)
# ---------------------------------------------------------------------------


def model_to_pu(model: Working) -> NetworkPU:
    """Собрать ``NetworkPU`` из ``Working`` (тонкая обёртка-адаптер).

    Извлекает контрактные таблицы узлов/ветвей (``to_numpy()``) и делегирует
    числовую конвертацию :func:`network_pu_from_tables`. Сам перевод в p.u.
    работает только над массивами контракта — модель здесь лишь источник таблиц.
    """
    return network_pu_from_tables(model.nodes.to_numpy(), model.branches.to_numpy())


def network_pu_from_tables(nodes_arr: np.ndarray, branches_arr: np.ndarray) -> NetworkPU:
    """Собрать ``NetworkPU`` из контрактных таблиц узлов/ветвей (numpy).

    **Массивное ядро** входного конвертера: вход — структурированные массивы
    входного слоя контракта (``SEInput`` nodes/branches; колонки —
    :data:`gridstate.contract.tables.NODES` / ``BRANCHES``), выход — внутреннее
    p.u.-представление SE. Зависимости от ``Working`` нет — массивный путь
    обращается сюда напрямую.

    Учитываются только активные узлы (``status=True``) и активные ветви.
    Импедансы и проводимости ветвей трактуются как заданные на стороне «от».

    Raises:
        ValueError: если нет узлов/slack-узла, не указан ``voltage_nominal``,
            или ветвь ссылается на несуществующий узел.
    """
    if len(nodes_arr) == 0:
        raise ValueError("Working пустой: нет узлов")

    # Активные узлы; индексация позиционная по полученному фильтру.
    active_nodes = nodes_arr[nodes_arr["status"]]
    if len(active_nodes) == 0:
        raise ValueError("Все узлы помечены как неактивные (status=False)")

    bus_ids = active_nodes["id"].astype(np.int64, copy=True)
    vn_kv = active_nodes["voltage_nominal"].astype(np.float64, copy=True)
    if np.any(vn_kv <= 0):
        bad = bus_ids[vn_kv <= 0].tolist()
        raise ValueError(f"voltage_nominal ≤ 0 у узлов {bad}")

    bus_type = active_nodes["node_type"].astype(np.int8, copy=True)

    slack_positions = np.where(bus_type == 2)[0]
    if slack_positions.size == 0:
        raise ValueError("В Working нет slack-узла (node_type=SLACK). SE требует ровно один slack.")
    if slack_positions.size > 1:
        # Если несколько — берём с минимальным balance_priority (т.е. первичный).
        priorities = active_nodes["balance_priority"][slack_positions]
        slack_idx = int(slack_positions[int(np.argmin(priorities))])
    else:
        slack_idx = int(slack_positions[0])

    # Карта id → позиционный индекс
    id_to_pos: dict[int, int] = {int(nid): pos for pos, nid in enumerate(bus_ids)}

    # Базы (поэлементно для каждой шины)
    z_base = vn_kv**2 / BASE_MVA  # Ом

    # Шунты узлов (См → p.u.)
    bus_g_shunt = active_nodes["shunt_g"].astype(np.float64) * z_base
    bus_b_shunt = active_nodes["shunt_b"].astype(np.float64) * z_base

    # Инъекции (МВт/МВАр → p.u.)
    bus_p_inj = (active_nodes["generation_p"] - active_nodes["load_p"]).astype(
        np.float64
    ) / BASE_MVA
    bus_q_inj = (active_nodes["generation_q"] - active_nodes["load_q"]).astype(
        np.float64
    ) / BASE_MVA

    # Углы — уже в радианах согласно NODE_DTYPE; численно проверяем диапазон.
    voltage_angle = active_nodes["voltage_angle"].astype(np.float64)
    if np.any(np.abs(voltage_angle) > 2 * math.pi):
        raise ValueError(
            "voltage_angle вне диапазона [−2π, +2π] — возможно, в модели угол хранится в "
            "градусах. NODE_DTYPE требует радиан."
        )

    # ---- Ветви ----
    if len(branches_arr) == 0:
        # Сеть без ветвей формально валидна (одиночный узел) — продолжаем.
        active_branches = branches_arr
    else:
        active_branches = branches_arr[branches_arr["status"]]

    # Фильтруем ветви, у которых хоть один конец вне активных узлов.
    if len(active_branches) > 0:
        keep = np.array(
            [
                int(b["from_node"]) in id_to_pos and int(b["to_node"]) in id_to_pos
                for b in active_branches
            ],
            dtype=bool,
        )
        active_branches = active_branches[keep]

    n_branch = len(active_branches)
    branch_ids = (
        active_branches["id"].astype(np.int64, copy=True)
        if n_branch > 0
        else np.empty(0, dtype=np.int64)
    )
    from_idx = np.empty(n_branch, dtype=np.int64)
    to_idx = np.empty(n_branch, dtype=np.int64)
    branch_r = np.empty(n_branch, dtype=np.float64)
    branch_x = np.empty(n_branch, dtype=np.float64)
    branch_g = np.empty(n_branch, dtype=np.float64)
    branch_b = np.empty(n_branch, dtype=np.float64)
    branch_g_from = np.empty(n_branch, dtype=np.float64)
    branch_b_from = np.empty(n_branch, dtype=np.float64)
    branch_g_to = np.empty(n_branch, dtype=np.float64)
    branch_b_to = np.empty(n_branch, dtype=np.float64)
    tap_ratio = np.empty(n_branch, dtype=np.float64)
    phase_shift = np.empty(n_branch, dtype=np.float64)

    for i, b in enumerate(active_branches):
        f_pos = id_to_pos[int(b["from_node"])]
        t_pos = id_to_pos[int(b["to_node"])]
        from_idx[i] = f_pos
        to_idx[i] = t_pos

        # Импеданс приведён к стороне «от» (pandapower-конвенция)
        zb = float(z_base[f_pos])
        branch_r[i] = float(b["resistance"]) / zb
        branch_x[i] = float(b["reactance"]) / zb
        branch_g[i] = float(b["conductance"]) * zb
        branch_b[i] = float(b["susceptance"]) * zb
        branch_g_from[i] = float(b["conductance_from"]) * zb
        branch_b_from[i] = float(b["susceptance_from"]) * zb
        branch_g_to[i] = float(b["conductance_to"]) * zb
        branch_b_to[i] = float(b["susceptance_to"]) * zb

        tap = float(b["tap_ratio"])
        if tap <= 0:
            tap = 1.0
        # Конвенция входного формата: ``tap_ratio`` записан как **physical**
        # turn ratio HV:LV (например 2.27 для trafo 750/330). После
        # base-нормализации (V_base_node = voltage_nominal_node) разные
        # стороны trafo имеют разные V_base, и в p.u. ratio становится
        # K_pu = K_physical / (V_base_from / V_base_to). Для idealn
        # trafo K_pu = 1.0; off-nominal tap (~±10%) приводит к 0.9-1.1.
        # Без этой нормализации yff/yft в build_ybus получает
        # double-counting turn ratio: на ветви 750/330 при flat init
        # h(P_flow_to) выходит 2160 p.u. = 216 ГВт (см. Group D в
        # docs/audit/audit_runaway_node.md).
        vn_from = float(vn_kv[f_pos])
        vn_to = float(vn_kv[t_pos])
        if vn_from > 0 and vn_to > 0 and not math.isclose(vn_from, vn_to):
            tap = tap / (vn_from / vn_to)
        tap_ratio[i] = tap
        phase_shift[i] = float(b["phase_shift"])

    return NetworkPU(
        n_bus=len(bus_ids),
        n_branch=n_branch,
        bus_ids=bus_ids,
        bus_vn_kv=vn_kv,
        bus_type=bus_type,
        slack_idx=slack_idx,
        branch_ids=branch_ids,
        from_idx=from_idx,
        to_idx=to_idx,
        branch_r=branch_r,
        branch_x=branch_x,
        branch_g=branch_g,
        branch_b=branch_b,
        branch_g_from=branch_g_from,
        branch_b_from=branch_b_from,
        branch_g_to=branch_g_to,
        branch_b_to=branch_b_to,
        tap_ratio=tap_ratio,
        phase_shift=phase_shift,
        bus_g_shunt=bus_g_shunt,
        bus_b_shunt=bus_b_shunt,
        bus_p_injection=bus_p_inj,
        bus_q_injection=bus_q_inj,
    )


# ---------------------------------------------------------------------------
# Внутренний → внешний (запись результатов SE в Working)
# ---------------------------------------------------------------------------


def compute_node_results_pu(
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    *,
    ybus: csr_matrix | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Числовое ядро записи по узлам: pu-решение → именованные величины (массивы).

    Чистая функция над массивами (без ``Working``): переводит модули
    напряжений в кВ и, при заданной ``ybus``, считает узловые инъекции
    ``S = V·conj(Ybus·V)·base_mva`` (МВт/МВАр).

    Returns:
        ``(voltage_kv, p_inj_mw, q_inj_mvar)`` — массивы длины ``n_bus`` в
        порядке ``network_pu.bus_ids``; ``p_inj_mw``/``q_inj_mvar`` = ``None``,
        если ``ybus`` не передана.
    """
    voltage_kv = v_pu * network_pu.bus_vn_kv
    if ybus is None:
        return voltage_kv, None, None
    v_complex_full = v_pu * np.exp(1j * delta_rad)
    i_bus = ybus @ v_complex_full
    s_bus = v_complex_full * np.conj(i_bus)
    return voltage_kv, s_bus.real * BASE_MVA, s_bus.imag * BASE_MVA


def compute_branch_results_pu(
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    yf: csr_matrix,
    yt: csr_matrix,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Числовое ядро записи по ветвям: pu-решение + Yf/Yt → перетоки/токи/потери.

    Чистая функция над массивами (без ``Working``). Все величины — в
    именованных единицах (МВт/МВАр/А), в порядке ``network_pu.branch_ids``.

    Returns:
        ``(p_from_mw, q_from_mvar, p_to_mw, q_to_mvar, i_from_a, i_to_a,
        loss_p, loss_q)``.
    """
    v_complex_full = v_pu * np.exp(1j * delta_rad)
    i_from = yf @ v_complex_full  # (n_branch,) комплексный, в p.u.
    i_to = yt @ v_complex_full
    s_from = v_complex_full[network_pu.from_idx] * np.conj(i_from)
    s_to = v_complex_full[network_pu.to_idx] * np.conj(i_to)

    # Базовые токи на стороне каждой ветви (А)
    sqrt3 = math.sqrt(3.0)
    i_base_from = BASE_MVA * 1000.0 / (sqrt3 * network_pu.bus_vn_kv[network_pu.from_idx])
    i_base_to = BASE_MVA * 1000.0 / (sqrt3 * network_pu.bus_vn_kv[network_pu.to_idx])

    p_from_mw = s_from.real * BASE_MVA
    q_from_mvar = s_from.imag * BASE_MVA
    p_to_mw = s_to.real * BASE_MVA
    q_to_mvar = s_to.imag * BASE_MVA
    i_from_a = np.abs(i_from) * i_base_from
    i_to_a = np.abs(i_to) * i_base_to

    # Потери в ветви: знаковая convention в power_from_p/power_to_p такая,
    # что P_from > 0 = поток входит в ветвь from-стороны, P_to > 0 = выходит
    # в to-сторону. Поэтому потери = P_from + P_to (не P_from − P_to).
    loss_p = p_from_mw + p_to_mw
    loss_q = q_from_mvar + q_to_mvar
    return (
        p_from_mw,
        q_from_mvar,
        p_to_mw,
        q_to_mvar,
        i_from_a,
        i_to_a,
        loss_p,
        loss_q,
    )


def write_results_to_model(
    model: Working,
    v_pu: np.ndarray,
    delta_rad: np.ndarray,
    network_pu: NetworkPU,
    yf: csr_matrix | None = None,
    yt: csr_matrix | None = None,
    ybus: csr_matrix | None = None,
) -> None:
    """Записать результат SE обратно в ``Working`` (именованные единицы).

    Args:
        model: ``Working``, обновляется in-place.
        v_pu: (n_bus,) — модули напряжений в p.u. (соответствуют ``network_pu.bus_ids``).
        delta_rad: (n_bus,) — углы напряжений в радианах.
        network_pu: внутреннее представление, использованное для расчёта.
        yf: (n_branch × n_bus) — матрица «от» в p.u. Если задана вместе с ``yt``,
            пересчитываются перетоки, токи и потери ветвей.
        yt: (n_branch × n_bus) — матрица «до».
        ybus: (n_bus × n_bus) — матрица узлов в p.u. Если задана, заполняются
            ``p_inj_calc``/``q_inj_calc``/``imbalance_p``/``imbalance_q`` узлов.

    Заполняет:

    Узлы (``NodeCollection``):
        - ``voltage_magnitude`` (кВ = v_pu × voltage_nominal),
        - ``voltage_angle`` (рад);
        - при ``ybus`` — ``p_inj_calc``/``q_inj_calc`` (МВт/МВАр) и
          ``imbalance_p`` = ``p_inj_calc`` − (``generation_p`` − ``load_p``),
          ``imbalance_q`` аналогично.

    Ветви (``BranchCollection``) — при ``yf, yt``:
        - ``power_from_p/q`` (МВт/МВАр), ``power_to_p/q``,
        - ``current_from/to`` (А);
        - ``loss_p`` = ``power_from_p`` + ``power_to_p`` (МВт),
          ``loss_q`` аналогично;
        - ``loading_pct`` = max(I_from, I_to) / ``current_limit_normal`` × 100,
          если лимит задан (иначе 0).
    """
    if v_pu.shape != (network_pu.n_bus,):
        raise ValueError(f"v_pu должен быть длины {network_pu.n_bus}, получено {v_pu.shape}")
    if delta_rad.shape != (network_pu.n_bus,):
        raise ValueError(
            f"delta_rad должен быть длины {network_pu.n_bus}, получено {delta_rad.shape}"
        )

    # ---- Узлы ---- (числовое ядро — на массивах; запись в модель — ниже)
    voltage_kv, p_inj_mw, q_inj_mvar = compute_node_results_pu(
        v_pu, delta_rad, network_pu, ybus=ybus
    )

    for pos, node_id in enumerate(network_pu.bus_ids.tolist()):
        update: dict[str, Any] = {
            "voltage_magnitude": float(voltage_kv[pos]),
            "voltage_angle": float(delta_rad[pos]),
        }
        if p_inj_mw is not None and q_inj_mvar is not None:
            node = model.nodes.get_by_id(int(node_id))
            net_p_meas = float(node.generation_p) - float(node.load_p) if node is not None else 0.0
            net_q_meas = float(node.generation_q) - float(node.load_q) if node is not None else 0.0
            p_calc = float(p_inj_mw[pos])
            q_calc = float(q_inj_mvar[pos])
            update["p_inj_calc"] = p_calc
            update["q_inj_calc"] = q_calc
            update["imbalance_p"] = p_calc - net_p_meas
            update["imbalance_q"] = q_calc - net_q_meas
        model.nodes.update(int(node_id), update)

    if yf is None or yt is None or network_pu.n_branch == 0:
        return

    # ---- Ветви: Sf/St в p.u., затем перевод в МВт/МВАр и А (числовое ядро) ----
    (
        p_from_mw,
        q_from_mvar,
        p_to_mw,
        q_to_mvar,
        i_from_a,
        i_to_a,
        loss_p,
        loss_q,
    ) = compute_branch_results_pu(v_pu, delta_rad, network_pu, yf, yt)

    for i, branch_id in enumerate(network_pu.branch_ids.tolist()):
        update_b: dict[str, Any] = {
            "power_from_p": float(p_from_mw[i]),
            "power_from_q": float(q_from_mvar[i]),
            "power_to_p": float(p_to_mw[i]),
            "power_to_q": float(q_to_mvar[i]),
            "current_from": float(i_from_a[i]),
            "current_to": float(i_to_a[i]),
            "loss_p": float(loss_p[i]),
            "loss_q": float(loss_q[i]),
        }
        # loading_pct: при наличии current_limit_normal
        branch = model.branches.get_by_id(int(branch_id))
        if branch is not None:
            i_max = float(branch.current_limit_normal)
            if i_max > 0:
                i_max_used = max(float(i_from_a[i]), float(i_to_a[i]))
                update_b["loading_pct"] = float(i_max_used / i_max * 100.0)
        model.branches.update(int(branch_id), update_b)
