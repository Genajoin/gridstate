"""Адаптер ``pandapower.net`` → контракт gridstate (``SEInput``).

Конвертирует описание сети из ``pandapower`` в контрактные структурированные
массивы (:data:`gridstate.contract.tables.SE_INPUT`) и оборачивает их в
:class:`~gridstate.contract.runtime.SEInput` (``derived=None`` — формат-зависимые
шаги пайплайна пропускаются: модель уже несёт измерения/режим).

Единицы (контракт; перевод в p.u. делает :mod:`gridstate.units`):

* ``BASE_MVA = 100`` фиксирована;
* импеданс ветвей — в **Омах**, приведённый к стороне ``from`` (для линии — bus
  ``from``, для трансформатора — bus ``hv``);
* проводимости/восприимчивости (и шунты узлов, и зарядная b ветвей) — в
  **Сименсах**;
* ``voltage_nominal`` — в кВ; узловые инжекции generation/load — в МВт/МВАр.

``pandapower`` импортируется лениво: пакет ``gridstate`` numpy/scipy-only, а
pandapower нужен только тому, кто строит фикстуры (dev-extra ``[test-models]``).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from gridstate.constants import BranchType, NodeType
from gridstate.contract.runtime import SEInput
from gridstate.contract.tables import SE_INPUT, SE_OUTPUT
from gridstate.working import Working


if TYPE_CHECKING:
    import pandapower as pp  # noqa: F401


_PP_IMPORT_HINT = (
    "Для работы адаптера from_pandapower нужен пакет pandapower. Он не входит в "
    "рантайм-зависимости gridstate (numpy/scipy-only); установите dev-extra:\n"
    "    pip install 'gridstate[test-models]'\n"
    "pandapower нужен только для ГЕНЕРАЦИИ фикстур, не для прогона SE."
)


def _require_pandapower() -> Any:
    """Ленивый guarded-импорт pandapower (подсказка про extra при отсутствии)."""
    try:
        import pandapower as pp
    except ImportError as exc:  # pragma: no cover — путь без pandapower
        raise ImportError(_PP_IMPORT_HINT) from exc
    return pp


def _io_dtype(in_schema: Any, out_schema: Any) -> np.dtype:
    """Объединённый dtype входного+выходного слоёв таблицы (как ``Working.empty``).

    Пайплайн пишет OUTPUT-колонки (``voltage_magnitude``/``p_inj_calc``/перетоки/
    ``estimated_si`` …) прямо в живые коллекции, поэтому backing-массивы должны
    нести и INPUT/WORKING, и OUTPUT-колонки — ровно как нулевые коллекции
    :meth:`gridstate.working.Working.empty`.
    """
    in_dt = in_schema.input_dtype()
    fields = list(in_dt.descr)
    have = set(in_dt.names or ())
    out_dt = out_schema.output_dtype()
    for name in out_dt.names or ():
        if name not in have:
            fields.append((name, out_dt[name].str))
    return np.dtype(fields)


_NODES_DTYPE = _io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes)
_BRANCHES_DTYPE = _io_dtype(SE_INPUT.branches, SE_OUTPUT.branches)
_MEASUREMENTS_DTYPE = _io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements)
_GENERATORS_DTYPE = SE_INPUT.generators.input_dtype()


def from_pandapower(
    net: Any,
    *,
    measurements: np.ndarray | None = None,
) -> SEInput:
    """Построить ``SEInput`` из ``pandapower``-сети ``net``.

    Args:
        net: ``pandapower.auxiliary.pandapowerNet`` — описание сети. Должен иметь
            ровно один ``ext_grid`` (он становится slack-узлом). ``net.sn_mva``
            ожидается равным 100 (контрактная база); иначе бросается ``ValueError``.
        measurements: опциональный контрактный структурный массив измерений
            (dtype = ``SE_INPUT.measurements.input_dtype()``). Если ``None`` —
            таблица измерений пустая (генератор фикстур добавит синтетический
            z-вектор из решения PF).

    Returns:
        :class:`SEInput` с ``derived=None`` (модель уже несёт узлы/ветви/режим/
        измерения; XML/формат-зависимые шаги пайплайна пропускаются).

    Raises:
        ImportError: если pandapower не установлен (см. extra ``[test-models]``).
        ValueError: при ≠1 ext_grid, ``net.sn_mva ≠ 100`` или пустой сети.
    """
    _require_pandapower()  # подсказка про extra; ниже работаем только с net-таблицами

    sn_mva = float(getattr(net, "sn_mva", 100.0))
    if not math.isclose(sn_mva, 100.0, rel_tol=1e-9):
        raise ValueError(
            f"from_pandapower поддерживает только net.sn_mva == 100 (контрактная база), "
            f"получено {sn_mva}."
        )
    if len(net.bus) == 0:
        raise ValueError("Сеть пустая: нет шин (net.bus).")
    if len(net.ext_grid) != 1:
        raise ValueError(
            f"Адаптер требует ровно один ext_grid (slack), получено {len(net.ext_grid)}."
        )

    f_hz = float(getattr(net, "f_hz", 50.0))

    nodes = _build_nodes(net)
    branches = _build_branches(net, f_hz)
    generators = _build_generators(net)
    meas = _coerce_measurements(measurements)

    working = Working.from_arrays(
        nodes=nodes,
        branches=branches,
        measurements=meas,
        generators=generators,
    )
    return SEInput(model=working, derived=None)


def _coerce_measurements(measurements: np.ndarray | None) -> np.ndarray:
    """Привести измерения к dtype рабочего слоя (INPUT⊕OUTPUT-колонки).

    Принимает массив контрактного INPUT-dtype (или уже I/O-dtype) и копирует
    его поля в массив :data:`_MEASUREMENTS_DTYPE` (с OUTPUT-колонками, которые
    пайплайн заполняет). ``None`` → пустая таблица.
    """
    if measurements is None:
        return np.zeros(0, dtype=_MEASUREMENTS_DTYPE)
    src = np.asarray(measurements)
    out = np.zeros(len(src), dtype=_MEASUREMENTS_DTYPE)
    for name in src.dtype.names or ():
        if name in (out.dtype.names or ()):
            out[name] = src[name]
    return out


def measurement_array(n: int) -> np.ndarray:
    """Пустой (нулевой) контрактный массив измерений длины ``n`` для заполнения.

    Хелпер для построителей фикстур: возвращает массив dtype рабочего слоя
    (INPUT⊕OUTPUT), поля ``status``/``quality`` остаются дефолтными — заполните
    ``id``/``object_type``/``object_id``/``measurement_type``/``value``/
    ``variance``/``branch_side``/``status``.
    """
    return np.zeros(n, dtype=_MEASUREMENTS_DTYPE)


# ---------------------------------------------------------------------------
# Узлы
# ---------------------------------------------------------------------------


def _build_nodes(net: Any) -> np.ndarray:
    """``net.bus`` + инжекции (load/gen/sgen/ext_grid/shunt) → массив узлов."""
    bus_index = list(net.bus.index)
    n = len(bus_index)
    pos_of = {bid: i for i, bid in enumerate(bus_index)}

    arr = np.zeros(n, dtype=_NODES_DTYPE)
    arr["id"] = np.asarray(bus_index, dtype=np.int64)
    arr["voltage_nominal"] = net.bus.vn_kv.to_numpy(dtype=np.float64)
    arr["status"] = net.bus.in_service.to_numpy(dtype=bool)
    arr["node_type"] = int(NodeType.PQ)
    # Плоский старт: V = Vном, δ = 0 (солвер перезапишет).
    arr["voltage_magnitude"] = arr["voltage_nominal"]
    arr["voltage_angle"] = 0.0

    has_gen = np.zeros(n, dtype=bool)
    has_load = np.zeros(n, dtype=bool)

    # Нагрузка (consumer, +).
    if len(net.load) > 0:
        for bus, p, q, in_svc in zip(
            net.load.bus, net.load.p_mw, net.load.q_mvar, net.load.in_service, strict=False
        ):
            if not bool(in_svc):
                continue
            i = pos_of[bus]
            arr["load_p"][i] += float(p)
            arr["load_q"][i] += float(q)
            has_load[i] = True

    # Генерация PV (net.gen): задаёт node_type=PV (если узел не slack).
    if len(net.gen) > 0:
        for bus, p, in_svc in zip(net.gen.bus, net.gen.p_mw, net.gen.in_service, strict=False):
            if not bool(in_svc):
                continue
            i = pos_of[bus]
            arr["generation_p"][i] += float(p)
            has_gen[i] = True
            arr["node_type"][i] = int(NodeType.PV)
        # Vsetpoint для PV (vm_pu задан на gen).
        if "vm_pu" in net.gen.columns:
            for bus, vm, in_svc in zip(
                net.gen.bus, net.gen.vm_pu, net.gen.in_service, strict=False
            ):
                if not bool(in_svc):
                    continue
                i = pos_of[bus]
                arr["voltage_setpoint"][i] = float(vm) * float(arr["voltage_nominal"][i])

    # Статическая генерация (net.sgen) — PQ-инжекция (+).
    if len(net.sgen) > 0:
        for bus, p, q, in_svc in zip(
            net.sgen.bus, net.sgen.p_mw, net.sgen.q_mvar, net.sgen.in_service, strict=False
        ):
            if not bool(in_svc):
                continue
            i = pos_of[bus]
            arr["generation_p"][i] += float(p)
            arr["generation_q"][i] += float(q)
            has_gen[i] = True

    # Шунты (net.shunt): q_mvar/p_mw заданы при vn_kv шунта; переводим в См.
    # B[См] = -Q_MVAr / (vn_shunt_kv² ) ; знак: Q_MVAr>0 = поглощение реактива
    # (индуктивный) → отрицательная susceptance. P_MW>0 = активные потери → G>0.
    if len(net.shunt) > 0:
        for bus, p_mw, q_mvar, vn_kv, step, in_svc in zip(
            net.shunt.bus,
            net.shunt.p_mw,
            net.shunt.q_mvar,
            net.shunt.vn_kv,
            net.shunt.step,
            net.shunt.in_service,
            strict=False,
        ):
            if not bool(in_svc):
                continue
            i = pos_of[bus]
            vn = float(vn_kv)
            if vn <= 0:
                continue
            s = float(step)
            arr["shunt_g"][i] += s * float(p_mw) / (vn * vn)
            arr["shunt_b"][i] += -s * float(q_mvar) / (vn * vn)

    # ext_grid → slack-узел; несёт генерацию-баланс (P/Q не известны до PF — 0).
    eg_bus = int(net.ext_grid.bus.iloc[0])
    i_slack = pos_of[eg_bus]
    arr["node_type"][i_slack] = int(NodeType.SLACK)
    has_gen[i_slack] = True
    if "vm_pu" in net.ext_grid.columns:
        vm = float(net.ext_grid.vm_pu.iloc[0])
        arr["voltage_setpoint"][i_slack] = vm * float(arr["voltage_nominal"][i_slack])
        arr["voltage_magnitude"][i_slack] = vm * float(arr["voltage_nominal"][i_slack])

    arr["exist_gen"] = has_gen.astype(np.int8)
    arr["exist_load"] = has_load.astype(np.int8)
    return arr


# ---------------------------------------------------------------------------
# Ветви
# ---------------------------------------------------------------------------


def _build_branches(net: Any, f_hz: float) -> np.ndarray:
    """``net.line`` + ``net.trafo`` → массив ветвей (импеданс Ом, шунты См)."""
    rows: list[dict[str, Any]] = []
    next_bid = 1

    # ----- Линии -----
    for _, ln in net.line.iterrows():
        length = float(ln.length_km)
        parallel = max(int(ln.parallel), 1)
        # Последовательный импеданс: r_ohm_per_km·length / parallel (Ом).
        r = float(ln.r_ohm_per_km) * length / parallel
        x = float(ln.x_ohm_per_km) * length / parallel
        # Зарядная susceptance: B = 2π·f·C·length·parallel (См), C в нФ/км.
        b_total = 2.0 * math.pi * f_hz * float(ln.c_nf_per_km) * 1e-9 * length * parallel
        g_total = float(ln.g_us_per_km) * 1e-6 * length * parallel
        rows.append(
            {
                "id": next_bid,
                "from_node": int(ln.from_bus),
                "to_node": int(ln.to_bus),
                "branch_type": int(BranchType.LINE),
                "status": bool(ln.in_service),
                "resistance": r,
                "reactance": x,
                # Серийный шунт ветви (Π-схема): половина зарядной на каждый конец.
                "conductance_from": g_total / 2.0,
                "susceptance_from": b_total / 2.0,
                "conductance_to": g_total / 2.0,
                "susceptance_to": b_total / 2.0,
                "tap_ratio": 1.0,
                "phase_shift": 0.0,
                "current_limit_normal": float(ln.max_i_ka) * 1000.0
                if "max_i_ka" in net.line.columns and not _is_nan(ln.max_i_ka)
                else 0.0,
            }
        )
        next_bid += 1

    # ----- Трансформаторы -----
    for _, tr in net.trafo.iterrows():
        parallel = max(int(tr.parallel), 1)
        sn = float(tr.sn_mva)
        vn_hv = float(tr.vn_hv_kv)
        vn_lv = float(tr.vn_lv_kv)
        # Короткое замыкание: z%, r% на базе sn_mva, приведено к HV-стороне.
        z_pu = float(tr.vk_percent) / 100.0  # на базе sn_mva, vn_hv
        r_pu = float(tr.vkr_percent) / 100.0
        x_pu = math.sqrt(max(z_pu * z_pu - r_pu * r_pu, 0.0))
        z_base_hv = vn_hv * vn_hv / sn  # Ом (на собственной базе трафо)
        r_ohm = r_pu * z_base_hv / parallel
        x_ohm = x_pu * z_base_hv / parallel

        # Коэф. трансформации (физический, HV:LV) с учётом РПН.
        tap_factor = 1.0
        if not _is_nan(getattr(tr, "tap_pos", float("nan"))):
            tap_pos = float(tr.tap_pos)
            tap_neutral = float(tr.tap_neutral) if not _is_nan(tr.tap_neutral) else 0.0
            tap_step = float(tr.tap_step_percent) if not _is_nan(tr.tap_step_percent) else 0.0
            delta = (tap_pos - tap_neutral) * tap_step / 100.0
            tap_side = str(getattr(tr, "tap_side", "hv"))
            # РПН на HV: ratio растёт; на LV — обратно.
            tap_factor = (1.0 + delta) if tap_side == "hv" else 1.0 / (1.0 + delta)
        tap_ratio = (vn_hv / vn_lv) * tap_factor

        shift = float(tr.shift_degree) if not _is_nan(tr.shift_degree) else 0.0

        rows.append(
            {
                "id": next_bid,
                "from_node": int(tr.hv_bus),
                "to_node": int(tr.lv_bus),
                "branch_type": int(BranchType.TRANSFORMER),
                "status": bool(tr.in_service),
                "resistance": r_ohm,
                "reactance": x_ohm,
                # Намагничивающую ветвь (i0/pfe) опускаем: на IEEE-кейсах
                # пренебрежимо мала, z-вектор её не несёт.
                "tap_ratio": tap_ratio,
                "phase_shift": math.radians(shift),
                "current_limit_normal": 0.0,
            }
        )
        next_bid += 1

    arr = np.zeros(len(rows), dtype=_BRANCHES_DTYPE)
    for i, row in enumerate(rows):
        for key, value in row.items():
            if key in (arr.dtype.names or ()):
                arr[i][key] = value
    # parallel_id обязателен для KEY; единичные ветви → 1.
    if "parallel_id" in (arr.dtype.names or ()):
        arr["parallel_id"] = 1
    return arr


# ---------------------------------------------------------------------------
# Генераторы
# ---------------------------------------------------------------------------


def _build_generators(net: Any) -> np.ndarray:
    """``net.gen`` + ``net.sgen`` → таблица генераторов (вход контракта).

    На IEEE-фикстурах генерация уже агрегирована в узловую инжекцию
    (``node.generation_p/q``); отдельная таблица генераторов нужна контракту как
    непустая структура, но границы box не требуются. Заполняем минимально.
    """
    rows: list[tuple[int, int, float, float]] = []
    gid = 1
    if len(net.gen) > 0:
        for bus, p in zip(net.gen.bus, net.gen.p_mw, strict=False):
            rows.append((gid, int(bus), float(p), 0.0))
            gid += 1
    if len(net.sgen) > 0:
        for bus, p, q in zip(net.sgen.bus, net.sgen.p_mw, net.sgen.q_mvar, strict=False):
            rows.append((gid, int(bus), float(p), float(q)))
            gid += 1

    arr = np.zeros(len(rows), dtype=_GENERATORS_DTYPE)
    for i, (g_id, node_id, p, q) in enumerate(rows):
        arr[i]["id"] = g_id
        arr[i]["node_id"] = node_id
        arr[i]["power_output"] = p
        arr[i]["reactive_output"] = q
        arr[i]["status"] = True
    return arr


def _is_nan(value: Any) -> bool:
    """True, если ``value`` — NaN/None (pandapower кодирует «нет значения» как NaN)."""
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False
