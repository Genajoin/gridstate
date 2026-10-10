"""Адаптер ``pandapower.net`` → контракт gridstate (``SEInput``).

Конвертирует описание сети из ``pandapower`` в контрактные структурированные
массивы (:data:`gridstate.contract.tables.SE_INPUT`) и оборачивает их в
:class:`~gridstate.contract.containers.SEInput` (``derived=None`` — формат-зависимые
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
from gridstate.contract.containers import SEInput
from gridstate.contract.tables import SE_INPUT, SE_OUTPUT, io_dtype
from gridstate.units import BASE_MVA
from gridstate.working import Working


if TYPE_CHECKING:
    from collections.abc import Iterator

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


# Backing-массивы несут INPUT/WORKING ⊕ OUTPUT-колонки — ровно как нулевые
# коллекции Working.empty(): пайплайн пишет OUTPUT-колонки (voltage_magnitude/
# p_inj_calc/перетоки/estimated_si) прямо в живые коллекции. Единый билдер —
# io_dtype (contract.tables).
_NODES_DTYPE = io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes)
_BRANCHES_DTYPE = io_dtype(SE_INPUT.branches, SE_OUTPUT.branches)
_MEASUREMENTS_DTYPE = io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements)
_GENERATORS_DTYPE = SE_INPUT.generators.input_dtype()


def from_pandapower(
    net: Any,
    *,
    measurements: np.ndarray | None = None,
    trafo_model: str | None = None,
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
        trafo_model: equivalent circuit of the transformer magnetising branch,
            ``"t"`` or ``"pi"`` (the pandapower power-flow option of the same
            name). ``None`` takes the option of the last power flow run on
            ``net`` (``net._options["trafo_model"]``), or pandapower's default
            ``"t"`` if the network has not been solved.

    Returns:
        :class:`SEInput` with ``derived=None`` (the model already carries nodes,
        branches, the operating point and measurements; the pipeline steps that
        need derived inputs are skipped).

    Raises:
        ImportError: если pandapower не установлен (см. extra ``[test-models]``).
        ValueError: при ≠1 ext_grid, ``net.sn_mva ≠ 100``, пустой сети или
            неизвестном ``trafo_model``.
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
    model = _resolve_trafo_model(net, trafo_model)

    nodes = _build_nodes(net)
    branches = _build_branches(net, f_hz, model)
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


def network_injection(net: Any) -> tuple[Any, Any]:
    """Nodal network injection of a solved ``net``: generation - load, MW / Mvar.

    pandapower's ``res_bus`` is the sum over all bus elements *including* the
    bus shunts. gridstate models a bus shunt inside the Y-bus (the adapter writes
    it to ``shunt_g`` / ``shunt_b``), so the injection it estimates,
    ``V * conj(Ybus V)``, excludes the shunt power. This helper adds the shunt
    power back: ``-res_bus + res_shunt`` per bus.

    Returns:
        ``(p_mw, q_mvar)`` as pandas Series indexed by the ``net.bus`` index.
    """
    p = -net.res_bus.p_mw.copy()
    q = -net.res_bus.q_mvar.copy()
    if len(net.shunt):
        p = p.add(net.res_shunt.p_mw.groupby(net.shunt.bus).sum(), fill_value=0.0)
        q = q.add(net.res_shunt.q_mvar.groupby(net.shunt.bus).sum(), fill_value=0.0)
    return p, q


_TRAFO_MODELS = ("t", "pi")


def _resolve_trafo_model(net: Any, trafo_model: str | None) -> str:
    """Transformer equivalent circuit: explicit argument, last power flow option, ``"t"``."""
    if trafo_model is None:
        options = getattr(net, "_options", None) or {}
        trafo_model = str(options.get("trafo_model", "t"))
    if trafo_model not in _TRAFO_MODELS:
        raise ValueError(f"trafo_model must be one of {_TRAFO_MODELS}, got {trafo_model!r}.")
    return trafo_model


# ---------------------------------------------------------------------------
# Узлы
# ---------------------------------------------------------------------------


def _iter_bus_rows(
    table: Any, pos_of: dict[Any, int], *columns: str
) -> Iterator[tuple[int, list[Any]]]:
    """Yield ``(node_pos, [col_values...])`` for in-service rows of a pp element table.

    Factors out the shared preamble of the per-element loops (load/gen/sgen/
    shunt): skip out-of-service rows and resolve the bus index to a node
    position. ``columns`` are the extra columns to read alongside the mandatory
    ``bus`` / ``in_service``; iteration order matches ``zip`` over the columns.
    """
    if len(table) == 0:
        return
    extra = [getattr(table, c) for c in columns]
    for bus, in_svc, *vals in zip(table.bus, table.in_service, *extra, strict=False):
        if not bool(in_svc):
            continue
        yield pos_of[bus], vals


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
    for i, (p, q) in _iter_bus_rows(net.load, pos_of, "p_mw", "q_mvar"):
        arr["load_p"][i] += float(p)
        arr["load_q"][i] += float(q)
        has_load[i] = True

    # Генерация PV (net.gen): задаёт node_type=PV (если узел не slack).
    for i, (p,) in _iter_bus_rows(net.gen, pos_of, "p_mw"):
        arr["generation_p"][i] += float(p)
        has_gen[i] = True
        arr["node_type"][i] = int(NodeType.PV)
    # Vsetpoint для PV (vm_pu задан на gen).
    if len(net.gen) > 0 and "vm_pu" in net.gen.columns:
        for i, (vm,) in _iter_bus_rows(net.gen, pos_of, "vm_pu"):
            arr["voltage_setpoint"][i] = float(vm) * float(arr["voltage_nominal"][i])

    # Статическая генерация (net.sgen) — PQ-инжекция (+).
    for i, (p, q) in _iter_bus_rows(net.sgen, pos_of, "p_mw", "q_mvar"):
        arr["generation_p"][i] += float(p)
        arr["generation_q"][i] += float(q)
        has_gen[i] = True

    # Шунты (net.shunt): q_mvar/p_mw заданы при vn_kv шунта; переводим в См.
    # B[См] = -Q_MVAr / (vn_shunt_kv² ) ; знак: Q_MVAr>0 = поглощение реактива
    # (индуктивный) → отрицательная susceptance. P_MW>0 = активные потери → G>0.
    for i, (p_mw, q_mvar, vn_kv, step) in _iter_bus_rows(
        net.shunt, pos_of, "p_mw", "q_mvar", "vn_kv", "step"
    ):
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


def _set_row(row: np.void, names: tuple[str, ...], values: dict[str, Any]) -> None:
    """Записать поля ``values`` прямо в строку структурного массива (по имени)."""
    for key, value in values.items():
        if key in names:
            row[key] = value


def _wye_delta(
    z_series: complex, y_mag: complex, r_ratio: float, x_ratio: float
) -> tuple[complex, complex, complex]:
    """T equivalent of a transformer -> pi equivalent (star-delta transform).

    The series impedance ``z_series`` is split into a from-side part
    (``r_ratio`` / ``x_ratio`` of R / X) and a to-side part, with the
    magnetising admittance ``y_mag`` at the star point. Returns the pi
    equivalent ``(z_series, y_from, y_to)``: the series impedance and the full
    shunt admittances at the from and the to end. Same arithmetic as
    pandapower's ``build_branch._wye_delta`` (all values in one per-unit base).
    """
    za = complex(z_series.real * r_ratio, z_series.imag * x_ratio)
    zb = complex(z_series.real * (1.0 - r_ratio), z_series.imag * (1.0 - x_ratio))
    zc = 1.0 / y_mag
    z_sum = za * zb + za * zc + zb * zc
    return z_sum / zc, zb / z_sum, za / z_sum


def _trafo_value(tr: Any, column: str, default: float) -> float:
    """``tr[column]`` as float, ``default`` when the column is absent or NaN."""
    try:
        value = float(getattr(tr, column))
    except (AttributeError, TypeError, ValueError):  # absent column, None / pd.NA
        return default
    return default if math.isnan(value) else value


def _build_branches(net: Any, f_hz: float, trafo_model: str = "t") -> np.ndarray:
    """``net.line`` + ``net.trafo`` → массив ветвей (импеданс Ом, шунты См)."""
    # pandas is not a gridstate dependency: it arrives with pandapower, so it is
    # imported here (only for pd.isna on table cells), never at module import.
    import pandas as pd  # type: ignore[import-untyped]

    # Пишем прямо в преаллоцированный массив (без промежуточного list[dict]):
    # число ветвей известно заранее = линии + трансформаторы.
    arr = np.zeros(len(net.line) + len(net.trafo), dtype=_BRANCHES_DTYPE)
    names = arr.dtype.names or ()
    next_bid = 1
    bi = 0

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
        _set_row(
            arr[bi],
            names,
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
                if "max_i_ka" in net.line.columns and not pd.isna(ln.max_i_ka)
                else 0.0,
            },
        )
        next_bid += 1
        bi += 1

    # ----- Трансформаторы -----
    # The equations of pandapower's own branch model (build_branch: series
    # impedance, magnetising admittance, T -> pi conversion), so that the
    # Y-bus built from the adapter output equals pandapower's.
    for _, tr in net.trafo.iterrows():
        parallel = max(int(tr.parallel), 1)
        sn = float(tr.sn_mva)

        # Rated voltages adjusted by the tap changer position; the ratio
        # (physical, HV:LV) follows from them.
        vn_hv = float(tr.vn_hv_kv)
        vn_lv = float(tr.vn_lv_kv)
        if not pd.isna(getattr(tr, "tap_pos", float("nan"))):
            tap_pos = float(tr.tap_pos)
            tap_neutral = float(tr.tap_neutral) if not pd.isna(tr.tap_neutral) else 0.0
            tap_step = float(tr.tap_step_percent) if not pd.isna(tr.tap_step_percent) else 0.0
            delta = (tap_pos - tap_neutral) * tap_step / 100.0
            if str(getattr(tr, "tap_side", "hv")) == "hv":
                vn_hv *= 1.0 + delta
            else:
                vn_lv *= 1.0 + delta
        tap_ratio = vn_hv / vn_lv

        # The same ratio in p.u. of the bus voltages, as ``model_to_pu`` gets it.
        vn_hv_bus = float(net.bus.at[tr.hv_bus, "vn_kv"])
        vn_lv_bus = float(net.bus.at[tr.lv_bus, "vn_kv"])
        tap_pu = tap_ratio
        if vn_hv_bus > 0 and vn_lv_bus > 0 and not math.isclose(vn_hv_bus, vn_lv_bus):
            tap_pu = tap_ratio / (vn_hv_bus / vn_lv_bus)

        # pandapower's per-unit branch data (base: BASE_MVA and the LV bus
        # voltage). Both the short-circuit impedance and the magnetising
        # admittance sit behind the ideal transformer, at the tap-adjusted LV
        # rated voltage.
        z_base_lv = vn_lv_bus * vn_lv_bus / BASE_MVA
        z_ohm = float(tr.vk_percent) / 100.0 * vn_lv * vn_lv / sn / parallel
        r_ohm = float(tr.vkr_percent) / 100.0 * vn_lv * vn_lv / sn / parallel
        x_ohm = math.copysign(math.sqrt(max(z_ohm * z_ohm - r_ohm * r_ohm, 0.0)), z_ohm)
        z_series = complex(r_ohm, x_ohm) / z_base_lv
        # Magnetising admittance: pfe_kw -> G, i0_percent -> |Y|, B inductive.
        p_fe_mw = float(tr.pfe_kw) / 1000.0
        y_mag_mva = float(tr.i0_percent) / 100.0 * sn
        b_mag_mva = -math.sqrt(max(y_mag_mva * y_mag_mva - p_fe_mw * p_fe_mw, 0.0))
        y_mag = complex(p_fe_mw, b_mag_mva) * parallel / (vn_lv * vn_lv) * z_base_lv
        if y_mag == 0:
            y_from = y_to = 0j
        elif trafo_model == "t":
            # The leakage impedance is split between the two windings around the
            # magnetising branch; the pi equivalent changes the series impedance too.
            z_series, y_from, y_to = _wye_delta(
                z_series,
                y_mag,
                _trafo_value(tr, "leakage_resistance_ratio_hv", 0.5),
                _trafo_value(tr, "leakage_reactance_ratio_hv", 0.5),
            )
        else:
            y_from = y_to = y_mag / 2.0

        # Contract units. The impedance goes to the from (HV) side by the actual
        # ratio: build_ybus uses Ys = |t|^2 / z_from (see gridstate.ybus). The
        # shunts are in S of the from-side base, the one model_to_pu converts
        # every branch quantity with. pandapower places the from-end shunt
        # behind the ideal transformer (Yff = (Ys + y_from) / |t|^2), while
        # build_ybus connects it at the from node, before the ideal
        # transformer (Yff = Ys / |t|^2 + Yc_from): refer it to the from side
        # by dividing by |t|^2. The to-end shunt needs no conversion.
        z_base_hv = vn_hv_bus * vn_hv_bus / BASE_MVA
        z_from = z_series * tap_pu * tap_pu * z_base_hv
        y_from /= tap_pu * tap_pu * z_base_hv
        y_to /= z_base_hv

        shift = float(tr.shift_degree) if not pd.isna(tr.shift_degree) else 0.0

        _set_row(
            arr[bi],
            names,
            {
                "id": next_bid,
                "from_node": int(tr.hv_bus),
                "to_node": int(tr.lv_bus),
                "branch_type": int(BranchType.TRANSFORMER),
                "status": bool(tr.in_service),
                "resistance": z_from.real,
                "reactance": z_from.imag,
                "conductance_from": y_from.real,
                "susceptance_from": y_from.imag,
                "conductance_to": y_to.real,
                "susceptance_to": y_to.imag,
                "tap_ratio": tap_ratio,
                "phase_shift": math.radians(shift),
                "current_limit_normal": 0.0,
            },
        )
        next_bid += 1
        bi += 1

    # parallel_id обязателен для KEY; единичные ветви → 1.
    if "parallel_id" in names:
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
