"""Шунты/реактансы: реакторы→node.shunt_b, нормализация X короткозамыкателей.

Выделено из telemetry/topology.py (Ф4 раскол по концернам).

**Ф4.1 (слайс 2):** обе функции расщеплены на ``_*_on_arrays``-**ядро над
контрактными numpy-массивами** (мутирует переданные массивы in-place, возвращает
stats — PSC-free) + тонкий адаптер (``to_numpy().copy()`` → ядро →
``update_from_array``). Массивные операции дословно прежние (тот же порядок
обхода, те же float-вычисления) → бит-в-бит; обе функции **уже** писали через
``update_from_array`` (``normalize_breaker_reactance`` — первый шаг каждого
пайплайна, т.е. round-trip ветвей валидирован canon'ом). Ранний возврат при
пустых reactors/branches оставлен в адаптере — дословно сохраняет «без to_numpy
при отсутствии данных».
"""

from __future__ import annotations

from typing import Any

from gridstate.units import BASE_MVA


# XmlFormat (power-system-core) подменяет ветви-«короткозамыкатели» с R=X=0
# на R=0, X=`_X_MIN_OHM`=1.0 Ом (см. `power_system/formats/xml_format.py`).
# Значение фиксировано в Омах → в p.u. зависит от класса напряжения узла.
# Константа должна совпадать с `_X_MIN_OHM` в загрузчике.
_BREAKER_X_SENTINEL_OHM = 1.0


def apply_reactors_to_node_shunt(model, *, sign: int = 1) -> dict[str, int | float]:
    """Сложить B/G активных реакторов в ``shunt_b/g`` их узлов.

    XML целевой системы хранит ШР (шунтирующие реакторы) в отдельной таблице
    ``<REACTORS>`` со ссылкой на узел через атрибут ``NODE``. XmlFormat
    загружает их в ``model.raw_tables['reactors']`` (с B в См после
    конверсии из мкСм), но **не суммирует** в ``model.nodes.shunt_b``.
    Без этого Y-bus игнорирует ШР, и расчётный Q узла не учитывает
    реактивную компенсацию зарядной мощности линии.

    Эта функция:
    1. Перебирает реакторы со ``status=True`` (активные после применения
       ON_LINE-топологии);
    2. Если узел тоже активен — суммирует ``sign·B/G`` реактора в
       ``shunt_b/g`` соответствующего узла.

    Применять **после** применения ON_LINE-топологии (которая
    активирует реакторы по их ON_LINE-формулам).

    Знак (``sign``): конвенция входного формата ``reactors.susceptance`` обратна
    EE Y-bus — входной формат ``B>0``=ШР (индуктивный, поглощает Q, понижает V),
    ``B<0``=БК (ёмкостный), тогда как в EE ``shunt_b>0``=ёмкостный.
    Поэтому физически корректный по узловому Q-балансу вариант —
    ``sign=-1`` (``shunt_b -= B``): с ним расчётный ``q_node_shunt``
    совпадает по знаку и величине с ``dq_shunt`` эталонной SE (на
    региональных моделях). **Однако** A/B на
    4 региональных моделях × IPM (поверх breaker-fix) показал, что ``sign=-1`` как
    default регрессирует V/δ (на региональных моделях dD_p50 +283%, dV_p50 +31%) —
    неверный знак в production исторически компенсировал недостающую
    легальную реактивную нагрузку (см. issue_reactor_sign_q_balance,
    memory ``reactor_sign_q_balance_finding``). Поэтому **default
    ``sign=1`` сохранён** (= историческое поведение, бит-в-бит no-op);
    ``sign=-1`` оставлен как opt-in рычаг для Q-баланс-экспериментов и
    будущих combo (когда недостающие данные будут восстановлены легально).

    Args:
        model: PowerSystemModel от XmlFormat.
        sign: знак при сложении B/G. ``1`` (default) — историческое
            поведение; ``-1`` — физически корректный по Q-балансу (но
            регрессирует V/δ как production-default, см. выше).

    Returns:
        ``{"applied": N, "sum_b_added_S": float, "sum_g_added_S": float}``.
    """
    reac = model.raw_tables.get("reactors")
    if reac is None or len(reac) == 0:
        return {"applied": 0, "sum_b_added_S": 0.0, "sum_g_added_S": 0.0}

    nodes = model.nodes.to_numpy().copy()
    stats = _apply_reactors_on_arrays(nodes, reac, sign=sign)
    model.nodes.update_from_array(nodes)
    return stats


def _apply_reactors_on_arrays(
    nodes_arr: Any, reactors_arr: Any, *, sign: int = 1
) -> dict[str, int | float]:
    """Сложить ``sign·B/G`` active-реакторов в ``shunt_b/g`` их узлов (мутирует ``nodes_arr``).

    Читает ``node.{id,status,shunt_b,shunt_g}`` и raw ``reactors.{node_id,status,
    susceptance,conductance}`` (B/G в мкСм → ×1e-6 в См). Возвращает
    ``{"applied":N,"sum_b_added_S":float,"sum_g_added_S":float}``. ``reactors_arr``
    может быть ``None``/пустым (тогда no-op).
    """
    if reactors_arr is None or len(reactors_arr) == 0:
        return {"applied": 0, "sum_b_added_S": 0.0, "sum_g_added_S": 0.0}

    by_id: dict[int, int] = {int(nodes_arr[i]["id"]): i for i in range(len(nodes_arr))}
    node_status: dict[int, bool] = {
        int(nodes_arr[i]["id"]): bool(nodes_arr[i]["status"]) for i in range(len(nodes_arr))
    }

    applied = 0
    sum_b = 0.0
    sum_g = 0.0
    for r in reactors_arr:
        if not bool(r["status"]):
            continue
        nid = int(r["node_id"])
        if nid == 0 or not node_status.get(nid, False):
            continue
        idx = by_id.get(nid)
        if idx is None:
            continue
        # Реактор B/G в model.raw_tables['reactors'] хранятся в См
        # (XmlFormat НЕ конвертирует мкСм→См для них; смотри ниже).
        # На ряде XML-выгрузок значения типа -272109 — это мкСм
        # (B_SI ≈ -0.272 См), т.е. конверсию делаем ЗДЕСЬ.
        # ``sign`` флипает всю конвенцию входной формат→EE (см. docstring).
        b_add = sign * float(r["susceptance"]) * 1e-6
        g_add = sign * float(r["conductance"]) * 1e-6
        nodes_arr[idx]["shunt_b"] = float(nodes_arr[idx]["shunt_b"]) + b_add
        nodes_arr[idx]["shunt_g"] = float(nodes_arr[idx]["shunt_g"]) + g_add
        sum_b += b_add
        sum_g += g_add
        applied += 1

    return {"applied": applied, "sum_b_added_S": sum_b, "sum_g_added_S": sum_g}


def normalize_breaker_reactance(model, *, eps_pu: float = 1e-3) -> dict[str, int | float]:
    """Привести X ветвей-«короткозамыкателей» к volt-aware значению ``X_pu=eps_pu``.

    XmlFormat подменяет ветви с ``R=X=0`` (секции, выключатели, блок-связи) на
    ``R=0, X=1.0 Ом`` — ФИКСИРОВАННО в Омах, без учёта класса напряжения. После
    ``model_to_pu`` это даёт ``X_pu = 1.0 / (Vn²/S_base)``: на 500 кВ
    ``X_pu≈4e-5`` (норма), но на блочной шине 10.5 кВ ``X_pu≈0.9`` — инъекция
    P/Q генератора роняет V до ~0.79, и валидный V-замер ложно отбраковывается
    солвером (``|r/σ|≈6``). Поэтому β-выбросы ``dV_max`` возникают именно на
    LV-классах 6-16 кВ (блочные ген-шины после ``aggregate_generators_to_node``).

    Фикс (рецепт ``rastr_format_quirks`` #4: ``x=1e-3 p.u.``, НЕ ``1.0 Ом``):
    для каждой такой ветви ``X = eps_pu·(Vn²/S_base)`` → ``X_pu=eps_pu`` на любом
    классе. ``Vn`` берётся из узла ``from`` (или ``to``). ``R=0`` сохраняется.
    Ветви без определённого ``Vn>0`` не трогаются.

    Применять **сразу после** загрузки модели, до применения телеметрии.
    A/B на 4 региональных моделях × IPM: ``dV_max`` 0.190→0.123 (−35%), 0.182→0.174 (−4%),
    на части моделей — точный no-op (сентинелей нет). Нейтрально по dSta.

    Args:
        model: PowerSystemModel от XmlFormat.
        eps_pu: целевой ``X_pu`` (default ``1e-3``, рецепт ``rastr_format_quirks`` #4).

    Returns:
        ``{"normalized": N, "eps_pu": eps_pu}``.
    """
    arr = model.branches.to_numpy()
    if arr is None or len(arr) == 0:
        return {"normalized": 0, "eps_pu": eps_pu}

    nodes = model.nodes.to_numpy()
    arr = arr.copy()
    n = _normalize_breaker_reactance_on_arrays(arr, nodes, eps_pu=eps_pu)
    if n:
        model.branches.update_from_array(arr)
    return {"normalized": n, "eps_pu": eps_pu}


def _normalize_breaker_reactance_on_arrays(
    branches_arr: Any, nodes_arr: Any, *, eps_pu: float = 1e-3
) -> int:
    """Привести X сентинел-ветвей (R=0, X=1.0 Ом) к ``X_pu=eps_pu`` (мутирует ``branches_arr``).

    Читает ``branch.{resistance,reactance,from_node,to_node}`` и
    ``node.{id,voltage_nominal}``; пишет ``branch.reactance``. ``Vn`` берётся из
    узла ``from`` (фолбэк ``to``); ветви без ``Vn>0`` не трогаются. Возвращает
    число нормализованных ветвей.
    """
    vn_by_id: dict[int, float] = {
        int(nodes_arr[i]["id"]): float(nodes_arr[i]["voltage_nominal"])
        for i in range(len(nodes_arr))
    }

    n = 0
    for i in range(len(branches_arr)):
        r = float(branches_arr[i]["resistance"])
        x = float(branches_arr[i]["reactance"])
        # Сентинел загрузчика: R ровно 0 и X ровно 1.0 Ом. Реальные ветви
        # всегда имеют R>0, поэтому совпадение уникально идентифицирует
        # подставленный «короткозамыкатель».
        # Идемпотентность: после нормализации X = eps_pu·(Vn²/S_base) ≠ 1.0
        # на всех классах, КРОМЕ совпадения Vn²=S_base/eps_pu (при default
        # 1e-3 и S_base=100 это Vn≈316 кВ — нет в реальных классах). Даже
        # на этой точке повторный матч пересчитывает X в то же значение
        # (value-idempotent), завышая лишь счётчик "normalized".
        if abs(r) > 1e-12 or abs(x - _BREAKER_X_SENTINEL_OHM) > 1e-9:
            continue
        vn = vn_by_id.get(int(branches_arr[i]["from_node"]), 0.0)
        if vn <= 0.0:
            vn = vn_by_id.get(int(branches_arr[i]["to_node"]), 0.0)
        if vn <= 0.0:
            continue
        branches_arr[i]["reactance"] = eps_pu * (vn * vn / BASE_MVA)
        n += 1

    return n
