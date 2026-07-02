"""Шунты/реактансы: реакторы→node.shunt_b, нормализация X короткозамыкателей.

**Декомпозиция:** каждая функция расщеплена на ``_*_on_arrays``-**ядро над
контрактными numpy-массивами** (мутирует переданные массивы in-place, возвращает
stats) + тонкий адаптер (``to_numpy().copy()`` → ядро → ``update_from_array``).
Ранний возврат при пустых reactors/branches оставлен в адаптере — без ``to_numpy``
при отсутствии данных.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gridstate.units import BASE_MVA
from gridstate.utils import id_to_pos_map


if TYPE_CHECKING:
    from gridstate.working import Working


# Некоторые входные форматы подменяют ветви-«короткозамыкатели» с R=X=0
# на R=0, X=1.0 Ом. Значение фиксировано в Омах → в p.u. зависит от класса
# напряжения узла. Константа должна совпадать с sentinel-X входного формата —
# имя публичное: производитель данных сверяет свою конвенцию кросс-тестом.
BREAKER_X_SENTINEL_OHM = 1.0
# Прежнее приватное имя — обратная совместимость импортов (telemetry.topology).
_BREAKER_X_SENTINEL_OHM = BREAKER_X_SENTINEL_OHM


def apply_reactors_to_node_shunt(model: Working) -> dict[str, int | float]:
    """Сложить B/G активных шунтов в ``shunt_b/g`` их узлов.

    Шунтирующие элементы (реакторы/БК) задаются таблицей ``shunts`` (B/G в См,
    со знаком и ON_LINE-статусом, применёнными во внешнем источнике данных). Без
    этого Y-bus игнорирует шунтовую компенсацию, и расчётный Q узла её не учитывает.

    Перебирает активные строки ``shunts`` (``status=True``) и для каждой, чей узел
    тоже активен, суммирует её ``B/G`` в ``shunt_b/g`` узла. Применять **после**
    применения ON_LINE-топологии.

    Returns:
        ``{"applied": N, "sum_b_added_S": float, "sum_g_added_S": float}``.
    """
    shunts_coll = getattr(model, "shunts", None)
    shunts_arr = shunts_coll.to_numpy() if shunts_coll is not None else None
    if shunts_arr is None or len(shunts_arr) == 0:
        return {"applied": 0, "sum_b_added_S": 0.0, "sum_g_added_S": 0.0}
    nodes = model.nodes.to_numpy().copy()
    stats = _aggregate_shunts_on_arrays(nodes, shunts_arr)
    model.nodes.update_from_array(nodes)
    return stats


def _aggregate_shunts_on_arrays(nodes_arr: Any, shunts_arr: Any) -> dict[str, int | float]:
    """Сложить B/G активных шунтов (таблица ``shunts``) в ``node.shunt_b/g``.

    ``shunts`` уже в См с применённым знаком и ON_LINE-статусом. Фильтрует по
    ``shunts.status`` И активности узла, суммирует в порядке строк. Мутирует
    ``nodes_arr``. Ранний возврат при пустых ``shunts`` — в адаптере
    :func:`apply_reactors_to_node_shunt` (без ``to_numpy`` при отсутствии данных).
    """
    by_id = id_to_pos_map(nodes_arr["id"])
    node_status: dict[int, bool] = {
        int(i): bool(s) for i, s in zip(nodes_arr["id"], nodes_arr["status"], strict=True)
    }

    applied = 0
    sum_b = 0.0
    sum_g = 0.0
    for s in shunts_arr:
        if not bool(s["status"]):
            continue
        nid = int(s["node_id"])
        if nid == 0 or not node_status.get(nid, False):
            continue
        idx = by_id.get(nid)
        if idx is None:
            continue
        b_add = float(s["susceptance"])
        g_add = float(s["conductance"])
        nodes_arr[idx]["shunt_b"] = float(nodes_arr[idx]["shunt_b"]) + b_add
        nodes_arr[idx]["shunt_g"] = float(nodes_arr[idx]["shunt_g"]) + g_add
        sum_b += b_add
        sum_g += g_add
        applied += 1

    return {"applied": applied, "sum_b_added_S": sum_b, "sum_g_added_S": sum_g}


def normalize_breaker_reactance(model: Working, *, eps_pu: float = 1e-3) -> dict[str, int | float]:
    """Привести X ветвей-«короткозамыкателей» к volt-aware значению ``X_pu=eps_pu``.

    Входной формат подменяет ветви с ``R=X=0`` (секции, выключатели, блок-связи) на
    ``R=0, X=1.0 Ом`` — ФИКСИРОВАННО в Омах, без учёта класса напряжения. После
    ``model_to_pu`` это даёт ``X_pu = 1.0 / (Vn²/S_base)``: на 500 кВ
    ``X_pu≈4e-5`` (норма), но на блочной шине 10.5 кВ ``X_pu≈0.9`` — инъекция
    P/Q генератора роняет V до ~0.79, и валидный V-замер ложно отбраковывается
    солвером (``|r/σ|≈6``). Поэтому β-выбросы ``dV_max`` возникают именно на
    LV-классах 6-16 кВ (блочные ген-шины после ``aggregate_generators_to_node``).

    Фикс (``x=1e-3 p.u.``, НЕ ``1.0 Ом``):
    для каждой такой ветви ``X = eps_pu·(Vn²/S_base)`` → ``X_pu=eps_pu`` на любом
    классе. ``Vn`` берётся из узла ``from`` (или ``to``). ``R=0`` сохраняется.
    Ветви без определённого ``Vn>0`` не трогаются.

    Применять **сразу после** загрузки модели, до применения телеметрии.
    Снижает ложную отбраковку V-замеров на LV-классах; на моделях без
    сентинельных ветвей — точный no-op.

    Args:
        model: рабочая модель (Working).
        eps_pu: целевой ``X_pu`` (default ``1e-3``).

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
        int(i): float(v) for i, v in zip(nodes_arr["id"], nodes_arr["voltage_nominal"], strict=True)
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
        if abs(r) > 1e-12 or abs(x - BREAKER_X_SENTINEL_OHM) > 1e-9:
            continue
        vn = vn_by_id.get(int(branches_arr[i]["from_node"]), 0.0)
        if vn <= 0.0:
            vn = vn_by_id.get(int(branches_arr[i]["to_node"]), 0.0)
        if vn <= 0.0:
            continue
        branches_arr[i]["reactance"] = eps_pu * (vn * vn / BASE_MVA)
        n += 1

    return n
