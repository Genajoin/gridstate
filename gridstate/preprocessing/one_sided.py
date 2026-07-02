"""Одностороннее отключение ветвей — физика свёртки + ПАРКОВАННЫЙ детектор.

⚠️ ВАЖНО (правка эксперта-пользователя 2026-05-31): НЕЛЬЗЯ выводить одностороннее
отключение из статуса УЗЛА. У эталонной SE признак СТА ветви **не бинарный**: помимо 0 (вкл)
и 1 (откл с обеих сторон) есть ещё **два состояния — отключена со стороны НАЧАЛА и со
стороны КОНЦА** (per-branch-per-end). Это свойство КОНКРЕТНОЙ ЛИНИИ на КОНКРЕТНОМ её
конце, а НЕ статуса узла. Подстанция-шина держит МНОГО линий: сама шина включена
(``node.sta=0``), но одна линия от неё отключена с этого конца. Гасить весь узел,
чтобы пометить одну линию односторонней — **испорченная физика** (тогда ВСЕ линии узла
стали бы односторонними). Поэтому детектор «node off + branch on» ниже —
**НЕ настоящая односторонность**, он на это НЕ способен.

Состояние данных (2026-05-31): небинарного ``sta`` НЕТ ни в одном доступном источнике —
5 дампов эталонной SE (``sta∈{0,1}``) и 41 тестовая модель во входном формате (``sta=0``
поголовно). Экспорт во входной формат, видимо, схлопывает per-end-статус в бинарный.
Настоящий признак, вероятно, в нативной БД эталонной SE до экспорта. **Пока такой модели
нет — реализовать корректную
физику односторонности нельзя; алгоритм ПАРКУЕТСЯ (default OFF, не в production).**

Что ОСТАЁТСЯ валидным и пригодится, когда модель с per-end STA появится:

* :func:`_driving_point_shunt` — КОРРЕКТНАЯ физика свёртки энергизированной линии с
  открытым дальним концом в шунт живого узла: ``Y_seen = yc_live + ys·yc_dead/
  (ys+yc_dead)`` (точная Kron-редукция открытого узла из 2-портовой Y). Для чистой
  зарядной линии → ``≈ полная B`` (не B/2: series-импеданс ничтожен против шунта,
  дальняя половина B «протягивается» к живому узлу) + Ферранти. Триггер для неё должен
  приходить из per-branch STA (FROM_OPEN/TO_OPEN), а НЕ из статуса узла.

* :func:`classify_branch_connectivity` — node-status-производный классификатор. ⚠️ Это
  НЕ STA эталонной SE: ``FROM_OPEN``/``TO_OPEN`` здесь выводятся из статусов узлов и НЕ
  отражают настоящую односторонность (узел односторонней линии остаётся ON). Оставлен
  как утилита диагностики node-каскада, не как источник one-sided.

:func:`fold_one_sided_branches` — ПАРКОВАН: его детектор (node off + branch on)
физически неверен (см. выше). Не включать в production. Эмпирически на региональных
моделях это всё равно no-op (folded=0). Сохранён вместе с физикой ради будущей
корректной реализации.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from gridstate.working import Working


__all__ = [
    "classify_branch_connectivity",
    "fold_one_sided_branches",
]


# Не-бинарные состояния связности ветви (выводимые, не хранимые).
ON = "on"  # ветвь активна, оба конца активны
OFF = "off"  # ветвь неактивна ИЛИ оба конца неактивны
FROM_OPEN = "from_open"  # ветвь активна, начало (from) отключено, конец (to) живой
TO_OPEN = "to_open"  # ветвь активна, конец (to) отключён, начало (from) живое

_LINE = 0  # BranchType.LINE


def classify_branch_connectivity(model: Working) -> dict[int, str]:
    """Вернуть node-status-производный статус связности каждой ветви.

    ⚠️ НЕ путать с STA эталонной SE: ``FROM_OPEN``/``TO_OPEN`` здесь выводятся из статусов
    УЗЛОВ и НЕ отражают настоящее одностороннее отключение (его признак — non-binary
    per-branch STA эталонной SE, а узел односторонней линии остаётся ON). Это утилита
    диагностики node-каскада, не источник one-sided. См. docstring модуля.

    Returns:
        ``{branch_id: state}``, где ``state`` ∈ {``ON``, ``OFF``, ``FROM_OPEN``,
        ``TO_OPEN``}. ``FROM_OPEN`` — начало отключено (линия под напряжением со
        стороны ``to``); ``TO_OPEN`` — конец отключён (под напряжением со стороны
        ``from``).
    """
    nodes = model.nodes.to_numpy()
    nstat = {int(n["id"]): bool(n["status"]) for n in nodes}
    out: dict[int, str] = {}
    for b in model.branches.to_numpy():
        bid = int(b["id"])
        if not bool(b["status"]):
            out[bid] = OFF
            continue
        f_on = nstat.get(int(b["from_node"]), False)
        t_on = nstat.get(int(b["to_node"]), False)
        if f_on and t_on:
            out[bid] = ON
        elif f_on and not t_on:
            out[bid] = TO_OPEN
        elif t_on and not f_on:
            out[bid] = FROM_OPEN
        else:
            out[bid] = OFF
    return out


def _driving_point_shunt(
    r: float,
    x: float,
    g: float,
    b: float,
    g_live: float,
    b_live: float,
    g_dead: float,
    b_dead: float,
) -> complex:
    """Driving-point admittance линии с открытым дальним концом (физ. См).

    Π-схема: живой конец видит свой шунт ``yc_live`` параллельно с
    [series ``ys`` → мёртвый шунт ``yc_dead`` на землю]. Точно:

        Y_seen = yc_live + ys · yc_dead / (ys + yc_dead)

    где ``ys = 1/(r+jx)``, ``yc_* = (g_*+j b_*) + (g+j b)/2``. Для чистой зарядной
    линии (r,x→0, g_*=b_*=0) → ``Y_seen ≈ j·b`` (полная B), что и ожидается
    физически (последовательный импеданс ничтожен против шунта).
    """
    ys = 1.0 / complex(r, x)
    yc_live = complex(g_live, b_live) + complex(g, b) * 0.5
    yc_dead = complex(g_dead, b_dead) + complex(g, b) * 0.5
    denom = ys + yc_dead
    if denom == 0:
        return yc_live
    return yc_live + ys * yc_dead / denom


def fold_one_sided_branches(
    model: Working,
    *,
    require_charging: bool = True,
    eps_shunt_s: float = 1e-9,
    lines_only: bool = True,
) -> dict[str, object]:
    """ПАРКОВАН (физически неверный детектор). Свернуть «активная ветвь + один
    отключённый УЗЕЛ-конец» в шунт живого узла.

    ⚠️ Детектор по статусу УЗЛА — НЕ настоящая односторонность эталонной SE (см. docstring
    модуля): одностороннее отключение — per-branch-per-end признак (non-binary STA),
    а узел односторонней линии остаётся включённым. Эта функция сработает лишь когда
    весь узел погашен — что НЕ соответствует физике (тогда все линии узла «односторонни»).
    Default OFF, не в production. Физика свёртки (:func:`_driving_point_shunt`) верна и
    переиспользуема, когда появится модель с per-branch STA как источником триггера.

    Запускать **ДО** ``disable_orphan_branches`` — иначе он погасит ветвь раньше.

    Args:
        model: модель сети.
        require_charging: складывать только ветви с ненулевым шунтом (|шунт| >
            ``eps_shunt_s``). По умолчанию ``True`` — breaker'ы/трансформаторы без
            зарядной B (нечего сохранять) пропускаются и достаются обычному
            ``disable_orphan_branches``. Делает шаг no-op на данных без charging.
        eps_shunt_s: порог «ненулевого» шунта, См.
        lines_only: складывать только ``branch_type == LINE`` с ``tap≈1``. Для
            трансформаторов/tap-ветвей driving-point с переводом базы напряжения
            некорректен в физ. единицах, а зарядная мощность у них ничтожна.

    Returns:
        ``{"folded": N, "skipped_zero_shunt": N, "skipped_breaker": N,
        "skipped_transformer": N, "q_folded_mvar_at_vnom": float, "samples": [...]}``.
    """
    nodes = model.nodes.to_numpy()
    nstat = {int(n["id"]): bool(n["status"]) for n in nodes}
    vnom = {int(n["id"]): float(n["voltage_nominal"]) for n in nodes}

    folded = 0
    skipped_zero_shunt = 0
    skipped_breaker = 0
    skipped_transformer = 0
    q_folded_mvar_at_vnom = 0.0
    samples: list[dict[str, object]] = []

    for b in model.branches.to_numpy():
        if not bool(b["status"]):
            continue
        f, t = int(b["from_node"]), int(b["to_node"])
        f_on, t_on = nstat.get(f, False), nstat.get(t, False)
        if f_on == t_on:
            continue  # ON (оба) или OFF (оба) — не односторонняя
        live, dead = (f, t) if f_on else (t, f)

        if lines_only:
            is_line = int(b["branch_type"]) == _LINE
            tap_unity = abs(float(b["tap_ratio"]) - 1.0) <= 1e-6
            if not (is_line and tap_unity):
                skipped_transformer += 1
                continue

        r, x = float(b["resistance"]), float(b["reactance"])
        if r == 0.0 and x == 0.0:
            skipped_breaker += 1
            continue

        g, bb = float(b["conductance"]), float(b["susceptance"])
        gf, bf = float(b["conductance_from"]), float(b["susceptance_from"])
        gt, bt = float(b["conductance_to"]), float(b["susceptance_to"])
        # шунты живого/мёртвого конца
        if live == f:
            g_live, b_live, g_dead, b_dead = gf, bf, gt, bt
        else:
            g_live, b_live, g_dead, b_dead = gt, bt, gf, bf

        shunt_mag = abs(bb) + abs(bf) + abs(bt) + abs(g) + abs(gf) + abs(gt)
        if require_charging and shunt_mag <= eps_shunt_s:
            skipped_zero_shunt += 1
            continue

        y_seen = _driving_point_shunt(r, x, g, bb, g_live, b_live, g_dead, b_dead)

        node = model.nodes.get_by_id(live)
        if node is None:
            continue
        # Single mutation style: both the shunt bump on the live node and the branch
        # de-energization go through the collection ``.update`` API.
        model.nodes.update(
            live,
            {
                "shunt_g": float(node.shunt_g) + y_seen.real,
                "shunt_b": float(node.shunt_b) + y_seen.imag,
            },
        )
        model.branches.update(int(b["id"]), {"status": False})

        vn = vnom.get(live, 0.0)
        # Q[МВАр] = b[См]·(vn[кВ])²:  b·(vn·1e3 В)² = b·vn²·1e6 ВАр = b·vn² МВАр.
        q_mvar = y_seen.imag * vn * vn  # ёмкостная Q при V=Vnom
        folded += 1
        q_folded_mvar_at_vnom += q_mvar
        if len(samples) < 30:
            samples.append(
                {
                    "branch_id": int(b["id"]),
                    "name": str(b["name"]),
                    "live": live,
                    "dead": dead,
                    "vn_kv": vn,
                    "y_seen_s": (y_seen.real, y_seen.imag),
                    "q_mvar": round(q_mvar, 3),
                }
            )
    return {
        "folded": folded,
        "skipped_zero_shunt": skipped_zero_shunt,
        "skipped_breaker": skipped_breaker,
        "skipped_transformer": skipped_transformer,
        "q_folded_mvar_at_vnom": q_folded_mvar_at_vnom,
        "samples": samples,
    }
