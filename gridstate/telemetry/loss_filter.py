"""Physics check of the reactive balance of a line (pi model).

:func:`compute_expected_q_imbalance_mvar` estimates the physically expected
``|Q_beg + Q_end|`` of a line from its charging susceptance (``V^2 * B``) and
the series reactive losses (``(P^2 + Q^2) / V^2 * X``).
:func:`is_branch_q_consistent_with_physics` compares an observed
``|Q_beg + Q_end|`` with that expectation; the telemetry kernel uses it as a
charging-aware threshold instead of a flat one (on a 750 kV line with a large
charging susceptance the expected imbalance exceeds 1000 Mvar, which a flat
50 Mvar threshold would wrongly reject).
"""

from __future__ import annotations


def compute_expected_q_imbalance_mvar(
    *,
    vn_kv: float,
    susceptance_si: float,
    reactance_si: float,
    p_typical_mw: float = 0.0,
    q_typical_mvar: float = 0.0,
) -> float:
    """Расчётное ``|Q_beg + Q_end|`` для линии через физику π-схемы.

    На активной ВЛ (без трафо) сумма Q-потоков на обоих концах =
    Q-потери в реактивности минус Q-генерация зарядной B::

        Q_beg + Q_end = (P²+Q²)/V²·X  -  V²·B

    где B — полная зарядная (хранится в нашей storage как
    ``branch.susceptance``, в Сименсах) — делится пополам в шунты
    обоих концов π-схемы (V²·B/2 + V²·B/2 = V²·B).

    Для приблизительной оценки используем ``Vnom`` вместо реального V
    (на ВЛ V обычно близок к номиналу ±5 %). Также можно опустить
    Q-потери в X если ``p_typical/q_typical = 0`` — на 750 кВ ВЛ они
    обычно << зарядной мощности.

    Args:
        vn_kv: номинальное напряжение ветви, кВ (берём из узла начала).
        susceptance_si: ``branch.susceptance``, См (полная B линии,
            хранится со знаком: +у capacitive ВЛ, ‒у indictive ШР).
        reactance_si: ``branch.reactance``, Ом.
        p_typical_mw: типичный P-поток через ветвь, МВт (если знаем —
            добавим Q-потери X-серии; default 0 = только charging).
        q_typical_mvar: типичный Q-поток.

    Returns:
        |Σ Q_beg + Q_end| в МВар (положительное число).

    Notes:
        Единицы: V [кВ], B [См] → V²·B имеет размерность (кВ)² · См =
        (1000 В)² · См = 10⁶ ВА = МВА. Поэтому V²·B напрямую в МВар.
        Аналогично (P²+Q²)/V²·X = (МВА)²/(кВ)² · Ом = МА² · Ом = МВт²/В.
        Конвертация: 1 МА = 10⁶ А; 1 МА² · Ом = 10¹² ВА = 10⁶ МВар.
        Корректировка: (P²+Q²)/V²·X где P,Q в МВт/МВар, V в кВ →
        результат в МВар напрямую (через единицы входного формата S = VI·1e-3).
    """
    charging_mvar = vn_kv * vn_kv * abs(susceptance_si)
    if vn_kv > 0 and abs(reactance_si) > 1e-12:
        s2 = p_typical_mw * p_typical_mw + q_typical_mvar * q_typical_mvar
        x_loss_mvar = s2 / (vn_kv * vn_kv) * abs(reactance_si)
    else:
        x_loss_mvar = 0.0
    return abs(charging_mvar - x_loss_mvar)


def is_branch_q_consistent_with_physics(
    *,
    q_observed_mvar: float,
    vn_kv: float,
    susceptance_si: float,
    reactance_si: float,
    p_typical_mw: float = 0.0,
    floor_mvar: float = 50.0,
    rel_pct: float = 30.0,
) -> tuple[bool, float, float, float]:
    """Sanity-check: совпадает ли |Q_beg+Q_end| с физикой π-схемы.

    Расширение flat-threshold ``|Q_beg+Q_end|>const``: использует
    расчётное ожидание через зарядную B и потери в X (V_flat=V_nom).

    Outlier-логика: ``excess = ||Q_obs| - |Q_expected||``; tolerance
    ``tol = max(floor_mvar, rel_pct/100 · |Q_expected|)``. Outlier если
    ``excess > tol``.

    Use case на 750 кВ ВЛ с большой зарядной B:
    ``|Q_obs|≈1100, |Q_exp|=V²·B=1066`` → excess=34 < tol≈max(50, 320)
    → **keep** (flat ``q>50 МВар`` ложно дропал).

    Use case на 220 кВ ВЛ с малой B:
    ``|Q_obs|=80, |Q_exp|=5`` → excess=75 > tol=max(50, 1.5) → **drop**
    (flat ``q>100 МВар`` пропускал).

    Returns:
        (is_consistent, expected_mvar, observed_mvar, excess_mvar)
    """
    q_obs_abs = abs(q_observed_mvar)
    q_exp_abs = compute_expected_q_imbalance_mvar(
        vn_kv=vn_kv,
        susceptance_si=susceptance_si,
        reactance_si=reactance_si,
        p_typical_mw=p_typical_mw,
        q_typical_mvar=q_observed_mvar / 2.0,
    )
    excess = abs(q_obs_abs - q_exp_abs)
    tol = max(floor_mvar, rel_pct / 100.0 * q_exp_abs)
    return (excess <= tol, q_exp_abs, q_obs_abs, excess)
