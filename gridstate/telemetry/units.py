"""Утилиты единиц измерения и дисперсий для телеметрии.

Перенесены из ``tests/_ti_loader.py`` для использования из gridstate.telemetry
без зависимости на ``tests/`` пакет.
"""

from __future__ import annotations


def normalize_guid(g: str) -> str:
    """Нормализация GUID к lower-case, без фигурных скобок."""
    return g.strip("{}").lower()


_VARIANCE_V_FRAC = 0.01  # 1% от номинала напряжения
_VARIANCE_P_BASE = 1.0  # МВт² floor
_VARIANCE_Q_BASE = 1.0  # МВАр² floor
_VARIANCE_P_FRAC = 0.02  # 2% от значения мощности
_VARIANCE_Q_FRAC = 0.02


def variance_voltage(value_kv: float, vn_kv: float) -> float:
    """σ²_V в kV²: 1% от номинала, минимум 0.1 кВ.

    Не зависит от текущего value (variance прайор по nominal).

    ``value_kv`` is ignored; it stays in the signature for existing positional
    callers.
    """
    sigma = max(_VARIANCE_V_FRAC * vn_kv, 0.1)
    return float(sigma * sigma)


def variance_power(
    value_mva: float,
    base: float = _VARIANCE_P_BASE,
    sigma_frac: float = _VARIANCE_P_FRAC,
) -> float:
    """σ² в МВт²: ``sigma_frac`` от |value|, минимум 0.5 МВт + base floor.

    Для P-измерений default ``sigma_frac=0.02`` (2 % от |value|).
    For Q branch-flow measurements on large models a larger value
    (0.05–0.07) is sometimes justified: the Q flow depends strongly on V and
    δ, and a tight σ_Q forces the estimate to bend the voltage of a whole
    area to fit broken or inconsistent Q telemetry.
    """
    sigma = max(sigma_frac * abs(value_mva), 0.5)
    return float(sigma * sigma + base)


def variance_branch_q(
    value_mvar: float,
    *,
    charging_mvar: float = 0.0,
    charging_alpha: float = 0.0,
    base: float = _VARIANCE_Q_BASE,
    sigma_frac: float = _VARIANCE_Q_FRAC,
) -> float:
    """σ² для branch-Q с **charging-aware** полом (единый источник истины).

    На длинных HV/EHV-ВЛ модельный branch-Q доминирован зарядной мощностью
    π-схемы ``Q_charge ≈ V²·B`` (на 750 кВ — сотни-тысячи МВАр), тогда как
    измеренный нетто-Q конца линии мал. Плоская ``σ_frac·|Q_meas|`` (1–3 МВАр
    при малом Q) выдаёт нормированный резидуал в десятки-сотни σ и тянет V
    региона вниз через нормальные уравнения. Поэтому вводим минимальный
    абсолютный пол ``σ_charge = charging_alpha·|charging_mvar|``::

        σ² = max( variance_power(value, sigma_frac), (charging_alpha·|charging_mvar|)² )

    где ``charging_mvar = |B_si|·Vn²`` (полная зарядная, см.
    :func:`gridstate.telemetry.loss_filter.compute_expected_q_imbalance_mvar`).
    Mirrors the inline logic of ``apply_telemetry``
    (``gridstate.telemetry.apply_resolved``) for callers that build measurement
    variances themselves.

    Args:
        value_mvar: измеренный Q-поток ветви, МВАр.
        charging_mvar: ``|B_si|·Vn²`` ветви, МВАр (полная зарядная). 0 = нет.
        charging_alpha: доля зарядной, идущая в σ-пол. 0 = выкл (= ``variance_power``).
        base: floor МВАр².
        sigma_frac: относительная σ (default региональной модели 0.07).

    Returns:
        σ² в МВАр².
    """
    var_frac = variance_power(value_mvar, base=base, sigma_frac=sigma_frac)
    if charging_alpha > 0.0 and charging_mvar != 0.0:
        sigma_charge = charging_alpha * abs(charging_mvar)
        return max(var_frac, float(sigma_charge * sigma_charge))
    return var_frac
