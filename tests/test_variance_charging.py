"""Тесты charging-aware σ_Q (:func:`gridstate.telemetry.noise.variance_branch_q`).

Единый источник истины для branch-Q дисперсии: σ² = max(σ²_frac, (α·|B|·Vn²)²).
``apply_telemetry`` uses it, and so can callers that build measurement variances
themselves. A parameter sweep confirmed α=0.10 as the best value on four large
production models; this test pins the semantics of the formula.
"""

from __future__ import annotations

import math

import pytest

from gridstate.telemetry.noise import variance_branch_q, variance_power


def test_alpha_zero_equals_variance_power() -> None:
    """α=0 → ровно ``variance_power`` (нет charging-пола)."""
    for q in (5.0, 50.0, 500.0):
        assert variance_branch_q(
            q, charging_mvar=300.0, charging_alpha=0.0, sigma_frac=0.07
        ) == pytest.approx(variance_power(q, sigma_frac=0.07))


def test_zero_charging_equals_variance_power() -> None:
    """charging_mvar=0 (нет B) → ``variance_power`` даже при α>0."""
    assert variance_branch_q(
        20.0, charging_mvar=0.0, charging_alpha=1.0, sigma_frac=0.07
    ) == pytest.approx(variance_power(20.0, sigma_frac=0.07))


def test_charging_floor_dominates_small_q() -> None:
    """При малом Q charging-пол доминирует: σ ≈ α·|charging|.

    Q=20 МВАр, frac=0.07 → σ_frac≈1.7; charging=125, α=0.10 → σ_charge=12.5.
    max ⇒ σ=12.5.
    """
    var = variance_branch_q(20.0, charging_mvar=125.0, charging_alpha=0.10, sigma_frac=0.07)
    assert math.sqrt(var) == pytest.approx(12.5, abs=0.05)


def test_frac_dominates_large_q() -> None:
    """При большом измеренном Q доминирует σ_frac (charging-пол ниже).

    Q=1000, frac=0.07 → σ_frac=70; charging=125, α=0.10 → σ_charge=12.5 < 70.
    max ⇒ σ≈70 (== variance_power).
    """
    var = variance_branch_q(1000.0, charging_mvar=125.0, charging_alpha=0.10, sigma_frac=0.07)
    assert var == pytest.approx(variance_power(1000.0, sigma_frac=0.07))
    assert math.sqrt(var) == pytest.approx(70.0, abs=0.1)


def test_alpha_scales_floor_linearly() -> None:
    """σ_charge = α·|charging| линеен по α (при доминировании пола)."""
    ch = 200.0
    s010 = math.sqrt(variance_branch_q(10.0, charging_mvar=ch, charging_alpha=0.10))
    s050 = math.sqrt(variance_branch_q(10.0, charging_mvar=ch, charging_alpha=0.50))
    s100 = math.sqrt(variance_branch_q(10.0, charging_mvar=ch, charging_alpha=1.00))
    assert s010 == pytest.approx(20.0, abs=0.1)
    assert s050 == pytest.approx(100.0, abs=0.1)
    assert s100 == pytest.approx(200.0, abs=0.1)


def test_charging_mvar_sign_irrelevant() -> None:
    """Знак charging_mvar не важен (берётся |·|): ШР (B<0) и БК (B>0) симметричны."""
    pos = variance_branch_q(10.0, charging_mvar=150.0, charging_alpha=0.5)
    neg = variance_branch_q(10.0, charging_mvar=-150.0, charging_alpha=0.5)
    assert pos == pytest.approx(neg)
