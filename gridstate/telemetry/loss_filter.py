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
    """Expected ``|Q_beg + Q_end|`` for a line from π-model physics.

    On an in-service line (no transformer) the sum of the Q flows at both ends =
    Q losses in the reactance minus the Q generation of the charging B::

        Q_beg + Q_end = (P²+Q²)/V²·X  -  V²·B

    where B is the full charging susceptance (stored in our data model as
    ``branch.susceptance``, in Siemens) and is split in half into the shunts
    of both π-model ends (V²·B/2 + V²·B/2 = V²·B).

    For a rough estimate we use ``Vnom`` instead of the actual V
    (on a line V is usually close to nominal ±5 %). The Q losses in X can also
    be omitted if ``p_typical/q_typical = 0``: on 750 kV lines they are
    usually << the charging power.

    Args:
        vn_kv: nominal branch voltage, kV (taken from the from-node).
        susceptance_si: ``branch.susceptance``, S (full B of the line,
            stored with sign: positive for capacitive lines, negative for inductive shunt reactors).
        reactance_si: ``branch.reactance``, Ohm.
        p_typical_mw: typical P flow through the branch, MW (if known,
            adds the series-X Q losses; default 0 = charging only).
        q_typical_mvar: typical Q flow.

    Returns:
        |Σ Q_beg + Q_end| in MVAr (a positive number).

    Notes:
        Units: V [kV], B [S] → V²·B has dimension (kV)² · S =
        (1000 V)² · S = 10⁶ VA = MVA. Hence V²·B is directly in MVAr.
        Likewise (P²+Q²)/V²·X = (MVA)²/(kV)² · Ohm = MA² · Ohm = MW²/V.
        Conversion: 1 MA = 10⁶ A; 1 MA² · Ohm = 10¹² VA = 10⁶ MVAr.
        Correction: (P²+Q²)/V²·X with P,Q in MW/MVAr, V in kV →
        the result is directly in MVAr (through the units of the input format S = VI·1e-3).
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
    """Sanity check: does |Q_beg+Q_end| agree with the π-model physics.

    An extension of the flat threshold ``|Q_beg+Q_end|>const``: uses
    the expected value from the charging B and the losses in X (V_flat=V_nom).

    Outlier logic: ``excess = ||Q_obs| - |Q_expected||``; tolerance
    ``tol = max(floor_mvar, rel_pct/100 · |Q_expected|)``. A measurement is an outlier if
    ``excess > tol``.

    Use case on a 750 kV line with a large charging B:
    ``|Q_obs|≈1100, |Q_exp|=V²·B=1066`` → excess=34 < tol≈max(50, 320)
    → **keep** (the flat ``q>50 MVAr`` wrongly dropped it).

    Use case on a 220 kV line with a small B:
    ``|Q_obs|=80, |Q_exp|=5`` → excess=75 > tol=max(50, 1.5) → **drop**
    (the flat ``q>100 MVAr`` let it pass).

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
