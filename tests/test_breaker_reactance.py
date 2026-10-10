"""Tests for `normalize_breaker_reactance`.

The loader replaces "short-circuit" branches (R=X=0: bus sections, breakers,
tie links) with R=0, X=1.0 Ohm, FIXED in Ohms. After model_to_pu this
gives an X_pu that depends on the voltage class: at 10.5 kV X_pu≈0.9 (a disaster
for unit generator buses), at 500 kV X_pu≈4e-5 (fine). `normalize_breaker_reactance`
brings X to a voltage-aware value X_pu=eps_pu on every class.
"""

from __future__ import annotations

import pytest

from gridstate.telemetry import normalize_breaker_reactance
from gridstate.telemetry.shunts import BREAKER_X_SENTINEL_OHM
from gridstate.units import BASE_MVA


def _toy_model_with_breakers():
    from gridstate.constants import BranchType, NodeType
    from gridstate.working import Working

    m = Working.empty()
    for nid, vn in ((1, 10.5), (2, 10.5), (3, 500.0), (4, 500.0), (5, 0.0), (6, 0.0), (7, 0.0)):
        m.nodes.add(
            {
                "id": nid,
                "name": f"N{nid}",
                "voltage_nominal": vn,
                "status": True,
                "node_type": int(NodeType.PQ),
            }
        )

    def _br(bid, frm, to, r, x):
        m.branches.add(
            {
                "id": bid,
                "name": f"B{bid}",
                "from_node": frm,
                "to_node": to,
                "resistance": r,
                "reactance": x,
                "parallel_id": 1,
                "tap_ratio": 1.0,
                "status": True,
                "branch_type": int(BranchType.LINE),
            }
        )

    S = BREAKER_X_SENTINEL_OHM
    _br(100, 1, 2, 0.0, S)  # LV sentinel -> normalised
    _br(200, 3, 4, 0.0, S)  # HV sentinel -> normalised
    _br(300, 2, 3, 6.05, 30.25)  # real overhead line -> NOT touched (R>0)
    _br(400, 5, 6, 0.0, S)  # sentinel, but Vn unavailable -> skipped
    _br(500, 7, 3, 0.0, S)  # from Vn=0 -> fallback to the to-node (500 kV)
    return m


def test_normalize_breaker_reactance_synthetic():
    """Sentinels R=0,X=1.0 -> X_pu=eps_pu on every class; everything else is left alone."""
    m = _toy_model_with_breakers()
    eps = 1e-3

    stats = normalize_breaker_reactance(m, eps_pu=eps)

    # Normalised: B100 (LV), B200 (HV), B500 (fallback to the to-node). B400 is skipped
    # (no Vn), B300 is a real branch (R>0).
    assert stats["normalized"] == 3, stats
    assert stats["eps_pu"] == eps

    by_id = {int(b["id"]): b for b in m.branches.to_numpy()}

    # LV sentinel: X = eps·(10.5²/100); X_pu = X/(Vn²/Sbase) == eps.
    x100 = float(by_id[100]["reactance"])
    assert x100 == pytest.approx(eps * (10.5**2 / BASE_MVA))
    assert x100 / (10.5**2 / BASE_MVA) == pytest.approx(eps)

    # HV sentinel.
    x200 = float(by_id[200]["reactance"])
    assert x200 == pytest.approx(eps * (500.0**2 / BASE_MVA))
    assert x200 / (500.0**2 / BASE_MVA) == pytest.approx(eps)

    # fallback to the to-node (500 kV).
    x500 = float(by_id[500]["reactance"])
    assert x500 == pytest.approx(eps * (500.0**2 / BASE_MVA))

    # The real overhead line (R>0) is untouched.
    assert float(by_id[300]["resistance"]) == pytest.approx(6.05)
    assert float(by_id[300]["reactance"]) == pytest.approx(30.25)

    # Sentinel without Vn is left as is.
    assert float(by_id[400]["reactance"]) == pytest.approx(BREAKER_X_SENTINEL_OHM)

    # R=0 is preserved on the normalised branches.
    assert float(by_id[100]["resistance"]) == 0.0
    assert float(by_id[200]["resistance"]) == 0.0


def test_normalize_breaker_reactance_idempotent():
    """A repeated call is a no-op (after normalisation X≠1.0 Ohm, so the sentinel no longer matches)."""
    m = _toy_model_with_breakers()
    normalize_breaker_reactance(m)
    stats2 = normalize_breaker_reactance(m)
    assert stats2["normalized"] == 0, stats2


def test_normalize_breaker_reactance_no_branches():
    """Model without branches -> 0 normalisations, no crash."""
    from gridstate.working import Working

    m = Working.empty()
    stats = normalize_breaker_reactance(m)
    assert stats["normalized"] == 0
