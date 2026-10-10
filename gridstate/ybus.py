"""Construction of the admittance matrices in p.u. for SE.

``Ybus`` is the complex sparse nodal admittance matrix of dimension
``n_bus × n_bus``; ``Yf`` / ``Yt`` are (n_branch × n_bus) matrices such that
``Ifrom = Yf · V`` and ``Ito = Yt · V`` (all in p.u.).

Formulas of each branch (``from`` → ``to``) referred to the "from" side:

.. code::

    Ysf = 1 / (r + j·x)                # series admittance, p.u.
    Yc_from = (g_from + j·b_from) + (g + j·b)/2     # total "from" shunt
    Yc_to   = (g_to   + j·b_to  ) + (g + j·b)/2     # total "to" shunt
    t       = tap_ratio · exp(j·phase_shift)        # complex ratio
    Ys      = Ysf · |t|²               # see below on the impedance side

    Yff = (Ys + Yc_from) / (t · conj(t))  = Ysf + Yc_from / |t|²
    Yft = − Ys / conj(t)                  = − Ysf · t
    Ytf = − Ys / t                        = − Ysf · conj(t)
    Ytt = Ys + Yc_to                      = Ysf · |t|² + Yc_to

Impedance side. In the input format the ``resistance``/``reactance``
of a transformer are Ohms on the "from" side: the impedance is placed at the start of the branch,
the ideal transformer at its end. The formulas above are written for an impedance
located behind the ideal transformer (the ``makeYbus`` form), so they
use the admittance referred to the "to" side by the actual (not the
nominal) ratio: ``Ys = Ysf·|t|²``. Referring by the nominal
ratio would distort the effective impedance by a factor of ``1/|t|²``
(``t`` is the ratio in p.u., see ``units.model_to_pu``), and the error
would change together with the on-load tap changer position. For lines and transformers with the nominal
ratio ``|t| = 1`` there is no referral: ``Ys = Ysf``.

The "from" shunt is divided by ``|t|²`` (its dependence on the tap
is accounted for by the shunt recomputation when the tap changer is applied,
``telemetry.rpn._apply_tap_steps_on_arrays``).

Nodal shunts are added to the ``Ybus`` diagonal; parallel branches
are summed automatically through the ``coo_matrix`` assembly with identical
indices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


if TYPE_CHECKING:
    from gridstate.units import NetworkPU


def series_admittance(network_pu: NetworkPU) -> np.ndarray:
    """Series admittance of branches behind the ideal transformer, p.u.

    ``Ys = |t|² / (r + j·x)``: the impedance is given on the "from" side and
    is referred to the "to" side by the actual ratio ``|t| = tap_ratio``
    (p.u.). For branches with ``|t| = 1`` it is simply ``1 / (r + j·x)``.
    """
    z = network_pu.branch_r + 1j * network_pu.branch_x
    return cast("np.ndarray", network_pu.tap_ratio**2 / z)


def build_ybus(
    network_pu: NetworkPU,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Build ``(Ybus, Yf, Yt)`` in p.u.

    Args:
        network_pu: p.u. representation of the network.

    Returns:
        (Ybus, Yf, Yt) — three CSR matrices with complex128:
            - ``Ybus``: ``(n_bus × n_bus)``;
            - ``Yf``:   ``(n_branch × n_bus)``;
            - ``Yt``:   ``(n_branch × n_bus)``.

    Raises:
        ValueError: if a branch has ``r = x = 0`` (singular series
            admittance — a disconnected branch must be excluded already in
            ``model_to_pu`` via ``status=False``).
    """
    n_bus = network_pu.n_bus
    n_branch = network_pu.n_branch

    if n_branch == 0:
        # Nodal shunts only.
        ysh = network_pu.bus_g_shunt + 1j * network_pu.bus_b_shunt
        ybus = csr_matrix(
            (ysh.astype(np.complex128), (np.arange(n_bus), np.arange(n_bus))),
            shape=(n_bus, n_bus),
            dtype=np.complex128,
        )
        empty = csr_matrix((0, n_bus), dtype=np.complex128)
        return cast("csr_matrix", ybus), cast("csr_matrix", empty), cast("csr_matrix", empty)

    # ---- Branch parameters ----
    z = network_pu.branch_r + 1j * network_pu.branch_x
    if np.any(z == 0):
        bad = network_pu.branch_ids[z == 0].tolist()
        raise ValueError(
            f"Branches with zero impedance (R=X=0) are not allowed: branch_ids={bad}. "
            "Exclude them via status=False or replace them with a small R/X."
        )

    yc_from = (
        network_pu.branch_g_from
        + 1j * network_pu.branch_b_from
        + (network_pu.branch_g + 1j * network_pu.branch_b) * 0.5
    )
    yc_to = (
        network_pu.branch_g_to
        + 1j * network_pu.branch_b_to
        + (network_pu.branch_g + 1j * network_pu.branch_b) * 0.5
    )

    tap = network_pu.tap_ratio * np.exp(1j * network_pu.phase_shift)
    # The impedance is given on the "from" side: refer it behind the ideal
    # transformer by the actual ratio, see the module docstring.
    ys = series_admittance(network_pu)

    yff = (ys + yc_from) / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ytt = ys + yc_to

    f = network_pu.from_idx
    t = network_pu.to_idx
    rng = np.arange(n_branch, dtype=np.int64)

    # Yf: (n_branch × n_bus). Row k has Yff[k] in column from_idx[k] and
    # Yft[k] in column to_idx[k].
    yf = coo_matrix(
        (np.concatenate([yff, yft]), (np.concatenate([rng, rng]), np.concatenate([f, t]))),
        shape=(n_branch, n_bus),
        dtype=np.complex128,
    ).tocsr()

    yt = coo_matrix(
        (np.concatenate([ytf, ytt]), (np.concatenate([rng, rng]), np.concatenate([f, t]))),
        shape=(n_branch, n_bus),
        dtype=np.complex128,
    ).tocsr()

    # Ybus = Cf^T · Yf + Ct^T · Yt + diag(Y_shunt).
    # Here it is more compact directly via COO: for each branch we add 4 entries,
    # parallel branches and opposite pairs are summed by sum_duplicates.
    ybus_rows = np.concatenate([f, f, t, t])
    ybus_cols = np.concatenate([f, t, f, t])
    ybus_vals = np.concatenate([yff, yft, ytf, ytt])

    # Nodal shunts — onto the diagonal.
    bus_idx = np.arange(n_bus, dtype=np.int64)
    bus_ysh = (network_pu.bus_g_shunt + 1j * network_pu.bus_b_shunt).astype(np.complex128)

    ybus_rows = np.concatenate([ybus_rows, bus_idx])
    ybus_cols = np.concatenate([ybus_cols, bus_idx])
    ybus_vals = np.concatenate([ybus_vals, bus_ysh])

    ybus = coo_matrix(
        (ybus_vals, (ybus_rows, ybus_cols)),
        shape=(n_bus, n_bus),
        dtype=np.complex128,
    ).tocsr()
    ybus.sum_duplicates()
    return cast("csr_matrix", ybus), cast("csr_matrix", yf), cast("csr_matrix", yt)
