"""Построение матриц проводимостей в p.u. для SE.

``Ybus`` — комплексная разреженная матрица узловых проводимостей размерности
``n_bus × n_bus``; ``Yf`` / ``Yt`` — матрицы (n_branch × n_bus), такие что
``Ifrom = Yf · V`` и ``Ito = Yt · V`` (всё в p.u.).

Формулы каждой ветви (``from`` → ``to``) с приведением к стороне «от»:

.. code::

    Ysf = 1 / (r + j·x)                # последовательная проводимость, p.u.
    Yc_from = (g_from + j·b_from) + (g + j·b)/2     # суммарный шунт «от»
    Yc_to   = (g_to   + j·b_to  ) + (g + j·b)/2     # суммарный шунт «до»
    t       = tap_ratio · exp(j·phase_shift)        # комплексный коэф.
    Ys      = Ysf · |t|²               # см. ниже про сторону сопротивления

    Yff = (Ys + Yc_from) / (t · conj(t))  = Ysf + Yc_from / |t|²
    Yft = − Ys / conj(t)                  = − Ysf · t
    Ytf = − Ys / t                        = − Ysf · conj(t)
    Ytt = Ys + Yc_to                      = Ysf · |t|² + Yc_to

Сторона сопротивления. Во входном формате ``resistance``/``reactance``
трансформатора — Омы на стороне «от»: сопротивление включено у начала ветви,
идеальный трансформатор — у конца. Формулы выше записаны для сопротивления,
стоящего за идеальным трансформатором (вид ``makeYbus``), поэтому в них
подставлена проводимость, приведённая к стороне «до» фактическим (а не
номинальным) коэффициентом: ``Ys = Ysf·|t|²``. Приведение номинальным
коэффициентом искажало бы эффективное сопротивление в ``1/|t|²`` раз
(``t`` — коэффициент в p.u., см. ``units.model_to_pu``), причём ошибка
менялась бы вместе с отпайкой РПН. У линий и трансформаторов с номинальным
коэффициентом ``|t| = 1`` приведения нет: ``Ys = Ysf``.

Шунт «от» делится на ``|t|²`` (его зависимость от отпайки
учитывает пересчёт шунта при применении РПН,
``telemetry.rpn._apply_tap_steps_on_arrays``).

Шунты узлов добавляются на диагональ ``Ybus``; параллельные ветви
суммируются автоматически через ``coo_matrix``-сборку с одинаковыми
индексами.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


if TYPE_CHECKING:
    from gridstate.units import NetworkPU


def series_admittance(network_pu: NetworkPU) -> np.ndarray:
    """Последовательная проводимость ветвей за идеальным трансформатором, p.u.

    ``Ys = |t|² / (r + j·x)``: сопротивление задано на стороне «от» и
    приводится к стороне «до» фактическим коэффициентом ``|t| = tap_ratio``
    (p.u.). У ветвей с ``|t| = 1`` — просто ``1 / (r + j·x)``.
    """
    z = network_pu.branch_r + 1j * network_pu.branch_x
    return cast("np.ndarray", network_pu.tap_ratio**2 / z)


def build_ybus(
    network_pu: NetworkPU,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Построить ``(Ybus, Yf, Yt)`` в p.u.

    Args:
        network_pu: p.u.-представление сети.

    Returns:
        (Ybus, Yf, Yt) — три CSR-матрицы с complex128:
            - ``Ybus``: ``(n_bus × n_bus)``;
            - ``Yf``:   ``(n_branch × n_bus)``;
            - ``Yt``:   ``(n_branch × n_bus)``.

    Raises:
        ValueError: если у ветви ``r = x = 0`` (сингулярность последовательной
            проводимости — несоединённая ветвь должна быть исключена ещё в
            ``model_to_pu`` через ``status=False``).
    """
    n_bus = network_pu.n_bus
    n_branch = network_pu.n_branch

    if n_branch == 0:
        # Только узловые шунты.
        ysh = network_pu.bus_g_shunt + 1j * network_pu.bus_b_shunt
        ybus = csr_matrix(
            (ysh.astype(np.complex128), (np.arange(n_bus), np.arange(n_bus))),
            shape=(n_bus, n_bus),
            dtype=np.complex128,
        )
        empty = csr_matrix((0, n_bus), dtype=np.complex128)
        return cast("csr_matrix", ybus), cast("csr_matrix", empty), cast("csr_matrix", empty)

    # ---- Параметры ветвей ----
    z = network_pu.branch_r + 1j * network_pu.branch_x
    if np.any(z == 0):
        bad = network_pu.branch_ids[z == 0].tolist()
        raise ValueError(
            f"Ветви с нулевым импедансом (R=X=0) недопустимы: branch_ids={bad}. "
            "Исключите их через status=False или замените малым R/X."
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
    # Сопротивление задано на стороне «от»: приводим его за идеальный
    # трансформатор фактическим коэффициентом, см. docstring модуля.
    ys = series_admittance(network_pu)

    yff = (ys + yc_from) / (tap * np.conj(tap))
    yft = -ys / np.conj(tap)
    ytf = -ys / tap
    ytt = ys + yc_to

    f = network_pu.from_idx
    t = network_pu.to_idx
    rng = np.arange(n_branch, dtype=np.int64)

    # Yf: (n_branch × n_bus). Row k имеет Yff[k] в столбце from_idx[k] и
    # Yft[k] в столбце to_idx[k].
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
    # Здесь компактнее напрямую через COO: для каждой ветви добавляем 4 элемента,
    # параллельные ветви и встречные пары суммируются sum_duplicates'ом.
    ybus_rows = np.concatenate([f, f, t, t])
    ybus_cols = np.concatenate([f, t, f, t])
    ybus_vals = np.concatenate([yff, yft, ytf, ytt])

    # Узловые шунты — на диагональ.
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
