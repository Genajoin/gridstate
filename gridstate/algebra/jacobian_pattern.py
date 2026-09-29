"""Measurement Jacobian on a fixed sparsity pattern.

The pattern of ``H`` depends only on the topology (``Ybus``/``Yf``/``Yt``),
the measurement index and the state layout, all of which are fixed for a
:class:`~gridstate.algebra.base.BaseAlgebra` instance. :class:`JacobianPattern`
builds it once; each evaluation then fills the ``nnz`` values with vectorised
numpy operations instead of ~180 sparse products and constructions.

The values are bit-identical to the generic sparse path of
``BaseAlgebra.evaluate_jacobian``: every product follows the same order of
operations, and complex products use the textbook formula without fused
multiply-add (:func:`_cmul`), which is how scipy's sparse kernels compute them.
numpy's own complex multiply may use FMA on x86-64 and differ in the last bit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix

from gridstate.z_vector import (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_QNAG,
    KIND_CURRENT,
    KIND_NODE_BALANCE_P,
    KIND_NODE_BALANCE_Q,
    KIND_POWER_INJECTION_P,
    KIND_POWER_INJECTION_Q,
    KIND_POWER_P,
    KIND_POWER_Q,
    KIND_VOLTAGE,
    OBJ_BRANCH,
    OBJ_NODE,
    SIDE_FROM,
    SIDE_TO,
)


if TYPE_CHECKING:
    from gridstate.algebra.base import BaseAlgebra


# Families of derivative matrices a measurement row is taken from.
_BUS, _FROM, _TO, _I_FROM, _I_TO = range(5)
# Part of the state a derivative refers to.
_VA, _VM = 0, 1
# Component of the complex derivative stored in H.
_RE, _IM = 0, 1

_NODE_KINDS_WITHOUT_V = (
    KIND_BOX_PRIOR_PGEN,
    KIND_BOX_PRIOR_QGEN,
    KIND_BOX_PRIOR_PNAG,
    KIND_BOX_PRIOR_QNAG,
)


class UnsupportedMeasurementError(ValueError):
    """The measurement index holds a combination the pattern does not cover."""


def _cmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Complex product without fused multiply-add, as in scipy's sparse kernels."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    out = np.empty(np.broadcast(a, b).shape, dtype=np.complex128)
    ar, ai, br, bi = a.real, a.imag, b.real, b.imag
    out.real = ar * br - ai * bi
    out.imag = ar * bi + ai * br
    return out


class _Family:
    """CSR pattern of ``Y ∪ {(extra_rows[k], extra_cols[k])}``.

    For every position keeps the index into ``Y.data`` (``-1`` when the
    position comes only from the extra set) and whether it belongs to the
    extra set (the diagonal of ``Ybus`` or the ``(l, s_l)`` selector of a
    branch end).
    """

    def __init__(self, Y: csr_matrix, extra_rows: np.ndarray, extra_cols: np.ndarray) -> None:
        Y = csr_matrix(Y, copy=True)
        Y.sum_duplicates()
        Y.sort_indices()
        n_rows, n_cols = Y.shape
        y_rows = np.repeat(np.arange(n_rows, dtype=np.int64), np.diff(Y.indptr))
        keys_y = y_rows * n_cols + Y.indices.astype(np.int64)
        keys_e = extra_rows.astype(np.int64) * n_cols + extra_cols.astype(np.int64)
        keys = np.union1d(keys_y, keys_e)
        self.rows = keys // n_cols
        self.cols = keys % n_cols
        self.y_idx = np.full(keys.size, -1, dtype=np.int64)
        self.y_idx[np.searchsorted(keys, keys_y)] = np.arange(keys_y.size)
        self.is_extra = np.zeros(keys.size, dtype=bool)
        self.is_extra[np.searchsorted(keys, keys_e)] = True
        self.has_y = self.y_idx >= 0
        self.y = np.where(self.has_y, Y.data[np.maximum(self.y_idx, 0)], 0)
        indptr = np.zeros(n_rows + 1, dtype=np.int64)
        np.add.at(indptr, self.rows + 1, 1)
        self.indptr = np.cumsum(indptr)
        self.size = keys.size


class JacobianPattern:
    """Fixed-pattern evaluator of ``BaseAlgebra.evaluate_jacobian``.

    Raises:
        UnsupportedMeasurementError: the measurement index has a
            ``(kind, object_kind, branch_side)`` combination not covered here;
            the caller falls back to the generic path.
    """

    def __init__(self, alg: BaseAlgebra) -> None:
        self._alg = alg
        n = alg.n_bus
        mi = alg.meas_index
        kind, obj, pos, side = mi.kind, mi.object_kind, mi.object_pos, mi.branch_side
        m = len(mi)
        node = obj == OBJ_NODE
        branch = obj == OBJ_BRANCH

        handled = np.zeros(m, dtype=bool)
        for k in (
            KIND_POWER_INJECTION_P,
            KIND_POWER_INJECTION_Q,
            KIND_NODE_BALANCE_P,
            KIND_NODE_BALANCE_Q,
            KIND_VOLTAGE,
            *_NODE_KINDS_WITHOUT_V,
        ):
            handled |= (kind == k) & node
        if alg.n_branch > 0:
            for k in (KIND_POWER_P, KIND_POWER_Q, KIND_CURRENT):
                handled |= (kind == k) & branch & ((side == SIDE_FROM) | (side == SIDE_TO))
        if not handled.all():
            raise UnsupportedMeasurementError("unsupported measurement combination")

        buses = np.arange(n)
        self._families: dict[int, _Family] = {_BUS: _Family(alg.ybus, buses, buses)}
        self._need_current = bool(np.any((kind == KIND_CURRENT) & branch))
        if alg.n_branch > 0:
            lines = np.arange(alg.n_branch)
            self._families[_FROM] = _Family(alg.yf, lines, np.asarray(alg.from_idx))
            self._families[_TO] = _Family(alg.yt, lines, np.asarray(alg.to_idx))
            if self._need_current:
                none = np.empty(0, dtype=np.int64)
                self._families[_I_FROM] = _Family(alg.yf, none, none)
                self._families[_I_TO] = _Family(alg.yt, none, none)

        non_slack = np.asarray(alg.layout.non_slack_idx, dtype=np.int64)
        va_col = np.full(n, -1, dtype=np.int64)
        va_col[non_slack] = np.arange(non_slack.size)
        n_va = non_slack.size

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        sources: list[np.ndarray] = []  # (family, part, component, position in family)

        def add(mask: np.ndarray, family: int, component: int) -> None:
            rows_h = np.where(mask)[0]
            if rows_h.size == 0:
                return
            fam = self._families[family]
            at = pos[rows_h].astype(np.int64)
            count = fam.indptr[at + 1] - fam.indptr[at]
            r = np.repeat(rows_h, count)
            starts = np.repeat(fam.indptr[at], count)
            k = starts + np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
            c = fam.cols[k]
            keep = va_col[c] >= 0
            nk = int(keep.sum())
            rows.append(r[keep])
            cols.append(va_col[c[keep]])
            sources.append(
                np.stack(
                    [np.full(nk, family), np.full(nk, _VA), np.full(nk, component), k[keep]], 1
                )
            )
            rows.append(r)
            cols.append(n_va + c)
            sources.append(
                np.stack(
                    [np.full(r.size, family), np.full(r.size, _VM), np.full(r.size, component), k],
                    1,
                )
            )

        add((kind == KIND_POWER_INJECTION_P) & node, _BUS, _RE)
        add((kind == KIND_POWER_INJECTION_Q) & node, _BUS, _IM)
        add((kind == KIND_NODE_BALANCE_P) & node, _BUS, _RE)
        add((kind == KIND_NODE_BALANCE_Q) & node, _BUS, _IM)
        if alg.n_branch > 0:
            add((kind == KIND_POWER_P) & branch & (side == SIDE_FROM), _FROM, _RE)
            add((kind == KIND_POWER_P) & branch & (side == SIDE_TO), _TO, _RE)
            add((kind == KIND_POWER_Q) & branch & (side == SIDE_FROM), _FROM, _IM)
            add((kind == KIND_POWER_Q) & branch & (side == SIDE_TO), _TO, _IM)
            if self._need_current:
                add((kind == KIND_CURRENT) & branch & (side == SIDE_FROM), _I_FROM, _RE)
                add((kind == KIND_CURRENT) & branch & (side == SIDE_TO), _I_TO, _RE)

        # Constant entries: |V| rows and the box-variable columns.
        v_rows = np.where((kind == KIND_VOLTAGE) & node)[0]
        const_rows = [v_rows]
        const_cols = [n_va + pos[v_rows].astype(np.int64)]
        const_vals = [np.ones(v_rows.size)]
        n_cols = 2 * n - 1
        if alg.layout.has_box:
            n_box = alg.layout.n_box
            box = alg._build_balance_jacobian_block(m, n_box).tocoo()
            const_rows.append(box.row.astype(np.int64))
            const_cols.append(n_cols + box.col.astype(np.int64))
            const_vals.append(box.data)
            n_cols += n_box
        self.shape = (m, n_cols)

        var_rows = np.concatenate(rows) if rows else np.empty(0, dtype=np.int64)
        var_cols = np.concatenate(cols) if cols else np.empty(0, dtype=np.int64)
        src = np.concatenate(sources) if sources else np.empty((0, 4), dtype=np.int64)
        all_rows = np.concatenate([var_rows, *const_rows])
        all_cols = np.concatenate([var_cols, *const_cols])
        order = np.lexsort((all_cols, all_rows))
        slot = np.empty_like(order)
        slot[order] = np.arange(order.size)
        self._indices = all_cols[order].astype(np.int32)
        self._indptr = np.concatenate([[0], np.cumsum(np.bincount(all_rows, minlength=m))]).astype(
            np.int32
        )
        self._nnz = order.size
        self._const_slot = slot[var_rows.size :]
        self._const_vals = np.concatenate(const_vals)
        var_slot = slot[: var_rows.size]
        self._groups: list[tuple[int, int, int, np.ndarray, np.ndarray]] = []
        for key in np.unique(src[:, :3], axis=0):
            sel = np.all(src[:, :3] == key, axis=1)
            family, part, component = (int(x) for x in key)
            self._groups.append((family, part, component, var_slot[sel], src[sel, 3]))

    def evaluate(self, v: np.ndarray, delta: np.ndarray) -> csr_matrix:
        """``H(V, δ)`` with the same values as the generic sparse path."""
        alg = self._alg
        V = v * np.exp(1j * delta)
        Vn = V / np.abs(V)
        derivs: dict[int, tuple[np.ndarray, np.ndarray]] = {_BUS: self._bus(V, Vn)}
        if alg.n_branch > 0:
            derivs[_FROM] = self._branch_end(V, Vn, self._families[_FROM], alg.yf, alg.from_idx)
            derivs[_TO] = self._branch_end(V, Vn, self._families[_TO], alg.yt, alg.to_idx)
            if self._need_current:
                derivs[_I_FROM] = self._current(V, Vn, self._families[_I_FROM], alg.yf)
                derivs[_I_TO] = self._current(V, Vn, self._families[_I_TO], alg.yt)

        data = np.empty(self._nnz, dtype=np.float64)
        data[self._const_slot] = self._const_vals
        for family, part, component, slots, k in self._groups:
            values = derivs[family][part][k]
            data[slots] = values.imag if component == _IM else values.real
        return csr_matrix((data, self._indices, self._indptr), shape=self.shape)

    def _bus(self, V: np.ndarray, Vn: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``∂S_bus/∂δ`` and ``∂S_bus/∂|V|`` on the pattern (see ``_dSbus_dV``)."""
        f = self._families[_BUS]
        Ibus = self._alg.ybus @ V
        r, c, ex, hy, y = f.rows, f.cols, f.is_extra, f.has_y, f.y
        # dVm = V_i·conj(y_ij·Vn_j) + [i = j] conj(I_i)·Vn_i
        off = np.zeros(f.size, dtype=np.complex128)
        off[hy] = _cmul(V[r[hy]], np.conj(_cmul(y[hy], Vn[c[hy]])))
        diag = _cmul(np.conj(Ibus[r[ex]]), Vn[r[ex]])
        d_vm = off.copy()
        d_vm[ex] = np.where(hy[ex], off[ex] + diag, diag)
        # dVa = (j·V_i)·conj([i = j] I_i − y_ij·V_j)
        yv = np.zeros(f.size, dtype=np.complex128)
        yv[hy] = _cmul(y[hy], V[c[hy]])
        inner = np.where(hy, -yv, 0)
        inner[ex] = np.where(hy[ex], Ibus[r[ex]] - yv[ex], Ibus[r[ex]])
        jV = 1j * V.astype(np.complex128)
        d_va = _cmul(jV[r], np.conj(inner))
        return d_va, d_vm

    @staticmethod
    def _branch_end(
        V: np.ndarray, Vn: np.ndarray, f: _Family, Y: csr_matrix, end: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``∂S_branch/∂δ`` and ``∂S_branch/∂|V|`` of one end (see ``_dSbr_dV``).

        ``end[l]`` is the node of branch ``l`` at this end; the extra positions
        of the family are ``(l, end[l])``.
        """
        I_br = Y @ V
        r, c, ex, hy, y = f.rows, f.cols, f.is_extra, f.has_y, f.y
        Vs = V[np.asarray(end)][r]
        # dVa = j·(conj(I_l)·V_s[(l, s_l)] − V_s·conj(y·V_j))
        yv = np.zeros(f.size, dtype=np.complex128)
        yv[hy] = _cmul(Vs[hy], np.conj(_cmul(y[hy], V[c[hy]])))
        sel = _cmul(np.conj(I_br[r[ex]]), V[c[ex]])
        diff = np.where(hy, -yv, 0)
        diff[ex] = np.where(hy[ex], sel - yv[ex], sel)
        d_va = 1j * diff
        # dVm = V_s·conj(y·Vn_j) + [(l, s_l)] conj(I_l)·Vn_s
        yvn = np.zeros(f.size, dtype=np.complex128)
        yvn[hy] = _cmul(Vs[hy], np.conj(_cmul(y[hy], Vn[c[hy]])))
        sel_n = _cmul(np.conj(I_br[r[ex]]), Vn[c[ex]])
        d_vm = yvn.copy()
        d_vm[ex] = np.where(hy[ex], yvn[ex] + sel_n, sel_n)
        return d_va, d_vm

    @staticmethod
    def _current(
        V: np.ndarray, Vn: np.ndarray, f: _Family, Y: csr_matrix
    ) -> tuple[np.ndarray, np.ndarray]:
        """``∂|I_branch|/∂δ`` and ``∂|I_branch|/∂|V|`` (see ``_dImbr_dV``)."""
        I_br = Y @ V
        abs_I = np.abs(I_br)
        norm = np.where(abs_I > 0, np.conj(I_br) / np.where(abs_I > 0, abs_I, 1.0), 0.0 + 0j)
        r, c = f.rows, f.cols
        iy = _cmul(norm[r], f.y)
        a = _cmul(iy, V[c])
        b = _cmul(iy, Vn[c])
        return -(a.imag) + 0j, b.real + 0j
