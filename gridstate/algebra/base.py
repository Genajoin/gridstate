"""Базовая алгебра SE: функция измерений ``h(E)`` и якобиан ``H = ∂h/∂E``.

Adapted from pandapower:
    pandapower/estimation/algorithm/matrix_base.py (class ``BaseAlgebra``).
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Отличия от оригинала:
    - убрана зависимость от PYPOWER-индексов ``BUS_TYPE``, ``idx_bus/brch``;
    - вместо ``ExtendedPPCI`` используются ``NetworkPU``, ``MeasurementIndex``
      и ``StateLayout``;
    - все операции в p.u., базы — в ``gridstate/units.py``;
    - порядок строк якобиана / вектора ``h`` соответствует порядку измерений
      в ``MeasurementIndex`` (то есть тот же порядок, что и в ``z``).

Основные комплексные величины (для напряжения ``V_k = V·exp(j·δ)``)::

    I_bus  = Ybus · V
    S_bus  = V ⊙ conj(I_bus)            инъекция мощности (p.u.)
    I_from = Yf · V                      ток в ветви со стороны «от»
    S_from = V[from] ⊙ conj(I_from)     переток на стороне «от»
    I_to   = Yt · V
    S_to   = V[to]   ⊙ conj(I_to)       переток на стороне «до»

Якобиан имеет размерность ``(m × (2n−1))``: первые ``n−1`` столбцов отвечают
за ``δ`` неслэк-узлов, остальные ``n`` — за ``V`` всех узлов.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack

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
    from gridstate.state import StateLayout
    from gridstate.units import NetworkPU
    from gridstate.z_vector import MeasurementIndex


def _diag(values: np.ndarray) -> csr_matrix:
    """Разреженная диагональ ``diag(values)``."""
    n = values.shape[0]
    idx = np.arange(n)
    return cast("csr_matrix", csr_matrix((values, (idx, idx)), shape=(n, n)))


def _inverse_pos_map(node_pos_arr: np.ndarray) -> np.ndarray:
    """Обратная карта ``pos → индекс в node_pos_arr`` (нет узла → ``−1``).

    ``inv[node_pos_arr[k]] = k`` для всех ``k``; остальные ячейки ``−1``.
    Дублей в ``node_pos_arr`` по построению ``StateLayout`` нет (каждый
    активный узел добавляется один раз, см. ``preprocessing/ipm_setup.py``),
    поэтому простое прямое присваивание корректно совпадает с прежним
    ``np.where(node_pos_arr == target)[0][0]`` (первое = единственное).
    Для пустой секции возвращается массив размера 1 (только ``−1``).
    """
    if node_pos_arr.size == 0:
        return np.full(1, -1, dtype=np.int64)
    size = int(node_pos_arr.max()) + 1
    inv = np.full(size, -1, dtype=np.int64)
    inv[node_pos_arr.astype(np.int64)] = np.arange(node_pos_arr.size, dtype=np.int64)
    return inv


class BaseAlgebra:
    """Объект, фиксирующий топологию (Ybus/Yf/Yt + ``MeasurementIndex`` +
    ``StateLayout``) и умеющий вычислять ``h(V, δ)`` и ``H(V, δ)`` при очередных
    значениях состояния.
    """

    def __init__(
        self,
        ybus: csr_matrix,
        yf: csr_matrix,
        yt: csr_matrix,
        meas_index: MeasurementIndex,
        layout: StateLayout,
        network_pu: NetworkPU,
    ) -> None:
        self.ybus = ybus
        self.yf = yf
        self.yt = yt
        self.meas_index = meas_index
        self.layout = layout
        self.from_idx = network_pu.from_idx
        self.to_idx = network_pu.to_idx
        self.n_bus = network_pu.n_bus
        self.n_branch = network_pu.n_branch

    # ------------------------------------------------------------------ h(E)
    def evaluate_h(
        self,
        v: np.ndarray,
        delta: np.ndarray,
        *,
        pgen_estimated: np.ndarray | None = None,
        qgen_estimated: np.ndarray | None = None,
        pnag_estimated: np.ndarray | None = None,
        qnag_estimated: np.ndarray | None = None,
    ) -> np.ndarray:
        """Прогнозы измерений ``h(E)`` — массив длины m.

        В IPM-режиме принимает дополнительные box-vars для узлов из
        ``layout.{pgen,qgen,pnag,qnag}_node_pos``. Используются для
        ``KIND_NODE_BALANCE_P/Q``-измерений::

            h_balance_P[i] = Sbus[i].real - (Pgen_est[i] - Pnag_est[i])
            h_balance_Q[i] = Sbus[i].imag - (Qgen_est[i] - Qnag_est[i])

        Узлы без соответствующей box-vars вкладываются как 0 в скобки —
        это эквивалентно zero-injection для transit-узлов.

        В WLS-режиме (все box-аргументы ``None``) поведение идентично
        прежнему.
        """
        if v.shape != (self.n_bus,) or delta.shape != (self.n_bus,):
            raise ValueError(
                f"v, delta должны быть длины n_bus={self.n_bus}; "
                f"получено v.shape={v.shape}, delta.shape={delta.shape}"
            )

        V = v * np.exp(1j * delta)
        Ibus = self.ybus @ V
        Sbus = V * np.conj(Ibus)

        if self.n_branch > 0:
            If = self.yf @ V
            It = self.yt @ V
            Sf = V[self.from_idx] * np.conj(If)
            St = V[self.to_idx] * np.conj(It)
        else:
            empty = np.zeros(0, dtype=np.complex128)
            If = It = Sf = St = empty

        kind = self.meas_index.kind
        ok = self.meas_index.object_kind
        op = self.meas_index.object_pos
        side = self.meas_index.branch_side

        h = np.zeros(len(self.meas_index), dtype=np.float64)

        m = (kind == KIND_VOLTAGE) & (ok == OBJ_NODE)
        if m.any():
            h[m] = np.abs(V[op[m]])

        m = (kind == KIND_POWER_INJECTION_P) & (ok == OBJ_NODE)
        if m.any():
            h[m] = Sbus[op[m]].real

        m = (kind == KIND_POWER_INJECTION_Q) & (ok == OBJ_NODE)
        if m.any():
            h[m] = Sbus[op[m]].imag

        m = (kind == KIND_POWER_P) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
        if m.any():
            h[m] = Sf[op[m]].real
        m = (kind == KIND_POWER_P) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
        if m.any():
            h[m] = St[op[m]].real

        m = (kind == KIND_POWER_Q) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
        if m.any():
            h[m] = Sf[op[m]].imag
        m = (kind == KIND_POWER_Q) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
        if m.any():
            h[m] = St[op[m]].imag

        m = (kind == KIND_CURRENT) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
        if m.any():
            h[m] = np.abs(If[op[m]])
        m = (kind == KIND_CURRENT) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
        if m.any():
            h[m] = np.abs(It[op[m]])

        # ----- Узловой balance P/Q (IPM-режим) -----
        # h_balance_P[i] = Sbus[i].real - (Pgen_est[i] - Pnag_est[i])
        # Box-секции пусты в WLS-режиме → balance-meas нет в meas_index.
        m = (kind == KIND_NODE_BALANCE_P) & (ok == OBJ_NODE)
        if m.any():
            box_p = self._build_box_node_contrib(
                op[m],
                pgen_estimated,
                pnag_estimated,
                self.layout.pgen_node_pos,
                self.layout.pnag_node_pos,
            )
            h[m] = Sbus[op[m]].real - box_p
        m = (kind == KIND_NODE_BALANCE_Q) & (ok == OBJ_NODE)
        if m.any():
            box_q = self._build_box_node_contrib(
                op[m],
                qgen_estimated,
                qnag_estimated,
                self.layout.qgen_node_pos,
                self.layout.qnag_node_pos,
            )
            h[m] = Sbus[op[m]].imag - box_q

        # ----- Soft-prior для box-vars (KIND_BOX_PRIOR_*) -----
        # Применяется только в IPM-режиме (есть layout с box-секциями).
        # h(x) = текущее значение box-var на узле; z несёт ЯКОРЬ prior'а
        # (init-значение переменной — материализованный load/gen в p.u.;
        # см. build_ipm_setup) → штраф на |x − z|. Раньше z был 0, и tight
        # bus-equiv prior пиннил gross-пары эквивалентов к нулю.
        if self.layout is not None and self.layout.has_box:
            for prior_kind, est_arr, node_pos_arr in (
                (KIND_BOX_PRIOR_PGEN, pgen_estimated, self.layout.pgen_node_pos),
                (KIND_BOX_PRIOR_QGEN, qgen_estimated, self.layout.qgen_node_pos),
                (KIND_BOX_PRIOR_PNAG, pnag_estimated, self.layout.pnag_node_pos),
                (KIND_BOX_PRIOR_QNAG, qnag_estimated, self.layout.qnag_node_pos),
            ):
                m = (kind == prior_kind) & (ok == OBJ_NODE)
                if not m.any() or est_arr is None or est_arr.size == 0:
                    continue
                # Обратная карта pos → индекс в node_pos_arr; нет узла → −1.
                inv = _inverse_pos_map(node_pos_arr)
                idx_meas = np.where(m)[0]
                pos_meas = op[idx_meas].astype(np.int64)
                # Защита выхода за границы inv (pos ≥ inv.size → нет в массиве).
                in_range = pos_meas < inv.size
                j = np.full(pos_meas.shape, -1, dtype=np.int64)
                j[in_range] = inv[pos_meas[in_range]]
                # j < 0 → узла нет в node_pos_arr → 0.0 (как в прежнем цикле).
                vals = np.where(j >= 0, est_arr[np.where(j >= 0, j, 0)], 0.0)
                h[idx_meas] = vals

        return h

    @staticmethod
    def _build_box_node_contrib(
        node_pos: np.ndarray,
        gen_estimated: np.ndarray | None,
        nag_estimated: np.ndarray | None,
        gen_node_pos: np.ndarray,
        nag_node_pos: np.ndarray,
    ) -> np.ndarray:
        """Вычислить ``Pgen[i] − Pnag[i]`` (или Q-аналог) для узлов из ``node_pos``.

        Поскольку box-секции хранят значения только для узлов из
        ``*_node_pos``, делаем lookup через позиционный индекс. Узлы,
        отсутствующие в ``gen/nag_node_pos``, вкладываются как 0.
        """
        out = np.zeros(node_pos.shape, dtype=np.float64)
        node_pos_i = node_pos.astype(np.int64)
        if gen_estimated is not None and gen_estimated.size and gen_node_pos.size:
            # lookup[pos] = est, NaN для узлов без box-var; дубли в gen_node_pos
            # (по построению StateLayout их нет) → последний выигрывает, как в
            # прежнем цикле присваивания.
            gen_lookup = np.full(int(gen_node_pos.max()) + 1, np.nan)
            gen_lookup[gen_node_pos.astype(np.int64)] = gen_estimated.astype(np.float64)
            mask = node_pos_i < gen_lookup.size
            vals = gen_lookup[node_pos_i[mask]]
            out[mask] += np.where(np.isnan(vals), 0.0, vals)
        if nag_estimated is not None and nag_estimated.size and nag_node_pos.size:
            nag_lookup = np.full(int(nag_node_pos.max()) + 1, np.nan)
            nag_lookup[nag_node_pos.astype(np.int64)] = nag_estimated.astype(np.float64)
            mask = node_pos_i < nag_lookup.size
            vals = nag_lookup[node_pos_i[mask]]
            out[mask] -= np.where(np.isnan(vals), 0.0, vals)
        return out

    # ----------------------------------------------------------- H = ∂h/∂E
    def evaluate_jacobian(self, v: np.ndarray, delta: np.ndarray) -> csr_matrix:
        """Якобиан h по состоянию E — ``(m × (2n−1))`` sparse."""
        if v.shape != (self.n_bus,) or delta.shape != (self.n_bus,):
            raise ValueError(
                f"v, delta должны быть длины n_bus={self.n_bus}; "
                f"получено v.shape={v.shape}, delta.shape={delta.shape}"
            )
        n = self.n_bus
        m_total = len(self.meas_index)

        keep_cols = np.concatenate([self.layout.non_slack_idx, n + np.arange(n)]).astype(np.int64)

        if m_total == 0:
            return cast("csr_matrix", csr_matrix((0, 2 * n - 1)))

        V = v * np.exp(1j * delta)

        dSbus_dVm, dSbus_dVa = self._dSbus_dV(V)
        empty_branch = cast("csr_matrix", csr_matrix((0, n)))
        if self.n_branch > 0:
            dSf_dVm, dSf_dVa = self._dSbr_dV(V, side=SIDE_FROM)
            dSt_dVm, dSt_dVa = self._dSbr_dV(V, side=SIDE_TO)
            dIfm_dVm, dIfm_dVa = self._dImbr_dV(V, side=SIDE_FROM)
            dItm_dVm, dItm_dVa = self._dImbr_dV(V, side=SIDE_TO)
        else:
            dSf_dVm = dSf_dVa = dSt_dVm = dSt_dVa = empty_branch
            dIfm_dVm = dIfm_dVa = dItm_dVm = dItm_dVa = empty_branch

        kind = self.meas_index.kind
        ok = self.meas_index.object_kind
        op = self.meas_index.object_pos
        side = self.meas_index.branch_side

        # Каждая часть: (global_row_indices_in_z, sparse-блок (count, 2n))
        parts: list[tuple[np.ndarray, csr_matrix]] = []

        def add(rows: np.ndarray, dVa: csr_matrix, dVm: csr_matrix) -> None:
            block = cast("csr_matrix", hstack([dVa, dVm], format="csr"))
            parts.append((rows, block))

        # --- P_inj / Q_inj на узлах ---
        mask = (kind == KIND_POWER_INJECTION_P) & (ok == OBJ_NODE)
        if mask.any():
            rg = np.where(mask)[0]
            pos = op[mask]
            add(rg, dSbus_dVa[pos, :].real, dSbus_dVm[pos, :].real)
        mask = (kind == KIND_POWER_INJECTION_Q) & (ok == OBJ_NODE)
        if mask.any():
            rg = np.where(mask)[0]
            pos = op[mask]
            add(rg, dSbus_dVa[pos, :].imag, dSbus_dVm[pos, :].imag)

        # --- Узловой balance (IPM): V/δ-зависимость как у P/Q_inj ---
        # ∂(Sbus[i] - (Pgen-Pnag))/∂V = ∂Sbus.real/∂V (та же)
        # Box-vars столбцы добавляются ниже отдельно.
        mask = (kind == KIND_NODE_BALANCE_P) & (ok == OBJ_NODE)
        if mask.any():
            rg = np.where(mask)[0]
            pos = op[mask]
            add(rg, dSbus_dVa[pos, :].real, dSbus_dVm[pos, :].real)
        mask = (kind == KIND_NODE_BALANCE_Q) & (ok == OBJ_NODE)
        if mask.any():
            rg = np.where(mask)[0]
            pos = op[mask]
            add(rg, dSbus_dVa[pos, :].imag, dSbus_dVm[pos, :].imag)

        # --- P / Q перетоков по ветвям ---
        if self.n_branch > 0:
            mask = (kind == KIND_POWER_P) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dSf_dVa[pos, :].real, dSf_dVm[pos, :].real)
            mask = (kind == KIND_POWER_P) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dSt_dVa[pos, :].real, dSt_dVm[pos, :].real)
            mask = (kind == KIND_POWER_Q) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dSf_dVa[pos, :].imag, dSf_dVm[pos, :].imag)
            mask = (kind == KIND_POWER_Q) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dSt_dVa[pos, :].imag, dSt_dVm[pos, :].imag)

            # |I| ветвей
            mask = (kind == KIND_CURRENT) & (ok == OBJ_BRANCH) & (side == SIDE_FROM)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dIfm_dVa[pos, :], dIfm_dVm[pos, :])
            mask = (kind == KIND_CURRENT) & (ok == OBJ_BRANCH) & (side == SIDE_TO)
            if mask.any():
                rg = np.where(mask)[0]
                pos = op[mask]
                add(rg, dItm_dVa[pos, :], dItm_dVm[pos, :])

        # --- |V| на узлах ---
        mask = (kind == KIND_VOLTAGE) & (ok == OBJ_NODE)
        if mask.any():
            rg = np.where(mask)[0]
            pos = op[mask]
            count = rg.shape[0]
            zero_va = cast("csr_matrix", csr_matrix((count, n)))
            eye_vm = cast(
                "csr_matrix",
                csr_matrix(
                    (np.ones(count, dtype=np.float64), (np.arange(count), pos)),
                    shape=(count, n),
                ),
            )
            add(rg, zero_va, eye_vm)

        # --- Soft-prior box-vars: V/δ-зависимости нет, только box-колонки ---
        # h = box-var, ∂h/∂V = 0, ∂h/∂δ = 0; box-блок добавится отдельно.
        for prior_kind in (
            KIND_BOX_PRIOR_PGEN,
            KIND_BOX_PRIOR_QGEN,
            KIND_BOX_PRIOR_PNAG,
            KIND_BOX_PRIOR_QNAG,
        ):
            mask = (kind == prior_kind) & (ok == OBJ_NODE)
            if mask.any():
                rg = np.where(mask)[0]
                count = rg.shape[0]
                zero_va = cast("csr_matrix", csr_matrix((count, n)))
                zero_vm = cast("csr_matrix", csr_matrix((count, n)))
                add(rg, zero_va, zero_vm)

        if not parts:
            return cast("csr_matrix", csr_matrix((m_total, 2 * n - 1)))

        all_rows = np.concatenate([rg for rg, _ in parts])
        H_full = cast("csr_matrix", vstack([blk for _, blk in parts], format="csr"))

        # Сборка может пропустить измерения с неподдерживаемым типом — в этом
        # случае их позиции в z останутся не заполнены и якобиан получится
        # короче. z_vector такое не пропускает (raise в _convert_*), но
        # подстрахуемся явной проверкой.
        if all_rows.shape[0] != m_total:
            missing = sorted(set(range(m_total)) - set(all_rows.tolist()))
            raise ValueError(
                f"Якобиан собран не для всех {m_total} измерений; "
                f"пропущены позиции в z: {missing}. "
                "Это означает, что в MeasurementIndex есть неизвестная "
                "комбинация (kind, object_kind, branch_side)."
            )

        # Перестановка строк, чтобы порядок совпадал с порядком в z.
        order = np.argsort(all_rows, kind="stable")
        H_full = H_full[order, :]

        H_E = H_full[:, keep_cols]

        # IPM: добавить столбцы для box-vars (Pgen, Qgen, Pnag, Qnag).
        # Для строк с balance-kind ставим ±1 в нужном столбце, остальные 0.
        if self.layout.has_box:
            n_box = self.layout.n_box
            box_block = self._build_balance_jacobian_block(m_total, n_box)
            H_E = cast("csr_matrix", hstack([H_E, box_block], format="csr"))

        return cast("csr_matrix", H_E.tocsr())

    def _build_balance_jacobian_block(self, m_total: int, n_box: int) -> csr_matrix:
        """Sparse-блок ``(m_total × n_box)`` с ±1 для balance-meas.

        Колонки идут в порядке: ``[Pgen, Qgen, Pnag, Qnag]``.

        Для ``KIND_NODE_BALANCE_P`` на узле i:
            -1 в столбце Pgen[i] (если i ∈ pgen_node_pos)
            +1 в столбце Pnag[i] (если i ∈ pnag_node_pos)

        Аналогично для Q-balance.
        """
        kind = self.meas_index.kind
        ok = self.meas_index.object_kind
        op = self.meas_index.object_pos

        pgen_pos = self.layout.pgen_node_pos
        qgen_pos = self.layout.qgen_node_pos
        pnag_pos = self.layout.pnag_node_pos
        qnag_pos = self.layout.qnag_node_pos
        off_qgen = int(pgen_pos.size)
        off_pnag = off_qgen + int(qgen_pos.size)
        off_qnag = off_pnag + int(pnag_pos.size)

        # Обратные карты pos → индекс в секции (нет узла → −1). Дублей в
        # *_node_pos по построению StateLayout нет (см. _inverse_pos_map),
        # поэтому совпадает с прежним np.where(...)[0][0] (первое = единств.).
        inv_pgen = _inverse_pos_map(pgen_pos)
        inv_qgen = _inverse_pos_map(qgen_pos)
        inv_pnag = _inverse_pos_map(pnag_pos)
        inv_qnag = _inverse_pos_map(qnag_pos)

        row_blocks: list[np.ndarray] = []
        col_blocks: list[np.ndarray] = []
        data_blocks: list[np.ndarray] = []

        def _emit(mask: np.ndarray, inv: np.ndarray, offset: int, sign: float) -> None:
            """Добавить ``sign`` в столбец ``offset + inv[pos]`` для мер из ``mask``.

            Меры без узла в секции (``inv[pos] < 0`` либо ``pos`` вне ``inv``)
            пропускаются — как ``if j.size`` в прежних циклах.
            """
            idx_meas = np.where(mask)[0]
            if idx_meas.size == 0:
                return
            pos_meas = op[idx_meas].astype(np.int64)
            in_range = pos_meas < inv.size
            j = np.full(pos_meas.shape, -1, dtype=np.int64)
            j[in_range] = inv[pos_meas[in_range]]
            valid = j >= 0
            if not valid.any():
                return
            row_blocks.append(idx_meas[valid].astype(np.int64))
            col_blocks.append(offset + j[valid])
            data_blocks.append(np.full(int(valid.sum()), sign, dtype=np.float64))

        # P-balance: ∂h/∂Pgen=-1 (offset 0), ∂h/∂Pnag=+1 (offset pnag).
        mask_bp = (kind == KIND_NODE_BALANCE_P) & (ok == OBJ_NODE)
        _emit(mask_bp, inv_pgen, 0, -1.0)
        _emit(mask_bp, inv_pnag, off_pnag, +1.0)

        # Q-balance: ∂h/∂Qgen=-1 (offset qgen), ∂h/∂Qnag=+1 (offset qnag).
        mask_bq = (kind == KIND_NODE_BALANCE_Q) & (ok == OBJ_NODE)
        _emit(mask_bq, inv_qgen, off_qgen, -1.0)
        _emit(mask_bq, inv_qnag, off_qnag, +1.0)

        # Soft-prior: ∂h/∂{box-var на узле} = +1; иначе 0.
        # offset для каждого kind: pgen=0, qgen=|pgen|, pnag=|pgen|+|qgen|,
        # qnag=|pgen|+|qgen|+|pnag|.
        for prior_kind, inv, offset in (
            (KIND_BOX_PRIOR_PGEN, inv_pgen, 0),
            (KIND_BOX_PRIOR_QGEN, inv_qgen, off_qgen),
            (KIND_BOX_PRIOR_PNAG, inv_pnag, off_pnag),
            (KIND_BOX_PRIOR_QNAG, inv_qnag, off_qnag),
        ):
            _emit((kind == prior_kind) & (ok == OBJ_NODE), inv, offset, +1.0)

        if not row_blocks:
            return cast("csr_matrix", csr_matrix((m_total, n_box)))
        return cast(
            "csr_matrix",
            csr_matrix(
                (
                    np.concatenate(data_blocks),
                    (np.concatenate(row_blocks), np.concatenate(col_blocks)),
                ),
                shape=(m_total, n_box),
            ),
        )

    # ------------------------------------------------------- частные блоки
    def _dSbus_dV(self, V: np.ndarray) -> tuple[csr_matrix, csr_matrix]:
        """``∂S_bus/∂V`` и ``∂S_bus/∂δ`` — обе ``(n × n)`` sparse complex.

        Используются формулы pandapower (``matrix_base._dSbus_dv``)::

            dS_bus/dVm = diag(V) · conj(Ybus · diag(V/|V|)) +
                         conj(diag(I_bus)) · diag(V/|V|)
            dS_bus/dVa = j · diag(V) · conj(diag(I_bus) − Ybus · diag(V))
        """
        Ibus = self.ybus @ V
        Vnorm = V / np.abs(V)
        diagV = _diag(V.astype(np.complex128))
        diagIbus = _diag(Ibus.astype(np.complex128))
        diagVnorm = _diag(Vnorm.astype(np.complex128))

        dSbus_dVm = diagV @ (self.ybus @ diagVnorm).conjugate() + diagIbus.conjugate() @ diagVnorm
        dSbus_dVa = 1j * diagV @ (diagIbus - self.ybus @ diagV).conjugate()
        return cast("csr_matrix", dSbus_dVm.tocsr()), cast("csr_matrix", dSbus_dVa.tocsr())

    def _dSbr_dV(self, V: np.ndarray, side: int) -> tuple[csr_matrix, csr_matrix]:
        """``∂S_branch/∂V`` и ``∂S_branch/∂δ`` — обе ``(n_branch × n_bus)``.

        Адаптация ``matrix_base._dSbr_dv``: для стороны «от» используется
        ``Yf`` и индексы ``from_idx``; для «до» — ``Yt`` и ``to_idx``.
        """
        if side == SIDE_FROM:
            Y = self.yf
            s = self.from_idx
        else:
            Y = self.yt
            s = self.to_idx

        nl = self.n_branch
        nb = self.n_bus
        il = np.arange(nl)
        I_br = Y @ V
        Vnorm = V / np.abs(V)

        diagVs = cast(
            "csr_matrix",
            csr_matrix((V[s].astype(np.complex128), (il, il)), shape=(nl, nl)),
        )
        diagI = cast(
            "csr_matrix",
            csr_matrix((I_br.astype(np.complex128), (il, il)), shape=(nl, nl)),
        )
        diagV_full = _diag(V.astype(np.complex128))
        diagVnorm_full = _diag(Vnorm.astype(np.complex128))

        sel_V = cast(
            "csr_matrix",
            csr_matrix((V[s].astype(np.complex128), (il, s)), shape=(nl, nb)),
        )
        sel_Vnorm = cast(
            "csr_matrix",
            csr_matrix((Vnorm[s].astype(np.complex128), (il, s)), shape=(nl, nb)),
        )

        dS_dVa = 1j * (diagI.conjugate() @ sel_V - diagVs @ (Y @ diagV_full).conjugate())
        dS_dVm = diagVs @ (Y @ diagVnorm_full).conjugate() + diagI.conjugate() @ sel_Vnorm
        return cast("csr_matrix", dS_dVm.tocsr()), cast("csr_matrix", dS_dVa.tocsr())

    def _dImbr_dV(self, V: np.ndarray, side: int) -> tuple[csr_matrix, csr_matrix]:
        """``∂|I_branch|/∂V`` и ``∂|I_branch|/∂δ`` — обе ``(n_branch × n_bus)`` real.

        ``matrix_base._dImbr_dV``: для нулевых токов в ветви производная
        принимается равной 0 (сингулярность ``conj(I)/|I|``).
        """
        Y = self.yf if side == SIDE_FROM else self.yt
        I_br = Y @ V
        Vnorm = V / np.abs(V)
        abs_I = np.abs(I_br)

        # Регуляризация деления на ноль.
        norm_diag = np.where(abs_I > 0, np.conj(I_br) / np.where(abs_I > 0, abs_I, 1.0), 0.0 + 0j)
        diagInorm = _diag(norm_diag.astype(np.complex128))

        diagV_full = _diag(V.astype(np.complex128))
        diagVnorm_full = _diag(Vnorm.astype(np.complex128))

        a = diagInorm @ Y @ diagV_full
        b = diagInorm @ Y @ diagVnorm_full

        # ``-Im(a)`` и ``Re(b)``; sparse_matrix.imag/real возвращают sparse.
        dIm_dVa = cast("csr_matrix", (-a.imag).tocsr())
        dIm_dVm = cast("csr_matrix", b.real.tocsr())
        return dIm_dVm, dIm_dVa
