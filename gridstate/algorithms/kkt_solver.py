"""Решатель последовательности KKT-систем (нормальных уравнений) Ньютона.

Внутри одного solve (WLS LM-цикл, IPM inner-Newton) каждая итерация решает
``G·d = rhs`` с ``G = HᵀR⁻¹H (+ диагональная добавка)`` — симметричной
положительно определённой матрицей, чья sparsity-структура между
итерациями практически неизменна: меняются только значения (V/δ в
тригонометрии H, huber-веса в R⁻¹, barrier-диагональ). Это позволяет
сделать символьный анализ один раз и далее только числовую
рефакторизацию — ``scipy.sparse.linalg.spsolve`` так не умеет (каждый
вызов = полный анализ + LU).

Бэкенд ``"cholmod"`` использует CHOLMOD из SuiteSparse через cvxopt
(manylinux-wheels, лёгкая optional-зависимость): ``cholmod.symbolic`` на
паттерн + ``cholmod.numeric`` на итерацию, режим simplicial LDLᵀ
(``options['supernodal']=0``) — поздние IPM-итерации квази-определённы
(barrier ~1e14 на диагонали), supernodal-LLᵀ на них срывается с
реюзнутым fill-order. На ОДУ_Юга (339 систем, n=16k, nnz=237k) —
17.4s spsolve → 1.2s (×14). Фолбэк на вырожденной матрице (потеря
наблюдаемости) — единичный ``spsolve`` с warning.

Грабли стабилизации паттерна: scipy prune'ит численные нули и в
sparse-matmul, и в сложении, поэтому паттерн G «дрожит» на единицы
элементов между итерациями (236564..236569 nnz на Юге), а проекция на
superset сложением с нулевой паттерн-матрицей разваливается (нулевые
суммы тоже prune'ятся). Решение — паттерн-матрица с данными
``±PATTERN_EPS=1e-300``: сумма в «дырах» не нулевая (не prune'ится), а
для реальных значений G (≫1e-280) добавка за пределами разрядности
double — данные бит-в-бит те же.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


logger = logging.getLogger(__name__)

# Значение паттерн-матрицы superset'а: не 0.0 (prune при сложении), но
# заведомо за пределами double-разрядности любых реальных значений G.
PATTERN_EPS = 1e-300


def _cvxopt_available() -> bool:
    try:
        import cvxopt.cholmod  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_backend(backend: str) -> str:
    """Развернуть ``"auto"`` в конкретный бэкенд по наличию cvxopt.

    Args:
        backend: ``"auto"`` | ``"cholmod"`` | ``"scipy"``.

    Returns:
        ``"cholmod"`` или ``"scipy"``.

    Raises:
        ImportError: явный запрос ``"cholmod"`` без установленного cvxopt.
        ValueError: неизвестное имя бэкенда.
    """
    if backend == "scipy":
        return "scipy"
    if backend == "cholmod":
        if not _cvxopt_available():
            raise ImportError(
                "kkt_solver='cholmod' требует cvxopt (pip install cvxopt "
                "или gridstate[fast]); для автофолбэка используйте 'auto'"
            )
        return "cholmod"
    if backend == "auto":
        return "cholmod" if _cvxopt_available() else "scipy"
    raise ValueError(f"неизвестный kkt_solver: {backend!r}; доступны 'auto', 'cholmod', 'scipy'")


class KKTSolver:
    """Решатель ``G·d = rhs`` с реюзом символьной факторизации.

    Один экземпляр живёт на один itерационный цикл (solve_wls / solve_ipm):
    держит накопительный superset-паттерн и symbolic-фактор CHOLMOD,
    рефакторизуя только значения, пока паттерн не вырастет.

    Бэкенд ``"scipy"`` — прежнее поведение (spsolve) бит-в-бит.
    """

    def __init__(self, backend: str = "auto") -> None:
        self.backend = resolve_backend(backend)
        # superset-паттерн (csc с data=PATTERN_EPS), cvxopt-матрица и
        # symbolic-фактор; живут пока паттерн стабилен.
        self._pattern: csc_matrix | None = None
        self._A: Any = None
        self._F: Any = None

    def solve(self, G: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        """Решить ``G·d = rhs``; G — SPD csc.

        Каскад устойчивости cholmod-бэкенда: поздние IPM-итерации дают
        квази-определённые G (barrier ~1e14 на диагонали, rcond до 1e-24),
        на которых LLᵀ с реюзнутым fill-order (AMD от ранней матрицы
        паттерна) может сорваться в not-PD, хотя fresh-анализ от текущей
        матрицы проходит. Поэтому: (1) retry со сбросом паттерна — свежий
        ``symbolic`` живёт дальше и для следующих систем хвоста
        μ-расписания; (2) лишь затем единичный ``spsolve``-фолбэк
        (честно вырожденная G — потеря наблюдаемости).

        Simplicial LDLᵀ (``options['supernodal']=0``) как лекарство
        ОТВЕРГНУТ: диагональный D без пивотинга шумит на тех же
        квази-определённых системах — IPM Юга стопорился (stalled, it=4
        вместо 9 при бесфейловой факторизации).
        """
        if self.backend == "scipy":
            return np.asarray(spsolve(G, rhs), dtype=np.float64).ravel()
        try:
            return self._solve_cholmod(G, rhs)
        except ArithmeticError:
            pass
        try:
            self._pattern = None  # retry: fresh symbolic от текущей G
            return self._solve_cholmod(G, rhs)
        except ArithmeticError:
            logger.warning("kkt cholmod: матрица не PD и при fresh-анализе — фолбэк на spsolve")
            return np.asarray(spsolve(G, rhs), dtype=np.float64).ravel()

    def _solve_cholmod(self, G: csc_matrix, rhs: np.ndarray) -> np.ndarray:
        from cvxopt import cholmod, matrix, spmatrix

        if self._pattern is None:
            P = G.copy()
            P.sort_indices()
            changed = True
        else:
            P = G + self._pattern
            # ВАЖНО: scipy не гарантирует сортировку indices у результата
            # сложения — а V-update ниже кладёт P.data в cvxopt-матрицу,
            # хранящую данные в canonical CSC-порядке. Без sort_indices()
            # значения рассаживаются мимо позиций (мусорная матрица,
            # каскад not-PD). Сортируем ДО сравнения с паттерном.
            P.sort_indices()
            changed = (
                P.nnz != self._pattern.nnz
                or not np.array_equal(P.indices, self._pattern.indices)
                or not np.array_equal(P.indptr, self._pattern.indptr)
            )
        if changed:
            pat = P.copy()
            pat.data = np.full_like(pat.data, PATTERN_EPS)
            self._pattern = pat
            coo = P.tocoo()
            self._A = spmatrix(
                coo.data,
                coo.row.astype(np.int64),
                coo.col.astype(np.int64),
                P.shape,
            )
            self._F = cholmod.symbolic(self._A)
        else:
            # Порядок data в cvxopt CSC совпадает со scipy canonical csc.
            self._A.V = matrix(P.data)
        cholmod.numeric(self._A, self._F)
        b = matrix(np.asarray(rhs, dtype=np.float64))
        cholmod.solve(self._F, b)
        return np.asarray(b, dtype=np.float64).ravel()


__all__ = ["PATTERN_EPS", "KKTSolver", "resolve_backend"]
