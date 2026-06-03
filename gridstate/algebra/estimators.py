"""Функции стоимости для SE: WLS, SHGM, LAV, QC, QL.

Adapted from pandapower:
    pandapower/estimation/algorithm/estimator.py.
Copyright (c) 2016-2025 University of Kassel and Fraunhofer IEE, Kassel.
Licensed under BSD 3-Clause; see the LICENSE file (Third-Party Notices).

Каждая функция принимает нормированные остатки ``r̃ = r/σ`` и возвращает
либо скалярное значение J, либо вектор взвешенных r для IRWLS.

Таблица:

    WLS   J = Σ r̃²                                    стандартный МНК
    SHGM  ρ(r̃), убывающий Φ при больших r̃              Schweppe-Huber GM
    LAV   J = Σ |r̃|                                    линейная по остаткам
    QC    J = r̃² при |r̃|<a, иначе a²                 квадратично-константная
    QL    J = r̃² при |r̃|<a, иначе линейная             квадратично-линейная
"""

from __future__ import annotations

import numpy as np


def wls_cost(normalized_residuals: np.ndarray) -> float:
    """J = Σ r̃²."""
    raise NotImplementedError


def shgm_weights(normalized_residuals: np.ndarray, tuning_constant: float = 1.5) -> np.ndarray:
    """Диагональ ``Φ`` (длины m) для SHGM — убывает при больших |r̃|."""
    raise NotImplementedError


def lav_cost(normalized_residuals: np.ndarray) -> float:
    """J = Σ |r̃|."""
    raise NotImplementedError


def qc_cost(normalized_residuals: np.ndarray, a: float = 3.0) -> float:
    """Quadratic-Constant cost."""
    raise NotImplementedError


def ql_cost(normalized_residuals: np.ndarray, a: float = 3.0) -> float:
    """Quadratic-Linear cost."""
    raise NotImplementedError
