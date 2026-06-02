"""Расчёт системных потерь активной/реактивной мощности после SE.

Модуль *диагностический*, в основной pipeline (``estimate``) не
подключается. Используется для сверки SE с эталонными отчётами и
для системных отчётов о потерях.

Конвенции:

* В ``PowerSystemModel`` (конвенция входного формата) ``branch.susceptance`` —
  это **полная** ёмкостная проводимость ветви B (См). В Π-схеме она
  делится пополам: B/2 в шунт «от» и B/2 в шунт «до». То же для
  ``branch.conductance`` (активные потери на корону).
* ``branch.tap_ratio`` хранится как **physical** turn ratio HV:LV (например
  18.58 для 110/6 кВ через коэффициент трансформации, 2.22 для 500/220).
  Внутри SE он нормируется к p.u.: ``K_pu = K_phys / (V_base_from /
  V_base_to)`` (см. ``gridstate/units.py``). Off-nominal-tap (±10%) даёт
  ``K_pu ∈ [0.9..1.1]``. Для расчёта потерь работаем в p.u.,
  переиспользуя ``model_to_pu`` + ``build_ybus`` (та же y-bus, что и SE).
* У трансформатора во входном формате *всё* g+jb обычно вынесено в сторону "от"
  через ``conductance_from`` / ``susceptance_from`` (``branch_g/b``
  остаётся нулевым).
* Узловые шунты — ``node.shunt_g + j·node.shunt_b`` (См). Реакторы (ШР)
  через ``apply_reactors_to_node_shunt`` уже сложены в ``node.shunt_b``,
  поэтому отдельно не суммируются.

Формулы (на одну ветвь, всё в p.u. после ``model_to_pu``):

.. code::

    V_f, V_t — комплексные напряжения узлов (V_pu·exp(jδ));
    y_ser = 1/(r + j·x);
    t     = tap_ratio_pu · exp(j·phase_shift);

    # Поток на стороне «от»/«до» — те же формулы, что в build_ybus:
    S_from = V_f · conj(Y_ff · V_f + Y_ft · V_t);
    S_to   = V_t · conj(Y_tf · V_f + Y_tt · V_t);

    # Серийные потери (R+jX часть, без шунта Π-схемы):
    I_ser = (V_f / t  − V_t) · y_ser;
    S_series = |I_ser|² · (r + j·x);

    # Общие потери ветви (включая шунт):
    S_loss_total = S_from + S_to;

    # Шунтовая часть — по разности:
    S_loss_shunt = S_loss_total − S_series;

  Узловой шунт (p.u.):

.. code::

    S_shunt_node = |V|² · conj(g_sh + j·b_sh)

Все ``S`` затем умножаются на ``BASE_MVA=100`` для перевода в МВт/МВАр.

Знак Q:

* в r (active): всегда положительные (рассеяние);
* в x (series-reactive): положительные на индуктивной серии (x>0);
* в b (shunt-capacitive ВЛ): в нашей storage ``b < 0`` (конвенция
  входного формата), и формула ``S = V·conj(I) = |V|²·conj(Y)`` даёт
  ``Im(S) = -|V|²·b > 0`` — это «потребление» в storage-конвенции, но
  физически зарядная B ВЛ **выдаёт** Q. Поэтому Q-компоненты ШУНТОВ
  (узловых и ветвевых) в отчёте **отрицаются** (``_SHUNT_Q_SIGN = -1``),
  чтобы совпадать по знаку с dq эталонной SE (ёмкостная выдача < 0, индуктивный
  ШР > 0) и быть физически осмысленными. Series-Q (R/X) НЕ трогается
  (индуктивное поглощение > 0 в обеих конвенциях). Это правка ТОЛЬКО
  отчётности — SE/Y-bus используют storage-конвенцию без изменений.

Для эталонного отчёта:

* ``area.dp`` ≡ ``dp_line + dp_tran + dp_shunt + dp_xx`` (проверено
  численно, расхождение порядка 1e-13);
* ``dp_line/dp_tran`` — активные потери в R соответственно ВЛ и трафо;
* ``dp_shunt`` — потери в шунтах **узлов** (``node.psh``);
* ``dp_xx`` — потери холостого хода трансформаторов (паспортные G_xx
  трафо, рассеяние в стали); в Π-схеме это активная компонента шунта,
  сидящего на стороне "от" трансформатора.
* ``dp_nag`` — НЕ потери в нагрузке, а часть генерации района; не
  входит в баланс потерь.

Для реактивных компонент аналогично:

* ``dq`` ≡ ``dq_line + dq_tran + dq_shunt + dq_xx``;
* ``dq_line/dq_tran`` — потребление Q в X серии ВЛ/трафо (положительные);
* ``dq_shunt`` — Q-потребление/генерация в узловых шунтах
  (положительное у индуктивных ШР, отрицательное у конденсаторов);
* ``dq_xx`` — Q-генерация зарядной B ВЛ (отрицательная, выдача в сеть).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from gridstate.units import BASE_MVA, model_to_pu
from gridstate.ybus import build_ybus


if TYPE_CHECKING:
    from power_system import PowerSystemModel


# Знак Q-шунта в отчёте о потерях. Наша storage-susceptance инвертирована
# относительно физической (b<0 ёмкостный, b>0 индуктивный — конвенция
# входного формата), поэтому ``Im(S)=−|V|²·b`` даёт Q-шунт со знаком, обратным
# dq эталонной SE и физическому смыслу «потерь» (выдача Q ёмкостью = отрицательные
# потери). Отрицаем, чтобы отчёт совпадал по знаку с эталонной SE и был физически
# осмыслен. НЕ влияет на SE/Y-bus — только на compute_system_losses.
_SHUNT_Q_SIGN = -1.0


@dataclass
class BranchLossRow:
    """Per-branch разбивка потерь (физ. единицы, МВт/МВАр)."""

    branch_id: int
    from_node: int
    to_node: int
    tip: str  # "line" / "trafo"
    p_loss_series: float
    q_loss_series: float
    p_loss_shunt: float
    q_loss_shunt: float
    p_loss_total: float
    q_loss_total: float


@dataclass
class SystemLosses:
    """Сводка по системе."""

    total_p_loss_mw: float
    total_q_loss_mvar: float
    # Разбивка по типам, МВт/МВАр
    p_line_series: float
    q_line_series: float
    p_trafo_series: float
    q_trafo_series: float
    p_branch_shunt: float
    q_branch_shunt: float
    p_node_shunt: float
    q_node_shunt: float
    # Per-branch list
    per_branch: list[BranchLossRow] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_p_loss_mw": self.total_p_loss_mw,
            "total_q_loss_mvar": self.total_q_loss_mvar,
            "breakdown": {
                "line_series_p": self.p_line_series,
                "line_series_q": self.q_line_series,
                "trafo_series_p": self.p_trafo_series,
                "trafo_series_q": self.q_trafo_series,
                "branch_shunt_p": self.p_branch_shunt,
                "branch_shunt_q": self.q_branch_shunt,
                "node_shunt_p": self.p_node_shunt,
                "node_shunt_q": self.q_node_shunt,
            },
            "per_branch": [
                {
                    "branch_id": r.branch_id,
                    "from_node": r.from_node,
                    "to_node": r.to_node,
                    "tip": r.tip,
                    "p_loss_series_mw": r.p_loss_series,
                    "q_loss_series_mvar": r.q_loss_series,
                    "p_loss_shunt_mw": r.p_loss_shunt,
                    "q_loss_shunt_mvar": r.q_loss_shunt,
                    "p_loss_total_mw": r.p_loss_total,
                    "q_loss_total_mvar": r.q_loss_total,
                }
                for r in self.per_branch
            ],
        }


def _branch_is_trafo_pu(
    g: float,
    b: float,
    g_from: float,
    b_from: float,
    tap: float,
) -> bool:
    """Признак трансформатора по p.u.-параметрам ветви.

    Эвристика: tap != 1 (off-nominal) ИЛИ шунт сосредоточен в одной стороне
    (g_from/b_from != 0 при g==b==0).
    """
    if not math.isclose(tap, 1.0, rel_tol=0.0, abs_tol=1e-6):
        return True
    return (abs(g) < 1e-12 and abs(b) < 1e-12) and (abs(g_from) > 1e-12 or abs(b_from) > 1e-12)


def compute_system_losses(
    model: PowerSystemModel,
    v_pu: np.ndarray | None = None,
    delta_rad: np.ndarray | None = None,
) -> SystemLosses:
    """Посчитать активные/реактивные потери системы по результатам SE.

    Расчёт идёт в p.u. через ``gridstate.units.model_to_pu``, что
    гарантирует консистентность с y-bus, использовавшейся в SE
    (см. ``gridstate/ybus.py``). Финальные тоталы переводятся в МВт/МВАр
    через ``BASE_MVA=100``.

    Если ``v_pu``/``delta_rad`` не переданы — V/δ берутся из
    ``model.nodes.voltage_magnitude / voltage_angle`` (т.е. уже записанные
    результаты SE; см. ``write_results_to_model``).

    Args:
        model: ``PowerSystemModel`` после ``estimate()``.
        v_pu: (n_bus_active,) — модули V в p.u. (опционально).
        delta_rad: (n_bus_active,) — углы в радианах (опционально).

    Returns:
        ``SystemLosses`` — тоталы + разбивка по типам + per-branch list.
    """
    network_pu = model_to_pu(model)
    n_bus = network_pu.n_bus
    n_branch = network_pu.n_branch

    if v_pu is None:
        # Из model.nodes.voltage_magnitude (после write_results_to_model).
        v_kv = np.empty(n_bus, dtype=np.float64)
        delta_arr = np.empty(n_bus, dtype=np.float64)
        for pos, nid in enumerate(network_pu.bus_ids.tolist()):
            node = model.nodes.get_by_id(int(nid))
            v_kv[pos] = float(node.voltage_magnitude) if node is not None else 0.0
            delta_arr[pos] = float(node.voltage_angle) if node is not None else 0.0
        v_pu_arr = np.where(network_pu.bus_vn_kv > 0, v_kv / network_pu.bus_vn_kv, 1.0)
        delta = delta_arr
    else:
        v_pu_arr = np.asarray(v_pu, dtype=np.float64)
        if v_pu_arr.shape != (n_bus,):
            raise ValueError(f"v_pu length {v_pu_arr.shape} != n_bus_active {n_bus}")
        if delta_rad is None:
            raise ValueError("delta_rad обязателен, если v_pu задан")
        delta = np.asarray(delta_rad, dtype=np.float64)

    # Комплексные V в p.u.
    v_complex = v_pu_arr * np.exp(1j * delta)

    # ---- Узловые шунты (в p.u., S = |V|² · conj(Y)) ----
    y_sh = network_pu.bus_g_shunt + 1j * network_pu.bus_b_shunt  # p.u.
    s_node_shunt_pu = (v_pu_arr**2) * np.conj(y_sh)
    p_node_shunt = float(s_node_shunt_pu.real.sum()) * BASE_MVA
    # Q-шунт — в ФИЗИЧЕСКОЙ конвенции потерь (= dq эталонной SE): ёмкостный
    # шунт (БК) ВЫДАЁТ Q → отрицательные потери; индуктивный (ШР) ПОГЛОЩАЕТ
    # → положительные. Наша storage susceptance инвертирована (b<0 ёмкостный,
    # b>0 индуктивный — конвенция входного формата), формула ``Im(S)=−|V|²·b`` даёт
    # знак, обратный физическому, поэтому отрицаем (``_SHUNT_Q_SIGN``). См.
    # docstring модуля. SE/Y-bus НЕ затронуты — это только отчётность.
    q_node_shunt = _SHUNT_Q_SIGN * float(s_node_shunt_pu.imag.sum()) * BASE_MVA

    # ---- Ветви (vectorized) ----
    if n_branch == 0:
        return SystemLosses(
            total_p_loss_mw=p_node_shunt,
            total_q_loss_mvar=q_node_shunt,
            p_line_series=0.0,
            q_line_series=0.0,
            p_trafo_series=0.0,
            q_trafo_series=0.0,
            p_branch_shunt=0.0,
            q_branch_shunt=0.0,
            p_node_shunt=p_node_shunt,
            q_node_shunt=q_node_shunt,
            per_branch=[],
        )

    # Используем ту же y-bus, что строит SE — гарантирует консистентность
    # с branch.loss_p/q (записанным в write_results_to_model).
    _, yf, yt = build_ybus(network_pu)
    i_from = yf @ v_complex
    i_to = yt @ v_complex
    s_from = v_complex[network_pu.from_idx] * np.conj(i_from)
    s_to = v_complex[network_pu.to_idx] * np.conj(i_to)
    s_loss_total = s_from + s_to  # p.u.

    # Серийная часть отдельно — для разбивки. С тем же tap'ом.
    z_ser = network_pu.branch_r + 1j * network_pu.branch_x
    if np.any(z_ser == 0):
        raise ValueError("Ветви с r=x=0 в active set; должны быть отключены до SE.")
    y_ser = 1.0 / z_ser
    tap_complex = network_pu.tap_ratio * np.exp(1j * network_pu.phase_shift)
    v_f = v_complex[network_pu.from_idx]
    v_t = v_complex[network_pu.to_idx]
    v_f_eff = v_f / tap_complex
    i_ser = (v_f_eff - v_t) * y_ser
    s_series = (np.abs(i_ser) ** 2) * z_ser  # p.u.

    # Шунтовая часть — по разности (s_loss_total = s_series + s_shunt).
    s_shunt = s_loss_total - s_series

    # Признак трафо per-branch
    is_trafo = np.array(
        [
            _branch_is_trafo_pu(
                float(network_pu.branch_g[i]),
                float(network_pu.branch_b[i]),
                float(network_pu.branch_g_from[i]),
                float(network_pu.branch_b_from[i]),
                float(network_pu.tap_ratio[i]),
            )
            for i in range(n_branch)
        ],
        dtype=bool,
    )

    p_ser_mw = s_series.real * BASE_MVA
    q_ser_mvar = s_series.imag * BASE_MVA
    p_sh_mw = s_shunt.real * BASE_MVA
    # Q-шунт в физической конвенции потерь (зарядная B ВЛ ВЫДАЁТ Q →
    # отрицательно, как ``dq_xx`` эталонной SE); отрицаем из-за инвертированной
    # storage-susceptance (см. docstring + _SHUNT_Q_SIGN).
    q_sh_mvar = _SHUNT_Q_SIGN * s_shunt.imag * BASE_MVA

    p_line_ser = float(p_ser_mw[~is_trafo].sum())
    q_line_ser = float(q_ser_mvar[~is_trafo].sum())
    p_trafo_ser = float(p_ser_mw[is_trafo].sum())
    q_trafo_ser = float(q_ser_mvar[is_trafo].sum())
    p_branch_shunt = float(p_sh_mw.sum())
    q_branch_shunt = float(q_sh_mvar.sum())

    # Per-branch list. Используем оригинальные id узлов (не позиции).
    from_node_ids = network_pu.bus_ids[network_pu.from_idx]
    to_node_ids = network_pu.bus_ids[network_pu.to_idx]
    per_branch: list[BranchLossRow] = []
    for i in range(n_branch):
        per_branch.append(
            BranchLossRow(
                branch_id=int(network_pu.branch_ids[i]),
                from_node=int(from_node_ids[i]),
                to_node=int(to_node_ids[i]),
                tip="trafo" if is_trafo[i] else "line",
                p_loss_series=float(p_ser_mw[i]),
                q_loss_series=float(q_ser_mvar[i]),
                p_loss_shunt=float(p_sh_mw[i]),
                q_loss_shunt=float(q_sh_mvar[i]),
                p_loss_total=float(p_ser_mw[i] + p_sh_mw[i]),
                q_loss_total=float(q_ser_mvar[i] + q_sh_mvar[i]),
            )
        )

    total_p = p_line_ser + p_trafo_ser + p_branch_shunt + p_node_shunt
    total_q = q_line_ser + q_trafo_ser + q_branch_shunt + q_node_shunt

    return SystemLosses(
        total_p_loss_mw=total_p,
        total_q_loss_mvar=total_q,
        p_line_series=p_line_ser,
        q_line_series=q_line_ser,
        p_trafo_series=p_trafo_ser,
        q_trafo_series=q_trafo_ser,
        p_branch_shunt=p_branch_shunt,
        q_branch_shunt=q_branch_shunt,
        p_node_shunt=p_node_shunt,
        q_node_shunt=q_node_shunt,
        per_branch=per_branch,
    )
