"""Тесты раскладки вектора состояния SE (``gridstate.state``)."""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.state import (
    StateLayout,
    flat_start,
    flat_start_with_box,
    pack,
    unpack,
    unpack_full,
)


# --------------------------------------------------------------- StateLayout
class TestStateLayout:
    def test_size_is_two_n_minus_one(self) -> None:
        layout = StateLayout.from_slack(n_bus=14, slack_idx=0)
        assert layout.size == 27

    def test_from_slack_excludes_slack_in_order(self) -> None:
        layout = StateLayout.from_slack(n_bus=5, slack_idx=2)
        assert layout.slack_idx == 2
        assert layout.non_slack_idx.tolist() == [0, 1, 3, 4]

    def test_from_slack_at_zero(self) -> None:
        layout = StateLayout.from_slack(n_bus=4, slack_idx=0)
        assert layout.non_slack_idx.tolist() == [1, 2, 3]

    def test_from_slack_at_last(self) -> None:
        layout = StateLayout.from_slack(n_bus=4, slack_idx=3)
        assert layout.non_slack_idx.tolist() == [0, 1, 2]

    def test_invalid_slack_index(self) -> None:
        with pytest.raises(ValueError, match="slack_idx"):
            StateLayout(n_bus=3, slack_idx=5, non_slack_idx=np.array([0, 1]))

    def test_invalid_n_bus(self) -> None:
        with pytest.raises(ValueError, match="n_bus"):
            StateLayout(n_bus=0, slack_idx=0, non_slack_idx=np.array([], dtype=np.int64))

    def test_non_slack_must_exclude_slack(self) -> None:
        with pytest.raises(ValueError, match="не должен присутствовать"):
            StateLayout(n_bus=3, slack_idx=1, non_slack_idx=np.array([0, 1]))

    def test_non_slack_size_validated(self) -> None:
        with pytest.raises(ValueError, match="форму"):
            StateLayout(n_bus=4, slack_idx=0, non_slack_idx=np.array([1, 2]))


# ------------------------------------------------------------ pack / unpack
class TestPackUnpack:
    def test_round_trip_random(self) -> None:
        rng = np.random.default_rng(42)
        layout = StateLayout.from_slack(n_bus=10, slack_idx=3)
        delta = rng.uniform(-0.3, 0.3, size=layout.n_bus)
        v = rng.uniform(0.95, 1.05, size=layout.n_bus)
        slack_delta = float(delta[layout.slack_idx])

        e = pack(delta, v, layout)
        delta_out, v_out = unpack(e, layout, slack_delta=slack_delta)

        np.testing.assert_allclose(delta_out, delta)
        np.testing.assert_allclose(v_out, v)

    def test_pack_layout(self) -> None:
        layout = StateLayout.from_slack(n_bus=4, slack_idx=1)
        delta = np.array([0.1, 0.0, -0.2, 0.3])  # slack at idx=1
        v = np.array([1.0, 1.05, 0.98, 1.02])
        e = pack(delta, v, layout)

        # E = [δ[0], δ[2], δ[3], V[0], V[1], V[2], V[3]]
        np.testing.assert_allclose(e[:3], [0.1, -0.2, 0.3])
        np.testing.assert_allclose(e[3:], [1.0, 1.05, 0.98, 1.02])

    def test_unpack_substitutes_slack_delta(self) -> None:
        layout = StateLayout.from_slack(n_bus=3, slack_idx=0)
        e = np.array([0.05, -0.1, 1.0, 1.02, 0.97])
        delta, v = unpack(e, layout, slack_delta=0.0)
        assert delta[0] == 0.0  # slack
        assert delta[1] == 0.05
        assert delta[2] == -0.1
        np.testing.assert_allclose(v, [1.0, 1.02, 0.97])

    def test_pack_validates_shapes(self) -> None:
        layout = StateLayout.from_slack(n_bus=4, slack_idx=0)
        with pytest.raises(ValueError, match="delta"):
            pack(np.zeros(3), np.ones(4), layout)
        with pytest.raises(ValueError, match="v "):
            pack(np.zeros(4), np.ones(5), layout)

    def test_unpack_validates_shape(self) -> None:
        layout = StateLayout.from_slack(n_bus=4, slack_idx=0)
        with pytest.raises(ValueError):
            unpack(np.zeros(5), layout)


# ------------------------------------------------------------ flat_start
class TestFlatStart:
    def test_flat_start_zero_angles_unit_voltages(self) -> None:
        layout = StateLayout.from_slack(n_bus=14, slack_idx=0)
        e = flat_start(layout)
        assert e.shape == (layout.size,)
        # δ-часть = 0
        np.testing.assert_array_equal(e[: layout.n_bus - 1], np.zeros(13))
        # V-часть = 1
        np.testing.assert_array_equal(e[layout.n_bus - 1 :], np.ones(14))

    def test_flat_start_unpacks_to_uniform_state(self) -> None:
        layout = StateLayout.from_slack(n_bus=5, slack_idx=2)
        delta, v = unpack(flat_start(layout), layout)
        np.testing.assert_array_equal(delta, np.zeros(5))
        np.testing.assert_array_equal(v, np.ones(5))


# ------------------------------------------------------------ box-vars (IPM)
class TestBoxVarsLayout:
    """IPM: расширение StateLayout box-переменными."""

    def test_default_layout_has_no_box_vars(self) -> None:
        """from_slack даёт WLS-режим: размер = 2·n_bus−1, has_box=False."""
        layout = StateLayout.from_slack(n_bus=10, slack_idx=0)
        assert not layout.has_box
        assert layout.n_box == 0
        assert layout.size == 2 * 10 - 1

    def test_with_box_vars_size(self) -> None:
        """Layout с box-vars имеет соответствующий размер."""
        layout = StateLayout(
            n_bus=5,
            slack_idx=0,
            non_slack_idx=np.array([1, 2, 3, 4], dtype=np.int64),
            pgen_node_pos=np.array([1, 2], dtype=np.int64),  # 2 Pgen
            qgen_node_pos=np.array([1, 2], dtype=np.int64),  # 2 Qgen
            pnag_node_pos=np.array([2, 3, 4], dtype=np.int64),  # 3 Pnag
            qnag_node_pos=np.array([2, 3, 4], dtype=np.int64),  # 3 Qnag
        )
        # 4 (δ) + 5 (V) + 2 + 2 + 3 + 3 = 19
        assert layout.size == 19
        assert layout.n_box == 10
        assert layout.has_box

    def test_offsets_consistent(self) -> None:
        """Смещения секций согласованы с их размерами."""
        layout = StateLayout(
            n_bus=3,
            slack_idx=0,
            non_slack_idx=np.array([1, 2], dtype=np.int64),
            pgen_node_pos=np.array([1], dtype=np.int64),  # 1
            qgen_node_pos=np.array([1, 2], dtype=np.int64),  # 2
            pnag_node_pos=np.array([2], dtype=np.int64),  # 1
            qnag_node_pos=np.array([1, 2], dtype=np.int64),  # 2
        )
        assert layout.offset_delta == 0
        assert layout.offset_v == 2  # n_bus−1 = 2
        assert layout.offset_pgen == 5  # 2·n_bus−1 = 5
        assert layout.offset_qgen == 6  # 5 + 1
        assert layout.offset_pnag == 8  # 6 + 2
        assert layout.offset_qnag == 9  # 8 + 1
        assert layout.size == 11  # 9 + 2

    def test_box_index_validation(self) -> None:
        """Индексы box-vars должны быть в [0, n_bus)."""
        with pytest.raises(ValueError, match="pgen_node_pos"):
            StateLayout(
                n_bus=3,
                slack_idx=0,
                non_slack_idx=np.array([1, 2], dtype=np.int64),
                pgen_node_pos=np.array([5], dtype=np.int64),  # вне диапазона
            )

    def test_box_index_must_be_1d(self) -> None:
        """Box-индексы должны быть 1-D."""
        with pytest.raises(ValueError, match="должен быть 1-D"):
            StateLayout(
                n_bus=3,
                slack_idx=0,
                non_slack_idx=np.array([1, 2], dtype=np.int64),
                qgen_node_pos=np.array([[0, 1]], dtype=np.int64),
            )

    def test_backward_compat_pack_unpack_with_empty_box(self) -> None:
        """С пустыми box-полями pack/unpack работают как раньше."""
        layout = StateLayout(
            n_bus=4,
            slack_idx=0,
            non_slack_idx=np.array([1, 2, 3], dtype=np.int64),
        )
        delta = np.array([0.0, 0.05, -0.1, 0.2])
        v = np.array([1.0, 1.02, 0.98, 1.05])
        e = pack(delta, v, layout)
        d2, v2 = unpack(e, layout, slack_delta=0.0)
        np.testing.assert_allclose(d2, delta)
        np.testing.assert_allclose(v2, v)
        assert e.shape == (layout.size,)
        assert layout.size == 2 * 4 - 1

    def test_pack_with_box_vars(self) -> None:
        """pack правильно укладывает box-секции."""
        layout = StateLayout(
            n_bus=3,
            slack_idx=0,
            non_slack_idx=np.array([1, 2], dtype=np.int64),
            pgen_node_pos=np.array([0, 2], dtype=np.int64),
            pnag_node_pos=np.array([1], dtype=np.int64),
        )
        e = pack(
            delta=np.array([0.0, 0.1, -0.1]),
            v=np.array([1.0, 1.05, 0.98]),
            layout=layout,
            pgen_estimated=np.array([100.0, 50.0]),
            pnag_estimated=np.array([30.0]),
        )
        # E = [δ[1], δ[2], V[0], V[1], V[2], Pgen[0], Pgen[2], Pnag[1]]
        assert e.shape == (8,)
        assert e[0] == 0.1  # δ[1]
        assert e[1] == -0.1  # δ[2]
        assert e[2] == 1.0  # V[0]
        assert e[5] == 100.0  # Pgen[0]
        assert e[6] == 50.0  # Pgen[2]
        assert e[7] == 30.0  # Pnag[1]

    def test_pack_validates_box_arr_required(self) -> None:
        """В IPM-режиме box-массив обязателен если в layout есть box-vars."""
        layout = StateLayout(
            n_bus=3,
            slack_idx=0,
            non_slack_idx=np.array([1, 2], dtype=np.int64),
            pgen_node_pos=np.array([0], dtype=np.int64),
        )
        with pytest.raises(ValueError, match=r"pgen_estimated.*обязателен"):
            pack(
                delta=np.zeros(3),
                v=np.ones(3),
                layout=layout,
                # pgen_estimated отсутствует — должно сломаться
            )

    def test_unpack_full_returns_six_arrays(self) -> None:
        """unpack_full всегда возвращает 6-tuple, пустые секции = пустые."""
        layout = StateLayout(
            n_bus=3,
            slack_idx=0,
            non_slack_idx=np.array([1, 2], dtype=np.int64),
            pgen_node_pos=np.array([0, 1], dtype=np.int64),
        )
        e = pack(
            delta=np.array([0.0, 0.1, -0.1]),
            v=np.array([1.0, 1.05, 0.98]),
            layout=layout,
            pgen_estimated=np.array([100.0, 50.0]),
        )
        delta, v, pgen, qgen, pnag, qnag = unpack_full(e, layout)
        np.testing.assert_allclose(delta, [0.0, 0.1, -0.1])
        np.testing.assert_allclose(v, [1.0, 1.05, 0.98])
        np.testing.assert_allclose(pgen, [100.0, 50.0])
        assert qgen.size == 0  # пусто
        assert pnag.size == 0
        assert qnag.size == 0

    def test_flat_start_with_box(self) -> None:
        """flat_start_with_box инициализирует box-секции переданными значениями."""
        layout = StateLayout(
            n_bus=2,
            slack_idx=0,
            non_slack_idx=np.array([1], dtype=np.int64),
            pgen_node_pos=np.array([0], dtype=np.int64),
            qgen_node_pos=np.array([0], dtype=np.int64),
        )
        e = flat_start_with_box(
            layout,
            pgen_init=np.array([100.0]),
            qgen_init=np.array([50.0]),
        )
        assert e.shape == (layout.size,)  # 1 + 2 + 1 + 1 = 5
        # δ = 0
        assert e[0] == 0.0
        # V = 1
        assert e[1] == 1.0 and e[2] == 1.0
        # box
        assert e[3] == 100.0
        assert e[4] == 50.0

    def test_flat_start_no_box_equals_basic(self) -> None:
        """flat_start без box эквивалентен старой версии."""
        layout = StateLayout.from_slack(n_bus=5, slack_idx=2)
        e1 = flat_start(layout)
        np.testing.assert_array_equal(e1[:4], np.zeros(4))
        np.testing.assert_array_equal(e1[4:], np.ones(5))
