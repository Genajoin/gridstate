"""Бит-в-бит эквивалентность ``gridstate.working.Working`` модели-источнику.

Контейнер ``Working`` (numpy-backed + row-proxy) обязан воспроизводить ровно ту
поверхность model-API, которую читает/пишет working-слой ``pipeline.run``, и быть
бит-в-бит совместимым с моделью-источником на ней.

Субъект — небольшая модель, собранная программно: проверяем поле-в-поле
равенство ``to_numpy()`` и атрибутов прокси для всех строк (включая
``measurement.weight`` — единственную производную-колонку).
"""

from __future__ import annotations

import numpy as np
import pytest

from gridstate.working import Working, _ArrayCollection, _RowProxy


# Атрибуты Measurement, читаемые в 4 object-итерация-местах + units + result.
MEAS_READ_ATTRS = (
    "id",
    "status",
    "quality",
    "variance",
    "weight",
    "measurement_type",
    "object_type",
    "object_id",
    "branch_side",
    "value",
    "min_value",
    "max_value",
    "is_pseudo",
    "filter_flag",
    "estimated_si",
    "estimated_value",
    "residual",
)
# ``name`` НЕ включён: это строковое поле, усекаемое по ширине U-dtype колонки
# контракта. Пайплайн в object-итерации (z_vector/post_processing) читает только
# numeric-атрибуты (не усекаются); бит-в-бит самой ``name``-колонки (репрезентация
# массива, на которой работает пайплайн) проверяется в to_numpy-тестах.
# Сравнивать proxy.name (репрезентация массива) с объектом-источником.name (полная строка)
# некорректно — это расхождение лишь ширины dtype, не контейнера.


# --- фикстуры моделей --------------------------------------------------------


@pytest.fixture
def small_model():
    """Маленькая модель-источник: 3 узла, 2 ветви, 2 генератора, 4 измерения.

    Покрывает все ветки proxy-типов: int (id/object_type), bool (status/
    is_pseudo), float (value/variance/voltage_magnitude), str (name), и
    измерение БЕЗ явного weight (weight посчитается как 1/variance).
    """
    m = Working.empty()
    m.nodes.add(
        {
            "id": 1,
            "name": "Slack",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 110.5,
            "voltage_angle": 0.0,
            "status": True,
            "node_type": 2,
        }
    )
    m.nodes.add(
        {
            "id": 2,
            "name": "PQ-2",
            "voltage_nominal": 110.0,
            "voltage_magnitude": 108.3,
            "voltage_angle": -0.05,
            "status": True,
            "node_type": 0,
            "load_p": 30.0,
            "load_q": 12.0,
        }
    )
    m.nodes.add(
        {
            "id": 3,
            "name": "PQ-3",
            "voltage_nominal": 35.0,
            "voltage_magnitude": 34.1,
            "voltage_angle": -0.11,
            "status": False,
            "node_type": 0,
        }
    )
    m.branches.add(
        {
            "id": 10,
            "name": "L1-2",
            "from_node": 1,
            "to_node": 2,
            "parallel_id": 1,
            "resistance": 1.2,
            "reactance": 9.5,
            "status": True,
            "branch_type": 0,
        }
    )
    m.branches.add(
        {
            "id": 11,
            "name": "T2-3",
            "from_node": 2,
            "to_node": 3,
            "parallel_id": 1,
            "resistance": 0.5,
            "reactance": 14.0,
            "tap_ratio": 0.95,
            "status": True,
            "branch_type": 1,
        }
    )
    m.generators.add(
        {"id": 100, "node_id": 1, "power_output": 55.0, "reactive_output": 12.0, "status": True}
    )
    m.generators.add(
        {"id": 101, "node_id": 2, "power_output": 0.0, "reactive_output": 0.0, "status": False}
    )
    # U на узле; P на ветви (from); Q-injection на узле; bad-quality мера.
    m.measurements.add(
        {
            "id": 1000,
            "name": "U_n1",
            "object_type": 0,
            "object_id": 1,
            "measurement_type": 2,
            "value": 110.5,
            "variance": 0.25,  # weight property -> 4.0
            "status": True,
            "quality": 0,
            "branch_side": -1,
        }
    )
    m.measurements.add(
        {
            "id": 1001,
            "name": "P_b10_from",
            "object_type": 1,
            "object_id": 10,
            "measurement_type": 0,
            "value": 41.0,
            "variance": 4.0,
            "status": True,
            "quality": 0,
            "branch_side": 0,
        }
    )
    m.measurements.add(
        {
            "id": 1002,
            "name": "Qinj_n2",
            "object_type": 0,
            "object_id": 2,
            "measurement_type": 5,
            "value": -12.0,
            "variance": 9.0,
            "status": True,
            "quality": 0,
            "is_pseudo": True,
            "filter_flag": 0,
        }
    )
    m.measurements.add(
        {
            "id": 1003,
            "name": "P_b11_to_bad",
            "object_type": 1,
            "object_id": 11,
            "measurement_type": 0,
            "value": 39.0,
            "variance": 4.0,
            "status": False,
            "quality": 2,
            "branch_side": 1,
        }
    )
    return m


# --- тесты -------------------------------------------------------------------


def test_from_model_to_numpy_field_for_field(small_model):
    """(1) ``to_numpy()`` каждой коллекции поле-в-поле == источник."""
    w = Working.from_model(small_model)
    for name in ("nodes", "branches", "measurements", "generators"):
        src_arr = getattr(small_model, name).to_numpy()
        w_arr = getattr(w, name).to_numpy()
        assert w_arr.dtype == src_arr.dtype, name
        assert np.array_equal(w_arr, src_arr), f"{name} массивы не идентичны"


def test_measurement_iteration_attrs_match_source(small_model):
    """(2) Итерация measurements: все читаемые атрибуты == источник, тот же порядок."""
    w = Working.from_model(small_model)
    src_list = list(small_model.measurements)
    w_list = list(w.measurements)
    assert len(w_list) == len(src_list)
    for src_m, w_m in zip(src_list, w_list, strict=True):
        for attr in MEAS_READ_ATTRS:
            src_v = getattr(src_m, attr)
            w_v = getattr(w_m, attr)
            assert type(w_v) is type(src_v), f"{attr}: тип {type(w_v)} != {type(src_v)}"
            if isinstance(src_v, float):
                assert w_v == pytest.approx(src_v), f"{attr}: {w_v} != {src_v}"
            else:
                assert w_v == src_v, f"{attr}: {w_v} != {src_v}"


def test_measurement_weight_is_property_value(small_model):
    """``weight`` (= 1/variance) материализован в колонку идентично."""
    w = Working.from_model(small_model)
    for src_m, w_m in zip(list(small_model.measurements), list(w.measurements), strict=True):
        assert w_m.weight == pytest.approx(src_m.weight)
    # узел id=1000 variance=0.25 -> weight=4.0
    m1000 = w.measurements.get_by_id(1000)
    assert m1000.weight == pytest.approx(4.0)


def test_node_get_by_id_and_update(small_model):
    """(3) get_by_id().V == источник; update точечно меняет только одну строку."""
    w = Working.from_model(small_model)
    for nid in (1, 2, 3):
        src_n = small_model.nodes.get_by_id(nid)
        w_n = w.nodes.get_by_id(nid)
        assert w_n.voltage_magnitude == pytest.approx(src_n.voltage_magnitude)
        assert w_n.voltage_angle == pytest.approx(src_n.voltage_angle)
        assert w_n.status == src_n.status
        assert w_n.node_type == src_n.node_type
    assert w.nodes.get_by_id(99999) is None

    before = w.nodes.to_numpy()
    # dict-стиль (позиционный dict)
    w.nodes.update(2, {"voltage_magnitude": 100.0})
    # kwargs-стиль
    w.nodes.update(2, voltage_angle=-0.2)
    arr = w.nodes.to_numpy()
    idx2 = {int(r["id"]): i for i, r in enumerate(arr)}[2]
    assert arr[idx2]["voltage_magnitude"] == pytest.approx(100.0)
    assert arr[idx2]["voltage_angle"] == pytest.approx(-0.2)
    # другие строки нетронуты
    for i in range(len(arr)):
        if i == idx2:
            continue
        assert np.array_equal(arr[i], before[i]), "update задел чужую строку"
    # запись через get_by_id-прокси тоже видна в to_numpy
    w.nodes.get_by_id(1).voltage_magnitude = 111.0
    assert w.nodes.to_numpy()[0]["voltage_magnitude"] == pytest.approx(111.0)


def test_branch_get_by_id(small_model):
    """get_by_id на ветвях: ключевые/working-колонки == источник."""
    w = Working.from_model(small_model)
    for bid in (10, 11):
        src_b = small_model.branches.get_by_id(bid)
        w_b = w.branches.get_by_id(bid)
        for attr in (
            "id",
            "from_node",
            "to_node",
            "parallel_id",
            "status",
            "resistance",
            "reactance",
            "tap_ratio",
            "branch_type",
        ):
            src_v = getattr(src_b, attr)
            w_v = getattr(w_b, attr)
            assert type(w_v) is type(src_v), f"{attr}"
            if isinstance(src_v, float):
                assert w_v == pytest.approx(src_v), attr
            else:
                assert w_v == src_v, attr


def test_add_measurement_appends_and_uniqueness(small_model):
    """(4) add → новый id, len+1, в конце to_numpy, виден итерацией; дубль → ValueError."""
    w = Working.from_model(small_model)
    n0 = len(w.measurements)
    new_id = w.measurements.add(
        {
            "id": 2000,
            "name": "synthetic",
            "object_type": 0,
            "object_id": 2,
            "measurement_type": 4,
            "value": 5.0,
            "variance": 1.0,
            "status": True,
            "is_pseudo": True,
        }
    )
    assert new_id == 2000
    assert len(w.measurements) == n0 + 1
    arr = w.measurements.to_numpy()
    assert int(arr[-1]["id"]) == 2000  # появилась в КОНЦЕ
    assert arr[-1]["name"] == "synthetic"
    assert bool(arr[-1]["is_pseudo"]) is True
    # незаданные поля — ноль dtype
    assert arr[-1]["quality"] == 0
    # видна итерацией, по правильному idx
    assert [int(r.id) for r in w.measurements][-1] == 2000
    assert w.measurements.get_by_id(2000) is not None
    # дубль id → ValueError
    with pytest.raises(ValueError):
        w.measurements.add(
            {"id": 2000, "object_type": 0, "object_id": 1, "measurement_type": 2, "value": 1.0}
        )


def test_add_matches_source_semantics(small_model):
    """add в исходный и в производный Working дают идентичный to_numpy()."""
    w = Working.from_model(small_model)
    row = {
        "id": 2001,
        "name": "m2001",
        "object_type": 1,
        "object_id": 10,
        "measurement_type": 1,
        "value": 7.5,
        "variance": 2.0,
        "status": True,
        "branch_side": 0,
    }
    small_model.measurements.add(dict(row))
    w.measurements.add(dict(row))
    assert np.array_equal(w.measurements.to_numpy(), small_model.measurements.to_numpy())


def test_add_many_matches_per_row_add(small_model):
    """add_many(rows) бит-в-бит эквивалентен последовательным add(): та же
    to_numpy(), тот же порядок, те же id-индексы. (O(n²)→O(n) оптимизация
    вставки псевдо-измерений должна быть поведенчески нейтральной.)"""
    rows = [
        {
            "id": 3000 + i,
            "name": f"m{i}",
            "object_type": 0,
            "object_id": i,
            "measurement_type": 4,
            "value": float(i),
            "variance": 2.0,  # weight = 1/variance = 0.5 (вес не задан явно)
            "status": True,
            "is_pseudo": True,
        }
        for i in range(5)
    ]
    w_seq = Working.from_model(small_model)
    for r in rows:
        w_seq.measurements.add(dict(r))
    w_bulk = Working.from_model(small_model)
    returned = w_bulk.measurements.add_many([dict(r) for r in rows])

    assert returned == [3000 + i for i in range(5)]
    assert np.array_equal(
        w_bulk.measurements.to_numpy(), w_seq.measurements.to_numpy()
    )
    # weight посчитан из variance так же, как в add()
    assert float(w_bulk.measurements.to_numpy()[-1]["weight"]) == 0.5
    # id-индекс согласован: get_by_id находит все вставленные
    for i in range(5):
        assert w_bulk.measurements.get_by_id(3000 + i) is not None


def test_add_many_rejects_duplicate_ids(small_model):
    """add_many ловит дубль и с уже существующим id, и внутри пакета."""
    w = Working.from_model(small_model)
    existing = int(w.measurements.to_numpy()[0]["id"])
    base = {"object_type": 0, "object_id": 1, "measurement_type": 4, "value": 1.0}
    # дубль с существующим
    with pytest.raises(ValueError):
        w.measurements.add_many([{"id": existing, **base}])
    # дубль внутри пакета
    with pytest.raises(ValueError):
        w.measurements.add_many([{"id": 4000, **base}, {"id": 4000, **base}])
    # пустой пакет — no-op
    assert w.measurements.add_many([]) == []


def test_update_from_array_roundtrip(small_model):
    """(5) update_from_array: модификация колонки отражается, id→idx корректен."""
    w = Working.from_model(small_model)
    arr = w.nodes.to_numpy()
    arr["voltage_magnitude"] = arr["voltage_magnitude"] + 1.0
    arr[0]["status"] = False
    w.nodes.update_from_array(arr)
    out = w.nodes.to_numpy()
    assert np.array_equal(out, arr)
    # индекс пересобран: get_by_id по-прежнему адресует правильную строку
    n2 = w.nodes.get_by_id(2)
    assert n2.voltage_magnitude == pytest.approx(
        arr["voltage_magnitude"][{int(r["id"]): i for i, r in enumerate(arr)}[2]]
    )
    assert w.nodes.get_by_id(1).status is False
    # dtype-strict: чужой dtype отвергается
    with pytest.raises(ValueError):
        w.nodes.update_from_array(w.measurements.to_numpy())


def test_iteration_order_matches_to_numpy_and_source(small_model):
    """(6) порядок итерации == порядок to_numpy == порядок list(источник)."""
    w = Working.from_model(small_model)
    for name in ("nodes", "branches", "measurements", "generators"):
        coll = getattr(w, name)
        src_coll = getattr(small_model, name)
        iter_ids = [int(p.id) for p in coll]
        numpy_ids = [int(x) for x in coll.to_numpy()["id"]]
        src_ids = [int(o.id) for o in src_coll]
        assert iter_ids == numpy_ids == src_ids, name
        assert coll.ids == numpy_ids
        assert coll.get_ids() == numpy_ids


def test_len_and_getitem(small_model):
    w = Working.from_model(small_model)
    assert len(w.nodes) == 3
    assert len(w.branches) == 2
    assert len(w.generators) == 2
    assert len(w.measurements) == 4
    assert isinstance(w.nodes[0], _RowProxy)
    assert int(w.nodes[0].id) == 1
    assert int(w.nodes[-1].id) == 3


def test_from_model_does_not_mutate_input(small_model):
    """Working независим от Input: правки working не текут в модель-источник."""
    w = Working.from_model(small_model)
    src_before = small_model.nodes.to_numpy()
    w.nodes.update(1, {"voltage_magnitude": 222.0})
    w.nodes.get_by_id(2).status = False
    assert np.array_equal(small_model.nodes.to_numpy(), src_before)


def test_array_collection_direct_construction():
    """``_ArrayCollection`` принимает голый structured-массив, dtype фиксируется."""
    dt = np.dtype([("id", "i4"), ("value", "f8"), ("status", "bool")])
    arr = np.array([(1, 1.5, True), (2, 2.5, False)], dtype=dt)
    coll = _ArrayCollection(arr)
    assert len(coll) == 2
    assert coll.dtype == dt
    assert coll.get_by_id(2).value == pytest.approx(2.5)
    assert coll.get_by_id(2).status is False
    # копия: мутация исходного arr не задевает коллекцию
    arr[0]["value"] = 99.0
    assert coll.get_by_id(1).value == pytest.approx(1.5)
