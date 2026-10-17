"""gridstate-native ``Working`` container of the SE working layer.

Purpose
==========

``Working`` is a **self-contained numpy-backed container** of the working network model
on top of the contract structured arrays. It reproduces exactly the
model-API surface that preprocessing + the solver + the ``SEResult`` assembly read/write
— and NOTHING beyond it. The runtime paths of the container depend only on
numpy: no external libraries.

Surface
-----------

The container holds 4 main collections (``nodes`` / ``branches`` /
``measurements`` / ``generators``) and 3 domain input tables (``tap_steps`` /
``load_characteristics`` / ``shunts``) — exactly what preprocessing and the solver
read and write.

Each collection (:class:`_ArrayCollection`) is a thin wrapper over a single numpy
structured array (its dtype is fixed at initialization — it is also the contract
dtype of the corresponding table) plus an ``id → row`` index. Object iteration and
``get_by_id`` return a :class:`_RowProxy` — a light view onto ONE row of the backing
array that reads/writes the columns directly (attribute == column name).

``weight`` semantics
--------------------

The only "derived" column is the measurements ``weight`` (``1/variance``):
on ``add`` it is materialized from ``variance`` (see ``weight_from_variance``).
NOTE: if preprocessing later changes ``variance`` via ``update``, the ``weight``
column is NOT recomputed automatically — the solver reads ``variance`` (not
``weight``), see the contract ``MEASUREMENTS.weight``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Row proxy: a light view onto one row of the collection's backing array.
# ---------------------------------------------------------------------------


class _RowProxy:
    """Proxy onto ONE row of the numpy structured array of a collection.

    Columns are accessed via attributes (attribute name == contract column name):

    * reading ``proxy.name`` → a python scalar from ``arr[idx][name]`` (the numpy scalar
      is cast to the python type: ``int`` for integers, ``bool`` for
      booleans, ``float`` for reals, ``str`` for strings);
    * writing ``proxy.name = v`` → writes into ``arr[idx][name]`` (numpy itself casts
      to the column dtype), mutating the backing array in place.

    The proxy does NOT copy the row: it points to ``(arr, idx)``, so writes
    are visible in ``to_numpy()`` and survive losing the reference to the proxy itself — exactly like
    mutating a live object obtained via ``get_by_id``/iteration.
    """

    __slots__ = ("_arr", "_idx")

    def __init__(self, arr: np.ndarray, idx: int) -> None:
        # Via object.__setattr__ — otherwise our __setattr__ would intercept and reach
        # into the backing array for the nonexistent column "_arr".
        object.__setattr__(self, "_arr", arr)
        object.__setattr__(self, "_idx", idx)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is called only if the ordinary lookup (including __slots__)
        # did not find the attribute — so name is treated as a column name.
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        try:
            value = arr[idx][name]
        except (ValueError, KeyError, IndexError) as exc:
            raise AttributeError(name) from exc
        return _scalar_to_python(value)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _RowProxy.__slots__:
            object.__setattr__(self, name, value)
            return
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        if name not in (arr.dtype.names or ()):
            raise AttributeError(name)
        # numpy casts value to the column dtype when assigning into a row field.
        arr[idx][name] = value

    def __repr__(self) -> str:
        arr = object.__getattribute__(self, "_arr")
        idx = object.__getattribute__(self, "_idx")
        oid = arr[idx]["id"] if "id" in (arr.dtype.names or ()) else "?"
        return f"_RowProxy(id={oid}, idx={idx})"


def _scalar_to_python(value: Any) -> Any:
    """numpy scalar of a single cell → native python type.

    The goal is for ``proxy.attr`` to have a native python type
    (``int``/``float``/``bool``/``str``) rather than stay ``np.int32`` etc. This
    makes comparisons and ``int(...)``/``float(...)`` in consumers identical.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.str_):
        return str(value)
    # numpy string fields (U*) already yield python str via row indexing;
    # normalize bytes/other just in case.
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


# Measurement constructor defaults that differ from the dtype zero (zero-fill gives 0 /
# False / "" — which is enough for all the OTHER fields). The contract owns the
# defaults: variance=0.01, plausibility bounds ∓9999, status=True by
# default, branch_side=-1 (N/A). ``weight`` is computed separately as 1/variance
# if not set explicitly — see ``weight_from_variance`` below.
_MEASUREMENT_ADD_DEFAULTS: dict[str, Any] = {
    "variance": 0.01,
    "min_value": -9999.0,
    "max_value": 9999.0,
    "status": True,
    "branch_side": -1,
}


# ---------------------------------------------------------------------------
# Collection: a wrapper over a single structured array + an id→row index.
# ---------------------------------------------------------------------------


class _ArrayCollection:
    """numpy-backed collection of the objects of one contract table.

    Provides the compact object API used by the SE working layer:
    ``to_numpy`` / ``update_from_array`` / ``update`` / ``get_by_id`` / ``add`` /
    ``__iter__`` / ``__len__`` / ``ids`` / ``get_ids``.

    The backing store is a single structured array ``self._arr``; its dtype is fixed
    at initialization (from the passed array) and checked on
    ``update_from_array``/``add``. Row order == object order.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        add_defaults: dict[str, Any] | None = None,
        weight_from_variance: bool = False,
    ) -> None:
        self._dtype: np.dtype = array.dtype
        self._arr: np.ndarray = array.copy()
        # Defaults for unspecified columns in ``add`` (see _MEASUREMENT_ADD_DEFAULTS)
        # + the computed default ``weight=1/variance`` for measurements.
        self._add_defaults: dict[str, Any] = dict(add_defaults) if add_defaults else {}
        self._weight_from_variance: bool = weight_from_variance
        self._rebuild_index()

    # --- internals ---

    def _rebuild_index(self) -> None:
        # The id index is built lazily on first lookup: bulk rewrites
        # (``update_from_array``) are far more frequent than id lookups.
        self._index: dict[int, int] | None = None

    @property
    def _id_index(self) -> dict[int, int]:
        """``id → row`` index of the current backing array (built on demand)."""
        index = self._index
        if index is None:
            ids = self._arr["id"].tolist()
            index = dict(zip(ids, range(len(ids)), strict=True))
            self._index = index
        return index

    # --- reading ---

    def to_numpy(self, *, copy: bool = True) -> np.ndarray:
        """Backing array of the collection.

        By default a fresh copy (every call returns a new array the caller
        may mutate freely). ``copy=False`` returns a read-only view of the
        current backing array without copying it: for read-only consumers
        that finish with it before the collection is mutated (the view sees
        in-place updates and is detached by ``update_from_array``/``add``).
        """
        if copy:
            fresh: np.ndarray = self._arr.copy()
            return fresh
        view: np.ndarray = self._arr.view()
        view.flags.writeable = False
        return view

    def get_by_id(self, object_id: int) -> _RowProxy | None:
        """Proxy onto the row by ``id`` (or ``None`` if there is no such id)."""
        idx = self._id_index.get(int(object_id))
        if idx is None:
            return None
        return _RowProxy(self._arr, idx)

    def get_ids(self) -> list[int]:
        """List of ``id``s in row order."""
        return [int(self._arr[i]["id"]) for i in range(len(self._arr))]

    @property
    def ids(self) -> list[int]:
        """List of ``id``s in row order."""
        return self.get_ids()

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    # --- writing ---

    def update_from_array(self, array: np.ndarray, *, copy: bool = True) -> None:
        """Full rebuild of the collection from an array (dtype-strict, order preserved).

        The array IS the collection's data — we copy it and rebuild the index.
        ``update_from_array`` uses a strict dtype check.

        ``copy=False`` takes ownership of a writeable ``array`` instead of
        copying it: the caller hands over a private array (typically a fresh
        ``to_numpy()`` copy it has just edited) and must not touch it
        afterwards. A read-only array is still copied.
        """
        if array.dtype != self._dtype:
            raise ValueError("Array dtype must match collection dtype")
        self._arr = array.copy() if copy or not array.flags.writeable else array
        self._rebuild_index()

    def update(self, object_id: int, data: dict | None = None, /, **kwargs: Any) -> None:
        """Update the fields of a row by ``id`` point-wise.

        Compatible with both calls in the code:

        * dict style ``coll.update(id, {"status": False})`` (a positional dict);
        * kwargs style ``coll.update(id, status=False)``.

        Unknown keys (no such column) are silently ignored.
        """
        oid = int(object_id)
        if oid not in self._id_index:
            raise ValueError(f"object with id={oid} not found")
        idx = self._id_index[oid]
        fields = dict(data) if data else {}
        if kwargs:
            fields.update(kwargs)
        names = self._arr.dtype.names or ()
        for key, value in fields.items():
            if key in names:
                self._arr[idx][key] = value

    def add(self, row_data: dict) -> int:
        """Add a row to the END; return its ``id`` (insertion order is preserved).

        ``id`` is mandatory and unique (``ValueError`` on a duplicate).
        Unspecified columns are filled with the object constructor's defaults:
        non-zero ones from ``self._add_defaults`` (see ``_MEASUREMENT_ADD_DEFAULTS``),
        the rest with the dtype zero (zero/False/""). ``weight``, if not set explicitly
        and ``weight_from_variance=True``, is computed as 1/variance.
        ``row_data`` may contain keys outside the dtype — they are ignored.
        """
        if "id" not in row_data:
            raise ValueError("id is required")
        new_id = int(row_data["id"])
        if new_id in self._id_index:
            raise ValueError(f"object with id={new_id} already exists")

        names = self._arr.dtype.names or ()
        new_row = np.zeros(1, dtype=self._dtype)
        # 1) non-zero constructor defaults, 2) passed values (override them).
        for key, value in self._add_defaults.items():
            if key in names:
                new_row[0][key] = value
        for key, value in row_data.items():
            if key in names:
                new_row[0][key] = value
        # weight = 1/variance if the weight is not set explicitly.
        if self._weight_from_variance and "weight" in names and "weight" not in row_data:
            var = float(new_row[0]["variance"])
            new_row[0]["weight"] = (1.0 / var) if var > 0 else 1.0

        new_idx = len(self._arr)
        self._arr = np.append(self._arr, new_row)
        self._id_index[new_id] = new_idx
        return new_id

    def add_many(self, rows: list[dict]) -> list[int]:
        """Add rows in bulk to the END with ONE array concatenation.

        The semantics of each row are identical to :meth:`add` (mandatory unique
        ``id``, constructor defaults, ``weight`` = 1/variance, extra keys
        ignored). The difference — ``self._arr`` grows once for the whole batch.

        ``add`` does ``np.append`` (a copy of the whole array) on EVERY call: inserting
        ``k`` rows into an array of length ``n`` is O(k·n) — the bottleneck of
        pseudo-measurements on large models (tens of seconds → hundreds). ``add_many``
        inserts in O(n+k).
        """
        if not rows:
            return []

        names = self._arr.dtype.names or ()
        block = np.zeros(len(rows), dtype=self._dtype)
        new_ids: list[int] = []
        seen: set[int] = set()
        for i, row_data in enumerate(rows):
            if "id" not in row_data:
                raise ValueError("id is required")
            new_id = int(row_data["id"])
            if new_id in self._id_index or new_id in seen:
                raise ValueError(f"object with id={new_id} already exists")
            seen.add(new_id)
            # 1) non-zero constructor defaults, 2) passed values (override them).
            for key, value in self._add_defaults.items():
                if key in names:
                    block[i][key] = value
            for key, value in row_data.items():
                if key in names:
                    block[i][key] = value
            # weight = 1/variance if the weight is not set explicitly.
            if self._weight_from_variance and "weight" in names and "weight" not in row_data:
                var = float(block[i]["variance"])
                block[i]["weight"] = (1.0 / var) if var > 0 else 1.0
            new_ids.append(new_id)

        base = len(self._arr)
        self._arr = np.concatenate([self._arr, block])
        for offset, new_id in enumerate(new_ids):
            self._id_index[new_id] = base + offset
        return new_ids

    def copy(self) -> _ArrayCollection:
        """Independent copy of the collection: the array is copied (in the constructor),
        the config (``add_defaults`` / ``weight_from_variance``) is preserved.
        Mutations of the copy (``add`` / ``update_from_array``) do not reach the original.
        """
        return _ArrayCollection(
            self._arr,
            add_defaults=self._add_defaults,
            weight_from_variance=self._weight_from_variance,
        )

    # --- collection protocol ---

    def __iter__(self) -> Iterator[_RowProxy]:
        for i in range(len(self._arr)):
            yield _RowProxy(self._arr, i)

    def __len__(self) -> int:
        return len(self._arr)

    def __getitem__(self, index: int) -> _RowProxy:
        # Support for positional access by index.
        if index < 0:
            index += len(self._arr)
        if not 0 <= index < len(self._arr):
            raise IndexError(index)
        return _RowProxy(self._arr, index)

    def __repr__(self) -> str:
        return f"_ArrayCollection(count={len(self._arr)})"


# ---------------------------------------------------------------------------
# Copy-free access for internal hot paths. The working layer's own steps
# operate on ``_ArrayCollection``; any other collection object (a duck-typed
# model passed to a public helper) keeps the plain ``to_numpy()`` /
# ``update_from_array()`` protocol with the defensive copy.
# ---------------------------------------------------------------------------


def table_view(collection: Any) -> np.ndarray:
    """Array of ``collection`` for read-only use.

    A read-only view of a working collection (no copy), otherwise
    ``to_numpy()``. The caller must be done with it before the collection is
    mutated.
    """
    if isinstance(collection, _ArrayCollection):
        return collection.to_numpy(copy=False)
    return np.asarray(collection.to_numpy())


def table_copy(collection: Any) -> np.ndarray:
    """Private writable copy of the array of ``collection`` (copied once)."""
    if isinstance(collection, _ArrayCollection):
        return collection.to_numpy()
    return np.array(collection.to_numpy(), copy=True)


def table_replace(collection: Any, array: np.ndarray) -> None:
    """``update_from_array`` that hands a private array over without a copy.

    ``array`` must be a private copy (see :func:`table_copy`) the caller does
    not use afterwards.
    """
    if isinstance(collection, _ArrayCollection):
        collection.update_from_array(array, copy=False)
    else:
        collection.update_from_array(array)


# ---------------------------------------------------------------------------
# Working: the SE working layer = 4 main collections + 4 domain input tables
# (tap_steps / load_characteristics / shunts / load_characteristic_segments).
# ---------------------------------------------------------------------------


def _empty_domain(name: str) -> _ArrayCollection:
    """Empty collection of a domain input table with its contract dtype.

    ``name`` ∈ {tap_steps, load_characteristics, shunts}. Lazy import of the contract
    (as in :meth:`Working.empty`) — avoids a circular dependency on load.
    """
    from gridstate.contract import SE_INPUT

    schema = getattr(SE_INPUT, name)
    return _ArrayCollection(np.zeros(0, dtype=schema.input_dtype()))


class Working:
    """numpy-backed SE working layer.

    Holds 4 main collections (:class:`_ArrayCollection`) and 4 domain
    input tables (``tap_steps`` / ``load_characteristics`` / ``shunts`` /
    ``load_characteristic_segments``) —
    the surface that preprocessing and the solver read and write.
    """

    def __init__(
        self,
        *,
        nodes: _ArrayCollection,
        branches: _ArrayCollection,
        measurements: _ArrayCollection,
        generators: _ArrayCollection,
        tap_steps: _ArrayCollection | None = None,
        load_characteristics: _ArrayCollection | None = None,
        shunts: _ArrayCollection | None = None,
        load_characteristic_segments: _ArrayCollection | None = None,
    ) -> None:
        self.nodes = nodes
        self.branches = branches
        self.measurements = measurements
        self.generators = generators
        # Domain input-only tables (format-agnostic numeric inputs).
        # The default is an empty collection of the contract dtype.
        self.tap_steps = tap_steps if tap_steps is not None else _empty_domain("tap_steps")
        self.load_characteristics = (
            load_characteristics
            if load_characteristics is not None
            else _empty_domain("load_characteristics")
        )
        self.shunts = shunts if shunts is not None else _empty_domain("shunts")
        self.load_characteristic_segments = (
            load_characteristic_segments
            if load_characteristic_segments is not None
            else _empty_domain("load_characteristic_segments")
        )

    def copy(self) -> Working:
        """Deep independent copy of the working layer.

        Each of the 4 main collections and 3 domain tables is copied (the arrays
        are independent, the config is preserved). Guarantees that Input is read-only: when
        ``run()`` is given an already-built ``Working``
        (e.g. an npz input), the pipeline works on a copy — added
        pseudo-measurements and V/δ edits do NOT reach the passed object (otherwise
        a repeated ``run_se`` on the same input failed with a duplicate id).
        """
        return Working(
            nodes=self.nodes.copy(),
            branches=self.branches.copy(),
            measurements=self.measurements.copy(),
            generators=self.generators.copy(),
            tap_steps=self.tap_steps.copy(),
            load_characteristics=self.load_characteristics.copy(),
            shunts=self.shunts.copy(),
            load_characteristic_segments=self.load_characteristic_segments.copy(),
        )

    @classmethod
    def from_model(cls, model: Any) -> Working:
        """Build a ``Working`` from any model object with collections (the ``run()`` working layer).

        Each collection is seeded from ``model.X.to_numpy().copy()`` (the source
        returns a fresh array, ``_ArrayCollection`` copies it once more —
        independence from Input is guaranteed). No other model-level attributes
        are forwarded: the working layer does not read them.
        """

        def _domain(name: str) -> _ArrayCollection:
            # Domain input tables are not present in every source. Absence →
            # an empty collection.
            coll = getattr(model, name, None)
            if coll is None or not hasattr(coll, "to_numpy"):
                return _empty_domain(name)
            arr = coll.to_numpy()
            return _ArrayCollection(arr.copy()) if len(arr) > 0 else _empty_domain(name)

        return cls(
            nodes=_ArrayCollection(model.nodes.to_numpy().copy()),
            branches=_ArrayCollection(model.branches.to_numpy().copy()),
            measurements=_ArrayCollection(
                model.measurements.to_numpy().copy(),
                add_defaults=_MEASUREMENT_ADD_DEFAULTS,
                weight_from_variance=True,
            ),
            generators=_ArrayCollection(model.generators.to_numpy().copy()),
            tap_steps=_domain("tap_steps"),
            load_characteristics=_domain("load_characteristics"),
            shunts=_domain("shunts"),
            load_characteristic_segments=_domain("load_characteristic_segments"),
        )

    @classmethod
    def from_arrays(
        cls,
        *,
        nodes: np.ndarray,
        branches: np.ndarray,
        measurements: np.ndarray,
        generators: np.ndarray,
        tap_steps: np.ndarray | None = None,
        load_characteristics: np.ndarray | None = None,
        shunts: np.ndarray | None = None,
        load_characteristic_segments: np.ndarray | None = None,
    ) -> Working:
        """Build a ``Working`` directly from contract numpy arrays.

        Main entry: an external loader/test assembles the structured arrays of the
        ``SE_INPUT`` schema (nodes/branches/measurements/generators + domain
        tap_steps/load_characteristics/shunts) and passes them here. The arrays
        are copied (the input is read-only). Domain tables are optional (None → empty
        collection). ``measurements`` gets the same add defaults as
        :meth:`from_model`.
        """

        def _domain(arr: np.ndarray | None, name: str) -> _ArrayCollection:
            if arr is None or len(np.asarray(arr)) == 0:
                return _empty_domain(name)
            return _ArrayCollection(np.asarray(arr).copy())

        return cls(
            nodes=_ArrayCollection(np.asarray(nodes).copy()),
            branches=_ArrayCollection(np.asarray(branches).copy()),
            measurements=_ArrayCollection(
                np.asarray(measurements).copy(),
                add_defaults=_MEASUREMENT_ADD_DEFAULTS,
                weight_from_variance=True,
            ),
            generators=_ArrayCollection(np.asarray(generators).copy()),
            tap_steps=_domain(tap_steps, "tap_steps"),
            load_characteristics=_domain(load_characteristics, "load_characteristics"),
            shunts=_domain(shunts, "shunts"),
            load_characteristic_segments=_domain(
                load_characteristic_segments, "load_characteristic_segments"
            ),
        )

    @classmethod
    def empty(cls) -> Working:
        """Empty ``Working`` with contract dtypes.

        Seeds 4 zero-length collections with dtype = INPUT/WORKING ⊕ OUTPUT roles
        of the contract (``gridstate.contract``) — the full set of columns that the pipeline
        reads AND writes (``estimated_*`` / ``p_inj_calc`` / flows). Supports
        incremental construction via ``.nodes.add({...})`` / ``.get_by_id`` /
        ``.update`` — convenient for tests and synthetic data.
        """
        from gridstate.contract import SE_INPUT, SE_OUTPUT, io_dtype

        return cls.from_arrays(
            nodes=np.zeros(0, dtype=io_dtype(SE_INPUT.nodes, SE_OUTPUT.nodes)),
            branches=np.zeros(0, dtype=io_dtype(SE_INPUT.branches, SE_OUTPUT.branches)),
            measurements=np.zeros(0, dtype=io_dtype(SE_INPUT.measurements, SE_OUTPUT.measurements)),
            generators=np.zeros(0, dtype=SE_INPUT.generators.input_dtype()),
        )

    def __repr__(self) -> str:
        return (
            "Working("
            f"nodes={len(self.nodes)}, branches={len(self.branches)}, "
            f"measurements={len(self.measurements)}, generators={len(self.generators)}, "
            f"tap_steps={len(self.tap_steps)}, "
            f"load_characteristics={len(self.load_characteristics)}, "
            f"shunts={len(self.shunts)}, "
            f"load_characteristic_segments={len(self.load_characteristic_segments)})"
        )
