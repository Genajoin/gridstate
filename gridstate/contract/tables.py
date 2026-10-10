"""SE data contract schema: ``SEInput`` / ``SEOutput``.

This is the **declaration** of the contract — a formalised, minimal set of named
tables with fixed columns and types that state estimation (SE)
actually reads and writes. The schema fixes the data boundary at the input and output of SE.

**Minimality.** The contract includes ONLY the columns that gridstate actually
touches. The possible "tail" of a full-size network model (generator dynamics,
emergency automation, sections, virtual groups,
``kct``/``ves``/``type_ekv``, etc.) is deliberately NOT part of the contract.

**Column roles** (:class:`Role`) encode the three-layer model of the plan:

* :attr:`Role.KEY` — identity (``id``, ``from_node`` …), does not change.
* :attr:`Role.INPUT` — read-only input: SE reads it but does NOT mutate it.
* :attr:`Role.WORKING` — input that preprocessing **mutates** during
  the run (statuses, state, shunts, tap, initial V/δ). These columns are copied into the
  explicit working layer (``Working``), hence the role is singled out.
* :attr:`Role.OUTPUT` — solver result; lives in `SEOutput`, separate from the input.

``SEInput`` = tables with columns of roles {KEY, INPUT, WORKING} + "raw" tables;
``SEOutput`` = tables with columns of roles {KEY, OUTPUT}. The column ``voltage_magnitude``
is present in both (in the input — the initial guess, role WORKING; in the
output — the solved value, role OUTPUT): these are different layers, there is no conflict.

**Types are ours.** The dtype strings of the columns are fixed here: the contract
is self-contained and is the canonical source of types. The guard test
``tests/test_contract.py`` makes sure the contract stays true to what the code
actually reads/writes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum

import numpy as np


class Role(str, Enum):
    """Role of a column in the three-layer contract model (see the module docs)."""

    KEY = "key"
    INPUT = "input"
    WORKING = "working"
    OUTPUT = "output"


# Roles forming the input layer (``SEInput``) and the output layer (``SEOutput``).
_INPUT_ROLES: tuple[Role, ...] = (Role.KEY, Role.INPUT, Role.WORKING)
_OUTPUT_ROLES: tuple[Role, ...] = (Role.KEY, Role.OUTPUT)


@dataclass(frozen=True)
class ColumnSpec:
    """A single column of a contract table.

    Attributes:
        name: column name (== the object attribute name in the current model).
        dtype: numpy dtype string (``"f8"``, ``"i4"``, ``"bool"``, ``"U64"`` …).
        role: :class:`Role` — layer/mutability.
        required: whether it must be present in the input data (for INPUT/WORKING)
            or be guaranteed to be filled (for OUTPUT). ``False`` — absence/a
            default is acceptable.
        doc: brief purpose and which SE steps read/write it.
    """

    name: str
    dtype: str
    role: Role
    required: bool = True
    doc: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Schema of a single contract table (nodes/branches/measurements/generators/raw)."""

    name: str
    key: tuple[str, ...]
    columns: tuple[ColumnSpec, ...]
    doc: str = ""

    def column(self, name: str) -> ColumnSpec | None:
        """Spec of a column by name (or ``None``)."""
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def columns_by_role(self, *roles: Role) -> tuple[ColumnSpec, ...]:
        """Columns of the selected roles (declaration order is preserved)."""
        wanted = set(roles)
        return tuple(c for c in self.columns if c.role in wanted)

    def column_names(self, *roles: Role) -> tuple[str, ...]:
        """Names of the columns of the selected roles (or of all, if ``roles`` is empty)."""
        if not roles:
            return tuple(c.name for c in self.columns)
        return tuple(c.name for c in self.columns_by_role(*roles))

    def required_names(self, *roles: Role) -> tuple[str, ...]:
        """Names of the required columns of the selected roles."""
        cols = self.columns_by_role(*roles) if roles else self.columns
        return tuple(c.name for c in cols if c.required)

    def numpy_dtype(self, *roles: Role) -> np.dtype:
        """Build an ``np.dtype`` from the columns of the selected roles (or all)."""
        cols = self.columns_by_role(*roles) if roles else self.columns
        return np.dtype([(c.name, c.dtype) for c in cols])

    def input_dtype(self) -> np.dtype:
        """``np.dtype`` of the input layer of the table (roles KEY/INPUT/WORKING)."""
        return self.numpy_dtype(*_INPUT_ROLES)

    def output_dtype(self) -> np.dtype:
        """``np.dtype`` of the output layer of the table (roles KEY/OUTPUT)."""
        return self.numpy_dtype(*_OUTPUT_ROLES)


def io_dtype(in_schema: TableSchema, out_schema: TableSchema) -> np.dtype:
    """Combined INPUT+OUTPUT dtype of a contract table (single source of truth).

    Input-layer columns (KEY/INPUT/WORKING) first, followed by any OUTPUT columns
    not already present. This is the dtype the live working collections carry:
    the pipeline writes OUTPUT columns (``estimated_si``/``p_inj_calc``/flows)
    into the very same backing array, so it must hold both layers. Used by
    :meth:`gridstate.working.Working.empty`, the ``.npz`` loader
    (:mod:`gridstate.contract.serialize`) and the pandapower adapter.
    """
    in_dt = in_schema.input_dtype()
    io_fields = list(in_dt.descr)
    have = set(in_dt.names or ())
    out_dt = out_schema.output_dtype()
    for name in out_dt.names or ():
        if name not in have:
            io_fields.append((name, out_dt[name].str))
    return np.dtype(io_fields)


# ===========================================================================
# Nodes
# ===========================================================================

NODES = TableSchema(
    name="nodes",
    key=("id",),
    doc="Network nodes. Identity — id.",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique node identifier."),
        # --- read-only input ---
        ColumnSpec(
            "name", "U64", Role.INPUT, required=False, doc="Node name (diagnostics/output)."
        ),
        ColumnSpec(
            "area_id",
            "i4",
            Role.INPUT,
            required=False,
            doc="Area — state materialization by area-distribution.",
        ),
        ColumnSpec(
            "voltage_nominal",
            "f8",
            Role.INPUT,
            doc="Vnom, kV — base of the pu conversion (read everywhere).",
        ),
        ColumnSpec(
            "voltage_min",
            "f8",
            Role.INPUT,
            required=False,
            doc="Lower operating V limit; used by the V range filter.",
        ),
        ColumnSpec(
            "voltage_max",
            "f8",
            Role.INPUT,
            required=False,
            doc="Upper operating V limit; used by the V range filter.",
        ),
        ColumnSpec(
            "voltage_critical",
            "f8",
            Role.INPUT,
            required=False,
            doc="Critical (emergency) lower V limit; the lower threshold of the V filter.",
        ),
        ColumnSpec(
            "voltage_setpoint",
            "f8",
            Role.INPUT,
            required=False,
            doc="U_ZAD — setpoint V for the PV anchor.",
        ),
        ColumnSpec(
            "exist_load",
            "i1",
            Role.INPUT,
            required=False,
            doc="The node may carry load — write-split p_inj→load, IPM box.",
        ),
        ColumnSpec(
            "exist_gen",
            "i1",
            Role.INPUT,
            required=False,
            doc="The node may carry generation — write-split, IPM box, PV-promotion.",
        ),
        ColumnSpec(
            "load_model_id",
            "i4",
            Role.INPUT,
            required=False,
            doc=(
                "Reference to load_characteristics.id (0-based, -1=none) — the node's "
                "P(V)/Q(V) characteristic; read by apply_load_characteristic."
            ),
        ),
        ColumnSpec("load_p_min", "f8", Role.INPUT, required=False, doc="Lower P load — IPM box."),
        ColumnSpec(
            "load_p_max",
            "f8",
            Role.INPUT,
            required=False,
            doc="Upper P load — IPM box + materialization clamp.",
        ),
        ColumnSpec("load_q_min", "f8", Role.INPUT, required=False, doc="Lower Q load — IPM box."),
        ColumnSpec("load_q_max", "f8", Role.INPUT, required=False, doc="Upper Q load — IPM box."),
        # --- input mutated by preprocessing ---
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Whether the node is in service. Mutated by topology/telemetry/cascade.",
        ),
        ColumnSpec(
            "node_type",
            "i1",
            Role.WORKING,
            doc="0-PQ/1-PV/2-Slack. Mutated by refine_slack/refine_node_types.",
        ),
        ColumnSpec(
            "balance_priority",
            "i4",
            Role.WORKING,
            doc="Balancing priority; read by refine_slack, written by telemetry.",
        ),
        ColumnSpec("shunt_g", "f8", Role.WORKING, doc="Shunt G, S. Written by apply_reactors."),
        ColumnSpec(
            "shunt_b",
            "f8",
            Role.WORKING,
            doc="Shunt B, S. Written by apply_reactors (shunt reactor→shunt).",
        ),
        ColumnSpec(
            "generation_p",
            "f8",
            Role.WORKING,
            doc="P generation, MW — state. Written by generator materialization/aggregation.",
        ),
        ColumnSpec("generation_q", "f8", Role.WORKING, doc="Q generation, MVAr — state."),
        ColumnSpec(
            "generation_p_min",
            "f8",
            Role.WORKING,
            doc="Lower P generation — IPM box; overwritten by generator aggregation.",
        ),
        ColumnSpec(
            "generation_p_max", "f8", Role.WORKING, doc="Upper P generation — IPM box/aggregation."
        ),
        ColumnSpec(
            "generation_q_min", "f8", Role.WORKING, doc="Lower Q generation — IPM box/aggregation."
        ),
        ColumnSpec(
            "generation_q_max", "f8", Role.WORKING, doc="Upper Q generation — IPM box/aggregation."
        ),
        ColumnSpec(
            "load_p", "f8", Role.WORKING, doc="P load, MW — state. Written by materialization."
        ),
        ColumnSpec(
            "load_q", "f8", Role.WORKING, doc="Q load, MVAr — state. Written by materialization."
        ),
        ColumnSpec(
            "voltage_magnitude",
            "f8",
            Role.WORKING,
            doc="V magnitude, kV — initial guess/warm start; overwritten by the solver.",
        ),
        ColumnSpec(
            "voltage_angle",
            "f8",
            Role.WORKING,
            doc="V angle, rad — initial guess; overwritten by the solver.",
        ),
    ),
)


# Output layer of nodes (``SEOutput.nodes``) — a separate results namespace.
# V/δ are present here too (solved values) and in the NODES input (role WORKING,
# initial guess): different layers, hence no role conflict.
NODES_OUTPUT = TableSchema(
    name="nodes",
    key=("id",),
    doc="SE result per node (V/δ + injections/imbalances/load-generation estimates).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique node identifier."),
        ColumnSpec("voltage_magnitude", "f8", Role.OUTPUT, doc="Solved V magnitude, kV."),
        ColumnSpec("voltage_angle", "f8", Role.OUTPUT, doc="Solved V angle, rad."),
        ColumnSpec(
            "p_inj_calc", "f8", Role.OUTPUT, doc="Node P injection from the SE solution, MW."
        ),
        ColumnSpec(
            "q_inj_calc", "f8", Role.OUTPUT, doc="Node Q injection from the SE solution, MVAr."
        ),
        ColumnSpec(
            "imbalance_p", "f8", Role.OUTPUT, doc="P imbalance = p_inj_calc − (gen_p − load_p)."
        ),
        ColumnSpec("imbalance_q", "f8", Role.OUTPUT, doc="Q imbalance."),
        ColumnSpec("load_p_estimated", "f8", Role.OUTPUT, doc="Actual P load after SE, MW."),
        ColumnSpec("load_q_estimated", "f8", Role.OUTPUT, doc="Actual Q load after SE, MVAr."),
        ColumnSpec("generation_p_estimated", "f8", Role.OUTPUT, doc="Actual P generation, MW."),
        ColumnSpec("generation_q_estimated", "f8", Role.OUTPUT, doc="Actual Q generation, MVAr."),
        ColumnSpec(
            "solved",
            "i1",
            Role.OUTPUT,
            doc=(
                "1 — the node entered the SE computation (a V/δ solution exists); 0 — cut off "
                "by preprocessing (off/isolated): its OUTPUT fields are zero and "
                "voltage_magnitude carries the input value. A marker for "
                "consumers (promote/PF): do not take the zeros for an estimate."
            ),
        ),
    ),
)


# ===========================================================================
# Branches
# ===========================================================================

BRANCHES = TableSchema(
    name="branches",
    key=("id", "from_node", "to_node", "parallel_id"),
    doc="Branches (lines/transformers/reactors). Identity — id (+from/to/parallel).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique branch identifier."),
        ColumnSpec("from_node", "i4", Role.KEY, doc="From node."),
        ColumnSpec("to_node", "i4", Role.KEY, doc="To node."),
        ColumnSpec("parallel_id", "i2", Role.KEY, doc="Parallel branch number."),
        # --- read-only input ---
        ColumnSpec(
            "name", "U128", Role.INPUT, required=False, doc="Branch name (diagnostics/output)."
        ),
        ColumnSpec("branch_type", "i1", Role.INPUT, doc="0-line/1-transformer/2-reactor."),
        ColumnSpec(
            "current_limit_normal",
            "f8",
            Role.INPUT,
            required=False,
            doc="Current limit, A — for loading_pct.",
        ),
        ColumnSpec(
            "ti_p_from",
            "i4",
            Role.INPUT,
            required=False,
            doc="Telemetry reference P-from (side detection).",
        ),
        ColumnSpec(
            "ti_q_from", "i4", Role.INPUT, required=False, doc="Telemetry reference Q-from."
        ),
        ColumnSpec("ti_p_to", "i4", Role.INPUT, required=False, doc="Telemetry reference P-to."),
        ColumnSpec("ti_q_to", "i4", Role.INPUT, required=False, doc="Telemetry reference Q-to."),
        # --- input mutated by preprocessing ---
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Whether the branch is in service. Mutated by topology/cascade/telemetry.",
        ),
        ColumnSpec(
            "resistance",
            "f8",
            Role.WORKING,
            doc="R, Ohm. Mutated by normalize_breaker_reactance (R=X=0 short circuit).",
        ),
        ColumnSpec(
            "reactance", "f8", Role.WORKING, doc="X, Ohm. Mutated by normalize_breaker_reactance."
        ),
        ColumnSpec("conductance", "f8", Role.WORKING, doc="Series G, S. Recomputed on tap change."),
        ColumnSpec("susceptance", "f8", Role.WORKING, doc="Series B, S. Recomputed on tap change."),
        ColumnSpec("conductance_from", "f8", Role.WORKING, doc="Shunt G at the from end, S."),
        ColumnSpec("susceptance_from", "f8", Role.WORKING, doc="Shunt B at the from end, S."),
        ColumnSpec("conductance_to", "f8", Role.WORKING, doc="Shunt G at the to end, S."),
        ColumnSpec("susceptance_to", "f8", Role.WORKING, doc="Shunt B at the to end, S."),
        ColumnSpec(
            "tap_ratio", "f8", Role.WORKING, doc="Transformation ratio. Mutated by apply_rpn."
        ),
        ColumnSpec(
            "phase_shift", "f8", Role.WORKING, doc="Phase shift, rad. Mutated by apply_rpn."
        ),
        ColumnSpec(
            "tap_step_id",
            "i4",
            Role.INPUT,
            required=False,
            doc=(
                "Reference to tap_steps.id (0-based, -1=no tap changer). The tap position is selected "
                "outside the core, the core applies the ready row (apply_rpn)."
            ),
        ),
        ColumnSpec(
            "tap_side",
            "i1",
            Role.INPUT,
            required=False,
            doc="Regulation side: 0=from (HV)/1=to (LV). Reserved (default 0).",
        ),
        ColumnSpec(
            "tap_min",
            "f8",
            Role.INPUT,
            required=False,
            doc="Lower bound of the transformation ratio. Reserved for future optimization.",
        ),
        ColumnSpec(
            "tap_max",
            "f8",
            Role.INPUT,
            required=False,
            doc="Upper bound of the transformation ratio. Reserved.",
        ),
    ),
)


# Output layer of branches (``SEOutput.branches``) — flows/currents/losses/loading.
BRANCHES_OUTPUT = TableSchema(
    name="branches",
    key=("id",),
    doc="SE result per branch (P/Q flows at both ends, currents, losses, loading).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique branch identifier."),
        ColumnSpec("power_from_p", "f8", Role.OUTPUT, doc="P at the from end, MW."),
        ColumnSpec("power_from_q", "f8", Role.OUTPUT, doc="Q at the from end, MVAr."),
        ColumnSpec("power_to_p", "f8", Role.OUTPUT, doc="P at the to end, MW."),
        ColumnSpec("power_to_q", "f8", Role.OUTPUT, doc="Q at the to end, MVAr."),
        ColumnSpec("current_from", "f8", Role.OUTPUT, doc="Current at the from end, A."),
        ColumnSpec("current_to", "f8", Role.OUTPUT, doc="Current at the to end, A."),
        ColumnSpec("loss_p", "f8", Role.OUTPUT, doc="Active losses, MW."),
        ColumnSpec("loss_q", "f8", Role.OUTPUT, doc="Reactive losses, MVAr."),
        ColumnSpec("loading_pct", "f8", Role.OUTPUT, doc="Current loading, %."),
    ),
)


# ===========================================================================
# Measurements (z-vector + estimates)
# ===========================================================================

MEASUREMENTS = TableSchema(
    name="measurements",
    key=("id", "object_type", "object_id"),
    doc=(
        "Telemetry = the z-vector. The producer of the input builds the rows (resolving "
        "its own measurement definitions, plus synthetic rows); the pipeline then edits "
        "them, so most columns are WORKING."
    ),
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique measurement identifier."),
        ColumnSpec("object_type", "i1", Role.KEY, doc="0-node/1-branch/2-generator."),
        ColumnSpec("object_id", "i4", Role.KEY, doc="ID of the measured object."),
        # --- read-only input (defining/provenance) ---
        ColumnSpec("measurement_type", "i1", Role.INPUT, doc="0-P/1-Q/2-U/3-I/4-Pinj/5-Qinj."),
        ColumnSpec("min_value", "f8", Role.INPUT, required=False, doc="Lower plausible bound."),
        ColumnSpec("max_value", "f8", Role.INPUT, required=False, doc="Upper plausible bound."),
        ColumnSpec("name", "U128", Role.INPUT, required=False, doc="Measurement name."),
        # --- input mutated/derived by preprocessing ---
        ColumnSpec(
            "value",
            "f8",
            Role.WORKING,
            doc="Measurement value. Written by telemetry/materialization.",
        ),
        ColumnSpec(
            "variance",
            "f8",
            Role.WORKING,
            doc="Variance σ². Written by σ_Q-charging/filters. The solver builds R from it.",
        ),
        ColumnSpec(
            "weight",
            "f8",
            Role.WORKING,
            required=False,
            doc="Weight = 1/σ² (derived). Written by telemetry/filters; the solver reads variance, not weight.",
        ),
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Whether it is active in the z-vector. Muted/enabled by preprocessing.",
        ),
        ColumnSpec(
            "quality",
            "i1",
            Role.WORKING,
            doc="0-good/1-doubtful/2-bad. Written by filters/bad-data.",
        ),
        ColumnSpec(
            "branch_side", "i1", Role.WORKING, doc="0-from/1-to/-1-N/A (Yf vs Yt for P_to/Q_to)."
        ),
        ColumnSpec("is_pseudo", "bool", Role.WORKING, doc="Pseudo-prior (add_pseudo/synthesis)."),
        ColumnSpec("filter_flag", "i1", Role.WORKING, doc="Deactivation reason (0=ok, …)."),
        ColumnSpec(
            "source_code",
            "U16",
            Role.WORKING,
            doc="Measurement origin label (source/preprocessing).",
        ),
        ColumnSpec("source_guid", "U40", Role.WORKING, doc="GUID of the data source."),
    ),
)


# Output layer of measurements (``SEOutput.measurements``) — estimates and residuals.
MEASUREMENTS_OUTPUT = TableSchema(
    name="measurements",
    key=("id",),
    doc="SE result per measurement (estimated value + residual).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique measurement identifier."),
        ColumnSpec("estimated_si", "f8", Role.OUTPUT, doc="Estimate of our SE, original units."),
        ColumnSpec("residual", "f8", Role.OUTPUT, doc="Residual value − estimated."),
    ),
)


# ===========================================================================
# Generators (input; results are folded into node generation_*_estimated)
# ===========================================================================

GENERATORS = TableSchema(
    name="generators",
    key=("id", "node_id"),
    doc="Generators. SE reads powers/bounds, mutates only status (cascade from the node).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Unique generator identifier."),
        ColumnSpec("node_id", "i4", Role.KEY, doc="Connection node."),
        ColumnSpec("power_output", "f8", Role.INPUT, doc="Output P, MW (→ node gen injection)."),
        ColumnSpec("reactive_output", "f8", Role.INPUT, doc="Output Q, MVAr."),
        ColumnSpec("power_min", "f8", Role.INPUT, required=False, doc="Lower P — box/aggregation."),
        ColumnSpec("power_max", "f8", Role.INPUT, required=False, doc="Upper P — box/aggregation."),
        ColumnSpec(
            "reactive_min", "f8", Role.INPUT, required=False, doc="Lower Q — box/aggregation."
        ),
        ColumnSpec(
            "reactive_max", "f8", Role.INPUT, required=False, doc="Upper Q — box/aggregation."
        ),
        ColumnSpec(
            "status",
            "bool",
            Role.WORKING,
            doc="Whether in service. Mutated by apply_generator_status_from_node (node off ⇒ gen off).",
        ),
    ),
)


# ===========================================================================
# Domain numeric tables of the SE input (a format-agnostic replacement of the raw
# reactors/shema_ktr/load_models). Format resolution (selecting the on-load tap changer
# position, unit conversion, aggregation) is done in an external data source; only numeric,
# format-agnostic data arrives here. All columns are KEY/INPUT (the core reads them,
# does not mutate them).
# ===========================================================================

# Tap-changer steps (on-load and off-load): the row is selected outside the core
# (the producer decides which tap position applies); the core applies it to the
# branches (tap_ratio/phase_shift) and rescales the branch shunt (shunt_factor).
# The transformer physics stays in the format-agnostic core.
TAP_STEPS = TableSchema(
    name="tap_steps",
    key=("id", "branch_id"),
    doc="Selected tap changer step per branch (dimensionless, format-agnostic).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Step identifier (0-based)."),
        ColumnSpec("branch_id", "i4", Role.KEY, doc="Transformer branch (FK to branches.id)."),
        ColumnSpec("ktr_re", "f8", Role.INPUT, doc="Re of the selected ratio (1/tap convention)."),
        ColumnSpec("ktr_im", "f8", Role.INPUT, doc="Im of the selected ratio (phase shift)."),
        ColumnSpec("tap_ratio", "f8", Role.INPUT, doc="hypot(re,im) — target ratio magnitude."),
        ColumnSpec("phase_shift", "f8", Role.INPUT, doc="atan2(im,re) — target angle, rad."),
        ColumnSpec(
            "shunt_factor",
            "f8",
            Role.INPUT,
            doc="(tap_new/tap_old)², the branch shunt rescaling factor; 1.0 means no rescaling.",
        ),
    ),
)

# Static load characteristics P(V)/Q(V). Read by apply_load_characteristic
# (post_processing) via node.load_model_id. Immutable INPUT.
LOAD_CHARACTERISTICS = TableSchema(
    name="load_characteristics",
    key=("id",),
    doc="Polynomial load characteristics P(V)/Q(V); node.load_model_id refers here (0-based).",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Characteristic identifier (0-based)."),
        ColumnSpec("coeff_p_a0", "f8", Role.INPUT, doc="P: constant component."),
        ColumnSpec("coeff_p_a1", "f8", Role.INPUT, doc="P: linear in V."),
        ColumnSpec("coeff_p_a2", "f8", Role.INPUT, doc="P: quadratic in V."),
        ColumnSpec("coeff_q_b0", "f8", Role.INPUT, doc="Q: constant component."),
        ColumnSpec("coeff_q_b1", "f8", Role.INPUT, doc="Q: linear in V."),
        ColumnSpec("coeff_q_b2", "f8", Role.INPUT, doc="Q: quadratic in V."),
        ColumnSpec(
            "coeff_p_f", "f8", Role.INPUT, required=False, doc="P: frequency dependence. Reserved."
        ),
        ColumnSpec(
            "coeff_q_f", "f8", Role.INPUT, required=False, doc="Q: frequency dependence. Reserved."
        ),
    ),
)

# Shunt devices (reactors/capacitor banks), per device. The core aggregates the active ones
# into node.shunt_g/b for the Y-bus (a numeric step, replacing apply_reactors_to_node_shunt) and
# in the future switches steps (group_id/step — reserved).
SHUNTS = TableSchema(
    name="shunts",
    key=("id",),
    doc="Shunt devices per object; the core aggregates the active ones into node.shunt_g/b.",
    columns=(
        ColumnSpec("id", "i4", Role.KEY, doc="Shunt identifier (0-based)."),
        ColumnSpec("node_id", "i4", Role.INPUT, doc="Connection node (FK to nodes.id)."),
        ColumnSpec("conductance", "f8", Role.INPUT, doc="Device G, S (adapter: µS→S)."),
        ColumnSpec(
            "susceptance", "f8", Role.INPUT, doc="Device B, S (inductive shunt reactor <0)."
        ),
        ColumnSpec("status", "bool", Role.INPUT, doc="On/off (adapter: ON_LINE resolution)."),
        ColumnSpec(
            "group_id",
            "i4",
            Role.INPUT,
            required=False,
            doc="Capacitor bank group (switching). Reserved.",
        ),
        ColumnSpec(
            "step", "i4", Role.INPUT, required=False, doc="Current step within the group. Reserved."
        ),
    ),
)


# ===========================================================================
# Top-level contract
# ===========================================================================


@dataclass(frozen=True)
class SEInputSchema:
    """Input contract schema: a set of tables (roles KEY/INPUT/WORKING).

    The domain tables ``tap_steps``/``load_characteristics``/``shunts`` are
    format-agnostic numeric inputs (tap changer position selection, unit conversion,
    aggregation of shunts are done in an external data source). Input-only
    (roles KEY/INPUT).
    """

    nodes: TableSchema
    branches: TableSchema
    measurements: TableSchema
    generators: TableSchema
    tap_steps: TableSchema
    load_characteristics: TableSchema
    shunts: TableSchema

    def tables(self) -> tuple[TableSchema, ...]:
        """All tables in the order of field declaration (single source — the dataclass)."""
        return tuple(getattr(self, f.name) for f in fields(self))


@dataclass(frozen=True)
class SEOutputSchema:
    """Output contract schema: results (roles KEY/OUTPUT), keyed by id."""

    nodes: TableSchema
    branches: TableSchema
    measurements: TableSchema

    def tables(self) -> tuple[TableSchema, ...]:
        """All tables in the order of field declaration (single source — the dataclass)."""
        return tuple(getattr(self, f.name) for f in fields(self))


SE_INPUT = SEInputSchema(
    nodes=NODES,
    branches=BRANCHES,
    measurements=MEASUREMENTS,
    generators=GENERATORS,
    tap_steps=TAP_STEPS,
    load_characteristics=LOAD_CHARACTERISTICS,
    shunts=SHUNTS,
)

SE_OUTPUT = SEOutputSchema(
    nodes=NODES_OUTPUT,
    branches=BRANCHES_OUTPUT,
    measurements=MEASUREMENTS_OUTPUT,
)
