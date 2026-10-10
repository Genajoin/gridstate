"""Version of the SE data contract (``SEInput`` / ``SEOutput``).

The data contract is the public API of gridstate: a formalised set of named
tables with fixed columns and types (see :mod:`gridstate.contract.tables`).
Its **version** is a separate SemVer string, owned here and versioned
together with the gridstate package.

Version policy (SemVer for the data schema):

* **MAJOR** — breaking change: removing/renaming a required column,
  changing its dtype family, removing a table, changing the storage format
  of a table, or changing the **meaning** of an existing field (same name and
  dtype, different interpretation: data written under the old meaning would
  pass a minor-level check and still be read wrongly). Data built under an
  older major is incompatible.
* **MINOR** — backward-compatible (additive) change: a new optional
  column or table. Data of an older minor is still readable.
* **PATCH** — changes that do not affect the schema (documentation, validators,
  clarifying column doc strings).

The version travels with the data as metadata (``contract_version`` — see
:data:`CONTRACT_VERSION_KEY`); on input gridstate validates compatibility
(see :func:`gridstate.contract.validate.validate_input`). This turns
today's *implicit* coupling (we silently assume the DTYPE fields) into an
*explicit, verifiable, versioned* interface.
"""

from __future__ import annotations

from dataclasses import dataclass


# Current version of the SE data contract — the single place it is defined; bump per
# the policy above and update in sync whenever :mod:`gridstate.contract.tables` or
# :mod:`gridstate.contract.derived_tables` changes.
#
# 2.0.0 froze the canonical contract after the field clean-up (removed fields formula/
# source_numer/tip_ti/prv_num/validity_timeout/guid_measurement, NODES.sxn_id,
# DerivedInputs.snapshot/rpn_resolved) — a breaking schema change (MAJOR bump).
# 3.0.0 (MAJOR):
#   * the derived inputs are stored as typed tables + JSON metadata
#     (:mod:`gridstate.contract.derived_tables`) instead of a pickled blob, with a
#     neutral key vocabulary;
#   * the meaning of ``branches.conductance_from`` / ``susceptance_from`` changed for
#     transformers with an off-nominal ratio: the from-side shunt sits at the from
#     node, in the from-side base, before the ideal transformer (tap-independent).
# Data of major 2 fails the compatibility check; it is read only through the
# explicit legacy path ``load_se_input_npz(path, trusted=True)``.
CONTRACT_VERSION = "3.0.0"

# Major of the data the explicit legacy path of the ``.npz`` loader still reads
# (``load_se_input_npz(path, trusted=True)``).
LEGACY_MAJOR = 2

# Key under which the contract version is stored in the data metadata
# (``model.metadata`` / future ``SEInput.meta``). The validator reads it on input.
CONTRACT_VERSION_KEY = "contract_version"


@dataclass(frozen=True, order=True)
class ContractVersion:
    """Parsed SemVer contract version (``MAJOR.MINOR.PATCH``).

    Comparable as a tuple ``(major, minor, patch)`` (``order=True``), so it is
    suitable for direct comparisons. Pre-release/build metadata is
    deliberately unsupported — the contract uses plain SemVer.
    """

    major: int
    minor: int
    patch: int

    @classmethod
    def parse(cls, text: str) -> ContractVersion:
        """Parse a ``"MAJOR.MINOR.PATCH"`` string into a :class:`ContractVersion`.

        Raises:
            ValueError: if the format is not ``int.int.int``.
        """
        parts = str(text).strip().split(".")
        if len(parts) != 3:
            raise ValueError(f"Contract version must be 'MAJOR.MINOR.PATCH', got: {text!r}")
        try:
            major, minor, patch = (int(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"Non-numeric components of the contract version: {text!r}") from exc
        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(f"Negative components of the contract version: {text!r}")
        return cls(major, minor, patch)

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def is_compatible_with(self, schema: ContractVersion) -> bool:
        """Whether data of this version is compatible with the schema of version ``schema``.

        Rule (reader = ``schema``, data = ``self``):

        * **major** must match — otherwise the schema changed incompatibly;
        * the data's **minor** must not exceed the schema's minor: data of a
          newer minor may carry columns/tables that an older
          reader does not have (additivity is guaranteed only "backwards").
        * **patch** does not affect compatibility.
        """
        if self.major != schema.major:
            return False
        return self.minor <= schema.minor


def current_version() -> ContractVersion:
    """Parsed current contract version (:data:`CONTRACT_VERSION`)."""
    return ContractVersion.parse(CONTRACT_VERSION)


def check_compatibility(data_version: str) -> str | None:
    """Single compatibility check of a data version against the built-in contract.

    Returns ``None`` when ``data_version`` is compatible, otherwise a
    human-readable incompatibility reason (version-mismatch description). A
    malformed version string raises :class:`ValueError` (delegated to
    :meth:`ContractVersion.parse`) — callers decide whether to propagate or
    translate it into their own diagnostic. Both compatibility-check sites
    (the ``.npz`` loader and :func:`gridstate.contract.validate.validate_input`)
    route through this one implementation, wrapping the result into their own
    exception type / issue.
    """
    parsed = ContractVersion.parse(data_version)
    schema_v = current_version()
    if parsed.is_compatible_with(schema_v):
        return None
    return f"data of version {parsed} is incompatible with contract {schema_v}"


def is_data_compatible(data_version: str) -> bool:
    """Whether the data version ``data_version`` is compatible with the current contract.

    A thin wrapper over :func:`check_compatibility` for the typical case
    "check the input data against the built-in contract" (bool result).
    """
    return check_compatibility(data_version) is None
