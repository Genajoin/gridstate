"""Юнит-тесты парсера формул дорасчёта (``tests/_dor_parser.py``)."""

from __future__ import annotations

import pytest

from tests._dor_parser import (
    BinOp,
    Compare,
    Const,
    Deps,
    Evaluator,
    IfExpr,
    LogicOp,
    ParseError,
    TIRef,
    TMRef,
    TSRef,
    UnaryMinus,
    collect_deps,
    is_numeric_code,
    parse,
)


# ── Парсинг ──────────────────────────────────────────────────────────────────


def test_parse_const() -> None:
    assert parse("42") == Const(42.0)
    assert parse("3.14") == Const(3.14)


def test_parse_unary_minus() -> None:
    assert parse("-5") == UnaryMinus(Const(5.0))


def test_parse_tm_ref() -> None:
    assert parse('tm[1239, "I"]') == TMRef(1239, "I")
    assert parse('-tm[1239, "I"]') == UnaryMinus(TMRef(1239, "I"))


def test_parse_ti_ref() -> None:
    assert parse("ti[6769]") == TIRef(6769)
    assert parse("-ti[6769]") == UnaryMinus(TIRef(6769))


def test_parse_ts_ref() -> None:
    assert parse("ts[5448]") == TSRef(5448)


def test_parse_arithmetic() -> None:
    expr = parse('tm[100, "I"] + tm[101, "I"]')
    assert expr == BinOp("+", TMRef(100, "I"), TMRef(101, "I"))


def test_parse_arith_priority() -> None:
    # 2 + 3 * 4 → 2 + (3*4)
    assert parse("2 + 3 * 4") == BinOp("+", Const(2.0), BinOp("*", Const(3.0), Const(4.0)))


def test_parse_arith_with_const() -> None:
    expr = parse('tm[41962, "S"]+1')
    assert expr == BinOp("+", TMRef(41962, "S"), Const(1.0))


def test_parse_division() -> None:
    expr = parse("0.01*ti[2888]/2200")
    assert expr == BinOp("/", BinOp("*", Const(0.01), TIRef(2888)), Const(2200.0))


def test_parse_paren_sum() -> None:
    expr = parse("(ts[8754])+(ts[8755])+(ts[8756])")
    assert expr == BinOp(
        "+",
        BinOp("+", TSRef(8754), TSRef(8755)),
        TSRef(8756),
    )


def test_parse_comparison() -> None:
    assert parse("ti[21916]=0") == Compare("=", TIRef(21916), Const(0.0))
    assert parse('tm[7114, "I"]<201') == Compare("<", TMRef(7114, "I"), Const(201.0))


def test_parse_if_then_else() -> None:
    expr = parse("IF (ti[21916]=0) THEN (0) ELSE (1)")
    assert expr == IfExpr(
        Compare("=", TIRef(21916), Const(0.0)),
        Const(0.0),
        Const(1.0),
    )


def test_parse_if_with_logic() -> None:
    src = "IF ((ts[220]+ts[221]>0) & (ts[222]=1)) THEN (1) ELSE (0)"
    node = parse(src)
    assert isinstance(node, IfExpr)
    assert isinstance(node.cond, LogicOp)
    assert node.cond.op == "&"


def test_parse_multiline() -> None:
    src = 'IF (ti[1]=0)\nTHEN\n(\n   tm[2, "I"]\n)\nELSE\n(\n   0\n)'
    node = parse(src)
    assert isinstance(node, IfExpr)


def test_parse_error_garbage() -> None:
    with pytest.raises(ParseError):
        parse("garbage @ 123")


def test_parse_error_eof() -> None:
    with pytest.raises(ParseError):
        parse("tm[100,")


# ── Evaluator ────────────────────────────────────────────────────────────────


def make_evaluator(
    tm: dict | None = None, ti: dict | None = None, ts: dict | None = None
) -> Evaluator:
    tm = tm or {}
    ti = ti or {}
    ts = ts or {}
    return Evaluator(
        tm=lambda n, c: tm.get((n, c)),
        ti=lambda k: ti.get(k),
        ts=lambda k: ts.get(k),
    )


def test_eval_const() -> None:
    e = make_evaluator()
    assert e.evaluate(parse("42")) == 42.0


def test_eval_tm_resolved() -> None:
    e = make_evaluator(tm={(100, "I"): 5.5})
    assert e.evaluate(parse('tm[100, "I"]')) == 5.5


def test_eval_tm_unresolved() -> None:
    e = make_evaluator()
    assert e.evaluate(parse('tm[100, "I"]')) is None


def test_eval_unary_minus() -> None:
    e = make_evaluator(tm={(1, "I"): 7.0})
    assert e.evaluate(parse('-tm[1, "I"]')) == -7.0


def test_eval_arithmetic() -> None:
    e = make_evaluator(tm={(1, "I"): 3.0, (2, "I"): 4.0})
    assert e.evaluate(parse('tm[1, "I"] + tm[2, "I"]')) == 7.0
    assert e.evaluate(parse('tm[1, "I"] * tm[2, "I"]')) == 12.0


def test_eval_unresolved_propagates() -> None:
    # tm[1] resolved, tm[2] not → результат None
    e = make_evaluator(tm={(1, "I"): 3.0})
    assert e.evaluate(parse('tm[1, "I"] + tm[2, "I"]')) is None


def test_eval_division_by_zero() -> None:
    e = make_evaluator(ti={5: 0.0})
    assert e.evaluate(parse("100 / ti[5]")) is None


def test_eval_compare() -> None:
    e = make_evaluator(ti={1: 5.0})
    assert e.evaluate(parse("ti[1] = 5")) == 1.0
    assert e.evaluate(parse("ti[1] < 10")) == 1.0
    assert e.evaluate(parse("ti[1] > 10")) == 0.0


def test_eval_logic_and() -> None:
    e = make_evaluator(ts={1: 1.0, 2: 0.0})
    assert e.evaluate(parse("ts[1] & ts[2]")) == 0.0
    assert e.evaluate(parse("ts[1] & ts[1]")) == 1.0


def test_eval_logic_short_circuit() -> None:
    # ts[1]=0, ts[2] не задан — should return 0 без обращения к ts[2]
    e = make_evaluator(ts={1: 0.0})
    assert e.evaluate(parse("ts[1] & ts[2]")) == 0.0
    # ts[1]=1, ts[2] не задан — нужно обратиться к ts[2] → None
    e2 = make_evaluator(ts={1: 1.0})
    assert e2.evaluate(parse("ts[1] & ts[2]")) is None


def test_eval_if_then() -> None:
    e = make_evaluator(ti={1: 5.0, 2: 99.0})
    assert e.evaluate(parse("IF (ti[1]=5) THEN (ti[2]) ELSE (0)")) == 99.0


def test_eval_if_else() -> None:
    e = make_evaluator(ti={1: 0.0, 2: 99.0})
    assert e.evaluate(parse("IF (ti[1]=5) THEN (ti[2]) ELSE (0)")) == 0.0


def test_eval_real_formula_negation() -> None:
    """`-tm[1239, "I"]` — типичная формула P_to из tatarstan."""
    e = make_evaluator(tm={(1239, "I"): 5.823})
    assert e.evaluate(parse('-tm[1239, "I"]')) == pytest.approx(-5.823)


def test_eval_real_formula_aggregate_pn() -> None:
    """`tm[1]+tm[2]+tm[3]` — агрегат фидерных нагрузок (типично для PN)."""
    e = make_evaluator(tm={(1, "I"): 10.0, (2, "I"): 15.0, (3, "I"): 20.0})
    assert e.evaluate(parse('tm[1, "I"] + tm[2, "I"] + tm[3, "I"]')) == 45.0


# ── Dependency collection ────────────────────────────────────────────────────


def test_deps_empty() -> None:
    deps = collect_deps(parse("42"))
    assert deps == Deps(tm=frozenset(), ti=frozenset(), ts=frozenset())


def test_deps_simple() -> None:
    deps = collect_deps(parse('tm[100, "I"] + ti[5] - ts[7]'))
    assert deps.tm == frozenset({(100, "I")})
    assert deps.ti == frozenset({5})
    assert deps.ts == frozenset({7})


def test_deps_in_if() -> None:
    deps = collect_deps(parse('IF (ti[1]=0) THEN (tm[2, "I"]) ELSE (tm[3, "S"])'))
    assert deps.ti == frozenset({1})
    assert deps.tm == frozenset({(2, "I"), (3, "S")})


# ── Numeric code detection ──────────────────────────────────────────────────


def test_is_numeric_code() -> None:
    assert is_numeric_code("0")
    assert is_numeric_code("1")
    assert is_numeric_code("-5")
    assert is_numeric_code("100")
    assert not is_numeric_code("")
    assert not is_numeric_code('tm[1, "I"]')
    assert not is_numeric_code("0.5")  # дробное — формальная константа, не код типа
