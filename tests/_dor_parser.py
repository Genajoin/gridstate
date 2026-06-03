"""Парсер формул дорасчёта (поле ``ti.dor``) для моделей входного формата.

Формулы эталонной SE вычисляют значение ti как функцию от:
    * ``tm[N, "X"]`` — сырая телеметрия (TM-сигнал id=N с кодом качества X)
    * ``ti[K]``      — другой ti с key_Num=K
    * ``ts[K]``      — телесигнализация (статус, 0/1) с id=K
    * арифметика ``+ − ∗ /``
    * сравнения ``= < > <= >= <>``
    * логика ``& |``
    * условные ``IF (cond) THEN (a) ELSE (b)``
    * целые/вещественные константы

Парсер строит AST и предоставляет:
    * :func:`parse` — текст → AST.
    * :meth:`Evaluator.evaluate` — AST + lookup-callbacks → ``float | None``
      (None если хоть одна зависимость не разрешена).
    * :func:`collect_deps` — AST → множество tm/ti/ts ссылок.

Грамматика рекурсивного спуска::

    expr     = or_expr
    or_expr  = and_expr ('|' and_expr)*
    and_expr = comp_expr ('&' comp_expr)*
    comp_expr= sum (('=' | '<>' | '<=' | '>=' | '<' | '>') sum)?
    sum      = product (('+' | '-') product)*
    product  = factor (('*' | '/') factor)*
    factor   = '-' factor | atom
    atom     = number | tm_ref | ti_ref | ts_ref | if_expr | '(' expr ')'
    tm_ref   = 'tm' '[' INT ',' STR ']'
    ti_ref   = 'ti' '[' INT ']'
    ts_ref   = 'ts' '[' INT ']'
    if_expr  = 'IF' '(' expr ')' 'THEN' '(' expr ')' 'ELSE' '(' expr ')'

Замечания:
    * Пробельные символы (включая ``\\n``) игнорируются вне строковых литералов.
    * Сравнения и логика возвращают ``0.0`` или ``1.0`` (как входной формат).
    * Деление на 0 даёт ``None`` (рассматривается как «не разрешено»).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


# ── AST ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Const:
    value: float


@dataclass(frozen=True)
class TMRef:
    tm_id: int
    code: str  # одна буква: I, S, C, ...


@dataclass(frozen=True)
class TIRef:
    ti_id: int


@dataclass(frozen=True)
class TSRef:
    ts_id: int


@dataclass(frozen=True)
class UnaryMinus:
    arg: Node


@dataclass(frozen=True)
class BinOp:
    op: str  # '+', '-', '*', '/'
    left: Node
    right: Node


@dataclass(frozen=True)
class Compare:
    op: str  # '=', '<>', '<', '>', '<=', '>='
    left: Node
    right: Node


@dataclass(frozen=True)
class LogicOp:
    op: str  # '&', '|'
    left: Node
    right: Node


@dataclass(frozen=True)
class IfExpr:
    cond: Node
    then_: Node
    else_: Node


@dataclass(frozen=True)
class UnaryNot:
    arg: Node


@dataclass(frozen=True)
class FuncCall:
    name: str  # 'MAX' | 'MIN'
    args: tuple[Node, ...]


Node = (
    Const
    | TMRef
    | TIRef
    | TSRef
    | UnaryMinus
    | UnaryNot
    | BinOp
    | Compare
    | LogicOp
    | IfExpr
    | FuncCall
)


# ── Lexer ────────────────────────────────────────────────────────────────────

# Порядок важен: длинные операторы перед короткими, ключевые слова перед именами.
_TOKEN_SPEC = [
    ("WS", r"[ \t\r\n]+"),
    ("COMMENT", r"\{[^}]*\}"),
    ("LE", r"<="),
    ("GE", r">="),
    ("NE", r"<>|!="),
    ("EQ", r"="),
    ("LT", r"<"),
    ("GT", r">"),
    ("AND", r"&"),
    ("OR", r"\|"),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("LBRACK", r"\["),
    ("RBRACK", r"\]"),
    ("COMMA", r","),
    ("STRING", r'"[^"]*"'),
    ("NUMBER", r"\d+\.\d+|\.\d+|\d+"),
    ("IF", r"IF\b"),
    ("THEN", r"THEN\b"),
    ("ELSE", r"ELSE\b"),
    ("NOT", r"NOT\b"),
    ("MAX", r"MAX\b"),
    ("MIN", r"MIN\b"),
    ("TM", r"tm\b"),
    ("TI", r"ti\b"),
    ("TS", r"ts\b"),
    ("IDENT", r"[A-Za-z_][A-Za-z_0-9]*"),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


@dataclass
class Token:
    kind: str
    value: str
    pos: int


class ParseError(ValueError):
    """Синтаксическая ошибка парсинга dor-формулы."""


def tokenize(src: str) -> list[Token]:
    out: list[Token] = []
    pos = 0
    while pos < len(src):
        m = _TOKEN_RE.match(src, pos)
        if not m:
            raise ParseError(
                f"Неизвестный символ {src[pos]!r} в позиции {pos}: {src[pos : pos + 30]!r}"
            )
        kind = m.lastgroup
        if kind not in ("WS", "COMMENT"):
            out.append(Token(kind, m.group(), pos))
        pos = m.end()
    return out


# ── Parser ───────────────────────────────────────────────────────────────────


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.idx = 0

    def _peek(self) -> Token | None:
        return self.tokens[self.idx] if self.idx < len(self.tokens) else None

    def _eat(self, kind: str) -> Token:
        t = self._peek()
        if t is None or t.kind != kind:
            got = f"{t.kind}({t.value!r})" if t else "EOF"
            raise ParseError(f"Ожидался {kind}, получен {got} (idx={self.idx})")
        self.idx += 1
        return t

    def _accept(self, *kinds: str) -> Token | None:
        t = self._peek()
        if t is not None and t.kind in kinds:
            self.idx += 1
            return t
        return None

    def parse(self) -> Node:
        node = self._or_expr()
        if self._peek() is not None:
            t = self._peek()
            raise ParseError(f"Лишние токены после выражения: {t.kind}({t.value!r}) idx={self.idx}")
        return node

    def _or_expr(self) -> Node:
        left = self._and_expr()
        while self._accept("OR"):
            right = self._and_expr()
            left = LogicOp("|", left, right)
        return left

    def _and_expr(self) -> Node:
        left = self._comp_expr()
        while self._accept("AND"):
            right = self._comp_expr()
            left = LogicOp("&", left, right)
        return left

    def _comp_expr(self) -> Node:
        left = self._sum()
        t = self._peek()
        if t is not None and t.kind in {"EQ", "NE", "LT", "GT", "LE", "GE"}:
            self.idx += 1
            op_map = {"EQ": "=", "NE": "<>", "LT": "<", "GT": ">", "LE": "<=", "GE": ">="}
            right = self._sum()
            return Compare(op_map[t.kind], left, right)
        return left

    def _sum(self) -> Node:
        left = self._product()
        while True:
            t = self._peek()
            if t is None or t.kind not in {"PLUS", "MINUS"}:
                break
            self.idx += 1
            right = self._product()
            left = BinOp("+" if t.kind == "PLUS" else "-", left, right)
        return left

    def _product(self) -> Node:
        left = self._factor()
        while True:
            t = self._peek()
            if t is None or t.kind not in {"STAR", "SLASH"}:
                break
            self.idx += 1
            right = self._factor()
            left = BinOp("*" if t.kind == "STAR" else "/", left, right)
        return left

    def _factor(self) -> Node:
        if self._accept("MINUS"):
            return UnaryMinus(self._factor())
        if self._accept("NOT"):
            # NOT может опционально иметь скобки: `NOT(expr)` или `NOT expr`.
            return UnaryNot(self._factor())
        return self._atom()

    def _atom(self) -> Node:
        t = self._peek()
        if t is None:
            raise ParseError("Неожиданный конец выражения")
        if t.kind == "NUMBER":
            self.idx += 1
            return Const(float(t.value))
        if t.kind == "TM":
            self.idx += 1
            self._eat("LBRACK")
            num = self._eat("NUMBER")
            self._eat("COMMA")
            s = self._eat("STRING")
            self._eat("RBRACK")
            return TMRef(int(float(num.value)), s.value.strip('"'))
        if t.kind == "TI":
            self.idx += 1
            self._eat("LBRACK")
            num = self._eat("NUMBER")
            self._eat("RBRACK")
            return TIRef(int(float(num.value)))
        if t.kind == "TS":
            self.idx += 1
            self._eat("LBRACK")
            num = self._eat("NUMBER")
            self._eat("RBRACK")
            return TSRef(int(float(num.value)))
        if t.kind == "IF":
            self.idx += 1
            self._eat("LPAREN")
            cond = self._or_expr()
            self._eat("RPAREN")
            self._eat("THEN")
            self._eat("LPAREN")
            then_ = self._or_expr()
            self._eat("RPAREN")
            self._eat("ELSE")
            self._eat("LPAREN")
            else_ = self._or_expr()
            self._eat("RPAREN")
            return IfExpr(cond, then_, else_)
        if t.kind in {"MAX", "MIN"}:
            self.idx += 1
            name = t.value.upper()
            self._eat("LPAREN")
            args = [self._or_expr()]
            while self._accept("COMMA"):
                args.append(self._or_expr())
            self._eat("RPAREN")
            return FuncCall(name, tuple(args))
        if t.kind == "LPAREN":
            self.idx += 1
            inner = self._or_expr()
            self._eat("RPAREN")
            return inner
        raise ParseError(f"Неожиданный токен {t.kind}({t.value!r}) idx={self.idx}")


def parse(src: str) -> Node:
    """Распарсить текст dor-формулы в AST.

    Raises:
        ParseError: при синтаксической ошибке.
    """
    return _Parser(tokenize(src)).parse()


# ── Evaluator ────────────────────────────────────────────────────────────────

TmLookup = Callable[[int, str], "float | None"]
TiLookup = Callable[[int], "float | None"]
TsLookup = Callable[[int], "float | None"]


class Evaluator:
    """Вычисление AST-формулы при заданных lookup-функциях.

    Возвращает ``float`` если все зависимости разрешены, ``None`` иначе.
    Деление на 0 трактуется как «не разрешено» (None).
    """

    def __init__(
        self,
        tm: TmLookup,
        ti: TiLookup,
        ts: TsLookup,
    ) -> None:
        self.tm = tm
        self.ti = ti
        self.ts = ts

    def evaluate(self, node: Node) -> float | None:
        if isinstance(node, Const):
            return node.value
        if isinstance(node, TMRef):
            return self.tm(node.tm_id, node.code)
        if isinstance(node, TIRef):
            return self.ti(node.ti_id)
        if isinstance(node, TSRef):
            return self.ts(node.ts_id)
        if isinstance(node, UnaryMinus):
            v = self.evaluate(node.arg)
            return None if v is None else -v
        if isinstance(node, UnaryNot):
            v = self.evaluate(node.arg)
            return None if v is None else (1.0 if v == 0.0 else 0.0)
        if isinstance(node, FuncCall):
            vals = [self.evaluate(a) for a in node.args]
            if any(v is None for v in vals):
                return None
            if node.name == "MAX":
                return max(vals)
            if node.name == "MIN":
                return min(vals)
            return None  # неизвестная функция → не разрешено
        if isinstance(node, BinOp):
            a = self.evaluate(node.left)
            if a is None:
                return None
            b = self.evaluate(node.right)
            if b is None:
                return None
            if node.op == "+":
                return a + b
            if node.op == "-":
                return a - b
            if node.op == "*":
                return a * b
            if node.op == "/":
                if b == 0.0:
                    return None
                return a / b
            raise AssertionError(f"unknown BinOp {node.op}")
        if isinstance(node, Compare):
            a = self.evaluate(node.left)
            if a is None:
                return None
            b = self.evaluate(node.right)
            if b is None:
                return None
            if node.op == "=":
                return 1.0 if a == b else 0.0
            if node.op == "<>":
                return 1.0 if a != b else 0.0
            if node.op == "<":
                return 1.0 if a < b else 0.0
            if node.op == ">":
                return 1.0 if a > b else 0.0
            if node.op == "<=":
                return 1.0 if a <= b else 0.0
            if node.op == ">=":
                return 1.0 if a >= b else 0.0
            raise AssertionError(f"unknown Compare {node.op}")
        if isinstance(node, LogicOp):
            a = self.evaluate(node.left)
            if a is None:
                return None
            # short-circuit: если левый конец задаёт результат — правый не нужен
            if node.op == "&" and a == 0.0:
                return 0.0
            if node.op == "|" and a != 0.0:
                return 1.0
            b = self.evaluate(node.right)
            if b is None:
                return None
            if node.op == "&":
                return 1.0 if (a != 0.0 and b != 0.0) else 0.0
            if node.op == "|":
                return 1.0 if (a != 0.0 or b != 0.0) else 0.0
            raise AssertionError(f"unknown LogicOp {node.op}")
        if isinstance(node, IfExpr):
            c = self.evaluate(node.cond)
            if c is None:
                return None
            return self.evaluate(node.then_ if c != 0.0 else node.else_)
        raise AssertionError(f"unknown node {type(node).__name__}")


# ── Dependency collection ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Deps:
    tm: frozenset[tuple[int, str]]
    ti: frozenset[int]
    ts: frozenset[int]


def collect_deps(node: Node) -> Deps:
    """Собрать множество tm/ti/ts ссылок, от которых зависит формула."""
    tm: set[tuple[int, str]] = set()
    ti: set[int] = set()
    ts: set[int] = set()

    def walk(n: Node) -> None:
        if isinstance(n, TMRef):
            tm.add((n.tm_id, n.code))
        elif isinstance(n, TIRef):
            ti.add(n.ti_id)
        elif isinstance(n, TSRef):
            ts.add(n.ts_id)
        elif isinstance(n, (UnaryMinus, UnaryNot)):
            walk(n.arg)
        elif isinstance(n, (BinOp, Compare, LogicOp)):
            walk(n.left)
            walk(n.right)
        elif isinstance(n, IfExpr):
            walk(n.cond)
            walk(n.then_)
            walk(n.else_)
        elif isinstance(n, FuncCall):
            for a in n.args:
                walk(a)
        # Const: нет зависимостей

    walk(node)
    return Deps(tm=frozenset(tm), ti=frozenset(ti), ts=frozenset(ts))


def is_numeric_code(src: str) -> bool:
    """Проверить, что строка — числовой код типа дорасчёта (`'0'`, `'1'`, ...).

    Такие записи — это **не формулы**, а классификатор типа дорасчёта во входном формате.
    Их следует пропускать при evaluation.
    """
    s = src.strip()
    if not s:
        return False
    return bool(re.fullmatch(r"-?\d+", s))
