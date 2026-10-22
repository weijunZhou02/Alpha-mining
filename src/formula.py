"""Formula reader: turns a formula string into numbers.

    parse("rank(ts_mean(close, 20))")  ->  a small tree of operations
    evaluate(tree, context, dialect)   ->  a wide DataFrame (date x coin)

Three dialects (the catalog's `notation` column):
    wq101   - 101 Formulaic Alphas  (rank, ts_rank, decay_linear, a ? b : c, ...)
    gtja191 - GTJA 191 report       (RANK, TSRANK, SMA(A,n,m), COUNT, SUMIF, REGBETA, ...)
    qlib    - Qlib Alpha158         ($close, Ref, Mean, IdxMax, Greater, ...)
"""
import re

import numpy as np
import pandas as pd

import operators as op

# ------------------------------------------------------------------ tokenizer

_TOKEN = re.compile(r"""
    (?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>[A-Za-z_$][A-Za-z0-9_.]*)
  | (?P<op>\|\||&&|==|!=|<=|>=|[?:<>=+\-*/^(),&|])
  | (?P<space>\s+)
""", re.X)


def tokenize(text):
    text = text.replace("./", "/").replace(".*", "*")  # GTJA matlab-style element-wise operators
    tokens, pos = [], 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise SyntaxError(f"cannot read {text[pos:pos + 20]!r}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "space":
            continue
        val = m.group()
        if kind == "name" and val.upper() == "OR":
            kind, val = "op", "||"
        elif kind == "name" and val.upper() == "AND":
            kind, val = "op", "&&"
        if kind == "op":
            val = {"=": "==", "&": "&&", "|": "||"}.get(val, val)
        tokens.append((kind, val))
    return tokens


# ------------------------------------------------------------------ parser (precedence climbing)

_BINARY = {"||": 2, "&&": 3, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4, "!=": 4,
           "+": 5, "-": 5, "*": 6, "/": 6, "^": 8}
_TERNARY_BP, _UNARY_BP = 1, 7


class _Parser:
    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.peek()
        if val is not None and tok[1] != val:
            raise SyntaxError(f"expected {val!r}, got {tok[1]!r} at token {self.i}")
        self.i += 1
        return tok

    def expr(self, min_bp=0):
        left = self.prefix()
        while True:
            kind, val = self.peek()
            if val == "?" and _TERNARY_BP >= min_bp:
                self.take("?")
                a = self.expr(0)
                self.take(":")
                b = self.expr(_TERNARY_BP)
                left = ("tern", left, a, b)
                continue
            bp = _BINARY.get(val) if kind == "op" else None
            if bp is None or bp < min_bp:
                return left
            self.take()
            right = self.expr(bp if val == "^" else bp + 1)  # ^ is right-associative
            left = ("bin", val, left, right)

    def prefix(self):
        kind, val = self.take()
        if kind == "num":
            return ("num", float(val))
        if val == "-":
            return ("neg", self.expr(_UNARY_BP))
        if val == "+":
            return self.expr(_UNARY_BP)
        if val == "(":
            e = self.expr(0)
            self.take(")")
            return e
        if kind == "name":
            if self.peek()[1] == "(":
                self.take("(")
                args = []
                if self.peek()[1] != ")":
                    args.append(self.expr(0))
                    while self.peek()[1] == ",":
                        self.take(",")
                        args.append(self.expr(0))
                self.take(")")
                return ("call", val, args)
            return ("var", val)
        raise SyntaxError(f"unexpected {val!r}")


def parse(text):
    p = _Parser(tokenize(text.strip().rstrip(";")))
    tree = p.expr(0)
    if p.i != len(p.t):
        raise SyntaxError(f"extra text after position {p.i}: {p.t[p.i:p.i + 5]}")
    return tree


# ------------------------------------------------------------------ evaluation

def _is_df(v):
    return isinstance(v, pd.DataFrame)


def _valid(a, b):
    return (a.notna() if _is_df(a) else True) & (b.notna() if _is_df(b) else True)


def _cmp(a, b, fn):
    """Comparisons give 1.0 / 0.0, and NaN when an input is missing (101 Alphas: true = 1)."""
    if not _is_df(a) and not _is_df(b):
        return float(fn(a, b))
    return fn(a, b).astype(float).where(_valid(a, b))


def _logic(a, b, both):
    if not _is_df(a) and not _is_df(b):
        return float((a > 0 and b > 0) if both else (a > 0 or b > 0))
    pa, pb = a > 0, b > 0
    out = (pa & pb) if both else (pa | pb)
    return out.astype(float).where(_valid(a, b))


def _ternary(c, a, b):
    if not _is_df(c):
        return a if c > 0 else b
    ref = c
    a = a if _is_df(a) else pd.DataFrame(a, index=ref.index, columns=ref.columns)
    b = b if _is_df(b) else pd.DataFrame(b, index=ref.index, columns=ref.columns)
    return a.where(c > 0, b).where(c.notna())


_BIN_FN = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: op._clean(a / b) if (_is_df(a) or _is_df(b)) else (a / b if b != 0 else np.nan),
    "^": op.power,
    "<": lambda a, b: _cmp(a, b, lambda p, q: p < q),
    ">": lambda a, b: _cmp(a, b, lambda p, q: p > q),
    "<=": lambda a, b: _cmp(a, b, lambda p, q: p <= q),
    ">=": lambda a, b: _cmp(a, b, lambda p, q: p >= q),
    "==": lambda a, b: _cmp(a, b, lambda p, q: p == q),
    "!=": lambda a, b: _cmp(a, b, lambda p, q: p != q),
    "&&": lambda a, b: _logic(a, b, True),
    "||": lambda a, b: _logic(a, b, False),
}


class Sequence:
    """GTJA SEQUENCE(n): the numbers 1..n, i.e. 'time' for a trend regression over n days."""

    def __init__(self, n):
        self.n = int(n)


def _num(v):
    if _is_df(v):
        raise TypeError("a window length must be a number, got a table")
    return float(v)


def _max_or_tsmax(args, ctx, ts_fn, pair_fn):
    """MAX(A, B): element-wise; MAX(A, n) with a number n >= 2: rolling max over n days."""
    a, b = args
    if _is_df(a) and not _is_df(b) and b >= 2:
        return ts_fn(a, b)
    return pair_fn(a, b)


def _regbeta(args, ctx):
    y, x = args[0], args[1]
    if isinstance(x, Sequence):
        n = int(args[2]) if len(args) > 2 else x.n
        return op.ts_regbeta(y, op.time_index(y), n)
    return op.ts_regbeta(y, x, _num(args[2]))


def _regresi(args, ctx):
    y, x = args[0], args[1]
    if isinstance(x, Sequence):
        return op.ts_regresi(y, op.time_index(y), x.n)
    return op.ts_regresi(y, x, _num(args[2]))


def _delay(args, ctx):
    return op.delay(args[0], _num(args[1]) if len(args) > 1 else 1)


def _delta(args, ctx):
    return op.delta(args[0], _num(args[1]) if len(args) > 1 else 1)


# name (lower case) -> function(args, ctx)
COMMON = {
    "abs": lambda a, c: np.abs(a[0]),
    "sign": lambda a, c: op.sign(a[0]),
    "log": lambda a, c: op.log(a[0]),
    "delay": _delay,
    "delta": _delta,
}

WQ101 = {
    **COMMON,
    "rank": lambda a, c: op.cs_rank(a[0], c.mask),
    "scale": lambda a, c: op.cs_scale(a[0], c.mask, _num(a[1]) if len(a) > 1 else 1.0),
    "indneutralize": lambda a, c: op.cs_demean_by_group(a[0], c.groups, c.mask),
    "correlation": lambda a, c: op.ts_corr(a[0], a[1], _num(a[2])),
    "covariance": lambda a, c: op.ts_cov(a[0], a[1], _num(a[2])),
    "sum": lambda a, c: op.ts_sum(a[0], _num(a[1])),
    "product": lambda a, c: op.ts_product(a[0], _num(a[1])),
    "stddev": lambda a, c: op.ts_std(a[0], _num(a[1])),
    "ts_min": lambda a, c: op.ts_min(a[0], _num(a[1])),
    "ts_max": lambda a, c: op.ts_max(a[0], _num(a[1])),
    "ts_argmax": lambda a, c: op.ts_argmax(a[0], _num(a[1])),
    "ts_argmin": lambda a, c: op.ts_argmin(a[0], _num(a[1])),
    "ts_rank": lambda a, c: op.ts_rank(a[0], _num(a[1])),
    "decay_linear": lambda a, c: op.decay_linear(a[0], _num(a[1])),
    "signedpower": lambda a, c: op.signed_power(a[0], a[1]),
    "min": lambda a, c: _max_or_tsmax(a, c, op.ts_min, op.min2),
    "max": lambda a, c: _max_or_tsmax(a, c, op.ts_max, op.max2),
}

GTJA = {
    **COMMON,
    "rank": lambda a, c: op.cs_rank(a[0], c.mask),
    "sum": lambda a, c: op.ts_sum(a[0], _num(a[1])),
    "mean": lambda a, c: op.ts_mean(a[0], _num(a[1])),
    "ma": lambda a, c: op.ts_mean(a[0], _num(a[1])),  # report typo for MEAN
    "delat": _delta,  # report typo for DELTA
    "std": lambda a, c: op.ts_std(a[0], _num(a[1])),
    "corr": lambda a, c: op.ts_corr(a[0], a[1], _num(a[2])),
    "coviance": lambda a, c: op.ts_cov(a[0], a[1], _num(a[2])),
    "covariance": lambda a, c: op.ts_cov(a[0], a[1], _num(a[2])),
    "tsrank": lambda a, c: op.ts_rank(a[0], _num(a[1])),
    "tsmax": lambda a, c: op.ts_max(a[0], _num(a[1])),
    "tsmin": lambda a, c: op.ts_min(a[0], _num(a[1])),
    "decaylinear": lambda a, c: op.decay_linear(a[0], _num(a[1])),
    "sma": lambda a, c: op.sma_gtja(a[0], _num(a[1]), _num(a[2])),
    "wma": lambda a, c: op.wma(a[0], _num(a[1])),
    "count": lambda a, c: op.ts_count(a[0], _num(a[1])),
    "sumif": lambda a, c: op.ts_sumif(a[0], _num(a[1]), a[2]),
    "prod": lambda a, c: op.ts_product(a[0], _num(a[1])),
    "highday": lambda a, c: op.ts_highday(a[0], _num(a[1])),
    "lowday": lambda a, c: op.ts_lowday(a[0], _num(a[1])),
    "sequence": lambda a, c: Sequence(_num(a[0])),
    "regbeta": _regbeta,
    "regresi": _regresi,
    "min": lambda a, c: _max_or_tsmax(a, c, op.ts_min, op.min2),
    "max": lambda a, c: _max_or_tsmax(a, c, op.ts_max, op.max2),
    # helpers used only by the corrected versions of garbled report formulas
    "sumacrange": lambda a, c: op.ts_sumac_range(a[0], _num(a[1])),
    "tsskew": lambda a, c: op.ts_skew(a[0], _num(a[1])),
    "regbetaif": lambda a, c: op.ts_regbeta(a[0].where(a[2] > 0), a[1].where(a[2] > 0), _num(a[3]), min_count=20),
}

QLIB = {
    "ref": lambda a, c: op.delay(a[0], _num(a[1])),
    "mean": lambda a, c: op.ts_mean(a[0], _num(a[1])),
    "std": lambda a, c: op.ts_std(a[0], _num(a[1])),
    "sum": lambda a, c: op.ts_sum(a[0], _num(a[1])),
    "slope": lambda a, c: op.ts_slope(a[0], _num(a[1])),
    "rsquare": lambda a, c: op.ts_rsquare(a[0], _num(a[1])),
    "resi": lambda a, c: op.ts_resi(a[0], _num(a[1])),
    "max": lambda a, c: op.ts_max(a[0], _num(a[1])),
    "min": lambda a, c: op.ts_min(a[0], _num(a[1])),
    "quantile": lambda a, c: op.ts_quantile(a[0], _num(a[1]), _num(a[2])),
    "rank": lambda a, c: op.ts_rank(a[0], _num(a[1])),
    "idxmax": lambda a, c: op.ts_argmax(a[0], _num(a[1])) + 1,  # Qlib counts 1..d (oldest = 1)
    "idxmin": lambda a, c: op.ts_argmin(a[0], _num(a[1])) + 1,
    "corr": lambda a, c: op.ts_corr(a[0], a[1], _num(a[2])),
    "log": lambda a, c: op.log(a[0]),
    "abs": lambda a, c: np.abs(a[0]),
    "greater": lambda a, c: op.max2(a[0], a[1]),
    "less": lambda a, c: op.min2(a[0], a[1]),
}

DIALECTS = {"wq101": WQ101, "gtja191": GTJA, "qlib": QLIB}


class Context:
    """What a formula can see: data tables (variables), universe mask, sector groups."""

    def __init__(self, variables, mask, groups=None):
        self.variables = variables  # name (lower case) -> DataFrame, or a callable(name) for patterns
        self.mask = mask
        self.groups = groups

    def var(self, name):
        key = name.lower()
        if key in self.variables:
            v = self.variables[key]
            return v() if callable(v) else v
        if "*" in self.variables:  # pattern resolver, e.g. adv20
            v = self.variables["*"](key)
            if v is not None:
                return v
        raise NameError(f"unknown variable {name!r}")


def evaluate(tree, ctx, dialect):
    funcs = DIALECTS[dialect]

    def ev(node):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return ctx.var(node[1])
        if kind == "neg":
            return -ev(node[1])
        if kind == "bin":
            return _BIN_FN[node[1]](ev(node[2]), ev(node[3]))
        if kind == "tern":
            return _ternary(ev(node[1]), ev(node[2]), ev(node[3]))
        if kind == "call":
            name = node[1].lower()
            if name not in funcs:
                raise NameError(f"unknown function {node[1]!r} in {dialect}")
            return funcs[name]([ev(a) for a in node[2]], ctx)
        raise ValueError(node)

    return ev(tree)
