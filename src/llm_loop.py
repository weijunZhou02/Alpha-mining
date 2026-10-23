"""Step 7: an LLM (Claude) invents new alphas; our code checks, scores and keeps or rejects them.

    .venv/bin/python src/llm_loop.py --fake --rounds 2                 # free test with ready-made formulas
    .venv/bin/python src/llm_loop.py --rounds 40 --per-round 5         # a real run (model from .env)

The API key and model come from .env (ANTHROPIC_API_KEY, LLM_MODEL). The key is never printed or logged.

Years. Everything here stays inside the 2021-2023 training period: the data is cut at 2023-12-31
before anything is computed, so 2024 and 2025+ cannot be read.
    search   2021-2022  each formula's own scores; the LLM sees them
    screen   2023 H1    gate: does a ridge blend of the broad list do better with it? The LLM sees the numbers
    confirm  2023 H2    gate: does the trees blend (the step 6 method, averaged over 3 seeds) earn at least as
                        much? The LLM only sees pass / fail
    Both gates compare "blend + candidate" with "blend + a dummy column of random numbers", so that adding
    a column by itself does not count as a gain.
Gates, in order: safety -> compute + cut-off test -> coverage -> own signal (search) -> not a twin
                 -> screen -> confirm -> kept (it joins both blends, so later tries must add to it)
Luck check at the start of each run: 20 pure-noise alphas and 20 random formulas go through the same
screen and confirm gates. About none should pass.
Records: data/llm/tries.jsonl (every try), kept.csv, runs.csv, luck_check.csv; raw prompts and replies in
logs/llm/ (not in git).
"""
import argparse
import json
import math
import signal
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import combine as cb
import evaluate as ev
import formula as fx
from alphas import AlphaEngine, _broadcast, load_catalog
from config import DATA, ROOT

OUT = DATA / "llm"
LOGS = ROOT / "logs" / "llm"
END = ev.PERIODS["train"][1]                      # 2023-12-31: nothing later is ever read
GAP = cb.GAP                                      # training rows end 2 days before the period they predict

MAX_CHARS, MAX_NODES, WINDOW_MAX = 300, 60, 250
CUTOFF_DATES = ("2021-11-15", "2022-08-15")
COMPUTE_SECONDS = 60
MIN_COVERAGE = 0.8                                # share of universe coin-days with a value
MIN_VARIED = 0.8                                  # share of days where coins get different values
SIGNAL_T = 2.0                                    # own signal: |rank t| >= 2 or money t >= 2 (2021-22)
TWIN = 0.7                                        # similarity at or above this = twin (as in step 5)
NEAR_REPEAT = 0.98                                # almost the same as an earlier try in this run
SCREEN_NET_BP, SCREEN_IC = 1.0, 0.001             # ridge blend: +1 bp/day after costs AND rank IC +0.001
CONFIRM_NET_BP = 0.0                              # trees blend: money after costs not lower
# Gains are measured against the blend plus a DUMMY column of random numbers, not against the bare blend:
# the free luck check showed that adding any column moves the ridge blend's money up (+1.5 bp/day on
# average) and reshuffles the trees blend (-5 bp/day). Trees are also averaged over 3 seeds.
RIDGE_DUMMIES, TREE_DUMMIES, TREE_SEEDS = 20, 4, (0, 1, 2)
MAX_INDNEUTRALIZE = 2                             # per round
HISTORY = 15                                      # latest tries shown to the LLM
LUCK_N = 20
RIDGE_PENALTIES = cb.RIDGE_PENALTIES
TREE_LEAVES, TREE_MAX_ITER, TREE_STEP = cb.TREE_LEAVES, cb.TREE_MAX_ITER, cb.TREE_STEP

# USD per million tokens: input, 5-minute cache write, cache read, output
PRICES = {
    "claude-haiku-5-5": (0.10, 0.125, 0.01, 0.50),
    "claude-sonnet-5-5": (2.0, 2.50, 0.10, 10.0),
    "claude-opus-5-5": (4.0, 5.0, 0.20, 20.0),
    "claude-fable-5-1": (10.0, 12.50, 0.25, 50.0),
    "fake": (0, 0, 0, 0),
}

THEMES = [
    "funding rate: its level, change, extremes or persistence",
    "perp premium: its level, change, or how it differs from funding",
    "order flow: taker_buy_ratio (who is aggressive, and is it changing?)",
    "trade size: volume per trade (volume / trades), and changes in it",
    "links to the market: moves against mkt_returns or btc_returns (beta, catching up, decoupling)",
    "shape of moves: up-days vs down-days, range (high - low), gaps, lopsided volatility",
    "slow signals (windows of 20 to 120 days) that change little from day to day and so trade cheaply",
    "liquidity changes: volume, adv, price impact (abs(returns) / volume)",
    "price position: close against its recent high, low or vwap",
    "mix two different families or data fields into one idea",
    "an idea from crypto research or market practice not covered by the existing list",
    "free choice: the idea you think is most likely to be new and to survive fees",
]

# Allowed functions: name -> (number of arguments, {argument index: smallest allowed window})
FUNCS = {
    "abs": (1, {}), "sign": (1, {}), "log": (1, {}), "rank": (1, {}), "scale": (1, {}),
    "indneutralize": (2, {}), "signedpower": (2, {}), "min": (2, {}), "max": (2, {}),
    "delay": (2, {1: 1}), "delta": (2, {1: 1}),
    "correlation": (3, {2: 3}), "covariance": (3, {2: 3}),
    "sum": (2, {1: 2}), "product": (2, {1: 2}), "stddev": (2, {1: 3}),
    "ts_min": (2, {1: 2}), "ts_max": (2, {1: 2}), "ts_argmax": (2, {1: 2}), "ts_argmin": (2, {1: 2}),
    "ts_rank": (2, {1: 3}), "decay_linear": (2, {1: 2}),
}
FIELDS = {"open", "high", "low", "close", "vwap", "volume", "returns", "funding", "premium", "trades",
          "taker_buy_ratio", "mkt_returns", "btc_returns"}


# ------------------------------------------------------------------ safety

def _nodes(t):
    kind = t[0]
    if kind in ("num", "var"):
        return 1
    if kind == "neg":
        return 1 + _nodes(t[1])
    if kind == "bin":
        return 1 + _nodes(t[2]) + _nodes(t[3])
    if kind == "tern":
        return 1 + sum(_nodes(x) for x in t[1:])
    return 1 + sum(_nodes(a) for a in t[2])


def _uses(t, name):
    if t[0] == "call":
        return t[1].lower() == name or any(_uses(a, name) for a in t[2])
    if t[0] in ("bin",):
        return _uses(t[2], name) or _uses(t[3], name)
    if t[0] == "tern":
        return any(_uses(x, name) for x in t[1:])
    if t[0] == "neg":
        return _uses(t[1], name)
    return False


def _check(t, parent=None, pos=None):
    """Return a reason string if the tree breaks a rule, else None."""
    kind = t[0]
    if kind == "num":
        return None
    if kind == "var":
        name = t[1].lower()
        if name.startswith("indclass."):
            return None if parent == "indneutralize" and pos == 1 else "IndClass.sector may only appear inside indneutralize(x, IndClass.sector)"
        if name in FIELDS:
            return None
        if name.startswith("adv") and name[3:].isdigit() and 2 <= int(name[3:]) <= WINDOW_MAX:
            return None
        return f"unknown field {t[1]!r}"
    if kind == "neg":
        return _check(t[1])
    if kind == "bin":
        return _check(t[2]) or _check(t[3])
    if kind == "tern":
        return _check(t[1]) or _check(t[2]) or _check(t[3])
    name, args = t[1].lower(), t[2]
    if name not in FUNCS:
        return f"unknown function {t[1]!r}"
    n_args, windows = FUNCS[name]
    if len(args) != n_args:
        return f"{name} takes {n_args} argument(s), got {len(args)}"
    for i, a in enumerate(args):
        if i in windows:
            if a[0] != "num" or a[1] != int(a[1]):
                return f"the window of {name} must be a plain whole number"
            if not windows[i] <= a[1] <= WINDOW_MAX:
                return f"the window of {name} must be from {windows[i]} to {WINDOW_MAX}"
            continue
        if name in ("min", "max") and a[0] == "num" and abs(a[1]) > 1:
            return f"{name}(a, b) is element-wise: a number in it must be between -1 and 1 (use ts_{name} for windows)"
        if name == "indneutralize" and i == 1 and not (a[0] == "var" and a[1].lower().startswith("indclass.")):
            return "indneutralize needs IndClass.sector as its second argument"
        why = _check(a, name, i)
        if why:
            return why
    return None


def _strip_sign(t):
    while True:
        if t[0] == "neg":
            t = t[1]
        elif t[0] == "bin" and t[1] == "*" and t[2] in (("num", -1.0), ("neg", ("num", 1.0))):
            t = t[3]
        elif t[0] == "bin" and t[1] == "*" and t[3] in (("num", -1.0), ("neg", ("num", 1.0))):
            t = t[2]
        else:
            return t


def canonical(t):
    """One spelling per formula (case, spaces and an outer sign removed): finds exact repeats."""
    def c(t):
        kind = t[0]
        if kind == "num":
            return repr(round(t[1], 8))
        if kind == "var":
            return t[1].lower()
        if kind == "neg":
            return f"-({c(t[1])})"
        if kind == "bin":
            return f"({c(t[2])}{t[1]}{c(t[3])})"
        if kind == "tern":
            return f"({c(t[1])}?{c(t[2])}:{c(t[3])})"
        return f"{t[1].lower()}({','.join(c(a) for a in t[2])})"
    return c(_strip_sign(t))


def check_formula(text):
    """(tree, None) if the formula may be computed, else (None, reason)."""
    if not isinstance(text, str) or not text.strip():
        return None, "empty formula"
    if len(text) > MAX_CHARS:
        return None, f"longer than {MAX_CHARS} characters"
    try:
        tree = fx.parse(text)
    except Exception as e:  # noqa: BLE001 - any parse failure is a refusal
        return None, f"syntax error: {e}"
    if _nodes(tree) > MAX_NODES:
        return None, f"more than {MAX_NODES} parts"
    why = _check(tree)
    return (None, why) if why else (tree, None)


# ------------------------------------------------------------------ computing

class Timeout(Exception):
    pass


def with_timeout(seconds, fn, *args):
    if not hasattr(signal, "SIGALRM"):     # Windows: no alarm signal; the size limits keep formulas fast
        return fn(*args)

    def stop(signum, frame):
        raise Timeout()
    old = signal.signal(signal.SIGALRM, stop)
    signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def cut_engine(eng, last):
    """The same engine with every table cut after `last`."""
    t = pd.Timestamp(last)
    return AlphaEngine({k: v.loc[:t] for k, v in eng.p.items()}, eng.universe.loc[:t], eng.sectors, eng.onboard)


def make_context(eng):
    """Formula variables: the 101-Alphas set plus the crypto fields."""
    ctx = eng.context("wq101")
    p = eng.p
    qv = p["quote_volume"]
    ctx.variables.update({
        "funding": p["funding"], "premium": p["premium"], "trades": p["trades"],
        "taker_buy_ratio": p["taker_buy_quote_volume"] / qv.where(qv > 0),
        "mkt_returns": _broadcast(eng.market_return(), p["close"]),
        "btc_returns": eng.bench("returns"),
    })
    return ctx


def compute(tree, ctx, universe):
    out = fx.evaluate(tree, ctx, "wq101")
    if not isinstance(out, pd.DataFrame):
        raise ValueError("the formula gives one number, not a value per coin")
    out = out.astype(float).replace([np.inf, -np.inf], np.nan)
    return out.where(universe)


def peeks_ahead(tree, full, cut_ctxs):
    """True if values up to a cut-off date change when later data is removed (look-ahead)."""
    for last, (ctx, uni) in cut_ctxs.items():
        t = pd.Timestamp(last)
        part = compute(tree, ctx, uni)
        a = full.loc[t - pd.Timedelta(days=10):t]
        b = part.reindex(index=a.index, columns=a.columns)
        if not np.isclose(a.to_numpy(), b.to_numpy(), rtol=1e-9, atol=1e-12, equal_nan=True).all():
            return True
    return False


# ------------------------------------------------------------------ the lab: data, gates and blends

def _std_cols(M):
    M = M - M.mean(axis=0)
    n = np.sqrt((M ** 2).sum(axis=0))
    n[n == 0] = 1
    return M / n


class Lab:
    def __init__(self, verbose=True):
        t0 = time.time()
        full = AlphaEngine.load()
        self.eng = cut_engine(full, END)                       # nothing after 2023 exists from here on
        del full
        self.ctx = make_context(self.eng)
        self.cut_ctxs = {}
        for d in CUTOFF_DATES:
            e = cut_engine(self.eng, d)
            self.cut_ctxs[d] = (make_context(e), e.universe)
        self.m = {k: ev.Market(k) for k in ("train", "search", "screen", "confirm")}
        self.rows = cb.Rows(self.m["train"].universe)
        self.date = self.rows.date
        self.y = self.rows.to_long(cb.centred_rank(self.m["train"].fwd[1], self.m["train"].universe))
        self.broad = pd.read_csv(ev_path("dedupe", "broad_list.csv"), index_col=0).index.tolist()
        X = pd.read_parquet(ev_path("combine", "features.parquet"))
        X = X[X.index.get_level_values(0) <= pd.Timestamp(END)]
        assert X.index.equals(self.rows.index), "features.parquet does not line up with the 2021-2023 universe"
        self.cols = {a: X[a].to_numpy(np.float32) for a in self.broad}
        net = pd.read_parquet(ev_path("daily_ls_net.parquet"))
        s = self.m["search"]
        self.nets = net[self.broad].loc[s.start:s.end].copy()     # date x alpha, daily net (step 3 directions)
        self.search_rows = self.date <= pd.Timestamp(s.end)
        self.kept = {}                                          # id -> vector
        self.run_vectors = []                                   # this run's tries, for near-repeats
        self._refresh_similarity()
        self.screen = RidgeScreen(self)
        self.confirm = TreesConfirm(self)
        if verbose:
            print(f"lab ready in {time.time() - t0:.0f}s: {len(self.broad)} broad alphas, rows {len(self.y):,}, "
                  f"data until {self.eng.universe.index[-1].date()}", flush=True)
            print(f"  screen base (ridge, 2023 H1): rank IC {self.screen.base['ic']:.4f}, net {self.screen.base['net_bp']:+.1f} bp/day, "
                  f"penalty {self.screen.penalty}; with dummy columns: IC {self.screen.ref['ic']:.4f}, net {self.screen.ref['net_bp']:+.1f}", flush=True)
            print(f"  confirm base (trees, 2023 H2): settled (numbers not shown to the LLM)", flush=True)

    def _refresh_similarity(self):
        names = self.broad + list(self.kept)
        M = np.column_stack([self.all_cols()[a][self.search_rows] for a in names]).astype(np.float64)
        self.sim_names, self.sim_M = names, _std_cols(M)

    def all_cols(self):
        return {**self.cols, **self.kept}

    def vector(self, alpha, direction):
        return np.nan_to_num(self.rows.to_long(cb.centred_rank(alpha.loc[:END], self.m["train"].universe) * direction)).astype(np.float32)

    # ---------------------------------------------------------- one candidate through every gate

    def test(self, text, luck=False):
        """Returns a record dict. stage = where it stopped: refused, compute, lookahead, coverage, weak, twin,
        screen, confirm or kept."""
        rec = {"formula": text}
        tree, why = check_formula(text)
        if why:
            return {**rec, "stage": "refused", "reason": why}
        rec["canonical"] = canonical(tree)
        t0 = time.time()
        try:
            alpha = with_timeout(COMPUTE_SECONDS, compute, tree, self.ctx, self.eng.universe)
            if with_timeout(COMPUTE_SECONDS, peeks_ahead, tree, alpha, self.cut_ctxs):
                return {**rec, "stage": "lookahead", "reason": "values changed when later data was removed"}
        except Timeout:
            return {**rec, "stage": "compute", "reason": f"took longer than {COMPUTE_SECONDS}s"}
        except Exception as e:  # noqa: BLE001
            return {**rec, "stage": "compute", "reason": f"could not compute: {type(e).__name__}: {str(e)[:120]}"}
        return self.test_values(alpha, rec, luck, t0)

    def test_values(self, alpha, rec, luck=False, t0=None):
        t0 = t0 or time.time()
        U = self.m["train"].universe
        a = alpha.reindex(index=U.index, columns=U.columns)
        has = a.notna() & U
        rec["coverage"] = round(float(has.sum().sum() / U.sum().sum()), 3)
        varied = (a.where(U).std(axis=1) > 0) & (has.sum(axis=1) >= ev.MIN_COINS)
        rec["varied"] = round(float(varied.mean()), 3)
        if rec["coverage"] < MIN_COVERAGE or rec["varied"] < MIN_VARIED:
            return {**rec, "stage": "coverage", "reason": f"values on {rec['coverage']:.0%} of coin-days, varied on {rec['varied']:.0%} of days (need 80%)"}
        s = self.m["search"]
        card, series = ev.score(a.loc[s.start:s.end], s)
        d = card["direction"]
        rec.update(direction=d, ic=card["ic"], ic_t=card["ic_t"], ls_bp=card["ls_bp"], ls_t=card["ls_t"],
                   net_bp=card["net_bp"], turnover=card["turnover"], verdict=card["verdict"])
        v = self.vector(alpha, d)
        if not luck:
            if not (abs(card["ic_t"]) >= SIGNAL_T or card["ls_t"] >= SIGNAL_T):
                return self._done(rec, "weak", f"own signal too weak in 2021-22 (rank t {card['ic_t']:+.1f}, money t {card['ls_t']:+.1f})", t0)
            twin = self.twin(v, series["net"])
            if twin:
                rec.update(twin_of=twin[0], twin_check=twin[1], twin_sim=round(twin[2], 3))
                return self._done(rec, "twin", f"twin of {twin[0]} ({twin[1]} similarity {twin[2]:.2f})", t0)
        sc = self.screen.test(v)
        rec.update(screen_ic_gain=sc["ic_gain"], screen_net_gain_bp=sc["net_gain_bp"])
        self.run_vectors.append(v[self.search_rows])
        if not (sc["net_gain_bp"] >= SCREEN_NET_BP and sc["ic_gain"] >= SCREEN_IC):
            return self._done(rec, "screen", f"ridge blend 2023 H1 vs a dummy column: rank IC {sc['ic_gain']:+.4f}, net {sc['net_gain_bp']:+.1f} bp/day "
                                             f"(needs IC >= +{SCREEN_IC} and net >= +{SCREEN_NET_BP:.0f})", t0)
        cf = self.confirm.test(v)
        rec.update(confirm_ic_gain=cf["ic_gain"], confirm_net_gain_bp=cf["net_gain_bp"])
        if cf["net_gain_bp"] < CONFIRM_NET_BP:
            return self._done(rec, "confirm", "passed the screen, but the trees blend did not earn more on the check half-year", t0)
        rec["_vector"] = v
        rec["_net"] = series["net"]
        return self._done(rec, "kept", "adds to both blends", t0)

    def _done(self, rec, stage, reason, t0):
        rec.update(stage=stage, reason=reason, seconds=round(time.time() - t0, 1))
        return rec

    def twin(self, v, net):
        z = _std_cols(v[self.search_rows].astype(np.float64)[:, None])[:, 0]
        sig = np.abs(self.sim_M.T @ z)
        i = int(np.argmax(sig))
        hits = []
        if sig[i] >= TWIN:
            hits.append((self.sim_names[i], "signal", float(sig[i])))
        money = self.nets.corrwith(net.reindex(self.nets.index), min_periods=250)
        if money.notna().any():
            j = money.idxmax()
            if money[j] >= TWIN:
                hits.append((j, "money", float(money[j])))
        for old in self.run_vectors:
            zo = _std_cols(old.astype(np.float64)[:, None])[:, 0]
            c = abs(float(z @ zo))
            if c >= NEAR_REPEAT:
                hits.append(("an earlier try in this run", "signal", c))
        return max(hits, key=lambda h: h[2]) if hits else None

    def keep(self, name, rec):
        self.kept[name] = rec["_vector"]
        self.nets[name] = rec["_net"].reindex(self.nets.index)
        self._refresh_similarity()
        self.screen.rebuild()
        self.confirm.rebuild()


def ev_path(*parts):
    return ev.PANEL.parent / "eval" / "/".join(parts)


class RidgeScreen:
    """Ridge blend of broad + kept alphas, learned on 2021-2022, traded on 2023 H1 (step 6 portfolio).
    The penalty is picked for the base blend only, then held fixed while candidates are tested."""

    def __init__(self, lab):
        self.lab = lab
        d = lab.date
        start = pd.Timestamp(lab.m["screen"].start)
        has_y = ~np.isnan(lab.y)
        self.train = (d <= start - GAP) & has_y
        self.inner = (d <= pd.Timestamp("2022-07-01") - GAP) & has_y
        self.hold = (d >= pd.Timestamp("2022-07-01")) & (d <= start - GAP) & has_y
        self.pred = (d >= start) & (d <= pd.Timestamp(lab.m["screen"].end))
        self.rebuild()

    def _X(self, rows):
        cols = self.lab.all_cols()
        return np.column_stack([cols[a][rows] for a in cols]).astype(np.float64)

    def rebuild(self):
        lab = self.lab
        Xi, Xh = self._X(self.inner), self._X(self.hold)
        A, b = Xi.T @ Xi / len(Xi), Xi.T @ lab.y[self.inner] / len(Xi)
        score = {p: cb.holdout_ic(Xh @ cb.ridge_weights(A, b, p), lab.y[self.hold], lab.rows.ri[self.hold]) for p in RIDGE_PENALTIES}
        self.penalty = max(score, key=score.get)
        self.Xt, self.Xp = self._X(self.train), self._X(self.pred)
        n = len(self.Xt)
        self.A, self.b = self.Xt.T @ self.Xt / n, self.Xt.T @ lab.y[self.train] / n
        self.base = self._judge(self.Xp @ cb.ridge_weights(self.A, self.b, self.penalty))
        rng = np.random.default_rng(7)
        refs = [self._with(rng.uniform(-0.5, 0.5, len(lab.y))) for _ in range(RIDGE_DUMMIES)]
        self.ref = {k: float(np.mean([r[k] for r in refs])) for k in ("ic", "net_bp")}

    def _judge(self, pred):
        wide = self.lab.rows.to_wide(pred, self.pred)
        return cb.summarize(cb.judge(wide, self.lab.m["screen"]))

    def _with(self, v):
        lab, n = self.lab, len(self.Xt)
        vt, vp = v[self.train].astype(np.float64), v[self.pred].astype(np.float64)
        a = self.Xt.T @ vt / n
        A = np.block([[self.A, a[:, None]], [a[None, :], np.array([[vt @ vt / n]])]])
        b = np.append(self.b, vt @ lab.y[self.train] / n)
        w = cb.ridge_weights(A, b, self.penalty)
        return self._judge(np.column_stack([self.Xp, vp]) @ w)

    def test(self, v):
        r = self._with(v)
        return {"ic_gain": round(r["ic"] - self.ref["ic"], 5), "net_gain_bp": round(r["net_bp"] - self.ref["net_bp"], 2)}


class TreesConfirm:
    """Trees blend (the step 6 method) of broad + kept alphas, walked forward month by month through 2023 H2.
    Tree size and number of trees are picked once for the base (learn on 2021-22, check on 2023 H1)."""

    def __init__(self, lab):
        self.lab = lab
        self.months = pd.date_range(lab.m["confirm"].start, lab.m["confirm"].end, freq="MS")
        self.rebuild()

    @staticmethod
    def model(leaves, n, seed=0):
        return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=n, max_leaf_nodes=leaves, min_samples_leaf=1000,
                                             l2_regularization=1.0, max_features=0.5, max_bins=63, early_stopping=False,
                                             random_state=seed)

    def _X(self, extra=None):
        cols = self.lab.all_cols()
        X = np.column_stack([cols[a] for a in cols])
        return X if extra is None else np.column_stack([X, extra])

    def rebuild(self):
        lab = self.lab
        d, has_y = lab.date, ~np.isnan(lab.y)
        X = self._X()
        s = pd.Timestamp(lab.m["screen"].start)
        inner = (d <= s - GAP) & has_y
        hold = (d >= s) & (d <= pd.Timestamp(lab.m["screen"].end)) & has_y
        best = (-np.inf, None, None)
        for leaves in TREE_LEAVES:
            m = self.model(leaves, TREE_MAX_ITER).fit(X[inner], lab.y[inner])
            for n, p in enumerate(m.staged_predict(X[hold]), 1):
                if n % TREE_STEP == 0:
                    sc = cb.holdout_ic(p, lab.y[hold], lab.rows.ri[hold])
                    if sc > best[0]:
                        best = (sc, leaves, n)
        _, self.leaves, self.n_trees = best
        rng = np.random.default_rng(11)
        refs = [self._walk(self._X(rng.uniform(-0.5, 0.5, len(lab.y)).astype(np.float32))) for _ in range(TREE_DUMMIES)]
        self.ref = {k: float(np.mean([r[k] for r in refs])) for k in ("ic", "net_bp")}

    def _walk(self, X):
        lab = self.lab
        d, has_y = lab.date, ~np.isnan(lab.y)
        pred = np.full(len(lab.y), np.nan)
        for m0 in self.months:
            train = (d <= m0 - GAP) & has_y
            rows = (d >= m0) & (d <= m0 + pd.offsets.MonthEnd(0))
            pred[rows] = np.mean([self.model(self.leaves, self.n_trees, s).fit(X[train], lab.y[train]).predict(X[rows])
                                  for s in TREE_SEEDS], axis=0)
        rows = ~np.isnan(pred)
        return cb.summarize(cb.judge(lab.rows.to_wide(pred[rows], rows), lab.m["confirm"]))

    def test(self, v):
        r = self._walk(self._X(v))
        return {"ic_gain": round(r["ic"] - self.ref["ic"], 5), "net_gain_bp": round(r["net_bp"] - self.ref["net_bp"], 2)}


# ------------------------------------------------------------------ the LLM

TOOL = {
    "name": "submit_alphas",
    "description": "Submit new alpha formulas for testing.",
    "input_schema": {
        "type": "object",
        "properties": {"alphas": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "idea": {"type": "string", "description": "One sentence: what the formula bets on, in plain English."},
                "why_new": {"type": "string", "description": "Why it differs from the existing alphas listed."},
                "why_survives_costs": {"type": "string", "description": "Why it should not trade too often (fees are 0.05% per trade)."},
                "formula": {"type": "string", "description": "The formula, 101-Alphas style, using only the allowed fields and functions."},
            },
            "required": ["idea", "why_new", "why_survives_costs", "formula"]}}},
        "required": ["alphas"],
    },
}


def read_env():
    path = ROOT / ".env"
    if not path.exists():
        return {}
    pairs = (l.split("=", 1) for l in path.read_text(encoding="utf-8").splitlines() if "=" in l and not l.lstrip().startswith("#"))
    return {k.strip(): v.strip() for k, v in pairs}


class Claude:
    def __init__(self, model, key):
        import anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=4)
        self.model = model

    def ask(self, system, user):
        t0 = time.time()
        r = self.client.messages.create(
            model=self.model, max_tokens=4000,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            tools=[TOOL], tool_choice={"type": "auto"},          # newer models refuse a forced tool choice
            messages=[{"role": "user", "content": user}])
        items = next((b.input for b in r.content if b.type == "tool_use"), {}).get("alphas")
        if items is None:                                       # answered in text: read the JSON from it
            items = parse_text_reply("".join(b.text for b in r.content if b.type == "text"))
        u = r.usage
        usage = {"input": u.input_tokens, "cache_write": u.cache_creation_input_tokens or 0,
                 "cache_read": u.cache_read_input_tokens or 0, "output": u.output_tokens, "seconds": round(time.time() - t0, 1)}
        return items, usage


class FakeLLM:
    """Ready-made formulas, to test the whole loop for free."""
    FORMULAS = [
        "-1 * decay_linear(abs(funding), 5)", "rank(ts_mean_typo(close, 5))", "-1 * rank(delta(premium, 3))",
        "decay_linear(taker_buy_ratio - ts_min(taker_buy_ratio, 20), 10)", "-1 * correlation(returns, mkt_returns, 30)",
        "rank(volume / trades) - rank(delay(volume / trades, 20))", "-1 * stddev(returns, 20) / stddev(returns, 60)",
        "delay(close, -1)", "-1 * ts_rank(funding, 60)", "rank(sum(premium, 10))",
    ]

    def __init__(self):
        self.model, self.i = "fake", 0

    def ask(self, system, user):
        n = 5
        items = []
        for _ in range(n):
            f = self.FORMULAS[self.i % len(self.FORMULAS)]
            self.i += 1
            items.append({"idea": "test formula", "why_new": "-", "why_survives_costs": "-", "formula": f})
        return items, {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0, "seconds": 0.0}


def parse_text_reply(text):
    """Formulas from a plain-text answer: the JSON object between the first { and the last }."""
    try:
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
    except ValueError:
        return []
    items = data.get("alphas", []) if isinstance(data, dict) else data
    return items if isinstance(items, list) else []


def cost_usd(model, u):
    p = PRICES.get(model, PRICES["claude-sonnet-5-5"])
    return (u["input"] * p[0] + u["cache_write"] * p[1] + u["cache_read"] * p[2] + u["output"] * p[3]) / 1e6


# ------------------------------------------------------------------ prompts

def system_prompt(lab):
    cat = load_catalog()
    core = pd.read_csv(ev_path("dedupe", "core_list.csv"), index_col=0)
    fam_lines = []
    for fam, ids in core.groupby("family", sort=False).groups.items():
        fam_lines.append(f"- {fam}: " + core.loc[ids[0], "family_note"])
        for a in ids[:2]:
            fam_lines.append(f"    {a}: {cat[a]['formula'][:200]}")
    by_cat = {}
    for a in lab.broad:
        by_cat.setdefault(cat[a]["category"], []).append(f"{a}: {cat[a]['formula'][:160]}")
    broad_lines = []
    for c, items in sorted(by_cat.items()):
        broad_lines.append(f"[{c}]")
        broad_lines += [f"  {x}" for x in items]
    return f"""You are a quantitative researcher inventing new daily alpha formulas for crypto perpetual futures.

THE MARKET
- Binance USDT perpetual futures, daily bars. Each day the universe is the 80 most-traded coins.
- An alpha gives every coin a score each day. We buy the top 20% and short the bottom 20% at the daily close,
  rebalance daily, and pay 0.05% fees per trade plus funding.
- The SIGN does not matter: an alpha that works backwards is flipped automatically.

FORMULA LANGUAGE (101-Alphas style)
Fields (one value per coin per day):
  open, high, low, close, vwap  prices in USDT
  volume          the day's trading value in USDT
  returns         close-to-close return
  advN            average daily volume over N days, e.g. adv20 (N from 2 to 250)
  funding         funding paid that day (sum of the 8-hourly rates; positive = longs pay shorts)
  premium         perpetual price minus the spot index at the close, as a fraction (often exactly 0)
  trades          number of trades that day
  taker_buy_ratio share of the day's volume where the buyer was the aggressor (0 to 1)
  mkt_returns     the whole market's return that day (volume-weighted; same value for every coin)
  btc_returns     BTC's return that day (same value for every coin)
  IndClass.sector the coin's Binance sector, ONLY as indneutralize(x, IndClass.sector)
Functions (d = window in days, a plain whole number):
  Cross-section (compares coins on the same day): rank(x), scale(x), indneutralize(x, IndClass.sector)
  Time series (each coin's own past): delay(x,d) delta(x,d) [d 1-250]; sum(x,d) product(x,d) ts_min(x,d)
    ts_max(x,d) ts_argmax(x,d) ts_argmin(x,d) decay_linear(x,d) [d 2-250]; stddev(x,d) ts_rank(x,d)
    correlation(x,y,d) covariance(x,y,d) [d 3-250]
  Element-wise: abs(x) sign(x) log(x) signedpower(x,e) min(a,b) max(a,b) (a constant inside min/max must be
    between -1 and 1; use ts_min/ts_max for windows)
  Operators: + - * / ^, comparisons < > <= >= == !=, && ||, and c ? a : b
Rules: at most {MAX_CHARS} characters and {MAX_NODES} parts; windows are plain numbers; only the fields and
functions above; no future data is available (and none can be reached).

HOW A FORMULA IS JUDGED (in this order)
1. It must compute and have values for at least 80% of coin-days, varying across coins.
2. Own signal in 2021-2022: rank-IC t-stat at least 2, or long-short money t-stat at least 2.
3. Not a twin: similarity below 0.7 to every existing alpha (by daily rankings or by daily profits).
4. Screen (2023 first half): add it to a ridge blend of the existing alphas. Compared with adding a column
   of random numbers, the blend's rank IC must rise by at least +{SCREEN_IC} and it must earn at least
   +{SCREEN_NET_BP:.0f} bp/day more after costs.
5. Confirm (2023 second half, numbers hidden): the step 6 trees blend must not earn less with it.
Kept alphas join the blends, so later formulas must add something beyond them too.

LESSONS SO FAR
- Ranking is not money. Crypto returns are lopsided: a formula can rank coins well every day and still lose
  money, because a few coins jump hugely.
- Fees decide a lot. Fast signals (1-3 day windows) change the portfolio every day and fees eat them.
  Smoothing (decay_linear, longer windows) makes a signal cheaper to trade.
- The blend is crowded: the {len(lab.broad)} alphas below already cover short-term reversal, volume cooling,
  price-volume divergence, size and liquidity, the perp premium and many technical patterns. Rewording them
  (another window, another wrapper) gives twins. NEW INFORMATION is most valuable: funding, premium, taker
  flow, trade size, links to BTC and the market, and new combinations.
- In earlier tests, wrapping everything in indneutralize did not help: use it only where sectors matter.

THE 8 FAMILIES OF THE BEST EXISTING ALPHAS
{chr(10).join(fam_lines)}

ALL {len(lab.broad)} EXISTING ALPHAS (copies of these are rejected; notation may be GTJA, Qlib or plain math)
{chr(10).join(broad_lines)}

WHAT TO RETURN
Always answer by calling the submit_alphas tool, never with plain text. Send exactly the number of formulas
asked for. Each one needs a one-sentence idea, why it
is new, why it should not trade too often, and the formula. Make the formulas different from each other and
from everything already tried."""


def round_message(rnd, rounds, theme, n, history, kept_rows, fix_list=None):
    lines = []
    if fix_list:
        lines.append(f"Some formulas from this round could not be used. Send exactly {len(fix_list)} corrected formula(s),"
                     " one per problem below, keeping each idea:")
        for f, why in fix_list:
            lines.append(f"- {f}\n    problem: {why}")
        return "\n".join(lines)
    lines.append(f"Round {rnd} of {rounds}. Theme for this round: {theme}.")
    lines.append(f"Send exactly {n} new formulas. At most {MAX_INDNEUTRALIZE} of them may use indneutralize.")
    if kept_rows:
        lines.append("\nKept so far (they are now part of the blends):")
        lines += [f"- {k}: {f}" for k, f in kept_rows]
    if history:
        lines.append(f"\nYour latest {len(history)} tries, newest first (scores: 2021-22 own signal; screen: 2023 H1 blend gain):")
        for h in history:
            lines.append(f"- {h['formula']}\n    {describe(h)}")
    else:
        lines.append("\nThis is the first round: no earlier tries yet.")
    return "\n".join(lines)


def describe(r):
    st = r.get("stage")
    def f(k, fmt):
        v = r.get(k)
        return format(v, fmt) if isinstance(v, (int, float)) else "n/a"
    own = ""
    if "ic" in r:
        own = (f"own: rank IC {f('ic', '+.4f')} (t {f('ic_t', '+.1f')}), money t {f('ls_t', '+.1f')}, "
               f"after costs {f('net_bp', '+.1f')} bp/day, turnover {f('turnover', '.2f')}. ")
    if st == "kept":
        return own + f"KEPT as {r.get('kept_as')}."
    if st == "confirm":
        return own + (f"screen passed (net {f('screen_net_gain_bp', '+.1f')} bp/day, IC {f('screen_ic_gain', '+.4f')}), "
                      "but the confirm half-year failed.")
    return own + r.get("reason", "")


# ------------------------------------------------------------------ random formulas for the luck check

def random_formula(rng):
    fields = ["close", "open", "high", "low", "vwap", "volume", "returns", "funding", "premium", "trades", "taker_buy_ratio"]

    def node(depth):
        if depth == 0:
            return str(rng.choice(fields))
        kind = rng.choice(["ts", "ts", "cs", "bin", "pair"])
        if kind == "ts":
            fn = rng.choice(["ts_rank", "decay_linear", "delta", "stddev", "sum", "ts_max", "ts_min"])
            return f"{fn}({node(depth - 1)}, {int(rng.choice([3, 5, 10, 20, 40]))})"
        if kind == "cs":
            return f"rank({node(depth - 1)})"
        if kind == "bin":
            return f"({node(depth - 1)} {rng.choice(['+', '-', '*', '/'])} {node(depth - 1)})"
        fn = rng.choice(["correlation", "covariance"])
        return f"{fn}({node(depth - 1)}, {node(depth - 1)}, {int(rng.choice([5, 10, 20]))})"
    return node(int(rng.integers(2, 4)))


def luck_check(lab, run, seed):
    """Pure noise and random formulas through the screen and confirm gates (no own-signal or twin filter)."""
    rng = np.random.default_rng(seed)
    U = lab.m["train"].universe
    out = []
    for i in range(LUCK_N):
        noise = pd.DataFrame(rng.normal(size=U.shape), index=U.index, columns=U.columns).where(U)
        r = lab.test_values(noise, {"formula": f"noise_{i + 1}"}, luck=True)
        out.append({"run": run, "kind": "noise", **{k: v for k, v in r.items() if not k.startswith("_")}})
    made, attempts = 0, 0
    while made < LUCK_N and attempts < 500:
        attempts += 1
        f = random_formula(rng)
        r = lab.test(f, luck=True)
        if r["stage"] in ("refused", "compute", "lookahead", "coverage"):
            continue
        made += 1
        out.append({"run": run, "kind": "random formula", **{k: v for k, v in r.items() if not k.startswith("_")}})
    df = pd.DataFrame(out)
    df["passed_screen"] = df["stage"].isin(["confirm", "kept"])
    df["passed_both"] = df["stage"] == "kept"
    lab.run_vectors.clear()      # luck items must not count as earlier tries
    return df


# ------------------------------------------------------------------ the run

def load_tries():
    path = OUT / "tries.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rec):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return str(o)


def clean(rec):
    out = {}
    for k, v in rec.items():
        if k.startswith("_"):
            continue
        if isinstance(v, (np.floating, float)):
            v = float(v)
            v = round(v, 6) if math.isfinite(v) else None
        elif isinstance(v, np.integer):
            v = int(v)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--per-round", type=int, default=5)
    ap.add_argument("--model", default=None, help="Claude model id (default: LLM_MODEL in .env)")
    ap.add_argument("--fake", action="store_true", help="use ready-made formulas instead of an LLM (free)")
    ap.add_argument("--budget", type=float, default=20.0, help="stop when the run has cost this many US dollars")
    ap.add_argument("--stop-after", type=int, default=20, help="stop after this many rounds in a row with nothing kept")
    ap.add_argument("--no-luck", action="store_true", help="skip the luck check (only for quick tests)")
    ap.add_argument("--out", default=None, help="write records here instead of data/llm (for test runs)")
    args = ap.parse_args()
    global OUT, LOGS
    if args.out:
        OUT = Path(args.out)
        LOGS = OUT / "logs"

    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    env = read_env()
    if args.fake:
        llm = FakeLLM()
    else:
        key = env.get("ANTHROPIC_API_KEY", "")
        if not key.startswith("sk-ant-"):
            raise SystemExit("No API key: put ANTHROPIC_API_KEY=... in the .env file (see README).")
        llm = Claude(args.model or env.get("LLM_MODEL", "claude-sonnet-5-5"), key)
    model = llm.model

    tries = load_tries()
    runs_path = OUT / "runs.csv"
    runs = pd.read_csv(runs_path) if runs_path.exists() else pd.DataFrame()
    run = int(runs["run"].max()) + 1 if len(runs) else 1
    next_try = max((t["try"] for t in tries), default=0) + 1
    seen = {t["canonical"] for t in tries if t.get("canonical")}
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"run {run}: model {model}, declared {args.rounds} rounds x {args.per_round} formulas, budget ${args.budget:.0f}", flush=True)

    lab = Lab()
    kept_path = OUT / "kept.csv"
    kept = pd.read_csv(kept_path) if kept_path.exists() else pd.DataFrame(columns=["id", "formula"])
    for _, k in kept.iterrows():                     # resume: earlier keepers rejoin the blends as they are
        tree, _ = check_formula(k["formula"])
        alpha = compute(tree, lab.ctx, lab.eng.universe)
        s = lab.m["search"]
        card, series = ev.score(alpha.loc[s.start:s.end], s)
        lab.keep(k["id"], {"_vector": lab.vector(alpha, card["direction"]), "_net": series["net"]})

    luck = None
    if not args.no_luck:
        t0 = time.time()
        luck = luck_check(lab, run, seed=run)
        luck_path = OUT / "luck_check.csv"
        old = pd.read_csv(luck_path) if luck_path.exists() else None      # runs can record different columns
        pd.concat([old, luck], ignore_index=True).to_csv(luck_path, index=False)
        print(f"luck check ({time.time() - t0:.0f}s): screen passed {int(luck['passed_screen'].sum())} of {len(luck)}, "
              f"both gates {int(luck['passed_both'].sum())} of {len(luck)}", flush=True)

    system = system_prompt(lab)
    (LOGS / f"run{run}_system.txt").write_text(system, encoding="utf-8")
    spent, calls, usage_sum, history = 0.0, 0, {"input": 0, "cache_write": 0, "cache_read": 0, "output": 0}, []
    kept_rows = list(zip(kept["id"], kept["formula"]))
    quiet, status, t_run = 0, "finished", time.time()
    run_tries = []

    def call(user, tag):
        nonlocal spent, calls
        items, u = llm.ask(system, user)
        calls += 1
        for k in usage_sum:
            usage_sum[k] += u[k]
        c = cost_usd(model, u)
        spent += c
        (LOGS / f"run{run}_{tag}.json").write_text(json.dumps({"user": user, "reply": items, "usage": u, "usd": c}, indent=1), encoding="utf-8")
        return items

    def handle(item, rnd, theme, fix, indn_used):
        nonlocal next_try
        item = item if isinstance(item, dict) else {"formula": str(item)}
        f = str(item.get("formula", "")).strip()
        base = {"run": run, "round": rnd, "try": next_try, "model": model, "theme": theme, "fix": fix,
                "idea": item.get("idea", ""), "why_new": item.get("why_new", ""), "why_costs": item.get("why_survives_costs", ""),
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        next_try += 1
        tree, why = check_formula(f)
        if tree is not None:
            can = canonical(tree)
            if can in seen:
                why = "already tried (same formula as an earlier try)"
            elif _uses(tree, "indneutralize") and indn_used[0] >= MAX_INDNEUTRALIZE:
                why = f"more than {MAX_INDNEUTRALIZE} formulas used indneutralize this round"
        if why:
            rec = {**base, "formula": f, "stage": "refused", "reason": why}
        else:
            seen.add(canonical(tree))
            if _uses(tree, "indneutralize"):
                indn_used[0] += 1
            rec = {**base, **lab.test(f)}
        if rec["stage"] == "kept":
            name = f"LLM_{len(kept_rows) + 1:03d}"
            rec["kept_as"] = name
            lab.keep(name, rec)
            kept_rows.append((name, f))
            row = {"id": name, "formula": f, "idea": rec["idea"], "run": run, "round": rnd, "try": rec["try"], "model": model,
                   **{k: rec.get(k) for k in ("direction", "ic", "ic_t", "ls_t", "net_bp", "turnover", "screen_ic_gain",
                                              "screen_net_gain_bp", "confirm_ic_gain", "confirm_net_gain_bp")}}
            pd.DataFrame([row]).to_csv(kept_path, mode="a", header=not kept_path.exists() or kept_path.stat().st_size == 0, index=False)
        rec = clean(rec)
        append_jsonl(OUT / "tries.jsonl", rec)
        run_tries.append(rec)
        print(f"  #{rec['try']:4d} {rec['stage']:9s} {f[:70]:70s} {rec.get('reason', '')[:70]}", flush=True)
        return rec

    try:
        for rnd in range(1, args.rounds + 1):
            if spent >= args.budget:
                status = f"stopped: budget ${args.budget:.0f} reached"
                break
            theme = THEMES[(rnd - 1) % len(THEMES)]
            user = round_message(rnd, args.rounds, theme, args.per_round, history[:HISTORY], kept_rows)
            items = call(user, f"round{rnd:03d}")[:args.per_round]
            print(f"round {rnd}/{args.rounds} ({theme[:40]}...), {len(items)} formulas, spent ${spent:.2f}", flush=True)
            indn_used = [0]
            recs = [handle(it, rnd, theme, False, indn_used) for it in items]
            broken = [(r["formula"], r["reason"]) for r in recs if r["stage"] in ("refused", "compute") and "already tried" not in r["reason"]
                      and "indneutralize this round" not in r["reason"]]
            if broken and spent < args.budget:
                fixes = call(round_message(rnd, args.rounds, theme, len(broken), [], kept_rows, fix_list=broken), f"round{rnd:03d}_fix")
                recs += [handle(it, rnd, theme, True, indn_used) for it in fixes[:len(broken)]]
            history = list(reversed(recs)) + history
            quiet = 0 if any(r["stage"] == "kept" for r in recs) else quiet + 1
            if quiet >= args.stop_after:
                status = f"stopped: nothing kept in {args.stop_after} rounds"
                break
    except KeyboardInterrupt:
        status = "stopped by hand"
    except Exception as e:  # noqa: BLE001 - record the failure, then re-raise
        status = f"error: {type(e).__name__}: {str(e)[:200]}"
        raise
    finally:
        st = pd.Series([r["stage"] for r in run_tries]).value_counts().to_dict() if run_tries else {}
        row = {"run": run, "model": model, "declared_rounds": args.rounds, "per_round": args.per_round, "started": started,
               "minutes": round((time.time() - t_run) / 60, 1), "status": status, "tries": len(run_tries),
               "kept": st.get("kept", 0), "passed_screen": st.get("kept", 0) + st.get("confirm", 0),
               "twins": st.get("twin", 0), "weak": st.get("weak", 0), "refused": st.get("refused", 0),
               "compute_errors": st.get("compute", 0), "coverage": st.get("coverage", 0), "lookahead": st.get("lookahead", 0),
               "luck_screen_passes": int(luck["passed_screen"].sum()) if luck is not None else None,
               "luck_both_passes": int(luck["passed_both"].sum()) if luck is not None else None,
               "calls": calls, **{f"tokens_{k}": v for k, v in usage_sum.items()}, "usd": round(spent, 3)}
        pd.DataFrame([row]).to_csv(runs_path, mode="a", header=not runs_path.exists(), index=False)
        print(f"\nrun {run} {status}: {row['tries']} tries, kept {row['kept']}, passed screen {row['passed_screen']}, "
              f"twins {row['twins']}, weak {row['weak']}, refused {row['refused']}, cost ${spent:.2f}, {row['minutes']} min", flush=True)


if __name__ == "__main__":
    main()
