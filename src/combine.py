"""Step 6: blend many alphas into one score, walking forward through 2024 (the validation year).

    .venv/bin/python src/combine.py      # after dedupe.py; about 10 min (the tree model is the slow part)

Rules agreed in the step 6 discussion (see README):
  - 2024 picks the METHOD, never the alphas: the core and broad lists come from 2021-2023 only.
  - Each alpha becomes its daily rank among the day's top 80, flipped so higher = better (direction
    from 2021-2023) and centred on 0. Missing = 0, the middle.
  - Models learn tomorrow's return, ranked among the day's coins.
  - Walk forward: at the start of each month of 2024, re-learn on all earlier days. Settings (ridge
    penalty, tree size) are picked inside each training window, using its last 6 months as a mini-test.
  - Portfolio: long the top 20%, short the bottom 20% (16 + 16 of 80), equal weight, daily.
    0.05% fee per unit traded + funding, as in step 3.
  - Judge: 2024 Sharpe after fees + funding. To pass, beat equal weight AND the best single alpha
    by training money t (A101_055), and make money after costs.
Writes data/eval/combine/.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import evaluate as ev
from alphas import AlphaEngine
from config import DATA

EVAL = DATA / "eval"
OUT = EVAL / "combine"
START, END = ev.PERIODS["train+valid"]
VALID_START = ev.PERIODS["valid"][0]
GAP = pd.Timedelta(days=2)    # last training day = 2 days before trading: its answer (next close) is known by then
HOLDOUT_MONTHS = 6            # mini-test inside each training window, for picking settings
IC_LOOKBACK = pd.Timedelta(days=365)
BUFFER = 0.3                  # the buffer version keeps a coin until it leaves the top/bottom 30%
RIDGE_PENALTIES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # big penalty -> weights proportional to each alpha's IC
TREE_LEAVES = (7, 31)
TREE_MAX_ITER, TREE_STEP = 300, 10
BENCHMARK = "A101_055"        # best single alpha by training money t (chosen before looking at 2024)
TOP = 1 / ev.N_GROUPS


# ------------------------------------------------------------------ long table: one row per (date, coin) in the universe

class Rows:
    """Maps between wide panels (date x coin) and long rows (only universe coin-days, sorted by date)."""

    def __init__(self, universe):
        self.dates, self.coins = universe.index, universe.columns
        self.ri, self.ci = np.nonzero(universe.to_numpy())
        self.date = self.dates[self.ri]
        self.index = pd.MultiIndex.from_arrays([self.date, self.coins[self.ci]], names=["date", "coin"])

    def to_long(self, wide):
        return wide.reindex(index=self.dates, columns=self.coins).to_numpy(dtype=np.float64)[self.ri, self.ci]

    def to_wide(self, values, rows):
        out = np.full((len(self.dates), len(self.coins)), np.nan)
        out[self.ri[rows], self.ci[rows]] = values
        days = np.unique(self.ri[rows])
        return pd.DataFrame(out[days], index=self.dates[days], columns=self.coins)


def centred_rank(wide, universe):
    r = wide.where(universe).rank(axis=1, pct=True)
    return r.sub(r.mean(axis=1), axis=0)


def build_features(ids, direction, universe, rows):
    """rows x alphas: daily rank among the universe, flipped so higher = better, centred on 0, missing = 0.
    Only 2021-2024 is kept: nothing from the locked test period is saved."""
    eng = AlphaEngine.load()
    t0 = time.time()
    cols = {}
    for i, a in enumerate(ids, 1):
        v = eng.compute(a).loc[START:END]
        cols[a] = np.nan_to_num(rows.to_long(centred_rank(v, universe) * direction[a])).astype(np.float32)
        if i % 40 == 0:
            print(f"  {i} alphas computed ({time.time() - t0:.0f}s)", flush=True)
    X = pd.DataFrame(cols, index=rows.index)
    X.to_parquet(OUT / "features.parquet")      # kept for step 7
    return X


# ------------------------------------------------------------------ daily correlation on long rows

def daily_corr(P, y, day):
    """Each day: correlation between each column of P and y. Rows must be sorted by day.
    Returns (day ids, array days x columns)."""
    P = P.reshape(len(y), -1).astype(np.float64)
    ok = ~np.isnan(y)
    P, y, day = P[ok], y[ok], day[ok]
    uniq, starts, counts = np.unique(day, return_index=True, return_counts=True)
    Pc = P - np.repeat(np.add.reduceat(P, starts) / counts[:, None], counts, axis=0)
    yc = y - np.repeat(np.add.reduceat(y, starts) / counts, counts)
    num = np.add.reduceat(Pc * yc[:, None], starts)
    den = np.sqrt(np.add.reduceat(Pc ** 2, starts) * np.add.reduceat(yc ** 2, starts)[:, None])
    ic = num / np.where(den > 0, den, np.nan)
    ic[counts < ev.MIN_COINS] = np.nan
    return uniq, ic


def holdout_ic(pred, y, day):
    """Mean daily rank IC of a prediction (used only to pick settings inside a training window)."""
    pr = pd.Series(pred).groupby(day).rank(pct=True).to_numpy()
    return float(np.nanmean(daily_corr(pr, y, day)[1]))


# ------------------------------------------------------------------ the methods

def fit_equal(ctx, cols):
    def run(sets):
        return ctx.X[sets.pred][:, cols].mean(axis=1), {}
    return run


def fit_ic_weighted(ctx, cols):
    """Weight = each alpha's mean daily IC over the last 12 months; negative -> 0."""
    days, ic = daily_corr(ctx.X[:, cols], ctx.y, ctx.rows.ri)
    ic_days = ctx.rows.dates[days]

    def run(sets):
        win = (ic_days > sets.last - IC_LOOKBACK) & (ic_days <= sets.last)
        w = np.clip(np.nanmean(ic[win], axis=0), 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(len(cols), 1 / len(cols))
        return ctx.X[sets.pred][:, cols] @ w, {"weights": w}
    return run


def ridge_weights(XtX, Xty, penalty):
    return np.linalg.solve(XtX + penalty * np.eye(len(Xty)), Xty)


def fit_ridge(ctx, cols, y=None):
    y_all = ctx.y if y is None else y

    def gram(rows):
        X = ctx.X[rows][:, cols].astype(np.float64)
        return X.T @ X / len(X), X.T @ y_all[rows] / len(X)

    def run(sets):
        A, b = gram(sets.inner)
        Xh = ctx.X[sets.hold][:, cols]
        score = {p: holdout_ic(Xh @ ridge_weights(A, b, p), y_all[sets.hold], ctx.rows.ri[sets.hold])
                 for p in RIDGE_PENALTIES}
        best = max(score, key=score.get)
        w = ridge_weights(*gram(sets.train), best)
        return ctx.X[sets.pred][:, cols] @ w, {"penalty": best, "weights": w}
    return run


def fit_trees(ctx, cols):
    """Shallow boosted trees. Tree size and number of trees are picked on the mini-test."""
    def model(leaves, n):
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=n, max_leaf_nodes=leaves, min_samples_leaf=1000,
            l2_regularization=1.0, max_features=0.5, max_bins=63, early_stopping=False, random_state=0)

    def run(sets):
        Xi, yi = ctx.X[sets.inner][:, cols], ctx.y[sets.inner]
        Xh, yh, dh = ctx.X[sets.hold][:, cols], ctx.y[sets.hold], ctx.rows.ri[sets.hold]
        best = (-np.inf, None, None)
        for leaves in TREE_LEAVES:
            m = model(leaves, TREE_MAX_ITER).fit(Xi, yi)
            for n, p in enumerate(m.staged_predict(Xh), 1):
                if n % TREE_STEP == 0:
                    s = holdout_ic(p, yh, dh)
                    if s > best[0]:
                        best = (s, leaves, n)
        _, leaves, n = best
        m = model(leaves, n).fit(ctx.X[sets.train][:, cols], ctx.y[sets.train])
        return m.predict(ctx.X[sets.pred][:, cols]), {"leaves": leaves, "trees": n}
    return run


class Sets:
    """Which rows each refit may use. Training rows end 2 days before the month starts."""

    def __init__(self, ctx, d0):
        d = ctx.rows.date
        has_y = ~np.isnan(ctx.y)
        self.month = d0
        self.last = d0 - GAP
        hold_start = d0 - pd.DateOffset(months=HOLDOUT_MONTHS)
        self.train = (d <= self.last) & has_y
        self.inner = (d <= hold_start - GAP) & has_y
        self.hold = (d >= hold_start) & (d <= self.last) & has_y
        self.pred = (d >= d0) & (d <= d0 + pd.offsets.MonthEnd(0))
        assert d[self.train].max() < d0 - pd.Timedelta(days=1) and d[self.pred].min() >= d0


def walk_forward(ctx, run, name):
    t0 = time.time()
    pred = np.full(len(ctx.y), np.nan)
    info = []
    for d0 in pd.date_range(VALID_START, END, freq="MS"):
        sets = Sets(ctx, d0)
        p, extra = run(sets)
        pred[sets.pred] = p
        info.append({"method": name, "month": d0.strftime("%Y-%m"), **extra})
    print(f"  {name:22s} walked forward in {time.time() - t0:5.1f}s", flush=True)
    rows = ~np.isnan(pred)
    return ctx.rows.to_wide(pred[rows], rows), info


# ------------------------------------------------------------------ portfolio and judging

def holdings(score, universe, keep=None):
    """Daily (long, short) weights: top 20% and bottom 20%, equal weight, each side summing to 1.
    Without a buffer this is step 3's rule (tied coins share places). With keep > TOP, a held coin
    stays until it drops out of the top (bottom) `keep` share; blends have no ties, so no sharing needed."""
    if keep is None or keep == TOP:
        g = ev.group_weights(score.reindex(index=universe.index, columns=universe.columns), universe)
        top, bot = g[ev.N_GROUPS].fillna(0), g[1].fillna(0)
        return top.div(top.sum(axis=1), axis=0).fillna(0), bot.div(bot.sum(axis=1), axis=0).fillna(0)
    S = score.reindex(index=universe.index, columns=universe.columns).where(universe).to_numpy()
    L, Sh = np.zeros_like(S), np.zeros_like(S)
    held_long, held_short = [], []
    for t in range(len(S)):
        ok = np.flatnonzero(~np.isnan(S[t]))
        n = len(ok)
        if n < ev.MIN_COINS:
            held_long, held_short = [], []
            continue
        k, kk = max(1, round(TOP * n)), max(1, round(keep * n))
        order = ok[np.argsort(S[t, ok], kind="stable")]       # worst ... best
        held_long = pick(order[::-1], held_long, k, kk)
        held_short = pick(order, held_short, k, kk)
        L[t, held_long], Sh[t, held_short] = 1 / k, 1 / k
    frame = lambda x: pd.DataFrame(x, index=universe.index, columns=universe.columns)
    return frame(L), frame(Sh)


def pick(best_first, held, k, kk):
    """Keep held coins still inside the first kk, then fill up to k with the best new ones."""
    held = set(held)
    kept = [c for c in best_first[:kk] if c in held][:k]
    new = [c for c in best_first if c not in set(kept)][:k - len(kept)]
    return kept + new


def judge(score, m, keep=None):
    """Daily results of trading `score` (higher = better, never flipped) on Market m."""
    long, short = holdings(score, m.universe, keep=keep)
    w = long - short           # the book actually held (a coin in both sides nets out)
    r = m.fwd[1].where(m.universe)
    gross = ev.group_mean(r, long) - ev.group_mean(r, short)   # same as step 3
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()                  # building the first book costs fees too
    cost = ev.FEE * turnover
    funding = -(w * m.funding_next.fillna(0)).sum(axis=1)
    ic = ev.row_spearman(score.reindex(index=m.dates, columns=m.universe.columns), m.fwd[1], m.universe)
    daily = pd.DataFrame({"ic": ic, "gross": gross, "cost": cost, "funding": funding,
                          "net": gross - cost + funding, "turnover": turnover})
    return daily.dropna(subset=["gross"])


def summarize(daily):
    net = daily["net"]
    cum = net.cumsum()
    monthly = net.groupby(net.index.to_period("M")).sum()
    return {"days": len(net), "ic": daily["ic"].mean(), "ic_t": ev.nw_tstat(daily["ic"], 1),
            "gross_bp": daily["gross"].mean() * 1e4, "cost_bp": daily["cost"].mean() * 1e4,
            "funding_bp": daily["funding"].mean() * 1e4, "net_bp": net.mean() * 1e4,
            "net_t": ev.nw_tstat(net, 1), "net_sharpe": ev.ann_sharpe(net),
            "annual_pct": net.mean() * 365 * 100, "max_drawdown_pct": (cum - cum.cummax()).min() * 100,
            "turnover": daily["turnover"].mean(), "months_up": int((monthly > 0).sum()), "months": len(monthly)}


def versus(a, b):
    """Is a better than b day by day? Mean daily difference and its t-stat."""
    d = (a["net"] - b["net"]).dropna()
    return d.mean() * 1e4, ev.nw_tstat(d, 1)


# ------------------------------------------------------------------ main

class Ctx:
    pass


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    core = pd.read_csv(EVAL / "dedupe" / "core_list.csv", index_col=0).index.tolist()
    assert core[0] == BENCHMARK, f"best core alpha is now {core[0]}: update BENCHMARK"
    broad = pd.read_csv(EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
    direction = pd.read_csv(EVAL / "scorecard.csv").set_index("id")["direction"]
    whole = ev.Market("train+valid")
    train, valid = ev.Market("train"), ev.Market("valid")

    ctx = Ctx()
    ctx.rows = Rows(whole.universe)
    X = build_features(broad, direction, whole.universe, ctx.rows)
    ctx.X = X.to_numpy(dtype=np.float32)
    ctx.y = ctx.rows.to_long(centred_rank(whole.fwd[1], whole.universe))   # tomorrow's return, ranked
    ci = [broad.index(a) for a in core]
    bi = list(range(len(broad)))
    print(f"features: {ctx.X.shape[0]:,} coin-days x {ctx.X.shape[1]} alphas ({time.time() - t0:.0f}s)")

    methods = {
        "Equal weight · core": fit_equal(ctx, ci),
        "IC-weighted · core": fit_ic_weighted(ctx, ci),
        "Ridge · core": fit_ridge(ctx, ci),
        "Ridge · broad": fit_ridge(ctx, bi),
        "Trees · broad": fit_trees(ctx, bi),
    }
    scores, settings = {}, []
    for name, run in methods.items():
        scores[name], info = walk_forward(ctx, run, name)
        settings += info

    # Honesty check: the same pipeline trained on shuffled answers (shuffled within each day) must score ~0.
    rng = np.random.default_rng(42)
    known = np.flatnonzero(~np.isnan(ctx.y))         # rows sorted by day, so this keeps each answer in its own day
    y_shuffled = ctx.y.copy()
    y_shuffled[known] = ctx.y[known[np.lexsort((rng.random(len(known)), ctx.rows.ri[known]))]]
    noise, _ = walk_forward(ctx, fit_ridge(ctx, bi, y=y_shuffled), "Shuffled answers")

    # Judge everything on 2024
    daily = {name: judge(s, valid) for name, s in scores.items()}
    results = {name: summarize(d) for name, d in daily.items()}
    best = max(results, key=lambda k: results[k]["net_sharpe"])
    buf_name = f"{best} + buffer"
    daily[buf_name] = judge(scores[best], valid, keep=BUFFER)
    results[buf_name] = summarize(daily[buf_name])

    singles, singles_daily = [], {}
    for a in core:
        wide = ctx.rows.to_wide(X[a].to_numpy(), np.ones(len(ctx.y), bool))
        d24 = judge(wide, valid)
        singles_daily[a] = d24
        s_tr, s_va = summarize(judge(wide, train)), summarize(d24)
        singles.append({"id": a, **{f"train_{k}": v for k, v in s_tr.items()},
                        **{f"y2024_{k}": v for k, v in s_va.items()}})
    singles = pd.DataFrame(singles).set_index("id")
    daily[f"Single alpha · {BENCHMARK}"] = singles_daily[BENCHMARK]
    results[f"Single alpha · {BENCHMARK}"] = summarize(singles_daily[BENCHMARK])
    ew_wide = ctx.rows.to_wide(ctx.X[:, ci].mean(axis=1), np.ones(len(ctx.y), bool))
    ew_train = summarize(judge(ew_wide, train))
    noise_result = summarize(judge(noise, valid))

    # Table of results, 2024
    ew, bench = daily["Equal weight · core"], daily[f"Single alpha · {BENCHMARK}"]
    res = pd.DataFrame(results).T
    res.index.name = "method"
    for col, ref in (("vs_equal", ew), ("vs_single", bench)):
        pairs = [versus(daily[k], ref) for k in res.index]
        res[f"{col}_bp"] = [p[0] for p in pairs]
        res[f"{col}_t"] = [p[1] for p in pairs]
    # Pass = beat both benchmarks AND make money after costs (beating two losers while losing money is no pass).
    bar = max(res.at["Equal weight · core", "net_sharpe"], res.at[f"Single alpha · {BENCHMARK}", "net_sharpe"], 0)
    res["passes_bar"] = res["net_sharpe"] > bar
    res.loc["Equal weight · core", "passes_bar"] = False
    res.loc[f"Single alpha · {BENCHMARK}", "passes_bar"] = False
    res["kind"] = ["benchmark" if k.startswith(("Single", "Equal")) else "buffer" if k.endswith("buffer")
                   else "blend" for k in res.index]
    res.to_csv(OUT / "results_2024.csv")

    q = []
    for name, d in daily.items():
        for qq, x in d["net"].groupby(d.index.quarter):
            q.append({"method": name, "quarter": f"Q{qq}", "net_pct": x.sum() * 100,
                      "sharpe": ev.ann_sharpe(x), "ic": d["ic"][d.index.quarter == qq].mean()})
    pd.DataFrame(q).to_csv(OUT / "quarters_2024.csv", index=False)
    pd.concat({k: d for k, d in daily.items()}, names=["method", "date"]).to_parquet(OUT / "daily_2024.parquet")
    singles.to_csv(OUT / "singles.csv")

    wrows = []
    for s in settings:
        if "weights" in s:
            cols = core if s["method"].endswith("core") else broad
            w = np.asarray(s["weights"])
            w = w / np.abs(w).sum() if np.abs(w).sum() > 0 else w
            wrows += [{"method": s["method"], "month": s["month"], "alpha": a, "weight": float(v)}
                      for a, v in zip(cols, w)]
    pd.DataFrame(wrows).to_csv(OUT / "weights.csv", index=False)
    pd.DataFrame([{k: v for k, v in s.items() if k != "weights"} for s in settings]).to_csv(
        OUT / "settings.csv", index=False)
    check = pd.DataFrame([{"what": "Shuffled answers", **noise_result},
                          {"what": "Equal weight · core, 2021-2023 (flattering: alphas picked on these years)",
                           **ew_train}]).set_index("what")
    check.to_csv(OUT / "checks.csv")

    show = ["ic", "ic_t", "gross_bp", "cost_bp", "funding_bp", "net_bp", "net_t", "net_sharpe",
            "annual_pct", "max_drawdown_pct", "turnover", "months_up", "vs_equal_t", "vs_single_t", "passes_bar"]
    pd.set_option("display.width", 250)
    print("\n2024 (validation year):")
    print(res[show].round(3).to_string())
    print(f"\nshuffled answers: IC {noise_result['ic']:+.4f} (t {noise_result['ic_t']:+.2f}), "
          f"Sharpe {noise_result['net_sharpe']:+.2f}")
    print(f"equal weight 2021-2023 (flattering): Sharpe {ew_train['net_sharpe']:.2f}, "
          f"net {ew_train['net_bp']:.1f} bp/day")
    print("\nsingle core alphas, 2021-2023 vs 2024 net Sharpe:")
    print(singles[["train_net_sharpe", "y2024_net_sharpe", "y2024_ic", "y2024_net_bp"]].round(2).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
