"""Top 5 / bottom 5: do the forecast's top picks rise tomorrow and its bottom picks fall? (2022-2023 only)

    .venv/bin/python src/top_bottom.py      # about 25 min -> data/eval/top_bottom/

The owner's goal: find coins that rise even on days the market falls, and coins that fall even on days
the market rises (moves of their own, not the market's).

Where the picks come from:
    top 80   the usual universe: the 80 most-traded coins each day
    all      every coin in our data that trades that day and has 60+ days of history (up to 594 coins)
How the picks are made:
    1. rank every coin with the step 6 trees (Trees · broad + LLM), take the 5 best and the 5 worst
    2. two models aimed straight at the extremes: "will it be one of tomorrow's 5 best?" and "... 5 worst?"
Checks, each against "any coin that day" as the no-skill baseline:
    top picks that rose, bottom picks that fell, split into market-up and market-down days
    (market day = the median top-80 coin's next-day return, the same days for every test).
Data cut at 2023-12-31. Models learn from 2021 and are refitted every month through 2022-2023.
"""
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import combine as cb
import cost_aware as ca
import evaluate as ev
import llm_loop as ll
from alphas import AlphaEngine
from config import DATA, MIN_HISTORY

OUT = DATA / "eval" / "top_bottom"
END = ev.PERIODS["train"][1]
K = 5


def all_coins_mask(eng):
    """Every coin that trades that day and has at least MIN_HISTORY days of history (the universe rule
    without the 'top 80 by trading value' part)."""
    close, qv = eng.p["close"], eng.p["quote_volume"]
    return close.notna() & (close.notna().cumsum() >= MIN_HISTORY) & (qv > 0)


def features(eng, mask, start, ids, direction, kept, rows):
    """Daily rank of each alpha inside `mask`, flipped so higher = better, centred, missing = 0."""
    e = AlphaEngine(eng.p, mask, eng.sectors, eng.onboard)          # cross-sectional steps now compare inside `mask`
    m = mask.loc[start:END]
    cols = []
    for a in ids:
        v = e.compute(a).loc[start:END]
        cols.append(np.nan_to_num(rows.to_long(cb.centred_rank(v, m) * direction[a])).astype(np.float32))
    fctx = ll.make_context(e)
    for k in kept.itertuples():
        tree, _ = ll.check_formula(k.formula)
        v = ll.compute(tree, fctx, mask).loc[start:END]
        cols.append(np.nan_to_num(rows.to_long(cb.centred_rank(v, m) * k.direction)).astype(np.float32))
    return np.column_stack(cols)


def classifier():
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=500,
                                          l2_regularization=1.0, max_features=0.5, max_bins=63, early_stopping=False,
                                          random_state=0)


def make_picks(name, mask, X, rows, fwd):
    """Walk forward through 2022-2023; returns {approach: (top picks, bottom picks)} as date x coin booleans."""
    ctx = cb.Ctx()
    ctx.rows, ctx.X = rows, X
    r = fwd.where(mask)
    ctx.y = rows.to_long(cb.centred_rank(r, mask))
    true_rank = r.rank(axis=1, ascending=False, method="first")
    n = r.notna().sum(axis=1)
    y_top = rows.to_long((true_rank <= K).astype(float).where(r.notna()))
    y_bot = rows.to_long(true_rank.gt(n - K, axis=0).astype(float).where(r.notna()))
    t0 = time.time()
    score = ca.predict(ctx, list(range(X.shape[1])), ca.TUNE_MONTHS, 1)
    print(f"  {name}: ranking trees walked forward ({time.time() - t0:.0f}s)", flush=True)
    t0 = time.time()
    p_top, p_bot = np.full(len(ctx.y), np.nan), np.full(len(ctx.y), np.nan)
    for d0 in ca.TUNE_MONTHS:
        w = ca.Window(ctx, d0, 2)
        p_top[w.pred] = classifier().fit(X[w.train], y_top[w.train]).predict_proba(X[w.pred])[:, 1]
        p_bot[w.pred] = classifier().fit(X[w.train], y_bot[w.train]).predict_proba(X[w.pred])[:, 1]
    keep = ~np.isnan(p_top)
    P_top, P_bot = rows.to_wide(p_top[keep], keep), rows.to_wide(p_bot[keep], keep)
    print(f"  {name}: top/bottom-5 models walked forward ({time.time() - t0:.0f}s)", flush=True)

    days = score.index
    ok = mask.loc[days] & r.loc[days].notna()

    def ends(s, best):
        rk = s.reindex(index=days, columns=ok.columns).where(ok).rank(axis=1, ascending=not best, method="first")
        return rk <= K
    return {"1. rank every coin, take the ends": (ends(score, True), ends(score, False)),
            "2. models aimed at the top/bottom 5": (ends(P_top, True), ends(P_bot, True))}, ok


def check(top, bot, ok, fwd, market_up):
    """Accuracy of the picks against 'any coin that day' in the same set."""
    days = ok.index
    R = fwd.loc[days].where(ok)
    up, down = R > 0, R < 0
    up_day = market_up.reindex(days).fillna(False).astype(bool)
    rate = lambda hit, where, sel=None: hit.where(where).loc[up_day.index if sel is None else sel].stack().mean()
    out = {}
    for label, sel in (("all days", days), ("market-DOWN days", days[~up_day]), ("market-UP days", days[up_day])):
        out[f"top 5 ROSE, {label}"] = rate(up, top, sel)
        out[f"any coin rose, {label}"] = rate(up, ok, sel)
        out[f"bottom 5 FELL, {label}"] = rate(down, bot, sel)
        out[f"any coin fell, {label}"] = rate(down, ok, sel)
    true_rank = R.rank(axis=1, ascending=False, method="first")
    n = R.notna().sum(axis=1)
    out["top picks truly in tomorrow's 5 best"] = (true_rank <= K).where(top).stack().mean()
    out["bottom picks truly in tomorrow's 5 worst"] = true_rank.gt(n - K, axis=0).where(bot).stack().mean()
    n_ok = n.loc[days].where(lambda x: x > 0)                         # the period's last day has no answer yet
    out["chance of that for any coin"] = (K / n_ok).mean()
    out["coins to pick from per day"] = n_ok.mean()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eng = ll.cut_engine(AlphaEngine.load(), END)                    # nothing after 2023 exists from here on
    m = ev.Market("train")
    fwd = m.fwd[1]
    start = m.start
    top80 = eng.universe.loc[start:END]
    whole = all_coins_mask(eng).loc[start:END].reindex(columns=top80.columns, fill_value=False)
    med80 = fwd.where(top80).median(axis=1)
    market_up = med80 > 0
    broad = pd.read_csv(cb.EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
    direction = pd.read_csv(cb.EVAL / "scorecard.csv").set_index("id")["direction"]
    kept = pd.read_csv(ll.OUT / "kept.csv")
    full_mask = {"top 80": eng.universe, "all": all_coins_mask(eng)}

    results = []
    for name, mask in (("top 80", top80), ("all", whole)):
        rows = cb.Rows(mask)
        t = time.time()
        X = features(eng, full_mask[name].reindex(columns=eng.universe.columns, fill_value=False), start, broad, direction, kept, rows)
        print(f"{name}: {X.shape[0]:,} coin-days x {X.shape[1]} alphas ({time.time() - t:.0f}s)", flush=True)
        picks, ok = make_picks(name, mask, X, rows, fwd)
        for approach, (top, bot) in picks.items():
            results.append({"pick from": name, "approach": approach, **check(top, bot, ok, fwd, market_up)})
    res = pd.DataFrame(results)
    res.to_csv(OUT / "accuracy_2022_2023.csv", index=False)
    show = res.set_index(["pick from", "approach"]).T
    pd.set_option("display.width", 250)
    print(f"\n2022-2023 ({len(market_up.loc['2022':])} days; market-up days {market_up.loc['2022':].mean():.0%}), shares in %:")
    print((show.drop(index=["coins to pick from per day"]) * 100).round(1).to_string())
    print(show.loc[["coins to pick from per day"]].round(0).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
