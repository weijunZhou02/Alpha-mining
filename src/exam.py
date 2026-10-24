"""Step 8: the final exam. Scores every sealed contender on 2025-01-01..2026-10-21, ONCE.

    .venv/bin/python src/exam.py --rehearsal   # the same code on 2024 (already used up): must reproduce the frozen
                                               # 2024 numbers exactly; nothing after 2024 is graded
    .venv/bin/python src/exam.py --open        # THE EXAM. Needs a passed rehearsal; refuses to run a second time

The plan was sealed in README ("Step 8 plan: the sealed envelope") before anyone looked at 2025-2026:
  contenders  Trees · broad (main, before LLM), Equal weight · core, Single alpha A101_055 (benchmarks),
              Trees · broad + LLM (after LLM), Option B (Trees · broad + LLM, cost-aware)
  rules       walk forward from 2021 month by month, settings picked inside each training window (step 6);
              long the top 20%, short the bottom 20%, daily, 0.05% fees plus funding, tied coins share places
  verdicts    "the LLM helped"  : Sharpe above Trees · broad AND day-by-day t >= 2
              "option B helped" : Sharpe above Trees · broad + LLM AND day-by-day t >= 2
  accuracy    (added 2026-10-24, before opening) approach 1: the trees rank the top 80; the 5 best are called UP,
              the 5 worst DOWN; daily accuracy = right calls / 10, against a random pick of the same coins.
              "Real skill" if the day-by-day edge over random has t >= 3. Headline: Trees · broad + LLM.
  extra       double fees, each year and quarter, the accuracy test on the whole market (not only the top 80).
Writes data/exam/ (the rehearsal writes data/exam/rehearsal_2024/).
"""
import argparse
import json
import subprocess
import time

import numpy as np
import pandas as pd

import combine as cb
import cost_aware as ca
import evaluate as ev
import llm_final as lf
import llm_loop as ll
import top_bottom as tb
from alphas import AlphaEngine
from config import DATA, ROOT

OUT = DATA / "exam"
REHEARSAL = OUT / "rehearsal_2024"
K = 5                                      # calls per side each day: 5 UP + 5 DOWN = 10
SKILL_T, MAYBE_T = 3.0, 2.0
BASE, CAND = "Trees · broad", "Trees · broad + LLM"
EW, SINGLE = "Equal weight · core", f"Single alpha · {cb.BENCHMARK}"
OPT_B = "Option B · Trees · broad + LLM, cost-aware"
TOP80, WHOLE = "top 80", "whole market"
SEALED = {"before-llm": ["data/eval/dedupe/broad_list.csv", "data/eval/dedupe/core_list.csv", "data/eval/scorecard.csv"],
          "after-llm": ["data/llm/kept.csv", "data/eval/cost_aware/choice.json"]}


# ------------------------------------------------------------------ inputs

def sealed_files_unchanged():
    """The alpha lists and settings must be byte-for-byte the ones frozen at the git tags."""
    for tag, paths in SEALED.items():
        for path in paths:
            frozen = subprocess.run(["git", "show", f"{tag}:{path}"], cwd=ROOT, capture_output=True, check=True).stdout
            assert (ROOT / path).read_bytes() == frozen, f"{path} differs from tag {tag}"


def features(eng, start, end, broad, direction, kept):
    """Rows x alphas for every coin-day in the engine's universe: the broad list, then the kept LLM alphas.
    Daily rank inside that universe, flipped so higher = better, centred on 0, missing = 0 (as in step 6)."""
    universe = eng.universe
    rows = cb.Rows(universe.loc[start:end])
    m = universe.loc[start:end]
    cols = {}
    t0 = time.time()
    for i, a in enumerate(broad, 1):
        v = eng.compute(a).loc[start:end]
        cols[a] = np.nan_to_num(rows.to_long(cb.centred_rank(v, m) * direction[a])).astype(np.float32)
        if i % 40 == 0:
            print(f"    {i} alphas ({time.time() - t0:.0f}s)", flush=True)
    fctx = ll.make_context(eng)
    for k in kept.itertuples():
        tree, why = ll.check_formula(k.formula)
        assert why is None, why
        v = ll.compute(tree, fctx, universe).loc[start:end]
        cols[k.id] = np.nan_to_num(rows.to_long(cb.centred_rank(v, m) * k.direction)).astype(np.float32)
    return rows, pd.DataFrame(cols, index=rows.index)


def make_ctx(rows, X, fwd, universe):
    ctx = cb.Ctx()
    ctx.rows, ctx.X = rows, X.to_numpy(np.float32)
    ctx.y = rows.to_long(cb.centred_rank(fwd, universe))       # tomorrow's return, ranked among the day's coins
    return ctx


def walk(ctx, run, months, name):
    """Step 6's walk forward over any months: refit at the start of each month on earlier rows only."""
    t0 = time.time()
    pred = np.full(len(ctx.y), np.nan)
    info = []
    for d0 in months:
        sets = cb.Sets(ctx, d0)
        p, extra = run(sets)
        pred[sets.pred] = p
        info.append({"method": name, "month": d0.strftime("%Y-%m"), **{k: v for k, v in extra.items() if k != "weights"}})
    print(f"  {name:28s} walked forward in {time.time() - t0:6.0f}s", flush=True)
    rows = ~np.isnan(pred)
    return ctx.rows.to_wide(pred[rows], rows), info


# ------------------------------------------------------------------ money

def double_fee(daily):
    d = daily.copy()
    d["net"] = d["net"] - d["cost"]
    d["cost"] = 2 * d["cost"]
    return d


def by_year_and_quarter(daily):
    years, quarters = [], []
    for name, d in daily.items():
        for y, part in d.groupby(d.index.year):
            years.append({"method": name, "year": y, **cb.summarize(part)})
        for (y, q), part in d.groupby([d.index.year, d.index.quarter]):
            quarters.append({"method": name, "quarter": f"{y} Q{q}", "days": len(part), "net_pct": part["net"].sum() * 100,
                             "sharpe": ev.ann_sharpe(part["net"]), "ic": part["ic"].mean()})
    return pd.DataFrame(years), pd.DataFrame(quarters)


# ------------------------------------------------------------------ accuracy: 5 UP calls + 5 DOWN calls a day

def pick_weights(score, ok, best):
    """Each coin's share of the day's K picks (best: the K highest scores, else the K lowest).
    Coins tied at the edge share the places, as in evaluate.group_weights."""
    s = score.where(ok)
    lo = s.rank(axis=1, method="min", ascending=not best) - 1      # a tied block fills places [lo, hi)
    hi = s.rank(axis=1, method="max", ascending=not best)
    w = (hi.clip(upper=K) - lo).clip(lower=0) / (hi - lo)
    return w.where(s.notna().sum(axis=1) >= ev.MIN_COINS, axis=0)


def accuracy_daily(score, universe, fwd, market_up):
    """One row per day: how many of the 10 calls were right, against a random pick from the same coins."""
    days = score.index
    U = universe.reindex(index=days, columns=score.columns, fill_value=False)
    s = score.reindex(index=days, columns=U.columns)
    ok = U & s.notna()                                          # coins the model could pick that day
    top, bot = pick_weights(s, ok, True), pick_weights(s, ok, False)
    R = fwd.reindex(index=days, columns=U.columns).where(ok)
    has = R.notna()
    up, down = (R > 0).astype(float).where(has), (R < 0).astype(float).where(has)
    wt, wb = top.where(has), bot.where(has)                     # a pick with no next-day price is left out
    n_t, n_b = wt.sum(axis=1).where(lambda x: x > 0), wb.sum(axis=1).where(lambda x: x > 0)
    hit_t, hit_b = (wt * up).sum(axis=1), (wb * down).sum(axis=1)
    any_rose, any_fell = up.mean(axis=1), down.mean(axis=1)
    best5, worst5 = pick_weights(R, has, True), pick_weights(R, has, False)   # tomorrow's real 5 best / 5 worst
    n = has.sum(axis=1)
    d = pd.DataFrame({
        "accuracy": (hit_t + hit_b) / (n_t + n_b),
        "random": (n_t * any_rose + n_b * any_fell) / (n_t + n_b),     # same number of UP / DOWN calls, random coins
        "right_of_10": 10 * (hit_t + hit_b) / (n_t + n_b),
        "top5_rose": hit_t / n_t, "any_rose": any_rose,
        "bottom5_fell": hit_b / n_b, "any_fell": any_fell,
        "top_in_best5": (wt * best5).sum(axis=1) / n_t, "bottom_in_worst5": (wb * worst5).sum(axis=1) / n_b,
        "chance_in_5": K / n, "coins": n, "market_up": market_up.reindex(days),
    })
    d = d[n_t.notna() & n_b.notna() & (n >= ev.MIN_COINS)]       # the last day has no answer yet
    d["edge"] = d["accuracy"] - d["random"]
    return d


def skill(t):
    return "real skill" if t >= SKILL_T else "maybe" if t >= MAYBE_T else "no clear skill"


def accuracy_summary(d):
    """Means are over days (every day counts the same); t-stats are day by day (Newey-West, 1 lag)."""
    t = ev.nw_tstat(d["edge"], 1)
    acc = d["accuracy"]
    return {
        "days": len(d), "accuracy_pct": acc.mean() * 100, "random_pct": d["random"].mean() * 100,
        "edge_points": d["edge"].mean() * 100, "edge_t": t, "verdict": skill(t),
        "days_beating_random_pct": (d["edge"] > 0).mean() * 100,
        "days_60pct_or_more": (acc >= 0.6 - 1e-9).mean() * 100, "days_70pct_or_more": (acc >= 0.7 - 1e-9).mean() * 100,
        "days_80pct_or_more": (acc >= 0.8 - 1e-9).mean() * 100, "days_40pct_or_less": (acc <= 0.4 + 1e-9).mean() * 100,
        "top5_rose_pct": d["top5_rose"].mean() * 100, "any_rose_pct": d["any_rose"].mean() * 100,
        "top5_edge_t": ev.nw_tstat(d["top5_rose"] - d["any_rose"], 1),
        "bottom5_fell_pct": d["bottom5_fell"].mean() * 100, "any_fell_pct": d["any_fell"].mean() * 100,
        "bottom5_edge_t": ev.nw_tstat(d["bottom5_fell"] - d["any_fell"], 1),
        "top_in_best5_pct": d["top_in_best5"].mean() * 100, "bottom_in_worst5_pct": d["bottom_in_worst5"].mean() * 100,
        "chance_in_5_pct": d["chance_in_5"].mean() * 100, "coins_per_day": d["coins"].mean(),
    }


def accuracy_tables(scores, pools, fwd, market_up):
    """scores: {(contender, pool): score}; pools: {pool: universe}. Returns (summary, daily, histogram)."""
    summary, daily, hist = [], {}, []
    for (name, pool), s in scores.items():
        d = accuracy_daily(s, pools[pool], fwd, market_up)
        daily[(name, pool)] = d
        parts = {"all days": d, "market-DOWN days": d[d["market_up"] == 0], "market-UP days": d[d["market_up"] == 1]}
        parts.update({str(y): part for y, part in d.groupby(d.index.year)})
        for label, part in parts.items():
            summary.append({"method": name, "pick from": pool, "days used": label, **accuracy_summary(part)})
        counts = d["right_of_10"].round().astype(int).value_counts().reindex(range(11), fill_value=0)
        hist.append({"method": name, "pick from": pool, **{f"{k} of 10": v for k, v in counts.items()}})
    return pd.DataFrame(summary), pd.concat(daily, names=["method", "pick from", "date"]), pd.DataFrame(hist)


# ------------------------------------------------------------------ one run (2024 rehearsal or the exam)

def run(whole, target, out, extra_months=None):
    """Walk forward from 2021 through `target`'s months and grade every contender on `target`.
    whole: Market covering 2021 .. target's end. extra_months: limit the whole-market extra (rehearsal only)."""
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    core = pd.read_csv(cb.EVAL / "dedupe" / "core_list.csv", index_col=0).index.tolist()
    assert core[0] == cb.BENCHMARK
    broad = pd.read_csv(cb.EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
    direction = pd.read_csv(cb.EVAL / "scorecard.csv").set_index("id")["direction"]
    kept = pd.read_csv(ll.OUT / "kept.csv")
    choice = json.loads((ca.OUT / "choice.json").read_text(encoding="utf-8"))
    assert choice["enters_exam"]
    version = ca.Version(**choice["chosen"])
    eng = ll.cut_engine(AlphaEngine.load(), whole.end)            # nothing after the target period exists
    months = pd.date_range(target.start, target.end, freq="MS")

    print(f"features, top 80, {whole.start} .. {whole.end}:", flush=True)
    rows, X = features(eng, whole.start, whole.end, broad, direction, kept)
    frozen = pd.read_parquet(cb.OUT / "features.parquet")
    old = X.index.get_level_values(0) <= pd.Timestamp(ev.PERIODS["valid"][1])
    assert X.index[old].equals(frozen.index) and np.array_equal(X.loc[old, broad].to_numpy(), frozen[broad].to_numpy()), \
        "the broad-list features for 2021-2024 differ from step 6's features.parquet"
    print(f"  {X.shape[0]:,} coin-days x {X.shape[1]} alphas; 2021-2024 part equals step 6's features exactly "
          f"({time.time() - t0:.0f}s)", flush=True)
    ctx = make_ctx(rows, X, whole.fwd[1], whole.universe)
    names = list(X.columns)
    base_cols, cand_cols = list(range(len(broad))), list(range(len(names)))
    core_cols = [names.index(a) for a in core]

    scores, settings = {}, []
    for name, fit in ((EW, cb.fit_equal(ctx, core_cols)), (BASE, cb.fit_trees(ctx, base_cols)),
                      (CAND, cb.fit_trees(ctx, cand_cols))):
        scores[name], info = walk(ctx, fit, months, name)
        settings += info
    scores[SINGLE] = ctx.rows.to_wide(X[cb.BENCHMARK].to_numpy(), np.ones(len(ctx.y), bool))

    # Honesty check (as in step 6): ridge trained on answers shuffled within each day must score about 0.
    rng = np.random.default_rng(42)
    known = np.flatnonzero(~np.isnan(ctx.y))
    y_shuffled = ctx.y.copy()
    y_shuffled[known] = ctx.y[known[np.lexsort((rng.random(len(known)), ctx.rows.ri[known]))]]
    noise, _ = walk(ctx, cb.fit_ridge(ctx, base_cols, y=y_shuffled), months, "Shuffled answers")

    # ---- money
    daily = {name: cb.judge(scores[name], target) for name in (BASE, EW, SINGLE, CAND)}
    vol = eng.p["returns"].rolling(ca.VOL_DAYS, min_periods=20).std()
    daily[OPT_B] = ca.run_version(version, {1: scores[CAND]}, target, vol)
    res = pd.DataFrame({k: cb.summarize(d) for k, d in daily.items()}).T
    res.index.name = "method"
    res["double_fee_sharpe"] = [cb.summarize(double_fee(daily[k]))["net_sharpe"] for k in res.index]
    for col, ref in (("vs_trees_broad", BASE), ("vs_trees_broad_llm", CAND)):
        pairs = [cb.versus(daily[k], daily[ref]) for k in res.index]
        res[f"{col}_bp"], res[f"{col}_t"] = [p[0] for p in pairs], [p[1] for p in pairs]
    llm_better = bool(res.at[CAND, "net_sharpe"] > res.at[BASE, "net_sharpe"])
    b_better = bool(res.at[OPT_B, "net_sharpe"] > res.at[CAND, "net_sharpe"])
    verdicts = {
        "the LLM helped?": lf.verdict(llm_better, res.at[CAND, "vs_trees_broad_t"]),
        "option B helped?": lf.verdict(b_better, res.at[OPT_B, "vs_trees_broad_llm_t"]),
        "best Sharpe after costs": res["net_sharpe"].idxmax(),
    }
    noise_res = cb.summarize(cb.judge(noise, target))
    years, quarters = by_year_and_quarter(daily)

    # ---- accuracy: top 80 for every score; the whole market for the headline model only
    med80 = whole.fwd[1].where(whole.universe).median(axis=1)
    market_up = (med80 > 0).where(med80.notna())
    pools = {TOP80: whole.universe}
    acc_scores = {(name, TOP80): scores[name].loc[target.start:target.end] for name in (CAND, BASE, EW, SINGLE)}

    print(f"features, whole market:", flush=True)
    all_mask = tb.all_coins_mask(eng).reindex(columns=eng.universe.columns, fill_value=False)
    eng_w = AlphaEngine(eng.p, all_mask, eng.sectors, eng.onboard)   # cross-sectional steps compare all coins
    rows_w, X_w = features(eng_w, whole.start, whole.end, broad, direction, kept)
    ctx_w = make_ctx(rows_w, X_w, whole.fwd[1], all_mask.loc[whole.start:whole.end])
    print(f"  {X_w.shape[0]:,} coin-days x {X_w.shape[1]} alphas", flush=True)
    w_months = months if extra_months is None else months[-extra_months:]
    acc_scores[(CAND, WHOLE)], info = walk(ctx_w, cb.fit_trees(ctx_w, cand_cols), w_months, f"{CAND} ({WHOLE})")
    settings += info
    pools[WHOLE] = all_mask
    acc, acc_daily, hist = accuracy_tables(acc_scores, pools, target.fwd[1], market_up)

    # ---- save
    res.to_csv(out / "results.csv")
    years.to_csv(out / "years.csv", index=False)
    quarters.to_csv(out / "quarters.csv", index=False)
    pd.concat(daily, names=["method", "date"]).to_parquet(out / "daily.parquet")
    pd.DataFrame(settings).to_csv(out / "settings.csv", index=False)
    acc.to_csv(out / "accuracy.csv", index=False)
    acc_daily.to_parquet(out / "accuracy_daily.parquet")
    hist.to_csv(out / "accuracy_histogram.csv", index=False)
    pd.concat({f"{n} | {p}": s.loc[target.start:target.end] for (n, p), s in acc_scores.items()},
              names=["method", "date"]).to_parquet(out / "scores.parquet")
    checks = {"shuffled answers IC": noise_res["ic"], "shuffled answers IC t": noise_res["ic_t"],
              "shuffled answers Sharpe": noise_res["net_sharpe"], "days": int(res.at[BASE, "days"]),
              "market-up days share": float(market_up.loc[target.start:target.end].mean())}
    (out / "verdicts.json").write_text(json.dumps({**verdicts, **checks}, indent=2, default=float), encoding="utf-8")

    # ---- print
    pd.set_option("display.width", 250)
    show = ["ic", "ic_t", "gross_bp", "cost_bp", "funding_bp", "net_bp", "net_t", "net_sharpe", "double_fee_sharpe",
            "annual_pct", "max_drawdown_pct", "turnover", "months_up", "months"]
    print(f"\n{target.start} .. {target.end}, walked forward month by month (money after fees and funding):")
    print(res[show].round(3).to_string())
    print(f"\nLLM: {CAND} vs {BASE}: {res.at[CAND, 'vs_trees_broad_bp']:+.2f} bp/day, "
          f"t {res.at[CAND, 'vs_trees_broad_t']:+.2f} -> {verdicts['the LLM helped?'].upper()}")
    print(f"Option B vs {CAND}: {res.at[OPT_B, 'vs_trees_broad_llm_bp']:+.2f} bp/day, "
          f"t {res.at[OPT_B, 'vs_trees_broad_llm_t']:+.2f} -> {verdicts['option B helped?'].upper()}")
    print(f"shuffled answers: IC {noise_res['ic']:+.4f} (t {noise_res['ic_t']:+.2f}), Sharpe {noise_res['net_sharpe']:+.2f}")
    print("\nEach year, Sharpe after costs:")
    print(years.pivot(index="method", columns="year", values="net_sharpe").round(2).to_string())
    a = acc.set_index(["method", "pick from", "days used"])
    cols = ["days", "accuracy_pct", "random_pct", "edge_points", "edge_t", "verdict", "days_beating_random_pct",
            "days_70pct_or_more", "top5_rose_pct", "any_rose_pct", "top5_edge_t", "bottom5_fell_pct", "any_fell_pct",
            "bottom5_edge_t", "top_in_best5_pct", "bottom_in_worst5_pct", "chance_in_5_pct"]
    print("\nAccuracy: 5 UP calls + 5 DOWN calls a day (percent):")
    print(a[cols].round(2).to_string())
    print("\nDays by number of right calls out of 10:")
    print(hist.set_index(["method", "pick from"]).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")
    return res, daily, acc


def rehearsal():
    """2024 with the exam code: the before contenders must give exactly step 6's numbers, and Trees · broad + LLM
    exactly the step 7 check's. 2024 is already used up; nothing here changes any choice."""
    res, _, _ = run(ev.Market("train+valid"), ev.Market("valid"), REHEARSAL, extra_months=2)
    step6 = pd.read_csv(cb.OUT / "results_2024.csv").set_index("method")
    step7 = pd.read_csv(ll.OUT / "final_2024.csv").set_index("method")
    rows = []
    for name in (BASE, EW, SINGLE, CAND):
        frozen = (step7 if name == CAND else step6).loc[name]
        for k in ("ic", "net_bp", "net_sharpe", "turnover"):
            rows.append({"method": name, "number": k, "exam code": res.at[name, k], "frozen": frozen[k],
                         "same": abs(res.at[name, k] - frozen[k]) < 1e-9})
    check = pd.DataFrame(rows)
    check.to_csv(REHEARSAL / "safety_check.csv", index=False)
    passed = bool(check["same"].all())
    (REHEARSAL / "safety_check.json").write_text(json.dumps({"passed": passed}), encoding="utf-8")
    print("\nSafety check (2024, exam code vs frozen numbers):")
    print(check.to_string(index=False))
    print("PASSED" if passed else "FAILED: do not open the exam")


def open_exam():
    if (OUT / "results.csv").exists():
        raise SystemExit("The exam has already been opened (data/exam/results.csv exists). It runs only once.")
    safety = REHEARSAL / "safety_check.json"
    if not safety.exists() or not json.loads(safety.read_text(encoding="utf-8"))["passed"]:
        raise SystemExit("Run the rehearsal first (--rehearsal); its safety check must pass.")
    run(ev.Market("all", unlock_test=True), ev.Market("test", unlock_test=True), OUT)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--rehearsal", action="store_true")
    g.add_argument("--open", action="store_true")
    args = ap.parse_args()
    sealed_files_unchanged()
    rehearsal() if args.rehearsal else open_exam()


if __name__ == "__main__":
    main()
