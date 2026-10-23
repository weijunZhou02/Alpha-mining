"""Step 7, final check: does adding the kept LLM alphas help the step 6 trees blend in 2024? RUN ONCE.

    .venv/bin/python src/llm_final.py

Declared before running (2026-10-23, pushed to GitHub before the first run):
- Contenders:
    "Trees · broad"          the step 6 method, unchanged (159 broad-list alphas)
    "Trees · broad + LLM"    the same, plus every alpha in data/llm/kept.csv (LLM_001, LLM_002)
- Same walk-forward as step 6 (src/combine.py): at the start of each month of 2024, re-learn on all
  earlier days, with settings picked inside each training window. Long the top 20%, short the bottom 20%,
  daily, 0.05% fees plus funding. Kept alphas use the direction found on 2021-2022 and data cut at
  2024-12-31 (2025+ stays locked).
- Verdict, on 2024 Sharpe after fees and funding:
    "helped"               higher Sharpe AND the day-by-day difference has t >= 2
    "no clear difference"  higher Sharpe but t < 2, or lower Sharpe but t > -2
    "hurt"                 lower Sharpe AND t <= -2
- Safety check: "Trees · broad" must reproduce data/eval/combine/results_2024.csv exactly, which proves
  the step 6 method was not changed.
- It refuses to run a second time (the result file is the marker). If a bug is found afterwards, the
  first result is kept and reported next to the fixed one.
Writes data/llm/final_2024.csv, final_2024_quarters.csv, final_2024_daily.parquet.
"""
import time

import numpy as np
import pandas as pd

import combine as cb
import evaluate as ev
import llm_loop as ll
from alphas import AlphaEngine

OUT = ll.OUT
LAST_DAY = ev.PERIODS["valid"][1]          # 2024-12-31
BASE, CAND = "Trees · broad", "Trees · broad + LLM"


def verdict(better, t):
    if better and t >= 2:
        return "helped"
    if not better and t <= -2:
        return "hurt"
    return "no clear difference"


def main():
    result_path = OUT / "final_2024.csv"
    if result_path.exists():
        raise SystemExit(f"The 2024 check has already been run ({result_path.name} exists). It runs only once.")
    t0 = time.time()
    broad = pd.read_csv(cb.EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
    kept = pd.read_csv(OUT / "kept.csv")
    whole, valid = ev.Market("train+valid"), ev.Market("valid")

    ctx = cb.Ctx()
    ctx.rows = cb.Rows(whole.universe)
    X = pd.read_parquet(cb.OUT / "features.parquet")
    assert X.index.equals(ctx.rows.index), "features.parquet does not line up with 2021-2024"
    eng = ll.cut_engine(AlphaEngine.load(), LAST_DAY)               # nothing after 2024 is computed
    fctx = ll.make_context(eng)
    extra = {}
    for k in kept.itertuples():
        tree, why = ll.check_formula(k.formula)
        assert why is None, why
        alpha = ll.compute(tree, fctx, eng.universe).loc[whole.start:whole.end]
        extra[k.id] = np.nan_to_num(ctx.rows.to_long(cb.centred_rank(alpha, whole.universe) * k.direction)).astype(np.float32)
    ctx.X = np.column_stack([X[broad].to_numpy(np.float32)] + list(extra.values()))
    ctx.y = ctx.rows.to_long(cb.centred_rank(whole.fwd[1], whole.universe))
    base_cols = list(range(len(broad)))
    cand_cols = list(range(len(broad) + len(extra)))
    print(f"data ready ({time.time() - t0:.0f}s): {len(broad)} broad alphas + {', '.join(extra)}", flush=True)

    scores, settings = {}, []
    for name, cols in ((BASE, base_cols), (CAND, cand_cols)):
        scores[name], info = cb.walk_forward(ctx, cb.fit_trees(ctx, cols), name)
        settings += info
    daily = {name: cb.judge(s, valid) for name, s in scores.items()}
    res = {name: cb.summarize(d) for name, d in daily.items()}

    frozen = pd.read_csv(cb.OUT / "results_2024.csv").set_index("method").loc[BASE]
    for k in ("ic", "net_bp", "net_sharpe", "turnover"):
        assert abs(res[BASE][k] - frozen[k]) < 1e-9, f"safety check failed: {BASE} {k} {res[BASE][k]} vs frozen {frozen[k]}"
    print("safety check passed: Trees · broad reproduces the step 6 numbers exactly", flush=True)

    diff_bp, diff_t = cb.versus(daily[CAND], daily[BASE])
    better = res[CAND]["net_sharpe"] > res[BASE]["net_sharpe"]
    v = verdict(better, diff_t)
    table = pd.DataFrame(res).T
    table.index.name = "method"
    table["vs_base_bp"] = [0.0, diff_bp]
    table["vs_base_t"] = [np.nan, diff_t]
    table["verdict"] = ["", v]
    table["kept_alphas"] = ["", ", ".join(extra)]
    table.to_csv(result_path)
    q = [{"method": n, "quarter": f"Q{qq}", "net_pct": x.sum() * 100, "sharpe": ev.ann_sharpe(x),
          "ic": d["ic"][d.index.quarter == qq].mean()}
         for n, d in daily.items() for qq, x in d["net"].groupby(d.index.quarter)]
    pd.DataFrame(q).to_csv(OUT / "final_2024_quarters.csv", index=False)
    pd.concat(daily, names=["method", "date"]).to_parquet(OUT / "final_2024_daily.parquet")

    show = ["ic", "ic_t", "gross_bp", "cost_bp", "funding_bp", "net_bp", "net_t", "net_sharpe", "annual_pct",
            "max_drawdown_pct", "turnover", "months_up"]
    pd.set_option("display.width", 220)
    print("\n2024, walked forward month by month:")
    print(table[show].round(3).to_string())
    print(f"\nday-by-day difference: {diff_bp:+.2f} bp/day, t {diff_t:+.2f}  ->  verdict: {v.upper()}")
    print(pd.DataFrame(q).pivot(index="method", columns="quarter", values="net_pct").round(2).to_string())
    print(f"\ndone in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
