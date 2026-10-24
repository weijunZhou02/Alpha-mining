"""After the exam (extra information, changes no result): the shuffled-answers check with 20 more seeds.

    .venv/bin/python src/exam_noise_check.py     # about 4 min -> data/exam/shuffled_answers_20_seeds.csv

The exam's one shuffled-answers run (seed 42) gave IC +0.018, t 2.9 on 2025-2026. Each run is one random mix of
the alphas used for the whole period, so its own t-stat says little; the spread over many seeds is the fair yardstick.
Also reports each random mix's daily accuracy (5 UP + 5 DOWN calls), to compare with the real model's.
"""
import numpy as np
import pandas as pd

import combine as cb
import evaluate as ev
import exam as ex
import llm_loop as ll
from alphas import AlphaEngine

whole, target = ev.Market("all", unlock_test=True), ev.Market("test", unlock_test=True)
broad = pd.read_csv(cb.EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
direction = pd.read_csv(cb.EVAL / "scorecard.csv").set_index("id")["direction"]
kept = pd.read_csv(ll.OUT / "kept.csv")
eng = ll.cut_engine(AlphaEngine.load(), whole.end)
rows, X = ex.features(eng, whole.start, whole.end, broad, direction, kept)
ctx = ex.make_ctx(rows, X, whole.fwd[1], whole.universe)
months = pd.date_range(target.start, target.end, freq="MS")
med80 = whole.fwd[1].where(whole.universe).median(axis=1)
market_up = (med80 > 0).where(med80.notna())
known = np.flatnonzero(~np.isnan(ctx.y))
out = []
for seed in [42] + list(range(1, 21)):
    rng = np.random.default_rng(seed)
    ys = ctx.y.copy()
    ys[known] = ctx.y[known[np.lexsort((rng.random(len(known)), ctx.rows.ri[known]))]]
    s, _ = ex.walk(ctx, cb.fit_ridge(ctx, list(range(len(broad))), y=ys), months, f"shuffled seed {seed}")
    r = cb.summarize(cb.judge(s, target))
    a = ex.accuracy_summary(ex.accuracy_daily(s, whole.universe, target.fwd[1], market_up))
    out.append({"seed": seed, "ic": r["ic"], "ic_t": r["ic_t"], "net_sharpe": r["net_sharpe"],
                "accuracy_pct": a["accuracy_pct"], "accuracy_edge_t": a["edge_t"]})
d = pd.DataFrame(out)
d.to_csv("data/exam/shuffled_answers_20_seeds.csv", index=False)
print(d.round(4).to_string(index=False))
x = d[d.seed != 42]
print(f"\n20 new seeds: mean IC {x.ic.mean():+.4f}, spread (sd) {x.ic.std():.4f}, |t| >= 2 in {(x.ic_t.abs() >= 2).sum()} of 20, "
      f"IC above +0.018 in {(x.ic >= 0.018).sum()} of 20")
print(f"accuracy of random mixes: mean {x.accuracy_pct.mean():.1f}%, best {x.accuracy_pct.max():.1f}%, worst {x.accuracy_pct.min():.1f}% "
      f"(Trees · broad + LLM: 55.9%)")
