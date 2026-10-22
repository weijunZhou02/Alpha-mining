"""Step 3: score every alpha on the training period (2021-2023).

    .venv/bin/python src/run_step3.py

Writes to data/eval/:
    scorecard.csv          one row per alpha (catalog info + scores)
    daily_ic.parquet       date x alpha: daily Rank IC (needed for combining later)
    daily_ls_net.parquet   date x alpha: daily long-short return after fees and funding
    alpha_ranks.parquet    (date, coin) x alpha: each alpha's daily rank inside the universe (for duplicates, step 5)
Also scores two fake alphas first, to prove the evaluator is honest:
    RANDOM  - pure noise: must score about zero
    ORACLE  - peeks at tomorrow's return: must score close to perfect (real alphas never can)
"""
import time

import numpy as np
import pandas as pd

import evaluate as ev
from alphas import NOT_COMPUTABLE, AlphaEngine
from config import DATA

OUT = DATA / "eval"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = ev.Market("train")
    eng = AlphaEngine.load()
    cards, ic, net, ranks = [], {}, {}, {}

    # --- honesty checks
    rng = np.random.default_rng(42)
    fake = {
        "RANDOM": pd.DataFrame(rng.normal(size=m.universe.shape), index=m.dates, columns=m.universe.columns),
        "ORACLE": m.fwd[1],
    }
    for name, values in fake.items():
        card, _ = ev.score(values, m)
        print(f"{name:7s} IC={card['ic']:+.4f}  t={card['ic_t']:+.1f}  long-short={card['ls_bp']:+.0f} bp/day"
              f"  verdict={card['verdict']}")
        cards.append({"id": name, "library": "check", "name": name, "category": "honesty check", **card})

    # --- every computable alpha
    t0 = time.time()
    for i, (alpha_id, row) in enumerate(eng.catalog.items(), 1):
        if alpha_id in NOT_COMPUTABLE:
            continue
        values = eng.compute(alpha_id).loc[m.start:m.end]
        card, series = ev.score(values, m)
        cards.append({"id": alpha_id, "library": row["library"], "name": row["name"],
                      "category": row["category"], "crypto_fit": row["crypto_fit"], **card})
        ic[alpha_id] = series["ic"]
        net[alpha_id] = series["net"]
        r = values.where(m.universe).rank(axis=1, pct=True)
        ranks[alpha_id] = r.stack().astype("float32")
        if i % 50 == 0:
            print(f"  {i} alphas scored ({time.time() - t0:.0f}s)", flush=True)

    sc = pd.DataFrame(cards)
    sc.to_csv(OUT / "scorecard.csv", index=False)
    pd.DataFrame(ic).to_parquet(OUT / "daily_ic.parquet")
    pd.DataFrame(net).to_parquet(OUT / "daily_ls_net.parquet")
    pd.DataFrame(ranks).to_parquet(OUT / "alpha_ranks.parquet")
    print(f"scored {len(ic)} alphas in {time.time() - t0:.0f}s")
    real = sc[sc.library != "check"]
    print("rank-only rule:", real["verdict_rank_only"].value_counts().to_dict())
    print("final verdict: ", real["verdict"].value_counts().to_dict())
    print("survives fees + funding (net t >= 2):", int(real["survives_costs"].sum()))


if __name__ == "__main__":
    main()
