"""Compute every catalog alpha once on the real panels and record whether it worked.

    .venv/bin/python src/check_all_alphas.py

This is a correctness/health check only (no prediction scores). Writes data/checks/alpha_compute_check.csv.
"""
import time
import traceback

import numpy as np
import pandas as pd

from alphas import NOT_COMPUTABLE, AlphaEngine
from config import DATA, RESEARCH_START


def main():
    eng = AlphaEngine.load()
    uni = eng.universe.loc[RESEARCH_START:]
    n_cells = int(uni.sum().sum())
    rows = []
    t_all = time.time()
    for alpha_id, row in eng.catalog.items():
        rec = {"id": alpha_id, "library": row["library"], "category": row["category"]}
        if alpha_id in NOT_COMPUTABLE:
            rec.update(status="skipped", note=NOT_COMPUTABLE[alpha_id])
            rows.append(rec)
            continue
        t0 = time.time()
        try:
            v = eng.compute(alpha_id).loc[RESEARCH_START:]
            vals = v.where(uni).to_numpy()
            finite = np.isfinite(vals)
            n_ok = int(finite.sum())
            per_day = finite.sum(axis=1)
            flat_days = int(sum(1 for i in range(len(vals)) if per_day[i] > 1 and np.nanstd(vals[i]) == 0))
            rec.update(status="ok", coverage=round(n_ok / n_cells, 4),
                       median_coins_per_day=int(np.median(per_day)),
                       days_all_equal=flat_days,
                       seconds=round(time.time() - t0, 2))
            if n_ok == 0:
                rec["status"] = "empty"
        except Exception as e:  # noqa: BLE001 - we want to list every failure
            rec.update(status="error", note=f"{type(e).__name__}: {e}"[:200], seconds=round(time.time() - t0, 2))
            traceback.print_exc(limit=2)
        rows.append(rec)
        print(f"{alpha_id:28s} {rec['status']:8s} {rec.get('coverage', '')}", flush=True)

    df = pd.DataFrame(rows)
    out = DATA / "checks"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "alpha_compute_check.csv", index=False)
    print(f"\n{len(df)} alphas in {time.time() - t_all:.0f}s")
    print(df["status"].value_counts().to_string())
    ok = df[df.status == "ok"]
    print("coverage of universe coin-days: median", ok["coverage"].median(), "| below 90%:", int((ok["coverage"] < 0.9).sum()))


if __name__ == "__main__":
    main()
