"""Step 4: prepare the data files behind the scorecard page.

    .venv/bin/python src/build_report_data.py      # after run_step3.py

Writes data/eval/report/:
    scorecard.csv     one row per alpha: scores (rounded as shown on the page) + catalog text
    weekly.csv        per alpha and week: running total of daily IC and of net long-short return,
                      both in the alpha's own direction
    klen_groups.csv   the day-range example: average vs typical (median) next-day return per group
    honesty.csv       the RANDOM and ORACLE fake alphas
"""
import numpy as np
import pandas as pd

import evaluate as ev
from alphas import AlphaEngine, load_catalog
from config import DATA

EVAL = DATA / "eval"
OUT = EVAL / "report"

ROUND = {"ic": 4, "ic_t": 2, "icir": 3, "ic_win": 3, "ic_2021": 4, "ic_2022": 4, "ic_2023": 4,
         "ic_h3": 4, "ic_h7": 4, "ic_h14": 4, "ic_late": 4, "coverage": 3, "monotonic": 2,
         "g1_bp": 1, "g2_bp": 1, "g3_bp": 1, "g4_bp": 1, "g5_bp": 1, "ls_bp": 1, "ls_t": 2,
         "ls_capped_bp": 1, "turnover": 2, "cost_bp": 1, "funding_bp": 1, "net_bp": 1, "net_t": 2,
         "net_annual_pct": 1}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(EVAL / "scorecard.csv")
    cat = load_catalog()

    honest = sc[sc.library == "check"][["id", "ic", "ic_t", "ls_bp", "verdict"]].copy()
    honest["ic_t"] = honest["ic_t"].map(lambda t: "inf" if np.isinf(t) else round(t, 2))
    honest["ic"] = honest["ic"].round(4)
    honest["ls_bp"] = honest["ls_bp"].round(1)
    honest["what"] = honest["id"].map({"RANDOM": "Pure noise: must score about zero",
                                       "ORACLE": "Peeks at tomorrow's return: must score about one"})
    honest.to_csv(OUT / "honesty.csv", index=False)

    real = sc[sc.library != "check"].copy()
    for col, d in ROUND.items():
        real[col] = real[col].round(d)
    real["ic_dir_t"] = (real["ic_t"] * real["direction"]).round(2)  # rank score in the alpha's own direction
    real["plain_english"] = real["id"].map(lambda i: cat[i]["plain_english"])
    real["formula"] = real["id"].map(lambda i: cat[i]["formula"])
    real["source"] = real["id"].map(lambda i: cat[i]["source"])
    real["year"] = real["id"].map(lambda i: cat[i]["year"])
    # Step 5 results, when present: which list each alpha is in, or which kept alpha it duplicates.
    dd = EVAL / "dedupe"
    has_dedupe = (dd / "membership.csv").exists()
    if has_dedupe:
        mem = pd.read_csv(dd / "membership.csv").set_index("id")
        real["kept"] = real["id"].map(mem["kept"])
        real["twin_of"] = real["id"].map(mem["twin_of"]).fillna("")
        real["twin_check"] = real["id"].map(mem["check"]).fillna("")
        real["twin_similarity"] = real["id"].map(mem["similarity"])
        core = pd.read_csv(dd / "core_list.csv", index_col=0)
        fam = core[["family", "family_note", "verdict", "category", "ls_t", "net_bp", "turnover", "n_twins", "twins"]].copy()
        fam["ls_t"] = fam["ls_t"].round(2)
        fam["net_bp"] = fam["net_bp"].round(1)
        fam["turnover"] = fam["turnover"].round(2)
        fam.index.name = "id"
        fam.to_csv(OUT / "families.csv")
        S = pd.read_parquet(dd / "signal_similarity.parquet")
        M = pd.read_parquet(dd / "money_similarity.parquet")
        ids = list(core.index)
        pairs = [{"a": a, "b": b, "signal": round(float(S.at[a, b]), 2), "money": round(float(M.at[a, b]), 2)}
                 for a in ids for b in ids]
        pd.DataFrame(pairs).to_csv(OUT / "core_similarity.csv", index=False)

    keep = ["id", "library", "name", "category", "verdict", "verdict_rank_only", "direction", "survives_costs",
            "ic", "ic_t", "ic_dir_t", "icir", "ic_win", "ic_2021", "ic_2022", "ic_2023", "years_same_sign",
            "ic_h3", "ic_h7", "ic_h14", "ic_late", "coverage", "g1_bp", "g2_bp", "g3_bp", "g4_bp", "g5_bp",
            "monotonic", "ls_bp", "ls_t", "ls_capped_bp", "turnover", "cost_bp", "funding_bp", "net_bp", "net_t",
            "net_annual_pct", "plain_english", "formula", "source", "year"]
    if has_dedupe:
        keep += ["kept", "twin_of", "twin_check", "twin_similarity"]
    real[keep].to_csv(OUT / "scorecard.csv", index=False)

    direction = real.set_index("id")["direction"]
    ic = pd.read_parquet(EVAL / "daily_ic.parquet")
    net = pd.read_parquet(EVAL / "daily_ls_net.parquet")
    cum_ic = (ic.fillna(0) * direction.reindex(ic.columns)).cumsum().resample("W-SUN").last()
    cum_net = (net.fillna(0).cumsum() * 100).resample("W-SUN").last()   # in % of capital (simple sum)
    weekly = pd.concat({"cum_ic": cum_ic.stack(), "cum_net_pct": cum_net.stack()}, axis=1).reset_index()
    weekly.columns = ["week", "id", "cum_ic", "cum_net_pct"]
    weekly["week"] = weekly["week"].dt.strftime("%Y-%m-%d")
    weekly["cum_ic"] = weekly["cum_ic"].round(3)
    weekly["cum_net_pct"] = weekly["cum_net_pct"].round(2)
    weekly[["id", "week", "cum_ic", "cum_net_pct"]].to_csv(OUT / "weekly.csv", index=False)

    m = ev.Market("train")
    a = AlphaEngine.load().compute("Q158_KLEN").loc[m.start:m.end]
    g = ev.group_weights(a, m.universe)
    r = m.fwd[1].where(m.universe)
    rows = []
    for k in range(1, 6):
        x = r.where(g[k] >= 0.5).stack().dropna()   # KLEN has almost no ties: a coin is (mostly) in one group
        label = {1: "1 · calmest coins", 5: "5 · wildest coins"}.get(k, str(k))
        rows.append({"group": k, "label": label, "mean_bp": round(x.mean() * 1e4, 1),
                     "median_bp": round(x.median() * 1e4, 1), "coin_days": len(x)})
    pd.DataFrame(rows).to_csv(OUT / "klen_groups.csv", index=False)

    if (EVAL / "combine" / "results_2024.csv").exists():
        combine_files(sc, cat)

    for f in sorted(OUT.glob("*.csv")):
        print(f"{f.name:20s} {f.stat().st_size / 1e6:6.2f} MB")


def combine_files(sc, cat):
    """Step 6 results for the page: one row per method, running totals, quarters, single alphas, weights."""
    cb = EVAL / "combine"
    res = pd.read_csv(cb / "results_2024.csv")
    keep = ["method", "kind", "ic", "ic_t", "gross_bp", "cost_bp", "funding_bp", "net_bp", "net_t", "net_sharpe",
            "annual_pct", "max_drawdown_pct", "turnover", "months_up", "months", "vs_equal_bp", "vs_equal_t",
            "vs_single_bp", "vs_single_t", "passes_bar"]
    res[keep].round({"ic": 4, "ic_t": 2, "gross_bp": 1, "cost_bp": 1, "funding_bp": 1, "net_bp": 1, "net_t": 2,
                     "net_sharpe": 2, "annual_pct": 1, "max_drawdown_pct": 1, "turnover": 2, "vs_equal_bp": 1,
                     "vs_equal_t": 2, "vs_single_bp": 1, "vs_single_t": 2}).assign(
        passes_bar=lambda d: d["passes_bar"].map({True: "yes", False: "no"}),
        months_up=lambda d: d["months_up"].astype(int)).to_csv(OUT / "combine_results.csv", index=False)

    daily = pd.read_parquet(cb / "daily_2024.parquet")
    cum = (daily["net"].unstack("method").fillna(0).cumsum() * 100).round(2)
    cum = cum.stack().rename("cum_net_pct").reset_index()
    cum["date"] = cum["date"].dt.strftime("%Y-%m-%d")
    cum[["method", "date", "cum_net_pct"]].to_csv(OUT / "combine_daily.csv", index=False)

    q = pd.read_csv(cb / "quarters_2024.csv")
    q.round({"net_pct": 2, "sharpe": 2, "ic": 4}).to_csv(OUT / "combine_quarters.csv", index=False)

    s = pd.read_csv(cb / "singles.csv").set_index("id")
    fam = pd.read_csv(EVAL / "dedupe" / "core_list.csv", index_col=0)["family"]
    out = pd.DataFrame({
        "id": s.index, "family": fam.reindex(s.index).values,
        "train_sharpe": s["train_net_sharpe"].round(2).values, "y2024_sharpe": s["y2024_net_sharpe"].round(2).values,
        "train_net_bp": s["train_net_bp"].round(1).values, "y2024_net_bp": s["y2024_net_bp"].round(1).values,
        "y2024_ic": s["y2024_ic"].round(4).values, "y2024_turnover": s["y2024_turnover"].round(2).values,
        "plain_english": [cat[i]["plain_english"] for i in s.index]})
    out.to_csv(OUT / "combine_singles.csv", index=False)

    w = pd.read_csv(cb / "weights.csv")
    w = w[w["method"].str.endswith("core")].copy()
    w["weight"] = w["weight"].round(4)
    w["family"] = w["alpha"].map(fam)
    w.to_csv(OUT / "combine_weights.csv", index=False)

    chk = pd.read_csv(cb / "checks.csv")
    chk["what"] = ["Shuffled answers" if w.startswith("Shuffled") else "Equal weight in 2021-2023 (in-sample: flattering)"
                   for w in chk["what"]]
    chk.round({"ic": 4, "ic_t": 2, "net_bp": 1, "net_sharpe": 2}).to_csv(OUT / "combine_checks.csv", index=False)


if __name__ == "__main__":
    main()
