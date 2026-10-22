"""Step 5: remove duplicate alphas (training period 2021-2023 only).

    .venv/bin/python src/dedupe.py      # after run_step3.py

Two alphas are duplicates if EITHER check says they are too similar (>= 0.7):
  signal: do they rank coins the same way each day? (correlation of daily ranks, sign ignored,
          so a flipped twin is still a twin)
  money:  do their daily long-short profits move together? (each in its own direction)
Picking: go down the list from best to worst and keep an alpha only if it is not too similar to
any alpha already kept. If a twin is almost as good (>= 90% of the money t) but trades much less
(<= 70% of the turnover), the cheaper one is kept instead.

Lists:
  core   strong + maybe alphas, ordered by money t                     -> simple combinations
  broad  core first, then every other alpha with some signal (|rank t| >= 2 or money t >= 2),
         ordered by |rank t| (their money t is near zero, so it can't order them)  -> learning models
Writes data/eval/dedupe/.
"""
import numpy as np
import pandas as pd

from config import DATA

EVAL = DATA / "eval"
OUT = EVAL / "dedupe"
THRESHOLD = 0.7
ALMOST_AS_GOOD, MUCH_CHEAPER = 0.9, 0.7

# Families of the core list, named by reading the formulas and checking which ones make money on the
# same days (money similarity 0.3-0.7 inside a family, mostly below 0.2 between families).
FAMILIES = {
    "A101_055": ("Price-volume divergence", "Price near the top of its range without a volume rush is good; with a rush, bad."),
    "CRYPTO_HMRW_PREMIUM": ("Perp premium", "Buy coins whose perpetual trades cheap against spot, short the expensive ones."),
    "A101_067": ("Fade recent pushes", "Bet against coins that just made a fresh high or jumped in the last day or two."),
    "GTJA_108": ("Fade recent pushes", "Bet against coins that just made a fresh high or jumped in the last day or two."),
    "GTJA_131": ("Fade recent pushes", "Bet against coins that just made a fresh high or jumped in the last day or two."),
    "A101_001": ("Fade recent pushes", "Bet against coins that just made a fresh high or jumped in the last day or two."),
    "Q158_IMAX5": ("Fade recent pushes", "Bet against coins that just made a fresh high or jumped in the last day or two."),
    "GTJA_102": ("Volume cooling", "Buy coins whose trading is quieter than usual, short coins where volume is surging."),
    "Q158_VMA60": ("Volume cooling", "Buy coins whose trading is quieter than usual, short coins where volume is surging."),
    "A101_007": ("Volume cooling", "Buy coins whose trading is quieter than usual, short coins where volume is surging."),
    "GTJA_095": ("Size and liquidity", "Buy coins with smaller, steadier trading value (the Liu-Tsyvinski-Wu effect)."),
    "Q158_WVMA5": ("Uneven volume", "Buy coins whose price-moving volume is steady, short coins with bursty volume."),
    "Q158_WVMA20": ("Uneven volume", "Buy coins whose price-moving volume is steady, short coins with bursty volume."),
    "A101_073": ("Big recent movers", "Short coins that moved a lot in either direction over the last few days."),
    "A101_059": ("Sector price-volume link", "Buy coins whose sector-adjusted price has not moved with volume lately, short those where it has."),
}


def signal_similarity(ranks):
    """|correlation| between alphas' daily ranks, pooled over all coin-days (missing = neutral)."""
    X = ranks.to_numpy(dtype=np.float64)
    X = np.nan_to_num(X - np.nanmean(X, axis=0))
    norm = np.sqrt((X ** 2).sum(axis=0))
    norm[norm == 0] = 1
    S = np.abs((X.T @ X) / np.outer(norm, norm))
    np.fill_diagonal(S, 1.0)
    return pd.DataFrame(S, index=ranks.columns, columns=ranks.columns)


def money_similarity(net):
    """Correlation between alphas' daily long-short profits (each already in its own direction)."""
    M = net.corr(min_periods=250).fillna(0).to_numpy().copy()
    np.fill_diagonal(M, 1.0)
    return pd.DataFrame(M, index=net.columns, columns=net.columns)


def pick(order, S, M, sc, kept=None):
    """Greedy pick. Returns (kept list, {alpha: (kept twin, which check, similarity)}).
    Alphas passed in `kept` are already decided and are never swapped out."""
    protected = set(kept or [])
    kept = list(kept or [])
    twin_of = {}
    for a in order:
        if a in kept:
            continue
        hits = [(k, "signal", S.at[a, k]) for k in kept if S.at[a, k] >= THRESHOLD]
        hits += [(k, "money", M.at[a, k]) for k in kept if M.at[a, k] >= THRESHOLD]
        if not hits:
            kept.append(a)
            continue
        twins = {k for k, _, _ in hits}
        k, how, sim = max(hits, key=lambda h: h[2])
        # Swap in a cheaper twin that is almost as good, if it clashes with nothing else.
        if (len(twins) == 1 and k not in protected and sc.at[a, "ls_t"] >= ALMOST_AS_GOOD * sc.at[k, "ls_t"] > 0
                and sc.at[a, "turnover"] <= MUCH_CHEAPER * sc.at[k, "turnover"]):
            kept[kept.index(k)] = a
            twin_of[k] = (a, how, sim)
            for x, (t, h, s) in list(twin_of.items()):
                if t == k:
                    twin_of[x] = (a, h, s)
            continue
        twin_of[a] = (k, how, sim)
    return kept, twin_of


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sc = pd.read_csv(EVAL / "scorecard.csv").set_index("id")
    sc = sc[sc["library"] != "check"]
    ranks = pd.read_parquet(EVAL / "alpha_ranks.parquet")[sc.index]
    net = pd.read_parquet(EVAL / "daily_ls_net.parquet")[sc.index]
    S = signal_similarity(ranks)
    M = money_similarity(net)
    S.astype("float32").to_parquet(OUT / "signal_similarity.parquet")
    M.astype("float32").to_parquet(OUT / "money_similarity.parquet")

    # stable sorts: exact ties (e.g. A101_055 and its sign-flipped copy GTJA_176) keep catalog order
    good = sc[sc["verdict"].isin(["strong", "maybe"])].sort_values("ls_t", ascending=False, kind="stable", key=lambda t: t.round(6)).index
    core, core_twins = pick(good, S, M, sc)
    some = sc[(sc["ic_t"].abs() >= 2) | (sc["ls_t"] >= 2)]
    rest = some.drop(index=[a for a in some.index if a in good]).assign(abs_t=lambda d: d["ic_t"].abs())
    rest_order = rest.sort_values("abs_t", ascending=False, kind="stable", key=lambda t: t.round(6)).index
    broad, broad_twins = pick(list(core) + list(rest_order), S, M, sc, kept=core)

    # Every alpha: which list it is in, or which kept alpha it duplicates.
    reps = pd.Index(broad)
    rows = []
    for a in sc.index:
        if a in core:
            status, rep, how, sim = "core", a, "", 1.0
        elif a in broad:
            status, rep, how, sim = "broad", a, "", 1.0
        elif a in core_twins or a in broad_twins:
            rep, how, sim = core_twins.get(a) or broad_twins[a]
            status = "twin"
        else:
            # weak alphas outside both lists: still show their nearest kept alpha
            s_best = S.loc[a, reps].idxmax()
            rep, how, sim = s_best, "signal", S.at[a, s_best]
            status = "weak"
        rows.append({"id": a, "kept": status, "twin_of": rep if status in ("twin", "weak") else "",
                     "check": how, "similarity": round(float(sim), 2)})
    membership = pd.DataFrame(rows)
    membership.to_csv(OUT / "membership.csv", index=False)

    tw = membership[membership["kept"] == "twin"].groupby("twin_of")["id"].apply(list)
    def lst(ids, extra=()):
        out = sc.loc[list(ids), ["verdict", "category", "ic_t", "ls_t", "net_bp", "turnover"]].copy()
        out["twins"] = [", ".join(tw.get(a, [])) for a in out.index]
        out["n_twins"] = [len(tw.get(a, [])) for a in out.index]
        return out
    core_df = lst(core)
    core_df["family"] = [FAMILIES.get(a, ("Unnamed", ""))[0] for a in core_df.index]
    core_df["family_note"] = [FAMILIES.get(a, ("", ""))[1] for a in core_df.index]
    missing = [a for a in core if a not in FAMILIES]
    if missing:
        print("WARNING: core alphas without a family name (update FAMILIES):", missing)
    core_df.to_csv(OUT / "core_list.csv")
    lst(broad).to_csv(OUT / "broad_list.csv")

    print(f"core:  {len(good)} strong+maybe -> {len(core)} distinct")
    print(lst(core)[["verdict", "category", "ls_t", "net_bp", "turnover", "n_twins", "twins"]].to_string())
    print(f"\nbroad: {len(some)} with some signal -> {len(broad)} distinct")
    print(membership["kept"].value_counts().to_dict())


if __name__ == "__main__":
    main()
