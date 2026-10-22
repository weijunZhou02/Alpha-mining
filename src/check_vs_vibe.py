"""Cross-check our alpha numbers against Vibe-Trading's independent code (third_party/vibe_trading).

    .venv/bin/python src/check_vs_vibe.py

Both codes get the SAME input: 30 liquid coins with complete data, 2022-01 to 2024-12, all coins
treated as "in the universe", strict full-window mode (Vibe-Trading's convention), coin-unit volume.
Compared on 2023-01..2024-12 (2022 is warm-up). For each alpha:
    exact      share of cells where the two numbers agree (to 1e-6, relative)
    rank_corr  average daily Spearman correlation across coins (same signal even if the scale differs?)
Writes data/checks/vibe_comparison.csv.
"""
import importlib
import re
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import operators as op
from alphas import AlphaEngine, load_catalog
from config import DATA, ROOT

VT = ROOT / "third_party" / "vibe_trading"
START, EVAL_START, END = "2022-01-01", "2023-01-01", "2024-12-31"
N_COINS = 30


def import_vibe():
    sys.path.insert(0, str(VT))
    cfg = types.SimpleNamespace(agent_tuning=types.SimpleNamespace(vibe_trading_disable_bottleneck=True))
    for name in ("src.config", "src.config.accessor"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["src.config.accessor"].get_env_config = lambda: cfg


def vibe_compute(family, stem, panel):
    mod = importlib.import_module(f"src.factors.zoo.{family}.{stem}")
    return mod.compute(panel)


def pick_coins(eng):
    u = eng.universe.loc[START:END]
    c = eng.p["close"].loc[START:END]
    full = u.all() & c.notna().all()
    qv = eng.p["quote_volume"].loc[START:END].mean()
    return qv[full].sort_values(ascending=False).index[:N_COINS].tolist()


QLIB_SHORTCUTS = {"ROC": "uses close/old close - 1 instead of Qlib's old close / close",
                  "BETA": "uses (close change / close) / d instead of Qlib's regression slope",
                  "RESI": "uses (close - mean) / close instead of Qlib's regression residual",
                  "CORD": "uses log((v+1)/(v_prev+1)) instead of Qlib's Log(v/v_prev + 1)"}


# Found by reading their code side by side with the paper/report while debugging differences.
MANUAL = {
    "A101_067": "uses x instead of the paper's ^ (power)",
    "GTJA_017": "uses |exponent| with the sign kept; report: plain power",
    "GTJA_101": "their formula text differs from the report",
    "GTJA_146": "report formula is incomplete; they square (return - EMA), the literal text squares the EMA",
    "GTJA_149": "20-day window and zeros on up days; report: 252 days, down days only",
    "GTJA_154": "multiplies by -1; the report has no sign flip",
    "GTJA_165": "keeps the literal operator order; we use (max - min) / std (rescaled range)",
    "GTJA_183": "keeps the literal operator order; we use (max - min) / std (rescaled range)",
    "GTJA_166": "report formula is garbled; we use rolling skewness, they keep the garbled constants",
    "GTJA_140": "adds an extra TSRANK(.., 7) step that is not in the report",
    "A101_045": "2-day correlations are always about +/-1; tiny rounding changes how ties are ranked",
    "GTJA_113": "2-day correlations are always about +/-1; tiny rounding changes how ties are ranked",
}


def vibe_shortcut(alpha_id, src, formula, lib):
    """Known places where Vibe-Trading's code departs from the original formula (from its own notes)."""
    reasons = [MANUAL[alpha_id]] if alpha_id in MANUAL else []
    m = re.fullmatch(r"Q158_([A-Z]+)\d+", alpha_id)
    if m and m.group(1) in QLIB_SHORTCUTS:
        reasons.append(QLIB_SHORTCUTS[m.group(1)])
    # Their sector alphas mention a "degraded" fallback that is only used when no sector map is given.
    text = re.sub(r"[^.\n]*sector[^.\n]*", "", src.lower())
    for key, label in [("approximated with", "shortens long windows"), ("truncated", "shortens long windows"),
                       ("window 50", "shortens long windows"), ("cap (warmup", "shortens long windows"),
                       ("proxy", "uses a proxy"), ("degraded", "simplified / fallback"),
                       ("simplified", "simplified"), ("falls back", "simplified / fallback"),
                       ("fallback cross-sectional", "simplified / fallback"),
                       ("multiply by -1", "adds a sign flip"), ("wma proxied", "replaces WMA"),
                       ("sumac = expanding", "reads SUMAC differently")]:
        if key in text and label not in reasons:
            reasons.append(label)
    if lib == "Alpha101" and any(float(x) % 1 >= 0.5 for x in re.findall(r"\b\d+\.\d+\b", formula) if float(x) > 1):
        reasons.append("rounds decimal windows (paper: round down)")
    if "^" in formula and re.search(r"out = \(?lhs \* rhs\)?", src) and "uses x instead" not in " ".join(reasons):
        reasons.append("uses x instead of the paper's ^ (power)")
    if "np.fmin" in src:
        reasons.append("its min ignores missing values (paper: missing stays missing)")
    return "; ".join(reasons)


def compare(ours, theirs):
    a = ours.loc[EVAL_START:END].astype(float)
    b = theirs.reindex_like(ours).loc[EVAL_START:END].astype(float)
    both = np.isfinite(a.to_numpy()) & np.isfinite(b.to_numpy())
    if both.sum() == 0:
        return dict(cells=0, exact=np.nan, rank_corr=np.nan,
                    ours_cov=float(np.isfinite(a.to_numpy()).mean()), theirs_cov=float(np.isfinite(b.to_numpy()).mean()))
    diff = np.abs(a.to_numpy() - b.to_numpy())
    tol = 1e-6 * (1 + np.abs(b.to_numpy()))
    exact = float((diff[both] <= tol[both]).mean())
    ra, rb = a.where(both).rank(axis=1), b.where(both).rank(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = ra.T.corrwith(rb.T).mean() if both.sum() > 0 else np.nan
    return dict(cells=int(both.sum()), exact=round(exact, 4), rank_corr=round(float(rc), 4),
                ours_cov=round(float(np.isfinite(a.to_numpy()).mean()), 4),
                theirs_cov=round(float(np.isfinite(b.to_numpy()).mean()), 4))


def main():
    import_vibe()
    full = AlphaEngine.load(volume_field="volume", adv_field="volume")
    coins = pick_coins(full)
    if "BTCUSDT" not in coins:
        coins = ["BTCUSDT"] + coins[:-1]
    sl = slice(START, END)
    panels = {k: v.loc[sl, coins].copy() for k, v in full.p.items()}
    # Put every coin's price near 100 (volume scaled the other way, so USDT volume is unchanged).
    # Vibe-Trading divides with a +1e-12 safety term that breaks on tiny numbers (e.g. SHIB at $0.00001).
    k = 100.0 / panels["close"].median()
    for f in ("open", "high", "low", "close", "vwap"):
        panels[f] = panels[f] * k
    panels["volume"] = panels["volume"] / k
    panels["returns"] = panels["close"] / panels["close"].shift(1) - 1
    universe = pd.DataFrame(True, index=panels["close"].index, columns=coins)
    eng = AlphaEngine(panels, universe, full.sectors.reindex(coins), full.onboard.reindex(coins),
                      volume_field="volume", adv_field="volume")
    sector_df = pd.DataFrame(np.broadcast_to(eng.sectors.to_numpy()[None, :], panels["close"].shape),
                             index=panels["close"].index, columns=coins)
    vt_panel = {"open": panels["open"], "high": panels["high"], "low": panels["low"], "close": panels["close"],
                "volume": panels["volume"], "vwap": panels["vwap"], "amount": panels["quote_volume"],
                "sector": sector_df,
                "benchmark_close": pd.DataFrame(np.broadcast_to(panels["close"]["BTCUSDT"].to_numpy()[:, None],
                                                                panels["close"].shape),
                                                index=panels["close"].index, columns=coins)}
    # Their GTJA code rebuilds VWAP from China A-share units: amount / (volume * 100 + 1).
    # Give those alphas an 'amount' that makes that rebuild equal to the real VWAP.
    vt_panel_vwap_fix = dict(vt_panel, amount=panels["vwap"] * (panels["volume"] * 100.0 + 1.0))
    op.set_min_frac(1.0)          # Vibe-Trading requires full windows
    op.set_flat_corr_zero(False)  # and leaves correlation with a flat series missing

    rows = []
    cat = load_catalog()
    for alpha_id, row in cat.items():
        lib = row["library"]
        if lib == "Alpha101":
            family, stem = "alpha101", f"alpha_{int(alpha_id[5:]):03d}"
        elif lib == "GTJA191":
            family, stem = "gtja191", f"alpha_{int(alpha_id[5:]):03d}"
        elif lib == "Alpha158":
            family, stem = "qlib158", alpha_id[5:].lower()
        else:
            continue
        rec = {"id": alpha_id}
        if not (VT / "src" / "factors" / "zoo" / family / f"{stem}.py").exists():
            rec["result"] = "not in Vibe-Trading"
            rows.append(rec)
            continue
        try:
            ours = eng.compute(alpha_id)
        except Exception as e:  # noqa: BLE001
            rec.update(result="ours failed", note=f"{type(e).__name__}: {e}"[:150])
            rows.append(rec)
            continue
        src = (VT / "src" / "factors" / "zoo" / family / f"{stem}.py").read_text(encoding="utf-8")
        uses_hack = "v * 100.0 + 1.0" in src
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                theirs = vibe_compute(family, stem, vt_panel_vwap_fix if uses_hack else vt_panel)
        except Exception as e:  # noqa: BLE001
            rec.update(result="theirs failed", note=f"{type(e).__name__}: {e}"[:150])
            rows.append(rec)
            continue
        rec.update(compare(ours, theirs))
        if rec["cells"] == 0:
            rec["result"] = "no overlap"
        elif rec["exact"] >= 0.99:
            rec["result"] = "same numbers"
        elif rec["rank_corr"] >= 0.99:
            rec["result"] = "same signal (scale differs)"
        else:
            rec["result"] = "differs"
            rec["their_shortcut"] = vibe_shortcut(alpha_id, src, row["formula"], lib)
        rows.append(rec)
        print(f"{alpha_id:12s} {rec['result']:28s} exact={rec.get('exact')} rank_corr={rec.get('rank_corr')}", flush=True)

    df = pd.DataFrame(rows)
    out = DATA / "checks"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "vibe_comparison.csv", index=False)
    print("\ncoins:", ", ".join(coins))
    df["library"] = df["id"].str.split("_").str[0]
    print(df.groupby(["library", "result"]).size().to_string())
    d = df[df["result"] == "differs"]
    known = d["their_shortcut"].fillna("") != ""
    print(f"\ndiffers: {len(d)}  (Vibe-Trading notes a shortcut: {int(known.sum())}, unexplained: {int((~known).sum())})")
    print(d.loc[~known, ["id", "exact", "rank_corr", "ours_cov", "theirs_cov"]].to_string(index=False))


if __name__ == "__main__":
    main()
