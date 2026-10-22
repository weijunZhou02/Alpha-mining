"""Turn the raw Binance files into daily panels: rows = UTC date, columns = coin.

    .venv/bin/python src/build_panels.py

Writes data/panel/<field>.parquet and data/meta/universe_members.csv.
"""
import json

import numpy as np
import pandas as pd

import binance_files as bf
from config import DOWNLOAD_START, META, MIN_HISTORY, PANEL, RANK_WINDOW, RAW, RESEARCH_START, TOP_N

FIELDS = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
          "taker_buy_volume", "taker_buy_quote_volume"]


def daily_funding(symbol):
    """Sum of funding rates settled during each UTC day (a 00:00 settlement belongs to the day before)."""
    f = bf.read_funding(symbol)
    if f.empty:
        return pd.Series(dtype=float)
    day = pd.to_datetime(f["calc_time"] - 1000, unit="ms").dt.normalize()
    return f.groupby(day)["last_funding_rate"].sum()


def main():
    meta = pd.read_csv(META / "symbols.csv").set_index("symbol")
    keep = pd.read_csv(META / "kept_symbols.csv")["symbol"].tolist()

    bars = {s: bf.read_klines(s) for s in keep}
    bars = {s: df for s, df in bars.items() if not df.empty}
    last_day = max(df.index.max() for df in bars.values())
    dates = pd.date_range(DOWNLOAD_START, last_day, freq="D", name="date")

    panels = {f: pd.DataFrame({s: df[f] for s, df in bars.items()}).reindex(dates) for f in FIELDS}
    close, volume, qv = panels["close"], panels["volume"], panels["quote_volume"]
    panels["vwap"] = (qv / volume).where(volume > 0)
    panels["returns"] = close / close.shift(1) - 1

    # A few exchange glitches (mostly 2023-09-19/20 and 2023-11-30): volume fields that don't add up.
    # Blank them instead of letting them leak into vwap or order-flow alphas.
    bad_vwap = (panels["vwap"] < panels["low"] * (1 - 1e-6)) | (panels["vwap"] > panels["high"] * (1 + 1e-6))
    panels["vwap"] = panels["vwap"].mask(bad_vwap)
    bad_taker = ((panels["taker_buy_volume"] > volume * (1 + 1e-9))
                 | (panels["taker_buy_quote_volume"] > qv * (1 + 1e-9)))
    for f in ("taker_buy_volume", "taker_buy_quote_volume"):
        panels[f] = panels[f].mask(bad_taker)
    n_glitches = int(bad_vwap.sum().sum() + bad_taker.sum().sum())

    # Universe: each day, the TOP_N coins by RANK_WINDOW-day average quote volume,
    # among coins that traded today and have at least MIN_HISTORY days of history.
    # Everything here uses data up to and including day t (no look-ahead).
    age = close.notna().cumsum()
    eligible = close.notna() & (volume > 0) & (age >= MIN_HISTORY)
    adv = qv.rolling(RANK_WINDOW, min_periods=RANK_WINDOW // 2).mean()
    rank = adv.where(eligible).rank(axis=1, ascending=False, method="first")
    panels["universe"] = (rank <= TOP_N)

    # Answer key for grading only: next day's close-to-close return. Never use it as an input.
    # Computed before removing zombie bars, so a coin's last trading day gets the return to its
    # delisting settlement price instead of a missing value.
    panels["fwd_ret_1d"] = close.shift(-1) / close - 1

    # After a delisting, Binance keeps printing flat zero-volume "zombie" bars for months. Remove them.
    last_trade = (volume > 0)[::-1].cummax()[::-1]
    zombie = close.notna() & ~last_trade
    n_zombie = int(zombie.sum().sum())
    for f in FIELDS + ["vwap", "returns", "fwd_ret_1d"]:
        panels[f] = panels[f].mask(zombie)
    close, volume = panels["close"], panels["volume"]

    panels["funding"] = pd.DataFrame({s: daily_funding(s) for s in bars}).reindex(dates)
    prem = {s: bf.read_klines(s, kind="premiumIndexKlines")["close"] for s in bars}
    panels["premium"] = pd.DataFrame({s: p for s, p in prem.items() if not p.empty}).reindex(dates)

    PANEL.mkdir(parents=True, exist_ok=True)
    for name, df in panels.items():
        df = df.reindex(columns=sorted(bars)).astype(float if name != "universe" else bool)
        df.index.name = "date"
        df.to_parquet(PANEL / f"{name}.parquet")

    # Per-coin summary
    uni = panels["universe"]
    members = pd.DataFrame({
        "first_date": close.apply(lambda c: c.first_valid_index()),
        "last_date": close.apply(lambda c: c.last_valid_index()),
        "days_in_universe": uni.sum(),
        "first_universe_date": uni.apply(lambda c: c.idxmax() if c.any() else None),
        "last_universe_date": uni.apply(lambda c: c[::-1].idxmax() if c.any() else None),
    })
    members = members.join(meta[["status", "onboard_date", "sectors"]])
    members.index.name = "symbol"
    members.sort_values("days_in_universe", ascending=False).to_csv(META / "universe_members.csv")

    # Health checks on the research period
    r = slice(RESEARCH_START, None)
    u = uni.loc[r]
    ret_u = panels["returns"].loc[r].where(u)
    fwd_missing = (u & panels["fwd_ret_1d"].loc[r].isna()).sum().sum()
    summary = {
        "dates": f"{dates[0].date()} to {dates[-1].date()} (research from {RESEARCH_START})",
        "coins_in_panels": len(bars),
        "coins_ever_in_universe": int(u.any().sum()),
        "universe_size_min_median_max": [int(u.sum(axis=1).min()), int(u.sum(axis=1).median()), int(u.sum(axis=1).max())],
        "universe_days": int(len(u)),
        "coin_days_in_universe": int(u.sum().sum()),
        "funding_coverage_in_universe": round(float(panels["funding"].loc[r].notna()[u].sum().sum() / u.sum().sum()), 3),
        "premium_coverage_in_universe": round(float(panels["premium"].loc[r].notna()[u].sum().sum() / u.sum().sum()), 3),
        "daily_returns_above_+100pct": int((ret_u > 1).sum().sum()),
        "daily_returns_below_-50pct": int((ret_u < -0.5).sum().sum()),
        "universe_coin_days_without_next_day_return": int(fwd_missing),
        "glitch_cells_blanked": n_glitches,
        "zombie_bars_removed": n_zombie,
        "missing_days_inside_coin_life": int(sum(close.loc[close[s].first_valid_index():close[s].last_valid_index(), s].isna().sum()
                                                 for s in close.columns)),
        "funding_missing_in_universe": int((u & panels["funding"].loc[r].isna()).sum().sum()),
        "panel_size_mb": round(sum(p.stat().st_size for p in PANEL.glob("*.parquet")) / 1e6, 1),
        "raw_size_mb": round(sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()) / 1e6, 1),
    }
    (PANEL / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
