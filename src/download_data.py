"""Download Binance USDT perpetual futures data (daily bars, funding, premium).

    .venv/bin/python src/download_data.py

Steps:
  1. Symbol list: every USDT perpetual in the archive (including delisted ones),
     minus stablecoins, gold, indices and the stock/commodity "TradFi" perps.
  2. Daily klines for all of them (needed to rank coins by trading value).
  3. Funding rate + premium index for every coin that was ever in the top KEEP_TOP_K.
  4. The current month via REST (the monthly archive only has finished months).

Re-running is cheap: files already on disk are skipped.
"""
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

import binance_files as bf
from config import DOWNLOAD_START, EXCLUDED_BASES, KEEP_TOP_K, META, MIN_HISTORY, RANK_WINDOW, RAW

WORKERS = 32


def pick_symbols():
    """USDT perpetuals on crypto underlyings, with Binance's own sector tags."""
    info = {s["symbol"]: s for s in bf.exchange_info()["symbols"]}
    rows = []
    for sym in bf.archive_symbols():
        if not sym.endswith("USDT") or "_" in sym:
            continue
        s = info.get(sym)
        base = s["baseAsset"] if s else sym[: -len("USDT")]
        if s and (s["contractType"] != "PERPETUAL" or s["underlyingType"] != "COIN"):
            continue  # stock / commodity / index / pre-market contracts
        if base in EXCLUDED_BASES:
            continue
        rows.append({
            "symbol": sym,
            "base": base,
            "status": s["status"] if s else "DELISTED (not in exchangeInfo)",
            "onboard_date": pd.to_datetime(s["onboardDate"], unit="ms").date() if s else None,
            "sectors": "|".join(t for t in s.get("underlyingSubType", []) if t != "Crypto") if s else "",
        })
    meta = pd.DataFrame(rows)
    META.mkdir(parents=True, exist_ok=True)
    meta.to_csv(META / "symbols.csv", index=False)
    return meta


def fetch_archive(kind, symbols):
    """List and download monthly archive files for many symbols in parallel."""
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as pool:
        key_lists = list(pool.map(lambda s: bf.monthly_files(kind, s, DOWNLOAD_START), symbols))
        keys = [k for ks in key_lists for k in ks]
        print(f"  {kind}: {len(keys):,} monthly files for {len(symbols)} symbols", flush=True)
        status = list(pool.map(bf.download_key, keys))
    counts = pd.Series(status).value_counts().to_dict() if status else {}
    print(f"  {kind}: {counts} in {time.time() - t0:.0f}s", flush=True)
    return keys


def ever_top_k(symbols, k):
    """Coins that were ever among the top-k by RANK_WINDOW-day average quote volume."""
    qv = {}
    for s in symbols:
        df = bf.read_klines(s)
        if not df.empty:
            qv[s] = df["quote_volume"]
    qv = pd.DataFrame(qv).sort_index()
    age = qv.notna().cumsum()
    adv = qv.rolling(RANK_WINDOW, min_periods=RANK_WINDOW // 2).mean().where(age >= MIN_HISTORY)
    rank = adv.rank(axis=1, ascending=False)
    return sorted(rank.columns[(rank <= k).any()])


def fetch_rest_tail(kind, symbols, start_ms, now_ms):
    with ThreadPoolExecutor(4) as pool:  # REST has rate limits: stay gentle
        n = list(pool.map(lambda s: bf.rest_tail(kind, s, start_ms, now_ms), symbols))
    print(f"  REST {kind}: {sum(n):,} rows for {sum(1 for x in n if x)} symbols", flush=True)


def main():
    print("1/4 symbols")
    meta = pick_symbols()
    symbols = meta["symbol"].tolist()
    print(f"  {len(symbols)} crypto USDT perpetuals ({(meta['status'] == 'TRADING').sum()} trading now)")

    print("2/4 daily klines (all symbols)")
    keys = fetch_archive("klines", symbols)

    # The monthly archive ends with the last finished month; REST fills the rest.
    last_month = max(re.search(r"(\d{4}-\d{2})\.zip$", k).group(1) for k in keys)
    tail_start = (pd.Period(last_month, "M") + 1).start_time
    start_ms = int(tail_start.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    trading = meta.loc[meta["status"] == "TRADING", "symbol"].tolist()
    print(f"  archive ends {last_month}; REST tail from {tail_start.date()}")
    fetch_rest_tail("klines", trading, start_ms, now_ms)

    print(f"3/4 picking coins that were ever top {KEEP_TOP_K}")
    keep = ever_top_k(symbols, KEEP_TOP_K)
    pd.Series(keep, name="symbol").to_csv(META / "kept_symbols.csv", index=False)
    print(f"  {len(keep)} coins kept")

    print("4/4 funding rate + premium index (kept coins)")
    fetch_archive("fundingRate", keep)
    fetch_archive("premiumIndexKlines", keep)
    keep_trading = [s for s in keep if s in set(trading)]
    fetch_rest_tail("fundingRate", keep_trading, start_ms, now_ms)
    fetch_rest_tail("premiumIndexKlines", keep_trading, start_ms, now_ms)

    patch_gaps(keep)
    size_mb = sum(p.stat().st_size for p in RAW.rglob("*") if p.is_file()) / 1e6
    n_files = sum(1 for p in RAW.rglob("*") if p.is_file())
    print(f"done: {n_files:,} raw files, {size_mb:.1f} MB in {RAW}")


def patch_gaps(symbols):
    """Fill holes left by the monthly archive.

    - Some monthly kline/premium files skip days (e.g. 2022-02-26..28, 2022-04-01..02);
      the daily archive has them.
    - A few relisted coins (ICP, TLM, BNX) miss old funding history in the archive; REST has it.
    Days where a coin really did not trade (delist/relist gaps) stay missing.
    """
    print("patch: filling missing days")
    keys = []
    for s in symbols:
        k = bf.read_klines(s)
        if k.empty:
            continue
        life = pd.date_range(k.index.min(), k.index.max(), freq="D")
        keys += [bf.daily_key("klines", s, d) for d in life.difference(k.index)]
        p = bf.read_klines(s, kind="premiumIndexKlines")
        if not p.empty:
            keys += [bf.daily_key("premiumIndexKlines", s, d) for d in k.index.difference(p.index)
                     if d >= p.index.min()]
    with ThreadPoolExecutor(WORKERS) as pool:
        status = list(pool.map(bf.download_key, keys))
    print(f"  daily files: {pd.Series(status).value_counts().to_dict() if status else {}}")

    # REST only answers for coins that still trade, so only those can be backfilled.
    meta = pd.read_csv(META / "symbols.csv")
    trading = set(meta.loc[meta["status"] == "TRADING", "symbol"])
    backfilled = 0
    for s in [s for s in symbols if s in trading]:
        if (RAW / "rest" / "fundingRate_backfill" / f"{s}.csv").exists():
            continue
        k, f = bf.read_klines(s), bf.read_funding(s)
        if k.empty:
            continue
        f_days = pd.to_datetime(f["calc_time"] - 1000, unit="ms").dt.normalize() if not f.empty else pd.Series([], dtype="datetime64[ns]")
        missing = k.index.difference(pd.DatetimeIndex(f_days.unique()))
        if len(missing) > 5:
            start = int(missing.min().timestamp() * 1000)
            end = int((missing.max() + pd.Timedelta(days=1)).timestamp() * 1000)
            if bf.funding_backfill(s, start, end):
                backfilled += 1
    print(f"  funding backfilled for {backfilled} coins")


if __name__ == "__main__":
    import sys
    if "--patch-only" in sys.argv:  # just fill holes in what is already downloaded
        patch_gaps(pd.read_csv(META / "kept_symbols.csv")["symbol"].tolist())
    else:
        main()
