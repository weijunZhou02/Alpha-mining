"""Download and read Binance USD-M futures files (bulk archive + REST for the newest days)."""
import io
import json
import re
import threading
import time
import urllib.parse
import zipfile

import pandas as pd
import requests

from config import KLINE_COLUMNS, RAW

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
FAPI = "https://fapi.binance.com"
DAY_MS = 86_400_000

# kind -> archive path pattern (monthly files)
ARCHIVE = {
    "klines": "data/futures/um/monthly/klines/{s}/1d/",
    "fundingRate": "data/futures/um/monthly/fundingRate/{s}/",
    "premiumIndexKlines": "data/futures/um/monthly/premiumIndexKlines/{s}/1d/",
}


_local = threading.local()


def _session():
    # One keep-alive session per thread: reusing connections is ~10x faster than a new TLS handshake per file.
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers["User-Agent"] = "crypto-alpha-research/0.1"
    return _local.s


def http_get(url, retries=5, timeout=30):
    """GET with retries. Returns bytes, or None on 400/404."""
    for attempt in range(retries):
        try:
            resp = _session().get(url, timeout=timeout)
            if resp.status_code in (400, 404):
                return None
            if resp.status_code in (418, 429):  # rate limited: back off hard
                time.sleep(30 * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.content
        except requests.RequestException:
            if attempt == retries - 1:
                raise
        time.sleep(2 ** attempt)
    raise RuntimeError(f"gave up on {url}")


def s3_list(prefix, delimiter="/"):
    """List sub-folders (when delimiter is set) and file keys under an archive prefix."""
    folders, keys, marker = [], [], ""
    while True:
        query = urllib.parse.urlencode({"delimiter": delimiter, "prefix": prefix, "marker": marker})
        xml = http_get(f"{S3}?{query}").decode()
        folders += [p for p in re.findall(r"<Prefix>(.*?)</Prefix>", xml) if p != prefix]
        keys += re.findall(r"<Key>(.*?)</Key>", xml)
        if "<IsTruncated>true</IsTruncated>" not in xml:
            return folders, keys
        marker = re.search(r"<NextMarker>(.*?)</NextMarker>", xml).group(1) if "<NextMarker>" in xml else keys[-1]


def archive_symbols():
    folders, _ = s3_list("data/futures/um/monthly/klines/")
    return sorted(f.rstrip("/").split("/")[-1] for f in folders)


def exchange_info():
    return json.loads(http_get(f"{FAPI}/fapi/v1/exchangeInfo"))


def monthly_files(kind, symbol, start_month):
    """Archive zip keys for one symbol, from start_month (YYYY-MM) on."""
    _, keys = s3_list(ARCHIVE[kind].format(s=symbol))
    out = []
    for k in keys:
        m = re.search(r"-(\d{4}-\d{2})\.zip$", k)
        if m and m.group(1) >= start_month:
            out.append(k)
    return out


def daily_key(kind, symbol, day):
    """Archive key of one daily file (used to fill days that are missing from the monthly files)."""
    d = pd.Timestamp(day).strftime("%Y-%m-%d")
    return f"data/futures/um/daily/{kind}/{symbol}/1d/{symbol}-1d-{d}.zip"


def funding_backfill(symbol, start_ms, end_ms):
    """Funding history from REST between two times (the archive misses some relisted coins)."""
    rows, t = [], start_ms
    while t < end_ms:
        data = http_get(f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&startTime={t}&endTime={end_ms}&limit=1000")
        page = json.loads(data) if data else []
        if not page:
            break
        rows += page
        t = int(page[-1]["fundingTime"]) + 1
        time.sleep(0.3)  # this endpoint allows ~500 requests per 5 minutes
    if not rows:
        return 0
    df = pd.DataFrame({"calc_time": [int(r["fundingTime"]) for r in rows],
                       "last_funding_rate": [float(r["fundingRate"]) for r in rows]})
    path = RAW / "rest" / "fundingRate_backfill" / f"{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


def local_path(key):
    # data/futures/um/{monthly,daily}/<kind>/<SYMBOL>/[1d/]<file>.zip -> raw/<kind>/<SYMBOL>/<file>.zip
    parts = key.split("/")
    return RAW / parts[4] / parts[5] / parts[-1]


def download_key(key):
    """Download one archive file unless it is already cached. Returns 'cached', 'ok' or 'missing'."""
    path = local_path(key)
    if path.exists() and path.stat().st_size > 0:
        return "cached"
    data = http_get(f"{FILES}/{key}")
    if data is None:
        return "missing"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return "ok"


# ---------- REST: the current month is not in the monthly archive yet ----------

def rest_tail(kind, symbol, start_ms, now_ms):
    """Fetch rows from start_ms to now via REST and save them as raw/rest/<kind>/<symbol>.csv."""
    if kind == "klines":
        url = f"{FAPI}/fapi/v1/klines?symbol={symbol}&interval=1d&startTime={start_ms}&limit=100"
    elif kind == "premiumIndexKlines":
        url = f"{FAPI}/fapi/v1/premiumIndexKlines?symbol={symbol}&interval=1d&startTime={start_ms}&limit=100"
    elif kind == "fundingRate":
        url = f"{FAPI}/fapi/v1/fundingRate?symbol={symbol}&startTime={start_ms}&limit=1000"
    else:
        raise ValueError(kind)
    data = http_get(url)
    if data is None:
        return 0
    rows = json.loads(data)
    if kind == "fundingRate":
        df = pd.DataFrame(rows)
        if df.empty:
            return 0
        df = pd.DataFrame({"calc_time": df["fundingTime"].astype("int64"),
                           "last_funding_rate": df["fundingRate"].astype(float)})
    else:
        df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
        df = df[df["close_time"].astype("int64") < now_ms]  # drop today's unfinished bar
    path = RAW / "rest" / kind / f"{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return len(df)


# ---------- reading ----------

def _read_zip_csv(path, columns):
    with zipfile.ZipFile(path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if isinstance(df.iloc[0, 0], str) and not str(df.iloc[0, 0]).isdigit():  # file has a header row
        df = pd.read_csv(io.BytesIO(raw))
        if columns is not None:
            df.columns = columns[: df.shape[1]]
    elif columns is not None:
        df.columns = columns[: df.shape[1]]
    return df


def _to_ms(series):
    s = pd.to_numeric(series, errors="coerce").astype("int64")
    return s.where(s < 10**14, s // 1000)  # some newer files use microseconds


def read_klines(symbol, kind="klines"):
    """All daily bars for a symbol (archive + REST tail), one row per UTC date."""
    parts = [_read_zip_csv(p, KLINE_COLUMNS) for p in sorted((RAW / kind / symbol).glob("*.zip"))]
    rest = RAW / "rest" / kind / f"{symbol}.csv"
    if rest.exists():
        parts.append(pd.read_csv(rest))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["open_time"] = _to_ms(df["open_time"])
    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    num = ["open", "high", "low", "close", "volume", "quote_volume", "trades",
           "taker_buy_volume", "taker_buy_quote_volume"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df.drop_duplicates("date", keep="last").set_index("date").sort_index()[num]


def read_funding(symbol):
    """Funding settlements (time in ms, rate)."""
    cols = ["calc_time", "funding_interval_hours", "last_funding_rate"]
    parts = []
    for p in sorted((RAW / "fundingRate" / symbol).glob("*.zip")):
        df = _read_zip_csv(p, None)
        if not isinstance(df.columns[0], str) or df.columns[0] not in cols:
            df.columns = cols[: df.shape[1]] if df.shape[1] == 3 else ["calc_time", "last_funding_rate"]
        parts.append(df[["calc_time", "last_funding_rate"]])
    for rest in (RAW / "rest" / "fundingRate" / f"{symbol}.csv",
                 RAW / "rest" / "fundingRate_backfill" / f"{symbol}.csv"):
        if rest.exists():
            parts.append(pd.read_csv(rest))
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=["calc_time", "last_funding_rate"])
    df = pd.concat(parts, ignore_index=True)
    df["calc_time"] = _to_ms(df["calc_time"])
    df["last_funding_rate"] = pd.to_numeric(df["last_funding_rate"], errors="coerce")
    return df.drop_duplicates("calc_time").sort_values("calc_time")
