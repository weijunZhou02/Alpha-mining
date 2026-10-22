"""Compute any alpha in the catalog on the crypto panels.

    eng = AlphaEngine.load()
    values = eng.compute("A101_012")      # wide DataFrame, NaN outside the daily universe

Crypto mapping used for published stock formulas:
    volume / VOLUME   -> USDT quote volume (comparable across coins; set volume_field="volume" for coin units)
    adv{d}, AMOUNT    -> average / daily USDT quote volume (dollar volume)
    benchmark index   -> BTCUSDT
    IndClass.*        -> Binance sector tag (first tag in data/meta/symbols.csv)
Published formulas are otherwise computed as written (price-unit outputs are NOT rescaled here).
"""
import csv
import re

import numpy as np
import pandas as pd

import formula as fx
import operators as op
from config import META, PANEL, ROOT

CATALOG = ROOT / "alpha_library" / "alpha_catalog.csv"
BENCH = "BTCUSDT"

# Corrected versions of report formulas with typos or garbled text (see catalog column formula_issues).
GTJA_FIXES = {
    10: "RANK(MAX(((RET<0)?STD(RET,20):CLOSE)^2,5))",
    22: "SMA(((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6)-DELAY((CLOSE-MEAN(CLOSE,6))/MEAN(CLOSE,6),3)),12,1)",
    23: "SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)/(SMA((CLOSE>DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1)"
        "+SMA((CLOSE<=DELAY(CLOSE,1)?STD(CLOSE,20):0),20,1))*100",
    28: "3*SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1)"
        "-2*SMA(SMA((CLOSE-TSMIN(LOW,9))/(TSMAX(HIGH,9)-TSMIN(LOW,9))*100,3,1),3,1)",
    30: "WMA((REGRESI(CLOSE/DELAY(CLOSE,1)-1,MKT,60))^2,20)",  # crypto: market = BTC only (no SMB/HML)
    36: "RANK(SUM(CORR(RANK(VOLUME),RANK(VWAP),6),2))",
    52: "SUM(MAX(0,HIGH-DELAY((HIGH+LOW+CLOSE)/3,1)),26)/SUM(MAX(0,DELAY((HIGH+LOW+CLOSE)/3,1)-LOW),26)*100",
    54: "(-1*RANK((STD(ABS(CLOSE-OPEN),10)+(CLOSE-OPEN))+CORR(CLOSE,OPEN,10)))",
    116: "REGBETA(CLOSE,SEQUENCE(20))",
    127: "(MEAN((100*(CLOSE-TSMAX(CLOSE,12))/(TSMAX(CLOSE,12)))^2,12))^(1/2)",
    146: "MEAN({R}-SMA({R},61,2),20)*({R}-SMA({R},61,2))/SMA(({R}-({R}-SMA({R},61,2)))^2,60,2)".replace(
        "{R}", "((CLOSE-DELAY(CLOSE,1))/DELAY(CLOSE,1))"),
    149: "REGBETAIF(CLOSE/DELAY(CLOSE,1)-1,BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1,"
         "BANCHMARKINDEXCLOSE<DELAY(BANCHMARKINDEXCLOSE,1),252)",
    165: "SUMACRANGE(CLOSE-MEAN(CLOSE,48),48)/STD(CLOSE,48)",
    166: "-1*TSSKEW(CLOSE/DELAY(CLOSE,1)-1,20)",
    180: "((MEAN(VOLUME,20)<VOLUME)?((-1*TSRANK(ABS(DELTA(CLOSE,7)),60))*SIGN(DELTA(CLOSE,7))):(-1*VOLUME))",
    # 181: the report uses the benchmark's price level; a co-skewness needs its returns ({B}).
    181: "SUM(((CLOSE/DELAY(CLOSE,1)-1)-MEAN((CLOSE/DELAY(CLOSE,1)-1),20))-({B}-MEAN({B},20))^2,20)"
         "/SUM(({B}-MEAN({B},20))^3,20)".replace("{B}", "(BANCHMARKINDEXCLOSE/DELAY(BANCHMARKINDEXCLOSE,1)-1)"),
    183: "SUMACRANGE(CLOSE-MEAN(CLOSE,24),24)/STD(CLOSE,24)",
    # 190: the first SUMIF misses the brackets around the growth rate {G}; the second SUMIF shows the intent.
    190: "LOG((COUNT({R}>{G},20)-1)*(SUMIF(({R}-{G})^2,20,{R}<{G}))/((COUNT({R}<{G},20))*(SUMIF(({R}-{G})^2,20,{R}>{G}))))"
         .replace("{R}", "(CLOSE/DELAY(CLOSE,1)-1)").replace("{G}", "((CLOSE/DELAY(CLOSE,19))^(1/20)-1)"),
}

# GTJA helper variables defined in the report appendix.
GTJA_DERIVED = {
    "dtm": "(OPEN<=DELAY(OPEN,1)?0:MAX((HIGH-OPEN),(OPEN-DELAY(OPEN,1))))",
    "dbm": "(OPEN>=DELAY(OPEN,1)?0:MAX((OPEN-LOW),(OPEN-DELAY(OPEN,1))))",
    "tr": "MAX(MAX(HIGH-LOW,ABS(HIGH-DELAY(CLOSE,1))),ABS(LOW-DELAY(CLOSE,1)))",
    "hd": "HIGH-DELAY(HIGH,1)",
    "ld": "DELAY(LOW,1)-LOW",
}

NOT_COMPUTABLE = {
    "A101_056": "needs market cap",
    "CRYPTO_LTW_MCAP": "needs market cap",
    "CRYPTO_LTW_VOLSCALED": "needs market cap",
    "CRYPTO_CKTZ_VALUE": "needs on-chain data",
    "CRYPTO_LT_ATTENTION": "needs Google Trends",
    "CRYPTO_LT_TSMOM": "market-timing signal (same value for every coin); tested separately later",
    "CRYPTO_FLPWZ_CTREND": "a trained model, not a formula; built in the combination step",
    "CRYPTO_PR_OI_CHANGE": "needs open-interest history",
    "CRYPTO_PR_OI_PRICE_DIV": "needs open-interest history",
    "CRYPTO_PR_LS_RATIO": "needs long/short ratio history",
    "CLASSIC_SMB": "needs market cap",
    "CLASSIC_HML": "needs accounting data",
    "CLASSIC_RMW": "needs accounting data",
    "CLASSIC_CMA": "needs accounting data",
    "CLASSIC_TURNOVER": "needs shares outstanding",
}


def load_catalog():
    with open(CATALOG, encoding="utf-8-sig") as fh:
        return {r["id"]: r for r in csv.DictReader(fh)}


def _broadcast(series, ref):
    """Copy one series (e.g. BTC's close) into every column of a table shaped like ref."""
    return pd.DataFrame(np.repeat(series.to_numpy()[:, None], ref.shape[1], axis=1), index=ref.index, columns=ref.columns)


class AlphaEngine:
    def __init__(self, panels, universe, sectors, onboard, volume_field="quote_volume", adv_field="quote_volume"):
        self.p = panels
        self.universe = universe
        self.sectors = sectors
        self.onboard = onboard
        self.volume_field = volume_field
        self.adv_field = adv_field
        self.catalog = load_catalog()
        self._cache = {}

    @classmethod
    def load(cls, **kw):
        names = ["open", "high", "low", "close", "volume", "quote_volume", "vwap", "returns", "trades",
                 "taker_buy_quote_volume", "funding", "premium"]
        panels = {n: pd.read_parquet(PANEL / f"{n}.parquet") for n in names}
        universe = pd.read_parquet(PANEL / "universe.parquet")
        meta = pd.read_csv(META / "symbols.csv").set_index("symbol")
        cols = panels["close"].columns
        sectors = meta.reindex(cols)["sectors"].fillna("").str.split("|").str[0].replace("", "unknown")
        onboard = pd.to_datetime(meta.reindex(cols)["onboard_date"])
        return cls(panels, universe, sectors, onboard, **kw)

    # ------------------------------------------------------------ shared inputs

    def _get(self, key, make):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    def volume(self):
        return self.p[self.volume_field]

    def adv(self, d):
        return self._get(f"adv{d}", lambda: op.ts_mean(self.p[self.adv_field], d))

    def bench(self, field):
        return self._get(f"bench_{field}", lambda: _broadcast(self.p[field][BENCH], self.p["close"]))

    def market_return(self):
        """Volume-weighted return of yesterday's universe (a stand-in for a cap-weighted crypto index)."""
        def make():
            w = op.ts_mean(self.p["quote_volume"], 30).shift(1).where(self.universe.shift(1, fill_value=False))
            r = self.p["returns"]
            ok = w.notna() & r.notna()
            return (r.where(ok) * w.where(ok)).sum(axis=1) / w.where(ok).sum(axis=1)
        return self._get("mkt_series", make)

    # ------------------------------------------------------------ formula contexts

    def context(self, dialect):
        p = self.p
        base = {"open": p["open"], "high": p["high"], "low": p["low"], "close": p["close"], "vwap": p["vwap"]}
        if dialect == "wq101":
            v = {**base, "volume": self.volume(), "returns": p["returns"]}

            def pattern(name):
                m = re.fullmatch(r"adv(\d+)", name)
                if m:
                    return self.adv(int(m.group(1)))
                if name.startswith("indclass"):
                    return 0.0  # placeholder: indneutralize always uses the sector map
                return None
            v["*"] = pattern
        elif dialect == "gtja191":
            v = {**base, "volume": self.volume(), "vol": self.volume(), "amount": p["quote_volume"],
                 "ret": p["returns"], "hgih": p["high"],
                 "banchmarkindexclose": lambda: self.bench("close"), "benchmarkindexclose": lambda: self.bench("close"),
                 "banchmarkindexopen": lambda: self.bench("open"), "benchmarkindexopen": lambda: self.bench("open"),
                 "mkt": lambda: self.bench("returns")}
            for name, text in GTJA_DERIVED.items():
                v[name] = (lambda t=text, n=name: self._get(f"gtja_{n}", lambda: self.run(t, "gtja191")))
        elif dialect == "qlib":
            v = {"$" + k: x for k, x in base.items()}
            v["$volume"] = self.volume()
        else:
            raise ValueError(dialect)
        return fx.Context(v, self.universe, self.sectors)

    def run(self, text, dialect, mask_output=False):
        out = fx.evaluate(fx.parse(text), self.context(dialect), dialect)
        if not isinstance(out, pd.DataFrame):
            raise ValueError("formula produced a single number, not a table")
        return out.where(self.universe) if mask_output else out

    # ------------------------------------------------------------ public

    def formula_for(self, alpha_id):
        row = self.catalog[alpha_id]
        if row["library"] == "GTJA191":
            n = int(alpha_id.split("_")[1])
            return GTJA_FIXES.get(n, row["formula"]), "gtja191"
        return row["formula"], row["notation"]

    def compute(self, alpha_id):
        if alpha_id in NOT_COMPUTABLE:
            raise NotImplementedError(NOT_COMPUTABLE[alpha_id])
        if alpha_id in CUSTOM:
            out = CUSTOM[alpha_id](self)
        else:
            text, dialect = self.formula_for(alpha_id)
            out = self.run(text, dialect)
        return out.astype(float).where(self.universe)


# ---------------------------------------------------------------- custom (math-notation) alphas

def _beta_vs(e, x, d):
    return op.ts_regbeta(e.p["returns"], x, d)


def _resid_vol(e, x, d):
    r = e.p["returns"]
    return op.ts_std(r, d) * np.sqrt((1 - op.ts_corr(r, x, d) ** 2).clip(lower=0))


def _delay_r2(e):
    """LTW DELAY: extra R^2 from adding 1- and 2-day lagged market returns (365-day window)."""
    r = e.p["returns"]
    m = e.market_return()
    lags = [m, m.shift(1), m.shift(2)]
    X = [_broadcast(s, r) for s in lags]
    ry = np.stack([op.ts_corr(r, x, 365).to_numpy() for x in X], axis=-1)              # (T, N, 3)
    Rxx = np.empty((len(m), 3, 3))
    for i in range(3):
        for j in range(3):
            Rxx[:, i, j] = 1.0 if i == j else lags[i].rolling(365, min_periods=244).corr(lags[j]).to_numpy()
    ok = np.isfinite(Rxx).all(axis=(1, 2))
    inv = np.full_like(Rxx, np.nan)
    inv[ok] = np.linalg.inv(Rxx[ok])
    r2_full = np.einsum("tni,tij,tnj->tn", ry, inv, ry)
    r2_one = ry[:, :, 0] ** 2
    return pd.DataFrame(r2_full - r2_one, index=r.index, columns=r.columns)


def _age_days(e):
    c = e.p["close"]
    first = c.apply(lambda s: s.first_valid_index())
    start = e.onboard.fillna(first)
    days = (c.index.to_numpy()[:, None] - pd.to_datetime(start).to_numpy()[None, :]) / np.timedelta64(1, "D")
    return pd.DataFrame(days, index=c.index, columns=c.columns).where(c.notna())


def _ltw_mom(k):
    return lambda e: e.p["close"] / op.delay(e.p["close"], 7 * k) - 1


def _cum_up(e):
    """GTJA 143 (recursive SELF): running product of (1 + return) over up days only."""
    r = e.p["returns"]
    return (1 + r.where(r > 0, 0.0)).where(r.notna()).cumprod()


CUSTOM = {
    "GTJA_143": _cum_up,
    # Liu, Tsyvinski & Wu (weekly characteristics, computed daily on 7-day windows)
    "CRYPTO_LTW_PRC": lambda e: op.log(e.p["close"]),
    "CRYPTO_LTW_MAXDPRC": lambda e: op.ts_max(e.p["close"], 7),
    "CRYPTO_LTW_AGE": _age_days,
    **{f"CRYPTO_LTW_R{k}": _ltw_mom(k) for k in (1, 2, 3, 4, 8, 16, 50, 100)},
    "CRYPTO_LTW_VOL": lambda e: op.log(op.ts_mean(e.p["volume"], 7)),
    "CRYPTO_LTW_PRCVOL": lambda e: op.log(op.ts_mean(e.p["quote_volume"], 7)),
    "CRYPTO_LTW_BETA": lambda e: _beta_vs(e, _broadcast(e.market_return(), e.p["close"]), 365),
    "CRYPTO_LTW_BETA2": lambda e: _beta_vs(e, _broadcast(e.market_return(), e.p["close"]), 365) ** 2,
    "CRYPTO_LTW_IDIOVOL": lambda e: _resid_vol(e, _broadcast(e.market_return(), e.p["close"]), 365),
    "CRYPTO_LTW_RETVOL": lambda e: op.ts_std(e.p["returns"], 7),
    "CRYPTO_LTW_RETSKEW": lambda e: op.ts_skew(e.p["returns"], 7),
    "CRYPTO_LTW_RETKURT": lambda e: op.ts_kurt(e.p["returns"], 7),
    "CRYPTO_LTW_MAXRET": lambda e: op.ts_max(e.p["returns"], 7),
    "CRYPTO_LTW_DELAY": _delay_r2,
    "CRYPTO_LTW_STDPRCVOL": lambda e: op.log(op.ts_std(e.p["quote_volume"], 7)),
    "CRYPTO_LTW_DAMIHUD": lambda e: op.ts_mean(np.abs(e.p["returns"]) / e.p["quote_volume"].where(e.p["quote_volume"] > 0), 7),
    # other crypto papers
    "CRYPTO_DOB_MOM": lambda e: e.p["close"] / op.delay(e.p["close"], 14) - 1,
    "CRYPTO_DOB_REV": lambda e: -(e.p["close"] / op.delay(e.p["close"], 90) - 1),
    "CRYPTO_BBD_VOLREV": lambda e: -op.cs_rank(e.p["returns"], e.universe)
        * (1 - op.cs_rank(e.p["quote_volume"] / op.ts_mean(e.p["quote_volume"], 30), e.universe)),
    "CRYPTO_LUWZ_MAX": lambda e: op.ts_max(e.p["returns"], 7),
    "CRYPTO_SST_CARRY": lambda e: op.ts_mean(e.p["funding"], 3) * 365,
    "CRYPTO_HMRW_PREMIUM": lambda e: e.p["premium"],
    # practitioner ideas
    "CRYPTO_PR_TAKER_BUY": lambda e: op.ts_sum(e.p["taker_buy_quote_volume"], 7) / op.ts_sum(e.p["quote_volume"], 7),
    "CRYPTO_PR_TRADE_SIZE": lambda e: e.p["quote_volume"] / e.p["trades"].where(e.p["trades"] > 0),
    "CRYPTO_PR_FUNDING_LEVEL": lambda e: -op.ts_mean(e.p["funding"], 3),
    "CRYPTO_PR_FUNDING_CHANGE": lambda e: op.delta(op.ts_mean(e.p["funding"], 3), 3),
    "CRYPTO_PR_BTC_RESID_REV": lambda e: -(e.p["returns"] - _beta_vs(e, e.bench("returns"), 30) * e.bench("returns")),
    "CRYPTO_PR_BTC_BETA": lambda e: -_beta_vs(e, e.bench("returns"), 60),
    # classic equity factors, crypto versions
    "CLASSIC_MKT_BETA": lambda e: _beta_vs(e, e.bench("returns"), 252),
    "CLASSIC_MOM": lambda e: op.delay(e.p["close"], 21) / op.delay(e.p["close"], 252) - 1,
    "CLASSIC_STREV": lambda e: -(e.p["close"] / op.delay(e.p["close"], 21) - 1),
    "CLASSIC_BAB": lambda e: -_beta_vs(e, e.bench("returns"), 252),
    "CLASSIC_ILLIQ": lambda e: op.ts_mean(np.abs(e.p["returns"]) / e.p["quote_volume"].where(e.p["quote_volume"] > 0), 21),
    "CLASSIC_IVOL": lambda e: _resid_vol(e, e.bench("returns"), 21),
    "CLASSIC_MAX": lambda e: -op.ts_max(e.p["returns"], 21),
    "CLASSIC_HIGH52W": lambda e: e.p["close"] / op.ts_max(e.p["close"], 365),
}
