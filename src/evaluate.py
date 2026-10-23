"""Score card for an alpha: does it predict returns? (step 3)

Settings agreed on (see README):
  horizon 1 day (plus 3, 7, 14 for decay) · trade at the same close (plus a 1-day-late check)
  Rank IC among the day's top-80 · 5 equal-weight groups, returns relative to the 80-coin average
  real returns kept (plus a 1%/99% capped check) · missing next-day returns dropped and counted
  cost 0.05% per trade + funding · strong = |t| >= 3 and same sign in >= 2 of 3 years
  negative alphas flipped (direction = -1) · Newey-West t-stats for overlapping horizons
Only the training period is used unless a later period is asked for explicitly.
"""
import math

import numpy as np
import pandas as pd

from config import PANEL

PERIODS = {
    "train": ("2021-01-01", "2023-12-31"),
    "valid": ("2024-01-01", "2024-12-31"),
    "train+valid": ("2021-01-01", "2024-12-31"),   # step 6: walk forward from train into valid
    # step 7 splits the training years: the LLM sees "search" scores and the "screen" gate's numbers,
    # never the "confirm" half-year's numbers (only pass / fail)
    "search": ("2021-01-01", "2022-12-31"),
    "screen": ("2023-01-01", "2023-06-30"),
    "confirm": ("2023-07-01", "2023-12-31"),
    "test": ("2025-01-01", "2026-10-21"),         # sealed with the step 8 plan; newer days are reported separately
}
HORIZONS = (1, 3, 7, 14)
N_GROUPS = 5
FEE = 0.0005           # 0.05% per unit of notional traded (Binance standard taker fee)
MIN_COINS = 20         # skip days with fewer coins than this
STRONG_T, MAYBE_T = 3.0, 2.0


# ------------------------------------------------------------------ shared data

class Market:
    """Everything an alpha is graded against, cut to one period."""

    def __init__(self, period="train", unlock_test=False):
        if period == "test" and not unlock_test:
            raise PermissionError("the test period is locked until the final exam (pass unlock_test=True)")
        start, end = PERIODS[period]
        self.period, self.start = period, start
        close = pd.read_parquet(PANEL / "close.parquet")
        self.end = end or str(close.index[-1].date())
        uni = pd.read_parquet(PANEL / "universe.parquet")
        fwd1 = pd.read_parquet(PANEL / "fwd_ret_1d.parquet")
        funding = pd.read_parquet(PANEL / "funding.parquet")
        sl = slice(self.start, self.end)
        self.universe = uni.loc[sl]
        self.dates = self.universe.index
        self.fwd = {1: fwd1.loc[sl]}
        for h in HORIZONS[1:]:
            self.fwd[h] = (close.shift(-h) / close - 1).loc[sl]
        self.fwd_late = fwd1.shift(-1).loc[sl]          # return from t+1 to t+2 (trade one day late)
        self.funding_next = funding.shift(-1).loc[sl]   # funding paid while holding from t to t+1
        # Purge: an answer key may not reach past the end of the period (no peeking into the next period).
        end_ts = pd.Timestamp(self.end)
        for h in HORIZONS:
            self.fwd[h] = self.fwd[h].copy()
            self.fwd[h].loc[self.dates > end_ts - pd.Timedelta(days=h)] = np.nan
        self.fwd_late = self.fwd_late.copy()
        self.fwd_late.loc[self.dates > end_ts - pd.Timedelta(days=2)] = np.nan


# ------------------------------------------------------------------ statistics

def row_spearman(a, b, mask):
    """Each day: rank correlation between a and b over coins where both exist and mask is True."""
    ok = mask & a.notna() & b.notna()
    ra = a.where(ok).rank(axis=1)
    rb = b.where(ok).rank(axis=1)
    ra = ra.sub(ra.mean(axis=1), axis=0)
    rb = rb.sub(rb.mean(axis=1), axis=0)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
    ic = num / den.where(den > 0)
    return ic.where(ok.sum(axis=1) >= MIN_COINS)


def nw_tstat(x, lags):
    """t-stat of the mean with a Newey-West correction for overlapping / autocorrelated days."""
    x = pd.Series(x).dropna().to_numpy()
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    var = e @ e / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (e[k:] @ e[:-k]) / n
    if var <= 1e-18:  # no day-to-day variation at all (e.g. a perfect cheat): infinitely sure
        return math.copysign(math.inf, x.mean()) if x.mean() != 0 else np.nan
    return x.mean() / math.sqrt(var / n)


def group_weights(alpha, mask, n=N_GROUPS):
    """{k: date x coin share of each coin in group k} for n equal groups by today's score (1 = lowest).
    Tied coins share the places they fill equally: if 40 coins tie for the top 16 places, each gets
    16/40 of a place. Breaking ties by coin name instead would pick coins alphabetically."""
    a = alpha.where(mask & alpha.notna())
    count = a.notna().sum(axis=1)
    lo = a.rank(axis=1, method="min") - 1     # a tied block fills places [lo, hi)
    hi = a.rank(axis=1, method="max")
    size = hi - lo
    ok = count >= MIN_COINS
    out = {}
    for k in range(1, n + 1):
        g_lo, g_hi = count * (k - 1) / n, count * k / n
        overlap = hi.clip(upper=g_hi, axis=0) - lo.clip(lower=g_lo, axis=0)
        out[k] = (overlap.clip(lower=0) / size).where(ok, axis=0)
    return out


def group_mean(r, w):
    """Weighted average of r in one group, each day (coins with no r are left out)."""
    w = w.where(r.notna())
    return (w * r).sum(axis=1, min_count=1) / w.sum(axis=1).where(lambda s: s > 0)


def capped(df, q=0.01):
    lo, hi = df.quantile(q, axis=1), df.quantile(1 - q, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def ann_sharpe(x):
    x = pd.Series(x).dropna()
    return x.mean() / x.std() * math.sqrt(365) if len(x) > 30 and x.std() > 0 else np.nan


# ------------------------------------------------------------------ the score card

def score(alpha, m):
    """Return (score card dict, daily series dict) for one alpha on Market m."""
    alpha = alpha.reindex(index=m.dates, columns=m.universe.columns)
    mask = m.universe
    has_val = alpha.notna() & mask
    card = {"coverage": round(float(has_val.sum().sum() / mask.sum().sum()), 4)}

    ic = row_spearman(alpha, m.fwd[1], mask)
    n_days = int(ic.notna().sum())
    card.update(days=n_days, ic=ic.mean(), ic_std=ic.std(), ic_t=nw_tstat(ic, 1))
    card["icir"] = card["ic"] / card["ic_std"] if card["ic_std"] and card["ic_std"] > 0 else np.nan
    card["ic_win"] = float((ic > 0).sum() / max(n_days, 1))
    direction = 1 if (card["ic"] or 0) >= 0 else -1
    card["direction"] = direction

    by_year = ic.groupby(ic.index.year).mean()
    for y, v in by_year.items():
        card[f"ic_{y}"] = v
    card["years_same_sign"] = int((np.sign(by_year) == direction).sum())
    card["years"] = int(by_year.notna().sum())

    for h in HORIZONS[1:]:
        ich = row_spearman(alpha, m.fwd[h], mask)
        card[f"ic_h{h}"] = ich.mean()
        card[f"ic_h{h}_t"] = nw_tstat(ich, h)
    card["ic_late"] = row_spearman(alpha, m.fwd_late, mask).mean()

    # 5 groups, equal weight; returns relative to the day's universe average
    g = group_weights(alpha, mask)
    r = m.fwd[1].where(mask)
    r_rel = r.sub(r.mean(axis=1), axis=0)
    r_cap = capped(r).sub(capped(r).mean(axis=1), axis=0)
    gm, gm_cap = {}, {}
    for k in range(1, N_GROUPS + 1):
        gm[k] = group_mean(r_rel, g[k])
        gm_cap[k] = group_mean(r_cap, g[k])
        card[f"g{k}_bp"] = gm[k].mean() * 1e4
    means = [card[f"g{k}_bp"] for k in range(1, N_GROUPS + 1)]
    # +1 = perfect staircase (group 1 < 2 < ... < 5), -1 = perfect staircase downwards
    card["monotonic"] = pd.Series(means).rank().corr(pd.Series(range(1, N_GROUPS + 1), dtype=float))

    # Long the best group, short the worst (in the alpha's own direction), daily rebalanced
    top, bot = (N_GROUPS, 1) if direction == 1 else (1, N_GROUPS)
    ls = gm[top] - gm[bot]
    ls_cap = gm_cap[top] - gm_cap[bot]
    w_long, w_short = g[top].fillna(0), g[bot].fillna(0)
    w = w_long.div(w_long.sum(axis=1), axis=0).fillna(0) - w_short.div(w_short.sum(axis=1), axis=0).fillna(0)
    turnover = w.diff().abs().sum(axis=1)                   # notional traded per day (both legs; 4 = full swap)
    cost = FEE * turnover
    funding = -(w * m.funding_next.fillna(0)).sum(axis=1)   # longs pay positive funding, shorts receive it
    net = ls - cost + funding
    card.update(ls_bp=ls.mean() * 1e4, ls_t=nw_tstat(ls, 1), ls_sharpe=ann_sharpe(ls),
                ls_capped_bp=ls_cap.mean() * 1e4,
                turnover=turnover.mean(), cost_bp=cost.mean() * 1e4, funding_bp=funding.mean() * 1e4,
                net_bp=net.mean() * 1e4, net_t=nw_tstat(net, 1), net_sharpe=ann_sharpe(net),
                net_annual_pct=net.mean() * 365 * 100)

    # Rule agreed in step 3: rank score only.
    t = abs(card["ic_t"]) if card["ic_t"] == card["ic_t"] else 0
    need = 2 if card["years"] >= 3 else card["years"]
    rank_ok = t >= STRONG_T and card["years_same_sign"] >= need
    card["verdict_rank_only"] = "strong" if rank_ok else ("maybe" if t >= MAYBE_T else "no")

    # Final verdict adds the money check: the long-short portfolio (before costs) must also earn money
    # in the same direction. Crypto returns are lopsided, so a rank score can be strong while the money is not.
    lt = card["ls_t"] if card["ls_t"] == card["ls_t"] else 0
    if rank_ok and lt >= STRONG_T:
        card["verdict"] = "strong"
    elif t >= MAYBE_T and lt >= MAYBE_T:
        card["verdict"] = "maybe"
    elif t >= STRONG_T:
        card["verdict"] = "rank only"   # usually right, but the occasional big miss eats the profit
    else:
        card["verdict"] = "no"
    card["survives_costs"] = bool((card["net_t"] == card["net_t"]) and card["net_t"] >= MAYBE_T)
    return card, {"ic": ic, "ls": ls, "net": net}
