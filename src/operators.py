"""Operator library: the building blocks that alpha formulas are made of.

Every input and output is a wide DataFrame (rows = date, columns = coin).

Two kinds of operators:
  * time-series (ts_*, delay, delta, ...): run down each coin's own history, using ALL of its days
    (also days when it was outside the top-80 universe);
  * cross-sectional (cs_*): compare coins on the same day, ONLY among coins in that day's universe.

Missing-data rule: a rolling value is computed when at least MIN_FRAC of the window has data,
otherwise it is NaN. Set MIN_FRAC = 1.0 for "strict" mode (full window required).
"""
import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

MIN_FRAC = 2 / 3
FLAT_CORR_ZERO = True  # correlation with a flat series = 0 (False: leave it missing, Vibe-Trading style)


def set_min_frac(frac):
    global MIN_FRAC
    MIN_FRAC = frac


def set_flat_corr_zero(flag):
    global FLAT_CORR_ZERO
    FLAT_CORR_ZERO = flag


def _n(d):
    """Window length: formulas sometimes use decimals like 9.91 -> floor (101 Alphas rule)."""
    n = int(math.floor(float(d)))
    if n < 1:
        raise ValueError(f"window must be >= 1, got {d}")
    return n


def _mp(n):
    """Minimum number of observations needed in a window of length n."""
    return max(1, math.ceil(n * MIN_FRAC))


def _like(values, ref):
    return pd.DataFrame(values, index=ref.index, columns=ref.columns)


def _clean(df):
    return df.replace([np.inf, -np.inf], np.nan)


def _windows(x, n, func, need_last=True, min_count=None):
    """Apply func to every rolling window. func gets an array (T, cols, n), oldest -> newest, NaN = missing.

    Windows that are too empty (or whose last value is missing, if need_last) become NaN.
    Columns are processed in chunks to keep memory small.
    """
    a = x.to_numpy(dtype=float)
    T, N = a.shape
    pad = np.vstack([np.full((n - 1, N), np.nan), a])
    out = np.full((T, N), np.nan)
    step = max(1, int(3e7 // max(1, T * n)))
    for j in range(0, N, step):
        w = sliding_window_view(pad[:, j:j + step], n, axis=0)  # (T, c, n)
        with np.errstate(all="ignore"):
            out[:, j:j + step] = func(w)
    count = pd.DataFrame(a).notna().rolling(n, min_periods=1).sum().to_numpy()
    bad = count < (min_count if min_count is not None else _mp(n))
    if need_last:
        bad |= np.isnan(a)
    out[bad] = np.nan
    return _like(out, x)


# ---------------------------------------------------------------- element-wise

def sign(x):
    return np.sign(x)


def log(x):
    return _clean(np.log(x.where(x > 0))) if isinstance(x, pd.DataFrame) else math.log(x)


def signed_power(x, a):
    return np.sign(x) * np.abs(x) ** a


def power(x, a):
    with np.errstate(all="ignore"):
        out = x ** a
    return _clean(out) if isinstance(out, pd.DataFrame) else out


def _pick(a, b, take_a):
    """Element-wise choice between a and b (either may be a number); NaN if either is NaN."""
    if not isinstance(a, pd.DataFrame) and not isinstance(b, pd.DataFrame):
        return a if take_a(a, b) else b
    if not isinstance(a, pd.DataFrame):
        a, b = b, a
    out = a.where(take_a(a, b), b)
    return out.where(a.notna() & (b.notna() if isinstance(b, pd.DataFrame) else True))


def max2(a, b):
    """Element-wise larger of two values."""
    return _pick(a, b, lambda p, q: p >= q)


def min2(a, b):
    """Element-wise smaller of two values."""
    return _pick(a, b, lambda p, q: p <= q)


# ---------------------------------------------------------------- time-series

def delay(x, d=1):
    return x.shift(_n(d))


def delta(x, d=1):
    return x - x.shift(_n(d))


def ts_sum(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=_mp(n)).sum()


def ts_mean(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=_mp(n)).mean()


def ts_std(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=max(2, _mp(n))).std(ddof=1)


def ts_var(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=max(2, _mp(n))).var(ddof=1)


def ts_skew(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=max(3, _mp(n))).skew()


def ts_kurt(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=max(4, _mp(n))).kurt()


def ts_min(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=_mp(n)).min()


def ts_max(x, d):
    n = _n(d)
    return x.rolling(n, min_periods=_mp(n)).max()


def ts_quantile(x, d, q):
    n = _n(d)
    return x.rolling(n, min_periods=_mp(n)).quantile(q)


def ts_product(x, d):
    n = _n(d)
    return _windows(x, n, lambda w: np.nanprod(w, axis=2), need_last=False)


def ts_count(cond, d):
    """How many of the last d days the condition (1/0) was true."""
    return ts_sum(cond, d)


def ts_sumif(x, d, cond):
    return ts_sum(x.where(cond > 0, 0.0).where(x.notna() & cond.notna()), d)


def ts_rank(x, d):
    """Where today's value sits among the last d values: percentile in (0, 1], ties averaged."""
    n = _n(d)

    def f(w):
        last = w[:, :, -1:]
        less = (w < last).sum(axis=2)
        equal = (w == last).sum(axis=2)
        valid = (~np.isnan(w)).sum(axis=2)
        return (less + (equal + 1) / 2) / valid

    return _windows(x, n, f)


def ts_argmax(x, d):
    """Position of the max inside the window, 0 = oldest day, d-1 = today (ties -> most recent)."""
    n = _n(d)

    def f(w):
        filled = np.where(np.isnan(w), -np.inf, w)
        return n - 1 - np.argmax(filled[:, :, ::-1], axis=2)

    return _windows(x, n, f)


def ts_argmin(x, d):
    n = _n(d)

    def f(w):
        filled = np.where(np.isnan(w), np.inf, w)
        return n - 1 - np.argmin(filled[:, :, ::-1], axis=2)

    return _windows(x, n, f)


def ts_highday(x, d):
    """GTJA HIGHDAY: how many days ago the max of the last d days happened (0 = today)."""
    return (_n(d) - 1) - ts_argmax(x, d)


def ts_lowday(x, d):
    return (_n(d) - 1) - ts_argmin(x, d)


def _weighted(x, d, weights):
    """Weighted average over the window; weights oldest -> newest; missing days are skipped."""
    n = _n(d)
    wts = np.asarray(weights, dtype=float)

    def f(w):
        ok = ~np.isnan(w)
        num = np.where(ok, w, 0.0) @ wts
        den = ok.astype(float) @ wts
        return num / den

    return _windows(x, n, f)


def decay_linear(x, d):
    """Weights d, d-1, ..., 1 with today getting d (101 Alphas / GTJA DECAYLINEAR)."""
    n = _n(d)
    return _weighted(x, n, np.arange(1, n + 1))


def wma(x, d):
    """GTJA WMA: weights 0.9^i, i = days before today (today weight 1)."""
    n = _n(d)
    return _weighted(x, n, 0.9 ** np.arange(n - 1, -1, -1))


def sma_gtja(x, n, m):
    """GTJA SMA(A, n, m): Y_t = (m * A_t + (n - m) * Y_{t-1}) / n, i.e. an EMA with alpha = m / n."""
    ema = x.ewm(alpha=float(m) / float(n), adjust=False, ignore_na=True, min_periods=_mp(_n(n))).mean()
    return ema.where(x.notna())


def ts_corr(x, y, d):
    """Rolling correlation. If either series is flat in the window (no movement), the answer is 0:
    nothing moved together. (Otherwise one flat stretch would blank out every formula built on it.)"""
    n = _n(d)
    ok = x.notna() & y.notna()
    mp = max(2, _mp(n))
    xs, ys = x.where(ok), y.where(ok)
    corr = _clean(xs.rolling(n, min_periods=mp).corr(ys))
    flat = (xs.rolling(n, min_periods=mp).std() < 1e-12) | (ys.rolling(n, min_periods=mp).std() < 1e-12)
    return corr.mask(flat, 0.0 if FLAT_CORR_ZERO else np.nan).clip(-1, 1)


def ts_cov(x, y, d):
    n = _n(d)
    ok = x.notna() & y.notna()
    return x.where(ok).rolling(n, min_periods=max(2, _mp(n))).cov(y.where(ok))


def _reg_moments(y, x, n, min_count):
    ok = y.notna() & x.notna()
    ym, xm = y.where(ok), x.where(ok)
    r = lambda s: s.rolling(n, min_periods=min_count).mean()
    mx, my = r(xm), r(ym)
    cov = r(xm * ym) - mx * my
    var = r(xm * xm) - mx * mx
    return mx, my, cov, var, r(ym * ym) - my * my


def ts_regbeta(y, x, d, min_count=None):
    """Slope of a rolling regression of y on x over d days."""
    n = _n(d)
    _, _, cov, var, _ = _reg_moments(y, x, n, min_count or max(3, _mp(n)))
    return _clean(cov / var.where(var > 1e-30))


def ts_regresi(y, x, d):
    """Today's residual of a rolling regression of y on x over d days."""
    n = _n(d)
    mx, my, cov, var, _ = _reg_moments(y, x, n, max(3, _mp(n)))
    beta = _clean(cov / var.where(var > 1e-30))
    return y - (my - beta * mx) - beta * x


def time_index(ref):
    """A table holding 1, 2, 3, ... down each column (the 'time' axis for trend regressions)."""
    return _like(np.broadcast_to(np.arange(1, len(ref) + 1, dtype=float)[:, None], ref.shape).copy(), ref)


def ts_slope(x, d):
    return ts_regbeta(x, time_index(x), d)


def ts_rsquare(x, d):
    return ts_corr(x, time_index(x), d) ** 2


def ts_resi(x, d):
    return ts_regresi(x, time_index(x), d)


def ts_sumac_range(x, d):
    """Max minus min of the running sum of x inside each window (rescaled-range numerator)."""
    n = _n(d)

    def f(w):
        c = np.nancumsum(w, axis=2)
        return c.max(axis=2) - c.min(axis=2)

    return _windows(x, n, f)


# ---------------------------------------------------------------- cross-sectional (within universe)

def cs_rank(x, mask):
    """Percentile rank among today's universe coins: (0, 1], ties averaged."""
    return x.where(mask).rank(axis=1, method="average", pct=True)


def cs_zscore(x, mask):
    x = x.where(mask)
    return _clean(x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1, ddof=1), axis=0))


def cs_scale(x, mask, a=1.0):
    """Resize so that the absolute values add up to a across today's universe."""
    x = x.where(mask)
    return _clean(x.div(x.abs().sum(axis=1), axis=0) * a)


def cs_demean_by_group(x, groups, mask):
    """Subtract the average of the coin's group (sector) on the same day."""
    x = x.where(mask)
    out = x.copy()
    for g in pd.unique(groups.values):
        cols = groups.index[groups.values == g]
        cols = [c for c in cols if c in x.columns]
        if cols:
            out[cols] = x[cols].sub(x[cols].mean(axis=1), axis=0)
    return out


def cs_winsorize(x, mask, q=0.01):
    """Cap each day's values at its q and 1-q quantiles (within the universe)."""
    x = x.where(mask)
    lo, hi = x.quantile(q, axis=1), x.quantile(1 - q, axis=1)
    return x.clip(lower=lo, upper=hi, axis=0)
