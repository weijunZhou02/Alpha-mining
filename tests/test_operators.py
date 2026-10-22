"""Hand-calculated checks: tiny inputs where the right answer can be worked out on paper."""
import math

import numpy as np
import pandas as pd
import pytest

import operators as op

NAN = np.nan


def col(values):
    """One coin, one value per day."""
    return pd.DataFrame({"A": values}, index=pd.date_range("2024-01-01", periods=len(values)), dtype=float)


def vals(df):
    return df["A"].tolist()


def same(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        if isinstance(w, float) and math.isnan(w):
            assert math.isnan(g), (got, want)
        else:
            assert g == pytest.approx(w, abs=1e-9), (got, want)


@pytest.fixture(autouse=True)
def strict():
    """Most checks use strict mode (full window needed); the missing-data rule is tested on its own."""
    op.set_min_frac(1.0)
    yield
    op.set_min_frac(2 / 3)


# --- simple time-series --------------------------------------------------------

def test_delay_and_delta():
    x = col([1, 2, 3, 4, 5])
    same(vals(op.delay(x, 1)), [NAN, 1, 2, 3, 4])
    same(vals(op.delta(x, 2)), [NAN, NAN, 2, 2, 2])         # 3-1, 4-2, 5-3


def test_mean_sum_window_includes_today():
    x = col([1, 2, 3, 4, 5])
    same(vals(op.ts_mean(x, 3)), [NAN, NAN, 2, 3, 4])      # day 3: (1+2+3)/3
    same(vals(op.ts_sum(x, 3)), [NAN, NAN, 6, 9, 12])


def test_std_is_sample_std():
    x = col([2, 4, 4, 4, 5, 5, 7, 9])                       # population std 2, sample std sqrt(32/7)
    assert vals(op.ts_std(x, 8))[-1] == pytest.approx(math.sqrt(32 / 7))


def test_product_min_max():
    x = col([1, 2, 3, 4])
    same(vals(op.ts_product(x, 3)), [NAN, NAN, 6, 24])
    same(vals(op.ts_max(col([1, 5, 2, 3]), 3)), [NAN, NAN, 5, 5])
    same(vals(op.ts_min(col([4, 5, 2, 3]), 3)), [NAN, NAN, 2, 2])


def test_ts_rank_is_percentile_of_today():
    same(vals(op.ts_rank(col([3, 1, 2]), 3)), [NAN, NAN, 2 / 3])   # 2 is the middle of {3,1,2}
    same(vals(op.ts_rank(col([1, 2, 3]), 3)), [NAN, NAN, 1.0])     # today is the highest
    same(vals(op.ts_rank(col([2, 2, 2]), 3)), [NAN, NAN, 2 / 3])   # ties share the average rank


def test_argmax_highday():
    same(vals(op.ts_argmax(col([1, 3, 2]), 3)), [NAN, NAN, 1])     # 0 = oldest, 2 = today
    same(vals(op.ts_argmax(col([5, 5, 1]), 3)), [NAN, NAN, 1])     # tie -> most recent
    same(vals(op.ts_argmin(col([1, 3, 2]), 3)), [NAN, NAN, 0])
    same(vals(op.ts_highday(col([1, 3, 2]), 3)), [NAN, NAN, 1])    # the high was 1 day ago
    same(vals(op.ts_lowday(col([3, 2, 1]), 3)), [NAN, NAN, 0])     # the low is today


def test_decay_linear_gives_today_the_biggest_weight():
    # weights 1, 2, 3 (today = 3): (1*1 + 2*2 + 3*3) / 6
    same(vals(op.decay_linear(col([1, 2, 3]), 3)), [NAN, NAN, 14 / 6])


def test_gtja_wma_weights_point_nine():
    # weights 0.9^2, 0.9^1, 0.9^0 for oldest -> today
    want = (0.81 * 1 + 0.9 * 2 + 1 * 3) / (0.81 + 0.9 + 1)
    same(vals(op.wma(col([1, 2, 3]), 3)), [NAN, NAN, want])


def test_gtja_sma_recursion():
    # Y1 = 10, Y2 = (1*20 + 2*10)/3 = 13.333, Y3 = (1*30 + 2*13.333)/3 = 18.889 ; strict: first 2 hidden
    y2 = (20 + 2 * 10) / 3
    y3 = (30 + 2 * y2) / 3
    same(vals(op.sma_gtja(col([10, 20, 30]), 3, 1)), [NAN, NAN, y3])
    op.set_min_frac(0.0)
    same(vals(op.sma_gtja(col([10, 20, 30]), 3, 1)), [10, y2, y3])


def test_corr_cov():
    x = col([1, 2, 3, 4])
    assert vals(op.ts_corr(x, col([2, 4, 6, 8]), 4))[-1] == pytest.approx(1.0)
    assert vals(op.ts_corr(x, col([8, 6, 4, 2]), 4))[-1] == pytest.approx(-1.0)
    assert vals(op.ts_cov(col([1, 2, 3]), col([1, 2, 3]), 3))[-1] == pytest.approx(1.0)  # sample variance
    assert vals(op.ts_corr(x, col([5, 5, 5, 5]), 4))[-1] == 0.0                          # flat line: no co-movement
    assert math.isnan(vals(op.ts_corr(col([1, 2]), col([5, 5]), 4))[-1])                 # too little data: missing


def test_regression_family():
    assert vals(op.ts_regbeta(col([2, 4, 6]), col([1, 2, 3]), 3))[-1] == pytest.approx(2.0)
    assert vals(op.ts_slope(col([1, 3, 5, 7]), 4))[-1] == pytest.approx(2.0)
    # y = 1, 2, 4 on x = 1, 2, 3: slope 1.5, intercept -2/3, fitted 3.8333 at x=3 -> residual 0.16667
    assert vals(op.ts_regresi(col([1, 2, 4]), col([1, 2, 3]), 3))[-1] == pytest.approx(4 - (-2 / 3 + 4.5))
    assert vals(op.ts_rsquare(col([1, 2, 3]), 3))[-1] == pytest.approx(1.0)


def test_count_sumif_sumac():
    cond = col([1, 0, 1, 1])
    same(vals(op.ts_count(cond, 3)), [NAN, NAN, 2, 2])
    same(vals(op.ts_sumif(col([5, 6, 7, 8]), 3, cond)), [NAN, NAN, 12, 15])  # 5+7, 7+8
    same(vals(op.ts_sumac_range(col([1, -2, 3]), 3)), [NAN, NAN, 3])        # running sums 1,-1,2


def test_quantile():
    same(vals(op.ts_quantile(col([1, 2, 3, 4, 5]), 5, 0.5)), [NAN, NAN, NAN, NAN, 3])


# --- element-wise ----------------------------------------------------------------

def test_elementwise():
    x = col([-2, 0, 3])
    same(vals(op.signed_power(x, 2)), [-4, 0, 9])
    same(vals(op.log(col([1, -1, math.e]))), [0, NAN, 1])                    # log of a negative -> missing
    same(vals(op.max2(x, 0)), [0, 0, 3])
    same(vals(op.min2(0, x)), [-2, 0, 0])
    same(vals(op.max2(col([1, NAN, 5]), col([2, 2, 2]))), [2, NAN, 5])


# --- cross-sectional (within the universe) ---------------------------------------

def day(values, inside):
    cols = list("ABCD")[:len(values)]
    x = pd.DataFrame([values], columns=cols, dtype=float)
    mask = pd.DataFrame([inside], columns=cols)
    return x, mask


def test_rank_only_inside_universe():
    x, mask = day([10, 30, 20, 99], [True, True, True, False])   # D is not in today's top 80
    same(op.cs_rank(x, mask).iloc[0].tolist(), [1 / 3, 1.0, 2 / 3, NAN])


def test_scale_zscore_group():
    x, mask = day([1, -3], [True, True])
    same(op.cs_scale(x, mask).iloc[0].tolist(), [0.25, -0.75])                # |0.25| + |-0.75| = 1
    x, mask = day([1, 2, 3], [True, True, True])
    same(op.cs_zscore(x, mask).iloc[0].tolist(), [-1, 0, 1])
    x, mask = day([1, 3, 10, 20], [True] * 4)
    groups = pd.Series(["L1", "L1", "Meme", "Meme"], index=list("ABCD"))
    same(op.cs_demean_by_group(x, groups, mask).iloc[0].tolist(), [-1, 1, -5, 5])


# --- the rules we agreed on -------------------------------------------------------

def test_missing_data_rule_two_thirds():
    x = col([1, NAN, 3])
    same(vals(op.ts_mean(x, 3)), [NAN, NAN, NAN])           # strict: a gap in the window -> missing
    op.set_min_frac(2 / 3)
    same(vals(op.ts_mean(x, 3)), [NAN, NAN, 2.0])           # 2 of 3 days present -> use them
    same(vals(op.ts_mean(col([1, NAN, NAN]), 3)), [NAN, NAN, NAN])  # only 1 of 3 -> missing


def test_no_look_ahead():
    """Changing tomorrow's number must not change any value up to today."""
    rng = np.random.default_rng(0)
    a = col(rng.normal(size=40))
    b = a.copy()
    b.iloc[-1] = 1000.0
    funcs = [lambda x: op.ts_mean(x, 5), lambda x: op.ts_rank(x, 5), lambda x: op.decay_linear(x, 5),
             lambda x: op.ts_argmax(x, 5), lambda x: op.sma_gtja(x, 5, 2), lambda x: op.ts_slope(x, 5),
             lambda x: op.ts_corr(x, x.shift(1), 5), lambda x: op.wma(x, 5), lambda x: op.ts_product(x, 5)]
    for f in funcs:
        pd.testing.assert_frame_equal(f(a).iloc[:-1], f(b).iloc[:-1])
