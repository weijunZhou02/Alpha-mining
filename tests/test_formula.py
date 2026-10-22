"""Checks for the formula reader: does it read formulas the way a person would?"""
import numpy as np
import pandas as pd
import pytest

import formula as fx
import operators as op


def num(text):
    return fx.evaluate(fx.parse(text), fx.Context({}, None), "wq101")


def test_math_order():
    assert num("1 + 2 * 3") == 7
    assert num("(1 + 2) * 3") == 9
    assert num("-2 ^ 2") == -4            # power first, then the minus sign
    assert num("2 ^ 3 ^ 2") == 512        # 2^(3^2)
    assert num("10 - 4 - 3") == 3         # left to right
    assert num(".5 + 2.") == 2.5          # 101-style numbers
    assert num("1 + 1e-12") == 1 + 1e-12  # Qlib-style tiny numbers


def test_conditions():
    assert num("1 < 2 ? 10 : 20") == 10
    assert num("1 > 2 ? 10 : 3 > 2 ? 30 : 40") == 30   # nested a ? b : (c ? d : e)
    assert num("(1 < 2) + (3 < 2)") == 1               # true counts as 1
    assert num("(1 < 2) || (3 < 2)") == 1
    assert num("(1 < 2) && (3 < 2)") == 0


@pytest.fixture
def ctx():
    idx = pd.date_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [1, 2, 3, 2, 5], "B": [5, 4, 3, 4, 1]}, index=idx, dtype=float)
    open_ = pd.DataFrame({"A": [1, 1, 4, 2, 4], "B": [5, 5, 3, 3, 2]}, index=idx, dtype=float)
    vol = pd.DataFrame({"A": [10, 20, 30, 40, 50], "B": [5, 5, 5, 5, 5]}, index=idx, dtype=float)
    mask = pd.DataFrame(True, index=idx, columns=["A", "B"])
    op.set_min_frac(1.0)
    yield fx.Context({"close": close, "open": open_, "volume": vol,
                      "$close": close, "$high": close}, mask), close, open_, vol
    op.set_min_frac(2 / 3)


def test_101_style(ctx):
    c, close, open_, vol = ctx
    got = fx.evaluate(fx.parse("(close > open) ? 1 : -1"), c, "wq101")
    want = np.where(close > open_, 1.0, -1.0)
    np.testing.assert_array_equal(got.to_numpy(), want)
    got = fx.evaluate(fx.parse("rank(close)"), c, "wq101")
    pd.testing.assert_frame_equal(got, op.cs_rank(close, c.mask))


def test_gtja_style(ctx):
    c, close, open_, vol = ctx
    # sum of volume on up days over 2 days
    got = fx.evaluate(fx.parse("SUM((CLOSE>DELAY(CLOSE,1)?VOLUME:0),2)"), c, "gtja191")
    assert got["A"].tolist()[2:] == [50.0, 30.0, 50.0]      # A rises on days 2,3,5
    # '=' means equal in GTJA
    got = fx.evaluate(fx.parse("CLOSE=DELAY(CLOSE,1)?0:1"), c, "gtja191")
    assert got["A"].tolist()[1:] == [1.0, 1.0, 1.0, 1.0]
    # MAX(A, number>=2) is a rolling max; MAX(A, 0) is element-wise
    pd.testing.assert_frame_equal(fx.evaluate(fx.parse("MAX(CLOSE,2)"), c, "gtja191"), op.ts_max(close, 2))
    got = fx.evaluate(fx.parse("MAX(CLOSE-OPEN,0)"), c, "gtja191")
    assert got["A"].tolist() == [0.0, 1.0, 0.0, 0.0, 1.0]
    # REGBETA against SEQUENCE(n) = trend slope
    pd.testing.assert_frame_equal(fx.evaluate(fx.parse("REGBETA(CLOSE,SEQUENCE(3))"), c, "gtja191"),
                                  op.ts_slope(close, 3))


def test_qlib_style(ctx):
    c, close, *_ = ctx
    got = fx.evaluate(fx.parse("Ref($close, 1)/$close"), c, "qlib")
    pd.testing.assert_frame_equal(got, close.shift(1) / close)
    got = fx.evaluate(fx.parse("IdxMax($high, 3)/3"), c, "qlib")
    assert got["A"].tolist()[2:] == [1.0, 2 / 3, 1.0]      # 1-based position of the high in the window


def test_bad_formulas_fail_loudly():
    with pytest.raises(SyntaxError):
        fx.parse("rank(close")
    with pytest.raises(NameError):
        fx.evaluate(fx.parse("nosuchthing(1)"), fx.Context({}, None), "wq101")
