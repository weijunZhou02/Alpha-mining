"""Hand checks for the evaluator's statistics."""
import math

import numpy as np
import pandas as pd
import pytest

import evaluate as ev


def table(rows):
    return pd.DataFrame(rows, dtype=float)


def test_rank_ic_perfect_and_reversed(monkeypatch):
    monkeypatch.setattr(ev, "MIN_COINS", 3)
    a = table([[1, 2, 3, 4]])
    mask = a.notna()
    assert ev.row_spearman(a, table([[10, 20, 30, 40]]), mask).iloc[0] == pytest.approx(1.0)
    assert ev.row_spearman(a, table([[4, 3, 2, 1]]), mask).iloc[0] == pytest.approx(-1.0)
    # coins outside the universe are ignored: only A, B, C count -> still perfect
    mask = table([[1, 1, 1, 0]]).astype(bool)
    assert ev.row_spearman(a, table([[1, 2, 3, -99]]), mask).iloc[0] == pytest.approx(1.0)


def test_too_few_coins_gives_no_ic(monkeypatch):
    monkeypatch.setattr(ev, "MIN_COINS", 20)
    a = table([[1, 2, 3, 4]])
    assert math.isnan(ev.row_spearman(a, a, a.notna()).iloc[0])


def test_groups(monkeypatch):
    monkeypatch.setattr(ev, "MIN_COINS", 5)
    a = table([list(range(10))])                  # 10 coins -> 2 per group
    g = ev.group_weights(a, a.notna(), n=5)
    label = sum(k * g[k] for k in g)
    assert label.iloc[0].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert all(g[k].iloc[0].sum() == pytest.approx(2) for k in g)


def test_tied_coins_share_places(monkeypatch):
    monkeypatch.setattr(ev, "MIN_COINS", 5)
    # 10 coins: 6 tied at 0, then 1..4. Top group (2 places) = coins 3 and 4. Bottom group (2 places)
    # falls inside the tie, so each of the 6 tied coins gets 2/6 of a place, whatever its name.
    a = table([[0, 0, 0, 0, 0, 0, 1, 2, 3, 4]])
    g = ev.group_weights(a, a.notna(), n=5)
    assert g[5].iloc[0].tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    assert g[1].iloc[0].tolist() == pytest.approx([1 / 3] * 6 + [0] * 4)
    assert sum(g[k].iloc[0] for k in g).tolist() == pytest.approx([1] * 10)   # every coin is fully placed
    # an uneven count splits the coin on the border: 7 coins -> 1.4 places per group
    b = table([list(range(7))])
    gb = ev.group_weights(b, b.notna(), n=5)
    assert gb[1].iloc[0].tolist() == pytest.approx([1, 0.4, 0, 0, 0, 0, 0])


def test_newey_west_matches_plain_t_without_lags():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1, 500)
    plain = x.mean() / (x.std(ddof=0) / math.sqrt(len(x)))
    assert ev.nw_tstat(x, 0) == pytest.approx(plain)
    # overlapping data (a 7-day moving sum of noise) must get a SMALLER t-stat after the correction
    y = pd.Series(rng.normal(0.05, 1, 2000)).rolling(7).sum().dropna()
    naive = y.mean() / (y.std(ddof=0) / math.sqrt(len(y)))
    assert abs(ev.nw_tstat(y, 7)) < abs(naive)
