"""Checks for the step 7 loop's safety rules (no data files needed)."""
import numpy as np
import pandas as pd
import pytest

import formula as fx
import llm_loop as ll


@pytest.mark.parametrize("text", [
    "-1 * decay_linear(abs(funding), 5)",
    "rank(volume / trades) - rank(delay(volume / trades, 20))",
    "correlation(returns, mkt_returns, 30)",
    "indneutralize(ts_rank(premium, 20), IndClass.sector)",
    "returns > 0 ? max(returns, 0) : min(returns, 0)",
    "adv20 / adv60",
    "signedpower(taker_buy_ratio - 0.5, 2)",
])
def test_allowed_formulas_pass(text):
    tree, why = ll.check_formula(text)
    assert why is None and tree is not None


@pytest.mark.parametrize("text, words", [
    ("rank(skew(close, 5))", "unknown function"),
    ("rank(open_interest)", "unknown field"),
    ("delay(close, -1)", "plain whole number"),          # a negative lag would look into the future
    ("delta(close, 2.5)", "plain whole number"),
    ("ts_rank(close, 300)", "from 3 to 250"),
    ("correlation(close, volume, 2)", "from 3 to 250"),   # 2-day correlations are only +-1 and rounding noise
    ("sum(close, adv20)", "plain whole number"),
    ("rank(close) + IndClass.sector", "only appear inside indneutralize"),
    ("indneutralize(close, volume)", "IndClass.sector"),
    ("max(close, 5)", "between -1 and 1"),
    ("rank(close, 5)", "takes 1 argument"),
    ("rank((close", "syntax error"),
    ("", "empty"),
])
def test_rule_breaks_are_refused(text, words):
    tree, why = ll.check_formula(text)
    assert tree is None and words in why


def test_size_limits():
    assert "characters" in ll.check_formula("rank(close)" + " + rank(close)" * 40)[1]
    many = "+".join(["open"] * 31)                # 31 fields + 30 pluses = 61 parts, but only 154 characters
    assert "parts" in ll.check_formula(many)[1]


def test_canonical_ignores_case_spaces_and_outer_sign():
    c = lambda t: ll.canonical(fx.parse(t))
    assert c("-1 * rank(close)") == c("RANK( close )") == c("-rank(close)") == c("rank(close) * -1")
    assert c("rank(delta(close, 3))") != c("rank(delta(close, 5))")


def _ctx(df):
    mask = pd.DataFrame(True, index=df.index, columns=df.columns)
    return fx.Context({"close": df}, mask), mask


def test_cutoff_test_catches_look_ahead(monkeypatch):
    dates = pd.date_range("2021-01-01", periods=400, freq="D")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(100 + rng.normal(size=(400, 5)).cumsum(axis=0), index=dates, columns=list("ABCDE"))
    cut = "2021-08-15"
    full_ctx, full_mask = _ctx(close)
    cut_ctxs = {cut: _ctx(close.loc[:cut])}
    honest = fx.parse("ts_rank(delta(close, 3), 20)")
    assert not ll.peeks_ahead(honest, ll.compute(honest, full_ctx, full_mask), cut_ctxs)
    # Negative lags are refused twice already (by check_formula and by operators._n), so plant a cheating function:
    monkeypatch.setitem(fx.WQ101, "peek", lambda a, c: a[0].shift(-1))   # tomorrow's value
    cheat = fx.parse("peek(close)")
    assert ll.peeks_ahead(cheat, ll.compute(cheat, full_ctx, full_mask), cut_ctxs)


def test_random_formulas_follow_the_rules():
    rng = np.random.default_rng(1)
    for _ in range(300):
        f = ll.random_formula(rng)
        assert ll.check_formula(f)[1] is None, f


def test_fake_llm_and_feedback_text():
    items, usage = ll.FakeLLM().ask("system", "user")
    assert len(items) == 5 and all("formula" in i for i in items) and usage["output"] == 0
    rec = {"formula": "rank(close)", "stage": "weak", "reason": "own signal too weak", "ic": None, "ic_t": float("nan")}
    assert "n/a" in ll.describe(rec) and "too weak" in ll.describe(rec)
    hidden = {"formula": "x", "stage": "confirm", "screen_net_gain_bp": 1.5, "screen_ic_gain": 0.001,
              "confirm_net_gain_bp": -3.0, "reason": ""}
    text = ll.describe(hidden)
    assert "-3.0" not in text and "confirm half-year failed" in text      # confirm numbers never reach the LLM
    msg = ll.round_message(3, 40, ll.THEMES[2], 5, [hidden], [("LLM_001", "rank(close)")])
    assert "Round 3 of 40" in msg and "At most 2" in msg and "LLM_001" in msg


def test_cost():
    u = {"input": 1_000_000, "cache_write": 0, "cache_read": 1_000_000, "output": 1_000_000}
    assert ll.cost_usd("claude-sonnet-5-5", u) == pytest.approx(2.0 + 0.10 + 10.0)


def test_text_reply_fallback():
    text = 'Here you go:\n```json\n{"alphas": [{"idea": "x", "formula": "rank(close)"}]}\n```'
    assert ll.parse_text_reply(text) == [{"idea": "x", "formula": "rank(close)"}]
    assert ll.parse_text_reply("no json here") == []
