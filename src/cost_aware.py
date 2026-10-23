"""Option B: a cost- and risk-aware way to trade the "Trees · broad + LLM" signal.

    .venv/bin/python src/cost_aware.py        # choose the settings on 2022-2023 (about 10 min)

Agreed design (README, "Option B"):
  - Same signal: the step 6 trees on the 159 broad-list alphas plus the kept LLM alphas.
  - Choices tried: what the trees predict (next 1 day or next 5 days) x smoothing of the daily score
    (none, 3-day average, 7-day average) = 6 versions. Fixed rules on in all 6:
      smaller positions in wilder coins: size proportional to 1 / 30-day volatility, inside each side
      volatility targeting: the whole book scales by (usual volatility / last 30 days' volatility),
                            between 0.25x and 2x, using only past days
  - Settings are chosen by learning from 2021 and trading 2022-2023 month by month (data cut at 2023-12-31).
    The best Sharpe after costs wins; all 6 are reported, plus the winner with each fixed rule switched off.
  - It enters the final exam only if it beats the plain version (1-day target, no smoothing, equal weight,
    no volatility targeting) on 2022-2023 after costs.
  - Double fees (0.10%) are reported as extra information only.
Writes data/eval/cost_aware/.
"""
import json
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import combine as cb
import evaluate as ev
import llm_loop as ll
from alphas import AlphaEngine
from config import DATA

OUT = DATA / "eval" / "cost_aware"
TUNE_MONTHS = pd.date_range("2022-01-01", "2023-12-01", freq="MS")
VOL_DAYS = 30
SCALE_MIN, SCALE_MAX = 0.25, 2.0


@dataclass(frozen=True)
class Version:
    target_days: int = 1          # what the trees learn: the next 1 or 5 days' return (ranked)
    smooth_days: int = 1          # average of the last n daily scores (1 = no smoothing)
    inverse_vol: bool = True      # smaller positions in wilder coins
    vol_target: bool = True       # scale the whole book to keep its swings steady

    @property
    def name(self):
        if self == PLAIN:
            return "plain (1-day, no smoothing, equal weight)"
        parts = [f"{self.target_days}-day target", "no smoothing" if self.smooth_days == 1 else f"{self.smooth_days}-day average"]
        if not self.inverse_vol:
            parts.append("equal weight")
        if not self.vol_target:
            parts.append("no vol targeting")
        return ", ".join(parts)


PLAIN = Version(1, 1, False, False)
GRID = [Version(t, s) for t in (1, 5) for s in (1, 3, 7)]


# ------------------------------------------------------------------ the signal

def forward_return(m, close, days):
    """Next `days` days' return on Market m's days, blank where it would reach past the period's end.
    1 day = the step 6 answer key itself (m.fwd[1])."""
    if days == 1:
        return m.fwd[1]
    fwd = (close.shift(-days) / close - 1).reindex(index=m.universe.index, columns=m.universe.columns).copy()
    fwd.loc[fwd.index > pd.Timestamp(m.end) - pd.Timedelta(days=days)] = np.nan
    return fwd.where(m.universe)


class Window:
    """Rows a monthly refit may use (like combine.Sets, but the gap grows with the target: days + 1)."""

    def __init__(self, ctx, d0, gap_days):
        d = ctx.rows.date
        has_y = ~np.isnan(ctx.y)
        gap = pd.Timedelta(days=gap_days)
        hold_start = d0 - pd.DateOffset(months=cb.HOLDOUT_MONTHS)
        self.last = d0 - gap
        self.train = (d <= self.last) & has_y
        self.inner = (d <= hold_start - gap) & has_y
        self.hold = (d >= hold_start) & (d <= self.last) & has_y
        self.pred = (d >= d0) & (d <= d0 + pd.offsets.MonthEnd(0))
        assert d[self.train].max() <= d0 - gap and d[self.pred].min() >= d0


def predict(ctx, cols, months, target_days):
    """Walk forward: refit the step 6 trees at the start of each month on earlier rows only."""
    run = cb.fit_trees(ctx, cols)
    pred = np.full(len(ctx.y), np.nan)
    for d0 in months:
        w = Window(ctx, d0, target_days + 1)
        p, _ = run(w)
        pred[w.pred] = p
    rows = ~np.isnan(pred)
    return ctx.rows.to_wide(pred[rows], rows)


# ------------------------------------------------------------------ trading the signal

def daily_rank(score, universe):
    return cb.centred_rank(score, universe.reindex_like(score))


def smooth(rank, days):
    return rank if days == 1 else rank.rolling(days, min_periods=1).mean().where(rank.notna())


def sides(score, universe, vol, inverse_vol):
    """(long, short) weights, each side summing to 1: top and bottom 20%, ties shared as in step 3."""
    long, short = cb.holdings(score, universe)
    if inverse_vol:
        inv = (1 / vol.reindex_like(long)).replace([np.inf, -np.inf], np.nan)
        inv = inv.mask(inv.isna(), inv.median(axis=1), axis=0)        # a coin with no volatility yet: typical size
        long, short = long * inv, short * inv
        long = long.div(long.sum(axis=1).where(lambda s: s > 0), axis=0).fillna(0)
        short = short.div(short.sum(axis=1).where(lambda s: s > 0), axis=0).fillna(0)
    return long, short


def trade(score, m, version, vol, fee=ev.FEE):
    """Daily results of trading `score` on Market m with the version's sizing rules (like combine.judge)."""
    U = m.universe
    score = score.reindex(index=U.index, columns=U.columns)
    long, short = sides(score, U, vol, version.inverse_vol)
    r = m.fwd[1].where(U)
    gross = ev.group_mean(r, long) - ev.group_mean(r, short)
    if version.vol_target:
        held = gross.where(score.notna().sum(axis=1) >= ev.MIN_COINS)
        rv = held.rolling(VOL_DAYS, min_periods=20).std()
        usual = rv.expanding(min_periods=60).median()
        scale = (usual / rv).shift(1).clip(SCALE_MIN, SCALE_MAX).fillna(1.0)   # yesterday's information only
    else:
        scale = pd.Series(1.0, index=U.index)
    w = (long - short).mul(scale, axis=0)
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    cost = fee * turnover
    funding = -(w * m.funding_next.fillna(0)).sum(axis=1)
    gross = gross * scale
    ic = ev.row_spearman(score, m.fwd[1], U)
    daily = pd.DataFrame({"ic": ic, "gross": gross, "cost": cost, "funding": funding,
                          "net": gross - cost + funding, "turnover": turnover, "scale": scale})
    return daily.dropna(subset=["gross"])


def run_version(version, preds, m, vol, fee=ev.FEE):
    score = smooth(daily_rank(preds[version.target_days], m.universe), version.smooth_days)
    return trade(score, m, version, vol, fee)


# ------------------------------------------------------------------ choosing the settings (2022-2023)

def build_ctx(last_day):
    """Features (broad + kept LLM alphas) and engine for 2021-2023, with every table cut at last_day."""
    assert last_day <= ev.PERIODS["train"][1], "option B is chosen on 2021-2023 only"
    m = ev.Market("train")
    ctx = cb.Ctx()
    ctx.rows = cb.Rows(m.universe)
    broad = pd.read_csv(cb.EVAL / "dedupe" / "broad_list.csv", index_col=0).index.tolist()
    X = pd.read_parquet(cb.OUT / "features.parquet")
    X = X[X.index.get_level_values(0) <= pd.Timestamp(m.end)]
    assert X.index.equals(ctx.rows.index)
    eng = ll.cut_engine(AlphaEngine.load(), last_day)
    fctx = ll.make_context(eng)
    extra = []
    for k in pd.read_csv(ll.OUT / "kept.csv").itertuples():
        tree, why = ll.check_formula(k.formula)
        assert why is None, why
        a = ll.compute(tree, fctx, eng.universe).loc[m.start:m.end]
        extra.append(np.nan_to_num(ctx.rows.to_long(cb.centred_rank(a, m.universe) * k.direction)).astype(np.float32))
    ctx.X = np.column_stack([X[broad].to_numpy(np.float32)] + extra)
    vol = eng.p["returns"].rolling(VOL_DAYS, min_periods=20).std()
    return ctx, m, eng, vol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    last = ev.PERIODS["train"][1]                                      # 2023-12-31: nothing later is read
    ctx, m, eng, vol = build_ctx(last)
    cols = list(range(ctx.X.shape[1]))
    preds = {}
    for days in (1, 5):
        ctx.y = ctx.rows.to_long(cb.centred_rank(forward_return(m, eng.p["close"], days), m.universe))
        t = time.time()
        preds[days] = predict(ctx, cols, TUNE_MONTHS, days)
        print(f"trees walked forward through 2022-2023, {days}-day target: {time.time() - t:.0f}s", flush=True)

    rows, daily = [], {}
    winner_checks = []
    for v in [PLAIN] + GRID:
        d = run_version(v, preds, m, vol)
        d2 = run_version(v, preds, m, vol, fee=2 * ev.FEE)
        daily[v.name] = d
        rows.append({"version": v.name, **asdict(v), **cb.summarize(d), "double_fee_sharpe": cb.summarize(d2)["net_sharpe"],
                     "mean_scale": d["scale"].mean()})
    res = pd.DataFrame(rows).set_index("version")
    grid = res.loc[[v.name for v in GRID]]
    best_name = grid["net_sharpe"].idxmax()
    best = next(v for v in GRID if v.name == best_name)
    for v in (Version(best.target_days, best.smooth_days, False, best.vol_target),
              Version(best.target_days, best.smooth_days, best.inverse_vol, False)):
        d = run_version(v, preds, m, vol)
        winner_checks.append({"version": v.name, **asdict(v), **cb.summarize(d),
                              "double_fee_sharpe": cb.summarize(run_version(v, preds, m, vol, fee=2 * ev.FEE))["net_sharpe"],
                              "mean_scale": d["scale"].mean()})
    res = pd.concat([res, pd.DataFrame(winner_checks).set_index("version")])
    res["role"] = ["plain"] + ["grid"] * len(GRID) + ["winner, one rule off"] * len(winner_checks)
    diff_bp, diff_t = cb.versus(daily[best_name], daily[PLAIN.name])
    enters = bool(res.at[best_name, "net_sharpe"] > res.at[PLAIN.name, "net_sharpe"])
    res.to_csv(OUT / "choose_2022_2023.csv")
    pd.concat(daily, names=["version", "date"]).to_parquet(OUT / "daily_2022_2023.parquet")
    choice = {"chosen": asdict(best), "name": best_name, "enters_exam": enters,
              "sharpe_2022_2023": res.at[best_name, "net_sharpe"], "plain_sharpe_2022_2023": res.at[PLAIN.name, "net_sharpe"],
              "vs_plain_bp": diff_bp, "vs_plain_t": diff_t}
    (OUT / "choice.json").write_text(json.dumps(choice, indent=2, default=float), encoding="utf-8")

    show = ["net_sharpe", "net_bp", "gross_bp", "cost_bp", "funding_bp", "turnover", "max_drawdown_pct", "ic", "double_fee_sharpe", "mean_scale"]
    pd.set_option("display.width", 230)
    print("\n2022-2023, walked forward (data cut at 2023-12-31):")
    print(res[show].round(3).to_string())
    print(f"\nwinner: {best_name} | vs plain {diff_bp:+.1f} bp/day (t {diff_t:+.2f}) | enters the exam: {enters}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
