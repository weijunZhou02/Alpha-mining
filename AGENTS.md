# Guide for AI agents

This file is for any AI coding agent (Claude Code, Codex, Cursor…) working in this repository. Read it fully before changing anything. The human guide is [README.md](README.md).

## Who you are working with

- The owner is learning alpha research while building this project. Write to them in **plain, simple English**: short sentences, no jargon without a one-line explanation. They sometimes mix in Chinese terms.
- **Work step by step, and discuss before building.** For every new step, first send a short table of design choices with your recommendation for each, then wait. The owner replies "all OK" or changes items. Do not build a step whose design has not been approved.
- A bug fix inside an approved step needs no approval, but say clearly what changed and how results moved.
- Report results honestly, including bad ones. Never present an in-sample number as if it were a real test.

## The project in one paragraph

Daily data for Binance USDT perpetual futures, 2020 to today. Each day the universe is the top 80 coins by 30-day average trading value. 507 published alphas (101 Alphas, GTJA 191, Qlib Alpha158, crypto papers) are computed from formulas and scored. Then duplicates are removed, and the survivors are combined into one score per coin per day. Step 7 will put an LLM on top to invent new alphas (as in the Alpha-GPT paper).

## Rules that keep the results honest (do not break these)

1. **Periods are fixed in `src/evaluate.py` → `PERIODS`:**
   - train: 2021-01-01 to 2023-12-31. Every choice about alphas (scores, verdicts, duplicate lists, directions) uses this period only.
   - valid: 2024. Used **once**, in step 6, to choose a **combining method**. It must not be used to choose or drop alphas: that would spoil it as a test.
   - test: 2025-01-01 to 2026-10-21. **Locked.** `Market("test")` raises unless `unlock_test=True`. Only step 8 (the final exam) may unlock it, once, after the owner says so. Never unlock it to "just have a look", and never show the owner or the LLM any 2025–2026 result before then.
   - **The step 8 plan is sealed** (README → "Step 8 plan: the sealed envelope", git tag `before-llm`). It fixes the contenders and how the winner is decided. Do not change it. The "before LLM" contenders must keep giving exactly the 2024 numbers in `data/eval/combine/results_2024.csv`. If step 7 needs changes to `combine.py` or `evaluate.py`, keep the old behaviour available for the "before" contenders, or run them from the tag (`git worktree add ../before before-llm`).
2. **No look-ahead.**
   - `data/panel/fwd_ret_1d.parquet` is the answer key (next day's return). It may only be used for grading, never as an input to an alpha.
   - Time-series operators see each coin's full past history.
   - Cross-sectional operators compare only the coins in that day's universe.
   - Walk-forward training rows end 2 days before the day being traded.
3. **The honesty checks must keep passing.**
   - Step 3: a RANDOM alpha must score about 0, and an ORACLE (tomorrow's return) about 1.
   - Step 6: a model trained on shuffled answers must score about 0.
   If one fails, the code is wrong; do not explain it away.
4. **Money check.** In crypto, a good rank IC does not mean the alpha makes money: returns are very skewed, and wild coins usually fall a little but sometimes jump a lot. A "strong" alpha needs |rank-IC t| ≥ 3, the same sign in at least 2 of the 3 training years, **and** long-short t ≥ 3. Always report money after fees (0.05% per unit traded) and funding.
5. **Ties are shared, never broken by coin name.** Use `evaluate.group_weights` for any top/bottom group. Coins tied at a group's edge share the places equally.
6. **Few, pre-declared tries.** When comparing methods on 2024, declare the list first (about 6) and report all of them. Trying many until one wins is overfitting.
7. **Do not commit** `data/raw/`, `.venv/`, logs, or `data/eval/alpha_ranks.parquet` (see `.gitignore`).

## Setup

- Python 3.11+. Create `.venv` and `pip install -r requirements.txt`.
- **Windows:** set `PYTHONUTF8=1` (PowerShell: `$env:PYTHONUTF8 = "1"`; permanent: `setx PYTHONUTF8 1`). Run scripts as `.venv\Scripts\python src\<script>.py`. Paths use `pathlib`, so they work on both systems.
- Tests: `python -m pytest tests -q` (29 tests, hand-calculated). Run them after any change to `operators.py`, `formula.py` or `evaluate.py`.
- Scripts import each other by plain module name (`import evaluate as ev`). Run them from anywhere as `python src/<script>.py`, or add `src` to `sys.path` in notebooks.
- `data/raw/` is not in git. You only need it to rebuild panels (`src/download_data.py` then `src/build_panels.py`, about 11 min). Everything else runs from `data/panel/`.

## Pipeline and files

| Step | Run | Writes | Time |
| --- | --- | --- | --- |
| 1 data | `src/download_data.py`, `src/build_panels.py` | `data/raw/`, `data/panel/*.parquet`, `data/meta/` | ~11 min |
| 2 check alphas | `src/check_all_alphas.py`, `src/check_vs_vibe.py` | `data/checks/` | ~2 min |
| 3 score | `src/run_step3.py` | `data/eval/scorecard.csv`, `daily_ic.parquet`, `daily_ls_net.parquet`, `alpha_ranks.parquet` | ~5 min |
| 5 dedupe | `src/dedupe.py` | `data/eval/dedupe/` (`core_list.csv`, `broad_list.csv`, `membership.csv`, similarity matrices) | ~1 min |
| 6 combine | `src/combine.py` | `data/eval/combine/` | ~3 min |
| 4 report | `src/build_report_data.py` | `data/eval/report/*.csv` (files behind the report page) | ~30 s |

Rebuild order after changing the evaluator: `run_step3.py` → `dedupe.py` → `combine.py` → `build_report_data.py`.

| File | What is in it |
| --- | --- |
| `src/config.py` | Paths, universe settings (`TOP_N=80`, `RANK_WINDOW=30`, `MIN_HISTORY=60`), excluded stablecoins |
| `src/binance_files.py`, `src/download_data.py`, `src/build_panels.py` | Download and clean the data |
| `src/operators.py` | ~45 operators. A window needs ≥ 2/3 of its days (`MIN_FRAC`); window lengths are floored; a correlation with a flat series is 0 |
| `src/formula.py` | Tokenizer, parser and evaluator for 3 formula styles: `wq101`, `gtja191`, `qlib` |
| `src/alphas.py` | `AlphaEngine`: maps crypto data to formula variables, `GTJA_FIXES` (typos in the GTJA report), `CUSTOM` (hand-written alphas), `NOT_COMPUTABLE` |
| `src/evaluate.py` | `PERIODS`, `Market(period)`, `score(alpha, market)` → score card, `group_weights`, `row_spearman`, `nw_tstat` |
| `src/run_step3.py` | Scores every alpha on train, plus the RANDOM and ORACLE checks |
| `src/dedupe.py` | Duplicate removal; `FAMILIES` names the core alphas' families |
| `src/combine.py` | Walk-forward blends, portfolio (`holdings`, `judge`), 2024 results |
| `src/build_report_data.py` | CSV files for the report page |
| `alpha_library/` | `alpha_catalog.csv` (507 alphas: formula, notation, plain English, category, source, year, crypto fit); `build_catalog.py` rebuilds it from `sources/` |
| `report/scorecard_page.html` | Copy of the report page's HTML (the live page is a claude.ai dashboard) |
| `third_party/vibe_trading/` | MIT-licensed reference code, used only by `check_vs_vibe.py` |

## Key APIs

```python
import sys; sys.path.insert(0, "src")
import evaluate as ev
from alphas import AlphaEngine

eng = AlphaEngine.load()                          # loads data/panel
x = eng.compute("A101_055")                       # catalog alpha: DataFrame date x coin, NaN outside the universe
y = eng.run("rank(-delta(close, 3))", "wq101").where(eng.universe)   # any new formula text
m = ev.Market("train")                            # "train", "valid", "train+valid"; "test" is locked
card, series = ev.score(y.loc[m.start:m.end], m)  # card: ic, ic_t, ls_t, net_bp, net_sharpe, turnover, verdict…
```

- Variables available to formulas:
  - `wq101`: open, high, low, close, vwap, volume, returns, advN (N-day average volume).
  - `gtja191`: also amount, ret, the benchmark (BTC) and the GTJA helpers.
  - `qlib`: `$open` … `$volume`.
  - `volume` means USDT volume (`quote_volume`).
- Crypto-only panels (`funding`, `premium`, `trades`, `taker_buy_quote_volume`) are loaded in `eng.p` and used by `CUSTOM` alphas, and step 7's formulas can use them: `llm_loop.make_context` adds `funding`, `premium`, `trades`, `taker_buy_ratio`, `mkt_returns` and `btc_returns` to the `wq101` style.
- `data/eval/combine/features.parquet`: index (date, coin) covering every universe coin-day from 2021 to 2024, with one column per broad-list alpha (159). Each value is the daily rank, centred on 0 and flipped so higher = better. Missing = 0. It is the input table for the blends.
- Directions (+1/−1) come from the training rank IC: `data/eval/scorecard.csv` → `direction`.

## Where things stand (2026-10-22)

- Step 3: 4 strong, 32 maybe, 341 rank only, 115 no (out of 492). 3 survive costs.
- Step 5: core list of 15 alphas in 8 families (price-volume divergence, perp premium, fade recent pushes, volume cooling, size and liquidity, uneven volume, big recent movers, sector price-volume link). Broad list: 159.
- Step 6 (2024 walk-forward):
  - Every method still ranks coins (rank IC t 3 to 11), but money collapsed.
  - The equal-weight blend made +42 bp/day before costs in 2021–2023 (in-sample) but only +0.5 bp/day in 2024. After costs: Sharpe −1.03.
  - The best single alpha (A101_055): Sharpe −1.49.
  - Boosted trees on the broad list is the only pass: +10.9 bp/day before costs, +1.2 after, Sharpe +0.09. That is not distinguishable from zero.
  - In 2024 the long side (quiet, calm coins) lagged by 7 bp/day while the short side still earned. Fees (~2× turnover a day) cost ~10 bp/day.
- Platform notes:
  - LightGBM needs `libomp`, which was not available on the Mac, so `HistGradientBoostingRegressor` from scikit-learn is used. On Windows, `pip install lightgbm` works out of the box if you want the real thing; it is not required.
  - pandas 3 is used (copy-on-write; `DataFrame.stack()` keeps NaN).

- 2026-10-23: the owner chose the "sealed envelope" for step 8. "Before LLM" is frozen at tag `before-llm`. The step 7 result gets its own tag (`after-llm`) before the exam is opened, and both are scored together once.

## Step 7, the LLM loop (built; first real run done)

- Code: `src/llm_loop.py`, with tests in `tests/test_llm_loop.py`. The README section "Step 7" has the gates and the results.
- The owner chose Claude through the API, run on the Mac. The Mac is the official machine. Results from the owner's Windows PC (local Qwen, see `step7_llm_runs_summary.md` on GitHub) are set aside: their steps 3-6 differed slightly because of floating-point rounding.
- The API key lives only in `.env` (ANTHROPIC_API_KEY, LLM_MODEL). Never print, log or commit it.
- Do not loosen the gates after seeing results. They were calibrated on luck items **before** the paid run. Any change to them is a design change that the owner approves first.
- Every run looks at 2023 again. Declare runs and rounds before starting, and report all tries.
- Kept so far: LLM_001 and LLM_002 (`data/llm/kept.csv`).
- Open decisions for the owner:
  1. More runs, for example with new data (Binance `metrics`: open interest, long/short ratios). Do this before the 2024 check, not after.
  2. The one-time 2024 check: Trees · broad + kept against Trees · broad.
  3. Freezing the `after-llm` tag for step 8.
