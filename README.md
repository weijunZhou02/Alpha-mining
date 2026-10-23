# Crypto Alpha Demo

A small alpha-mining project for crypto, built step by step:

**data → operator library → evaluator → report → remove duplicates → combine → LLM loop → final exam**

It follows the idea of the Alpha-GPT paper: an LLM proposes alpha formulas, and code computes them, scores them and gives the results back. Steps 0–6 build and test the machinery on 507 published alphas. Step 7 adds the LLM.

> **AI agents:** read [AGENTS.md](AGENTS.md) first. It has the rules that keep the results honest.

## Plan

| Step | What | Status |
| --- | --- | --- |
| 0 | Alpha library (`alpha_library/`): 507 published alphas with crypto-fit notes | done |
| 1 | Data: download Binance USDT perpetuals and build daily panels | done |
| 2 | Operator library + formula reader: computes 492 of 507 catalog alphas | done |
| 3 | Evaluator: Rank IC, ICIR, 5 groups, turnover, fees + funding, money check (`src/evaluate.py`, `src/run_step3.py`) | done |
| 4 | Report page: [Crypto alpha scorecard](https://claude.ai/artifact/XXQ1ThZC3xpi4dXtFYUzmB) (data from `src/build_report_data.py`, page copy in `report/`) | done |
| 5 | Remove duplicates (`src/dedupe.py`): core list 15 alphas in 8 families, broad list 159 | done |
| 6 | Combine (`src/combine.py`): equal weight, IC-weighted, ridge, boosted trees, walked forward through 2024 | done |
| 7 | LLM loop: new alphas judged by what they add to the combination (design not yet agreed) | next |
| 8 | Final exam: 2025-01 to 2026-10, run once, "before LLM" vs "after LLM" ([plan sealed](#step-8-plan-the-sealed-envelope)) | |

## Quick start

### Windows (PowerShell)

```powershell
git clone https://github.com/weijunZhou02/Alpha-mining.git
cd Alpha-mining
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"                       # read and write files as UTF-8 (set it in every new window)
.venv\Scripts\python -m pytest tests -q      # 29 tests should pass
.venv\Scripts\python src\combine.py          # re-runs step 6 in about 3 minutes
```

To keep `PYTHONUTF8` permanently: `setx PYTHONUTF8 1`, then open a new window.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest tests -q
```

Python 3.11 or newer. The commands below use the macOS form (`.venv/bin/python src/x.py`). On Windows, write `.venv\Scripts\python src\x.py`.

### What is in the repository, and what is not

| Included | Not included (rebuild it) |
| --- | --- |
| All code, tests and the alpha library | `data/raw/` (62k files, 244 MB): `src/download_data.py` gets it in about 10 min |
| `data/panel/` cleaned daily tables (71 MB), `data/meta/` | `data/eval/alpha_ranks.parquet` (61 MB): `src/run_step3.py` rebuilds it in about 5 min |
| `data/eval/` scores, duplicate lists, step 6 results and `combine/features.parquet` | `.venv/`, logs |

You only need `data/raw/` to rebuild the panels or add newer days. Everything from step 2 onward runs from `data/panel/`.

## Data: how to run

```bash
.venv/bin/python src/download_data.py    # about 10 min the first time; later runs only fetch new files
.venv/bin/python src/build_panels.py     # about 1 min: raw files -> data/panel/*.parquet
```

Result of the first run (2026-10-22):
- 679 crypto USDT perps were downloaded, 594 of them kept (every coin ever in the top 120).
- 538 coins were in the top-80 universe at some point since 2021 (539 counting the 2020 warm-up).
- The universe has 80 coins a day (70-79 in Jan-Mar 2021, when fewer coins had 60 days of history).
- Data runs from 2020-01-01 to 2026-10-21. Raw data is 78 MB of files (244 MB on disk), and the panels are 71 MB.
- `data/meta/symbols.csv` has Binance's own sector tags (Layer-1, DeFi, AI, Meme…) for sector-neutral tests.
- `data/panel/summary.json` has the health checks.

## Step 2: computing alphas

| File | What |
| --- | --- |
| `src/operators.py` | About 45 building blocks: time-series (`ts_mean`, `ts_rank`, `decay_linear`, GTJA `SMA`/`WMA`, regressions…), cross-sectional (`cs_rank`, `cs_scale`, sector demean), element-wise |
| `src/formula.py` | Reads formula text in 3 styles (101 Alphas, GTJA, Qlib) and computes it |
| `src/alphas.py` | `AlphaEngine`: crypto data mapping, 18 corrected GTJA formulas, 44 hand-written crypto/classic factors |
| `tests/` | Hand-calculated tests: `.venv/bin/python -m pytest tests` |
| `src/check_all_alphas.py` | Computes all alphas once and records coverage → `data/checks/alpha_compute_check.csv` |
| `src/check_vs_vibe.py` | Compares with Vibe-Trading's code → `data/checks/vibe_comparison.csv` |

```python
import sys; sys.path.insert(0, "src")
from alphas import AlphaEngine
eng = AlphaEngine.load()
x = eng.compute("A101_012")   # rows = date, columns = coin, blank outside the daily top 80
```

Rules:
- Time-series tools use each coin's full history.
- Cross-sectional tools compare only today's top 80.
- A window needs at least 2/3 of its days.
- A correlation with a flat series is 0.
- `volume` means USDT volume.
- The benchmark is BTC.
- The industry is Binance's first sector tag.

## Step 3: scoring alphas (training period 2021-2023 only)

```bash
.venv/bin/python src/run_step3.py    # about 5 min -> data/eval/
```

Verdicts in `data/eval/scorecard.csv`:
- `verdict_rank_only`: the rule agreed first. |rank-IC t| ≥ 3 and the same sign in at least 2 of 3 years.
- `verdict`: adds the **money check**. The long-short portfolio (before costs) must also have t ≥ 3 in the same direction for `strong`, or t ≥ 2 for `maybe`. `rank only` means the rank score is strong but the money isn't. Crypto returns are lopsided: wild coins usually fall a little but sometimes jump a lot, so rank scores can mislead.
- `survives_costs`: still profitable (t ≥ 2) after 0.05% fees and funding.

Results: 4 strong, 32 maybe, 341 rank only, 115 no; 3 survive costs.

**Ties are shared fairly.** If several coins have exactly the same score at the edge of a group, they share the places equally (40 coins tied for 16 places get 16/40 of a place each). The first version picked tied coins in alphabetical order. That mattered for 120 alphas that take only a few values (0/1, "which of the last 5 days"…), and for the perp premium, which is exactly 0 for about 20 coins a day. Fixing it changed 13 verdicts, all near the t = 2 border. Rank IC was not affected.

Honesty checks run every time:
- a RANDOM alpha must score about 0
- an ORACLE that peeks at tomorrow must score about 1

The 2025+ test period is locked in code (`Market("test")` refuses without `unlock_test=True`).

## Step 5: removing duplicates (training period only)

```bash
.venv/bin/python src/dedupe.py              # needs data/eval/alpha_ranks.parquet from step 3
.venv/bin/python src/build_report_data.py   # refresh the report data
```

How it works:
- Two alphas are duplicates when they rank coins alike (|correlation| ≥ 0.7, so a flipped twin counts) **or** make money on the same days (≥ 0.7).
- Go from best to worst money t, and keep an alpha only if it isn't a duplicate of one already kept. A twin that is almost as good (≥ 90% of the money t) but trades much less (≤ 70% of the turnover) replaces it.

The lists:
- `core_list.csv`: 15 alphas in 8 families, from 36 strong + maybe. Used for simple combinations. Families are named in `FAMILIES` in `src/dedupe.py`.
- `broad_list.csv`: 159 alphas, from 421 with some signal. Used for learning models.
- `membership.csv`: every alpha's status (core, broad, twin, weak) and its closest kept alpha.

## Step 6: combining alphas (tested on 2024)

```bash
.venv/bin/python src/combine.py             # about 3 min -> data/eval/combine/
.venv/bin/python src/build_report_data.py
```

Rules agreed before running:
- **2024 picks the method, never the alphas.** The core and broad lists come from 2021–2023 only.
- Each alpha becomes its daily rank among the top 80, flipped so higher = better, centred on 0. Missing = 0.
- Models learn tomorrow's return, ranked among the day's coins.
- Walk forward: at the start of each month of 2024, re-learn on all earlier days (training rows end 2 days before). Settings (ridge penalty, tree size, number of trees) are picked inside each training window, using its last 6 months as a mini-test.
- Portfolio as in step 3: long the top 20%, short the bottom 20%, equal weight, daily, 0.05% fees + funding.
- **Pass** = a higher 2024 Sharpe after costs than both benchmarks (equal weight, and the best single alpha by training money t: A101_055) **and** money made after costs.

Methods: equal weight (core), IC-weighted (core, last 12 months), ridge (core and broad), boosted trees (broad; scikit-learn `HistGradientBoostingRegressor`, the same kind of model as LightGBM), plus a "buffer" version of the best one (keep a coin until it leaves the top/bottom 30%).

Results, 2024 (after fees and funding):

| Method | Rank IC | Before costs bp/day | After costs bp/day | Sharpe |
| --- | --- | --- | --- | --- |
| Equal weight · core | 0.033 | +0.5 | −11.7 | −1.03 |
| IC-weighted · core | 0.038 | −6.8 | −18.1 | −1.53 |
| Ridge · core | 0.042 | −9.5 | −19.8 | −1.64 |
| Ridge · broad | 0.092 | −5.4 | −13.6 | −0.97 |
| **Trees · broad** | **0.101** | **+10.9** | **+1.2** | **+0.09** |
| Trees · broad + buffer | 0.101 | +0.8 | −5.3 | −0.38 |
| Single alpha A101_055 | 0.021 | −7.2 | −14.9 | −1.49 |

What it means:
- **The ranking still works in 2024; the money does not.** Every method still ranks coins better than chance (rank IC t from 3 to 11), but before costs the equal-weight blend earned only +0.5 bp/day, against +42 bp/day in 2021–2023.
- The short side (betting against hot coins) still earned +7 bp/day. The long side (quiet, calm, cheap-premium coins) lagged the market by 7 bp/day.
- Costs then decide it: about 2× turnover a day costs ~10 bp/day in fees, plus ~2 bp/day of funding.
- Trees on the broad list is the only method that passes, but its +1.2 bp/day (t = 0.09) cannot be told apart from zero.
- The buffer cut fees in half but lost the edge: the signal fades within days.
- On their own, 12 of the 15 core alphas lost money in 2024. The 2021–2023 numbers were flattering because the alphas were picked on those years. Following the rule, we do **not** re-pick alphas based on 2024.
- Honesty check: the same ridge blend trained on shuffled answers scores about zero in 2024 (Sharpe −0.31, rank t 1.3), so the walk-forward does not leak the future.

Files in `data/eval/combine/`: `results_2024.csv`, `quarters_2024.csv`, `daily_2024.parquet`, `singles.csv`, `weights.csv`, `settings.csv`, `checks.csv`, and `features.parquet`. That last one is a (date, coin) × 159 alphas table of direction-adjusted ranks for 2021–2024, ready for step 7.

## Step 8 plan: the sealed envelope

Agreed on 2026-10-23, **before anyone has looked at 2025–2026**. The git tag `before-llm` marks this exact version of the code, the alpha lists and this plan. To rebuild it anytime: `git checkout before-llm`.

**The exam paper**
- Test period: **2025-01-01 to 2026-10-21**, the data we have today (about 22 months). If newer days are added later, they are reported separately as extra; they don't change the verdict.
- It is opened **once**, in one run that scores every contender below together. Until then, `Market("test")` stays locked and nobody looks at 2025–2026 results, not even for one alpha.

**The contenders, fixed now**

| Group | Contender | Defined by |
| --- | --- | --- |
| Before LLM (main) | **Trees · broad**, the step 6 winner | `src/combine.py` and `data/eval/dedupe/broad_list.csv` at tag `before-llm` |
| Benchmark | Equal weight · core | `core_list.csv` at tag `before-llm` |
| Benchmark | Single alpha A101_055 | the best alpha by 2021–2023 money t |
| After LLM | The final blend from step 7 | Frozen with its own tag (`after-llm`) **before** the exam is opened |

**Rules, the same as 2024**
- Walk forward: each contender re-learns at the start of every test month using only earlier days, with settings picked inside its own training window, exactly as in step 6.
- Long the top 20%, short the bottom 20%, daily. 0.05% fees plus funding. Tied coins share places.

**How the winner is decided, written before seeing anything**
- Main score: Sharpe after fees and funding over the whole test period.
- **"The LLM helped"** only if the after-LLM blend has a higher Sharpe than Trees · broad **and** the day-by-day difference has t ≥ 2. A higher Sharpe with t < 2 is reported as "no clear difference".
- Also reported for every contender: money before and after costs, t, worst drop, turnover, rank IC, each quarter, and each year (2025, 2026).
- Safety check before opening: the exam code must reproduce the 2024 numbers in `data/eval/combine/results_2024.csv` exactly for the "before" contenders. That proves they were not changed.
- If a bug is found after opening, both the original and the fixed result are reported. Nothing is re-run quietly.

## Data folders

```
Binance ──download_data.py──▶ data/raw/ ──build_panels.py──▶ data/panel/  +  data/meta/
           (about 10 min, incremental)    (originals)      (about 1 min)   (research tables)   (coin info)
```

| Folder | What | Shape | Rule |
| --- | --- | --- | --- |
| `data/raw/` | Files exactly as Binance gives them: `klines/` (daily bars), `fundingRate/`, `premiumIndexKlines/`, one zip per coin per month; `rest/` holds October from the live API plus funding backfill | Long: one row per coin per day (or per funding time) | Never edited. Not on GitHub: re-download it. |
| `data/panel/` | Clean research tables, one parquet per field | Wide: rows = date, columns = coin | Rebuilt by `build_panels.py` |
| `data/meta/` | Facts about coins, not daily numbers: `symbols.csv` (status, listing date, sectors), `kept_symbols.csv`, `universe_members.csv` | One row per coin | Rebuilt by the scripts |
| `data/eval/` | Scores (step 3), duplicate lists (step 5), combining results (step 6), report files (step 4) | CSV and parquet | Rebuilt by the step scripts |
| `data/checks/` | Coverage of every alpha, comparison with Vibe-Trading | CSV | Rebuilt by the check scripts |

Cleaning done by `build_panels.py`:
- Fills days the monthly archive skipped, using daily files.
- Blanks 31 glitch cells (Sep/Nov 2023).
- Removes about 42k flat "zombie" bars that Binance prints after a delisting. The last trading day's next-day return stays: it's the return to the settlement price.

## Data spec (step 1)

**Source:** Binance USDⓈ-M perpetual futures, from the free bulk archive at `data.binance.vision` (no API key).
- One venue, so prices are consistent.
- Shorting is easy, so long-short tests are realistic.
- It includes funding rates and open interest.

**Bar:** daily (`1d`, UTC 00:00 to 24:00). Move to 4h or 1h later.

**Period:** 2021-01 to today (2020 is downloaded as warm-up history only).

**Universe, rebuilt every day:**
- Top 80 USDT perps by 30-day average quote volume.
- Drop stablecoins, and any coin with fewer than 60 days of history.
- Keep coins that were later delisted, otherwise the test only sees survivors.

**Tables:**

| Tier | File type | Columns we use |
| --- | --- | --- |
| 1 (start) | `klines/1d` | open, high, low, close, volume, quote_volume, count, taker_buy_volume, taker_buy_quote_volume |
| 2 | `fundingRate` (monthly) | calc_time, funding_interval_hours, last_funding_rate (usually every 8h, summed to daily) |
| 2 | `premiumIndexKlines/1d` | perp premium over the index price (exactly 0 whenever the index sits inside the perp's bid/ask) |
| 2 (not yet) | `metrics` (daily, 5-min rows) | sum_open_interest(_value), top-trader and account long/short ratios, taker long/short volume ratio |
| 3 (later) | CoinGecko | market cap, sector categories |

**Derived fields:**
- `vwap = quote_volume / volume`
- `ret = close / close.shift(1) - 1`
- `taker_buy_ratio = taker_buy_quote_volume / quote_volume`

**Layout:** one parquet file per field, with rows = date and columns = symbol.

```
data/panel/close.parquet, open.parquet, ..., quote_volume.parquet, funding.parquet, premium.parquet
data/panel/universe.parquet   True/False per date and coin (tradable that day?)
data/panel/fwd_ret_1d.parquet answer key: next day's return. ONLY for grading, never an input.
```

## Credits

- 101 Formulaic Alphas (Kakushadze 2016), Guotai Junan 191 alphas (2017 report), Qlib Alpha158 (Microsoft): formulas collected in `alpha_library/` with their sources.
- `third_party/vibe_trading/`: MIT-licensed factor code used only to cross-check our numbers (see its `LICENSE` and `SOURCE.txt`).
