# Step 7 LLM loop: summary of runs 1–4 (2026-10-23)

This summary is written for a planning agent. It records how each run worked, what changed between runs and why, what came out, and what limits the results, so that a cloud-API plan can be designed from it.

Read [AGENTS.md](../AGENTS.md) first: it has the project rules, and they bind any new plan too.

The raw record:
- `data/llm/tries.jsonl` has one line per try.
- `data/llm/runs.csv`, `data/llm/luck_check.csv` and `data/llm/accepted.csv` hold the per-run, luck-check and accepted-alpha records.
- `logs/llm/round_*.json` holds the raw prompts and replies (not in git).

## 1. Bottom line

| | |
| --- | --- |
| Runs | 4: one smoke test and three full runs of 40 rounds each |
| LLM | local Ollama `qwen2.5-coder:7b` on an RTX 2060 (6 GB), temperature 0.8, then 1.0 |
| Formulas tried | 634, all scored on 2021–2023 only |
| Accepted | 1 (`LLM_001`, run 4), cleared the bar by +0.0002 |
| Luck check (noise alphas) | 0 of 80 passed |
| 2024 (validation year) | **not used by step 7** (used once in step 6, to pick the combining method) |
| 2025+ (test) | locked, never touched |
| Verdict | The machinery works and is honest. A 7B local model found almost nothing that adds to the 159-alpha blend. The one accepted alpha is weak: its money comes mostly from 2021, and it lowers the money of the final trees blend even on 2023. |

## 2. Shared mechanism (all runs)

The code is in `src/llm_loop.py`. One run is up to 40 rounds, and each round works like this:

1. **Prompt.**
   - The system prompt gives:
     - the formula syntax (`wq101` style) and the fields: `open high low close vwap volume returns advN funding premium taker_buy_ratio trades`;
     - the functions, with windows that must be plain integers from 1 to 250;
     - one simplified example per core family (8 families);
     - how formulas are judged;
     - a note that the sign does not matter (negative alphas are flipped);
     - the JSON reply format `{"alphas": [{"idea", "formula"}]}`.
   - The user message gives the recent tries with their 2021–2023 scores, the accepted alphas, the current blend score on 2023, and a request for 5 new formulas.
   - **No 2024 or 2025+ number is ever shown.**
2. **LLM call:** Ollama `/api/chat` with a JSON-schema `format`, `num_ctx` 8192.
3. **Safety checks** (`check_formula`):
   - the text is parsed by `formula.py` and never run as Python;
   - at most 300 characters and 60 parse nodes;
   - only the listed fields and functions, with the right argument counts;
   - windows must be literal numbers from 1 to 250;
   - `IndClass.*` is allowed only inside `indneutralize`;
   - a formula that was already tried is refused.
4. **Compute** in a helper process, with a 60 s time limit, plus a **cut-off test** (`peeks_ahead`). The values on 2022-03-15 and 2023-09-20 must not change when the data is cut at those dates; this catches any look-ahead.
5. **Fix pass:** formulas refused for syntax or compute errors go back to the LLM once, with the error message.
6. **Gates**, all on 2021–2023 only (`Lab.test`):
   - **(a)** It covers at least 80% of universe coin-days and differs across coins on at least 80% of days.
   - **(b)** It is not a twin. The signal similarity (pooled |corr| of daily ranks) and the money similarity (corr of daily long-short net) must both be below 0.7 against the broad list and the accepted alphas.
   - **(c)** It adds to the blend.
     - A ridge regression on broad + accepted + new is learned on 2021-01 to 2022-12-30 (a 2-day gap) and scored on 2023.
     - The scoring uses step 6's portfolio: long the top 20%, short the bottom 20%, 0.05% fees plus funding.
     - It is **kept if the rank IC rises by at least +0.002 and money after costs does not fall.**
     - An accepted alpha joins the blend, so later candidates must add to it.
7. **Scores also logged:** the standalone step-3 score card on 2021–2023 (rank IC t, long-short t, net bp, turnover, verdict).
8. **Luck check:** at the start of every run, 20 pure-noise alphas go through gates (a)–(c). Almost none should pass.
9. **Logging and resume:** each try is appended to `tries.jsonl`, and a new run continues the round and try numbers and rebuilds the blend with the accepted alphas.
10. **Final check** (`src/llm_final.py`): trees on broad + accepted, walked forward monthly through 2024, next to trees on broad alone. **It runs once only.** `--dry-run` runs the same code on 2023.

## 3. Settings per run

| Setting | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Purpose | smoke test | first full run (v1 design) | fix a noisy gate, fix low variety | stop the LLM copying its best tries |
| Rounds × formulas | 1 × 5 | 40 × 5 | 40 × 5 | 40 × 5 |
| Temperature | 0.8 | 0.8 | **1.0** | 1.0 |
| History shown | last 30 | last 30 | **15 latest + 5 best** | **15 latest only** |
| Theme per round | no | no | **yes, 14 rotating** | yes |
| Ridge penalty in gate (c) | re-picked per candidate | re-picked | **fixed at the base blend's pick** | fixed |
| `ts_corr` rounded to 10 decimals | no | no | **yes** (steps 3–6 rerun) | yes |
| Broad list size | 158 | 158 | 159 | 159 |
| Base blend on 2023 (rank IC, net bp/day) | 0.1135, +10.4 | 0.1135, +10.4 | 0.1103, +9.4 | 0.1103, +9.4 |
| Fix pass pairs fixes correctly | no (a bug) | yes | yes | yes |
| "Sign does not matter" note | no | yes | yes | yes |

The 14 rotating themes are the 8 core families ("a new twist on…"), interleaved with funding, premium, price shape, volatility, `taker_buy_ratio`, trades and "mix two families".

## 4. Results per run

| | Run 1 | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- | --- |
| Tries | 6 | 201 | 210 | 217 |
| Accepted | 0 | 0 | 0 | **1** |
| Refused: repeat / other errors | 1 / 1 | 39 / 3 | 10 / 28 | 2 / 34 |
| Failed coverage | 0 | 0 | 3 | 8 |
| Twins | 1 | 34 | 14 | 49 |
| Reached gate (c) | 3 | 125 | 155 | 124 |
| Gate (c) gain: median / max | −0.0007 / −0.0002 | −0.0022 / +0.0012 | +0.0001 / +0.0019 | 0.0000 / **+0.0022** |
| Near misses (gain of +0.0015 or more) | 0 | 0 | 5 | 2 |
| Standalone verdict "strong" (step 3 rule) | 0 | 10 | 4 | 23 |
| Luck check: passed / median gain | 0 of 20 / −0.0020 | 0 of 20 / −0.0023 | 0 of 20 / 0.0000 | 0 of 20 / −0.0000 |
| Fix attempts (passed safety) | 1 (0) | 1 (1) | 14 (5) | 17 (7) |
| LLM calls / prompt tokens / reply tokens | 2 / 2.3k / 0.3k | 43 / 124k / 9k | 52 / 153k / 14k | 53 / 128k / 13k |
| Minutes, total (LLM share) | 1.4 (0.5) | 18.6 (11.2) | 22.9 (14.0) | 23.4 (12.8) |

Gate (c) gains in runs 1–2 cannot be compared with runs 3–4. In runs 1–2 the penalty was re-picked for each candidate, which by itself pushed gains down by about 0.002 (see run 2), and the base blend differs (158 against 159 alphas).

Ingredient share (fraction of formulas that contain the word):

| | Run 2 | Run 3 | Run 4 |
| --- | --- | --- | --- |
| `indneutralize` | 55% | 98% | 94% |
| `funding` | 73% | 43% | 19% |
| `volume` | 75% | 68% | 63% |
| `decay_linear` | 51% | 65% | 50% |
| `premium` | 5% | 9% | 11% |
| `taker_buy_ratio` | 22% | 21% | 10% |
| `trades` | 5% | 16% | 12% |
| `close` / `high` / `low` | 5% / 0% / 0% | 37% / 23% / 17% | 32% / 14% / 14% |
| `correlation` | 21% | 11% | 7% |
| `ts_rank` | 0% | 2% | 0% |

### Run 1: smoke test (1 round)

- **Logic:** prove the whole chain works end to end on this PC.
- **Result:** 6 tries, 0 accepted.
  - 3 reached gate (c), 1 was a twin of CLASSIC_ILLIQ (0.90), and 1 used an unknown function (`skew`).
  - The luck check passed 0 of 20.
  - Raw replies showed that Qwen answers a fix request by resending the whole list with only the broken formula changed. The code took the first item (a repeat) and dropped the real fix.
- **Changes before run 2:**
  - `pair_fixes` now skips formulas already tried.
  - The fix prompt asks for the fixed formulas only.
  - The system prompt says the sign does not matter.

### Run 2: first full run (v1 design)

- **Logic:** the approved design as built.
- **Result:** 201 tries, 0 accepted. The best try was `-1 * decay_linear((funding * taker_buy_ratio) * volume, 20)`, at +0.0012.
- **Finding 1, the LLM narrowed.** In rounds 1–10 against rounds 31–40:
  - `indneutralize` went from 22% to 100%;
  - `funding` went from 57% to 78%;
  - the price fields (`close`, `high`, `vwap`) went from about 20% to 0%.
  - 39 exact repeats.
- **Finding 2, gate (c) was noisy.**
  - Re-picking the ridge penalty (from 1e-4 to 10, on the last 6 months of 2022) for each candidate made the pick jump between 0.01, 0.001 and 0.0001.
  - For pure noise, that alone moved the blend IC by −0.002 to −0.003; with the penalty held fixed, noise moves it by at most 0.0007.
  - The best LLM tries were not affected (their penalty did not jump), but other candidates could be rejected unfairly.
- **Finding 3, results depended on the machine.**
  - A 2-day correlation should be exactly ±1, but about 9,600 values carried rounding noise (0.9999999999999998), so `rank()` split true ties.
  - 69 alphas differed from the GitHub numbers, and the broad list came out 158 instead of 159.
- **Changes before run 3 (owner approved):**
  - Hold the penalty fixed at the base blend's pick, and re-pick it only after an accept.
  - A theme per round.
  - History of 15 latest + 5 best.
  - Temperature 1.0.
  - Round `ts_corr` to 10 decimals and rerun steps 3–6:
    - the verdict counts and the core list stayed the same;
    - the broad list is 159, with A101_068 in place of GTJA_119 against GitHub;
    - trees · broad in 2024 came out at −2.0 bp/day, Sharpe −0.15.

### Run 3: themes and a fixed gate

- **Logic:**
  - Remove the measurement noise, so that gains reflect only the new column.
  - Force variety with a different theme each round.
  - Show the best tries, so the LLM can build on what worked.
- **Result:** 210 tries, 0 accepted.
  - 5 near misses, all one idea: premium or funding × volume, smoothed and scaled by yesterday's volume, sector-neutral. The best was `-1 * indneutralize(decay_linear(premium * volume, 20) / delay(volume, 1), IndClass.sector)`, at +0.0019 and +3.8 bp/day.
  - The luck check median is now 0.0000, so the fix worked.
  - The price fields came back, but **98% of formulas used `indneutralize`**: the LLM copied the wrapper from its "best tries".
  - Repeats fell from 39 to 10, but syntax errors rose to 28, probably from the higher temperature.
- **Change before run 4 (owner approved):** hide the best tries (`HISTORY_BEST = 0`).

### Run 4: best tries hidden

- **Logic:** stop the LLM copying its own best formulas, so that it explores more.
- **Result:** 217 tries, **1 accepted**.
  - `funding` fell to 19%, but `indneutralize` stayed at 94%. The wrapper was also in the latest tries, so the LLM still saw it.
  - Twins rose to 49: the LLM rediscovered known alphas (GTJA_095 ×8, GTJA_030 ×7, Q158_VMA60 ×6, CRYPTO_HMRW_PREMIUM ×4).
  - 23 tries were "strong" on their own, yet only 1 added to the blend.
- **Accepted:** `LLM_001 = -1 * indneutralize(decay_linear(abs(funding) * taker_buy_ratio, 3), IndClass.sector)`, gain +0.0022, net +0.6 bp/day.

## 5. The accepted alpha, LLM_001 (all numbers on 2021–2023)

- **Alone it looks strong.**
  - Rank IC 0.040, t 10.1, with the same sign every year (0.046, 0.031, 0.042).
  - Long-short t 5.2, higher than any of the 492 catalog alphas.
  - +21 bp/day after fees and funding, turnover 1.19 a day.
  - It still works traded one day late (IC 0.031).
- **The money is mostly 2021:** +64.6 bp/day in 2021, −5.4 in 2022, +3.6 in 2023.
- **The idea is really `abs(funding)`:** short coins with extreme funding of either sign.
  - `-1 * decay_linear(abs(funding), 3)` alone scores IC t 11.2 and +22.9 bp/day.
  - The `taker_buy_ratio` part adds nothing; on its own it loses −17.9 bp/day.
- **The trees blend earns less with it, even on 2023.** This is the `llm_final.py --dry-run`, which walks trees month by month through 2023 and is flattering, because 2023 picked the alpha:

| 2023 | Rank IC | Net bp/day | Sharpe | Funding bp/day |
| --- | --- | --- | --- | --- |
| Trees · broad | 0.124 | +15.0 | 1.21 | −7.9 |
| Trees · broad + LLM_001 | 0.126 | +10.8 | 0.84 | −9.0 |

The difference is −4.2 bp/day, t −1.0. Gate (c) used ridge, where LLM_001 helped by a hair; the trees model is the step 6 method, and it does better without it.

## 6. What limits the results

**Model-limited (a stronger LLM should help):**
- Repetition: 52 exact repeats over the runs.
- Convergence on one wrapper (`indneutralize` in 94–98% of formulas) and one idea family (funding or premium × volume).
- Syntax errors: 28–34 per run once the temperature was 1.0.
- Rediscovering catalog alphas: 49 twins in run 4.
- Only 8 example formulas are in the prompt. The LLM does not see the 159 broad-list alphas, so it cannot avoid them on purpose.

**Not model-limited (a stronger LLM will not fix these):**
- **A crowded blend.** 159 deduplicated alphas from 492 published ones already cover most simple daily price, volume, funding and premium ideas. Many LLM formulas were strong alone (verdict "strong": 10, 4 and 23 per run) but added nothing.
- **Thin data.** Daily bars for about 80 coins, with 11 fields.
- **The step 6 problem is money after costs in 2024, not ranking.** The blend's rank IC holds in 2024 (t from 3 to 11), but fees (about 2× turnover a day, about 10 bp/day) and a market change wipe out the profit. Better ranking on 2021–2023 does not address that.
- **Overuse of the selection data.** 634 tries plus 80 noise alphas have been scored on 2023, so it is partly fitted. Any further search raises the chance of lucky passes, as with LLM_001.

## 7. Facts for designing a cloud-API plan

**Where the code changes**
- The only LLM call is `ask(messages, model, host)` in `src/llm_loop.py`.
  - It POSTs to `{host}/api/chat` (Ollama) with `format = SCHEMA`.
  - It returns `(content, {"prompt_tokens", "reply_tokens", "seconds"})`.
  - A cloud client must keep that return contract.
- `main()` checks `{host}/api/tags` for the model at start; this needs a provider switch.
- `parse_reply` already tolerates code fences and extra text.
- Command-line flags: `--model`, `--host`, `--rounds`, `--per-round`, `--luck-only`, `--no-luck`.
- `runs.csv` and every try record the model name, so runs from different models can be compared.

**Size of a run (measured)**
- About 2.4–2.9k prompt tokens and 210–265 reply tokens per call.
- About 43–53 calls per 40-round run.
- So about 125–155k prompt tokens and 9–14k reply tokens per run.
- The system prompt is about 1.3k tokens.
- A bigger context, for example the full broad list as twin-avoidance info, would add tokens.

**Time**
- Scoring takes about 2.5–3.5 s per formula on this PC (compute, cut-off test, score card, twins, ridge), so about 10 minutes per 200 tries.
- The local LLM took 11–14 minutes per run. With a fast cloud model, scoring becomes the main cost.

**Keys**
- The owner sets the API key themselves, in an environment variable or in `.env` (already in `.gitignore`).
- The key must never be written to logs. `logs/` is git-ignored, but raw prompts and replies go there.

**The resume trap**
- A new run resumes from `tries.jsonl` and `accepted.csv`: the round and try numbers continue, and `LLM_001` is in the blend.
- A plan that wants a clean comparison must decide whether the cloud model starts from the blend with or without `LLM_001`. Either way, all earlier tries stay in the record, because every look at 2023 counts.

## 8. Rules any plan must keep (from AGENTS.md)

1. The periods are fixed.
   - All alpha choices use train 2021–2023.
   - 2024 is used for one final check of the finished blend (`llm_final.py`, once), never to pick or drop alphas.
   - 2025+ is locked.
2. The LLM never sees a 2024 or 2025+ number.
3. Formulas go through `formula.py` only, never Python `eval`. The safety checks and the cut-off test stay.
4. The luck check runs every run and must stay at about 0 passes.
5. The bar (+0.002 rank IC, net not falling) is not lowered after seeing results. Any change to the gates is a design change.
6. Declare the tries before running: state the number of runs and rounds up front, and report all tries, including failed ones.
7. Before building, the owner approves a short design table ("all OK"). Write to the owner in plain, simple English.

## 9. Open questions for the cloud-API plan

1. **Provider and model**, and the budget per run and in total. This is the owner's choice.
2. **A fair comparison with Qwen:** one pre-declared run with the same rules, bar and 40 rounds, compared on tries, accepts, near misses, repeats, twins and ingredient variety.
3. **What a big model should see:** the full broad list (159 formulas) or family summaries, so it can avoid twins on purpose. It might also see why tries were twins (the twin's name is already logged).
4. **Whether to keep the gate as is:** ridge as a fast proxy, while the final method is trees. LLM_001 passed ridge and hurt trees. Options include a trees check of the few accepted alphas inside 2021–2023 (for example, inner months of 2023), or a stricter bar.
5. **How to limit further fitting of 2023:** for example, select on 2023 H1 and confirm on 2023 H2 before accepting, or a fixed total of tries.
6. **Whether the objective should change first**, which the 2024 results suggest: slower signals or a 3–7 day target to cut fees, or new data (Binance `metrics`: open interest and long/short ratios), which gives the LLM information the current alphas lack.

## 10. Files

| File | What |
| --- | --- |
| `src/llm_loop.py` | the loop: prompt, `THEMES`, Ollama client, safety checks, gates, luck check, logging and resume |
| `src/llm_final.py` | the one-time 2024 check (`--dry-run` = on 2023) |
| `tests/test_llm_loop.py` | 24 tests: safety rules, cut-off test, fix pairing, themes, prompt history |
| `data/llm/tries.jsonl` | all 634 tries: formula, idea, stage, reason, all scores, model, theme, run, round |
| `data/llm/accepted.csv` | LLM_001 |
| `data/llm/runs.csv` | one row per run |
| `data/llm/luck_check.csv` | 80 noise alphas (20 per run) |
| `data/llm/final_dryrun_2023.csv` | the dry-run table in section 5 |
| `logs/llm/round_*.json` | raw prompts, replies and token counts (not in git) |
