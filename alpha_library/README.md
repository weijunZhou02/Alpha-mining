# Alpha Library

`alpha_catalog.csv` lists 507 published alphas in one table. Open it in Excel or Numbers, or load it with pandas. Each row is one alpha: its formula, a plain-English explanation, where it was published, and whether it can be used on crypto.

## What's inside

| Library | Rows | From | Market | Year |
| --- | --- | --- | --- | --- |
| `Alpha101` | 101 | Kakushadze, *101 Formulaic Alphas* (Wilmott; arXiv:1601.00991) | US equities | 2016 |
| `GTJA191` | 191 | Guotai Junan Securities, 基于短周期价量特征的多因子选股体系 (Quant Topic No. 93) | China A-shares | 2017 |
| `Alpha158` | 158 | Microsoft Qlib, Alpha158 feature set | China A-shares | 2020 |
| `Crypto-academic` | 35 | Crypto papers (Liu-Tsyvinski-Wu, Dobrynskaya, Bianchi et al., Li et al., Fieberg et al., Cong et al., BIS, He et al.) | Crypto | 2021-2025 |
| `Crypto-practitioner` | 9 | Common trader ideas that use free Binance data (no paper checked) | Crypto | - |
| `Classic-equity` | 13 | Famous academic factors (CAPM, Fama-French, momentum, BAB, Amihud, MAX…) | US equities | 1964-2015 |

Where the formulas come from:
- 101: extracted from your own copy, `../../101 Formulaic Alphas.pdf`.
- GTJA 191: extracted from the original 2017 report PDF.
- Alpha158: generated from Qlib's own code (`qlib/contrib/data/loader.py`).

Of the 191 GTJA alphas, 29 are exact copies of a 101 Alpha and 26 more are close variants. The `related_to` column says which.

## Columns

| Column | Meaning |
| --- | --- |
| `id` | Unique id, e.g. `A101_021`, `GTJA_067`, `Q158_KMID`, `CRYPTO_LTW_R3` |
| `library` | Which collection it comes from |
| `category` | Momentum / trend, Short-term reversal, Price-volume, Volume / activity, Volatility / risk, Intraday bar, VWAP deviation, Technical indicator, Mined combo, Size / liquidity, Market-relative (beta), Carry / derivatives, Order flow, Attention / sentiment, Value / fundamental, Price level |
| `formula` | The formula as published |
| `notation` | Which operator language the formula uses: `wq101`, `gtja191`, `qlib`, or `math` |
| `plain_english` | What the formula measures and how to read it |
| `evidence` | For crypto papers: what the paper found |
| `data_needed` | Which data you need to compute it |
| `original_market`, `source`, `venue`, `year` | Where and when it was published |
| `crypto_fit` | `direct`: works on crypto as-is · `adapt`: needs a small change · `weak`: meaning breaks in 24/7 markets · `hard`: needs paid data · `no`: no crypto version |
| `crypto_note` | What to change for crypto |
| `related_to` | GTJA alphas that copy or resemble a 101 Alpha |
| `formula_issues` | Known typos or missing arguments in the original text |

## Rules for using stock alphas on crypto

1. **Map the data.**
   - `volume` → base volume (coins traded).
   - `amount`, dollar volume and `adv{d}` → **quote volume** (USDT traded).
   - `vwap` → quote volume ÷ base volume.
   - `returns` → close-to-close daily return.
2. **The open has no gap.** Crypto trades 24/7, so today's open is almost exactly yesterday's close. Anything built on `open - delay(close, 1)` is noise (3 alphas, marked `weak`). `(close - open) / open` simply becomes today's return.
3. **Make outputs unit-free.** One coin can cost $0.0001 and another $60,000. Divide price-unit outputs by close and volume-unit outputs by average volume, or use ranks (marked `adapt`).
4. **Industry → sector.** For `indneutralize` and `IndClass`, use CoinGecko categories (L1, DeFi, meme, AI…), or demean across all coins. Many alts are mostly "BTC plus noise", so removing BTC beta often matters more than removing the sector.
5. **Benchmark → BTC.** Where an alpha needs a benchmark index, use BTC or a market-cap-weighted crypto index.
6. **Windows are in days.** Stocks trade about 250 days a year and crypto trades 365. A "250-day" window is about 8 months in crypto; use 365 for a full year.
7. **Horizons are faster.** Crypto momentum lives in 1-4 weeks, and 1-month+ lookbacks tend to reverse (`CRYPTO_LTW_R*`, `CRYPTO_DOB_*`).
8. **Fundamentals don't transfer.** Book value, earnings and similar fields don't exist for coins. On-chain data is the nearest thing, and it's mostly paid.

## Not included (pointers)

Big equity anomaly databases. They're mostly accounting-based, so they're less useful for crypto:
- Chen & Zimmermann, *Open Source Asset Pricing*: 200+ signals with code.
- Jensen, Kelly & Pedersen, *Is There a Replication Crisis in Finance?*: 153 factors, global data.
- Hou, Xue & Zhang, *Replicating Anomalies*: 452 anomalies.
- Harvey, Liu & Zhu (2016), the paper in this folder: 316 factors, plus the multiple-testing warning.

Ready-made code for 101 + GTJA 191 + Alpha158 exists in the open-source Vibe-Trading "Alpha Zoo" (HKUDS, MIT license). It's useful later as a cross-check when we write our own operator library.

## Rebuild

```bash
python3 build_catalog.py
```

Source files are in `sources/`:
- the extracted formula JSONs;
- `gtja191_notes.py` (plain English + known typos);
- `crypto_and_classic.py` (crypto and classic factors).

The 101 plain-English text is read from `../../101 Formulaic Alphas in Plain English.md`.
