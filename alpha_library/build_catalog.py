"""Build alpha_catalog.csv from the source files in ./sources.

Run:  python3 build_catalog.py
"""
import csv
import difflib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
NOTES_101_MD = SRC / "alpha101_plain_english.md"   # copy of "101 Formulaic Alphas in Plain English.md"
sys.path.insert(0, str(SRC))

from gtja191_notes import NOTES as GTJA_NOTES, ISSUES as GTJA_ISSUES, PRICE_UNIT, VOLUME_UNIT, BENCHMARK, GAP  # noqa: E402
from crypto_and_classic import CRYPTO_ACADEMIC, CRYPTO_PRACTITIONER, CLASSIC_EQUITY  # noqa: E402

COLUMNS = ["id", "library", "name", "category", "formula", "notation", "plain_english", "evidence",
           "data_needed", "original_market", "source", "venue", "year", "crypto_fit", "crypto_note",
           "related_to", "formula_issues"]

TYPE_MAP_101 = {"Trend & momentum": "Momentum / trend", "Volume surprise": "Volume / activity",
                "Industry-neutral combo": "Mined combo"}

PRICE_UNIT_NOTE = "Output is in price units: divide by close (or use % changes) before comparing coins with very different prices."
VOLUME_UNIT_NOTE = "Output is in raw volume units: use quote volume (USDT) or divide by average volume before comparing coins."
GAP_NOTE = "Uses open vs yesterday's close. Crypto trades 24/7, so today's open is almost exactly yesterday's close and this gap is ~0 noise."


def load_101_notes():
    """Read Type + Plain English per alpha from the 101 notes doc."""
    notes = {}
    for line in NOTES_101_MD.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| \d+ \|", line):
            continue
        cells = [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line)[1:-1]]
        notes[int(cells[0])] = (TYPE_MAP_101.get(cells[1], cells[1]), cells[2])
    assert len(notes) == 101, len(notes)
    return notes


def rows_alpha101(formulas, notes):
    rows = []
    gap_re = re.compile(r"open\s*-\s*delay\(close,\s*1\)", re.I)
    price_unit = {9, 12, 23, 24, 41, 46, 49, 51}
    for n in range(1, 102):
        f = formulas[n]
        cat, text = notes[n]
        data = ["OHLCV", "vwap"]
        fit, crypto_notes = "direct", []
        if "IndClass" in f:
            data.append("sector map")
            fit = "adapt"
            crypto_notes.append("Needs a sector map (indneutralize): use CoinGecko categories, or skip and demean across all coins.")
        if re.search(r"\bcap\b", f):
            data.append("market cap")
            fit = "adapt"
            crypto_notes.append("Needs market cap: get it from CoinGecko.")
        if re.search(r"adv\d+", f):
            crypto_notes.append("adv{d} = average daily dollar volume: use the average quote volume (USDT).")
        if n in price_unit:
            fit = "adapt" if fit == "direct" else fit
            crypto_notes.append(PRICE_UNIT_NOTE)
        if gap_re.search(f):
            fit = "weak"
            crypto_notes.append(GAP_NOTE)
        rows.append(dict(
            id=f"A101_{n:03d}", library="Alpha101", name=f"Alpha#{n}", category=cat, formula=f,
            notation="wq101", plain_english=text, evidence="", data_needed=" + ".join(data),
            original_market="US equities", source="Kakushadze - 101 Formulaic Alphas",
            venue="Wilmott Magazine; arXiv:1601.00991", year=2016, crypto_fit=fit,
            crypto_note=" ".join(crypto_notes), related_to="", formula_issues=""))
    return rows


def _norm_gtja(s):
    s = s.upper()
    s = re.sub(r"MEAN\(VOLUME,\s*(\d+)\)", r"ADV\1", s)
    for k, v in {"CORR(": "CORRELATION(", "STD(": "STDDEV(", "COVIANCE(": "COVARIANCE(",
                 "TSRANK(": "TS_RANK(", "DECAYLINEAR(": "DECAY_LINEAR(", "TSMIN(": "TS_MIN(",
                 "TSMAX(": "TS_MAX(", "RET": "RETURNS", "PROD(": "PRODUCT("}.items():
        s = s.replace(k, v)
    s = s.replace("RETURNSURNS", "RETURNS")
    return re.sub(r"[\s()]", "", s)


def match_101(gtja_formulas, a101_formulas):
    """For each GTJA alpha, find the most similar 101 alpha (ratio >= 0.8)."""
    a_norm = {n: re.sub(r"[\s()]", "", f.upper()) for n, f in a101_formulas.items()}
    out = {}
    for g, f in gtja_formulas.items():
        gn = _norm_gtja(f)
        best_n, best_r = max(((n, difflib.SequenceMatcher(None, gn, an).ratio()) for n, an in a_norm.items()),
                             key=lambda x: x[1])
        if best_r >= 0.8:
            out[g] = (best_n, best_r)
    return out


def rows_gtja191(formulas, a101_formulas, notes_101):
    matches = match_101(formulas, a101_formulas)
    rows = []
    for n in range(1, 192):
        f = formulas[n]
        cat, text = GTJA_NOTES[n]
        related = ""
        if n in matches:
            m, r = matches[n]
            related = f"Same formula as Alpha101 #{m}" if r == 1.0 else f"Similar to Alpha101 #{m} (similarity {r:.2f})"
        if text is None:
            m, r = matches[n]
            assert r == 1.0, (n, m, r)
            cat = notes_101[m][0]
            text = f"Same formula as 101 Alpha #{m}. " + notes_101[m][1]
        data = ["OHLCV", "vwap"]
        if "AMOUNT" in f:
            data.append("amount (quote volume)")
        fit, crypto_notes = "direct", []
        if n in BENCHMARK:
            fit = "adapt"
            if n == 30:
                data.append("Fama-French factors")
                crypto_notes.append("Replace Fama-French factors with crypto market / size / momentum factors (Liu-Tsyvinski-Wu).")
            else:
                data.append("benchmark index")
                crypto_notes.append("Use BTC or a crypto market index as the benchmark.")
        if n in PRICE_UNIT:
            fit = "adapt" if fit == "direct" else fit
            crypto_notes.append(PRICE_UNIT_NOTE)
        if n in VOLUME_UNIT:
            fit = "adapt" if fit == "direct" else fit
            crypto_notes.append(VOLUME_UNIT_NOTE)
        if n in GAP:
            fit = "weak"
            crypto_notes.append(GAP_NOTE)
        rows.append(dict(
            id=f"GTJA_{n:03d}", library="GTJA191", name=f"Alpha{n}", category=cat, formula=f,
            notation="gtja191", plain_english=text, evidence="", data_needed=" + ".join(data),
            original_market="China A-shares",
            source="Guotai Junan Securities (Li Chen, Liu Fubing) - 基于短周期价量特征的多因子选股体系 (Quant Topic No. 93)",
            venue="Broker research report", year=2017, crypto_fit=fit, crypto_note=" ".join(crypto_notes),
            related_to=related, formula_issues=GTJA_ISSUES.get(n, "")))
    return rows


# Alpha158 families, following qlib/contrib/data/loader.py (Alpha158DL.get_feature_config).
KBAR = [
    ("KMID", "($close-$open)/$open", "Candle body: (close - open) / open. In 24/7 markets this is about the same as today's return."),
    ("KLEN", "($high-$low)/$open", "Day range: (high - low) / open."),
    ("KMID2", "($close-$open)/($high-$low+1e-12)", "Candle body as a share of the full day range."),
    ("KUP", "($high-Greater($open, $close))/$open", "Upper shadow (high above the body) as % of open."),
    ("KUP2", "($high-Greater($open, $close))/($high-$low+1e-12)", "Upper shadow as a share of the range."),
    ("KLOW", "(Less($open, $close)-$low)/$open", "Lower shadow (low below the body) as % of open."),
    ("KLOW2", "(Less($open, $close)-$low)/($high-$low+1e-12)", "Lower shadow as a share of the range."),
    ("KSFT", "(2*$close-$high-$low)/$open", "Where the close sits vs the middle of the range, as % of open."),
    ("KSFT2", "(2*$close-$high-$low)/($high-$low+1e-12)", "Where the close sits in the range: -1 at the low, +1 at the high."),
]
ROLLING = [
    ("ROC", "Ref($close, {d})/$close", "Close {d} days ago / today's close (the inverse of the {d}-day return).", "Momentum / trend"),
    ("MA", "Mean($close, {d})/$close", "{d}-day average close / today's close.", "Momentum / trend"),
    ("STD", "Std($close, {d})/$close", "{d}-day standard deviation of close / close.", "Volatility / risk"),
    ("BETA", "Slope($close, {d})/$close", "Slope of the close over {d} days / close (trend speed).", "Momentum / trend"),
    ("RSQR", "Rsquare($close, {d})", "R-squared of a straight-line fit to the last {d} closes (how clean the trend is).", "Momentum / trend"),
    ("RESI", "Resi($close, {d})/$close", "Gap between today's close and the {d}-day straight-line trend / close.", "Momentum / trend"),
    ("MAX", "Max($high, {d})/$close", "Highest high of {d} days / close.", "Momentum / trend"),
    ("MIN", "Min($low, {d})/$close", "Lowest low of {d} days / close.", "Momentum / trend"),
    ("QTLU", "Quantile($close, {d}, 0.8)/$close", "80th percentile of {d}-day closes / close.", "Momentum / trend"),
    ("QTLD", "Quantile($close, {d}, 0.2)/$close", "20th percentile of {d}-day closes / close.", "Momentum / trend"),
    ("RANK", "Rank($close, {d})", "Percentile of today's close within its last {d} closes.", "Momentum / trend"),
    ("RSV", "($close-Min($low, {d}))/(Max($high, {d})-Min($low, {d})+1e-12)", "Position of the close inside the {d}-day high-low range (0-1), like stochastic %K.", "Technical indicator"),
    ("IMAX", "IdxMax($high, {d})/{d}", "Days since the {d}-day high / {d} (Aroon-style).", "Technical indicator"),
    ("IMIN", "IdxMin($low, {d})/{d}", "Days since the {d}-day low / {d} (Aroon-style).", "Technical indicator"),
    ("IMXD", "(IdxMax($high, {d})-IdxMin($low, {d}))/{d}", "Gap between the timing of the {d}-day high and low; large = the low came after the high (downward momentum).", "Technical indicator"),
    ("CORR", "Corr($close, Log($volume+1), {d})", "{d}-day correlation between close and log volume.", "Price-volume"),
    ("CORD", "Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), {d})", "{d}-day correlation between daily price change and daily volume change.", "Price-volume"),
    ("CNTP", "Mean($close>Ref($close, 1), {d})", "Share of up days in the last {d}.", "Technical indicator"),
    ("CNTN", "Mean($close<Ref($close, 1), {d})", "Share of down days in the last {d}.", "Technical indicator"),
    ("CNTD", "Mean($close>Ref($close, 1), {d})-Mean($close<Ref($close, 1), {d})", "Up-day share minus down-day share over {d} days.", "Technical indicator"),
    ("SUMP", "Sum(Greater($close-Ref($close, 1), 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", "Total gains / total absolute moves over {d} days (RSI-like, 0-1).", "Technical indicator"),
    ("SUMN", "Sum(Greater(Ref($close, 1)-$close, 0), {d})/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", "Total losses / total absolute moves over {d} days.", "Technical indicator"),
    ("SUMD", "(Sum(Greater($close-Ref($close, 1), 0), {d})-Sum(Greater(Ref($close, 1)-$close, 0), {d}))/(Sum(Abs($close-Ref($close, 1)), {d})+1e-12)", "(Gains - losses) / total absolute moves over {d} days.", "Technical indicator"),
    ("VMA", "Mean($volume, {d})/($volume+1e-12)", "{d}-day average volume / today's volume.", "Volume / activity"),
    ("VSTD", "Std($volume, {d})/($volume+1e-12)", "{d}-day standard deviation of volume / today's volume.", "Volume / activity"),
    ("WVMA", "Std(Abs($close/Ref($close, 1)-1)*$volume, {d})/(Mean(Abs($close/Ref($close, 1)-1)*$volume, {d})+1e-12)", "How uneven volume-weighted price moves were over {d} days (std / mean).", "Volume / activity"),
    ("VSUMP", "Sum(Greater($volume-Ref($volume, 1), 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)", "Total volume increases / total absolute volume changes over {d} days.", "Volume / activity"),
    ("VSUMN", "Sum(Greater(Ref($volume, 1)-$volume, 0), {d})/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)", "Total volume decreases / total absolute volume changes over {d} days.", "Volume / activity"),
    ("VSUMD", "(Sum(Greater($volume-Ref($volume, 1), 0), {d})-Sum(Greater(Ref($volume, 1)-$volume, 0), {d}))/(Sum(Abs($volume-Ref($volume, 1)), {d})+1e-12)", "(Volume increases - decreases) / total absolute volume changes over {d} days.", "Volume / activity"),
]
WINDOWS = [5, 10, 20, 30, 60]


def rows_alpha158():
    base = dict(library="Alpha158", notation="qlib", evidence="", original_market="China A-shares (Qlib default)",
                source="Microsoft Qlib - Alpha158 feature set (Yang et al., Qlib: An AI-oriented Quantitative Investment Platform)",
                venue="Open-source library; arXiv:2009.11189", year=2020, related_to="", formula_issues="")
    model_note = "Designed as a model input feature, not a standalone alpha."
    rows = []
    for name, formula, text in KBAR:
        rows.append(dict(base, id=f"Q158_{name}", name=name, category="Intraday bar", formula=formula,
                         plain_english=text, data_needed="OHLCV", crypto_fit="direct", crypto_note=model_note))
    for field in ["OPEN", "HIGH", "LOW", "VWAP"]:
        rows.append(dict(base, id=f"Q158_{field}0", name=f"{field}0", category="Price level",
                         formula=f"${field.lower()}/$close", plain_english=f"Today's {field.lower()} / today's close.",
                         data_needed="OHLCV" + (" + vwap" if field == "VWAP" else ""), crypto_fit="direct",
                         crypto_note=model_note + (" vwap = quote volume / base volume." if field == "VWAP" else "")))
    for name, formula, text, cat in ROLLING:
        for d in WINDOWS:
            rows.append(dict(base, id=f"Q158_{name}{d}", name=f"{name}{d}", category=cat,
                             formula=formula.format(d=d), plain_english=text.format(d=d), data_needed="OHLCV",
                             crypto_fit="direct", crypto_note=model_note))
    assert len(rows) == 158, len(rows)
    return rows


def rows_extra():
    rows = []
    for lib, items, notation in [("Crypto-academic", CRYPTO_ACADEMIC, "math"),
                                 ("Crypto-practitioner", CRYPTO_PRACTITIONER, "math"),
                                 ("Classic-equity", CLASSIC_EQUITY, "math")]:
        for r in items:
            rows.append(dict(r, library=lib, notation=notation, related_to="", formula_issues=""))
    return rows


def main():
    a101 = {int(k): v for k, v in json.loads((SRC / "alpha101_formulas.json").read_text(encoding="utf-8")).items()}
    gtja = {int(k): v for k, v in json.loads((SRC / "gtja191_formulas.json").read_text(encoding="utf-8")).items()}
    notes_101 = load_101_notes()

    rows = rows_alpha101(a101, notes_101) + rows_gtja191(gtja, a101, notes_101) + rows_alpha158() + rows_extra()
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate ids"
    for r in rows:
        missing = set(COLUMNS) - set(r)
        assert not missing, (r["id"], missing)

    out = HERE / "alpha_catalog.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as fh:  # utf-8-sig so Excel shows Chinese correctly
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in COLUMNS})

    print(f"Wrote {len(rows)} alphas to {out.name}")
    print("By library:", dict(Counter(r["library"] for r in rows)))
    print("By crypto fit:", dict(Counter(r["crypto_fit"] for r in rows)))
    print("By category:", dict(Counter(r["category"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
