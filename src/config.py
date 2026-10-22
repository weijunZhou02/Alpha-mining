"""Settings shared by the data scripts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PANEL = DATA / "panel"
META = DATA / "meta"

# 2020 is downloaded only as warm-up history for lookback windows; research starts in 2021.
DOWNLOAD_START = "2020-01"
RESEARCH_START = "2021-01-01"

TOP_N = 80          # coins in the universe each day
RANK_WINDOW = 30    # days of average quote volume used to rank coins
MIN_HISTORY = 60    # days a coin must have traded before it can join the universe
KEEP_TOP_K = 120    # panels, funding and premium cover every coin that was ever in the top 120

# Not crypto risk: stablecoins, gold tokens, fiat, and Binance index contracts.
EXCLUDED_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "FDUSD", "DAI", "USDE", "USD1", "PYUSD", "EUR", "AEUR",
    "PAXG", "XAUT", "BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL",
}

KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
