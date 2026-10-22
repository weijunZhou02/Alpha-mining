# 101 Formulaic Alphas in Plain English

Oct 22, 2026 · @Charly

## How to read the formulas

Every alpha gives each stock one score per day: a high score means buy, a low score means short, and only the order across stocks matters. The formulas come from Kakushadze, [101 Formulaic Alphas](https://arxiv.org/abs/1601.00991) (Wilmott, 2016), and use the inputs and operators below. Read each formula from the inside out. The exact formulas are in Appendix A of the paper, and WorldQuant keeps the rights to them, so this doc explains each one in words: keep the paper open beside it.

### Input data

| Name | Plain English |
| --- | --- |
| `open`, `close`, `high`, `low` | Today's opening, closing, highest and lowest price |
| `volume` | Shares traded today |
| `vwap` | Average price paid per share today, weighted by volume (today's fair price) |
| `returns` | Today's close-to-close return, e.g. -0.02 = down 2% |
| `adv{d}` (e.g. `adv20`) | Average daily dollar volume over the past d days (normal trading activity) |
| `cap` | Market capitalization (company size) |
| `IndClass.sector / industry / subindustry` | The stock's group in an industry classification such as GICS |

### Operators

| Operator | Plain English | Compares across |
| --- | --- | --- |
| `rank(x)` | Compared with all other stocks today, how high is x? Gives 0 to 1 | Stocks |
| `scale(x)` | Resize x so the absolute values add up to 1 across stocks | Stocks |
| `indneutralize(x, g)` | x minus the average x of its industry group, so only the difference from peers is left | Stocks in the same group |
| `delay(x, d)` | x as it was d days ago | Time |
| `delta(x, d)` | x today minus x d days ago: how much x changed | Time |
| `sum(x, d)`, `product(x, d)` | Sum, or product, of x over the past d days | Time |
| `stddev(x, d)` | How bumpy x was over the past d days | Time |
| `correlation(x, y, d)` | Over the past d days, did x and y move up and down together? Gives -1 to +1 | Time |
| `covariance(x, y, d)` | Like correlation, but keeps the size of the moves | Time |
| `ts_min(x, d)`, `ts_max(x, d)` (also `min`, `max` with a window) | Lowest, or highest, x of the past d days | Time |
| `ts_argmin(x, d)`, `ts_argmax(x, d)` | On which of the past d days x was lowest, or highest | Time |
| `ts_rank(x, d)` | Compared with this stock's own past d days, how high is x today? | Time |
| `decay_linear(x, d)` | Average of x over the past d days, with recent days weighted more (weights d, d-1, ..., 1) | Time |
| `signedpower(x, a)`, `x^a` | x raised to the power a, keeping its sign | - |
| `abs(x)`, `sign(x)`, `log(x)` | Size of x / direction of x (+1 or -1) / natural log of x | - |
| `a ? b : c` | If a is true, use b, otherwise use c | - |
| `a < b`, `a \|\| b` | A true/false test; used as a number, true = 1 and false = 0 | - |
| `-1 * (...)` | Flip the sign: bet against what the inside measures | - |

Windows with decimals, such as 9.91 days, are rounded down to whole days. Many of the later alphas (from about #48 on) use `indneutralize`, so they compare a stock with its industry peers rather than with the whole market.

## Alphas 1-10

Alphas 1-10 are short and readable: most bet against a recent move, and three of them link price with volume.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 1 | Short-term reversal | For each of the last 5 days, take the close if the stock rose or the 20-day volatility if it fell, square it, and find the day with the largest value. A squared price always beats a squared volatility, so this really finds the week's highest up-day close. Score = which day that was, ranked and centered on 0. Read as "days since the peak", a fresh peak scores low (short); code that counts the other way turns it into momentum. |
| 2 | Price-volume | Over 6 days, compares two daily rankings: how much volume grew versus 2 days earlier, and how much the stock gained from open to close. Volume jumps landing on up days (hype) → short; landing on down days (panic selling) → buy. |
| 3 | Price-volume | Over 10 days, checks whether the opening price and the volume, each ranked against all stocks every day, rise and fall together. Together (a crowd piling in) → short; opposite → buy. |
| 4 | Short-term reversal | Ranks today's low against all stocks, then asks where that rank sits within the stock's own last 9 days. Top of its 9-day range (strong versus the market lately) → short; bottom → buy. A plain rubber-band reversal. |
| 5 | VWAP deviation | Multiplies two ranks: how far today's open is above the 10-day average vwap, and how far today's close is above today's vwap (made negative). All scores are 0 or below; the most negative are stocks that opened high and closed strong, which it shorts. Stocks that opened low or faded intraday sit near 0, on the buy side. |
| 6 | Price-volume | The 10-day correlation between raw opening price and raw volume, flipped. Rising together → short; moving apart → buy. The raw version of #3, so one huge-volume day can dominate it. |
| 7 | Volume surprise | Acts only when today's volume is above its 20-day average. Then it bets against the 7-day price change, harder when the move is unusually large versus the stock's own last 60 days: big drop → buy, big rise → short. Every quiet stock gets -1, the lowest score. |
| 8 | Short-term reversal | Multiplies the sum of the last 5 opens by the sum of the last 5 returns (roughly price level × recent performance) and compares it with the same number 10 days ago. Grown → short; shrunk → buy. A reversal of the last week versus two weeks earlier. |
| 9 | Short-term reversal | Looks at the last 5 daily price changes. If all 5 were up, or all 5 were down, it follows today's change. On every other day, which is most days, it bets against today's change. A one-day reversal that switches off during clean streaks. |
| 10 | Short-term reversal | Same as #9 with a 4-day lookback, then ranked across stocks: follow today's move during a clean 4-day streak, otherwise bet against it. |

## Alphas 11-20

Alphas 11-20 use volume to confirm or doubt a price move, and add the first gap and candle-body signals.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 11 | VWAP deviation | Over the last 3 days, takes the largest and the smallest gap of vwap above the close, ranks both and adds them, then multiplies by the rank of the 3-day volume change. High score = the stock kept closing below its average price while volume grew → buy. |
| 12 | Short-term reversal | If volume rose today, bet against today's price change; if volume fell, follow it. Volume decides whether today's move is trusted. The simplest volume-confirmed reversal in the set. |
| 13 | Price-volume | The 5-day covariance between the close's rank and volume's rank (both versus all stocks), ranked and flipped. Like #3, but covariance keeps the size of the co-movement, not only its direction. |
| 14 | Short-term reversal | Takes the 3-day change in daily returns (is performance improving?), ranks it and flips it, then multiplies by the 10-day correlation of open and volume. When price and volume move together, improving stocks get shorted; when they move apart, the sign flips. |
| 15 | Price-volume | Each day, takes the 3-day correlation between the high's rank and volume's rank, ranks it, adds the last 3 days, and flips. Highs and heavy volume repeatedly arriving together → short. |
| 16 | Price-volume | Same as #13 with the day's high instead of the close: 5-day covariance of high rank and volume rank, ranked and flipped. |
| 17 | Short-term reversal | Multiplies three ranks: how close the close is to its 10-day high (made negative), how much the daily price change is speeding up, and how unusual volume is versus normal over 5 days. Most negative (short) when a stock is at a 10-day high, accelerating, on heavy volume. |
| 18 | Intraday bar | Adds three things and flips: how bumpy the daily candle bodies (close minus open) were over 5 days, today's candle body, and how closely close and open tracked each other over 10 days. Big up candles, jumpy bodies and tight open-close tracking → short. |
| 19 | Short-term reversal | Bets against the direction of the last 7 days' price change, sized by 1 plus the rank of the 1-year return. Short-term reversal, bet harder on long-term winners. |
| 20 | Intraday bar | Compares today's open with yesterday's high, close and low, ranks each gap, and multiplies them with a minus sign. Opening above all three (a strong gap up) gives the most negative score → short; a gap down sits near 0, on the buy side. |

## Alphas 21-30

Alphas 21-30 bring in if-then rules (bands and trend filters) and the first long-horizon inputs, 100 days and more.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 21 | Short-term reversal | A mini Bollinger band. If the 2-day average close is above the 8-day average plus one 8-day standard deviation → -1 (short); below the 8-day average minus one standard deviation → +1 (buy). Inside the band: +1 if volume is at or above its 20-day average, else -1. Output is only +1 or -1. |
| 22 | Price-volume | How much the 5-day correlation between the high and volume has risen over the last 5 days, times the rank of 20-day price volatility, flipped. A strengthening high-volume link → short, especially in volatile stocks. |
| 23 | Short-term reversal | Only when today's high is above its 20-day average high: bet against the 2-day change in the high. Every other stock gets 0. Fades fresh pushes above the recent range. |
| 24 | Trend & momentum | Checks the long trend: how much the 100-day average price rose over 100 days, as a share of the price 100 days ago. 5% or less (weak trend) → short in proportion to how far the price is above its 100-day low. Stronger uptrend → bet against the last 3 days' change. |
| 25 | Short-term reversal | Multiplies minus today's return, 20-day dollar volume, vwap, and the distance from the close down from the high, then ranks. Stocks that fell, closed well below their high, and are large and liquid score high → buy. Because dollar volume and price are multiplied in, big expensive stocks dominate. |
| 26 | Price-volume | Each day, the 5-day correlation between where volume sits in its own 5-day range and where the high sits in its own 5-day range; takes the largest of the last 3 days and flips. Volume and highs climbing together → short. |
| 27 | Price-volume | Averages the last 2 days of the 6-day correlation between volume rank and vwap rank, then ranks across stocks. Top half → -1 (short), bottom half → +1 (buy). A two-bucket signal with nothing in between. |
| 28 | Intraday bar | Adds the 5-day correlation of 20-day dollar volume with the low to the gap between the day's midpoint, (high + low) / 2, and the close, then scales. Closing below the midpoint → high score, buy; closing near the high → short. |
| 29 | Short-term reversal | Many layers (ranks, a log, a scale, a 2-day min, a 5-day min) wrapped around one core: minus the 5-day price change. Plus a small term: where the return from 6 days ago, flipped, sits in its 5-day range. In effect a smoothed 5-day reversal; most layers barely change the ordering. |
| 30 | Short-term reversal | Adds the signs of the last 3 daily price changes (up = +1, down = -1), ranks the total, and takes 1 minus that rank, so three up days score low. Multiplies by the share of the last 20 days' volume traded in the last 5. Buys recent losers, more so when volume is concentrated in the last week. |

## Alphas 31-40

Alphas 31-40 start blending several signals with weights, and mix short-term reversal with 200-250 day context.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 31 | Short-term reversal | Adds three parts: a smoothed, many-times-ranked reversal of the 10-day price change, a ranked reversal of the 3-day change, and +1 or -1 depending on whether dollar volume and the low moved together over 12 days. Mainly buys stocks that fell over 3 and 10 days. |
| 32 | Short-term reversal | Two scaled parts: the 7-day average close minus today's close (below the average → buy), plus 20 times the 230-day correlation between vwap and the close 5 days earlier. The first is mean reversion to a moving average; the second, weighted heavily, favors stocks whose prices move smoothly over the long run. |
| 33 | Intraday bar | Ranks open divided by close, minus 1. Fell during the day (close below open) → high rank, buy; rose intraday → short. A pure one-day intraday reversal (the power of 1 does nothing). |
| 34 | Short-term reversal | Adds two parts and ranks: 1 minus the rank of 2-day volatility over 5-day volatility (calming down → higher), and 1 minus the rank of today's price change (fell → higher). Buys stocks that fell today while their volatility settles. |
| 35 | Volume surprise | Multiplies three time-series ranks: volume versus its own last 32 days (high is good), 1 minus close + high - low versus its last 16 days (low is good), and 1 minus today's return versus its last 32 days (low is good). Buys stocks that are weak by their own standards on unusually heavy volume. |
| 36 | Price-volume | A blend of five ranks with hand-set weights: 15-day correlation of the candle body with yesterday's volume (2.21), today's drop from open to close (0.7), a reversed return from 6 days ago (0.73), the absolute 6-day correlation of vwap and dollar volume (1), and the 200-day average minus the open, times the candle body (0.6). Mostly price-volume with reversal add-ons. |
| 37 | Intraday bar | Adds two ranks: the 200-day correlation between yesterday's intraday drop (open minus close) and today's close, and today's intraday drop. Fell during the day → higher score, buy. |
| 38 | Short-term reversal | Minus the rank of where the close sits in its 10-day range, times the rank of close over open. Near a 10-day high and up a lot intraday → most negative, short. |
| 39 | Short-term reversal | Bets against the 7-day price change, amplified when volume over the last 9 days (recent days weighted more) was light versus normal, then sized by 1 plus the rank of the 1-year return. Fades weekly moves made on thin volume, harder for long-term winners. |
| 40 | Price-volume | Minus the rank of the high's 10-day volatility, times the 10-day correlation of high and volume. Volatile stocks whose highs and volume move together → short; moving apart flips it to buy. |

## Alphas 41-50

Alphas 41-50 include the two simplest vwap signals (#41, #42), a family of trend-speed rules (#46, #49), and the first industry-neutral alpha (#48).

| Alpha | Type | Plain English |
| --- | --- | --- |
| 41 | VWAP deviation | The geometric middle of the day's range (square root of high × low) minus vwap. Positive when most shares traded below the middle of the range → buy. One of the simplest in the set. |
| 42 | VWAP deviation | Rank of vwap minus close, divided by rank of vwap plus close. Closing below the day's average price → high score, buy; closing above → short. The paper calls it a same-day (delay-0) mean-reversion alpha: it trades at the close of the day it is computed. |
| 43 | Volume surprise | Multiplies two time-series ranks: today's volume over its 20-day average, versus the last 20 days; and the 7-day price drop, versus the last 8 days. Buys stocks falling hard on unusually heavy volume. |
| 44 | Price-volume | The 5-day correlation between the raw high and volume's rank (versus all stocks), flipped. Highs and relative volume rising together → short. |
| 45 | Price-volume | Multiplies three parts and flips: the rank of the 20-day average of week-old closes, the 2-day correlation of close and volume, and the rank of the 2-day correlation between 5-day and 20-day sums of closes. Driven mostly by the very short 2-day price-volume correlation. |
| 46 | Trend & momentum | Compares the price's speed over the last 10 days with its speed over the 10 days before, in dollars per day. Speeding up by more than $0.25 a day → -1 (short); slowing down → +1 (buy); in between → bet against today's change. The threshold is in dollars, so a $10 and a $500 stock are treated very differently. |
| 47 | Volume surprise | Relative volume (today over its 20-day average) weighted toward cheap stocks, times today's high versus its 5-day average (scaled by how far the close fell from the high), minus the rank of the 5-day change in vwap. High score for cheap stocks on heavy volume that closed off the high while their average price fell over the week → buy. |
| 48 | Trend & momentum | Measures over 250 days whether today's price change tends to repeat yesterday's (a correlation), multiplies it by today's change over price, subtracts the subindustry average, and divides by the 1-year sum of squared returns. Where moves tend to persist it follows today's move; where they tend to reverse it bets against it. |
| 49 | Trend & momentum | The same speed comparison as #46 with one rule: slowed down by more than $0.10 a day versus the prior 10 days → +1 (buy); otherwise bet against today's price change. |
| 50 | Price-volume | Each day, ranks the 5-day correlation between volume rank and vwap rank; takes the highest of the last 5 days and flips. A recent strong link between volume and price → short. |

## Alphas 51-60

Alphas 51-60 finish the trend-speed family and the bar-shape signals; from #58 on, windows have decimals and industry-relative inputs become common.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 51 | Trend & momentum | The same speed comparison as #46 and #49: slowed down by more than $0.05 a day versus the prior 10 days → +1 (buy); otherwise bet against today's price change. |
| 52 | Trend & momentum | Multiplies three parts: how much the 5-day low has dropped versus 5 days ago (a recent dip), the rank of the return from about 11 months ago to 1 month ago (classic long-term momentum that skips the last month), and where today's volume sits in its last 5 days. Buys dips in long-term winners on rising volume; shorts them when their lows are rising. |
| 53 | Intraday bar | Measures where the close sits in the day's range as ((close - low) - (high - close)) / (close - low), and bets against its change over 9 days. Closes creeping up toward the highs versus 9 days ago → short. Blows up when the close equals the low. |
| 54 | Intraday bar | Minus the close's position in the day's range (0 = at the low, 1 = at the high), times (open / close) to the 5th power. Mostly: close near the high → short, near the low → buy; the open-to-close term makes the signal stronger or weaker. |
| 55 | Price-volume | Finds where the close sits within its 12-day low-to-high range, ranks it, and takes its 6-day correlation with volume's rank, flipped. Pushing to the top of the range on rising relative volume → short. |
| 56 | Short-term reversal | Minus the rank of the 10-day return divided by a short recent-return total (roughly the last 4 days), times the rank of today's return × market cap. Large companies that rose today get shorted hardest. The only alpha that uses market cap; the ratio blows up when the recent total is near zero. |
| 57 | VWAP deviation | Close minus vwap, divided by a 2-day smoothed rank of how many days ago the 30-day closing high was, flipped. Closing above vwap → short, below → buy; the divisor scales the bet by the timing of the recent high. |
| 58 | Industry-neutral combo | Subtracts the sector average from vwap, takes its 3-day correlation with volume, smooths it over 7 days (recent days weighted more), and asks where today's value sits in its last 5 days, flipped. Sector-relative price and volume rising together → short. |
| 59 | Industry-neutral combo | The same recipe as #58 at the industry level with longer windows (4-day correlation, 16-day smoothing, 8-day time rank). It mixes vwap with itself (0.73 × vwap + 0.27 × vwap), which is just vwap. |
| 60 | Intraday bar | Two scaled parts: where the close sits in the day's range times volume, ranked (weight 2), minus the rank of how many days ago the 10-day closing high was. Flipped, so closing near the high on heavy volume → short. Close to the classic accumulation/distribution idea. |

## Alphas 61-70

From #61 on, almost every alpha joins two signals with a comparison (a < b, giving 1 or 0), a power (a ^ b), or max/min, and uses decimal windows that look machine-tuned. Windows below are rounded down, as the paper specifies.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 61 | Mined combo | Outputs 1 or 0. Compares the rank of how far vwap is above its 16-day low with the rank of the 17-day correlation between vwap and 180-day dollar volume. Score 1 (buy side) when the first is lower: the stock sits near its recent low relative to how tightly its price tracks liquidity. |
| 62 | Mined combo | Outputs -1 or 0. Compares the 9-day correlation of vwap with 22-day summed dollar volume against a yes/no flag for "the high and midpoint rank above the open today" (the stock traded up from its open). -1 (short) when the correlation ranks lower than that flag. |
| 63 | Industry-neutral combo | Two smoothed ranks subtracted and flipped: the 2-day change in the industry-relative close (smoothed 8 days), minus the 13-day correlation of a vwap/open blend with 37-day summed 180-day dollar volume (smoothed 12 days). Shorts stocks that rose versus their industry; buys those whose price tracks liquidity closely. |
| 64 | Mined combo | Outputs -1 or 0. -1 (short) when the 3-day price change (mostly vwap) outranks the 16-day correlation of 12-day summed prices (mostly the low) with 12-day summed 120-day dollar volume. In effect: short recent risers whose price does not track liquidity. |
| 65 | Mined combo | Outputs -1 or 0. -1 (short) when how far the open is above its 13-day low outranks the 6-day correlation of (almost pure) vwap with 8-day summed 60-day dollar volume. Shorts stocks that have bounced far off their recent low. The 0.008 weight on the open makes the blend nearly pure vwap. |
| 66 | Mined combo | Adds two parts and flips: the rank of the smoothed 3-day change in vwap, and where a smoothed bar-shape ratio, (low - vwap) / (open - midpoint), sits in its last 6 days. Mainly shorts stocks whose average price rose over the last few days. The low is mixed with itself (0.97 × low + 0.03 × low). |
| 67 | Industry-neutral combo | Rank of how far today's high is above the lowest high of the last 2 days, raised to the power of the rank of the 6-day correlation between sector-relative vwap and subindustry-relative dollar volume, flipped. A bigger new high → more negative (short); the power term sets how much that counts. |
| 68 | Mined combo | Outputs -1 or 0. -1 (short) when the 1-day rise in a close/low blend outranks where the 8-day correlation of high rank and 15-day dollar volume rank sits in its last 13 days. Shorts today's risers when the price-liquidity link is weak. |
| 69 | Industry-neutral combo | Rank of the largest 2-day rise in industry-relative vwap over the last 4 days, raised to the power of where the 4-day correlation of a close/vwap blend with dollar volume sits in its last 9 days, flipped. Recent jumps versus the industry → short. |
| 70 | Industry-neutral combo | Rank of the 1-day change in vwap, raised to the power of where the 17-day correlation of industry-relative close with 50-day dollar volume sits in its last 17 days, flipped. Today's risers → short; the power term tunes the strength. |

## Alphas 71-80

Alphas 71-80 keep pairing a price-change signal with a price-liquidity correlation; #74 and #75 are near mirror images of each other.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 71 | Mined combo | The larger of two time-series ranks: the smoothed 18-day correlation between the close's 3-day position and 180-day dollar volume's 12-day position (ranked over 15 days), and a smoothed, squared rank of (low + open) minus twice vwap (ranked over 4 days). High when either the price-liquidity link is unusually strong or the low and open sit well above vwap → buy. |
| 72 | Mined combo | A ratio of two smoothed ranks: the 8-day correlation of the day's midpoint with 40-day dollar volume, divided by the 6-day correlation between vwap's 3-day position and volume's 18-day position. High when the slow price-liquidity link is strong but the fast price-volume link is weak → buy. |
| 73 | Mined combo | The larger of two signals, flipped: the rank of the smoothed 4-day change in vwap, and where the smoothed 2-day percentage drop of an open/low blend sits in its last 16 days. Shorts stocks that moved a lot either way recently, so it is hard to read cleanly. |
| 74 | Mined combo | Outputs -1 or 0. -1 (short) when the 15-day correlation of close with 37-day summed 30-day dollar volume ranks below the 11-day correlation of (almost pure) vwap rank with volume rank. Shorts stocks whose fast price-volume link beats their slow price-liquidity link. |
| 75 | Mined combo | Outputs 1 or 0. 1 (buy side) when the 4-day vwap-volume correlation ranks below the 12-day correlation of low rank with 50-day dollar volume rank. Buys stocks whose fast price-volume link is weak relative to the slow liquidity link: the mirror of #74. |
| 76 | Industry-neutral combo | The larger of two signals, flipped: the rank of the 1-day change in vwap (smoothed 11 days), and a heavily smoothed time rank of the 8-day correlation between sector-relative low and 81-day dollar volume. Shorts recent risers or stocks with an unusually strong sector-relative price-liquidity link. |
| 77 | Mined combo | The smaller of two smoothed ranks: the day's midpoint minus vwap (smoothed 20 days; the two highs in the formula cancel), and the 3-day correlation of the midpoint with 40-day dollar volume (smoothed 5 days). High only when both are high: trading below the middle of the range plus a positive price-liquidity link → buy. |
| 78 | Mined combo | The rank of the 6-day correlation of 19-day summed prices (low/vwap blend) with 19-day summed 40-day dollar volume, raised to the power of the rank of the 5-day vwap-volume rank correlation. High when the slow price-liquidity link is strong and the fast price-volume link is weak → buy. |
| 79 | Industry-neutral combo | Outputs 1 or 0. 1 (buy side) when the 1-day change in sector-relative price (close/open blend) ranks below the 14-day correlation between vwap's 3-day position and 150-day dollar volume's 9-day position. Favors stocks that did not rise versus their sector today. |
| 80 | Industry-neutral combo | Rank of the direction (+1 or -1) of the 4-day change in industry-relative price (mostly the open), raised to the power of where the 5-day correlation of high with 10-day dollar volume sits in its last 5 days, flipped. Rising versus its industry → short. |

## Alphas 81-90

Alphas 81-90 are mostly two-signal combos; #83 is the one readable exception, a close-versus-vwap signal scaled by range and volume.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 81 | Mined combo | Outputs -1 or 0. Left side: the 8-day correlation of vwap with 49-day summed 10-day dollar volume, wrapped in ranks, a 4th power, a 14-day product and a log. -1 (short) when that ranks below the 5-day vwap-volume rank correlation. |
| 82 | Industry-neutral combo | The smaller of two signals, flipped: the rank of the smoothed 1-day change in the open (14-day smoothing), and a time rank of the smoothed 17-day correlation between sector-relative volume and the open. Shorts only when both are high: rising opens plus a strong volume-open link. The open is mixed with itself (0.63 × open + 0.37 × open). |
| 83 | VWAP deviation | Let R = today's high-low range as a share of the 5-day average close. Score = rank of R two days ago × rank of volume × (vwap - close) / R. Positive when the close is below vwap (buy), larger on heavy volume, smaller when today's range is wide. |
| 84 | Mined combo | Where vwap's distance below its 15-day high sits in its last 20 days (between 0 and 1), raised to the power of the 4-day price change in dollars. Because the base is below 1, a 4-day rise shrinks the score toward 0 and a 4-day fall pushes it above 1, so it mostly buys recent losers. A base of exactly 0 with a falling price blows up. |
| 85 | Mined combo | The rank of the 9-day correlation of a high/close blend with 30-day dollar volume, raised to the power of the rank of the 7-day correlation between the midpoint's 3-day position and volume's 10-day position. Like #78: strong slow price-liquidity link and weak fast price-volume link → buy. |
| 86 | Mined combo | Outputs -1 or 0. -1 (short) when how far the close is above vwap outranks where the 6-day correlation of close with 14-day summed dollar volume sits in its last 20 days. Shorts stocks closing above their average price. The two opens in the formula cancel. |
| 87 | Industry-neutral combo | The larger of two signals, flipped: the rank of the smoothed 1-day change in a close/vwap blend, and a time rank of the smoothed absolute 13-day correlation between industry-relative 81-day dollar volume and the close. Shorts today's risers or stocks with a strong liquidity-price link of either sign. |
| 88 | Mined combo | The smaller of two signals: the smoothed rank of (open rank + low rank) minus (high rank + close rank), high after an intraday drop; and a short time rank of the smoothed correlation between the close's 8-day position and 60-day dollar volume's 20-day position. High only when both are high → buy. |
| 89 | Industry-neutral combo | Two time ranks subtracted: the smoothed 6-day correlation of the low with 10-day dollar volume, minus the smoothed 3-day change in industry-relative vwap. Buys stocks with a strong price-liquidity link that have not risen versus their industry. The low is mixed with itself (0.97 × low + 0.03 × low). |
| 90 | Industry-neutral combo | Rank of the close minus its 4-day high (stocks at a new high rank top), raised to the power of where the 5-day correlation of subindustry-relative 40-day dollar volume with the low sits in its last 3 days, flipped. At or near a 4-day high → short. |

## Alphas 91-101

Alphas 91-99 continue the two-signal combos; #100 is an industry-relative version of #60, and #101 closes the list with the simplest formula of all.

| Alpha | Type | Plain English |
| --- | --- | --- |
| 91 | Industry-neutral combo | Two parts subtracted and flipped: a double-smoothed 9-day correlation of industry-relative close with volume (time rank over 4 days), minus the rank of the smoothed 4-day correlation of vwap with 30-day dollar volume. Short when the industry-relative price-volume link is strong; buy when the plain price-liquidity link is strong. |
| 92 | Mined combo | The smaller of two time ranks: how often lately the day closed with a big down candle (fell from the open by more than half the day's range; smoothed 14 days), and the smoothed 7-day correlation of low rank with 30-day dollar volume rank. High only when both are high → buy. |
| 93 | Industry-neutral combo | A ratio: a time rank of the smoothed 17-day correlation between industry-relative vwap and 81-day dollar volume, divided by the rank of the smoothed 2-day change in a close/vwap blend. Strong price-liquidity link and no recent rise → buy. |
| 94 | Mined combo | Rank of how far vwap is above its 11-day low, raised to the power of a short time rank of the 18-day correlation between vwap's 19-day position and 60-day dollar volume's 4-day position, flipped. Far above the recent low → short. |
| 95 | Mined combo | Outputs 1 or 0. 1 (buy side) when how far the open is above its 12-day low ranks below an 11-day time rank of the (5th-powered) 12-day correlation of 19-day summed midpoints with 19-day summed 40-day dollar volume. Favors stocks still near their recent low. |
| 96 | Mined combo | The larger of two time ranks, flipped: the smoothed 3-day vwap-volume rank correlation, and the smoothed timing (days since its peak) of a 3-day correlation between the close's 7-day position and 60-day dollar volume's 4-day position. Shorts stocks with an unusually strong recent price-volume link. |
| 97 | Industry-neutral combo | Two parts subtracted and flipped: the rank of the smoothed 3-day change in industry-relative price (low/vwap blend), minus a heavily smoothed time rank of the 4-day correlation between the low's 7-day position and 60-day dollar volume's 17-day position. Rising versus its industry → short. |
| 98 | Mined combo | Two smoothed ranks subtracted: the 4-day correlation of vwap with 26-day summed 5-day dollar volume, minus a smoothed time rank of when (within 8 days) the 20-day open-liquidity rank correlation hit its low. Mostly buys stocks whose price tracks liquidity closely. |
| 99 | Mined combo | Outputs -1 or 0. -1 (short) when the 8-day correlation of 19-day summed midpoints with 19-day summed 60-day dollar volume ranks below the 6-day correlation of the low with volume. Like #74: shorts stocks whose fast price-volume link beats the slow price-liquidity link. |
| 100 | Industry-neutral combo | Two scaled, subindustry-relative parts: where the close sits in the day's range times volume, ranked (weight 1.5; neutralized twice, which is the same as once), minus \[the 5-day correlation of close with dollar-volume rank, minus the rank of how many days ago the 30-day low was\]. Flipped and multiplied by today's relative volume. Closing near the high on heavy volume, versus peers → short. An industry-relative upgrade of #60. |
| 101 | Intraday bar | Today's candle body divided by today's range (the 0.001 avoids dividing by zero): +1 = opened at the low and closed at the high, -1 = the reverse. Buys stocks that rose during the day. The paper calls it a momentum alpha traded the next day (delay-1). |

## Common patterns and quirks

The 101 alphas are short-term, mostly contrarian, and built from a small toolbox; the paper reports average holding periods of about 0.6 to 6.4 days, an average pairwise correlation of 15.9%, and performance before trading costs.

- **Reversal dominates.** Most readable alphas bet against a recent move. Clear momentum is rare: the trend-speed rules (#46, #49, #51), dip-buying in long-term winners (#52), and #101.
- **Price-volume links are the most common ingredient.** A correlation or covariance between a price and volume or dollar volume sits at the core of most alphas after #57.
- **Two eras.** #1-#57 are mostly readable. #58-#99 use decimal windows, comparisons, powers and max/min, which looks machine-tuned: hard to interpret and easy to overfit if you copy the shape.
- **Pieces that do nothing.** A price mixed with itself (#59, #66, #82, #89), terms that cancel (#77, #86), neutralizing twice (#100), and stacked ranks around one core (#29). They do not change the signal.
- **Unit traps.** Dollar thresholds (#46, #49, #51) and dollar differences (#5, #25) treat cheap and expensive stocks differently; `adv` is defined in dollars but #7 compares it with share volume.
- **Division traps.** #53, #56 and #84 can divide by a number near zero; clip or guard these in code.
- **Different output shapes.** Some alphas only output 1/0, -1/0 or +1/-1 (#21, #27, #61, #62 and others). Rank or neutralize every alpha before combining them.
- **Conventions the paper leaves open.** Whether `ts_argmax` / `ts_argmin` count days ago or days from the start (#1, #57, #60, #96, #98, #100), and the scale of `ts_rank`. Test both directions.
- **Industry data needed.** 18 alphas use `indneutralize` (#48, #58-#100 range), so you need an industry classification such as GICS to compute them.
