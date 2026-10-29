# strategy-backtest

Backtests ten well-documented trading strategies, and a bot that combines them, on free
historical data after trading costs: futures trend, the 10-month index filter, stock momentum,
volatility-scaled momentum, sector ETF momentum, low volatility, quality (gross profitability),
turn of the month, crypto trend and crypto cash-and-carry. Each one is tested on the years after
its paper was published, because published edges shrink once people know about them.

## Run it

```
.venv/Scripts/python selftest.py   # 147 checks; must pass before any number is trusted
.venv/Scripts/python run.py        # writes results/ and prints every table
```

Prove the checks catch mistakes (each must fail):

```
PEEK=1 .venv/Scripts/python selftest.py        # strategy #2 uses the month's own return
FREE_LUNCH=1 .venv/Scripts/python selftest.py  # costs are paid to you instead of by you
```

## Paper trading bot

Runs the stock-only strategy's tradable version ("Bot P" in run.py: SPY/BIL trend with the
turn-of-month layer, top 3 of 9 sector ETFs, USMV + QUAL, the rest in BIL) forward on live prices
with $10,000 of paper money. Nothing talks to a broker.

```
.venv/Scripts/python paper_bot.py run      # process every finished trading day since the last run
.venv/Scripts/python paper_bot.py status   # one-screen summary
```

- Windows Task Scheduler runs `paper_bot.py run` weekdays at 6:00 PM ("Paper trading bot" task,
  with pythonw so no window appears). If the laptop was off, the next run catches up day by day.
- Two books follow the same decisions: real timing trades at the next morning's open;
  backtest timing trades at the close. Their gap is one of the three step-4 checks (see
  `PASS_BAR` in paper_bot.py), judged after 63 trading days.
- Output in `paper/`: `dashboard.html` (open in a browser), `ledger.csv`, `trades.csv`,
  `state.json`, `run.log`.
- It will not use a day's prices until 45 minutes after that day's close, and it refuses to
  run twice at once (`paper/run.lock`).
- `replay START END DIR` runs the same engine over past days; selftest.py uses it to prove the
  engine matches the backtest month by month and never peeks ahead.
- To stop: `schtasks /Delete /TN "Paper trading bot" /F`. To start over: move `paper/state.json`
  aside and run `paper_bot.py start`.

## Files

- `data.py` downloads and caches every series in `data/` (delete a file to refresh it):
  Ken French's library (US stocks 1926 on), Yahoo ETF prices, Bitstamp BTC/ETH, BitMEX funding.
- `strategies.py` holds each strategy as a pure function, the combined bot, and the cost
  assumptions (`COSTS`).
- `run.py` holds the pass bar, written before any result was seen, the publication dates it
  splits on, the paper rebuilds, and the output tables.
- `selftest.py` holds the gates: no position may depend on data from the period it is held in,
  net may never beat gross, and known historical values must match.

## What it found (2026-09-28, data through August 2026)

- Held up on real funds after costs: #1 trend following (barely), #2 index trend filter,
  #5 sector momentum on the nine sector ETFs.
- Held up only before costs: #3 momentum, #6 low volatility, #7 quality. The real funds
  (MTUM, USMV, QUAL) all trailed SPY.
- Failed: #4 long-only scaled momentum, #8 turn of the month (faded after 1987), #9 crypto trend
  (lost to holding the coin), #10 as written (its exit rule switches 130 to 390 times a year).
- Stock-only bot, 1964-2026: 10.7% a year vs the market's 10.8%, with volatility of 10.4% vs 15.5%.
- Full bot as written lost 1.7% a year. The second try (carry never exits) made 11.6% vs 15.1%
  for SPY, but kept 44% of its money in BitMEX funding on average, so treat it as a new idea.
- The Bitcoin weekly momentum result from Liu and Tsyvinski did not reproduce on Bitstamp data.

## Day-trading paper bot (started 2026-10-01)

A high-risk, many-trades version. `daytrade_bot.py` runs three published
intraday rules on QQQ and SPY with $25,000 of paper money ($25k is the floor a real US account
needs to day trade freely), a third each, up to 4x leverage, flat every night:
opening range breakout (Zarattini & Aziz 2023), VWAP trend (Zarattini & Aziz 2023) and
last-half-hour momentum (Gao, Han, Li & Zhou 2018). Costs: 1 basis point slippage per side plus
$0.0035 a share, minimum $0.35 an order.

```
.venv/Scripts/python daytrade_bot.py run       # task "Day trading paper bot", weekdays 6:15 PM
.venv/Scripts/python daytrade_bot.py status
.venv/Scripts/python daytrade_bot.py history   # replay of the ~59 days Yahoo still has
.venv/Scripts/python selftest_daytrade.py      # no look-ahead, costs hurt, 4x cap, flat nightly
PEEK=1 .venv/Scripts/python selftest_daytrade.py   # must fail: decisions see the bar they trade on
```

Output in `daytrade/` (dashboard.html, trades.csv, state.json, run.log). 5-minute bars are
cached in `data/intraday/` so history outlives Yahoo's 60-day window.

Replay before the trial, 2026-07-10 to 2026-10-01 (59 sessions, 733 trades): -13.7% vs +1.6% for
holding SPY. ORB +4.9% (+9.7% before costs), VWAP -39% (-12% before costs), MOM -6.9%
(-2% before costs). To stop: `schtasks /Delete /TN "Day trading paper bot" /F`.
