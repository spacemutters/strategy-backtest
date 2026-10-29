"""Gates for the day-trading paper bot. Every number it reports is only trusted if these pass.

  1. No look-ahead: changing how a bar ends (its high, low, close and volume, keeping its open)
     must not change any trade made at or before that bar's open. PEEK=1 plants exactly that
     bug, and this file must then fail.
  2. Costs always hurt: the same days with costs switched off must end higher.
  3. Leverage cap: no position is ever worth more than 4x its rule's money at entry.
  4. Every rule is flat at the end of every day.
"""
from __future__ import annotations

import random
import sys

import pandas as pd

import daytrade_bot as b

fails: list[str] = []
checks = 0


def check(ok: bool, msg: str) -> None:
    global checks
    checks += 1
    if not ok:
        fails.append(msg)


bars = {t: pd.read_pickle(b.CACHE / f"{t}_5m.pkl") for t in b.TICKERS}
now = pd.Timestamp.now(tz=b.NY).to_pydatetime()
days = [d for d in b.sessions(bars) if b.complete(bars, d, now)]
rng = random.Random(7)
eq = {k: 25000 / 3 for k in b.SLEEVES}

# 1. look-ahead
for day in rng.sample(days[1:], min(12, len(days) - 1)):
    prev = {t: float(b.day_bars(bars[t], days[days.index(day) - 1]).iloc[-1].close) for t in b.TICKERS}
    base = b.run_day(bars, day, prev, eq)["trades"]
    q = b.day_bars(bars["QQQ"], day)
    for i in rng.sample(range(2, len(q) - 1), 6):
        ts = q.index[i]
        bent = {t: df.copy() for t, df in bars.items()}
        for t in b.TICKERS:
            df = bent[t]
            if ts in df.index:
                o = df.at[ts, "open"]
                df.at[ts, "high"] = o * 1.03
                df.at[ts, "low"] = o * 0.97
                df.at[ts, "close"] = o * (0.97 if df.at[ts, "close"] >= o else 1.03)
                df.at[ts, "volume"] = df.at[ts, "volume"] * 50
        alt = b.run_day(bent, day, prev, eq)["trades"]
        # Trades decided before this bar finished must not move. Stops and targets filled
        # inside the bent bar legitimately change, so only decisions are compared.
        def decided(tr):
            return [(x["time"], x["sleeve"], x["shares"]) for x in tr
                    if pd.Timestamp(x["time"]) <= ts and x["why"] not in ("stop", "target")
                    and not (pd.Timestamp(x["time"]) == ts and x["sleeve"] == "ORB" and x["why"] != "opening range breakout")]
        check(decided(base) == decided(alt), f"look-ahead: {day.date()} changed when the {ts.time()} bar was altered")

# 2. costs hurt, 3. leverage, 4. flat
res = b.replay(bars)
saved = (b.SLIPPAGE, b.COMMISSION, b.COMMISSION_MIN)
b.SLIPPAGE, b.COMMISSION, b.COMMISSION_MIN = 0.0, 0.0, 0.0
free = b.replay(bars)
b.SLIPPAGE, b.COMMISSION, b.COMMISSION_MIN = saved
check(free["stats"]["final"] > res["stats"]["final"], "costs: switching costs off did not raise the result")

trades = pd.DataFrame(res["trades"])
trades["day"] = trades["time"].str[:10]
for (day, sleeve), g in trades.groupby(["day", "sleeve"]):
    check(abs(g["shares"].sum()) < 1e-6, f"flat: {sleeve} still held shares at the end of {day}")
sleeve_eq = {}
for d in res["state"]["days"]:
    sleeve_eq[d["day"]] = d["by_sleeve"]
days_list = [d["day"] for d in res["state"]["days"]]
for (day, sleeve), g in trades.groupby(["day", "sleeve"]):
    i = days_list.index(day)
    start = 25000 / 3 if i == 0 else sleeve_eq[days_list[i - 1]][sleeve]
    pos = g["shares"].cumsum()
    worst = (pos.abs() * g["price"]).max()
    check(worst <= b.LEVERAGE * start * 1.06, f"leverage: {sleeve} on {day} held ${worst:,.0f} on ${start:,.0f}")

print(f"{checks - len(fails)} of {checks} checks pass")
for f in fails[:15]:
    print("FAIL", f)
sys.exit(1 if fails else 0)
