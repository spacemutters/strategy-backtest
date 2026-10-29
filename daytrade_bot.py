"""High-risk day-trading paper bot. No broker, no real money.

Three published intraday rules, each with a third of the paper account and up to 4x
leverage on its third, so the whole account never goes past 4x (the most a US
pattern-day-trader margin account allows):

  ORB   QQQ 5-minute opening range breakout (Zarattini & Aziz 2023). Trade the direction
        of the first 5-minute bar from the next bar's open, stop at the far side of that
        bar, target 10x the risk, flat at the close. Risks 1% of the sleeve per trade.
  VWAP  QQQ VWAP trend (Zarattini & Aziz 2023). Long while the last bar closed above the
        day's VWAP, short while below, flipping on every cross. Flat at the close.
  MOM   SPY intraday momentum (Gao, Han, Li & Zhou 2018). The return from yesterday's
        close to 10:00 sets the side for the last half hour (15:30 to the close).

How it runs: after each close it downloads that day's 5-minute bars from Yahoo and plays
the day forward one bar at a time. Every decision sees only bars that had finished, and
fills at the next bar's open with costs. `selftest_daytrade.py` proves that by planting a
look-ahead bug and checking the gate catches it.

Commands:
  daytrade_bot.py start [capital]   begin the trial at the last finished session
  daytrade_bot.py run               process every finished session since the last run
  daytrade_bot.py status            one-screen summary
  daytrade_bot.py history           replay the last ~59 days Yahoo still has (not the trial)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "daytrade"
CACHE = ROOT / "data" / "intraday"
NY = ZoneInfo("America/New_York")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) strategy-backtest"}
TICKERS = ["QQQ", "SPY"]
LEVERAGE = 4.0
SLEEVES = ["ORB", "VWAP", "MOM"]
SLIPPAGE = 0.0001          # 1 basis point per side
COMMISSION = 0.0035        # dollars per share, minimum 0.35 per order (a typical per-share broker rate)
COMMISSION_MIN = 0.35
ORB_RISK = 0.01
ORB_TARGET_R = 10.0
SETTLE = timedelta(minutes=45)
TRIAL_SESSIONS = 63

# Planted bug for the self-test: decisions may see the bar they trade on.
PEEK = os.environ.get("PEEK") == "1"


# ---------- data ----------

def fetch(ticker: str) -> pd.DataFrame:
    """5-minute regular-session bars, merged into a local cache so history outlives Yahoo's 60 days."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{ticker}_5m.pkl"
    old = pd.read_pickle(path) if path.exists() else None
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"interval": "5m", "range": "60d", "includePrePost": "false"},
                     headers=UA, timeout=60)
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ["open", "high", "low", "close", "volume"]},
                      index=pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(NY))
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df.index.time >= datetime.strptime("09:30", "%H:%M").time())
            & (df.index.time <= datetime.strptime("15:55", "%H:%M").time())]
    if old is not None:
        df = pd.concat([old[~old.index.isin(df.index)], df]).sort_index()
    df.to_pickle(path)
    return df


def sessions(bars: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    days = None
    for df in bars.values():
        d = set(pd.Index(df.index.date).unique())
        days = d if days is None else days & d
    return sorted(pd.Timestamp(x) for x in days)


def day_bars(df: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return df[df.index.date == day.date()]


def settled(day: pd.Timestamp, now: datetime) -> bool:
    close = datetime.combine(day.date(), datetime.strptime("16:00", "%H:%M").time(), NY)
    return now >= close + SETTLE


def complete(bars: dict, day: pd.Timestamp, now: datetime) -> bool:
    """A session counts only once its 15:55 bar exists for every ticker and the close has settled."""
    if not settled(day, now):
        return False
    for df in bars.values():
        d = day_bars(df, day)
        if len(d) < 70 or d.index[-1].time() != datetime.strptime("15:55", "%H:%M").time():
            return False  # half days and gaps are skipped rather than guessed at
    return True


# ---------- execution ----------

def cost(shares: float, price: float) -> float:
    return abs(shares) * price * SLIPPAGE + max(COMMISSION_MIN, abs(shares) * COMMISSION)


class Sleeve:
    def __init__(self, name: str, cash: float):
        self.name, self.cash, self.shares, self.ticker = name, cash, 0.0, None
        self.trades: list[dict] = []
        self.entry = None

    def value(self, price: float) -> float:
        return self.cash + self.shares * price

    def target(self, ticker: str, shares: float, price: float, when: pd.Timestamp, why: str) -> None:
        """Move to `shares` of `ticker` at `price`, paying costs. Positions never cross tickers."""
        if self.shares and ticker != self.ticker:
            raise ValueError("a sleeve trades one ticker per day")
        delta = shares - self.shares
        if abs(delta) < 1e-9:
            return
        c = cost(delta, price)
        self.cash -= delta * price + c
        self.trades.append({"time": when.isoformat(), "sleeve": self.name, "ticker": ticker,
                            "side": "buy" if delta > 0 else "sell", "shares": round(delta, 4),
                            "price": round(price, 4), "cost": round(c, 2), "why": why})
        self.shares, self.ticker = shares, (ticker if shares else None)


def max_shares(equity: float, price: float) -> float:
    return int(LEVERAGE * equity / price)


def run_day(bars: dict, day: pd.Timestamp, prev_close: dict, equity: dict) -> dict:
    """Play one session forward bar by bar. Returns new sleeve equities, trades and a day record."""
    out_trades, result = [], {}
    q = day_bars(bars["QQQ"], day)
    s = day_bars(bars["SPY"], day)

    # ORB on QQQ
    sl = Sleeve("ORB", equity["ORB"])
    first = q.iloc[0]
    direction = 1 if first.close > first.open else (-1 if first.close < first.open else 0)
    if direction and len(q) > 1:
        entry_i = 1
        px = q.iloc[entry_i].open
        stop = first.low if direction > 0 else first.high
        risk = abs(px - stop)
        if risk > 0:
            n = min(ORB_RISK * sl.cash / risk, max_shares(sl.cash, px))
            n = int(n) * direction
            if n:
                sl.target("QQQ", n, px, q.index[entry_i], "opening range breakout")
                tgt = px + direction * ORB_TARGET_R * risk
                done = False
                for i in range(entry_i, len(q)):
                    bar = q.iloc[i]
                    hit_stop = bar.low <= stop if direction > 0 else bar.high >= stop
                    hit_tgt = bar.high >= tgt if direction > 0 else bar.low <= tgt
                    if hit_stop:  # if both happen inside one bar, assume the stop came first
                        fill = min(bar.open, stop) if direction > 0 else max(bar.open, stop)
                        sl.target("QQQ", 0, fill, q.index[i], "stop")
                        done = True
                        break
                    if hit_tgt:
                        fill = max(bar.open, tgt) if direction > 0 else min(bar.open, tgt)
                        sl.target("QQQ", 0, fill, q.index[i], "target")
                        done = True
                        break
                if not done:
                    sl.target("QQQ", 0, q.iloc[-1].close, q.index[-1], "flat at the close")
    result["ORB"] = sl.cash
    out_trades += sl.trades

    # VWAP trend on QQQ
    sl = Sleeve("VWAP", equity["VWAP"])
    typical = (q.high + q.low + q.close) / 3
    vwap = (typical * q.volume).cumsum() / q.volume.cumsum().replace(0, float("nan"))
    side = 0
    for i in range(1, len(q)):
        seen = i if PEEK else i - 1          # the decision may only use the bar that just finished
        c, v = q.iloc[seen].close, vwap.iloc[seen]
        want = 1 if c > v else (-1 if c < v else side)
        px = q.iloc[i].open
        if want != side:
            n = max_shares(sl.value(px), px) * want
            sl.target("QQQ", n, px, q.index[i], "above VWAP" if want > 0 else "below VWAP")
            side = want
    sl.target("QQQ", 0, q.iloc[-1].close, q.index[-1], "flat at the close")
    result["VWAP"] = sl.cash
    out_trades += sl.trades

    # Intraday momentum on SPY
    sl = Sleeve("MOM", equity["MOM"])
    at10 = s[s.index.time < datetime.strptime("10:00", "%H:%M").time()]
    last = s[s.index.time >= datetime.strptime("15:30", "%H:%M").time()]
    pc = prev_close.get("SPY")
    if pc and len(at10) and len(last):
        r1 = at10.iloc[-1].close / pc - 1
        want = 1 if r1 > 0 else (-1 if r1 < 0 else 0)
        if want:
            px = last.iloc[0].open
            sl.target("SPY", max_shares(sl.cash, px) * want, px, last.index[0],
                      f"first half hour {r1:+.2%}")
            sl.target("SPY", 0, last.iloc[-1].close, last.index[-1], "flat at the close")
    result["MOM"] = sl.cash
    out_trades += sl.trades

    return {"equity": result, "trades": out_trades,
            "closes": {"QQQ": float(q.iloc[-1].close), "SPY": float(s.iloc[-1].close)}}


# ---------- state ----------

def new_state(capital: float, start_day: pd.Timestamp, closes: dict) -> dict:
    third = capital / 3
    return {"capital": capital, "start": str(start_day.date()), "last": str(start_day.date()),
            "equity": {k: third for k in SLEEVES}, "prev_close": closes,
            "spy_start": closes["SPY"], "days": [], "trades": 0}


def process(state: dict, bars: dict, now: datetime, log) -> int:
    n = 0
    for day in sessions(bars):
        if str(day.date()) <= state["last"]:
            continue
        if not settled(day, now):
            log(f"{day.date()} hasn't settled yet; will retry next run")
            break
        if not complete(bars, day, now):
            log(f"{day.date()} skipped: short session or missing bars")
            state["last"] = str(day.date())
            for t in TICKERS:  # keep yesterday's close current for the momentum rule
                d = day_bars(bars[t], day)
                if len(d):
                    state["prev_close"][t] = float(d.iloc[-1].close)
            continue
        r = run_day(bars, day, state["prev_close"], state["equity"])
        before = sum(state["equity"].values())
        state["equity"] = r["equity"]
        after = sum(r["equity"].values())
        state["prev_close"] = r["closes"]
        state["last"] = str(day.date())
        state["trades"] += len(r["trades"])
        state["days"].append({"day": str(day.date()), "equity": round(after, 2),
                              "pnl": round(after - before, 2), "trades": len(r["trades"]),
                              "by_sleeve": {k: round(v, 2) for k, v in r["equity"].items()},
                              "spy_hold": round(state["capital"] * r["closes"]["SPY"] / state["spy_start"], 2)})
        append_trades(r["trades"])
        n += 1
        log(f"{day.date()}: {len(r['trades'])} trades, account ${after:,.2f} ({after - before:+,.2f})")
    return n


def append_trades(trades: list[dict]) -> None:
    if not trades:
        return
    path = OUT / "trades.csv"
    df = pd.DataFrame(trades)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def save(state: dict) -> None:
    OUT.mkdir(exist_ok=True)
    tmp = OUT / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(OUT / "state.json")


def load() -> dict:
    return json.loads((OUT / "state.json").read_text())


def logger():
    OUT.mkdir(exist_ok=True)
    f = OUT / "run.log"

    def log(msg: str) -> None:
        line = f"{datetime.now(NY).strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line)
        with f.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


# ---------- report ----------

def stats(days: list[dict], capital: float) -> dict:
    if not days:
        return {}
    eq = pd.Series([capital] + [d["equity"] for d in days])
    rets = eq.pct_change().dropna()
    peak = eq.cummax()
    return {"sessions": len(days), "final": round(float(eq.iloc[-1]), 2),
            "return": float(eq.iloc[-1] / capital - 1),
            "worst_day": float(rets.min()), "best_day": float(rets.max()),
            "max_drawdown": float((eq / peak - 1).min()),
            "win_days": float((rets > 0).mean()),
            "trades": int(sum(d["trades"] for d in days)),
            "spy_hold": days[-1]["spy_hold"]}


def render(state: dict) -> None:
    from daytrade_dashboard import write
    write(state, stats(state["days"], state["capital"]), OUT / "dashboard.html")


# ---------- commands ----------

def all_bars() -> dict:
    return {t: fetch(t) for t in TICKERS}


def cmd_start(capital: float) -> None:
    log = logger()
    if (OUT / "state.json").exists():
        log("already started; move daytrade/state.json aside to start over")
        return
    bars = all_bars()
    now = datetime.now(NY)
    done = [d for d in sessions(bars) if complete(bars, d, now)]
    d0 = done[-1]
    closes = {t: float(day_bars(bars[t], d0).iloc[-1].close) for t in TICKERS}
    state = new_state(capital, d0, closes)
    save(state)
    render(state)
    log(f"started with ${capital:,.0f} at the close of {d0.date()}; first trading day is the next session")


def cmd_run() -> int:
    log = logger()
    lock = OUT / "run.lock"
    if lock.exists() and datetime.now().timestamp() - lock.stat().st_mtime < 1800:
        log("another run is in progress")
        return 1
    OUT.mkdir(exist_ok=True)
    lock.write_text(str(os.getpid()))
    try:
        state = load()
        bars = all_bars()
        n = process(state, bars, datetime.now(NY), log)
        save(state)
        render(state)
        st = stats(state["days"], state["capital"])
        log(f"processed {n} session(s); account ${st.get('final', state['capital']):,.2f}")
        return 0
    finally:
        lock.unlink(missing_ok=True)


def cmd_status() -> None:
    state = load()
    st = stats(state["days"], state["capital"])
    print(f"Day-trading paper bot, started {state['start']} with ${state['capital']:,.0f}")
    if not st:
        print("No finished trading day yet.")
        return
    print(f"{st['sessions']} of {TRIAL_SESSIONS} sessions, {st['trades']} trades")
    print(f"Account ${st['final']:,.2f} ({st['return']:+.2%}); holding SPY instead: ${st['spy_hold']:,.2f}")
    print(f"Worst day {st['worst_day']:+.2%}, worst drop from a high {st['max_drawdown']:+.2%}")
    for k, v in state["equity"].items():
        print(f"  {k}: ${v:,.2f}")


def replay(bars: dict, capital: float = 25000.0) -> dict:
    days = sessions(bars)
    now = datetime.now(NY)
    d0 = days[0]
    closes = {t: float(day_bars(bars[t], d0).iloc[-1].close) for t in TICKERS}
    state = new_state(capital, d0, closes)
    trades = []
    for day in days[1:]:
        if not complete(bars, day, now):
            continue
        r = run_day(bars, day, state["prev_close"], state["equity"])
        state["equity"], state["prev_close"] = r["equity"], r["closes"]
        trades += r["trades"]
        state["days"].append({"day": str(day.date()), "equity": round(sum(r["equity"].values()), 2),
                              "pnl": 0, "trades": len(r["trades"]),
                              "by_sleeve": {k: round(v, 2) for k, v in r["equity"].items()},
                              "spy_hold": round(capital * r["closes"]["SPY"] / state["spy_start"], 2)})
    return {"state": state, "trades": trades, "stats": stats(state["days"], capital)}


def cmd_history() -> None:
    res = replay(all_bars())
    st, state = res["stats"], res["state"]
    print(f"Replay of {st['sessions']} sessions ({state['days'][0]['day']} to {state['days'][-1]['day']}), "
          f"{st['trades']} trades, $25,000 start")
    print(f"Account ${st['final']:,.2f} ({st['return']:+.2%}) vs holding SPY ${st['spy_hold']:,.2f}")
    print(f"Worst day {st['worst_day']:+.2%}, best day {st['best_day']:+.2%}, "
          f"worst drop {st['max_drawdown']:+.2%}, up days {st['win_days']:.0%}")
    for k, v in state["equity"].items():
        print(f"  {k}: ${v:,.2f} from ${25000 / 3:,.2f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "history.json").write_text(json.dumps({"stats": st, "days": state["days"]}, indent=1))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]
    if cmd == "start":
        cmd_start(float(argv[1]) if len(argv) > 1 else 25000.0)
    elif cmd == "run":
        return cmd_run()
    elif cmd == "status":
        cmd_status()
    elif cmd == "history":
        cmd_history()
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
