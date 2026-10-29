"""Paper trading bot for the stock-only strategy (the tradable version, "Bot P" in run.py).

Paper money only: nothing here talks to a broker.

Once a day after the US market closes, for every trading day it has not processed yet, it:
  1. marks both paper books to that day's prices
  2. decides what to hold next, using only prices up to that close
  3. writes the ledger, trades, state and dashboard into paper/

Two books follow the same decisions:
  real      decides after the close, trades at the next morning's open, as a real bot would
  backtest  trades at the close the decision used, as the backtest assumed
The gap between them is part of step 4 of the doc's pass bar (PASS_BAR below).

Usage:
  python paper_bot.py start [--capital 10000]   begin a paper run from the last finished trading day
  python paper_bot.py run                       process every finished trading day since the last run
  python paper_bot.py status                    print a summary
  python paper_bot.py replay START END DIR      run the same engine over past days (for testing)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import exchange_calendars as xc
import numpy as np
import pandas as pd

import strategies as s

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper"
CACHE = ROOT / "data" / "paper_ohlc.pkl"
CAL = xc.get_calendar("XNYS", start="2000-01-01")  # the default only reaches back 20 years
COST = s.COSTS["etf"]
TICKERS = s.PAPER_TICKERS
SECTOR_NAMES = {"XLB": "Materials", "XLE": "Energy", "XLF": "Financials", "XLI": "Industrials",
                "XLK": "Technology", "XLP": "Consumer staples", "XLU": "Utilities", "XLV": "Health care",
                "XLY": "Consumer discretionary"}
TRIAL_SESSIONS = 63
GAP_LIMIT = 0.01
PASS_BAR = [
    "Every trading day was processed, and none used a price bar from before the market closed",
    "Every month's plan matched the backtest code rebuilt from scratch",
    "The real book (next-morning trades) ended within 1 percentage point of the backtest book (closing trades)",
]
SETTLE = timedelta(minutes=45)  # wait this long after the close before trusting the day's final prices

# selftest.py sets these to break a rule on purpose and prove its gates notice.
PEEK_LIVE = os.environ.get("PEEK_LIVE") == "1"      # month-end plans see the next day's prices
PARTIAL_BAR = os.environ.get("PARTIAL_BAR") == "1"  # treat today's bar as final before the close


# ---------- calendar ----------

def now_et() -> str:
    """Wall-clock time in New York, whatever time zone the process runs in."""
    return pd.Timestamp.now(tz="America/New_York").strftime("%Y-%m-%dT%H:%M:%S")



def D(x) -> pd.Timestamp:
    return pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize()


def next_session(d: pd.Timestamp) -> pd.Timestamp:
    return D(CAL.next_session(D(d)))


def month_sessions(d: pd.Timestamp) -> pd.DatetimeIndex:
    start = d.replace(day=1)
    return pd.DatetimeIndex([D(x) for x in CAL.sessions_in_range(start, start + pd.offsets.MonthEnd(0))])


def is_month_end(d: pd.Timestamp) -> bool:
    return next_session(d).month != d.month


def is_tom(d: pd.Timestamp) -> bool:
    ms = month_sessions(d)
    return d == ms[-1] or d in ms[:3]


def last_finished_session(now_utc: datetime) -> pd.Timestamp:
    """The latest trading day whose close was at least SETTLE ago. Never today's bar before the close."""
    now = pd.Timestamp(now_utc).tz_convert("UTC") if pd.Timestamp(now_utc).tzinfo else pd.Timestamp(now_utc, tz="UTC")
    settle = timedelta(hours=-24) if PARTIAL_BAR else SETTLE
    for sess in reversed(CAL.sessions_in_range(now.normalize().tz_localize(None) - timedelta(days=20),
                                               now.normalize().tz_localize(None))):
        if CAL.session_close(sess) + settle <= now:
            return D(sess)
    raise RuntimeError("no finished session in the last 20 days")


def tom_series(index: pd.DatetimeIndex) -> pd.Series:
    """Turn-of-month flags for every day in `index`, from the exchange calendar."""
    sessions = pd.DatetimeIndex([D(x) for x in CAL.sessions_in_range(index[0].replace(day=1),
                                                                      index[-1] + pd.offsets.MonthEnd(0))])
    return s.tom_mask_from_sessions(sessions).reindex(index).fillna(False)


# ---------- prices ----------

def fetch_prices(use_cache: bool = False) -> dict[str, pd.DataFrame]:
    """Adjusted close, adjusted open and raw close for every fund, daily since 2006."""
    if use_cache and CACHE.exists():
        return pd.read_pickle(CACHE)
    import yfinance as yf
    last_err = None
    for attempt in range(3):
        try:
            df = yf.download(TICKERS, start="2006-01-01", progress=False, auto_adjust=False, threads=True)
            close_raw, adj, open_raw = df["Close"][TICKERS], df["Adj Close"][TICKERS], df["Open"][TICKERS]
            break
        except Exception as e:  # network hiccups: try again, then give up for this run
            last_err = e
            time.sleep(20)
    else:
        raise RuntimeError(f"price download failed: {last_err}")
    factor = adj / close_raw
    prices = {"close": adj, "open": open_raw * factor, "raw_close": close_raw, "raw_open": open_raw}
    prices = {k: v.set_axis(pd.DatetimeIndex([D(x) for x in v.index])) for k, v in prices.items()}
    if use_cache:
        CACHE.parent.mkdir(exist_ok=True)
        pd.to_pickle(prices, CACHE)
    return prices


def complete_through(prices, day: pd.Timestamp, need_open: bool) -> bool:
    if day not in prices["close"].index:
        return False
    ok = prices["close"].loc[day].notna().all()
    return bool(ok and (not need_open or prices["open"].loc[day].notna().all()))


# ---------- the plan ----------

def make_plan(close: pd.DataFrame, t: pd.Timestamp) -> dict:
    """What to hold next month, decided at the close of `t`, the last trading day of a month.

    Uses the same functions as the backtest, fed only prices through `t`.
    """
    upto = next_session(t) if PEEK_LIVE else t
    data = close.loc[:upto].dropna(how="all")
    sleeves, _ = s.paper_sleeves(data, tom_series(data.index))
    sleeves = sleeves.dropna().loc[:t + pd.offsets.MonthEnd(0)]
    if len(sleeves) < 12:
        raise RuntimeError("fewer than 12 months of sleeve history")
    w = s.bot_weights(sleeves.iloc[-36:])
    spy_me = data["SPY"].resample("ME").last()
    avg10 = spy_me.rolling(10).mean()
    rm = data.resample("ME").last().pct_change()
    picks = s.sector_picks(rm[s.PAPER_SECTORS]).iloc[-1]
    r12 = np.expm1(np.log1p(rm[s.PAPER_SECTORS]).iloc[-12:].sum())
    nxt = next_session(t)
    return {
        "for_month": nxt.strftime("%Y-%m"),
        "decided_on": t.strftime("%Y-%m-%d"),
        "weights": {k: float(v) for k, v in w.items()},
        "cash": float(1 - w.sum()),
        "trend_on": bool(spy_me.iloc[-1] > avg10.iloc[-1]),
        "spy_month_end": float(spy_me.iloc[-1]),
        "spy_avg10": float(avg10.iloc[-1]),
        "sectors": [k for k, v in picks.items() if v > 0],
        "sector_12m": {k: float(v) for k, v in r12.sort_values(ascending=False).items()},
    }


def plan_for(state: dict, day: pd.Timestamp) -> dict:
    key = day.strftime("%Y-%m")
    for p in reversed(state["plans"]):
        if p["for_month"] == key:
            return p
    raise RuntimeError(f"no plan for {key}")


def trend_fund(plan: dict, day: pd.Timestamp) -> str:
    return "SPY" if plan["trend_on"] or is_tom(day) else "BIL"


def target_weights(plan: dict, trend: str) -> dict:
    w = plan["weights"]
    return {
        "trend": {trend: w["trend"]},
        "momentum": {tk: w["momentum"] / len(plan["sectors"]) for tk in plan["sectors"]},
        "defensive": {"USMV": w["defensive"] / 2, "QUAL": w["defensive"] / 2},
        "cash": {"BIL": plan["cash"]},
    }


# ---------- books ----------

def book_value(book: dict) -> float:
    return float(sum(v for sl in book.values() for v in sl.values()))


def by_fund(book: dict) -> dict:
    out: dict[str, float] = {}
    for sl in book.values():
        for tk, v in sl.items():
            out[tk] = out.get(tk, 0.0) + v
    return out


def mark(book: dict, ratio: pd.Series) -> dict:
    return {sl: {tk: v * (1.0 if tk == "USD" else float(ratio[tk])) for tk, v in h.items()} for sl, h in book.items()}


def trend_holding(book: dict) -> str | None:
    t = book.get("trend", {})
    return next(iter(t)) if t else None


def execute(book: dict, action: dict, when: str, day: pd.Timestamp, raw: pd.Series, name: str) -> tuple[dict, list]:
    """Apply a rebalance or a trend switch. Returns the new book and the trades."""
    old = by_fund(book)
    if action["kind"] == "rebalance":
        V = book_value(book)
        new = {sl: {tk: w * V for tk, w in h.items() if w > 0} for sl, h in action["targets"].items()}
        new_f = by_fund(new)
        turnover = sum(abs(new_f.get(tk, 0.0) - old.get(tk, 0.0)) for tk in set(new_f) | set(old) if tk != "USD")
        scale = (V - COST * turnover) / V
        new = {sl: {tk: v * scale for tk, v in h.items()} for sl, h in new.items()}
    else:  # switch the trend sleeve between SPY and BIL
        v = sum(book["trend"].values())
        new = copy.deepcopy(book)
        new["trend"] = {action["to"]: v - 2 * COST * v}
    new_f = by_fund(new)
    trades = []
    for tk in sorted(set(new_f) | set(old)):
        delta = new_f.get(tk, 0.0) - old.get(tk, 0.0)
        if tk == "USD" or abs(delta) < 0.005:
            continue
        price = float(raw[tk])
        trades.append({"date": day.strftime("%Y-%m-%d"), "when": when, "book": name, "fund": tk,
                       "action": "buy" if delta > 0 else "sell", "dollars": round(abs(delta), 2),
                       "price": round(price, 2), "shares": round(abs(delta) / price, 4)})
    return new, trades


def describe(action: dict | None) -> str:
    if action is None:
        return ""
    if action["kind"] == "rebalance":
        return action.get("why", "monthly rebalance")
    return "switch the trend part to " + ("the S&P 500" if action["to"] == "SPY" else "T-bills")


# ---------- one trading day ----------

def decide(state: dict, close: pd.DataFrame, t: pd.Timestamp) -> dict | None:
    """What to do at the close of `t` (backtest book) or the next open (real book)."""
    nxt = next_session(t)
    if is_month_end(t):
        plan = make_plan(close, t)
        state["plans"].append(plan)
        return {"kind": "rebalance", "targets": target_weights(plan, trend_fund(plan, nxt)),
                "why": f"monthly rebalance for {pd.Timestamp(plan['for_month']).strftime('%B')}"}
    want = trend_fund(plan_for(state, nxt), nxt)
    if want != trend_holding(state["books"]["backtest"]):
        return {"kind": "switch", "to": want}
    return None


def process_session(state: dict, prices: dict, t: pd.Timestamp, processed_at: str) -> None:
    # Everything below sees prices only through this close. PEEK_LIVE removes that cut on purpose.
    close, opn = (prices["close"] if PEEK_LIVE else prices["close"].loc[:t]), prices["open"].loc[:t]
    raw_close, raw_open = prices["raw_close"].loc[t], prices["raw_open"].loc[t]
    prev = D(state["last_session"])
    c_prev, c_t, o_t = close.loc[prev], close.loc[t], opn.loc[t]
    books, trades = state["books"], []

    # Real book: yesterday's decision fills at this morning's open.
    real = books["real"]
    if state["pending"]:
        real = mark(real, o_t / c_prev)
        real, tr = execute(real, state["pending"], "open", t, raw_open, "real")
        trades += tr
        real = mark(real, c_t / o_t)
        if state["bench_start"] is None:
            state["bench_start"] = float(o_t["SPY"])
    else:
        real = mark(real, c_t / c_prev)
    books["real"] = real
    books["backtest"] = mark(books["backtest"], c_t / c_prev)

    # Decide at this close. The backtest book trades now; the real book tomorrow morning.
    action = decide(state, close, t)
    if action is not None:
        books["backtest"], tr = execute(books["backtest"], action, "close", t, raw_close, "backtest")
        trades += tr
    state["pending"] = action

    bench = state["capital"] * float(c_t["SPY"]) / state["bench_start"] if state["bench_start"] else state["capital"]
    state["ledger"].append({"date": t.strftime("%Y-%m-%d"), "real": round(book_value(books["real"]), 2),
                            "backtest": round(book_value(books["backtest"]), 2), "spy": round(bench, 2),
                            "trend_fund": trend_holding(books["backtest"]), "action": describe(action),
                            "processed_at": processed_at})
    state["trades"] += trades
    state["last_session"] = t.strftime("%Y-%m-%d")


def new_state(prices: dict, t0: pd.Timestamp, capital: float) -> dict:
    """Start at the close of `t0` with cash; the backtest book buys at this close, the real book at the next open."""
    close = prices["close"].loc[:t0]
    prev_me = t0 if is_month_end(t0) else D(CAL.previous_session(month_sessions(t0)[0]))
    state = {"capital": capital, "start_session": t0.strftime("%Y-%m-%d"), "last_session": t0.strftime("%Y-%m-%d"),
             "created_at": now_et(), "bench_start": None,
             "books": {"real": {"start": {"USD": capital}}, "backtest": {"start": {"USD": capital}}},
             "pending": None, "plans": [make_plan(close, prev_me)], "trades": [], "ledger": [], "checks": {}}
    nxt = next_session(t0)
    plan = plan_for(state, nxt)
    action = {"kind": "rebalance", "targets": target_weights(plan, trend_fund(plan, nxt)), "why": "first buy"}
    state["books"]["backtest"], tr = execute(state["books"]["backtest"], action, "close", t0,
                                             prices["raw_close"].loc[t0], "backtest")
    state["trades"] += tr
    state["pending"] = action
    state["ledger"].append({"date": t0.strftime("%Y-%m-%d"), "real": capital,
                            "backtest": round(book_value(state["books"]["backtest"]), 2), "spy": capital,
                            "trend_fund": trend_holding(state["books"]["backtest"]), "action": "first buy",
                            "processed_at": now_et()})
    return state


def catch_up(state: dict, prices: dict, until: pd.Timestamp, log) -> int:
    done = 0
    day = next_session(D(state["last_session"]))
    while day <= until:
        if not complete_through(prices, day, need_open=True):
            log(f"prices for {day.date()} are not complete yet; will retry next run")
            break
        process_session(state, prices, day, now_et())
        done += 1
        day = next_session(day)
    return done


# ---------- checks (step 4) ----------

WEIGHT_TOLERANCE = 1e-5  # 0.001 percentage points, $0.10 on $10,000


def check_plans(state: dict, prices: dict) -> list[dict]:
    """Rebuild each month's plan with the backtest's own pipeline and compare.

    Sector picks and the trend switch must match exactly. Sleeve weights must match within
    WEIGHT_TOLERANCE: Yahoo returns dividend-adjusted prices that differ in the 7th decimal
    between downloads (seen 2026-09-29, two downloads two seconds apart), so a plan made from
    one download never matches another to the last digit. A logic error moves weights by far more.
    """
    last = D(state["last_session"])
    close = prices["close"].loc[:last]
    sleeves, bil_m = s.paper_sleeves(close, tom_series(close.index))
    held_w = s.build_bot(sleeves, bil_m).held
    rm = close.resample("ME").last().pct_change()
    held_sec = s.s5_sector(rm[s.PAPER_SECTORS]).held
    flags = s.trend_flags(close["SPY"].resample("ME").last()).shift(1)
    out = []
    for p in state["plans"]:
        m = pd.Timestamp(p["for_month"]) + pd.offsets.MonthEnd(0)
        if m not in held_w.index:
            out.append({"month": p["for_month"], "match": None, "note": "month not started yet"})
            continue
        w_diff = max(abs(float(held_w.loc[m, k]) - v) for k, v in p["weights"].items())
        w_ok = w_diff < WEIGHT_TOLERANCE
        sec_ok = sorted(held_sec.columns[held_sec.loc[m] > 0]) == sorted(p["sectors"])
        flag_ok = bool(flags.loc[m]) == p["trend_on"]
        out.append({"month": p["for_month"], "match": bool(w_ok and sec_ok and flag_ok),
                    "note": "" if (w_ok and sec_ok and flag_ok) else
                    f"weights {'ok' if w_ok else f'differ by up to {w_diff * 100:.5f} points'}, sectors {'ok' if sec_ok else 'differ'}, trend {'ok' if flag_ok else 'differs'}"})
    return out


def trial_status(state: dict) -> dict:
    start = D(state["start_session"])
    sessions = [D(x) for x in CAL.sessions_in_range(start, start + timedelta(days=140))]
    window = sessions[1:TRIAL_SESSIONS + 1]
    done = [D(r["date"]) for r in state["ledger"][1:]]
    last = state["ledger"][-1]
    gap = (last["real"] - last["backtest"]) / state["capital"]
    plans = state["checks"].get("plans", [])
    finished = len(done) >= TRIAL_SESSIONS
    crit = [
        all(d in done for d in window if d <= D(state["last_session"])),
        all(p["match"] is not False for p in plans) and any(p["match"] for p in plans),
        abs(gap) <= GAP_LIMIT,
    ]
    return {"day": len(done), "of": TRIAL_SESSIONS, "ends_on": window[-1].strftime("%Y-%m-%d"),
            "finished": finished, "criteria": crit, "gap": gap,
            "verdict": ("passes" if all(crit) else "fails") if finished else "in progress"}


# ---------- files ----------

def save(state: dict, folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    tmp = folder / "state.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, folder / "state.json")
    pd.DataFrame(state["ledger"]).to_csv(folder / "ledger.csv", index=False)
    pd.DataFrame(state["trades"]).to_csv(folder / "trades.csv", index=False)


def load(folder: Path) -> dict:
    return json.loads((folder / "state.json").read_text())


class Lock:
    """Refuse to run twice at once: two runs writing one ledger corrupts it."""

    def __init__(self, folder: Path):
        self.path = folder / "run.lock"

    def __enter__(self):
        if self.path.exists() and time.time() - self.path.stat().st_mtime < 1800:
            raise RuntimeError("another run is in progress (paper/run.lock is under 30 minutes old)")
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def logger(folder: Path):
    folder.mkdir(exist_ok=True)

    def log(msg: str):
        line = f"{now_et()} {msg}"
        with open(folder / "run.log", "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line)
    return log


# ---------- commands ----------

def cmd_start(capital: float, folder: Path = PAPER) -> None:
    """Start at the close of the last finished trading day. If Yahoo has not posted that day's
    final prices yet, record the start day and let the next scheduled run begin from it."""
    log = logger(folder)
    if (folder / "state.json").exists():
        raise RuntimeError("a paper run already exists; move paper/state.json aside to start over")
    t0 = last_finished_session(datetime.now(timezone.utc))
    (folder / "start.json").write_text(json.dumps({"capital": capital, "start_session": t0.strftime("%Y-%m-%d")}))
    if try_begin(folder, fetch_prices(), log) is None:
        log(f"waiting for Yahoo's final prices for {t0.date()}; the next run will start from that close")


def try_begin(folder: Path, prices: dict, log) -> dict | None:
    """Create the paper run from start.json once the start day's prices are complete."""
    spec = json.loads((folder / "start.json").read_text())
    t0 = D(spec["start_session"])
    if not complete_through(prices, t0, need_open=False):
        # If a later finished day is complete, Yahoo has skipped the start day: start there instead.
        later = [d for d in prices["close"].index
                 if t0 < d <= last_finished_session(datetime.now(timezone.utc))
                 and complete_through(prices, d, need_open=False)]
        if later:
            log(f"Yahoo never posted final prices for {t0.date()}; starting from {later[-1].date()} instead")
            spec["start_session"] = later[-1].strftime("%Y-%m-%d")
            (folder / "start.json").write_text(json.dumps(spec))
            return try_begin(folder, prices, log)
        from paper_dashboard import write_waiting
        write_waiting(folder, spec["start_session"], spec["capital"], now_et())
        return None
    with Lock(folder):
        state = new_state(prices, t0, spec["capital"])
        state["checks"]["plans"] = check_plans(state, prices)
        save(state, folder)
        render(state, folder)
    log(f"started with ${spec['capital']:,.0f} at the close of {t0.date()}")
    return state


def cmd_run(folder: Path = PAPER) -> int:
    log = logger(folder)
    if not (folder / "state.json").exists():
        if not (folder / "start.json").exists():
            raise RuntimeError("no paper run yet; run paper_bot.py start first")
        prices = fetch_prices()
        if try_begin(folder, prices, log) is None:
            log("still waiting for the start day's final prices")
            return 0
    with Lock(folder):
        state = load(folder)
        prices = fetch_prices()
        until = last_finished_session(datetime.now(timezone.utc))
        n = catch_up(state, prices, until, log)
        state["checks"]["plans"] = check_plans(state, prices)
        state["last_run"] = now_et()
        save(state, folder)
        render(state, folder)
        last = state["ledger"][-1]
        log(f"processed {n} day(s) through {state['last_session']}; real ${last['real']:,.2f}, "
            f"backtest ${last['backtest']:,.2f}, S&P ${last['spy']:,.2f}")
    return n


def cmd_status(folder: Path = PAPER) -> None:
    state = load(folder)
    last = state["ledger"][-1]
    ts = trial_status(state)
    print(f"Paper run since {state['start_session']}, day {ts['day']} of {ts['of']} (ends {ts['ends_on']}): {ts['verdict']}")
    print(f"Through {state['last_session']}: real ${last['real']:,.2f}, backtest ${last['backtest']:,.2f}, "
          f"S&P 500 ${last['spy']:,.2f}")
    print("Pending for next open:", describe(state["pending"]) or "nothing")
    for c, ok in zip(PASS_BAR, ts["criteria"]):
        print(f"  [{'ok' if ok else '--'}] {c}")


def cmd_replay(start: str, end: str, folder: Path, capital: float = 10000.0) -> dict:
    """Run the live engine over past days, as if run once per day, into `folder`."""
    prices = fetch_prices(use_cache=True)
    return replay_with(prices, D(start), D(end), folder, capital)


def replay_with(prices: dict, t0: pd.Timestamp, end: pd.Timestamp, folder: Path | None, capital: float = 10000.0) -> dict:
    state = new_state(prices, t0, capital)
    catch_up(state, prices, end, lambda m: None)
    state["checks"]["plans"] = check_plans(state, {k: v.loc[:end] for k, v in prices.items()})
    if folder is not None:
        save(state, folder)
        render(state, folder)
    return state


def render(state: dict, folder: Path) -> None:
    from paper_dashboard import write_dashboard
    write_dashboard(state, folder, trial_status(state), PASS_BAR, SECTOR_NAMES,
                    upcoming(state))


def upcoming(state: dict) -> list[dict]:
    """The next few scheduled moves, in plain words."""
    out = []
    day = D(state["last_session"])
    if state["pending"]:
        out.append({"date": next_session(day).strftime("%Y-%m-%d"),
                    "text": f"At the open: {describe(state['pending'])}"})
    probe = next_session(day)
    for _ in range(25):
        if is_month_end(probe):
            out.append({"date": probe.strftime("%Y-%m-%d"),
                        "text": "At the close: set next month's plan and rebalance"})
            break
        probe = next_session(probe)
    try:
        plan = plan_for(state, next_session(day))
        if not plan["trend_on"]:
            probe = next_session(day)
            for _ in range(25):
                nxt = next_session(probe)
                if is_tom(nxt) and not is_tom(probe):
                    out.append({"date": probe.strftime("%Y-%m-%d"),
                                "text": "At the close: buy the S&P 500 for the turn of the month"})
                    break
                probe = nxt
    except RuntimeError:
        pass
    return sorted(out, key=lambda x: x["date"])[:3]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Paper trading bot for the stock-only strategy")
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("--capital", type=float, default=10000.0)
    sub.add_parser("run")
    sub.add_parser("status")
    rp = sub.add_parser("replay")
    rp.add_argument("start")
    rp.add_argument("end")
    rp.add_argument("dir")
    a = ap.parse_args(argv)
    log = logger(PAPER)
    try:
        if a.cmd == "start":
            cmd_start(a.capital)
        elif a.cmd == "run":
            cmd_run()
        elif a.cmd == "status":
            cmd_status()
        elif a.cmd == "replay":
            st = cmd_replay(a.start, a.end, Path(a.dir))
            print(f"replayed {len(st['ledger']) - 1} days into {a.dir}")
        return 0
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
