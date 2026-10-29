"""The ten Tier 1 strategies from the "Top 50 Trading Strategies" doc, plus the combined bot.

Each strategy is a pure function of the data it is given, so selftest.py can feed it altered
data and prove that no position ever depends on prices from the period it is held in.

Every strategy returns a Run:
  monthly  DataFrame indexed by month end, columns gross and net (net = after COSTS)
  held     what it held, indexed by the end of the period it was held in; the decision
           for each period uses only data up to the end of the previous period
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from metrics import monthly_from

COSTS = {
    "etf": 0.0005,          # per side: spread and slippage on a liquid ETF at a zero-commission broker
    "crypto": 0.0025,       # per side: low-tier spot exchange fee plus spread
    "carry_switch": 0.003,  # per switch into or out of the spot + perpetual pair, both legs together
    "borrow": 0.005,        # per year, on the value of ETFs held short
    "fund_fee": 0.003,      # per year, charged to the stock-portfolio sleeves inside the bots
}

# selftest.py sets these to break a rule on purpose and prove its gates notice.
PEEK = os.environ.get("PEEK") == "1"
FREE_LUNCH = os.environ.get("FREE_LUNCH") == "1"


def _cost(x):
    return -x if FREE_LUNCH else x


@dataclass
class Run:
    monthly: pd.DataFrame
    held: pd.DataFrame | pd.Series | None


def _run(gross: pd.Series, net: pd.Series, held) -> Run:
    df = pd.DataFrame({"gross": gross, "net": net}).dropna(how="all")
    return Run(df, held)


# ---------- 1 Trend following across markets (ETF stand-ins for futures) ----------

def s1_trend(px: pd.DataFrame, rf_m: pd.Series, long_short: bool = True, target: float = 0.10) -> Run:
    """Each market: long if its 12-month return beat cash, short (or flat) if not.

    Positions are sized by inverse volatility and scaled so the whole book aims for `target`
    yearly volatility, using the past year of daily returns. Gross exposure is capped at 2x
    for long/short and 1x (no borrowing) for long/flat.
    """
    pm = px.resample("ME").last()
    rm = pm.pct_change()
    rd = px.pct_change()
    rf12 = np.exp(np.log1p(rf_m).rolling(12).sum()) - 1
    ex12 = (pm / pm.shift(12) - 1).sub(rf12.reindex(pm.index), axis=0)
    cap = 2.0 if long_short else 1.0

    rows = {}
    for t in pm.index:
        sig = np.sign(ex12.loc[t])
        if sig.isna().any():
            continue  # wait until every market has a full year of history
        if not long_short:
            sig = sig.clip(lower=0)
        hist = rd.loc[:t].iloc[-252:]
        raw = sig / (hist.std() * np.sqrt(252))
        if (raw != 0).sum() == 0:
            rows[t] = raw * 0.0
            continue
        cov = hist.cov() * 252
        pv = float(np.sqrt(raw @ cov @ raw))
        w = raw * target / pv
        gross_exp = w.abs().sum()
        if gross_exp > cap:
            w = w * cap / gross_exp
        rows[t] = w
    decided = pd.DataFrame(rows).T
    held = decided.shift(1).iloc[1:]

    rf = rf_m.reindex(held.index)
    ex = rm.reindex(held.index).sub(rf, axis=0)
    gross = rf + (held * ex).sum(axis=1)
    turnover = (held - held.shift(1).fillna(0)).abs().sum(axis=1)
    borrow = held.clip(upper=0).abs().sum(axis=1) * COSTS["borrow"] / 12
    net = gross - _cost(turnover * COSTS["etf"] + borrow)
    return _run(gross, net, held)


# ---------- 2 Index trend filter ----------

def s2_index_trend(mkt_m: pd.Series, rf_m: pd.Series) -> Run:
    """Hold the stock market while it closed above its 10-month average last month-end, else T-bills."""
    level = (1 + mkt_m).cumprod()
    sma = level.rolling(10).mean()
    decided = (level > sma).astype(float).where(sma.notna())
    held = (decided if PEEK else decided.shift(1)).dropna()
    mkt, rf = mkt_m.reindex(held.index), rf_m.reindex(held.index)
    gross = held * mkt + (1 - held) * rf
    # A switch sells one fund and buys the other: two sides. The first month is one side in.
    turnover = 2 * held.diff().abs()
    turnover.iloc[0] = 1.0
    net = gross - _cost(turnover * COSTS["etf"])
    return _run(gross, net, held)


def s2_with_tom(mkt_m: pd.Series, mkt_d: pd.Series, rf_d: pd.Series) -> Run:
    """#2 with #8 layered on: when the trend filter says cash, still hold stocks at the turn of the month."""
    level = (1 + mkt_m).cumprod()
    sma = level.rolling(10).mean()
    decided = (level > sma).astype(float).where(sma.notna())
    month_flag = decided.shift(1)  # held in month t, decided at the end of month t-1
    flag_d = pd.Series(month_flag.reindex(mkt_d.index.to_period("M").to_timestamp(how="end").normalize()).values,
                       index=mkt_d.index)
    held = np.maximum(flag_d, _tom_mask(mkt_d.index).astype(float)).where(flag_d.notna()).dropna()
    mkt, rf = mkt_d.reindex(held.index), rf_d.reindex(held.index)
    gross_d = held * mkt + (1 - held) * rf
    # Cash sits in the broker's interest-paying sweep, so each switch is one side.
    switch = held.diff().abs().fillna(1.0)
    net_d = gross_d - _cost(switch * COSTS["etf"])
    return _run(monthly_from(gross_d), monthly_from(net_d), held)


# ---------- 3 and 4 Momentum ----------

def s3_momentum(mom_m: pd.DataFrame) -> Run:
    """Top decile of US stocks by 12-month return skipping the last month (Ken French's portfolio).

    French rebuilds the portfolio monthly; its trading cost cannot be computed from returns alone,
    so net is left empty and run.py reports the break-even cost instead.
    """
    hi = mom_m["Hi PRIOR"]
    return _run(hi, pd.Series(np.nan, index=hi.index), None)


def _realized_vol(daily: pd.Series, days: int = 126) -> pd.Series:
    """Annualized volatility over the past `days`, sampled at each month end."""
    var = (daily ** 2).rolling(days).sum() * (252 / days)
    return np.sqrt(var).resample("ME").last()


def s4_wml_rebuild(mom_m: pd.DataFrame, mom_d: pd.DataFrame, target: float = 0.12) -> tuple[Run, Run]:
    """Barroso & Santa-Clara's test: winners minus losers, plain and scaled to 12% volatility."""
    wml_m = mom_m["Hi PRIOR"] - mom_m["Lo PRIOR"]
    wml_d = mom_d["Hi PRIOR"] - mom_d["Lo PRIOR"]
    scale = (target / _realized_vol(wml_d)).shift(1).reindex(wml_m.index).dropna()
    scaled = scale * wml_m.reindex(scale.index)
    plain = wml_m.reindex(scale.index)
    return _run(plain, plain, None), _run(scaled, scaled, scale)


def s4_scaled_momentum(mom_m: pd.DataFrame, mom_d: pd.DataFrame, rf_m: pd.Series, target: float = 0.15) -> Run:
    """Long-only #3, with exposure cut to target/realized volatility (never above 100%); the rest in T-bills."""
    hi_m = mom_m["Hi PRIOR"]
    exposure = (target / _realized_vol(mom_d["Hi PRIOR"])).clip(upper=1.0)
    held = exposure.shift(1).reindex(hi_m.index).dropna()
    rf = rf_m.reindex(held.index)
    gross = held * hi_m.reindex(held.index) + (1 - held) * rf
    return _run(gross, pd.Series(np.nan, index=gross.index), held)


# ---------- 5 Sector momentum ----------

def sector_picks(rets_m: pd.DataFrame, top: int = 3, lookback: int = 12) -> pd.DataFrame:
    """Weights decided at each month end: 1/top on the `top` best `lookback`-month returns."""
    logs = np.log1p(rets_m)
    ready = logs.rolling(lookback).count().eq(lookback).all(axis=1)
    mom = logs.rolling(lookback).sum()[ready]
    ranks = mom.rank(axis=1, ascending=False, method="first")
    return (ranks <= top).astype(float) / top


def s5_sector(rets_m: pd.DataFrame, top: int = 3, lookback: int = 12) -> Run:
    """Hold the `top` sectors with the best `lookback`-month return, equal weight, monthly."""
    decided = sector_picks(rets_m, top, lookback)
    held = decided.shift(1).iloc[1:]
    gross = (held * rets_m.reindex(held.index)).sum(axis=1)
    turnover = (held - held.shift(1).fillna(0)).abs().sum(axis=1)
    net = gross - _cost(turnover * COSTS["etf"])
    return _run(gross, net, held)


# ---------- 6 and 7 Low volatility, quality ----------

def s6_low_vol(var_m: pd.DataFrame) -> Run:
    """Lowest-variance fifth of US stocks, value-weighted, rebuilt monthly by Ken French."""
    lo = var_m["Lo 20"]
    return _run(lo, pd.Series(np.nan, index=lo.index), None)


def s7_quality(op_m: pd.DataFrame) -> Run:
    """Most profitable fifth of US stocks by operating profitability, value-weighted, rebuilt each June.

    French publishes operating profitability, not the doc's gross profitability; this is the stand-in.
    """
    hi = op_m["Hi 20"]
    return _run(hi, pd.Series(np.nan, index=hi.index), None)


# ---------- 8 Turn of the month ----------

def _tom_mask(idx: pd.DatetimeIndex) -> pd.Series:
    """True on the last trading day of each month and the first three of the next."""
    per = idx.to_period("M")
    ones = pd.Series(1, index=idx)
    from_start = ones.groupby(per).cumsum()
    from_end = ones.groupby(per).transform("sum") - from_start + 1
    return (from_end == 1) | (from_start <= 3)


def s8_turn_of_month(mkt_d: pd.Series, rf_d: pd.Series) -> Run:
    """Hold stocks from the close two days before month-end to the close of the third trading day."""
    held = _tom_mask(mkt_d.index).astype(float)
    gross_d = held * mkt_d + (1 - held) * rf_d
    switch = held.diff().abs().fillna(held.iloc[0])
    net_d = gross_d - _cost(switch * COSTS["etf"])
    return _run(monthly_from(gross_d), monthly_from(net_d), held)


# ---------- 9 Crypto trend ----------

def s9_crypto_trend(closes: pd.Series) -> tuple[Run, pd.Series]:
    """Hold the coin for a week when last week's return was positive; otherwise hold dollars.

    Returns the run and the weekly buy-and-hold returns for comparison.
    """
    wk = closes.resample("W-SUN").last()
    wk = wk.loc[: closes.index[-1]]  # drop a final week that has not finished yet
    wr = wk.pct_change()
    decided = (wr > 0).astype(float).where(wr.notna())
    held = decided.shift(1).dropna()
    r = held * wr.reindex(held.index)
    switch = held.diff().abs()
    switch.iloc[0] = held.iloc[0]
    net = r - _cost(switch * COSTS["crypto"])
    run = Run(pd.DataFrame({"gross": monthly_from(r), "net": monthly_from(net)}), held)
    run.weekly = pd.DataFrame({"gross": r, "net": net})
    return run, wr.reindex(held.index)


def crypto_pair(btc: Run, eth: Run) -> pd.Series:
    """Half BTC trend, half ETH trend, rebalanced weekly; BTC alone before ETH data starts."""
    w = pd.concat([btc.weekly["net"], eth.weekly["net"]], axis=1, keys=["btc", "eth"])
    both = w.dropna()
    pair = w["btc"].copy()
    pair.loc[both.index] = both.mean(axis=1)
    return monthly_from(pair)


# ---------- 10 Crypto cash-and-carry ----------

def s10_carry(funding: pd.Series, always_on: bool = False) -> Run:
    """Own BTC and short the same dollar amount of the perpetual; collect funding.

    Doc rule: hold the pair only while the last funding payment was positive. On BitMEX's
    inverse contract the BTC you own is the collateral, so return on capital = funding received.
    """
    if always_on:
        held = pd.Series(1.0, index=funding.index)
    else:
        held = (funding.shift(1) > 0).astype(float).iloc[1:]
    income = held * funding.reindex(held.index)
    switch = held.diff().abs()
    switch.iloc[0] = held.iloc[0]
    net = income - _cost(switch * COSTS["carry_switch"])
    return _run(monthly_from(income), monthly_from(net), held)


# ---------- The tradable stock-only bot (what the paper bot runs) ----------

PAPER_SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
PAPER_TICKERS = ["SPY", "BIL"] + PAPER_SECTORS + ["USMV", "QUAL"]


def trend_flags(month_end_level: pd.Series) -> pd.Series:
    """1.0 where the month-end level closed above its 10-month average, decided at that month end."""
    sma = month_end_level.rolling(10).mean()
    return (month_end_level > sma).astype(float).where(sma.notna())


def tom_mask_from_sessions(sessions: pd.DatetimeIndex) -> pd.Series:
    """Turn-of-month days from the exchange calendar, so a month still in progress is marked right."""
    return _tom_mask(pd.DatetimeIndex(sessions))


def s2_tom_etf(spy_d: pd.Series, bil_d: pd.Series, tom: pd.Series | None = None) -> Run:
    """#2 with #8 on real funds: SPY while above its 10-month average or at the turn of the month, else BIL.

    spy_d and bil_d are daily closes with dividends reinvested. A switch sells one fund and buys
    the other, so it costs two sides.
    """
    px = pd.concat([spy_d, bil_d], axis=1, keys=["SPY", "BIL"]).dropna()
    flags = trend_flags(px["SPY"].resample("ME").last())
    month_flag = flags.shift(1)
    month_key = px.index.to_period("M").to_timestamp(how="end").normalize()
    flag_d = pd.Series(month_flag.reindex(month_key).values, index=px.index)
    tom_d = (tom.reindex(px.index).fillna(False) if tom is not None else _tom_mask(px.index)).astype(float)
    held = np.maximum(flag_d, tom_d).where(flag_d.notna()).dropna()
    r = px.pct_change().reindex(held.index)
    gross_d = held * r["SPY"] + (1 - held) * r["BIL"]
    switch = held.diff().abs().fillna(1.0)
    net_d = gross_d - _cost(2 * switch * COSTS["etf"])
    return _run(monthly_from(gross_d), monthly_from(net_d), held)


def paper_sleeves(px: pd.DataFrame, tom: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly returns of the three sleeves the paper bot trades, plus BIL's monthly return for cash."""
    pm = px.resample("ME").last()
    rm = pm.pct_change()
    sleeves = pd.DataFrame({
        "trend": s2_tom_etf(px["SPY"], px["BIL"], tom).monthly.net,
        "momentum": s5_sector(rm[PAPER_SECTORS]).monthly.net,
        "defensive": 0.5 * rm["USMV"] + 0.5 * rm["QUAL"],
    })
    return sleeves, rm["BIL"]


# ---------- The bot ----------

def bot_weights(hist: pd.DataFrame, target: float = 0.10, cap: float = 1.0) -> pd.Series:
    """Equal-risk sleeve weights from past monthly sleeve returns, scaled toward `target` volatility."""
    inv = 1 / (hist.std() * np.sqrt(12))
    w = inv / inv.sum()
    pv = float(np.sqrt(w @ (hist.cov() * 12) @ w))
    return w * min(cap, target / pv)


def build_bot(sleeves: pd.DataFrame, rf_m: pd.Series, target: float = 0.10, cap: float = 1.0,
              lookback: int = 36, min_obs: int = 12) -> Run:
    """Equal risk per sleeve, whole bot scaled toward `target` yearly volatility, never borrowing.

    Weights for month t use only sleeve returns through month t-1. Unused money earns T-bills.
    """
    sleeves = sleeves.dropna()
    rows = {}
    for i in range(min_obs, len(sleeves)):
        t = sleeves.index[i]
        rows[t] = bot_weights(sleeves.iloc[max(0, i - lookback):i], target, cap)
    held = pd.DataFrame(rows).T
    rf = rf_m.reindex(held.index)
    invested = held.sum(axis=1)
    gross = (held * sleeves.reindex(held.index)).sum(axis=1) + (1 - invested) * rf
    turnover = (held - held.shift(1).fillna(0)).abs().sum(axis=1)
    net = gross - _cost(turnover * COSTS["etf"])
    return _run(gross, net, held)
