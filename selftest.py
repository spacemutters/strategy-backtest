"""Gates every backtest number depends on. Run: python selftest.py

1. No peeking. Every input is altered after a cutoff date. What each strategy holds in the
   first period after the cutoff was decided at the cutoff, so it must not change.
2. Costs only cost. Net return is never above gross, and some cost is actually charged.
3. Data spot checks against known historical values.

Prove the gates fire by breaking the rules on purpose (each must print FAIL and exit 1):
  PEEK=1 python selftest.py        # strategy #2 decides each month using that month's own return
  FREE_LUNCH=1 python selftest.py  # every cost is paid to you instead of by you
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import data as d
import strategies as s

rng = np.random.default_rng(7)
failures: list[str] = []
passes = 0


def alter(x, cut, kind):
    """Scramble everything strictly after `cut` hard enough to flip buy/sell signals.

    Prices get multiplied by large random factors; returns get their sign flipped and noise added.
    A gentle nudge is not enough: a first version of this gate used 3% noise at two dates and
    missed a deliberately planted peek, because no signal happened to flip.
    """
    x = x.copy()
    mask = x.index > pd.Timestamp(cut)
    shape = x.loc[mask].shape
    if kind == "price":
        x.loc[mask] = x.loc[mask] * np.exp(rng.normal(0, 0.25, size=shape))
    else:
        x.loc[mask] = -x.loc[mask] + rng.normal(0, 0.10, size=shape)
    return x


def cuts_between(start, end, n, freq="ME"):
    dates = pd.date_range(start, end, freq=freq)
    return [str(t.date()) for t in dates[np.linspace(0, len(dates) - 1, n).astype(int)]]


def same_through_next(name, held_a, held_b, cut, must_change_later=True):
    global passes
    nxt = held_a.index[held_a.index > pd.Timestamp(cut)]
    if len(nxt) == 0:
        failures.append(f"{name}: nothing held after {cut}, test is empty")
        return
    upto = nxt[0]
    a, b = held_a.loc[:upto], held_b.loc[:upto]
    if a.shape != b.shape or not np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True):
        failures.append(f"{name}: holdings through {upto.date()} changed when only data after {cut} changed")
        return
    if must_change_later:
        la, lb = held_a.loc[upto:].iloc[1:], held_b.loc[upto:].iloc[1:]
        common = la.index.intersection(lb.index)
        if np.allclose(np.asarray(la.loc[common], float), np.asarray(lb.loc[common], float), equal_nan=True):
            failures.append(f"{name}: altering data after {cut} changed nothing at all, so the test proves nothing")
            return
    passes += 1


# ---------- load ----------
mkt = d.market_monthly()
mkt_d = d.market_daily()
mom_m, mom_d = d.french_table("mom_deciles_m"), d.french_table("mom_deciles_d")
ind12 = d.french_table("ind12_m")
px_trend = d.etf_prices(d.TREND_ETFS)
px_sector = d.etf_prices(d.SECTOR_ETFS)
btc = d.bitstamp_daily("btcusd")
funding = d.bitmex_funding()

# ---------- gate 1: no peeking ----------
# Many cutoffs per strategy: a signal only flips near its threshold, so one or two dates can miss a peek.
base = s.s2_index_trend(mkt.mkt, mkt.rf).held
for cut in cuts_between("1930-01-31", "2025-12-31", 40):
    same_through_next(f"#2 index trend @ {cut}", base, s.s2_index_trend(alter(mkt.mkt, cut, "ret"), mkt.rf).held, cut)

base = s.s2_with_tom(mkt.mkt, mkt_d.mkt, mkt_d.rf).held
for cut in cuts_between("1935-01-31", "2025-12-31", 8):
    b = s.s2_with_tom(alter(mkt.mkt, cut, "ret"), alter(mkt_d.mkt, cut, "ret"), mkt_d.rf).held
    same_through_next(f"#2+#8 @ {cut}", base, b, cut)

base = s.s1_trend(px_trend, mkt.rf).held
for cut in cuts_between("2009-01-31", "2025-12-31", 10):
    same_through_next(f"#1 trend @ {cut}", base, s.s1_trend(alter(px_trend, cut, "price"), mkt.rf).held, cut)

base_a = s.s4_scaled_momentum(mom_m, mom_d, mkt.rf).held
base_b = s.s4_wml_rebuild(mom_m, mom_d)[1].held
for cut in cuts_between("1930-01-31", "2025-12-31", 10):
    mm, md = alter(mom_m, cut, "ret"), alter(mom_d, cut, "ret")
    same_through_next(f"#4 scaled momentum @ {cut}", base_a, s.s4_scaled_momentum(mm, md, mkt.rf).held, cut)
    same_through_next(f"#4 rebuild scale @ {cut}", base_b, s.s4_wml_rebuild(mm, md)[1].held, cut)

base = s.s5_sector(ind12).held
for cut in cuts_between("1930-01-31", "2025-12-31", 20):
    same_through_next(f"#5 sectors @ {cut}", base, s.s5_sector(alter(ind12, cut, "ret")).held, cut)
sector_m = px_sector.resample("ME").last().pct_change()
base = s.s5_sector(sector_m).held
for cut in cuts_between("2001-01-31", "2025-12-31", 10):
    same_through_next(f"#5 sector ETFs @ {cut}", base, s.s5_sector(alter(sector_m, cut, "ret")).held, cut)

base = s.s8_turn_of_month(mkt_d.mkt, mkt_d.rf).held
for cut in ["1990-03-30", "2011-08-05"]:
    b = s.s8_turn_of_month(alter(mkt_d.mkt, cut, "ret"), mkt_d.rf).held
    same_through_next(f"#8 turn of month @ {cut}", base, b, cut, must_change_later=False)

base = s.s9_crypto_trend(btc)[0].held
for cut in cuts_between("2012-01-01", "2026-06-30", 12, freq="W-SUN"):  # Sundays, where weekly bars close
    same_through_next(f"#9 crypto trend @ {cut}", base, s.s9_crypto_trend(alter(btc, cut, "price"))[0].held, cut)

base = s.s10_carry(funding).held
for cut in ["2017-02-01 04:00", "2018-06-01 04:00", "2020-03-12 12:00", "2022-11-09 12:00", "2025-01-15 20:00"]:
    same_through_next(f"#10 carry @ {cut}", base, s.s10_carry(alter(funding, cut, "ret")).held, cut)

sleeves = pd.DataFrame({"x": s.s2_index_trend(mkt.mkt, mkt.rf).monthly.net,
                        "y": s.s5_sector(ind12).monthly.net}).dropna()
base = s.build_bot(sleeves, mkt.rf).held
for cut in cuts_between("1935-01-31", "2025-12-31", 10):
    same_through_next(f"bot weights @ {cut}", base, s.build_bot(alter(sleeves, cut, "ret"), mkt.rf).held, cut)

# ---------- gate 2: costs only cost ----------
costed = {
    "#1": s.s1_trend(px_trend, mkt.rf),
    "#2": s.s2_index_trend(mkt.mkt, mkt.rf),
    "#5": s.s5_sector(ind12),
    "#8": s.s8_turn_of_month(mkt_d.mkt, mkt_d.rf),
    "#9": s.s9_crypto_trend(btc)[0],
    "#10": s.s10_carry(funding),
}
for k, run in costed.items():
    m = run.monthly.dropna()
    if (m.net > m.gross + 1e-12).any():
        failures.append(f"{k}: net beat gross in {int((m.net > m.gross + 1e-12).sum())} months, costs were paid to us")
    elif (m.gross - m.net).sum() <= 0:
        failures.append(f"{k}: no cost was charged at all")
    else:
        passes += 1

# ---------- gate 3: data spot checks ----------
checks = [
    ("US market, October 1987", float(mkt.mkt.loc["1987-10-31"]), -0.23, -0.22),
    ("BTC close 2017-12-17", float(btc.loc["2017-12-17"]), 18000, 20000),
    ("SPY total return 2008", float(px_trend.SPY.loc["2008-12-31"] / px_trend.SPY.loc["2007-12-31"] - 1), -0.38, -0.35),
    ("Turn-of-month days per full month", float(s._tom_mask(mkt_d.index).groupby(mkt_d.index.to_period("M")).sum().iloc[1:-1].mean()), 3.99, 4.01),
]
for name, value, lo, hi in checks:
    if lo <= value <= hi:
        passes += 1
    else:
        failures.append(f"data check {name}: got {value}, expected {lo} to {hi}")

# ---------- gate 4: the paper bot (selftest_paper.py) ----------
import selftest_paper

p4, f4 = selftest_paper.run()
passes += p4
failures += f4

print(f"{passes} checks passed, {len(failures)} failed")
for f in failures:
    print("FAIL", f)
sys.exit(1 if failures else 0)
