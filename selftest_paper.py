"""Gates for the paper bot. Run alone (python selftest_paper.py) or as part of selftest.py.

4a. The engine's closing-price book reproduces the backtest of the same strategy, month by month,
    and every monthly plan matches the backtest pipeline rebuilt from scratch.
4b. Scrambling prices after a month-end close changes nothing decided at or before that close.
4c. The bot never treats a day's prices as final until 45 minutes after that day's close.

Prove the gates fire by breaking the rules on purpose (each must print FAIL and exit 1):
  PEEK_LIVE=1 python selftest_paper.py    # decisions can see prices after the close they are made at
  PARTIAL_BAR=1 python selftest_paper.py  # today's bar counts as final before the market closes
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import paper_bot as pb
import strategies as s


def run(rng: np.random.Generator | None = None) -> tuple[int, list[str]]:
    rng = rng or np.random.default_rng(11)
    passes, failures = 0, []
    pp = pb.fetch_prices(use_cache=True)

    # 4a
    st = pb.replay_with(pp, pb.D("2015-01-02"), pb.D("2026-08-31"), None)
    led = pd.DataFrame(st["ledger"]).set_index("date")
    led.index = pd.to_datetime(led.index)
    eng = led["backtest"].resample("ME").last().pct_change().dropna()
    sl, bil_m = s.paper_sleeves(pp["close"], pb.tom_series(pp["close"].index))
    bt = s.build_bot(sl, bil_m).monthly.net
    both = pd.concat([eng, bt], axis=1, keys=["engine", "backtest"]).dropna()
    gap = (both.engine - both.backtest).abs()
    if len(both) >= 120 and gap.max() < 0.0005:
        passes += 1
    else:
        failures.append(f"paper engine vs backtest: {len(both)} months, worst monthly gap {gap.max():.4%} (limit 0.05%)")
    if all(c["match"] is not False for c in st["checks"]["plans"]):
        passes += 1
    else:
        failures.append("paper engine: a monthly plan did not match the backtest pipeline")

    # 4b
    def scrambled(prices, cut):
        out = {}
        for k, v in prices.items():
            v = v.copy()
            mask = v.index > pd.Timestamp(cut)
            v.loc[mask] = v.loc[mask] * np.exp(rng.normal(0, 0.25, size=v.loc[mask].shape))
            out[k] = v
        return out

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "processed_at"} for r in rows]

    base = pb.replay_with(pp, pb.D("2017-01-03"), pb.D("2025-06-30"), None)
    for cut in ["2018-12-31", "2020-03-31", "2022-06-30", "2024-10-31"]:
        alt = pb.replay_with(scrambled(pp, cut), pb.D("2017-01-03"), pb.D("2025-06-30"), None)
        a = [r for r in base["ledger"] if r["date"] <= cut]
        b = [r for r in alt["ledger"] if r["date"] <= cut]
        pa = [p for p in base["plans"] if p["decided_on"] <= cut]
        pb_ = [p for p in alt["plans"] if p["decided_on"] <= cut]
        if strip(a) != strip(b) or pa != pb_:
            failures.append(f"paper engine @ {cut}: something decided by that close changed when only later prices changed")
        elif strip(base["ledger"]) == strip(alt["ledger"]):
            failures.append(f"paper engine @ {cut}: scrambling later prices changed nothing, so the test proves nothing")
        else:
            passes += 1

    # 4c: Nov 26 2026 is Thanksgiving; Nov 27 closes early at 1:00 PM.
    et = "America/New_York"
    for now, want in [("2026-09-28 15:00", "2026-09-25"), ("2026-09-28 16:30", "2026-09-25"),
                      ("2026-09-28 17:00", "2026-09-28"), ("2026-09-26 12:00", "2026-09-25"),
                      ("2026-11-27 13:30", "2026-11-25"), ("2026-11-27 14:00", "2026-11-27")]:
        got = pb.last_finished_session(pd.Timestamp(now, tz=et).tz_convert("UTC").to_pydatetime())
        if got == pd.Timestamp(want):
            passes += 1
        else:
            failures.append(f"at {now} New York time the bot treats {got.date()} as finished; should be {want}")
    return passes, failures


if __name__ == "__main__":
    p, f = run()
    print(f"{p} paper checks passed, {len(f)} failed")
    for x in f:
        print("FAIL", x)
    sys.exit(1 if f else 0)
