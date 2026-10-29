"""Backtest the doc's Tier 1 strategies and the combined bot. Run: python run.py

Writes results/*.csv and results/summary.json and prints the tables.

THE PASS BAR (written before any result was seen; do not edit it to fit results)
Step 2, rebuild: rerun a published result on the same data window and compare.
Step 3, after publication: in the months after the defining paper came out, a strategy
holds up only if all three are true:
  (a) its Sharpe ratio after publication is at least half its Sharpe ratio before
      (skipped when there are fewer than 36 months before)
  (b) its Sharpe after publication beats its benchmark's Sharpe over the same months
      (benchmark: the US stock market for stock strategies, SPY for ETF versions, the coin
      itself for crypto trend); diversifiers and carry, which are not trying to beat stocks,
      need a Sharpe of at least 0.3 instead
  (c) it beat cash over those months
Net returns (after COSTS in strategies.py) are used wherever costs can be computed. Where they
cannot (Ken French's stock portfolios), gross returns are judged and the break-even cost is shown.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import data as d
import metrics as mt
import strategies as s

RESULTS = d.ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Month the defining paper came out (the doc's citation); "after" starts the following month.
PUBLISHED = {
    "#1": "2012-05", "#2": "2007-01", "#3": "1993-03", "#4": "2015-04", "#5": "1999-08",
    "#6": "2011-01", "#7": "2013-04", "#8": "1987-03", "#9": "2018-08", "#10": "2023-03",
}
BOT_SPLIT = "2007-01"


def load():
    mkt, mkt_d = d.market_monthly(), d.market_daily()
    px_all = d.etf_prices(d.TREND_ETFS + d.SECTOR_ETFS + d.FACTOR_ETFS)
    etf_m = px_all.resample("ME").last().pct_change()
    return {
        "mkt": mkt, "mkt_d": mkt_d, "etf_m": etf_m,
        "px_trend": px_all[d.TREND_ETFS],
        "mom_m": d.french_table("mom_deciles_m"), "mom_d": d.french_table("mom_deciles_d"),
        "var_m": d.french_table("var_m"), "op_m": d.french_table("op_m"), "ind12": d.french_table("ind12_m"),
        "btc": d.bitstamp_daily("btcusd"), "eth": d.bitstamp_daily("ethusd"),
        "funding": d.bitmex_funding(),
    }


def windows(series: pd.Series, published: str | None):
    out = {"full": series}
    if published:
        cut = pd.Timestamp(published) + pd.offsets.MonthEnd(0)
        out["before"] = series.loc[:cut]
        out["after"] = series.loc[cut + pd.offsets.MonthEnd(1):]
    return out


def judge(stats: dict, kind: str) -> tuple[str, str]:
    before, after, bench = stats.get("before"), stats["after"], stats["bench_after"]
    reasons = []
    if before and before["months"] >= 36 and not after["sharpe"] >= 0.5 * before["sharpe"]:
        reasons.append(f"Sharpe fell from {before['sharpe']:.2f} to {after['sharpe']:.2f}, more than half")
    if kind in ("diversifier", "carry"):
        if not after["sharpe"] >= 0.3:
            reasons.append(f"Sharpe {after['sharpe']:.2f} is under 0.3")
    elif not after["sharpe"] > bench["sharpe"]:
        reasons.append(f"Sharpe {after['sharpe']:.2f} did not beat the benchmark's {bench['sharpe']:.2f}")
    if not after["cagr"] > stats["cash_after"]:
        reasons.append("did not beat cash")
    return ("holds up" if not reasons else "fails"), "; ".join(reasons)


def main():
    D = load()
    mkt, rf = D["mkt"], D["mkt"].rf
    etf_m = D["etf_m"]
    spy = etf_m["SPY"]

    runs = {}
    runs["#1"] = ("Trend following, 8 ETF markets, long/short", s.s1_trend(D["px_trend"], rf), "diversifier", None)
    runs["#1 flat"] = ("Trend following, long or cash only", s.s1_trend(D["px_trend"], rf, long_short=False), "diversifier", None)
    runs["#2"] = ("Index trend filter (10-month average)", s.s2_index_trend(mkt.mkt, rf), "stock", mkt.mkt)
    runs["#2+#8"] = ("#2 with turn-of-month layer", s.s2_with_tom(mkt.mkt, D["mkt_d"].mkt, D["mkt_d"].rf), "stock", mkt.mkt)
    runs["#3"] = ("Stock momentum, top tenth", s.s3_momentum(D["mom_m"]), "stock", mkt.mkt)
    runs["#4"] = ("Momentum, volatility-scaled, long-only", s.s4_scaled_momentum(D["mom_m"], D["mom_d"], rf), "stock", mkt.mkt)
    runs["#5"] = ("Sector momentum, 12 industries since 1927", s.s5_sector(D["ind12"]), "stock", mkt.mkt)
    runs["#5 ETF"] = ("Sector momentum, 9 real sector ETFs", s.s5_sector(etf_m[d.SECTOR_ETFS]), "stock", spy)
    runs["#6"] = ("Low-volatility stocks, lowest fifth", s.s6_low_vol(D["var_m"]), "stock", mkt.mkt)
    runs["#7"] = ("Quality (operating profitability), top fifth", s.s7_quality(D["op_m"]), "stock", mkt.mkt)
    runs["#8"] = ("Turn of the month", s.s8_turn_of_month(D["mkt_d"].mkt, D["mkt_d"].rf), "stock", mkt.mkt)
    btc_run, btc_wr = s.s9_crypto_trend(D["btc"])
    eth_run, eth_wr = s.s9_crypto_trend(D["eth"])
    runs["#9 BTC"] = ("Crypto trend, Bitcoin", btc_run, "coin", mt.monthly_from(btc_wr))
    runs["#9 ETH"] = ("Crypto trend, Ether", eth_run, "coin", mt.monthly_from(eth_wr))
    runs["#10"] = ("Crypto cash-and-carry (doc rule: exit on negative funding)", s.s10_carry(D["funding"]), "carry", None)
    runs["#10 always"] = ("Crypto cash-and-carry, never exit", s.s10_carry(D["funding"], always_on=True), "carry", None)

    # Real funds that run #3, #6 and #7 today, after their own fees and trading costs.
    for tk, key in (("MTUM", "#3 fund"), ("USMV", "#6 fund"), ("QUAL", "#7 fund")):
        r = etf_m[tk].dropna()
        runs[key] = (f"{tk} (real fund)", s.Run(pd.DataFrame({"gross": r, "net": r}), None), "stock", spy)

    rows, verdicts, series_out = [], [], {}
    for key, (name, run, kind, bench) in runs.items():
        base = key.split()[0].replace("+#8", "")
        m = run.monthly
        costed = m["net"].notna().any()
        r = (m["net"] if costed else m["gross"]).dropna()
        series_out[key] = r
        pub = PUBLISHED.get(base) if "fund" not in key else None
        stats = {}
        for wname, wr in windows(r, pub).items():
            if len(wr) < 12:
                continue
            st = mt.summary(wr, rf)
            if bench is not None:
                b = bench.reindex(wr.index)
                bst = mt.summary(b, rf)
                st.update({f"bench_{k}": v for k, v in bst.items() if k in ("cagr", "vol", "sharpe", "max_dd")})
            st["cash_cagr"] = mt.cagr(rf.reindex(wr.index))
            stats[wname] = st
            rows.append({"key": key, "name": name, "window": wname, "costed": costed, **st})
        judged_window = "after" if "after" in stats else "full"
        a = stats[judged_window]
        bench_after = {"sharpe": a.get("bench_sharpe", float("nan"))}
        verdict, why = judge({"before": stats.get("before"), "after": a, "bench_after": bench_after,
                              "cash_after": a["cash_cagr"]}, kind)
        be = mt.breakeven_cost(r.loc[a["start"]:], bench) if (not costed and bench is not None) else None
        verdicts.append({"key": key, "name": name, "judged_on": judged_window, "window": f"{a['start']} to {a['end']}",
                         "costed": costed, "verdict": verdict, "why": why, "breakeven_cost": be})

    # ---------- the bots ----------
    fee = s.COSTS["fund_fee"] / 12
    crypto = s.crypto_pair(btc_run, eth_run)
    sleeves_a = pd.DataFrame({
        "trend": 0.5 * runs["#1"][1].monthly.net + 0.5 * crypto,
        "momentum": runs["#5 ETF"][1].monthly.net,
        "defensive": 0.5 * etf_m["USMV"] + 0.5 * etf_m["QUAL"],
        "carry": runs["#10"][1].monthly.net,
    })
    bot_a = s.build_bot(sleeves_a, rf)
    # Second try, added after the doc's #10 rule failed on switching costs: never exit the carry pair.
    sleeves_a2 = sleeves_a.assign(carry=runs["#10 always"][1].monthly.net)
    bot_a2 = s.build_bot(sleeves_a2, rf)
    sleeves_b = pd.DataFrame({
        "trend": runs["#2+#8"][1].monthly.net,
        "momentum": runs["#5"][1].monthly.net,
        "defensive": 0.5 * (runs["#6"][1].monthly.gross - fee) + 0.5 * (runs["#7"][1].monthly.gross - fee),
    })
    bot_b = s.build_bot(sleeves_b, rf)
    # The tradable version the paper bot runs: SPY/BIL trend with turn-of-month, sector ETFs, USMV+QUAL.
    px_paper = d.etf_prices(s.PAPER_TICKERS)
    sleeves_p, bil_m = s.paper_sleeves(px_paper)
    bot_p = s.build_bot(sleeves_p, bil_m)
    sixty_forty = 0.6 * spy + 0.4 * etf_m["IEF"]

    bots = {
        "Bot A": ("Full bot: trend, momentum, defensive, carry", bot_a, {"SPY": spy, "60/40": sixty_forty}, None),
        "Bot A2": ("Second try: Bot A with carry never exiting", bot_a2, {"SPY": spy, "60/40": sixty_forty}, None),
        "Bot B": ("Stock-only bot since 1964: #2+#8, #5, #6+#7", bot_b, {"US market": mkt.mkt}, BOT_SPLIT),
        "Bot P": ("Paper bot's tradable version: SPY/BIL, sector ETFs, USMV+QUAL", bot_p, {"SPY": spy, "60/40": sixty_forty}, None),
    }
    for key, (name, run, benches, split) in bots.items():
        r = run.monthly.net.dropna()
        series_out[key] = r
        for bname, b in benches.items():
            series_out[f"{key} bench {bname}"] = b.reindex(r.index)
        stats = {}
        for wname, wr in windows(r, split).items():
            st = mt.summary(wr, rf)
            for bname, b in benches.items():
                bst = mt.summary(b.reindex(wr.index), rf)
                st.update({f"{bname}_{k}": v for k, v in bst.items() if k in ("cagr", "vol", "sharpe", "max_dd")})
            st["cash_cagr"] = mt.cagr(rf.reindex(wr.index))
            stats[wname] = st
            rows.append({"key": key, "name": name, "window": wname, "costed": True, **st})
        first_bench = next(iter(benches))
        a = stats["after"] if "after" in stats else stats["full"]
        verdict, why = judge({"before": stats.get("before"), "after": a,
                              "bench_after": {"sharpe": a[f"{first_bench}_sharpe"]}, "cash_after": a["cash_cagr"]}, "stock")
        verdicts.append({"key": key, "name": name, "judged_on": "after" if "after" in stats else "full",
                         "window": f"{a['start']} to {a['end']}", "costed": True, "verdict": verdict, "why": why,
                         "breakeven_cost": None})
        run.held.to_csv(RESULTS / f"{key.replace(' ', '_').lower()}_weights.csv")
        sleeve_df = {"Bot A": sleeves_a, "Bot A2": sleeves_a2, "Bot B": sleeves_b, "Bot P": sleeves_p}[key]
        sleeve_df.to_csv(RESULTS / f"{key.replace(' ', '_').lower()}_sleeves.csv")

    # ---------- step 2 rebuilds ----------
    rebuilds = []
    t2 = runs["#2"][1].monthly.net.loc[:"2012-12"]
    m2 = mkt.mkt.reindex(t2.index)
    rebuilds.append({"paper": "Faber: 10-month filter vs buy-and-hold (paper 1901-2012)",
                     "paper_says": "10.18% vs 9.32% a year; worst drop 42% vs 84%",
                     "we_get": f"{mt.cagr(t2):.2%} vs {mt.cagr(m2):.2%} a year; worst drop {-mt.max_drawdown(t2):.0%} vs {-mt.max_drawdown(m2):.0%} ({t2.index[0]:%Y}-2012)"})
    plain, scaled = s.s4_wml_rebuild(D["mom_m"], D["mom_d"])
    p, q = plain.monthly.gross.loc["1927-03":"2011-12"], scaled.monthly.gross.loc["1927-03":"2011-12"]
    rebuilds.append({"paper": "Barroso & Santa-Clara: momentum Sharpe, plain vs scaled (1927-2011)",
                     "paper_says": "0.53 vs 0.97",
                     "we_get": f"{mt.sharpe(p):.2f} vs {mt.sharpe(q):.2f}; worst month {p.min():.0%} vs {q.min():.0%}"})
    md = D["mkt_d"].loc[:"2005-12-31"]
    tom = s._tom_mask(md.index)
    ex = md.mkt - md.rf
    rebuilds.append({"paper": "McConnell & Xu: stock returns above T-bills, turn-of-month days vs all others (1926-2005)",
                     "paper_says": "no reward for market risk except at the turn of the month",
                     "we_get": f"{ex[tom].mean() * 1e4:.1f} vs {ex[~tom].mean() * 1e4:.1f} basis points a day; "
                               f"other days' total excess {np.expm1(np.log1p(ex[~tom]).sum()):.0%} over 80 years"})
    wr = btc_wr.loc[:"2018-07-31"].dropna()
    x, y = wr.iloc[:-1].values, wr.iloc[1:].values
    slope = np.polyfit(x, y, 1)[0]
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    tstat = slope / (resid.std(ddof=2) / (x.std() * np.sqrt(len(x))))
    rebuilds.append({"paper": "Liu & Tsyvinski: Bitcoin this week's return predicting next week's (2011-2018)",
                     "paper_says": "slope 0.19, t = 3.73",
                     "we_get": f"slope {slope:.2f}, t = {tstat:.2f} (Bitstamp, {wr.index[0]:%b %Y}-{wr.index[-1]:%b %Y})"})
    lo = runs["#6"][1].monthly.gross.loc["1968-01":"2008-12"]
    rebuilds.append({"paper": "Baker, Bradley & Wurgler: $1 in the lowest-volatility fifth, 1968-2008",
                     "paper_says": "grew to $59.55 (top 1,000 stocks, their own volatility sort)",
                     "we_get": f"grew to ${(1 + lo).prod():.2f} (French's all-stock variance sort)"})
    f_all = s.s10_carry(D["funding"], always_on=True).monthly.gross.loc[:"2022-12"]
    rebuilds.append({"paper": "Schmeling, Schrimpf & Todorov: crypto carry averaged above 10% a year",
                     "paper_says": "above 10% a year, up to 60% at times",
                     "we_get": f"{(1 + f_all).prod() ** (12 / len(f_all)) - 1:.1%} a year always-on, BitMEX {f_all.index[0]:%Y}-2022; "
                               f"best 12 months {((1 + f_all).rolling(12).apply(np.prod, raw=True) - 1).max():.0%}"})

    pd.DataFrame(rows).to_csv(RESULTS / "summary.csv", index=False)
    pd.DataFrame(verdicts).to_csv(RESULTS / "verdicts.csv", index=False)
    pd.DataFrame(rebuilds).to_csv(RESULTS / "rebuilds.csv", index=False)
    pd.DataFrame(series_out).to_csv(RESULTS / "monthly_net.csv")
    (RESULTS / "summary.json").write_text(json.dumps(
        {"rows": rows, "verdicts": verdicts, "rebuilds": rebuilds}, indent=1, default=str))

    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 90)
    print("\nREBUILDS")
    for rb in rebuilds:
        print(f"- {rb['paper']}\n    paper: {rb['paper_says']}\n    ours:  {rb['we_get']}")
    df = pd.DataFrame(rows)
    show = df[["key", "window", "start", "end", "cagr", "vol", "sharpe", "max_dd", "bench_cagr", "bench_sharpe",
               "bench_max_dd", "cash_cagr"]].copy()
    for c in ("cagr", "vol", "max_dd", "bench_cagr", "bench_max_dd", "cash_cagr"):
        show[c] = (show[c] * 100).round(1)
    for c in ("sharpe", "bench_sharpe"):
        show[c] = show[c].round(2)
    print("\nSTRATEGIES (percent a year; Sharpe; worst drop)")
    print(show[~show.key.str.startswith("Bot")].to_string(index=False))
    bot_cols = [c for c in df.columns if c.startswith(("SPY_", "60/40_", "US market_"))]
    bshow = df[df.key.str.startswith("Bot")][["key", "window", "start", "end", "cagr", "vol", "sharpe", "max_dd"] + bot_cols]
    print("\nBOTS")
    print(bshow.round(3).to_string(index=False))
    print("\nVERDICTS (step 3: after publication)")
    print(pd.DataFrame(verdicts)[["key", "window", "costed", "verdict", "why", "breakeven_cost"]].to_string(index=False))


if __name__ == "__main__":
    main()
