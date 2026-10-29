"""Writes paper/dashboard.html: one static page, rebuilt after every run. No scripts, no network.

Colors (text contrast measured against the card surface):
  light  page #F4F5F7  surface #FFFFFF  ink #15171C 17.9:1  ink2 #5A606B 6.3:1
         accent #2354C2 6.7:1  gain #1A7547 5.7:1  loss #B3302A 6.2:1
  dark   page #0E1014  surface #171A20  ink #ECEEF2 15.0:1  ink2 #A3A9B5 7.4:1
         accent #7EA4FF 7.2:1  gain #5CC993 8.5:1  loss #FF8C82 7.7:1
Signature element: the 63-day trial track. Everything else stays quiet.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import pandas as pd

FUND_NAMES = {"SPY": "S&P 500", "BIL": "T-bills", "USMV": "Low volatility", "QUAL": "Quality"}
SLEEVE_NAMES = {"trend": "Trend", "momentum": "Sector momentum", "defensive": "Defensive", "cash": "Cash"}

CSS = """
:root{--page:#F4F5F7;--surface:#FFFFFF;--ink:#15171C;--ink2:#5A606B;--line:#E2E4E9;--track:#D5D8DE;
--accent:#2354C2;--gain:#1A7547;--loss:#B3302A;--bench:#8A909B;color-scheme:light dark}
@media (prefers-color-scheme:dark){:root{--page:#0E1014;--surface:#171A20;--ink:#ECEEF2;--ink2:#A3A9B5;
--line:#2A2E36;--track:#343944;--accent:#7EA4FF;--gain:#5CC993;--loss:#FF8C82;--bench:#7A808C}}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI Variable Text","Segoe UI",system-ui,sans-serif;
-webkit-font-smoothing:antialiased}
main{max-width:880px;margin:0 auto;padding:40px 16px 56px}
h1{font-size:28px;line-height:1.2;margin:0 0 4px;font-weight:650;letter-spacing:-0.01em}
h2{font-size:15px;margin:0 0 14px;font-weight:600}
.status{color:var(--ink2);margin:0 0 20px;font-size:14px}
.lead{font-size:20px;line-height:1.45;margin:0 0 28px;max-width:40em}
.num{font-variant-numeric:tabular-nums}
.gain{color:var(--gain)}.loss{color:var(--loss)}
.card{background:var(--surface);border-radius:14px;padding:20px 22px;margin:0 0 16px;border:1px solid var(--line)}
.track{display:grid;grid-template-columns:repeat(63,1fr);gap:3px;margin:4px 0 10px}
.track span{height:22px;border-radius:3px;background:var(--track)}
.track span.done{background:var(--accent)}
.track span.today{background:var(--accent);box-shadow:0 0 0 2px var(--surface),0 0 0 4px var(--accent)}
.meta{display:flex;justify-content:space-between;color:var(--ink2);font-size:13px;gap:12px;flex-wrap:wrap}
ul.checks{list-style:none;padding:0;margin:16px 0 0}
ul.checks li{display:grid;grid-template-columns:22px 1fr;gap:8px;padding:6px 0;border-top:1px solid var(--line);font-size:14px}
ul.checks li:first-child{border-top:0}
.mark{font-weight:700;text-align:center}
.mark.ok{color:var(--gain)}.mark.no{color:var(--loss)}.mark.wait{color:var(--ink2)}
table{width:100%;border-collapse:collapse;font-size:14px}
th{text-align:left;color:var(--ink2);font-weight:500;padding:6px 8px 6px 0;border-bottom:1px solid var(--line)}
td{padding:8px 8px 8px 0;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.r,th.r{text-align:right}
.why{color:var(--ink2);font-size:13px}
.sub{color:var(--ink2);font-size:13px;margin:-8px 0 14px}
svg text{font:12px -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;font-variant-numeric:tabular-nums}
footer{color:var(--ink2);font-size:13px;margin-top:28px}
.nw{white-space:nowrap}
.narrow{display:none}
ul.key{list-style:none;padding:0;margin:10px 0 0;font-size:13px;color:var(--ink2)}
ul.key li{display:flex;align-items:center;gap:8px;padding:2px 0}
ul.key .num{margin-left:auto;color:var(--ink)}
@media (max-width:560px){.lead{font-size:18px}.track span{height:16px}.hide-s{display:none}.card{padding:16px}
.wide{display:none}.narrow{display:block}}
"""


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{abs(x) * 100:.2f}%"


def day(s: str) -> str:
    ts = pd.Timestamp(s)
    return f"{ts.strftime('%a, %b')} {ts.day}"


def change_words(x: float) -> str:
    if abs(x) < 0.00005:
        return "even"
    cls = "gain" if x > 0 else "loss"
    return f'<span class="{cls}">{"up" if x > 0 else "down"} {pct(x)}</span>'


SERIES = [("real", "Real timing", "var(--accent)", "", 2.25),
          ("backtest", "Backtest timing", "var(--accent)", "5 4", 1.5),
          ("spy", "S&P 500", "var(--bench)", "", 1.75)]


def chart(ledger: list[dict], capital: float) -> str:
    """Two drawings of the same chart: labelled line ends for wide screens, a key below for phones."""
    if len(ledger) < 3:
        return '<p class="why">The chart fills in after a few trading days.</p>'
    key = "".join(
        f'<li><svg width="22" height="10" aria-hidden="true"><line x1="0" x2="22" y1="5" y2="5" stroke="{c}" '
        f'stroke-width="{w}" stroke-dasharray="{d}"/></svg>{html.escape(lbl)} <span class="num">{money(ledger[-1][k])}</span></li>'
        for k, lbl, c, d, w in SERIES)
    return (f'<div class="wide">{_chart_svg(ledger, capital, 800, 56, 640, True)}</div>'
            f'<div class="narrow">{_chart_svg(ledger, capital, 400, 50, 392, False)}<ul class="key">{key}</ul></div>')


def _chart_svg(ledger: list[dict], capital: float, W: int, left: int, right: int, end_labels: bool) -> str:
    H, y0, y1 = 260, 16, 236
    series = SERIES
    vals = [r[k] for r in ledger for k, *_ in series]
    lo, hi = min(vals + [capital]), max(vals + [capital])
    pad = max((hi - lo) * 0.12, capital * 0.002)
    lo, hi = lo - pad, hi + pad
    n = len(ledger)
    X = lambda i: left + i / (n - 1) * (right - left)
    Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0)
    parts = [f'<svg viewBox="0 0 {W} {H + 24}" width="100%" role="img" aria-label="Value of the paper account against the S&amp;P 500">']
    for frac in (0, 0.5, 1):
        v = lo + (hi - lo) * frac
        parts.append(f'<line x1="{left}" x2="{right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="var(--line)"/>'
                     f'<text x="{left - 8}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="var(--ink2)">${v:,.0f}</text>')
    parts.append(f'<line x1="{left}" x2="{right}" y1="{Y(capital):.1f}" y2="{Y(capital):.1f}" stroke="var(--ink2)" stroke-dasharray="2 4"/>')
    ends = []
    for key, label, color, dash, width in series:
        pts = " ".join(f"{X(i):.1f},{Y(r[key]):.1f}" for i, r in enumerate(ledger))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" '
                     f'stroke-dasharray="{dash}" stroke-linejoin="round"/>')
        ends.append([Y(ledger[-1][key]), f"{label} {money(ledger[-1][key])}", key == "real"])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
    for yv, text, strong in ends if end_labels else []:
        parts.append(f'<text x="{right + 10}" y="{yv + 4:.1f}" fill="var({"--ink" if strong else "--ink2"})" '
                     f'font-weight="{600 if strong else 400}">{html.escape(text)}</text>')
    first, last = ledger[0]["date"], ledger[-1]["date"]
    parts.append(f'<text x="{left}" y="{H + 16}" fill="var(--ink2)">{day(first)}</text>'
                 f'<text x="{right}" y="{H + 16}" text-anchor="end" fill="var(--ink2)">{day(last)}</text></svg>')
    return "".join(parts)


def holdings(state: dict, plan: dict, sector_names: dict) -> str:
    book = state["books"]["real"]
    total = sum(v for sl in book.values() for v in sl.values())
    rows = []
    trend_pct = (plan["spy_month_end"] / plan["spy_avg10"] - 1)
    whys = {
        "trend": (f"S&amp;P 500 ended {pd.Timestamp(plan['decided_on']).strftime('%B')} "
                  f"{pct(trend_pct)} {'above' if trend_pct > 0 else 'below'} its 10-month average"
                  + ("" if plan["trend_on"] else "; holds T-bills except at the turn of the month")),
        "momentum": "Best 12-month returns of the 9 sectors: " + ", ".join(
            f"{sector_names[t]} {'+' if plan['sector_12m'][t] >= 0 else '-'}{abs(plan['sector_12m'][t]) * 100:.0f}%"
            for t in plan["sectors"]),
        "defensive": "Low-volatility and quality funds, half each",
        "cash": "Held back to keep the bot's swings near 10% a year",
    }
    for sl in ("trend", "momentum", "defensive", "cash"):
        funds = book.get(sl, {})
        if not funds:
            continue
        first = True
        for tk, v in sorted(funds.items(), key=lambda kv: -kv[1]):
            name = FUND_NAMES.get(tk) or sector_names.get(tk, tk)
            rows.append(
                "<tr>"
                + (f'<td rowspan="{len(funds)}"><strong>{SLEEVE_NAMES[sl]}</strong>'
                   f'<div class="why">{whys[sl]}</div></td>' if first else "")
                + f'<td>{html.escape(name)} <span class="why">{tk}</span></td>'
                f'<td class="r num">{v / total * 100:.1f}%</td><td class="r num">{money(v)}</td></tr>')
            first = False
    if "start" in book:
        rows.append(f'<tr><td><strong>Waiting to buy</strong><div class="why">First trades fill at the next open</div></td>'
                    f'<td>Cash</td><td class="r num">100.0%</td><td class="r num">{money(total)}</td></tr>')
    return ('<table><thead><tr><th>Part</th><th>Fund</th><th class="r">Share</th><th class="r">Value</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def trades_table(state: dict, sector_names: dict) -> str:
    real = [t for t in state["trades"] if t["book"] == "real"][-10:][::-1]
    if not real:
        return '<p class="why">No real-timing trades yet; the first ones fill at the next open.</p>'
    name = lambda tk: FUND_NAMES.get(tk) or sector_names.get(tk, tk)
    rows = "".join(
        f'<tr><td class="num nw">{day(t["date"])}</td><td>{"Bought" if t["action"] == "buy" else "Sold"}</td>'
        f'<td>{html.escape(name(t["fund"]))} <span class="why">{t["fund"]}</span></td>'
        f'<td class="r num nw">{money(t["dollars"])}</td><td class="r num hide-s">{money(t["price"])}</td></tr>'
        for t in real)
    return ('<table><thead><tr><th>Date</th><th>Trade</th><th>Fund</th><th class="r">Amount</th>'
            f'<th class="r hide-s">Price</th></tr></thead><tbody>{rows}</tbody></table>')


def write_waiting(folder: Path, start_session: str, capital: float, checked_at: str) -> None:
    """Placeholder page while the start day's final prices are not posted yet."""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper trading bot</title>
<style>{CSS}</style></head><body><main>
<h1>Paper trading bot</h1>
<p class="status num">Last checked {day(checked_at)}, {pd.Timestamp(checked_at).strftime("%I:%M %p").lstrip("0")}</p>
<p class="lead">The trial starts with {money(capital).replace(".00", "")} of paper money at the close on {day(start_session)}.
Yahoo has not posted that day's final prices yet; the next run (8:00 AM or 6:00 PM) begins as soon as it does.</p>
<footer>Paper money only; nothing is sent to a broker.</footer>
</main></body></html>"""
    (folder / "dashboard.html").write_text(page, encoding="utf-8")


def write_dashboard(state: dict, folder: Path, trial: dict, pass_bar: list[str], sector_names: dict,
                    upcoming: list[dict]) -> None:
    ledger = state["ledger"]
    cap = state["capital"]
    last = ledger[-1]
    real_chg = last["real"] / cap - 1
    spy_chg = last["spy"] / cap - 1
    this_month = pd.Timestamp(state["last_session"]).strftime("%Y-%m")
    current = [p for p in state["plans"] if p["for_month"] <= this_month]
    plan = current[-1] if current else state["plans"][0]  # the plan behind what is held right now
    updated = datetime.fromisoformat(state.get("last_run", state["created_at"]))
    track = "".join(
        f'<span class="{"today" if i == trial["day"] - 1 else "done" if i < trial["day"] else ""}"></span>'
        for i in range(trial["of"]))
    marks = []
    for text, ok in zip(pass_bar, trial["criteria"]):
        if trial["finished"]:
            cls, sym, label = ("ok", "✓", "Passed") if ok else ("no", "✕", "Failed")
        else:
            cls, sym, label = ("ok", "✓", "On track") if ok else ("no", "✕", "Off track")
        marks.append(f'<li><span class="mark {cls}" aria-label="{label}">{sym}</span><span>{html.escape(text)}'
                     f'{f" <span class=why>(now {trial["gap"] * 100:+.2f} points)</span>" if "percentage point" in text else ""}'
                     "</span></li>")
    nxt = "".join(f'<tr><td class="num">{day(u["date"])}</td><td>{html.escape(u["text"])}</td></tr>' for u in upcoming) \
        or '<tr><td colspan="2" class="why">Nothing scheduled in the next few weeks.</td></tr>'
    verdict = {"in progress": "Verdict after the last day", "passes": "Passed step 4", "fails": "Did not pass step 4"}[trial["verdict"]]
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper trading bot</title>
<style>{CSS}</style></head><body><main>
<h1>Paper trading bot</h1>
<p class="status num">{f'Day {trial["day"]} of {trial["of"]}' if trial["day"] <= trial["of"] else f'Trial finished {day(trial["ends_on"])}'} · prices through {day(state["last_session"])} close · updated {updated.strftime("%b ")}{updated.day}, {updated.strftime("%I:%M %p").lstrip("0")}</p>
<p class="lead num">The paper account is at <strong>{money(last["real"])}</strong>, {change_words(real_chg)} since {day(state["start_session"])}. The S&amp;P 500 is {change_words(spy_chg)} over the same days.</p>
<section class="card" aria-label="Three-month trial">
<h2>Three-month trial</h2>
<div class="track" role="img" aria-label="{trial["day"]} of {trial["of"]} trading days done">{track}</div>
<div class="meta num"><span>Started {day(state["start_session"])}</span><span>{verdict}</span><span>Ends {day(trial["ends_on"])}</span></div>
<ul class="checks">{"".join(marks)}</ul>
</section>
<section class="card"><h2>Value of {money(cap).replace(".00", "")}</h2>
<p class="sub">Real timing trades at the next morning's open. Backtest timing trades at the close, as the backtest assumed.</p>
{chart(ledger, cap)}</section>
<section class="card"><h2>Holding now</h2>
<p class="sub">Plan for {pd.Timestamp(plan["for_month"]).strftime("%B")}, set at the close on {day(plan["decided_on"])}</p>
{holdings(state, plan, sector_names)}</section>
<section class="card"><h2>Next moves</h2><table><tbody>{nxt}</tbody></table></section>
<section class="card"><h2>Recent trades</h2>{trades_table(state, sector_names)}</section>
<footer>Paper money only; nothing is sent to a broker. Files are in C:\\Users\\space\\projects\\strategy-backtest\\paper.</footer>
</main></body></html>"""
    tmp = folder / "dashboard.html.tmp"
    tmp.write_text(page, encoding="utf-8")
    tmp.replace(folder / "dashboard.html")
