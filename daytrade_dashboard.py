"""One self-contained HTML page for the day-trading paper bot. Rewritten after every run."""
from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#1b1b1a;--muted:#5d5d58;--line:#e2e2dc;--up:#1d6b3a;--down:#a3261d;--accent:#1b1b1a}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--card:#1b1b1a;--ink:#f0f0ec;--muted:#a6a69e;--line:#2c2c29;--up:#6fcf8f;--down:#ff8a7e;--accent:#f0f0ec}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:26px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 22px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px}
.tile b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.tile span{color:var(--muted);font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:18px;overflow-x:auto}
h2{font-size:17px;margin:0 0 10px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:14px}
td,th{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{color:var(--muted);font-weight:600}
td:first-child,th:first-child{text-align:left}.up{color:var(--up)}.down{color:var(--down)}svg{width:100%;height:220px}
"""


def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x:+.2%}"


def cls(x: float) -> str:
    return "up" if x > 0 else ("down" if x < 0 else "")


def chart(days: list[dict], capital: float) -> str:
    if len(days) < 2:
        return "<p class='sub'>The chart appears after two finished trading days.</p>"
    bot = [capital] + [d["equity"] for d in days]
    spy = [capital] + [d["spy_hold"] for d in days]
    lo, hi = min(bot + spy), max(bot + spy)
    pad = (hi - lo) * 0.08 or 1
    lo, hi = lo - pad, hi + pad
    w, h = 940, 200

    def path(vals):
        n = len(vals) - 1
        return " ".join(f"{'M' if i == 0 else 'L'}{i / n * w:.1f},{h - (v - lo) / (hi - lo) * h:.1f}"
                        for i, v in enumerate(vals))
    return (f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none' role='img' "
            f"aria-label='Account value against holding SPY'>"
            f"<path d='{path(spy)}' fill='none' stroke='var(--muted)' stroke-width='2' stroke-dasharray='5 4'/>"
            f"<path d='{path(bot)}' fill='none' stroke='var(--accent)' stroke-width='2.5'/></svg>"
            "<p class='sub'>Solid: the bot. Dashed: just holding SPY.</p>")


def write(state: dict, st: dict, out: Path) -> None:
    cap = state["capital"]
    days = state["days"]
    if st:
        tiles = [("Account", money(st["final"]), pct(st["return"])),
                 ("Holding SPY instead", money(st["spy_hold"]), pct(st["spy_hold"] / cap - 1)),
                 ("Worst drop from a high", pct(st["max_drawdown"]), ""),
                 ("Trades", f"{st['trades']:,}", f"{st['sessions']} of 63 days")]
    else:
        tiles = [("Account", money(cap), "waiting for the first trading day"),
                 ("Trades", "0", "")]
    tile_html = "".join(f"<div class='tile'><span>{a}</span><b>{b}</b><span>{c}</span></div>" for a, b, c in tiles)

    third = cap / 3
    names = {"ORB": "Opening range breakout (QQQ)", "VWAP": "VWAP trend (QQQ)", "MOM": "Last half hour (SPY)"}
    sleeve_rows = "".join(
        f"<tr><td>{names[k]}</td><td>{money(v)}</td><td class='{cls(v - third)}'>{pct(v / third - 1)}</td></tr>"
        for k, v in state["equity"].items())

    day_rows = "".join(
        f"<tr><td>{d['day']}</td><td>{d['trades']}</td><td class='{cls(d['pnl'])}'>{d['pnl']:+,.2f}</td>"
        f"<td>{money(d['equity'])}</td></tr>" for d in reversed(days[-30:]))

    trades_path = out.parent / "trades.csv"
    trade_rows = ""
    if trades_path.exists():
        t = pd.read_csv(trades_path).tail(40).iloc[::-1]
        trade_rows = "".join(
            f"<tr><td>{html.escape(str(r.time)[5:16].replace('T', ' '))}</td><td>{r.sleeve}</td><td>{r.ticker}</td>"
            f"<td>{r.side}</td><td>{abs(r.shares):,.0f}</td><td>{r.price:,.2f}</td>"
            f"<td>{html.escape(str(r.why))}</td></tr>" for r in t.itertuples())

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Day-Trading Paper Bot</title>
<style>{CSS}</style></head><body><main>
<h1>Day-trading paper bot</h1>
<p class="sub">Paper money only. Started {state['start']} with {money(cap)}. Up to 4x leverage, flat every night.
Last updated after the {state['last']} session.</p>
<div class="tiles">{tile_html}</div>
<div class="card"><h2>Account vs. holding SPY</h2>{chart(days, cap)}</div>
<div class="card"><h2>The three rules</h2><table><tr><th>Rule</th><th>Now</th><th>Change</th></tr>{sleeve_rows}</table></div>
<div class="card"><h2>Days</h2><table><tr><th>Day</th><th>Trades</th><th>Profit</th><th>Account</th></tr>{day_rows or '<tr><td>None yet</td></tr>'}</table></div>
<div class="card"><h2>Latest trades</h2><table><tr><th>When</th><th>Rule</th><th>Fund</th><th>Side</th><th>Shares</th><th>Price</th><th>Why</th></tr>{trade_rows or '<tr><td>None yet</td></tr>'}</table></div>
</main></body></html>"""
    out.write_text(page, encoding="utf-8")
