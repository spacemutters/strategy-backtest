"""Download and cache every data series the backtest uses, then load them as pandas objects.

Sources (all free, no keys):
  - Ken French Data Library: US stock portfolios and T-bill rate, 1926 onward
  - Yahoo Finance (via yfinance): ETF prices with dividends reinvested
  - Bitstamp: daily BTC and ETH prices, BTC from 2011
  - BitMEX: Bitcoin perpetual funding rates, 2016 onward

Everything is cached under data/. Delete a file to force a fresh download.
Run `python data.py` to fetch everything once.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) strategy-backtest"}

# Last month every source covers. Ken French's library runs through this month.
END = "2026-08-31"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = {
    "factors_m": "F-F_Research_Data_Factors_CSV.zip",
    "factors_d": "F-F_Research_Data_Factors_daily_CSV.zip",
    "mom_deciles_m": "10_Portfolios_Prior_12_2_CSV.zip",
    "mom_deciles_d": "10_Portfolios_Prior_12_2_Daily_CSV.zip",
    "var_m": "Portfolios_Formed_on_VAR_CSV.zip",
    "op_m": "Portfolios_Formed_on_OP_CSV.zip",
    "ind12_m": "12_Industry_Portfolios_CSV.zip",
}

# ETF stand-ins for the futures markets in strategy #1, one per asset class.
TREND_ETFS = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "UUP"]
# The nine original sector SPDRs (Dec 1998). Real estate and communications came later.
SECTOR_ETFS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
# Real funds that run strategies #3, #6 and #7, for a net-of-fees check.
FACTOR_ETFS = ["MTUM", "USMV", "QUAL"]


# ---------- Ken French ----------

def _french_raw(key: str) -> str:
    path = DATA / f"french_{key}.csv"
    if not path.exists():
        r = requests.get(FRENCH_BASE + FRENCH_FILES[key], headers=UA, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        path.write_bytes(z.read(z.namelist()[0]))
    return path.read_text(encoding="latin-1")


def french_table(key: str, section: int = 0) -> pd.DataFrame:
    """Return one table from a French CSV as decimals (0.01 = 1%), indexed by period end.

    Files hold several tables (value-weighted monthly, equal-weighted monthly, annual...).
    Section 0 is the first table, which is value-weighted returns in every file used here.
    """
    lines = _french_raw(key).splitlines()
    tables, header, rows = [], None, []
    for line in lines:
        cells = [c.strip() for c in line.split(",")]
        first = cells[0]
        if first == "" and len(cells) > 1 and any(cells[1:]):
            if header is not None and rows:
                tables.append((header, rows))
            header, rows = cells[1:], []
        elif header is not None and first.isdigit() and len(first) in (6, 8):
            rows.append(cells)
        elif header is not None and rows and first == "":
            tables.append((header, rows))
            header, rows = None, []
        elif header is not None and rows and not first.isdigit():
            tables.append((header, rows))
            header, rows = None, []
    if header is not None and rows:
        tables.append((header, rows))

    header, rows = tables[section]
    stamp = rows[0][0]
    if len(stamp) == 6:
        idx = pd.to_datetime([r[0] for r in rows], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        idx = pd.to_datetime([r[0] for r in rows], format="%Y%m%d")
    df = pd.DataFrame([[float(x) for x in r[1:len(header) + 1]] for r in rows], index=idx, columns=header)
    df = df.mask(df <= -99.99) / 100.0
    return df.loc[:END]


def market_monthly() -> pd.DataFrame:
    """US stock market total return and T-bill rate, monthly: columns mkt, rf."""
    f = french_table("factors_m")
    return pd.DataFrame({"mkt": f["Mkt-RF"] + f["RF"], "rf": f["RF"]})


def market_daily() -> pd.DataFrame:
    f = french_table("factors_d")
    return pd.DataFrame({"mkt": f["Mkt-RF"] + f["RF"], "rf": f["RF"]})


# ---------- Yahoo ----------

def etf_prices(tickers: list[str]) -> pd.DataFrame:
    """Daily closes with dividends reinvested, one column per ticker."""
    out = {}
    for t in tickers:
        path = DATA / f"yahoo_{t}.csv"
        if not path.exists():
            import yfinance as yf
            df = yf.download(t, start="1990-01-01", end="2026-09-01", progress=False, auto_adjust=True)
            if df.empty:
                raise RuntimeError(f"Yahoo returned nothing for {t}")
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.rename(t).to_csv(path)
            time.sleep(1)
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        out[t] = s
    return pd.DataFrame(out).loc[:END]


# ---------- Crypto ----------

def bitstamp_daily(pair: str) -> pd.Series:
    """Daily closes from Bitstamp, e.g. pair='btcusd'."""
    path = DATA / f"bitstamp_{pair}.csv"
    if not path.exists():
        start = 1312156800  # 2011-08-01
        stop = int(pd.Timestamp("2026-09-01").timestamp())
        rows = {}
        while start < stop:
            r = requests.get(f"https://www.bitstamp.net/api/v2/ohlc/{pair}/",
                             params={"step": 86400, "limit": 1000, "start": start}, headers=UA, timeout=60)
            r.raise_for_status()
            chunk = r.json()["data"]["ohlc"]
            if not chunk:
                start += 1000 * 86400
                continue
            for c in chunk:
                rows[int(c["timestamp"])] = float(c["close"])
            last = max(int(c["timestamp"]) for c in chunk)
            start = last + 86400
            time.sleep(0.5)
        s = pd.Series(rows).sort_index()
        s.index = pd.to_datetime(s.index, unit="s")
        s = s[s > 0]
        s.rename(pair).to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.loc[:END]


def bitmex_funding(symbol: str = "XBTUSD") -> pd.Series:
    """Funding rate per funding period (8 hours; daily in 2016's first weeks), BitMEX perpetual."""
    path = DATA / f"bitmex_funding_{symbol}.csv"
    if not path.exists():
        rows, start = [], 0
        while True:
            r = requests.get("https://www.bitmex.com/api/v1/funding",
                             params={"symbol": symbol, "count": 500, "start": start, "reverse": "false"},
                             headers=UA, timeout=60)
            if r.status_code == 429:
                time.sleep(30)
                continue
            r.raise_for_status()
            chunk = r.json()
            if not chunk:
                break
            rows += [(c["timestamp"], c["fundingRate"]) for c in chunk]
            start += len(chunk)
            time.sleep(2.5)
        s = pd.Series({pd.Timestamp(t).tz_localize(None): f for t, f in rows}).sort_index()
        s.rename("funding").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    return s.loc[:"2026-08-31 23:59"]


if __name__ == "__main__":
    for k in FRENCH_FILES:
        t = french_table(k)
        print(f"french {k}: {t.index[0].date()} to {t.index[-1].date()}, columns {list(t.columns)[:6]}")
    p = etf_prices(TREND_ETFS + SECTOR_ETFS + FACTOR_ETFS)
    for c in p:
        s = p[c].dropna()
        print(f"yahoo {c}: {s.index[0].date()} to {s.index[-1].date()}")
    for pair in ("btcusd", "ethusd"):
        s = bitstamp_daily(pair)
        print(f"bitstamp {pair}: {s.index[0].date()} to {s.index[-1].date()}, {len(s)} days")
    f = bitmex_funding()
    print(f"bitmex XBTUSD funding: {f.index[0]} to {f.index[-1]}, {len(f)} periods")
