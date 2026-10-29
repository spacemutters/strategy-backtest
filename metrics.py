"""Performance numbers for monthly return series. All inputs are decimals (0.01 = 1%)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def cagr(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) == 0:
        return float("nan")
    return float((1 + r).prod() ** (12 / len(r)) - 1)


def vol(r: pd.Series) -> float:
    return float(r.dropna().std() * np.sqrt(12))


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0) -> float:
    """Annualized mean excess return over its volatility. Pass rf=0 for self-financing long-short returns."""
    ex = (r - rf).dropna()
    if len(ex) < 12 or ex.std() == 0:
        return float("nan")
    return float(ex.mean() / ex.std() * np.sqrt(12))


def max_drawdown(r: pd.Series) -> float:
    w = (1 + r.dropna()).cumprod()
    return float((w / w.cummax() - 1).min())


def summary(r: pd.Series, rf: pd.Series | float) -> dict:
    r = r.dropna()
    if isinstance(rf, pd.Series):
        rf = rf.reindex(r.index)
    return {
        "start": r.index[0].strftime("%Y-%m") if len(r) else "",
        "end": r.index[-1].strftime("%Y-%m") if len(r) else "",
        "months": len(r),
        "cagr": cagr(r),
        "vol": vol(r),
        "sharpe": sharpe(r, rf),
        "max_dd": max_drawdown(r),
    }


def breakeven_cost(r: pd.Series, bench: pd.Series) -> float:
    """Yearly cost (as a decimal) that would bring r's compound return down to bench's over the same months."""
    both = pd.concat([r, bench], axis=1).dropna()
    a, b = both.iloc[:, 0], both.iloc[:, 1]
    target = cagr(b)
    if cagr(a) <= target:
        return 0.0
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if cagr(a - mid / 12) > target:
            lo = mid
        else:
            hi = mid
    return lo


def monthly_from(r: pd.Series) -> pd.Series:
    """Compound daily or weekly returns into calendar months, indexed at month end."""
    r = r.dropna()
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    m.index = m.index.to_timestamp(how="end").normalize()
    return m
