import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    mu = excess.mean()
    sigma = excess.std(ddof=1)
    if sigma == 0 or np.isnan(sigma):
        return np.nan
    return np.sqrt(periods_per_year) * mu / sigma

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod()

def max_drawdown(returns: pd.Series) -> float:
    eq = equity_curve(returns)
    roll_max = eq.cummax()
    drawdown = (eq - roll_max) / roll_max
    return drawdown.min()

def cagr(returns: pd.Series, periods_per_year: int = 252) -> float:
    n = returns.shape[0]
    if n == 0:
        return np.nan
    total = (1 + returns.fillna(0)).prod()
    years = n / periods_per_year
    if years == 0:
        return np.nan
    return total ** (1 / years) - 1

def summary(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> dict:
    return {
        "Sharpe": sharpe_ratio(returns, rf, periods_per_year),
        "MaxDrawdown": max_drawdown(returns),
        "CAGR": cagr(returns, periods_per_year),
        "MeanReturn": returns.mean(),
        "StdReturn": returns.std(ddof=1),
    }
