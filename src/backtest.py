"""
backtest.py

Runs a rolling VaR backtest: for each day, compute VaR using only data up to
that point (no lookahead), then check whether the next day's actual loss
breached it. Then run the two standard statistical tests on the breach
sequence:

  - Kupiec POF test: are there the RIGHT NUMBER of breaches?
  - Christoffersen independence test: are the breaches CLUSTERED in time
    (which would suggest the model reacts too slowly to changing vol)?

Basel traffic light zones are also applied as a sanity check most risk teams
still quote informally even though FRTB has moved capital treatment on.
"""
import numpy as np
import pandas as pd
from scipy import stats

from var_engine import VaREngine


def rolling_backtest(returns: pd.DataFrame, weights: np.ndarray, portfolio_value: float,
                      method="historical", window=250, confidence=0.99):
    """
    method: 'historical', 'parametric', or 'monte_carlo'
    window: lookback used to estimate VaR each day (rolling, not expanding)
    """
    n = len(returns)
    port_ret = returns.values @ weights
    var_series = np.full(n, np.nan)
    pnl_series = np.full(n, np.nan)

    for t in range(window, n):
        hist_window = returns.iloc[t - window:t]
        eng = VaREngine(hist_window, weights, portfolio_value)
        if method == "historical":
            v = eng.historical_var(confidence)
        elif method == "parametric":
            v = eng.parametric_var(confidence)
        elif method == "monte_carlo":
            v = eng.monte_carlo_var(confidence, n_sims=5000)  # smaller n for speed in a loop
        else:
            raise ValueError(method)
        var_series[t] = v
        pnl_series[t] = port_ret[t] * portfolio_value  # realized P&L on day t

    df = pd.DataFrame({"pnl": pnl_series, "var": var_series}, index=returns.index)
    df = df.dropna()
    df["breach"] = df["pnl"] < -df["var"]
    return df


def kupiec_pof_test(breaches: pd.Series, confidence=0.99):
    """Proportion of failures test. Null: observed breach rate == 1-confidence."""
    n = len(breaches)
    x = breaches.sum()
    p = 1 - confidence
    p_hat = x / n

    if x == 0:
        # avoid log(0); LR still well-defined in the limit
        lr = -2 * n * np.log(1 - p)
    else:
        ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
        ll_alt = (n - x) * np.log(1 - p_hat) + x * np.log(p_hat)
        lr = -2 * (ll_null - ll_alt)

    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {"n_obs": n, "n_breaches": int(x), "breach_rate": p_hat,
            "expected_rate": p, "LR_stat": lr, "p_value": p_value,
            "reject_at_5pct": p_value < 0.05}


def christoffersen_independence_test(breaches: pd.Series):
    """Tests whether breaches cluster (i.e. today's breach predicts tomorrow's)."""
    b = breaches.astype(int).values
    n00 = n01 = n10 = n11 = 0
    for i in range(1, len(b)):
        prev, cur = b[i - 1], b[i]
        if prev == 0 and cur == 0:
            n00 += 1
        elif prev == 0 and cur == 1:
            n01 += 1
        elif prev == 1 and cur == 0:
            n10 += 1
        else:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1) if (n0 + n1) > 0 else 0

    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    ll_null = (n0 + n1 - n01 - n11) * _safe_log(1 - pi) + (n01 + n11) * _safe_log(pi)
    ll_alt = (n00 * _safe_log(1 - pi01) + n01 * _safe_log(pi01) +
              n10 * _safe_log(1 - pi11) + n11 * _safe_log(pi11))
    lr = -2 * (ll_null - ll_alt)
    p_value = 1 - stats.chi2.cdf(lr, df=1)
    return {"LR_stat": lr, "p_value": p_value, "independent_at_5pct": p_value >= 0.05}


def basel_traffic_light(n_breaches: int, n_obs: int = 250):
    """Basel III framework: green <=4, yellow 5-9, red >=10 breaches per 250 obs."""
    scaled = n_breaches * 250 / n_obs
    if scaled <= 4:
        return "GREEN"
    elif scaled <= 9:
        return "YELLOW"
    else:
        return "RED"
