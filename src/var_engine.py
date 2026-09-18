"""
var_engine.py

Three ways of getting to the same number: 1-day VaR and Expected Shortfall
for a portfolio. The point of building all three side by side is that they
agree in calm markets and disagree in exactly the way the FRM curriculum
predicts once the return distribution gets fat-tailed / non-normal.

  1. Historical Simulation  - no distribution assumption, just resample the past
  2. Parametric (Var-Cov)   - assumes normal returns, closed form, fast, blind to tails
  3. Monte Carlo            - simulate from a fitted distribution, flexible but slow

All three return VaR as a POSITIVE number representing a loss, in the same
currency units as the portfolio value passed in.
"""
import numpy as np
import pandas as pd
from scipy import stats


class VaREngine:

    def __init__(self, returns: pd.DataFrame, weights: np.ndarray, portfolio_value: float):
        """
        returns: daily simple returns, one column per asset
        weights: portfolio weights, must sum to 1, same order as returns.columns
        portfolio_value: current mark-to-market value of the portfolio
        """
        assert abs(weights.sum() - 1.0) < 1e-6, "weights should sum to 1"
        self.returns = returns
        self.weights = weights
        self.pv = portfolio_value
        self.port_returns = returns.values @ weights  # daily portfolio return series

    # ---------- Historical Simulation ----------
    def historical_var(self, confidence=0.99, window=None):
        rets = self.port_returns if window is None else self.port_returns[-window:]
        q = np.percentile(rets, (1 - confidence) * 100)
        var = -q * self.pv
        return max(var, 0.0)

    def historical_es(self, confidence=0.99, window=None):
        rets = self.port_returns if window is None else self.port_returns[-window:]
        cutoff = np.percentile(rets, (1 - confidence) * 100)
        tail = rets[rets <= cutoff]
        if len(tail) == 0:
            return self.historical_var(confidence, window)
        return max(-tail.mean() * self.pv, 0.0)

    # ---------- Parametric / Variance-Covariance ----------
    def parametric_var(self, confidence=0.99, window=None):
        rets = self.port_returns if window is None else self.port_returns[-window:]
        mu, sigma = rets.mean(), rets.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        var = -(mu + z * sigma) * self.pv
        return max(var, 0.0)

    def parametric_es(self, confidence=0.99, window=None):
        # closed form ES under normality: mu - sigma * phi(z)/(1-c)
        rets = self.port_returns if window is None else self.port_returns[-window:]
        mu, sigma = rets.mean(), rets.std(ddof=1)
        z = stats.norm.ppf(1 - confidence)
        es = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence)) * self.pv
        return max(es, 0.0)

    # ---------- Monte Carlo (fits a Student-t so it at least captures kurtosis) ----------
    def monte_carlo_var(self, confidence=0.99, n_sims=50000, window=None, seed=42):
        rets = self.port_returns if window is None else self.port_returns[-window:]
        # fit a student-t rather than assuming normal -- cheap way to get fatter tails
        dof, loc, scale = stats.t.fit(rets)
        rng = np.random.default_rng(seed)
        sims = stats.t.rvs(dof, loc=loc, scale=scale, size=n_sims, random_state=rng)
        q = np.percentile(sims, (1 - confidence) * 100)
        return max(-q * self.pv, 0.0)

    def monte_carlo_es(self, confidence=0.99, n_sims=50000, window=None, seed=42):
        rets = self.port_returns if window is None else self.port_returns[-window:]
        dof, loc, scale = stats.t.fit(rets)
        rng = np.random.default_rng(seed)
        sims = stats.t.rvs(dof, loc=loc, scale=scale, size=n_sims, random_state=rng)
        cutoff = np.percentile(sims, (1 - confidence) * 100)
        tail = sims[sims <= cutoff]
        return max(-tail.mean() * self.pv, 0.0)

    def summary(self, confidence=0.99, window=None):
        return pd.DataFrame({
            "VaR": {
                "Historical": self.historical_var(confidence, window),
                "Parametric": self.parametric_var(confidence, window),
                "Monte Carlo": self.monte_carlo_var(confidence, window=window),
            },
            "ES": {
                "Historical": self.historical_es(confidence, window),
                "Parametric": self.parametric_es(confidence, window),
                "Monte Carlo": self.monte_carlo_es(confidence, window=window),
            },
        })
