"""
Basic sanity tests for var_engine.py. Not exhaustive -- just enough to catch
a broken sign convention or a method that silently returns garbage.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from var_engine import VaREngine


@pytest.fixture
def sample_returns():
    # fixed seed so this test is deterministic across runs
    rng = np.random.default_rng(123)
    n, k = 500, 3
    data = rng.normal(loc=0.0002, scale=0.01, size=(n, k))
    dates = pd.bdate_range("2023-01-01", periods=n)
    return pd.DataFrame(data, index=dates, columns=["A", "B", "C"])


def test_weights_must_sum_to_one(sample_returns):
    bad_weights = np.array([0.5, 0.5, 0.5])
    with pytest.raises(AssertionError):
        VaREngine(sample_returns, bad_weights, 1_000_000)


def test_var_is_non_negative(sample_returns):
    weights = np.array([0.4, 0.3, 0.3])
    eng = VaREngine(sample_returns, weights, 1_000_000)
    assert eng.historical_var() >= 0
    assert eng.parametric_var() >= 0
    assert eng.monte_carlo_var(n_sims=2000) >= 0


def test_es_is_at_least_var(sample_returns):
    # ES should never be smaller than VaR at the same confidence level --
    # it's an average of losses beyond the VaR cutoff, so it's at least as bad
    weights = np.array([0.4, 0.3, 0.3])
    eng = VaREngine(sample_returns, weights, 1_000_000)
    assert eng.historical_es() >= eng.historical_var() - 1e-6
    assert eng.parametric_es() >= eng.parametric_var() - 1e-6


def test_higher_confidence_gives_higher_var(sample_returns):
    weights = np.array([0.4, 0.3, 0.3])
    eng = VaREngine(sample_returns, weights, 1_000_000)
    var_95 = eng.historical_var(confidence=0.95)
    var_99 = eng.historical_var(confidence=0.99)
    assert var_99 >= var_95


def test_var_scales_with_portfolio_value(sample_returns):
    weights = np.array([0.4, 0.3, 0.3])
    eng_small = VaREngine(sample_returns, weights, 1_000_000)
    eng_big = VaREngine(sample_returns, weights, 2_000_000)
    assert eng_big.historical_var() == pytest.approx(eng_small.historical_var() * 2, rel=1e-6)
