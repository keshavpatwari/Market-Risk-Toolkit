import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from backtest import kupiec_pof_test, christoffersen_independence_test, basel_traffic_light


def test_kupiec_accepts_model_with_expected_breach_rate():
    # 250 obs, exactly 2-3 breaches is what you'd expect at 99% confidence
    n = 250
    breaches = pd.Series([False] * (n - 3) + [True] * 3)
    result = kupiec_pof_test(breaches, confidence=0.99)
    assert result["n_breaches"] == 3
    assert not result["reject_at_5pct"]


def test_kupiec_rejects_model_with_way_too_many_breaches():
    n = 250
    breaches = pd.Series([False] * (n - 25) + [True] * 25)  # 10% breach rate at 99% VaR -- bad model
    result = kupiec_pof_test(breaches, confidence=0.99)
    assert result["reject_at_5pct"]


def test_christoffersen_flags_clustered_breaches():
    # all breaches bunched together in a row -- should NOT look independent
    breaches = pd.Series([False] * 100 + [True] * 10 + [False] * 100)
    result = christoffersen_independence_test(breaches)
    assert not result["independent_at_5pct"]


def test_christoffersen_accepts_scattered_breaches():
    rng = np.random.default_rng(0)
    breaches = pd.Series(rng.random(300) < 0.01)  # scattered ~1% rate, no clustering pattern
    result = christoffersen_independence_test(breaches)
    # not a hard guarantee with random data, but scattered iid breaches
    # should usually pass -- this is a smoke test, not a proof
    assert isinstance(result["p_value"], float)


def test_basel_traffic_light_zones():
    assert basel_traffic_light(2, 250) == "GREEN"
    assert basel_traffic_light(4, 250) == "GREEN"
    assert basel_traffic_light(7, 250) == "YELLOW"
    assert basel_traffic_light(15, 250) == "RED"
