import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from frtb_calculator import FRTBSensitivity, delta_risk_charge, risk_class_summary


def test_single_sensitivity_has_no_diversification():
    # with only one sensitivity there are no cross terms, so K_b should just
    # be the absolute weighted sensitivity
    sens = [FRTBSensitivity("Equity", "LargeCap_DM", "SPX", sensitivity=1_000_000, risk_weight=0.25)]
    charge, scenarios = delta_risk_charge(sens)
    expected = abs(1_000_000 * 0.25)
    assert np.isclose(charge, expected)


def test_offsetting_positions_reduce_charge_vs_gross():
    # two fully offsetting positions in the same bucket, high correlation,
    # should net down a lot relative to just adding the two gross charges
    sens = [
        FRTBSensitivity("Equity", "LargeCap_DM", "SPX", sensitivity=1_000_000, risk_weight=0.25),
        FRTBSensitivity("Equity", "LargeCap_DM", "SX5E", sensitivity=-1_000_000, risk_weight=0.25),
    ]
    charge, _ = delta_risk_charge(sens, intra_bucket_rho=0.9)
    gross = 1_000_000 * 0.25 + 1_000_000 * 0.25
    assert charge < gross


def test_worst_case_scenario_is_the_max():
    sens = [
        FRTBSensitivity("FX", "G10", "EURUSD", sensitivity=1_500_000, risk_weight=0.15),
        FRTBSensitivity("FX", "G10", "GBPUSD", sensitivity=800_000, risk_weight=0.15),
    ]
    charge, scenarios = delta_risk_charge(sens)
    assert charge == max(scenarios.values())


def test_risk_class_summary_totals_add_up():
    portfolio = {
        "FX": [FRTBSensitivity("FX", "G10", "EURUSD", sensitivity=1_000_000, risk_weight=0.15)],
        "Equity": [FRTBSensitivity("Equity", "LargeCap_DM", "SPX", sensitivity=500_000, risk_weight=0.25)],
    }
    correlations = {"FX": (0.6, 0.6), "Equity": (0.15, 0.15)}
    result = risk_class_summary(portfolio, correlations)
    manual_total = result["FX"]["capital_charge"] + result["Equity"]["capital_charge"]
    assert np.isclose(result["TOTAL_SBM_DELTA"], manual_total)
