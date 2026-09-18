"""
main.py

Run end to end:
  1. Load price history, build a 5-asset portfolio
  2. Compute 1-day 99% VaR and ES three ways (historical / parametric / MC)
  3. Rolling backtest each method over the sample, run Kupiec + Christoffersen
  4. Plot P&L vs VaR with breaches marked
  5. Run a sample FRTB SA delta risk charge on a small hypothetical book

Usage: python src/main.py
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))  # so var_engine/backtest import cleanly

from var_engine import VaREngine
from backtest import rolling_backtest, kupiec_pof_test, christoffersen_independence_test, basel_traffic_light
from frtb_calculator import FRTBSensitivity, risk_class_summary

CONFIDENCE = 0.99
PORTFOLIO_VALUE = 10_000_000  # $10mm notional, arbitrary but round
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_returns():
    prices = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "prices.csv"),
                          index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return returns


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_var_section(returns):
    print_header("1-DAY 99% VaR & EXPECTED SHORTFALL -- FULL SAMPLE")
    weights = np.array([0.35, 0.15, 0.20, 0.15, 0.15])  # equity-tilted book
    assert len(weights) == returns.shape[1]

    eng = VaREngine(returns, weights, PORTFOLIO_VALUE)
    summary = eng.summary(confidence=CONFIDENCE)
    print(summary.round(0))

    spread = summary["VaR"].max() - summary["VaR"].min()
    print(f"\nSpread between methods: ${spread:,.0f} "
          f"({spread / summary['VaR'].mean() * 100:.1f}% of the average VaR)")
    print("Parametric usually understates the tail here because returns aren't normal "
          "-- that gap is the whole reason FRTB pushed toward ES.")
    return weights


def run_backtest_section(returns, weights):
    print_header("ROLLING BACKTEST (250-day window, 99% VaR)")
    for method in ["historical", "parametric", "monte_carlo"]:
        bt = rolling_backtest(returns, weights, PORTFOLIO_VALUE, method=method,
                               window=250, confidence=CONFIDENCE)
        kupiec = kupiec_pof_test(bt["breach"], confidence=CONFIDENCE)
        christ = christoffersen_independence_test(bt["breach"])
        zone = basel_traffic_light(kupiec["n_breaches"], kupiec["n_obs"])

        print(f"\n[{method.upper()}]")
        print(f"  observations: {kupiec['n_obs']}, breaches: {kupiec['n_breaches']} "
              f"(expected ~{kupiec['expected_rate'] * kupiec['n_obs']:.1f})")
        print(f"  Basel traffic light zone: {zone}")
        print(f"  Kupiec POF p-value: {kupiec['p_value']:.3f} "
              f"({'REJECT model' if kupiec['reject_at_5pct'] else 'not rejected'} at 5%)")
        print(f"  Christoffersen independence p-value: {christ['p_value']:.3f} "
              f"({'breaches cluster' if not christ['independent_at_5pct'] else 'no clustering'})")

        if method == "historical":
            plot_backtest(bt, method)

    return bt  # last one (monte_carlo) returned for convenience, not used further


def plot_backtest(bt, method):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bt.index, bt["pnl"], label="Daily P&L", color="#2b6cb0", linewidth=1)
    ax.plot(bt.index, -bt["var"], label="-VaR threshold", color="#c53030",
             linewidth=1, linestyle="--")
    breaches = bt[bt["breach"]]
    ax.scatter(breaches.index, breaches["pnl"], color="#c53030", zorder=5,
               label=f"Breaches (n={len(breaches)})", s=25)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.set_title(f"P&L vs {method.title()} VaR (99%, 250d rolling window)")
    ax.set_ylabel("USD")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"backtest_{method}.png")
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  saved chart -> {out_path}")


def run_frtb_section():
    print_header("FRTB STANDARDIZED APPROACH -- DELTA RISK CHARGE (illustrative)")
    # a small hypothetical trading book: a couple of equity positions, one
    # rates position, one FX position -- sensitivities are made up PV01/delta
    # figures for the demo, not pulled from a real book
    portfolio = {
        "GIRR": [
            FRTBSensitivity("GIRR", "USD", "10Y", sensitivity=45_000, risk_weight=0.017),
            FRTBSensitivity("GIRR", "USD", "5Y", sensitivity=-20_000, risk_weight=0.020),
        ],
        "Equity": [
            FRTBSensitivity("Equity", "LargeCap_DM", "SPX", sensitivity=3_500_000, risk_weight=0.25),
            FRTBSensitivity("Equity", "LargeCap_DM", "SX5E", sensitivity=-1_200_000, risk_weight=0.25),
        ],
        "FX": [
            FRTBSensitivity("FX", "G10", "EURUSD", sensitivity=1_500_000, risk_weight=0.15),
        ],
    }
    correlations = {
        "GIRR": (0.5, 0.5),      # tenors within a curve are more correlated
        "Equity": (0.15, 0.15),  # separate names in the same bucket, lower corr
        "FX": (0.6, 0.6),
    }

    result = risk_class_summary(portfolio, correlations)
    for rc, data in result.items():
        if rc == "TOTAL_SBM_DELTA":
            continue
        print(f"  {rc:10s} capital charge: ${data['capital_charge']:,.0f}  "
              f"(scenarios -> {', '.join(f'{k}: {v:,.0f}' for k, v in data['scenarios'].items())})")
    print(f"\n  TOTAL SBM DELTA CHARGE: ${result['TOTAL_SBM_DELTA']:,.0f}")
    print("  (real capital = this + vega + curvature + Default Risk Charge + RRAO,"
          " not included here)")


if __name__ == "__main__":
    returns = load_returns()
    weights = run_var_section(returns)
    run_backtest_section(returns, weights)
    run_frtb_section()
