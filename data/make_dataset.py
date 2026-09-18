"""
Builds a synthetic daily price history for a small 5-asset portfolio so the
rest of the project can run without a live market data feed.

Swap this out for a real pull later, e.g.:
    import yfinance as yf
    yf.download(["SPY","QQQ","EFA","TLT","GLD"], start="2019-01-01")

Innovations are drawn from a Student-t distribution instead of a normal
because equity/FX returns have fatter tails than a GBM model alone gives you,
and that fat-tailedness is exactly the thing that trips up parametric VaR
later on -- which is the point of the comparison.
"""
import numpy as np
import pandas as pd

np.random.seed(7)

ASSETS = ["SPX_IDX", "EURUSD", "UST_10Y_FUT", "WTI_CRUDE", "GOLD"]
N_DAYS = 750  # ~3 trading years

# rough annualized vol / drift assumptions per asset, nothing precise
ANNUAL_VOL = {"SPX_IDX": 0.18, "EURUSD": 0.09, "UST_10Y_FUT": 0.07,
              "WTI_CRUDE": 0.32, "GOLD": 0.15}
ANNUAL_DRIFT = {"SPX_IDX": 0.08, "EURUSD": 0.0, "UST_10Y_FUT": 0.02,
                "WTI_CRUDE": 0.03, "GOLD": 0.05}

START_PRICE = {"SPX_IDX": 4200, "EURUSD": 1.08, "UST_10Y_FUT": 112,
               "WTI_CRUDE": 78, "GOLD": 1950}

# a rough correlation matrix, not calibrated to anything real, just plausible
CORR = np.array([
    [1.00,  0.15, -0.30,  0.25, -0.10],
    [0.15,  1.00,  0.05,  0.10,  0.20],
    [-0.30, 0.05,  1.00, -0.05,  0.15],
    [0.25,  0.10, -0.05,  1.00,  0.10],
    [-0.10, 0.20,  0.15,  0.10,  1.00],
])


def build(n_days=N_DAYS, stress_block=True):
    dt = 1 / 252
    dof = 5  # student-t degrees of freedom -> fat tails

    L = np.linalg.cholesky(CORR)
    z = np.random.standard_t(dof, size=(n_days, len(ASSETS)))
    z = z @ L.T

    rets = np.zeros_like(z)
    for i, a in enumerate(ASSETS):
        mu = ANNUAL_DRIFT[a]
        sig = ANNUAL_VOL[a]
        # scale student-t so variance matches the target annual vol
        scale = sig * np.sqrt(dt) / np.sqrt(dof / (dof - 2))
        rets[:, i] = mu * dt + scale * z[:, i]

    if stress_block:
        # drop in a short synthetic stress window so backtests actually see
        # some breaches instead of a suspiciously clean run -- mimic a
        # vol spike + correlation-to-1 event like Mar 2020
        stress_start = int(n_days * 0.55)
        stress_len = 12
        shock = np.random.standard_t(4, size=(stress_len, len(ASSETS)))
        common = np.random.standard_t(4, size=stress_len)  # correlated crash
        for i, a in enumerate(ASSETS):
            sign = -1 if a != "GOLD" else 1  # gold benefits from flight to quality
            rets[stress_start:stress_start + stress_len, i] += (
                sign * 0.006 * np.abs(shock[:, i]) + sign * 0.004 * np.abs(common)
            )

    prices = np.zeros_like(rets)
    for i, a in enumerate(ASSETS):
        prices[:, i] = START_PRICE[a] * np.exp(np.cumsum(rets[:, i]))

    dates = pd.bdate_range("2022-01-03", periods=n_days)
    price_df = pd.DataFrame(prices, index=dates, columns=ASSETS)
    return price_df


if __name__ == "__main__":
    df = build()
    out_path = "data/prices.csv"
    df.to_csv(out_path)
    print(f"wrote {len(df)} rows x {len(df.columns)} assets to {out_path}")
    print(df.tail())
