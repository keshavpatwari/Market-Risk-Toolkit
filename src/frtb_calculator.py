"""
frtb_calculator.py

A simplified implementation of the FRTB Standardized Approach,
Sensitivities-Based Method (SBM), covering delta risk for three risk
classes: GIRR, Equity, and FX.

IMPORTANT SCOPE NOTE (say this out loud in an interview, it shows judgement
rather than gaps): this is a teaching implementation. It does NOT include
vega/curvature risk, the Default Risk Charge, or the Residual Risk Add-On,
and the risk weights/correlations below are illustrative values inspired by
the BCBS framework, not pulled from a current regulatory rulebook. A real
implementation would source risk weights and correlation parameters from the
latest local regulator's technical standards.

Mechanics implemented, which ARE the actual FRTB logic:
  1. Weighted sensitivity WS_k = risk_weight_k * sensitivity_k
  2. Intra-bucket aggregation:
       K_b = sqrt( sum(WS_k^2) + sum_{k != l} rho_kl * WS_k * WS_l )
  3. Cross-bucket aggregation using bucket correlation gamma_bc
  4. Three correlation scenarios (low / medium / high) -- final capital
     charge for a risk class is the WORST (max) of the three
"""
import numpy as np


class FRTBSensitivity:
    def __init__(self, risk_class: str, bucket: str, factor: str,
                 sensitivity: float, risk_weight: float):
        self.risk_class = risk_class
        self.bucket = bucket
        self.factor = factor
        self.sensitivity = sensitivity  # e.g. PV impact of a 1bp / 1% move
        self.risk_weight = risk_weight
        self.ws = sensitivity * risk_weight


def _bucket_capital(sensitivities, intra_bucket_rho):
    """K_b for a single bucket given a flat intra-bucket correlation."""
    ws = np.array([s.ws for s in sensitivities])
    if len(ws) == 0:
        return 0.0, ws
    sum_sq = np.sum(ws ** 2)
    cross = 0.0
    for i in range(len(ws)):
        for j in range(len(ws)):
            if i != j:
                cross += intra_bucket_rho * ws[i] * ws[j]
    k_b = np.sqrt(max(sum_sq + cross, 0.0))  # floor at 0, correlations can make this negative
    return k_b, ws


def _cross_bucket_capital(bucket_data, gamma_bc):
    """
    bucket_data: dict bucket_name -> (K_b, sum_of_WS_in_bucket)
    gamma_bc: correlation between buckets (flat, simplification)
    """
    names = list(bucket_data.keys())
    k = np.array([bucket_data[n][0] for n in names])
    s = np.array([bucket_data[n][1] for n in names])  # signed sum, for cross terms

    sum_sq = np.sum(k ** 2)
    cross = 0.0
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j:
                cross += gamma_bc * s[i] * s[j]
    total = np.sqrt(max(sum_sq + cross, 0.0))
    return total


def delta_risk_charge(sensitivities: list, intra_bucket_rho=0.6, gamma_bc=0.15):
    """
    sensitivities: list of FRTBSensitivity for ONE risk class
    Returns capital charge under low/medium/high correlation scenarios and
    picks the max, per the FRTB requirement.
    """
    buckets = {}
    for s in sensitivities:
        buckets.setdefault(s.bucket, []).append(s)

    scenarios = {"low": 0.75, "medium": 1.0, "high": 1.25}
    results = {}
    for scen_name, mult in scenarios.items():
        rho = min(intra_bucket_rho * mult, 1.0)
        gamma = min(gamma_bc * mult, 1.0)
        bucket_data = {}
        for b_name, b_sens in buckets.items():
            k_b, ws = _bucket_capital(b_sens, rho)
            bucket_data[b_name] = (k_b, ws.sum())
        results[scen_name] = _cross_bucket_capital(bucket_data, gamma)

    worst_case = max(results.values())
    return worst_case, results


def risk_class_summary(portfolio: dict, correlations: dict):
    """
    portfolio: {risk_class: [FRTBSensitivity, ...]}
    correlations: {risk_class: (intra_bucket_rho, gamma_bc)}
    """
    out = {}
    for rc, sens in portfolio.items():
        rho, gamma = correlations.get(rc, (0.6, 0.15))
        charge, scen = delta_risk_charge(sens, rho, gamma)
        out[rc] = {"capital_charge": charge, "scenarios": scen}
    out["TOTAL_SBM_DELTA"] = sum(v["capital_charge"] for k, v in out.items())
    return out
