#!/usr/bin/env python3
"""
drone_allocation.py — Computational Component

Risk-Constrained Operator Allocation for Autonomous Drone Swarms:
A Stochastic Framework for Multi-Zone Command and Control

Notes:
- Hardware tested on: Macbook Pro (M5, 24GB RAM, 2025)
- Last updated: Nov 2025

"""

import argparse
import sys
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

import numpy as np
from scipy import stats
import pandas as pd

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ═══════════════════════════════════════════════════════════════
# 1. CONFIGURATION
@dataclass
class SystemConfig:
    """Configuration for a multi-zone autonomous drone C2 system."""
    n_zones: int = 2             # M — number of operational zones
    lambda_day: float = 0.15     # daily target-acquisition probability per drone
    tau_minutes: float = 30.0    # intervention duration (minutes)
    n_flex: int = 1              # F — shared flex operators
    theta: float = 0.01          # maximum acceptable per-window overload probability

    @property
    def windows_per_day(self) -> float:
        """W = 1440 / τ non-overlapping intervention windows per day."""
        return 1440.0 / self.tau_minutes

    @property
    def p_window(self) -> float:
        """Per-window intervention probability p."""
        return intervention_probability(self.lambda_day, self.tau_minutes)

    @property
    def total_operators(self) -> int:
        """Total operators = M dedicated + F flex."""
        return self.n_zones + self.n_flex

    def summary(self) -> str:
        lines = [
            "═" * 60,
            "  SYSTEM CONFIGURATION",
            "═" * 60,
            f"  Zones (M)                  : {self.n_zones}",
            f"  Flex operators (F)         : {self.n_flex}",
            f"  Total operators (M + F)    : {self.total_operators}",
            f"  Daily acq. probability (λ) : {self.lambda_day:.4f}",
            f"  Intervention time (τ)      : {self.tau_minutes:.0f} min",
            f"  Windows per day (W)        : {self.windows_per_day:.0f}",
            f"  Per-window probability (p) : {self.p_window:.6f}",
            f"  Risk threshold (θ)         : {self.theta:.4f} ({self.theta:.2%})",
            "═" * 60,
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# 2. CORE PROBABILITY FUNCTIONS
def intervention_probability(lambda_day: float, tau_minutes: float) -> float:
    """
    Convert daily acquisition probability λ to per-window probability p.

    p = 1 − (1 − λ)^(1/W),  where W = 1440/τ.
    """
    if lambda_day <= 0:
        return 0.0
    if lambda_day >= 1:
        return 1.0
    W = 1440.0 / tau_minutes
    return 1.0 - (1.0 - lambda_day) ** (1.0 / W)


def overload_prob_two_zone(n: int, p: float) -> float:
    """
    Exact overload probability for symmetric two-zone, one-flex system.

    P_OL = 1 − α² − 2αβ,  where α = P(K ≤ 1), β = P(K = 2), K ~ Bin(n, p).

    Implements Theorem 4.1 from the paper.
    """
    if n <= 0 or p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    rv = stats.binom(n, p)
    alpha = rv.cdf(1)   # P(K ≤ 1)
    beta = rv.pmf(2)    # P(K = 2)
    result = 1.0 - alpha * alpha - 2.0 * alpha * beta
    return max(0.0, result)


def overload_prob_general(n_per_zone: int, p: float,
                          n_zones: int, n_flex: int) -> float:
    """
    General overload probability for M symmetric zones with F flex operators.

    Uses convolution of excess-demand PMFs (Theorem 5.1).
    """
    if n_per_zone <= 0 or p <= 0:
        return 0.0
    if p >= 1.0:
        return 1.0 if n_per_zone > 1 else 0.0

    rv = stats.binom(n_per_zone, p)

    # Build single-zone excess demand PMF: E = max(0, K − 1)
    max_e = min(n_per_zone, 30)  # truncate at 30 for efficiency
    excess_pmf = np.zeros(max_e + 1)
    excess_pmf[0] = rv.cdf(1)  # P(K ≤ 1) → E = 0
    for e in range(1, max_e + 1):
        excess_pmf[e] = rv.pmf(e + 1)  # P(K = e+1) → E = e
    # Ensure normalisation (truncation residual)
    residual = 1.0 - excess_pmf.sum()
    if residual > 0:
        excess_pmf[-1] += residual

    # M-fold convolution for total excess demand S = Σ E_m
    total_pmf = excess_pmf.copy()
    for _ in range(n_zones - 1):
        total_pmf = np.convolve(total_pmf, excess_pmf)

    # P(overload) = P(S > F)
    if n_flex + 1 < len(total_pmf):
        return float(total_pmf[n_flex + 1:].sum())
    return 0.0


def overload_prob_centralized(n_total: int, p: float,
                               n_operators: int) -> float:
    """
    Overload probability for a centralized operator pool.

    All n_total drones share n_operators interchangeable operators.
    P_OL = P(K > n_operators), K ~ Bin(n_total, p).
    """
    if n_total <= 0 or p <= 0:
        return 0.0
    rv = stats.binom(n_total, p)
    return float(1.0 - rv.cdf(n_operators))


# ═══════════════════════════════════════════════════════════════
# 3. FLEET-SIZE OPTIMIZATION
def max_drones_per_zone(p: float, n_zones: int = 2,
                        n_flex: int = 1, theta: float = 0.01) -> int:
    """
    Find maximum drones per zone such that P_OL ≤ θ, using binary search.

    Returns n* (integer).
    """
    if p <= 0:
        return 999999  # effectively unlimited
    lo, hi = 1, 500

    # Expand upper bound until we exceed the threshold
    while overload_prob_general(hi, p, n_zones, n_flex) < theta:
        hi *= 2
        if hi > 500000:
            return hi  # effectively unlimited

    # Binary search
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if overload_prob_general(mid, p, n_zones, n_flex) <= theta:
            lo = mid
        else:
            hi = mid - 1

    return lo


def max_drones_centralized(p: float, n_operators: int,
                           theta: float = 0.01) -> int:
    """Max drones for centralized pool with n_operators total."""
    if p <= 0:
        return 999999
    lo, hi = 1, 500
    while overload_prob_centralized(hi, p, n_operators) < theta:
        hi *= 2
        if hi > 500000:
            return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if overload_prob_centralized(mid, p, n_operators) <= theta:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ═══════════════════════════════════════════════════════════════
# 4. DAILY RISK AGGREGATION
def daily_overload_probability(n_per_zone: int, lambda_day: float,
                                tau_minutes: float, n_zones: int,
                                n_flex: int) -> float:
    """
    Probability of ≥ 1 overload event in a 24-hour period.

    Treats windows as independent:  P_day = 1 − (1 − P_window)^W.
    """
    p = intervention_probability(lambda_day, tau_minutes)
    p_window = overload_prob_general(n_per_zone, p, n_zones, n_flex)
    W = 1440.0 / tau_minutes
    return 1.0 - (1.0 - p_window) ** W


# ═══════════════════════════════════════════════════════════════
# 5. MONTE CARLO SIMULATION
def monte_carlo_overload(n_per_zone: int, p: float,
                         n_zones: int, n_flex: int,
                         n_sims: int = 1_000_000,
                         seed: int = 42) -> Tuple[float, float]:
    """
    Monte Carlo estimate of per-window overload probability.

    Returns (point_estimate, standard_error).
    """
    rng = np.random.default_rng(seed)
    K = rng.binomial(n_per_zone, p, size=(n_sims, n_zones))
    E = np.maximum(K - 1, 0)
    total_excess = E.sum(axis=1)
    overload = total_excess > n_flex
    p_hat = overload.mean()
    se = np.sqrt(p_hat * (1.0 - p_hat) / n_sims)
    return float(p_hat), float(se)


# ═══════════════════════════════════════════════════════════════
# 6. ANALYSIS / TABLE GENERATION
def baseline_demand_table(cfg: SystemConfig, n: int) -> pd.DataFrame:
    """Probability distribution of intervention demand for a single zone."""
    p = cfg.p_window
    rv = stats.binom(n, p)
    rows = []
    for k in range(min(n + 1, 6)):
        rows.append({"Demand K": k, "P(K=k)": rv.pmf(k),
                      "P(K=k) %": f"{rv.pmf(k) * 100:.4f}%"})
    rows.append({"Demand K": "≥6", "P(K=k)": 1 - rv.cdf(5),
                  "P(K=k) %": f"{(1 - rv.cdf(5)) * 100:.6f}%"})
    return pd.DataFrame(rows)


def overload_vs_fleet_table(cfg: SystemConfig) -> pd.DataFrame:
    """Overload probability for varying fleet sizes."""
    p = cfg.p_window
    rows = []
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 94, 95, 100,
              110, 120, 130, 140, 150, 175, 200]:
        p_ol_w = overload_prob_general(n, p, cfg.n_zones, cfg.n_flex)
        p_ol_d = daily_overload_probability(n, cfg.lambda_day,
                                            cfg.tau_minutes,
                                            cfg.n_zones, cfg.n_flex)
        rows.append({
            "Drones/Zone": n,
            "Total Drones": n * cfg.n_zones,
            "P_OL (window)": f"{p_ol_w:.6f}",
            "P_OL (window) %": f"{p_ol_w * 100:.4f}%",
            "P_OL (daily)": f"{p_ol_d:.4f}",
            "P_OL (daily) %": f"{p_ol_d * 100:.1f}%",
        })
    return pd.DataFrame(rows)


def operator_scaling_table(cfg: SystemConfig, max_flex: int = 8) -> pd.DataFrame:
    """Maximum fleet size vs. flex operator count."""
    p = cfg.p_window
    rows = []
    for f in range(1, max_flex + 1):
        n_max = max_drones_per_zone(p, cfg.n_zones, f, cfg.theta)
        total = n_max * cfg.n_zones
        total_ops = cfg.n_zones + f
        p_ol = overload_prob_general(n_max, p, cfg.n_zones, f)
        rows.append({
            "Flex Ops (F)": f,
            "Total Ops": total_ops,
            "Max n/Zone": n_max,
            "Total Drones": total,
            "Drones/Op": round(total / total_ops, 1),
            "P_OL": f"{p_ol:.6f}",
        })
    return pd.DataFrame(rows)


def sensitivity_table(cfg: SystemConfig) -> pd.DataFrame:
    """Max total drones as function of λ and τ."""
    tau_values = [10, 15, 20, 30, 45, 60]
    lambda_values = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    rows = []
    for lam in lambda_values:
        row = {"λ": lam}
        for tau in tau_values:
            p = intervention_probability(lam, tau)
            n_max = max_drones_per_zone(p, cfg.n_zones, cfg.n_flex, cfg.theta)
            row[f"τ={tau}"] = n_max * cfg.n_zones
        rows.append(row)
    return pd.DataFrame(rows)


def risk_threshold_table(cfg: SystemConfig) -> pd.DataFrame:
    """Max fleet size vs. risk threshold θ."""
    thetas = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
    p = cfg.p_window
    rows = []
    for theta in thetas:
        n_max = max_drones_per_zone(p, cfg.n_zones, cfg.n_flex, theta)
        p_ol = overload_prob_general(n_max, p, cfg.n_zones, cfg.n_flex)
        rows.append({
            "θ": f"{theta:.1%}",
            "Max n/Zone": n_max,
            "Total Drones": n_max * cfg.n_zones,
            "Actual P_OL": f"{p_ol:.6f}",
        })
    return pd.DataFrame(rows)


def pooling_comparison_table(cfg: SystemConfig) -> pd.DataFrame:
    """Decentralized vs. centralized capacity comparison."""
    p = cfg.p_window
    thetas = [0.001, 0.005, 0.01, 0.02, 0.05]
    rows = []
    for theta in thetas:
        n_dec = max_drones_per_zone(p, cfg.n_zones, cfg.n_flex, theta)
        total_dec = n_dec * cfg.n_zones
        total_cen = max_drones_centralized(p, cfg.total_operators, theta)
        gap = (total_cen - total_dec) / total_cen * 100 if total_cen > 0 else 0
        rows.append({
            "θ": f"{theta:.1%}",
            "Decentralized (total)": total_dec,
            "Centralized (total)": total_cen,
            "Capacity Gap": f"{gap:.1f}%",
        })
    return pd.DataFrame(rows)


def monte_carlo_validation_table(cfg: SystemConfig,
                                  n_sims: int = 1_000_000) -> pd.DataFrame:
    """Compare analytical and Monte Carlo results."""
    p = cfg.p_window
    rows = []
    for n in [30, 50, 70, 94, 100, 120, 150]:
        analytical = overload_prob_general(n, p, cfg.n_zones, cfg.n_flex)
        mc_est, mc_se = monte_carlo_overload(n, p, cfg.n_zones,
                                             cfg.n_flex, n_sims)
        rows.append({
            "n/Zone": n,
            "Analytical": f"{analytical:.6f}",
            "MC Estimate": f"{mc_est:.6f}",
            "MC 95% CI": f"±{1.96 * mc_se:.6f}",
            "Match": "✓" if abs(analytical - mc_est) < 3 * mc_se else "✗",
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
# 7. FIGURE GENERATION
FIGURE_DIR = "figures"
COLORS = ["#2171b5", "#e6550d", "#31a354", "#756bb1", "#d62728"]


def _ensure_fig_dir():
    os.makedirs(FIGURE_DIR, exist_ok=True)


def plot_overload_vs_drones(cfg: SystemConfig):
    """Figure 1: P_OL vs fleet size for different flex-operator counts."""
    _ensure_fig_dir()
    p = cfg.p_window
    n_range = np.arange(1, 301)

    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, f in enumerate([1, 2, 3, 4]):
        probs = [overload_prob_general(int(n), p, cfg.n_zones, f)
                 for n in n_range]
        ax.plot(n_range, probs, color=COLORS[idx], linewidth=2,
                label=f"F = {f} flex operator{'s' if f > 1 else ''}")

    ax.axhline(y=cfg.theta, color="red", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"θ = {cfg.theta:.0%} threshold")
    ax.set_xlabel("Drones per Zone (n)", fontsize=13)
    ax.set_ylabel("P(System Overload) per Window", fontsize=13)
    ax.set_title("System Overload Probability vs. Fleet Size per Zone", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 1)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig1_overload_vs_drones.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_sensitivity_heatmap(cfg: SystemConfig):
    """Figure 2: Heatmap of max total drones as function of (τ, λ)."""
    _ensure_fig_dir()
    tau_vals = np.arange(5, 65, 2)
    lam_vals = np.arange(0.02, 1.01, 0.02)
    Z = np.zeros((len(lam_vals), len(tau_vals)))
    for i, lam in enumerate(lam_vals):
        for j, tau in enumerate(tau_vals):
            p = intervention_probability(lam, tau)
            n_max = max_drones_per_zone(p, cfg.n_zones, cfg.n_flex, cfg.theta)
            Z[i, j] = n_max * cfg.n_zones

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[tau_vals[0], tau_vals[-1],
                           lam_vals[0], lam_vals[-1]],
                   cmap="viridis", interpolation="bilinear")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Maximum Total Drones", fontsize=12)
    ax.set_xlabel("Intervention Time τ (minutes)", fontsize=13)
    ax.set_ylabel("Daily Acquisition Probability λ", fontsize=13)
    ax.set_title(f"Maximum Fleet Capacity  (M = {cfg.n_zones},  "
                 f"F = {cfg.n_flex},  θ = {cfg.theta:.0%})", fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig2_sensitivity_heatmap.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_monte_carlo_validation(cfg: SystemConfig,
                                 n_sims: int = 500_000):
    """Figure 3: Analytical vs. Monte Carlo comparison."""
    _ensure_fig_dir()
    p = cfg.p_window
    n_range = np.arange(10, 201, 5)

    analytical, mc_est, mc_err = [], [], []
    for n in n_range:
        a = overload_prob_general(int(n), p, cfg.n_zones, cfg.n_flex)
        m, se = monte_carlo_overload(int(n), p, cfg.n_zones,
                                     cfg.n_flex, n_sims)
        analytical.append(a)
        mc_est.append(m)
        mc_err.append(1.96 * se)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_range, analytical, "b-", linewidth=2.5, label="Analytical (Exact)",
            zorder=3)
    ax.errorbar(n_range, mc_est, yerr=mc_err, fmt="o", color="#e6550d",
                markersize=3.5, capsize=2, linewidth=1,
                label=f"Monte Carlo (N = {n_sims:,}, 95% CI)", zorder=2)
    ax.axhline(y=cfg.theta, color="green", linestyle="--", linewidth=1.5,
               alpha=0.7, label=f"θ = {cfg.theta:.0%}")
    ax.set_xlabel("Drones per Zone (n)", fontsize=13)
    ax.set_ylabel("P(System Overload) per Window", fontsize=13)
    ax.set_title("Analytical vs. Monte Carlo Validation", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1)
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig3_monte_carlo_validation.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_operator_scaling(cfg: SystemConfig, max_flex: int = 8):
    """Figure 4: Fleet capacity scaling with flex operators."""
    _ensure_fig_dir()
    p = cfg.p_window
    flex_range = list(range(1, max_flex + 1))
    per_zone, total, drones_per_op = [], [], []

    for f in flex_range:
        n_max = max_drones_per_zone(p, cfg.n_zones, f, cfg.theta)
        per_zone.append(n_max)
        t = n_max * cfg.n_zones
        total.append(t)
        drones_per_op.append(t / (cfg.n_zones + f))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    ax1.bar(flex_range, total, color="#2171b5", alpha=0.85, edgecolor="white")
    ax1.set_xlabel("Flex Operators (F)", fontsize=13)
    ax1.set_ylabel("Maximum Total Drones", fontsize=13)
    ax1.set_title(f"Fleet Capacity vs. Flex Operators  (θ = {cfg.theta:.0%})",
                  fontsize=13)
    ax1.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(total):
        ax1.text(flex_range[i], v + max(total) * 0.01, str(v),
                 ha="center", fontsize=10, fontweight="bold")

    ax2.plot(flex_range, drones_per_op, "o-", color="#e6550d",
             linewidth=2, markersize=8)
    ax2.set_xlabel("Flex Operators (F)", fontsize=13)
    ax2.set_ylabel("Drones per Operator", fontsize=13)
    ax2.set_title("Operator Efficiency vs. Flex Operators", fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig4_operator_scaling.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_pooling_comparison(cfg: SystemConfig):
    """Figure 5: Decentralized vs. centralized capacity."""
    _ensure_fig_dir()
    p = cfg.p_window
    thetas = np.logspace(-4, -0.5, 30)
    dec_total, cen_total = [], []

    for theta in thetas:
        n_d = max_drones_per_zone(p, cfg.n_zones, cfg.n_flex, theta)
        dec_total.append(n_d * cfg.n_zones)
        cen_total.append(max_drones_centralized(p, cfg.total_operators, theta))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thetas * 100, cen_total, "-", color="#31a354", linewidth=2.5,
            label="Centralized Pool")
    ax.plot(thetas * 100, dec_total, "--", color="#2171b5", linewidth=2.5,
            label="Decentralized (Zone-Based)")
    ax.fill_between(thetas * 100, dec_total, cen_total,
                    alpha=0.15, color="gray", label="Capacity Gap")
    ax.set_xlabel("Risk Threshold θ (%)", fontsize=13)
    ax.set_ylabel("Maximum Total Drones", fontsize=13)
    ax.set_title("Centralized vs. Decentralized C2 Capacity", fontsize=14)
    ax.set_xscale("log")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, which="both")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig5_pooling_comparison.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_daily_risk_surface(cfg: SystemConfig):
    """Figure 6: Daily overload risk vs. fleet size and acquisition rate."""
    _ensure_fig_dir()
    n_range = np.arange(10, 201, 5)
    lam_range = np.arange(0.05, 0.55, 0.05)

    Z = np.zeros((len(lam_range), len(n_range)))
    for i, lam in enumerate(lam_range):
        for j, n in enumerate(n_range):
            Z[i, j] = daily_overload_probability(
                int(n), lam, cfg.tau_minutes, cfg.n_zones, cfg.n_flex)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(Z * 100, aspect="auto", origin="lower",
                   extent=[n_range[0], n_range[-1],
                           lam_range[0], lam_range[-1]],
                   cmap="RdYlGn_r", interpolation="bilinear",
                   vmin=0, vmax=100)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Daily Overload Probability (%)", fontsize=12)
    ax.set_xlabel("Drones per Zone (n)", fontsize=13)
    ax.set_ylabel("Daily Acquisition Probability λ", fontsize=13)
    ax.set_title(f"Daily System Overload Risk  "
                 f"(M = {cfg.n_zones},  F = {cfg.n_flex},  "
                 f"τ = {cfg.tau_minutes:.0f} min)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "fig6_daily_risk.pdf")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# 8. PRINTING HELPERS
def _header(title: str):
    w = 60
    print()
    print("─" * w)
    print(f"  {title}")
    print("─" * w)


def print_all_tables(cfg: SystemConfig, n_sims: int = 500_000):
    """Print every analysis table."""
    n_star = max_drones_per_zone(cfg.p_window, cfg.n_zones,
                                 cfg.n_flex, cfg.theta)

    _header("BASELINE DEMAND DISTRIBUTION  (n = n*)")
    print(baseline_demand_table(cfg, n_star).to_string(index=False))

    _header("OVERLOAD PROBABILITY vs. FLEET SIZE")
    print(overload_vs_fleet_table(cfg).to_string(index=False))

    _header(f"MAXIMUM FLEET SIZE:  n* = {n_star}/zone,  "
            f"{n_star * cfg.n_zones} total")
    p_ol = overload_prob_general(n_star, cfg.p_window,
                                 cfg.n_zones, cfg.n_flex)
    p_day = daily_overload_probability(n_star, cfg.lambda_day,
                                       cfg.tau_minutes,
                                       cfg.n_zones, cfg.n_flex)
    print(f"  Per-window overload: {p_ol:.6f} ({p_ol:.4%})")
    print(f"  Per-day overload:    {p_day:.4f}  ({p_day:.2%})")

    _header("OPERATOR SCALING  (varying F)")
    print(operator_scaling_table(cfg).to_string(index=False))

    _header("SENSITIVITY ANALYSIS  (max total drones)")
    print(sensitivity_table(cfg).to_string(index=False))

    _header("RISK THRESHOLD ANALYSIS")
    print(risk_threshold_table(cfg).to_string(index=False))

    _header("CENTRALIZED vs. DECENTRALIZED CAPACITY")
    print(pooling_comparison_table(cfg).to_string(index=False))


def print_monte_carlo(cfg: SystemConfig, n_sims: int = 1_000_000):
    """Print Monte Carlo validation."""
    _header(f"MONTE CARLO VALIDATION  (N = {n_sims:,} simulations)")
    print(monte_carlo_validation_table(cfg, n_sims).to_string(index=False))


# ═══════════════════════════════════════════════════════════════
# 9. MAIN
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Risk-Constrained Operator Allocation for "
                    "Autonomous Drone Swarms — Computational Component",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python drone_allocation.py --all
  python drone_allocation.py --tables --figures
  python drone_allocation.py --all --lambda-day 0.20 --tau 15
  python drone_allocation.py --all --zones 3 --flex 2 --theta 0.005
        """,
    )
    # Analysis flags
    ap.add_argument("--all", action="store_true",
                    help="Run all analyses (tables + figures + MC)")
    ap.add_argument("--tables", action="store_true",
                    help="Print all analysis tables")
    ap.add_argument("--figures", action="store_true",
                    help="Generate all figures")
    ap.add_argument("--monte-carlo", action="store_true",
                    help="Run Monte Carlo validation")

    # Parameters
    ap.add_argument("--lambda-day", type=float, default=0.15,
                    help="Daily target-acquisition probability (default: 0.15)")
    ap.add_argument("--tau", type=float, default=30.0,
                    help="Intervention duration in minutes (default: 30)")
    ap.add_argument("--zones", type=int, default=2,
                    help="Number of operational zones M (default: 2)")
    ap.add_argument("--flex", type=int, default=1,
                    help="Number of flex operators F (default: 1)")
    ap.add_argument("--theta", type=float, default=0.01,
                    help="Per-window overload risk threshold (default: 0.01)")
    ap.add_argument("--mc-sims", type=int, default=1_000_000,
                    help="Monte Carlo replications (default: 1000000)")
    return ap


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve what to run
    do_tables = args.all or args.tables
    do_figures = args.all or args.figures
    do_mc = args.all or args.monte_carlo

    if not (do_tables or do_figures or do_mc):
        print("No action selected. Use --all, --tables, --figures, "
              "or --monte-carlo.")
        print("Run with --help for full usage information.")
        sys.exit(0)

    cfg = SystemConfig(
        n_zones=args.zones,
        lambda_day=args.lambda_day,
        tau_minutes=args.tau,
        n_flex=args.flex,
        theta=args.theta,
    )

    print(cfg.summary())

    if do_tables:
        print_all_tables(cfg)

    if do_mc:
        print_monte_carlo(cfg, args.mc_sims)

    if do_figures:
        _header("GENERATING FIGURES")
        print("  This may take a minute for heatmaps and MC validation ...\n")
        plot_overload_vs_drones(cfg)
        plot_sensitivity_heatmap(cfg)
        plot_monte_carlo_validation(cfg, n_sims=min(args.mc_sims, 500_000))
        plot_operator_scaling(cfg)
        plot_pooling_comparison(cfg)
        plot_daily_risk_surface(cfg)
        print(f"\n  All figures saved to ./{FIGURE_DIR}/")

    print("\n" + "═" * 60)
    print("  COMPLETE")
    print("═" * 60)


if __name__ == "__main__":
    main()
