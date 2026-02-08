# Optimal Human-on-the-Loop Architectures for Autonomous Drone Swarms

**A Risk-Constrained Stochastic Framework for Multi-Zone Command and Control**

---

## Overview

This project provides a complete research paper and computational implementation for determining optimal operator-to-drone ratios in multi-zone autonomous drone operations under human-on-the-loop command-and-control constraints.

The framework answers a critical national security question: *How many autonomous platforms can a human operator structure responsibly oversee while maintaining meaningful human judgment over lethal engagement decisions?*

## Project Structure

```
drone-operator-allocation/
├── main.tex                 # Complete LaTeX paper (~25 pages)
├── drone_allocation.py      # Python computational component
├── requirements.txt         # Python dependencies
├── figures/                 # Generated figures (created by code)
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Analysis

```bash
python drone_allocation.py --all
```

This produces all tables (printed to stdout) and all figures (saved to `./figures/`).

### 3. Compile the Paper

```bash
pdflatex main.tex
pdflatex main.tex   # run twice for table of contents
```

## Usage Examples

```bash
# Tables only
python drone_allocation.py --tables

# Figures only
python drone_allocation.py --figures

# Monte Carlo validation only
python drone_allocation.py --monte-carlo

# Custom parameters: 3 zones, 2 flex ops, 20% daily rate, 15-min intervention
python drone_allocation.py --all --zones 3 --flex 2 --lambda-day 0.20 --tau 15 --theta 0.005
```

## Key Results (Baseline Configuration)

| Parameter             | Value          |
|-----------------------|----------------|
| Zones                 | 2              |
| Daily acquisition rate| 15%            |
| Intervention time     | 30 min         |
| Operators             | 3 (2 ded + 1 flex) |
| **Max drones (θ=1%)** | **~190 total** |
| Drones per operator   | ~63            |

## Mathematical Framework

The model treats target acquisitions as independent Bernoulli trials within discrete intervention windows. Key constructs:

- **Per-window probability**: `p = 1 − (1 − λ)^(1/W)` where `W = 1440/τ`
- **Demand per zone**: `K ~ Bin(n, p)`
- **Excess demand**: `E = max(0, K − 1)`
- **Overload**: `Σ E_m > F` (total excess exceeds flex operator count)

For the two-zone, one-flex case:

```
P_OL = 1 − α² − 2αβ
```

where `α = P(K ≤ 1)` and `β = P(K = 2)`.

## Generated Figures

| Figure | Description |
|--------|-------------|
| `fig1_overload_vs_drones` | Overload probability vs. fleet size for varying flex operators |
| `fig2_sensitivity_heatmap` | Max fleet capacity as function of (τ, λ) |
| `fig3_monte_carlo_validation` | Analytical vs. Monte Carlo comparison |
| `fig4_operator_scaling` | Fleet capacity and efficiency vs. operator count |
| `fig5_pooling_comparison` | Centralized vs. decentralized C2 capacity |
| `fig6_daily_risk` | Daily overload risk surface |

## Author

Tadhg Taylor-McGreal

## License

This work is provided for academic and defense research purposes.
