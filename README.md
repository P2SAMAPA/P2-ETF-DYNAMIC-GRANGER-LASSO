# P2-DYNAMIC-GRANGER-LASSO

**Dynamic Granger-LASSO — Time-Varying Causal Connectivity**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Engine Does

This engine computes a **rolling-window Granger causality test** using **adaptive LASSO** that penalizes connections that fade over time, producing a **living causality matrix** that evolves with market conditions.

### Theory

**Granger Causality:**
- X Granger-causes Y if past values of X help predict Y
- Captures predictive relationships in time series

**Adaptive LASSO:**
- L1 regularization with adaptive weights
- Selects sparse set of causal connections
- Penalizes connections that fade over time

**Rolling Window:**
- Slides a window over time
- Captures evolving relationships
- Produces time-varying connectivity

**Decay Penalty:**
- Connections that weaken over time are penalized
- Emphasizes recent, stable relationships

---

## Key Metrics

| Metric | What it tells you |
|--------|-------------------|
| **z-score** | Cross-sectional ranking of causality strength |
| **Causality Score** | Strength of causal relationship |
| **Significant** | Whether causality is statistically significant |
| **p-value** | Statistical significance |
| **Selected Lags** | Which lags are predictive |

---

## Windows

| Window | Purpose |
|--------|---------|
| 63d | Short-term causal connectivity |
| 126d | Medium-term causal connectivity |
| 252d | Core signal (primary) |
| 504d | Long-term causal connectivity |

---

## Interpretation

| z-score | Action | Meaning |
|---------|--------|---------|
| **> 0.1** | BUY | Strong causal relationship detected |
| **-0.1 to 0.1** | HOLD | Weak or no causal relationship |
| **< -0.1** | SELL | Negative causal relationship |

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-DYNAMIC-GRANGER-LASSO
cd P2-DYNAMIC-GRANGER-LASSO
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday.

Required secret: HF_TOKEN

References
Granger, C. W. J. (1969). Investigating Causal Relations by Econometric Models and Cross-spectral Methods. Econometrica.

Zou, H. (2006). The Adaptive LASSO and Its Oracle Properties. Journal of the American Statistical Association.

Runge, J., et al. (2019). Detecting and quantifying causal associations in large nonlinear time series datasets. Science Advances.
