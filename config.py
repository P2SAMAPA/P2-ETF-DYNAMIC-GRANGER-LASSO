"""
config.py  —  Configuration for Dynamic Granger-LASSO Engine
=============================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - GRANGER: Granger causality parameters
  - LASSO: Adaptive LASSO parameters
  - ROLLING: Rolling window parameters
  - WINDOWS: Time windows for analysis
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-dynamic-granger-lasso-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
WINDOW_LABELS = {
    63: "63d  (~3 months) — Short-term",
    126: "126d (~6 months) — Medium-term",
    252: "252d (~1 year) — Core Signal",
    504: "504d (~2 years) — Long-term",
}
PRIMARY_WINDOW = 252


# ── Granger Causality Parameters ────────────────────────────────────────────

GRANGER = {
    "max_lag": 5,              # Maximum lag for Granger causality
    "significance_level": 0.05, # Significance level for F-test
    "min_samples": 30,         # Minimum samples for reliable test
    "f_test": True,            # Use F-test for significance
}


# ── Adaptive LASSO Parameters ──────────────────────────────────────────────

LASSO = {
    "alpha": 0.01,             # L1 regularization strength
    "adaptive_weights": True,  # Use adaptive weights
    "max_iter": 1000,          # Maximum iterations
    "tol": 1e-4,               # Convergence tolerance
    "selection": "cv",         # Selection method: 'cv', 'bic', 'aic'
}


# ── Rolling Window Parameters ──────────────────────────────────────────────

ROLLING = {
    "window_size": 120,        # Rolling window size
    "step_size": 10,           # Step size for rolling
    "min_observations": 50,    # Minimum observations for estimation
    "decay_factor": 0.95,      # Exponential decay for connections
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
