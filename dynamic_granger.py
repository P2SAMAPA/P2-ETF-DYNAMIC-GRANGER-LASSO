"""
dynamic_granger.py  —  Dynamic Granger-LASSO Engine
====================================================

Implements:
- Adaptive LASSO Granger causality
- Rolling window estimation
- Time-varying connectivity matrix
- Connection decay penalization
"""

import numpy as np
import pandas as pd
from scipy.stats import f
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


class AdaptiveLASSO:
    """
    Adaptive LASSO for Granger causality.
    
    Uses adaptive weights to penalize connections that fade over time.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.alpha = config.get("alpha", 0.01)
        self.max_iter = config.get("max_iter", 1000)
        self.tol = config.get("tol", 1e-4)
        self.adaptive_weights = config.get("adaptive_weights", True)
        
    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray = None) -> Dict:
        """
        Fit adaptive LASSO.
        
        Args:
            X: Predictor matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            weights: Adaptive weights for each feature
        
        Returns:
            coefficients: Fitted coefficients
            selected: Selected features
        """
        n_samples, n_features = X.shape
        
        # Standardize
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0) + 1e-6
        X_norm = (X - X_mean) / X_std
        
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-6
        y_norm = (y - y_mean) / y_std
        
        # Initialize coefficients
        beta = np.zeros(n_features)
        
        # Adaptive weights
        if self.adaptive_weights and weights is not None:
            w = weights
        elif self.adaptive_weights:
            # Initial Ridge estimate for adaptive weights
            ridge_beta = np.linalg.lstsq(X_norm.T @ X_norm + np.eye(n_features) * 0.1, X_norm.T @ y_norm, rcond=None)[0]
            w = 1 / (np.abs(ridge_beta) + 1e-6)
            w = w / np.max(w)
        else:
            w = np.ones(n_features)
        
        # Coordinate descent for LASSO
        for iteration in range(self.max_iter):
            beta_old = beta.copy()
            
            for j in range(n_features):
                # Compute residual without feature j
                residual = y_norm - X_norm @ beta + X_norm[:, j] * beta[j]
                
                # Compute correlation
                corr = X_norm[:, j] @ residual
                
                # Soft thresholding with adaptive weight
                lambda_j = self.alpha * w[j]
                if corr > lambda_j:
                    beta[j] = (corr - lambda_j) / (X_norm[:, j] @ X_norm[:, j] + 1e-6)
                elif corr < -lambda_j:
                    beta[j] = (corr + lambda_j) / (X_norm[:, j] @ X_norm[:, j] + 1e-6)
                else:
                    beta[j] = 0
            
            # Check convergence
            if np.max(np.abs(beta - beta_old)) < self.tol:
                break
        
        # Denormalize coefficients
        beta_denorm = beta * (y_std / X_std)
        
        # Selected features
        selected = np.abs(beta) > 1e-6
        
        return {
            "coefficients": beta_denorm,
            "coefficients_norm": beta,
            "selected": selected,
            "n_selected": np.sum(selected),
            "iterations": iteration + 1
        }


class DynamicGrangerLASSO:
    """
    Dynamic Granger-LASSO for time-varying causal connectivity.
    
    Computes rolling-window Granger causality with adaptive LASSO
    that penalizes fading connections.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.max_lag = config.get("max_lag", 5)
        self.alpha = config.get("alpha", 0.01)
        self.window_size = config.get("window_size", 120)
        self.step_size = config.get("step_size", 10)
        self.min_observations = config.get("min_observations", 50)
        self.decay_factor = config.get("decay_factor", 0.95)
        
        self.lasso = AdaptiveLASSO(config)
        self.causality_history = []
        
    def granger_test(self, X: np.ndarray, y: np.ndarray, lag: int) -> Dict:
        """
        Perform Granger causality test using adaptive LASSO.
        
        Args:
            X: Predictor variable (n_samples,)
            y: Target variable (n_samples,)
            lag: Number of lags
        
        Returns:
            causality_score: Granger causality score
            selected_lags: Selected lags
            coefficients: LASSO coefficients
        """
        n_samples = len(y)
        
        if n_samples < self.min_observations:
            return {
                "causality_score": 0,
                "selected_lags": [],
                "coefficients": [],
                "significant": False
            }
        
        # Create lagged features
        X_lags = []
        for l in range(1, lag + 1):
            X_lags.append(X[:-l] if l < len(X) else np.zeros(len(y) - lag))
        
        # Align lengths
        min_len = min(len(y) - lag, len(X))
        y_aligned = y[lag:lag + min_len]
        X_lags_aligned = []
        for l in range(1, lag + 1):
            if l <= len(X):
                X_lags_aligned.append(X[lag-l:lag-l+min_len])
            else:
                X_lags_aligned.append(np.zeros(min_len))
        
        # Also include autoregressive terms
        y_lags = []
        for l in range(1, lag + 1):
            if l <= len(y_aligned):
                y_lags.append(y[lag-l:lag-l+len(y_aligned)])
            else:
                y_lags.append(np.zeros(len(y_aligned)))
        
        # Combine features
        features = np.column_stack(y_lags + X_lags_aligned)
        
        # Fit LASSO
        result = self.lasso.fit(features, y_aligned)
        
        # Check if X lags are selected
        n_y_lags = len(y_lags)
        x_coefficients = result["coefficients"][n_y_lags:]
        x_selected = result["selected"][n_y_lags:]
        
        # Causality score: sum of absolute X coefficients
        causality_score = np.sum(np.abs(x_coefficients))
        
        # Significance test (F-test)
        # Unrestricted model (with X lags)
        unrestricted_rss = np.sum((y_aligned - features @ result["coefficients"]) ** 2)
        
        # Restricted model (without X lags)
        y_features = features[:, :n_y_lags]
        y_beta = np.linalg.lstsq(y_features, y_aligned, rcond=None)[0]
        restricted_rss = np.sum((y_aligned - y_features @ y_beta) ** 2)
        
        # F-statistic
        n_x_lags = len(X_lags_aligned)
        df1 = n_x_lags
        df2 = len(y_aligned) - len(result["coefficients"]) - 1
        
        if df2 > 0:
            f_stat = ((restricted_rss - unrestricted_rss) / df1) / (unrestricted_rss / df2)
            p_value = 1 - f.cdf(f_stat, df1, df2)
            significant = p_value < 0.05
        else:
            p_value = 1.0
            significant = False
        
        selected_lags = [l for l, s in enumerate(x_selected, 1) if s]
        
        return {
            "causality_score": causality_score,
            "selected_lags": selected_lags,
            "coefficients": x_coefficients.tolist() if len(x_coefficients) > 0 else [],
            "significant": significant,
            "p_value": p_value,
            "f_stat": f_stat if df2 > 0 else 0
        }
    
    def compute_connectivity_matrix(self, data: np.ndarray, var_names: List[str]) -> Dict:
        """
        Compute full connectivity matrix using Granger-LASSO.
        
        Returns:
            matrix: NxN causality matrix
            scores: Causality scores
        """
        n_vars = len(var_names)
        n_samples = data.shape[1] if data.ndim > 1 else len(data)
        
        # Reshape data if needed
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        matrix = np.zeros((n_vars, n_vars))
        scores = {}
        
        # Compute pairwise Granger causality
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                
                X = data[i, :]
                y = data[j, :]
                
                result = self.granger_test(X, y, self.max_lag)
                scores[f"{var_names[i]}→{var_names[j]}"] = result
                matrix[i, j] = result["causality_score"] if result["significant"] else 0
        
        return {
            "matrix": matrix,
            "scores": scores,
            "var_names": var_names
        }
    
    def rolling_window_analysis(self, data: np.ndarray, var_names: List[str]) -> List[Dict]:
        """
        Compute rolling-window Granger causality analysis.
        
        Returns:
            windows: List of window results with time-varying connectivity
        """
        n_samples = data.shape[1] if data.ndim > 1 else len(data)
        window_size = self.window_size
        step_size = self.step_size
        
        results = []
        
        for start in range(0, n_samples - window_size, step_size):
            end = start + window_size
            window_data = data[:, start:end]
            
            # Decay weights for connections from previous window
            if results:
                decay_weights = self._compute_decay_weights(results[-1]["matrix"])
            else:
                decay_weights = None
            
            # Compute connectivity for this window
            connectivity = self.compute_connectivity_matrix(window_data, var_names)
            
            # Apply decay penalty to connections that faded
            if decay_weights is not None:
                connectivity["matrix"] = connectivity["matrix"] * decay_weights
            
            results.append({
                "start": start,
                "end": end,
                "connectivity": connectivity,
                "var_names": var_names
            })
        
        return results
    
    def _compute_decay_weights(self, prev_matrix: np.ndarray) -> np.ndarray:
        """Compute decay weights for connections."""
        decay = self.decay_factor
        weights = np.ones_like(prev_matrix)
        
        # Connections that were weak in previous window get penalized
        prev_strength = np.abs(prev_matrix)
        weights = 1 - (1 - decay) * (1 - prev_strength / (np.max(prev_strength) + 1e-6))
        
        return weights


def compute_dynamic_granger(
    prices: pd.Series,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute Dynamic Granger-LASSO for a single ticker.
    """
    returns = np.log(prices / prices.shift(1)).dropna().values
    
    if len(returns) < window:
        return {
            "causality_score": 0,
            "causality_strength": 0,
            "z_score": 0,
            "error": "Insufficient data"
        }
    
    try:
        # Use recent window
        train_returns = returns[-window:]
        
        # For single ticker, compute causality relative to macro
        # Use multiple lags as features
        if len(macro_df) > 0:
            macro = macro_df.values[-window:]
            # Compute Granger causality from macro to returns
            engine = DynamicGrangerLASSO(config)
            
            # Use macro as predictor for returns
            macro_flat = macro.flatten()[:min(10, len(macro))]
            macro_features = np.array([macro_flat[:len(train_returns)]])
            
            # Compute causality
            result = engine.granger_test(macro_features.flatten(), train_returns, config.get("max_lag", 5))
            causality_score = result["causality_score"]
            significant = result.get("significant", False)
        else:
            # Use lagged returns as features
            engine = DynamicGrangerLASSO(config)
            lagged_returns = train_returns[:-config.get("max_lag", 5)]
            result = engine.granger_test(lagged_returns, train_returns[config.get("max_lag", 5):], config.get("max_lag", 5))
            causality_score = result["causality_score"]
            significant = result.get("significant", False)
        
        # Compute signal
        signal = causality_score * (1 if significant else 0.5)
        
        return {
            "causality_score": causality_score,
            "causality_strength": signal,
            "z_score": signal,
            "significant": significant,
            "selected_lags": result.get("selected_lags", []),
            "p_value": result.get("p_value", 1.0),
            "error": None
        }
    except Exception as e:
        return {
            "causality_score": 0,
            "causality_strength": 0,
            "z_score": 0,
            "error": str(e)
        }


def compute_universe_dynamic_granger(
    prices_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    config: Dict,
    window: int = 252
) -> Dict:
    """
    Compute Dynamic Granger-LASSO for all ETFs in a universe.
    """
    results = {}
    
    for ticker in prices_df.columns:
        prices = prices_df[ticker]
        result = compute_dynamic_granger(prices, macro_df, config, window)
        
        results[ticker] = {
            "causality_score": result.get("causality_score", 0),
            "causality_strength": result.get("causality_strength", 0),
            "z_score": result.get("z_score", 0),
            "significant": result.get("significant", False),
            "selected_lags": result.get("selected_lags", []),
            "p_value": result.get("p_value", 1.0)
        }
    
    # Normalize z-scores
    z_values = np.array([r["z_score"] for r in results.values()])
    if len(z_values) > 1 and np.std(z_values) > 1e-6:
        mean_z = np.mean(z_values)
        std_z = np.std(z_values)
        for ticker, r in results.items():
            r["z_score"] = (r["z_score"] - mean_z) / std_z
    else:
        # Fallback: use causality score
        scores = np.array([r["causality_score"] for r in results.values()])
        if len(scores) > 1 and np.std(scores) > 1e-6:
            mean_s = np.mean(scores)
            std_s = np.std(scores)
            for ticker, r in results.items():
                r["z_score"] = (r["causality_score"] - mean_s) / std_s
        else:
            for r in results.values():
                r["z_score"] = 0
    
    return results
