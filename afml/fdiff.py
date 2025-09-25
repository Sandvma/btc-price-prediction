"""
Fractional differentiation utilities (López de Prado, AFML Ch. 5)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def get_weights_ffd(d: float, threshold: float = 1e-5, max_size: int = 10000) -> np.ndarray:
    """Compute fractional differentiation weights with binomial expansion.
    Stops when |w_k| < threshold or reaches max_size.
    """
    w=[1.0]
    k=1
    while k < max_size:
        w_k = -w[-1] * (d - (k-1)) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)
        k+=1
    return np.array(w, dtype=float)


def fractional_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5) -> pd.Series:
    """Fractional differencing with fixed width window (FFD).
    Uses weights from get_weights_ffd and convolution over a rolling window.
    """
    s = pd.Series(series).astype(float)
    w = get_weights_ffd(d, threshold)
    width = len(w)
    out = np.full(s.shape, np.nan, dtype=float)
    vals = s.values
    for i in range(width - 1, len(vals)):
        out[i] = np.dot(w[::-1], vals[i - width + 1:i + 1])
    return pd.Series(out, index=s.index, name=f"ffd_{d:.2f}")
