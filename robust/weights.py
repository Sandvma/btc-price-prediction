import numpy as np
import pandas as pd


def overlap_weights(events: pd.DataFrame, prices: pd.Series, tl_col: str = "t1") -> pd.Series:
    """Compute uniqueness weights for overlapping events.

    Each event has a start index (events.index) and an end time in column `tl_col`.
    The weight is the average inverse of the number of concurrent events over its lifespan.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame where index are start times and column tl_col gives end times.
    prices : pd.Series
        Price series used to define the final time if t1 is NaN.
    tl_col : str, optional
        Column name for end times, by default "t1".

    Returns
    -------
    pd.Series
        Series of uniqueness weights indexed by event start times.
    """
    # Determine all timestamps where events start or end
    all_times = set(events.index)
    all_times.update(events[tl_col].dropna())
    ts = sorted(all_times)
    # Count concurrent events at each timestamp
    count = pd.Series(0.0, index=ts)
    for t0, t1 in events[tl_col].items():
        if pd.isna(t1):
            t1 = prices.index[-1]
        count.loc[t0:t1] += 1.0
    # Compute weight for each event as mean inverse concurrency
    weights = {}
    for t0, t1 in events[tl_col].items():
        if pd.isna(t1):
            t1 = prices.index[-1]
        weights[t0] = float((1.0 / count.loc[t0:t1]).mean())
    return pd.Series(weights)
