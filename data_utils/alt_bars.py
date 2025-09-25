from __future__ import annotations
import pandas as pd


def dollar_bars(df: pd.DataFrame, dollar_threshold: float, price_col: str = 'Close', volume_col: str = 'Volume') -> pd.DataFrame:
    """Build dollar bars by accumulating dollar value until threshold is crossed.

    Returns a new DataFrame with OHLCV aggregated at bar boundaries.
    """
    px = df[price_col].fillna(method='ffill')
    vol = df[volume_col].fillna(0)
    dollar = (px * vol).values

    idx = []
    cum = 0.0
    for i, val in enumerate(dollar):
        cum += float(val)
        if cum >= dollar_threshold:
            idx.append(i)
            cum = 0.0
    # ensure last bar includes remainder
    if len(idx) == 0 or idx[-1] != len(df) - 1:
        idx.append(len(df) - 1)

    bars = []
    start = 0
    for end in idx:
        chunk = df.iloc[start:end + 1]
        bars.append({
            'Open': chunk['Open'].iloc[0],
            'High': chunk['High'].max(),
            'Low': chunk['Low'].min(),
            'Close': chunk['Close'].iloc[-1],
            'Volume': chunk['Volume'].sum(),
        })
        start = end + 1
    return pd.DataFrame(bars)
