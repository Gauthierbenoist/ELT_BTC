"""Causal volume features for fixed-timeframe bars.

Same causality contract as :mod:`elt_btc.features.ohlc` (trailing windows
only), enforced by the same prefix-invariance test. ``volume`` on a bar is
the sum of the 1m volumes inside it — known at bar close, like OHLC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volume_features(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Volume activity features (same index as ``bars``).

    Per window ``w``:
    - ``volume_rel_w``: volume relative to its trailing ``w``-bar mean —
      the classic "is activity unusual right now" signal.
    - ``volume_z_w``: z-score of log1p(volume) over the trailing window —
      heavy-tail robust version of the same idea.
    """
    volume = bars["volume"]
    log_volume = pd.Series(np.log1p(volume), index=bars.index)
    out: dict[str, pd.Series] = {}
    for window in windows:
        out[f"volume_rel_{window}"] = volume / volume.rolling(window).mean()
        rolling_mean = log_volume.rolling(window).mean()
        rolling_std = log_volume.rolling(window).std()
        out[f"volume_z_{window}"] = (log_volume - rolling_mean) / rolling_std
    return pd.DataFrame(out, index=bars.index)
