"""Causal volume features for fixed-timeframe bars.

Same causality contract as :mod:`elt_btc.features.ohlc` (trailing windows
only), enforced by the same prefix-invariance test. ``volume`` on a bar is
the sum of the 1m volumes inside it — known at bar close, like OHLC.

The window is expressed in **prediction bars**, so the feature scales with
the model timeframe automatically: with ``dataset.timeframe = "4h"`` a
window of 20 spans 20×4h, and switching to ``"1d"`` makes the same 20 span
20 days — no change needed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_volume_features(bars: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Volume z-score of the just-closed bar (same index as ``bars``).

    Per window ``w``:
    - ``volume_z_w``: z-score of the just-closed bar's activity against the
      distribution of the ``w - 1`` bars *strictly before* it. Volume is
      log-normal, so the score is computed on ``log1p(volume)`` (heavy-tail
      robust) as ``(x_t - mean) / std`` where ``mean``/``std`` come from the
      prior ``w - 1`` bars.

    The baseline excludes the current bar (``shift(1)``), so a bar never
    contributes to its own mean/std: with ``w = 20`` the reference is the
    previous 19 bars. This is strictly trailing — no look-ahead.

    Warm-up rows (fewer than ``w - 1`` prior bars) contain NaN.
    """
    log_volume = pd.Series(np.log1p(bars["volume"]), index=bars.index)
    prior = log_volume.shift(1)
    out: dict[str, pd.Series] = {}
    for window in windows:
        if window < 3:
            raise ValueError(f"volume window must be >= 3 for a z-score (got {window})")
        baseline_mean = prior.rolling(window - 1).mean()
        baseline_std = prior.rolling(window - 1).std()
        out[f"volume_z_{window}"] = (log_volume - baseline_mean) / baseline_std
    return pd.DataFrame(out, index=bars.index)
