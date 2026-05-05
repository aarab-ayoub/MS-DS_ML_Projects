from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.config import STATES


def estimate_transition_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Estimate an empirical DTMC transition matrix from consecutive state indices.
    """
    n_states = len(STATES)
    if "state_idx" not in df.columns:
        raise KeyError("Expected `state_idx` in dataframe.")

    s = df["state_idx"].to_numpy(dtype=int)
    counts = np.zeros((n_states, n_states), dtype=float)

    if len(s) > 1:
        for i in range(len(s) - 1):
            counts[s[i], s[i + 1]] += 1.0

    # Keep `FailedPermanent` absorbing if a row has no observations.
    row_sums = counts.sum(axis=1, keepdims=True)
    p = np.zeros_like(counts)
    for i in range(n_states):
        if row_sums[i, 0] > 0:
            p[i] = counts[i] / row_sums[i, 0]
        else:
            p[i, i] = 1.0
    return p


def estimate_generator_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Optional CTMC generator estimate from transitions and holding times.
    """
    n_states = len(STATES)
    if "state_idx" not in df.columns or "time_delta_hours" not in df.columns:
        raise KeyError("Expected `state_idx` and `time_delta_hours` in dataframe.")

    s = df["state_idx"].to_numpy(dtype=int)
    dt = df["time_delta_hours"].to_numpy(dtype=float)

    jump_counts = np.zeros((n_states, n_states), dtype=float)
    sojourn = np.zeros(n_states, dtype=float)

    for i in range(1, len(s)):
        prev = s[i - 1]
        cur = s[i]
        sojourn[prev] += max(dt[i], 1e-9)
        if prev != cur:
            jump_counts[prev, cur] += 1.0

    q = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        if sojourn[i] <= 0:
            continue
        for j in range(n_states):
            if i != j:
                q[i, j] = jump_counts[i, j] / sojourn[i]
        q[i, i] = -np.sum(q[i, :])
    return q

