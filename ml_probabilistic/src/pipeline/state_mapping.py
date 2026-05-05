from __future__ import annotations

import pandas as pd

from ..core.config import STATE_TO_IDX


def map_to_state(row: pd.Series) -> str:
    """
    Map telemetry-derived signals to one of the 5 canonical reliability states.

    Metrics-only heuristic (no direct label dependency):
    - `stress_score` (0..1 proxy from temp/power)
    - `time_delta_hours` (long telemetry gap can indicate outage/permanent fault)
    """
    stress = float(row.get("stress_score", 0.0))
    gap_h = float(row.get("time_delta_hours", 0.0))

    # Long gaps suggest hard failures / permanent outages.
    if gap_h >= 48.0:
        return "FailedPermanent"
    if gap_h >= 8.0:
        return "FailedRecoverable"

    if stress < 0.25:
        return "Healthy"
    if stress < 0.50:
        return "Degraded"
    if stress < 0.70:
        return "MaintenanceRequired"
    if stress < 0.85:
        return "FailedRecoverable"
    return "FailedPermanent"


def add_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["state"] = out.apply(map_to_state, axis=1)
    out["state_idx"] = out["state"].map(STATE_TO_IDX).astype(int)
    return out

