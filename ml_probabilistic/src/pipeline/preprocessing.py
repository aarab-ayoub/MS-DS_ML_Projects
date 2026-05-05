from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from ..core.config import XID_CODES

logger = logging.getLogger(__name__)


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _derive_xid_code(ecc_count: pd.Series) -> pd.Series:
    # Heuristic mapping: map observed error magnitude into one of the
    # 4 categorical XID codes used by the emission model.
    # If ecc_count is not exactly 0/2/3, bucket using rough thresholds.
    ecc = ecc_count.fillna(0).astype(float)
    xid = pd.Series(np.zeros(len(ecc), dtype=int), index=ecc.index)
    xid[ecc <= 0] = XID_CODES[0]
    xid[(ecc > 0) & (ecc <= 2)] = XID_CODES[1]
    xid[(ecc > 2) & (ecc <= 3)] = XID_CODES[2]
    xid[ecc > 3] = XID_CODES[3]
    return xid


def _minmax_clip(s: pd.Series, lo: float, hi: float) -> pd.Series:
    if hi <= lo:
        return pd.Series(np.zeros(len(s), dtype=float), index=s.index)
    out = (s - lo) / (hi - lo)
    return out.clip(0.0, 1.0)


def preprocess_real_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess real telemetry for:
    - state mapping (via `stress_score`)
    - DTMC/CTMC transition estimation (via `state_idx`)
    - HMM observation extraction (via standardized observation columns)
    """
    _require_columns(df, ["timestamp"])

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    t0 = df["timestamp"].iloc[0]
    df["time_hours"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
    df["time_delta_hours"] = df["time_hours"].diff().fillna(0.0)

    # Fill missing numeric values with medians for stability.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # --- Derive observation columns required by HMM code ---
    # Temperature: use the first GPU core temp avg if available.
    temp_col = "gpu0_core_temp_avg" if "gpu0_core_temp_avg" in df.columns else None
    if temp_col is None:
        temp_candidates = [c for c in df.columns if c.endswith("_temp_avg")]
        if not temp_candidates:
            raise KeyError("No temperature-like '*_temp_avg' column found.")
        temp_col = sorted(temp_candidates)[0]
        logger.warning("Using temperature column: %s", temp_col)

    power_col = "total_power_avg" if "total_power_avg" in df.columns else None
    if power_col is None:
        power_candidates = [c for c in df.columns if "power" in c.lower() and c.endswith("_avg")]
        if not power_candidates:
            raise KeyError("No power-like '*power*_avg' column found.")
        power_col = sorted(power_candidates)[0]
        logger.warning("Using power column: %s", power_col)

    # ecc_count proxy: dataset does not provide a dedicated ECC column in the
    # current extract, so we use `value` if present.
    if "value" in df.columns:
        ecc_count = df["value"].fillna(0).astype(float).round().astype(int)
    else:
        ecc_count = pd.Series(np.zeros(len(df), dtype=int), index=df.index)
        logger.warning("No `value` column found; setting ecc_count=0.")

    df["temperature"] = df[temp_col].astype(float)
    df["power_usage"] = df[power_col].astype(float)
    df["ecc_count"] = ecc_count
    df["xid_code"] = _derive_xid_code(df["ecc_count"])

    # utilization: normalize power into [0, 100] for a rough utilization proxy.
    power_lo = float(df["power_usage"].quantile(0.05))
    power_hi = float(df["power_usage"].quantile(0.95))
    df["utilization"] = _minmax_clip(df["power_usage"], power_lo, power_hi) * 100.0

    # retired_pages: heuristic proxy from ecc_count.
    df["retired_pages"] = (df["ecc_count"].astype(float) * 0.5).clip(lower=0.0)

    # --- Stress score for 5-state mapping (metrics-only) ---
    # We keep it strictly based on telemetry-derived signals, not on `value`.
    temp = df["temperature"]
    power = df["power_usage"]

    temp_lo = float(temp.quantile(0.05))
    temp_hi = float(temp.quantile(0.95))
    power_lo = float(power.quantile(0.05))
    power_hi = float(power.quantile(0.95))

    temp_n = _minmax_clip(temp, temp_lo, temp_hi)
    power_n = _minmax_clip(power, power_lo, power_hi)

    # Weighted stress: temperature dominates; power acts as a secondary indicator.
    df["stress_score"] = 0.7 * temp_n + 0.3 * power_n

    # Keep only columns that downstream pipeline will need (plus a few
    # telemetry columns for EDA/debugging).
    keep = [
        "timestamp",
        "time_hours",
        "time_delta_hours",
        "temperature",
        "ecc_count",
        "xid_code",
        "utilization",
        "power_usage",
        "retired_pages",
        "stress_score",
        temp_col,
        power_col,
    ]
    if "value" in df.columns:
        keep.append("value")
    df = df[[c for c in keep if c in df.columns]]

    return df

