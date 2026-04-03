from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryFeature:
    age_since_maintenance: int = 0
    rolling_ecc: float = 0.0



def update_memory_feature(mem: MemoryFeature, ecc_count: int, maintenance_done: bool) -> MemoryFeature:
    if maintenance_done:
        return MemoryFeature(age_since_maintenance=0, rolling_ecc=0.5 * mem.rolling_ecc)
    return MemoryFeature(
        age_since_maintenance=mem.age_since_maintenance + 1,
        rolling_ecc=0.9 * mem.rolling_ecc + float(ecc_count),
    )


def risk_multiplier(mem: MemoryFeature) -> float:
    age_factor = 1.0 + min(0.5, mem.age_since_maintenance / 1000.0)
    ecc_factor = 1.0 + min(0.7, mem.rolling_ecc / 50.0)
    return age_factor * ecc_factor


def apply_non_markov_adjustment(transition_row: np.ndarray, mem: MemoryFeature, fail_idx: int) -> np.ndarray:
    row = transition_row.copy().astype(float)
    mult = risk_multiplier(mem)
    row[fail_idx] *= mult
    row = np.clip(row, 1e-12, None)
    row /= row.sum()
    return row
