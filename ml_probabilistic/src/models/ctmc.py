from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def validate_generator(a: np.ndarray) -> None:
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Generator must be square.")
    off_diag = a.copy()
    np.fill_diagonal(off_diag, 0.0)
    if np.any(off_diag < -1e-12):
        raise ValueError("Off-diagonal intensities must be non-negative.")
    if not np.allclose(a.sum(axis=1), 0.0, atol=1e-8):
        raise ValueError("Each row of generator must sum to 0.")


def transition_matrix(a: np.ndarray, t: float) -> np.ndarray:
    validate_generator(a)
    if t < 0:
        raise ValueError("Time t must be non-negative")
    return expm(a * t)


def distribution_at_time(pi0: np.ndarray, a: np.ndarray, t: float) -> np.ndarray:
    pi0 = np.asarray(pi0, dtype=float)
    if not np.isclose(pi0.sum(), 1.0):
        raise ValueError("Initial distribution must sum to 1.")
    return pi0 @ transition_matrix(a, t)


def trajectory(pi0: np.ndarray, a: np.ndarray, time_grid: np.ndarray) -> np.ndarray:
    probs = []
    for t in time_grid:
        probs.append(distribution_at_time(pi0, a, float(t)))
    return np.vstack(probs)


def survival_probability(pi_t: np.ndarray, failed_idx: int) -> np.ndarray:
    return 1.0 - pi_t[:, failed_idx]
