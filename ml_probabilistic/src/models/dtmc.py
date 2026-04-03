from __future__ import annotations

import numpy as np


def validate_transition_matrix(p: np.ndarray) -> None:
    if p.ndim != 2 or p.shape[0] != p.shape[1]:
        raise ValueError("Transition matrix must be square.")
    if np.any(p < -1e-12):
        raise ValueError("Transition matrix has negative entries.")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Each row of transition matrix must sum to 1.")


def n_step_transition(p: np.ndarray, n: int) -> np.ndarray:
    validate_transition_matrix(p)
    if n < 0:
        raise ValueError("n must be non-negative")
    return np.linalg.matrix_power(p, n)


def distribution_after_n_steps(pi0: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    validate_transition_matrix(p)
    pi0 = np.asarray(pi0, dtype=float)
    if not np.isclose(pi0.sum(), 1.0):
        raise ValueError("Initial distribution must sum to 1.")
    return pi0 @ n_step_transition(p, n)


def fundamental_matrix(p: np.ndarray, transient_idx: list[int]) -> np.ndarray:
    validate_transition_matrix(p)
    q = p[np.ix_(transient_idx, transient_idx)]
    i = np.eye(q.shape[0])
    return np.linalg.inv(i - q)


def expected_steps_to_absorption(p: np.ndarray, transient_idx: list[int]) -> np.ndarray:
    n = fundamental_matrix(p, transient_idx)
    ones = np.ones((n.shape[0], 1))
    return (n @ ones).ravel()


def stationary_distribution_power(p: np.ndarray, max_iter: int = 5000, tol: float = 1e-10) -> np.ndarray:
    validate_transition_matrix(p)
    k = p.shape[0]
    pi = np.ones(k) / k
    for _ in range(max_iter):
        pi_next = pi @ p
        if np.max(np.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi
