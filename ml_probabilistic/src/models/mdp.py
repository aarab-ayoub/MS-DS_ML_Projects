from __future__ import annotations

import numpy as np


def value_iteration(
    transitions: np.ndarray,
    rewards: np.ndarray,
    gamma: float = 0.95,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve finite-state/action MDP with value iteration.

    transitions shape: (n_actions, n_states, n_states)
    rewards shape: (n_actions, n_states)
    """
    n_actions, n_states, _ = transitions.shape
    v = np.zeros(n_states, dtype=float)

    for _ in range(max_iter):
        q = np.zeros((n_actions, n_states), dtype=float)
        for a in range(n_actions):
            q[a] = rewards[a] + gamma * transitions[a] @ v
        v_next = q.max(axis=0)
        if np.max(np.abs(v_next - v)) < tol:
            v = v_next
            break
        v = v_next

    policy = q.argmax(axis=0)
    return v, policy
