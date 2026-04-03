from __future__ import annotations

import numpy as np


def belief_update(
    belief: np.ndarray,
    transition_a: np.ndarray,
    observation_likelihood: np.ndarray,
) -> np.ndarray:
    """Single-step POMDP belief update for one action and one received observation."""
    pred = belief @ transition_a
    nxt = pred * observation_likelihood
    z = nxt.sum()
    if z <= 1e-15:
        return np.ones_like(belief) / belief.size
    return nxt / z


def greedy_action_from_belief(
    belief: np.ndarray,
    transitions: np.ndarray,
    rewards: np.ndarray,
    value: np.ndarray,
    gamma: float,
) -> int:
    n_actions = transitions.shape[0]
    scores = np.zeros(n_actions, dtype=float)
    for a in range(n_actions):
        immediate = np.dot(belief, rewards[a])
        future = np.dot(belief @ transitions[a], value)
        scores[a] = immediate + gamma * future
    return int(np.argmax(scores))
