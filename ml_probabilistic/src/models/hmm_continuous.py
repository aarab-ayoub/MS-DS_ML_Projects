from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .ctmc import transition_matrix
from .hmm_discrete import emission_log_prob


def ct_hmm_filter(
    observations: List[Dict[str, float]],
    times: List[float],
    generator: np.ndarray,
    pi0: np.ndarray,
    scalar_key: Optional[str] = None,
) -> np.ndarray:
    """Forward filtering for CT hidden chain observed at irregular timestamps."""
    if len(observations) != len(times):
        raise ValueError("observations and times must have the same length")

    n_states = generator.shape[0]
    t_max = len(times)
    post = np.zeros((t_max, n_states), dtype=float)

    belief = pi0.astype(float).copy()
    belief /= belief.sum()

    for t in range(t_max):
        if t > 0:
            dt = float(times[t] - times[t - 1])
            p_dt = transition_matrix(generator, dt)
            belief = belief @ p_dt

        likelihood = np.zeros(n_states, dtype=float)
        for s in range(n_states):
            likelihood[s] = np.exp(emission_log_prob(s, observations[t], scalar_key))

        belief = belief * likelihood
        z = belief.sum()
        if z <= 1e-15:
            belief = np.ones(n_states) / n_states
        else:
            belief /= z
        post[t] = belief

    return post
