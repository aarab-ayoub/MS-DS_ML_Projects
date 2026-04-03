from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..core.config import EMISSION_PARAMS, STATE_TO_IDX, STATES, XID_CODES


def _log_gaussian(x: float, mu: float, sigma: float) -> float:
    sigma = max(1e-6, sigma)
    return -0.5 * math.log(2 * math.pi * sigma * sigma) - 0.5 * ((x - mu) / sigma) ** 2


def _log_poisson(k: int, lam: float) -> float:
    lam = max(1e-8, lam)
    return k * math.log(lam) - lam - math.lgamma(k + 1)


def _log_categorical(x: int, values: List[int], probs: List[float]) -> float:
    if x not in values:
        return math.log(1e-12)
    idx = values.index(x)
    return math.log(max(1e-12, probs[idx]))


def emission_log_prob(state_idx: int, obs: Dict[str, float], scalar_key: Optional[str] = None) -> float:
    params = EMISSION_PARAMS[STATES[state_idx]]

    if scalar_key == "ecc_count":
        return _log_poisson(int(obs["ecc_count"]), params["ecc_lambda"])
    if scalar_key == "temperature":
        mu, sigma = params["temperature"]
        return _log_gaussian(float(obs["temperature"]), mu, sigma)

    temp_mu, temp_sigma = params["temperature"]
    util_mu, util_sigma = params["utilization"]
    p_mu, p_sigma = params["power_usage"]

    ll = 0.0
    ll += _log_gaussian(float(obs["temperature"]), temp_mu, temp_sigma)
    ll += _log_poisson(int(obs["ecc_count"]), params["ecc_lambda"])
    ll += _log_categorical(int(obs["xid_code"]), XID_CODES, params["xid_probs"])
    ll += _log_gaussian(float(obs["utilization"]), util_mu, util_sigma)
    ll += _log_gaussian(float(obs["power_usage"]), p_mu, p_sigma)
    ll += _log_poisson(int(obs["retired_pages"]), params["retired_pages_lambda"])
    return ll


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return m + math.log(np.sum(np.exp(x - m)))


def viterbi(
    observations: List[Dict[str, float]],
    transition: np.ndarray,
    pi0: np.ndarray,
    scalar_key: Optional[str] = None,
) -> List[int]:
    n_states = transition.shape[0]
    t_max = len(observations)

    log_t = np.log(np.clip(transition, 1e-12, None))
    log_pi0 = np.log(np.clip(pi0, 1e-12, None))

    dp = np.full((t_max, n_states), -np.inf)
    back = np.zeros((t_max, n_states), dtype=int)

    for s in range(n_states):
        dp[0, s] = log_pi0[s] + emission_log_prob(s, observations[0], scalar_key)

    for t in range(1, t_max):
        for s in range(n_states):
            scores = dp[t - 1] + log_t[:, s]
            best_prev = int(np.argmax(scores))
            dp[t, s] = scores[best_prev] + emission_log_prob(s, observations[t], scalar_key)
            back[t, s] = best_prev

    path = [int(np.argmax(dp[-1]))]
    for t in range(t_max - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path


def forward_filter(
    observations: List[Dict[str, float]],
    transition: np.ndarray,
    pi0: np.ndarray,
    scalar_key: Optional[str] = None,
) -> np.ndarray:
    n_states = transition.shape[0]
    t_max = len(observations)
    log_t = np.log(np.clip(transition, 1e-12, None))

    alpha = np.full((t_max, n_states), -np.inf)
    for s in range(n_states):
        alpha[0, s] = math.log(max(1e-12, pi0[s])) + emission_log_prob(s, observations[0], scalar_key)

    for t in range(1, t_max):
        for s in range(n_states):
            alpha[t, s] = _logsumexp(alpha[t - 1] + log_t[:, s]) + emission_log_prob(
                s, observations[t], scalar_key
            )

    post = np.zeros_like(alpha)
    for t in range(t_max):
        z = _logsumexp(alpha[t])
        post[t] = np.exp(alpha[t] - z)
    return post


def decode_states_to_names(path: List[int]) -> List[str]:
    return [STATES[i] for i in path]
