from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class RLEnv:
    transitions: np.ndarray  # (A, S, S)
    rewards: np.ndarray  # (A, S)
    terminal_state: int
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.n_actions, self.n_states, _ = self.transitions.shape

    def reset(self, start_state: int = 0) -> int:
        self.s = start_state
        return self.s

    def step(self, a: int) -> Tuple[int, float, bool]:
        p = self.transitions[a, self.s]
        s_next = int(self.rng.choice(self.n_states, p=p))
        r = float(self.rewards[a, self.s])
        done = s_next == self.terminal_state
        self.s = s_next
        return s_next, r, done


def epsilon_greedy(q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        return int(rng.integers(0, q.shape[1]))
    return int(np.argmax(q[s]))


def q_learning(
    env: RLEnv,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = np.zeros(episodes, dtype=float)

    rng = np.random.default_rng(env.seed + 10)
    for ep in range(episodes):
        s = env.reset(0)
        total = 0.0
        for _ in range(max_steps):
            a = epsilon_greedy(q, s, epsilon, rng)
            s_next, r, done = env.step(a)
            total += r
            q[s, a] += alpha * (r + gamma * np.max(q[s_next]) - q[s, a])
            s = s_next
            if done:
                break
        returns[ep] = total
    return q, returns


def sarsa(
    env: RLEnv,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    returns = np.zeros(episodes, dtype=float)

    rng = np.random.default_rng(env.seed + 20)
    for ep in range(episodes):
        s = env.reset(0)
        a = epsilon_greedy(q, s, epsilon, rng)
        total = 0.0
        for _ in range(max_steps):
            s_next, r, done = env.step(a)
            total += r
            a_next = epsilon_greedy(q, s_next, epsilon, rng)
            q[s, a] += alpha * (r + gamma * q[s_next, a_next] - q[s, a])
            s, a = s_next, a_next
            if done:
                break
        returns[ep] = total
    return q, returns


def td0_policy_evaluation(
    env: RLEnv,
    policy: np.ndarray,
    episodes: int = 300,
    alpha: float = 0.1,
    gamma: float = 0.95,
    max_steps: int = 500,
) -> np.ndarray:
    v = np.zeros(env.n_states, dtype=float)
    for _ in range(episodes):
        s = env.reset(0)
        for _ in range(max_steps):
            a = int(policy[s])
            s_next, r, done = env.step(a)
            v[s] += alpha * (r + gamma * v[s_next] - v[s])
            s = s_next
            if done:
                break
    return v


def td_lambda_policy_evaluation(
    env: RLEnv,
    policy: np.ndarray,
    lam: float = 0.8,
    episodes: int = 300,
    alpha: float = 0.05,
    gamma: float = 0.95,
    max_steps: int = 500,
) -> np.ndarray:
    v = np.zeros(env.n_states, dtype=float)
    for _ in range(episodes):
        e = np.zeros(env.n_states, dtype=float)
        s = env.reset(0)
        for _ in range(max_steps):
            a = int(policy[s])
            s_next, r, done = env.step(a)
            delta = r + gamma * v[s_next] - v[s]
            e[s] += 1.0
            v += alpha * delta * e
            e *= gamma * lam
            s = s_next
            if done:
                break
    return v


def r_learning(
    env: RLEnv,
    episodes: int = 300,
    alpha: float = 0.1,
    beta: float = 0.01,
    epsilon: float = 0.1,
    max_steps: int = 500,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Average-reward R-learning (optional extension).

    Returns:
        q: action-value table
        rho: estimated average reward
        avg_rewards: episode mean rewards
    """
    q = np.zeros((env.n_states, env.n_actions), dtype=float)
    rho = 0.0
    avg_rewards = np.zeros(episodes, dtype=float)

    rng = np.random.default_rng(env.seed + 30)
    for ep in range(episodes):
        s = env.reset(0)
        rewards_ep = []
        for _ in range(max_steps):
            a = epsilon_greedy(q, s, epsilon, rng)
            s_next, r, done = env.step(a)
            rewards_ep.append(r)

            td = r - rho + np.max(q[s_next]) - q[s, a]
            q[s, a] += alpha * td

            greedy_a = int(np.argmax(q[s]))
            if a == greedy_a:
                rho += beta * td

            s = s_next
            if done:
                break

        avg_rewards[ep] = float(np.mean(rewards_ep)) if rewards_ep else 0.0

    return q, float(rho), avg_rewards
