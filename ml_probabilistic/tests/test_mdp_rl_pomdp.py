import numpy as np

from src.models.mdp import value_iteration
from src.models.pomdp import belief_update
from src.models.rl import RLEnv, q_learning


def test_value_iteration_runs():
    t = np.zeros((2, 3, 3), dtype=float)
    t[:, :, :] = np.eye(3)
    r = np.array([[1.0, 0.0, -1.0], [0.5, 0.2, -1.5]])
    v, pi = value_iteration(t, r)
    assert v.shape == (3,)
    assert pi.shape == (3,)


def test_q_learning_runs():
    t = np.zeros((2, 3, 3), dtype=float)
    t[:, :, :] = np.eye(3)
    r = np.zeros((2, 3), dtype=float)
    env = RLEnv(transitions=t, rewards=r, terminal_state=2)
    q, returns = q_learning(env, episodes=5, max_steps=5)
    assert q.shape == (3, 2)
    assert returns.shape == (5,)


def test_belief_update_normalized():
    b = np.array([0.6, 0.4])
    t = np.array([[0.8, 0.2], [0.1, 0.9]])
    o = np.array([0.7, 0.3])
    b2 = belief_update(b, t, o)
    assert np.isclose(b2.sum(), 1.0)
