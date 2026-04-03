from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from ..core.config import STATES


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_state_counts(df, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    counts = df["state"].value_counts().reindex(STATES, fill_value=0)
    plt.figure(figsize=(8, 4))
    counts.plot(kind="bar")
    plt.title("Hidden State Frequency (DT Trace)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_ct_reliability(time_grid: np.ndarray, pi_t: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 4))
    survival = 1.0 - pi_t[:, -1]
    plt.plot(time_grid, survival)
    plt.title("CTMC Survival Probability")
    plt.xlabel("Hours")
    plt.ylabel("P(not FailedPermanent)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def plot_learning_curves(ret_q: np.ndarray, ret_s: np.ndarray, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    plt.figure(figsize=(8, 4))
    plt.plot(ret_q, label="Q-learning", alpha=0.9)
    plt.plot(ret_s, label="SARSA", alpha=0.9)
    plt.title("RL Episodic Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()
