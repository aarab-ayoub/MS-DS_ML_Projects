from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

# Canonical hidden health states used across DTMC/CTMC/HMM/MDP/RL/POMDP.
STATES: List[str] = [
    "Healthy",
    "Degraded",
    "MaintenanceRequired",
    "FailedRecoverable",
    "FailedPermanent",
]

ACTIONS: List[str] = [
    "keep_running",
    "power_cap",
    "checkpoint",
    "reboot",
    "isolate_maintain",
]

STATE_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(STATES)}
ACTION_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(ACTIONS)}

BASE_TRANSITION = np.array(
    [
        [0.90, 0.06, 0.02, 0.01, 0.01],
        [0.15, 0.70, 0.10, 0.05, 0.00],
        [0.40, 0.00, 0.40, 0.15, 0.05],
        [0.35, 0.00, 0.20, 0.40, 0.05],
        [0.00, 0.00, 0.00, 0.00, 1.00],
    ],
    dtype=float,
)

# Off-diagonal CTMC intensities; diagonal set automatically.
BASE_GENERATOR = np.array(
    [
        [0.00, 0.06, 0.02, 0.01, 0.01],
        [0.15, 0.00, 0.10, 0.05, 0.00],
        [0.40, 0.00, 0.00, 0.15, 0.05],
        [0.35, 0.00, 0.20, 0.00, 0.05],
        [0.00, 0.00, 0.00, 0.00, 0.00],
    ],
    dtype=float,
)

# Simple action effects on key transitions (multiplicative factors).
ACTION_EFFECTS = {
    "keep_running": {},
    "power_cap": {(1, 3): 0.75, (2, 3): 0.80, (0, 1): 0.90},
    "checkpoint": {(3, 4): 0.85},
    "reboot": {(3, 0): 1.40, (3, 3): 0.65, (3, 4): 0.80},
    "isolate_maintain": {(2, 0): 1.25, (2, 3): 0.70, (2, 4): 0.75},
}

EMISSION_PARAMS = {
    "Healthy": {
        "temperature": (63.0, 3.0),
        "utilization": (87.0, 5.0),
        "power_usage": (590.0, 35.0),
        "ecc_lambda": 0.2,
        "xid_probs": [0.97, 0.02, 0.01, 0.00],
        "retired_pages_lambda": 0.02,
    },
    "Degraded": {
        "temperature": (71.0, 4.0),
        "utilization": (80.0, 8.0),
        "power_usage": (620.0, 45.0),
        "ecc_lambda": 1.5,
        "xid_probs": [0.80, 0.12, 0.07, 0.01],
        "retired_pages_lambda": 0.2,
    },
    "MaintenanceRequired": {
        "temperature": (76.0, 4.5),
        "utilization": (72.0, 10.0),
        "power_usage": (640.0, 55.0),
        "ecc_lambda": 4.0,
        "xid_probs": [0.55, 0.20, 0.20, 0.05],
        "retired_pages_lambda": 0.7,
    },
    "FailedRecoverable": {
        "temperature": (68.0, 7.0),
        "utilization": (25.0, 20.0),
        "power_usage": (300.0, 120.0),
        "ecc_lambda": 7.0,
        "xid_probs": [0.15, 0.20, 0.35, 0.30],
        "retired_pages_lambda": 1.2,
    },
    "FailedPermanent": {
        "temperature": (40.0, 10.0),
        "utilization": (2.0, 5.0),
        "power_usage": (60.0, 40.0),
        "ecc_lambda": 10.0,
        "xid_probs": [0.05, 0.10, 0.25, 0.60],
        "retired_pages_lambda": 2.0,
    },
}

XID_CODES = [0, 31, 48, 79]


@dataclass
class RewardWeights:
    uptime: float = 1.0
    failure: float = 15.0
    recovery: float = 3.0
    action: float = 1.0


@dataclass
class ProjectConfig:
    seed: int = 7
    dt_hours: float = 1.0
    horizon_steps: int = 300
    ctmc_horizon_hours: float = 240.0
    gamma: float = 0.95
    rl_episodes: int = 300
    rl_max_steps: int = 500
    td_eval_episodes: int = 300
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


def build_generator(off_diag: np.ndarray) -> np.ndarray:
    """Return a valid generator matrix from off-diagonal intensities."""
    a = off_diag.copy().astype(float)
    np.fill_diagonal(a, 0.0)
    row_sums = a.sum(axis=1)
    for i in range(a.shape[0]):
        a[i, i] = -row_sums[i]
    return a
