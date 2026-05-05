from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..core.config import (
    ACTIONS,
    ACTION_EFFECTS,
    ProjectConfig,
    STATES,
)
from ..models.ctmc import trajectory
from ..models.dtmc import expected_steps_to_absorption
from ..models.hmm_continuous import ct_hmm_filter
from ..models.hmm_discrete import forward_filter, viterbi
from ..models.mdp import value_iteration
from ..models.rl import RLEnv, q_learning, r_learning, sarsa, td0_policy_evaluation, td_lambda_policy_evaluation
from .data_loader import load_real_dataset
from .preprocessing import preprocess_real_dataframe
from .state_mapping import add_state_columns
from .transition_estimation import estimate_generator_matrix, estimate_transition_matrix

logger = logging.getLogger(__name__)


#cette fonction génère les matrices de transition pour chaque action en appliquant les effets d'action définis dans ACTION_EFFECTS.
def _build_action_transitions(base: np.ndarray, n_actions: int) -> np.ndarray:
    transitions = []
    for a_idx in range(n_actions):
        p = base.copy().astype(float)
        action_name = ACTIONS[a_idx]
        effects = ACTION_EFFECTS.get(action_name, {})
        for (i, j), mult in effects.items():
            p[i, j] *= mult
        # Re-normalize each row after applying action effects.
        row_sum = p.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        p = p / row_sum
        transitions.append(p)
    return np.stack(transitions, axis=0)

#cette fonction génère une matrice de récompenses pour chaque action et état, en attribuant des récompenses négatives plus sévères pour les états plus graves et les actions plus coûteuses.
def _build_rewards(n_actions: int, n_states: int) -> np.ndarray:
    rewards = np.zeros((n_actions, n_states), dtype=float)
    for a in range(n_actions):
        for s in range(n_states):
            if s == n_states - 1:
                rewards[a, s] = -15.0 - 0.2 * a
            elif s == n_states - 2:
                rewards[a, s] = -3.0 - 0.1 * a
            else:
                rewards[a, s] = 1.0 - 0.1 * a
    return rewards


def run_all(
    cfg: ProjectConfig,
    *,
    dataset_path: str | Path = "data/real",
    extract_dir: str | Path | None = None,
    enable_ctmc_estimation: bool = True,
) -> dict:
    out = {}

    project_root = Path(__file__).resolve().parents[2]
    dataset_path = Path(dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = (project_root / dataset_path).resolve()
    if extract_dir is None:
        extract_dir_path = dataset_path / "extracted"
    else:
        extract_dir_path = Path(extract_dir)
        if not extract_dir_path.is_absolute():
            extract_dir_path = (project_root / extract_dir_path).resolve()

    df_raw = load_real_dataset(
        project_root=project_root,
        dataset_path=dataset_path,
        extract_dir=extract_dir_path,
        extract_tars=True,
    )
    df_proc = preprocess_real_dataframe(df_raw)
    df_dt = add_state_columns(df_proc)
    out["raw_df"] = df_raw
    out["processed_df"] = df_dt
    out["dt_trace"] = df_dt
    out["ct_trace"] = df_dt.copy()

    p_emp = estimate_transition_matrix(df_dt)
    out["empirical_transition"] = p_emp

    transient = [0, 1, 2, 3]
    mttf_steps = expected_steps_to_absorption(p_emp, transient)
    out["dt_mttf_steps_from_states"] = mttf_steps

    # DTMC/HMM prior from first observed state.
    pi0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    first_state = int(df_dt["state_idx"].iloc[0])
    pi0[:] = 0.0
    pi0[first_state] = 1.0

    if enable_ctmc_estimation:
        q_emp = estimate_generator_matrix(df_dt)
    else:
        q_emp = np.zeros((len(STATES), len(STATES)), dtype=float)
    out["generator_matrix"] = q_emp

    time_grid = np.linspace(0.0, cfg.ctmc_horizon_hours, 200)
    pi_t = trajectory(pi0, q_emp, time_grid)
    out["ct_time_grid"] = time_grid
    out["ct_probabilities"] = pi_t

    observations = df_dt[["temperature", "ecc_count", "xid_code", "utilization", "power_usage", "retired_pages"]].to_dict(
        orient="records"
    )
    path = viterbi(observations, p_emp, pi0)
    post = forward_filter(observations, p_emp, pi0)
    out["hmm_path"] = path
    out["hmm_post"] = post

    if not df_dt.empty:
        ct_obs = df_dt[["temperature", "ecc_count", "xid_code", "utilization", "power_usage", "retired_pages"]].to_dict(
            orient="records"
        )
        ct_times = df_dt["time_hours"].tolist()
        out["ct_hmm_post"] = ct_hmm_filter(ct_obs, ct_times, q_emp, pi0)
    else:
        out["ct_hmm_post"] = np.empty((0, len(STATES)))

    n_actions = len(ACTIONS)
    n_states = len(STATES)
    transitions = _build_action_transitions(p_emp, n_actions)
    rewards = _build_rewards(n_actions, n_states)
    out["mdp_transitions"] = transitions
    out["mdp_rewards"] = rewards

    v_star, pi_star = value_iteration(transitions, rewards, gamma=cfg.gamma)
    # The terminal FailedPermanent state is operationally mapped to isolate_maintain.
    # The argmax action in terminal states is mathematically arbitrary because episodes stop there.
    pi_star = np.asarray(pi_star, dtype=int)
    pi_star[n_states - 1] = ACTIONS.index("isolate_maintain")
    out["mdp_v"] = v_star
    out["mdp_policy"] = pi_star

    env = RLEnv(transitions=transitions, rewards=rewards, terminal_state=n_states - 1, seed=cfg.seed)
    q_q, ret_q = q_learning(env, episodes=cfg.rl_episodes, max_steps=cfg.rl_max_steps, gamma=cfg.gamma)
    q_s, ret_s = sarsa(env, episodes=cfg.rl_episodes, max_steps=cfg.rl_max_steps, gamma=cfg.gamma)
    q_r, rho, ret_r = r_learning(env, episodes=cfg.rl_episodes, max_steps=cfg.rl_max_steps)
    v_td0 = td0_policy_evaluation(env, pi_star, episodes=cfg.td_eval_episodes, max_steps=cfg.rl_max_steps, gamma=cfg.gamma)
    v_tdl = td_lambda_policy_evaluation(
        env,
        pi_star,
        episodes=cfg.td_eval_episodes,
        max_steps=cfg.rl_max_steps,
        gamma=cfg.gamma,
    )

    out["q_learning_q"] = q_q
    out["q_learning_returns"] = ret_q
    out["sarsa_q"] = q_s
    out["sarsa_returns"] = ret_s
    out["r_learning_q"] = q_r
    out["r_learning_rho"] = rho
    out["r_learning_returns"] = ret_r
    out["td0_v"] = v_td0
    out["tdlambda_v"] = v_tdl

    return out


def summarize(results: dict) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "metric": [
                "DTMC_MTTF_from_Healthy_steps",
                "Mean_Q_learning_return",
                "Mean_SARSA_return",
                "Mean_R_learning_return",
                "R_learning_rho",
                "TD0_V_Healthy",
                "TDlambda_V_Healthy",
            ],
            "value": [
                float(results["dt_mttf_steps_from_states"][0]),
                float(np.mean(results["q_learning_returns"])),
                float(np.mean(results["sarsa_returns"])),
                float(np.mean(results["r_learning_returns"])),
                float(results["r_learning_rho"]),
                float(results["td0_v"][0]),
                float(results["tdlambda_v"][0]),
            ],
        }
    )
    return df
