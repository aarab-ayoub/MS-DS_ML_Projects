from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core.config import ACTIONS, ProjectConfig, STATES
from .generate_assets import build_dataset_bundle
from .models.non_markov import MemoryFeature, risk_multiplier, update_memory_feature
from .pipeline.experiments import run_all


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _reference_results(
    seed: int = 7,
    horizon_steps: int = 320,
    ctmc_hours: float = 168.0,
    rl_episodes: int | None = None,
    rl_max_steps: int | None = None,
    td_eval_episodes: int | None = None,
) -> dict:
    cfg = ProjectConfig(seed=seed)
    cfg.horizon_steps = horizon_steps
    cfg.ctmc_horizon_hours = ctmc_hours
    if rl_episodes is not None:
        cfg.rl_episodes = rl_episodes
    if rl_max_steps is not None:
        cfg.rl_max_steps = rl_max_steps
    if td_eval_episodes is not None:
        cfg.td_eval_episodes = td_eval_episodes
    return run_all(cfg)


def _save_dtmc(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    dt = res["dt_trace"]
    n_states = len(STATES)

    counts = np.zeros((n_states, n_states), dtype=float)
    for row in dt[["state_idx", "next_state_idx"]].itertuples(index=False):
        counts[int(row.state_idx), int(row.next_state_idx)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    emp_p = counts / row_sums

    pd.DataFrame(emp_p, index=STATES, columns=STATES).to_csv(out_dir / "dtmc_transition_empirical.csv")
    pd.DataFrame({"state": STATES, "mttf_steps": np.array(res["dt_mttf_steps_from_states"]).tolist() + [np.nan]}).to_csv(
        out_dir / "dtmc_mttf.csv", index=False
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(emp_p, cmap="viridis")
    plt.title("DTMC Transition Matrix")
    plt.colorbar(label="Probability")
    plt.xticks(range(n_states), STATES, rotation=45, ha="right")
    plt.yticks(range(n_states), STATES)
    plt.tight_layout()
    plt.savefig(out_dir / "dtmc_transition_heatmap.png", dpi=140)
    plt.close()


def _save_ctmc(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    ctmc_df = pd.DataFrame(
        {
            "time_hours": res["ct_time_grid"],
            **{f"p_{name}": res["ct_probabilities"][:, i] for i, name in enumerate(STATES)},
        }
    )
    ctmc_df.to_csv(out_dir / "ctmc_state_probabilities.csv", index=False)

    survival = 1.0 - res["ct_probabilities"][:, -1]
    plt.figure(figsize=(8, 4))
    plt.plot(res["ct_time_grid"], survival)
    plt.title("CTMC Survival Probability")
    plt.xlabel("Hours")
    plt.ylabel("P(not FailedPermanent)")
    plt.tight_layout()
    plt.savefig(out_dir / "ctmc_survival.png", dpi=140)
    plt.close()


def _save_hmm(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    dt = res["dt_trace"]
    post = res["hmm_post"]
    horizon = min(120, post.shape[0])

    pd.DataFrame(post, columns=[f"post_{s}" for s in STATES]).to_csv(out_dir / "hmm_forward_posterior.csv", index=False)
    pd.DataFrame(
        {
            "step": np.arange(len(res["hmm_path"])),
            "true_state_idx": dt["state_idx"].to_numpy()[: len(res["hmm_path"])],
            "viterbi_state_idx": np.array(res["hmm_path"]),
        }
    ).to_csv(out_dir / "hmm_true_vs_viterbi.csv", index=False)

    plt.figure(figsize=(9, 4))
    for i, name in enumerate(STATES):
        plt.plot(np.arange(horizon), post[:horizon, i], label=name)
    plt.title("HMM Forward Posterior")
    plt.xlabel("Step")
    plt.ylabel("Posterior")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "hmm_forward_posterior.png", dpi=140)
    plt.close()


def _save_mdp(res: dict, out_dir: Path, gamma: float) -> None:
    _ensure(out_dir)
    policy = np.array(res["mdp_policy"], dtype=int)
    values = np.array(res["mdp_v"], dtype=float)
    transitions = np.array(res["mdp_transitions"], dtype=float)
    rewards = np.array(res["mdp_rewards"], dtype=float)

    pd.DataFrame(
        {
            "state": STATES,
            "action_idx": policy,
            "action_name": [ACTIONS[i] for i in policy],
        }
    ).to_csv(out_dir / "mdp_policy_table.csv", index=False)

    pd.DataFrame({"state": STATES, "value": values}).to_csv(out_dir / "mdp_value_function.csv", index=False)

    q_vals = np.zeros((len(ACTIONS), len(STATES)), dtype=float)
    for a_idx in range(len(ACTIONS)):
        q_vals[a_idx] = rewards[a_idx] + gamma * (transitions[a_idx] @ values)
    pd.DataFrame(q_vals, index=ACTIONS, columns=STATES).to_csv(out_dir / "mdp_q_table.csv")

    policy_matrix = np.zeros((len(STATES), len(ACTIONS)), dtype=float)
    for s_idx, a_idx in enumerate(policy):
        policy_matrix[s_idx, a_idx] = 1.0

    plt.figure(figsize=(8, 4.8))
    plt.imshow(policy_matrix, cmap="Blues", aspect="auto")
    plt.title("MDP Optimal Policy Matrix")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.xticks(np.arange(len(ACTIONS)), ACTIONS, rotation=25, ha="right")
    plt.yticks(np.arange(len(STATES)), STATES)
    for s_idx, a_idx in enumerate(policy):
        plt.text(a_idx, s_idx, "1", ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "mdp_policy_matrix.png", dpi=140)
    plt.close()


def _save_rl(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    pd.DataFrame(
        {
            "episode": np.arange(len(res["q_learning_returns"])),
            "q_learning_return": res["q_learning_returns"],
            "sarsa_return": res["sarsa_returns"],
            "r_learning_avg_reward": res["r_learning_returns"],
        }
    ).to_csv(out_dir / "rl_returns.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(res["q_learning_returns"], label="Q-learning")
    plt.plot(res["sarsa_returns"], label="SARSA")
    plt.plot(res["r_learning_returns"], label="R-learning")
    plt.title("RL Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Return / Avg reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "rl_convergence.png", dpi=140)
    plt.close()


def _save_pomdp(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    post = np.array(res["hmm_post"], dtype=float)
    eps = 1e-12
    entropy = -np.sum(post * np.log(post + eps), axis=1)
    pd.DataFrame({"step": np.arange(len(entropy)), "belief_entropy": entropy}).to_csv(out_dir / "pomdp_belief_entropy.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(entropy)), entropy, color="tab:red")
    plt.title("POMDP Belief Entropy")
    plt.xlabel("Step")
    plt.ylabel("Entropy")
    plt.tight_layout()
    plt.savefig(out_dir / "pomdp_belief_entropy.png", dpi=140)
    plt.close()


def _save_non_markov(res: dict, out_dir: Path) -> None:
    _ensure(out_dir)
    dt = res["dt_trace"]
    mem = MemoryFeature()
    multipliers = []
    for row in dt[["ecc_count", "action"]].itertuples(index=False):
        maintenance_done = row.action == "isolate_maintain"
        mem = update_memory_feature(mem, int(row.ecc_count), maintenance_done)
        multipliers.append(risk_multiplier(mem))

    pd.DataFrame({"step": np.arange(len(multipliers)), "risk_multiplier": multipliers}).to_csv(
        out_dir / "non_markov_risk_multiplier.csv", index=False
    )

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(multipliers)), multipliers, color="tab:green")
    plt.title("Non-Markovian Risk Multiplier")
    plt.xlabel("Step")
    plt.ylabel("Multiplier")
    plt.tight_layout()
    plt.savefig(out_dir / "non_markov_risk_multiplier.png", dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one section or full pipeline for the project defense/demo")
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "dtmc", "ctmc", "hmm", "mdp", "rl", "pomdp", "non_markov", "bundle"],
    )
    parser.add_argument("--runs", type=int, default=30, help="Used when section=bundle")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=320)
    parser.add_argument("--ct_hours", type=float, default=168.0)
    parser.add_argument("--rl_episodes", type=int, default=None)
    parser.add_argument("--rl_max_steps", type=int, default=None)
    parser.add_argument("--td_eval_episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    if args.section == "bundle":
        build_dataset_bundle(project_root, runs=args.runs)
        print("Bundle generated in mandatory mode.")
        return

    out_root = project_root / "results" / "sections"
    _ensure(out_root)

    res = _reference_results(
        seed=args.seed,
        horizon_steps=args.horizon,
        ctmc_hours=args.ct_hours,
        rl_episodes=args.rl_episodes,
        rl_max_steps=args.rl_max_steps,
        td_eval_episodes=args.td_eval_episodes,
    )

    if args.section in ["all", "dtmc"]:
        _save_dtmc(res, out_root / "dtmc")
    if args.section in ["all", "ctmc"]:
        _save_ctmc(res, out_root / "ctmc")
    if args.section in ["all", "hmm"]:
        _save_hmm(res, out_root / "hmm")
    if args.section in ["all", "mdp"]:
        _save_mdp(res, out_root / "mdp", gamma=ProjectConfig().gamma)
    if args.section in ["all", "rl"]:
        _save_rl(res, out_root / "rl")
    if args.section in ["all", "pomdp"]:
        _save_pomdp(res, out_root / "pomdp")
    if args.section in ["all", "non_markov"]:
        _save_non_markov(res, out_root / "non_markov")

    print(f"Section run completed: {args.section}")


if __name__ == "__main__":
    main()
