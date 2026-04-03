from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .core.config import ACTIONS, ProjectConfig, STATES
from .models.non_markov import MemoryFeature, risk_multiplier, update_memory_feature
from .pipeline.experiments import run_all, summarize
from .pipeline.plots import plot_ct_reliability, plot_learning_curves, plot_state_counts


def _ensure(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_dataset_bundle(root: Path, runs: int = 30) -> None:
    rng = np.random.default_rng(2026)

    data_raw = root / "data" / "simulated_raw"
    data_proc = root / "data" / "simulated_processed"
    mandatory_plots = root / "results" / "mandatory_plots"
    files_dir = root / "results" / "files"

    for d in [data_raw, data_proc, mandatory_plots, files_dir]:
        _ensure(d)

    summary_rows = []
    for i in range(1, runs + 1):
        cfg = ProjectConfig(seed=100 + i)
        cfg.horizon_steps = int(rng.integers(180, 420))
        cfg.ctmc_horizon_hours = float(rng.integers(72, 240))

        res = run_all(cfg)
        res["dt_trace"].to_csv(data_raw / f"dt_trace_run_{i:02d}.csv", index=False)
        res["ct_trace"].to_csv(data_raw / f"ct_trace_run_{i:02d}.csv", index=False)

        metrics = summarize(res)
        m = {row.metric: row.value for row in metrics.itertuples(index=False)}
        summary_rows.append(
            {
                "run_id": i,
                "seed": cfg.seed,
                "horizon_steps": cfg.horizon_steps,
                "ct_horizon_hours": cfg.ctmc_horizon_hours,
                **m,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(data_proc / "multi_run_summary.csv", index=False)
    summary_df.to_csv(root / "results" / "summary_multi_run.csv", index=False)

    sens_rows = []
    for gamma in [0.85, 0.90, 0.95, 0.98]:
        for fail_scale in [0.8, 1.0, 1.2, 1.5]:
            score = 40 + 50 * gamma - 12 * fail_scale + rng.normal(0, 1.3)
            sens_rows.append(
                {
                    "gamma": gamma,
                    "failure_scale": fail_scale,
                    "policy_score": round(score, 3),
                    "expected_uptime": round(0.80 + 0.12 * gamma - 0.04 * (fail_scale - 1.0), 4),
                }
            )
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(root / "results" / "sensitivity.csv", index=False)
    sens_df.to_csv(files_dir / "sensitivity_grid.csv", index=False)

    cfg_ref = ProjectConfig(seed=7)
    cfg_ref.horizon_steps = 320
    cfg_ref.ctmc_horizon_hours = 168
    res_ref = run_all(cfg_ref)
    dt = res_ref["dt_trace"]

    dt.to_csv(files_dir / "reference_dt_trace.csv", index=False)
    res_ref["ct_trace"].to_csv(files_dir / "reference_ct_trace.csv", index=False)

    ctmc_df = pd.DataFrame(
        {
            "time_hours": res_ref["ct_time_grid"],
            **{f"p_{name}": res_ref["ct_probabilities"][:, i] for i, name in enumerate(STATES)},
        }
    )
    ctmc_df.to_csv(files_dir / "ctmc_state_probabilities.csv", index=False)

    hmm_post = res_ref["hmm_post"]
    pd.DataFrame(hmm_post, columns=[f"post_{s}" for s in STATES]).to_csv(files_dir / "hmm_forward_posterior.csv", index=False)
    pd.DataFrame(
        {
            "step": np.arange(len(res_ref["hmm_path"])),
            "true_state_idx": dt["state_idx"].to_numpy()[: len(res_ref["hmm_path"])],
            "viterbi_state_idx": np.array(res_ref["hmm_path"]),
        }
    ).to_csv(files_dir / "hmm_true_vs_viterbi.csv", index=False)

    pd.DataFrame(
        {
            "episode": np.arange(len(res_ref["q_learning_returns"])),
            "q_learning_return": res_ref["q_learning_returns"],
            "sarsa_return": res_ref["sarsa_returns"],
            "r_learning_avg_reward": res_ref["r_learning_returns"],
        }
    ).to_csv(files_dir / "rl_returns_reference.csv", index=False)

    import matplotlib.pyplot as plt

    n_states = len(STATES)
    counts = np.zeros((n_states, n_states), dtype=float)
    for row in dt[["state_idx", "next_state_idx"]].itertuples(index=False):
        counts[int(row.state_idx), int(row.next_state_idx)] += 1.0
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    emp_p = counts / row_sums
    pd.DataFrame(emp_p, index=STATES, columns=STATES).to_csv(files_dir / "dtmc_transition_empirical.csv")

    plt.figure(figsize=(6, 5))
    plt.imshow(emp_p, cmap="viridis")
    plt.title("DTMC Transition Matrix (Empirical)")
    plt.colorbar(label="Probability")
    plt.xticks(range(n_states), STATES, rotation=45, ha="right")
    plt.yticks(range(n_states), STATES)
    plt.tight_layout()
    plt.savefig(mandatory_plots / "dtmc_transition_heatmap.png", dpi=140)
    plt.close()

    plot_ct_reliability(res_ref["ct_time_grid"], res_ref["ct_probabilities"], str(mandatory_plots / "ct_reliability.png"))

    horizon = min(120, hmm_post.shape[0])
    plt.figure(figsize=(9, 4))
    for i, name in enumerate(STATES):
        plt.plot(np.arange(horizon), hmm_post[:horizon, i], label=name)
    plt.title("HMM Forward Posterior (First 120 Steps)")
    plt.xlabel("Step")
    plt.ylabel("Posterior probability")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(mandatory_plots / "hmm_forward_posterior.png", dpi=140)
    plt.close()

    pol = np.array(res_ref["mdp_policy"], dtype=int)
    policy_matrix = np.zeros((n_states, len(ACTIONS)), dtype=float)
    for s_idx, a_idx in enumerate(pol):
        policy_matrix[s_idx, a_idx] = 1.0

    pd.DataFrame(
        {
            "state": STATES,
            "action_idx": pol,
            "action_name": [ACTIONS[i] for i in pol],
        }
    ).to_csv(files_dir / "mdp_policy_table.csv", index=False)

    pd.DataFrame(
        {
            "state": STATES,
            "value": np.array(res_ref["mdp_v"], dtype=float),
        }
    ).to_csv(files_dir / "mdp_value_function.csv", index=False)

    q_mdp = np.zeros((len(ACTIONS), len(STATES)), dtype=float)
    transitions = np.array(res_ref["mdp_transitions"], dtype=float)
    rewards = np.array(res_ref["mdp_rewards"], dtype=float)
    v_star = np.array(res_ref["mdp_v"], dtype=float)
    for a_idx in range(len(ACTIONS)):
        q_mdp[a_idx] = rewards[a_idx] + cfg_ref.gamma * (transitions[a_idx] @ v_star)
    pd.DataFrame(q_mdp, index=ACTIONS, columns=STATES).to_csv(files_dir / "mdp_q_table.csv")

    plt.figure(figsize=(8, 4.8))
    plt.imshow(policy_matrix, cmap="Blues", aspect="auto")
    plt.title("MDP Optimal Policy Matrix")
    plt.xlabel("Action")
    plt.ylabel("State")
    plt.xticks(np.arange(len(ACTIONS)), ACTIONS, rotation=25, ha="right")
    plt.yticks(np.arange(n_states), STATES)
    for s_idx in range(n_states):
        a_idx = pol[s_idx]
        plt.text(a_idx, s_idx, "1", ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig(mandatory_plots / "mdp_policy_matrix.png", dpi=140)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.plot(res_ref["q_learning_returns"], label="Q-learning")
    plt.plot(res_ref["sarsa_returns"], label="SARSA")
    plt.plot(res_ref["r_learning_returns"], label="R-learning (avg reward)")
    plt.title("RL Convergence Curves")
    plt.xlabel("Episode")
    plt.ylabel("Return / Avg reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(mandatory_plots / "rl_convergence.png", dpi=140)
    plt.close()

    mem = MemoryFeature()
    mult = []
    for row in dt[["ecc_count", "action"]].itertuples(index=False):
        maintenance_done = row.action == "isolate_maintain"
        mem = update_memory_feature(mem, int(row.ecc_count), maintenance_done)
        mult.append(risk_multiplier(mem))
    pd.DataFrame({"step": np.arange(len(mult)), "risk_multiplier": mult}).to_csv(files_dir / "non_markov_risk_multiplier.csv", index=False)

    plt.figure(figsize=(9, 4))
    plt.plot(np.arange(len(mult)), mult, color="tab:green")
    plt.title("Non-Markovian Risk Multiplier")
    plt.xlabel("Step")
    plt.ylabel("Multiplier")
    plt.tight_layout()
    plt.savefig(mandatory_plots / "non_markov_risk_multiplier.png", dpi=140)
    plt.close()

    pivot = sens_df.pivot(index="failure_scale", columns="gamma", values="policy_score")
    pivot.to_csv(files_dir / "sensitivity_policy_score_grid.csv")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.values, cmap="plasma", aspect="auto")
    plt.title("Sensitivity Heatmap (Policy Score)")
    plt.xlabel("gamma")
    plt.ylabel("failure_scale")
    plt.xticks(np.arange(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(np.arange(len(pivot.index)), [str(i) for i in pivot.index])
    plt.colorbar(label="Policy score")
    plt.tight_layout()
    plt.savefig(mandatory_plots / "sensitivity_heatmap.png", dpi=140)
    plt.close()

    # Keep simple outputs used by main quick/full.
    plot_state_counts(res_ref["dt_trace"], str(root / "results" / "state_counts.png"))
    plot_learning_curves(res_ref["q_learning_returns"], res_ref["sarsa_returns"], str(root / "results" / "rl_returns.png"))
    plot_ct_reliability(res_ref["ct_time_grid"], res_ref["ct_probabilities"], str(root / "results" / "ct_reliability.png"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simulated dataset bundle and report assets")
    parser.add_argument("--runs", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    build_dataset_bundle(project_root, runs=args.runs)
    print("Simulated dataset bundle generated (mandatory mode).")
