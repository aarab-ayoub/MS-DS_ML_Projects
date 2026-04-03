from __future__ import annotations

import argparse
import os

from .core.config import ProjectConfig
from .pipeline.experiments import run_all, summarize
from .pipeline.plots import plot_ct_reliability, plot_learning_curves, plot_state_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU cluster probabilistic reliability project")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--out", default="../results")
    parser.add_argument("--horizon-steps", type=int, default=None)
    parser.add_argument("--ct-hours", type=float, default=None)
    parser.add_argument("--rl-episodes", type=int, default=None)
    parser.add_argument("--rl-max-steps", type=int, default=None)
    parser.add_argument("--td-eval-episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ProjectConfig()
    if args.mode == "quick":
        cfg.horizon_steps = 120
        cfg.ctmc_horizon_hours = 72.0
        cfg.rl_episodes = 180
        cfg.td_eval_episodes = 180

    if args.horizon_steps is not None:
        cfg.horizon_steps = args.horizon_steps
    if args.ct_hours is not None:
        cfg.ctmc_horizon_hours = args.ct_hours
    if args.rl_episodes is not None:
        cfg.rl_episodes = args.rl_episodes
    if args.rl_max_steps is not None:
        cfg.rl_max_steps = args.rl_max_steps
    if args.td_eval_episodes is not None:
        cfg.td_eval_episodes = args.td_eval_episodes

    results = run_all(cfg)

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    summary = summarize(results)
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    plot_state_counts(results["dt_trace"], os.path.join(out_dir, "state_counts.png"))
    plot_ct_reliability(results["ct_time_grid"], results["ct_probabilities"], os.path.join(out_dir, "ct_reliability.png"))
    plot_learning_curves(
        results["q_learning_returns"],
        results["sarsa_returns"],
        os.path.join(out_dir, "rl_returns.png"),
    )

    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
