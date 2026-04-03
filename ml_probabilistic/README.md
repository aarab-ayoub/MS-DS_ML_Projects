# Stochastic Reliability Modeling for H100 GPU Clusters

This project implements a unified student-scale pipeline for:
- DTMC and CTMC reliability analysis
- HMM (discrete and continuous-time filtering)
- MDP control with discount factor
- RL algorithms: Q-learning, SARSA, TD(0), TD(lambda), optional R-learning
- POMDP belief-state updates
- Non-Markovian memory extension (age + cumulative ECC)

## Structure
- `src/core/`: shared configuration and constants
- `src/models/`: DTMC/CTMC/HMM/MDP/RL/POMDP/non-Markov models
- `src/pipeline/`: simulated data generation, orchestration, and plotting
- `src/`: entry points (`main.py`, `generate_assets.py`, `run_section.py`)
- `results/`: generated figures and summary metrics
- `chapters/`: LaTeX chapter stubs for report integration

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run quick experiment:
   ```bash
   .venv/bin/python -m src.main --mode quick --out results
   ```
   This mode is intentionally lightweight for sanity checks and can finish very fast.
3. Run heavier experiment (slower, more realistic for demo/defense):
   ```bash
   .venv/bin/python -m src.main --mode full --horizon-steps 600 --ct-hours 360 --rl-episodes 900 --rl-max-steps 900 --td-eval-episodes 900 --out results
   ```
4. Outputs:
   - `results/summary.csv`
   - `results/state_counts.png`
   - `results/ct_reliability.png`
   - `results/rl_returns.png`

## Mandatory Bundle (Simple Defense Mode)
Use this for your teacher demo when you want only required figures and CSV evidence.

```bash
.venv/bin/python -m src.generate_assets
```

Main outputs:
- `results/mandatory_plots/` (7 mandatory plots)
- `results/files/` (all key CSV files including `mdp_policy_table.csv`, `mdp_value_function.csv`, `mdp_q_table.csv`)
- `data/simulated_raw/` and `data/simulated_processed/`

## Section-by-Section Run Guide
Run one report section at a time (useful during oral defense).

```bash
# DTMC only
.venv/bin/python -m src.run_section --section dtmc

# CTMC only
.venv/bin/python -m src.run_section --section ctmc

# HMM only
.venv/bin/python -m src.run_section --section hmm

# MDP only
.venv/bin/python -m src.run_section --section mdp

# RL only
.venv/bin/python -m src.run_section --section rl

# POMDP only
.venv/bin/python -m src.run_section --section pomdp

# Non-Markov only
.venv/bin/python -m src.run_section --section non_markov

# All sections in one command
.venv/bin/python -m src.run_section --section all

# Heavier RL for one section
.venv/bin/python -m src.run_section --section rl --rl_episodes 1200 --rl_max_steps 900 --td_eval_episodes 1200
```

Section outputs are written to:
- `results/sections/dtmc/`
- `results/sections/ctmc/`
- `results/sections/hmm/`
- `results/sections/mdp/`
- `results/sections/rl/`
- `results/sections/pomdp/`
- `results/sections/non_markov/`

## Script Role Map (What to Explain to Teacher)
- `src/main.py`: quick or full integrated experiment; writes core summary.
- `src/pipeline/experiments.py`: central orchestrator that runs all models and returns unified results.
- `src/generate_assets.py`: generates complete simulated evidence bundle in mandatory mode.
- `src/run_section.py`: demo helper to run only one chapter/section and export corresponding plot + CSV files.
- `src/models/dtmc.py` and `src/models/ctmc.py`: reliability transition models (discrete and continuous time).
- `src/models/hmm_discrete.py` and `src/models/hmm_continuous.py`: hidden-state inference from observables.
- `src/models/mdp.py`: value iteration and optimal policy for control decisions.
- `src/models/rl.py`: tabular RL learning curves and policy learning behavior.
- `src/models/pomdp.py`: belief-state update and action selection under partial observability.
- `src/models/non_markov.py`: memory-augmented risk model (history-sensitive behavior).

## Source File Guide (What + Why)
- `src/__init__.py`: package marker for `src`; needed so `python -m src.*` works reliably.
- `src/main.py`: integrated runner for quick/full experiments; used for one-command project execution.
- `src/generate_assets.py`: builds the mandatory dataset/figure/CSV bundle; used for reproducible report assets.
- `src/run_section.py`: per-section runner (`dtmc`, `ctmc`, `hmm`, etc.); used for oral defense demos.
- `src/core/__init__.py`: core package marker/re-export; keeps config imports centralized.
- `src/core/config.py`: canonical states/actions, transition priors, emissions, and runtime knobs; this is the single source of truth for assumptions.
- `src/models/__init__.py`: model package marker; groups all probabilistic/control models in one namespace.
- `src/models/dtmc.py`: DTMC validation, powers, absorption, stationary distribution; needed for discrete-time reliability math.
- `src/models/ctmc.py`: CTMC generator validation, matrix exponential transitions, trajectories; needed for continuous-time reliability math.
- `src/models/hmm_discrete.py`: discrete HMM emission model, Forward and Viterbi; needed for hidden-state inference from telemetry.
- `src/models/hmm_continuous.py`: CT-HMM filtering over irregular timestamps; needed when event times are non-uniform.
- `src/models/mdp.py`: value iteration for optimal policy/value; needed for formal control decisions.
- `src/models/rl.py`: RL environment + Q-learning/SARSA/R-learning/TD evaluation; needed for learned control baselines.
- `src/models/pomdp.py`: belief update and belief-based action scoring; needed for partial observability control logic.
- `src/models/non_markov.py`: memory-feature risk adjustment; needed to capture history effects beyond pure Markov assumptions.
- `src/pipeline/__init__.py`: pipeline package marker; groups generation/orchestration/plotting utilities.
- `src/pipeline/simulated_data.py`: simulated DTMC/CTMC trace generator with action effects and emissions; needed to create reproducible data without external logs.
- `src/pipeline/experiments.py`: orchestrates all models into one result dict and summary metrics; needed for consistent cross-model experiments.
- `src/pipeline/plots.py`: shared plotting helpers for state counts, CT reliability, and RL curves; needed to avoid duplicate plotting code.

## Test File Guide (What + Why)
- `tests/test_dtmc_ctmc.py`: validates transition/generator properties and absorption behavior; protects core reliability math from regressions.
- `tests/test_hmm.py`: checks Forward/Viterbi output shape/sanity; protects hidden-state inference pipeline.
- `tests/test_mdp_rl_pomdp.py`: checks value iteration, RL loop execution, and normalized beliefs; protects control/inference integration.

## How To Run Tests
Run all tests:

```bash
.venv/bin/python -m pytest tests -q
```

Run one test file:

```bash
.venv/bin/python -m pytest tests/test_mdp_rl_pomdp.py -q
```

Run one specific test function:

```bash
.venv/bin/python -m pytest tests/test_dtmc_ctmc.py::test_ctmc_generator_and_transition -q
```

Note: prefer `pytest` commands above instead of `python tests/<file>.py` because tests use package imports.

## Data Dictionary (Column Meanings)
This section documents the CSV columns generated by the project.

### `data/simulated_raw/dt_trace_run_XX.csv`
- `step`: discrete timestep index.
- `time_hours`: elapsed simulated time in hours for that step.
- `state_idx`: numeric index of current hidden state.
- `state`: current hidden state name.
- `action_idx`: numeric index of selected action.
- `action`: selected action name.
- `next_state_idx`: numeric index of next hidden state.
- `next_state`: next hidden state name.
- `reward`: immediate reward after transition.
- `temperature`: simulated GPU temperature.
- `ecc_count`: simulated ECC error count at step.
- `xid_code`: simulated NVIDIA XID code category.
- `utilization`: simulated GPU utilization percentage.
- `power_usage`: simulated power draw.
- `retired_pages`: simulated retired memory pages count.

### `data/simulated_raw/ct_trace_run_XX.csv`
- `event_idx`: continuous-time event index.
- `time_hours`: event timestamp in hours.
- `holding_hours`: time spent in previous state before transition.
- `state_idx`: numeric index of current hidden state.
- `state`: current hidden state name.
- `action_idx`: numeric index of selected action.
- `action`: selected action name.
- `next_state_idx`: numeric index of next hidden state.
- `next_state`: next hidden state name.
- `reward`: immediate reward after transition.
- `temperature`: simulated GPU temperature at event.
- `ecc_count`: simulated ECC count at event.
- `xid_code`: simulated XID category at event.
- `utilization`: simulated utilization at event.
- `power_usage`: simulated power draw at event.
- `retired_pages`: simulated retired pages count at event.

### `data/simulated_processed/multi_run_summary.csv`
- `run_id`: run number.
- `seed`: random seed used for run.
- `horizon_steps`: DTMC horizon in steps.
- `ct_horizon_hours`: CTMC horizon in hours.
- `DTMC_MTTF_from_Healthy_steps`: expected DT steps before absorption from Healthy.
- `Mean_Q_learning_return`: mean episodic return for Q-learning.
- `Mean_SARSA_return`: mean episodic return for SARSA.
- `Mean_R_learning_return`: mean episodic average reward for R-learning.
- `R_learning_rho`: estimated long-run average reward in R-learning.
- `TD0_V_Healthy`: TD(0) value estimate at Healthy state.
- `TDlambda_V_Healthy`: TD(lambda) value estimate at Healthy state.

### `results/files/reference_dt_trace.csv`
- Same columns and meanings as `data/simulated_raw/dt_trace_run_XX.csv`.

### `results/files/reference_ct_trace.csv`
- Same columns and meanings as `data/simulated_raw/ct_trace_run_XX.csv`.

### `results/files/ctmc_state_probabilities.csv`
- `time_hours`: continuous time grid.
- `p_Healthy`: probability of Healthy state at time t.
- `p_Degraded`: probability of Degraded state at time t.
- `p_MaintenanceRequired`: probability of MaintenanceRequired state at time t.
- `p_FailedRecoverable`: probability of FailedRecoverable state at time t.
- `p_FailedPermanent`: probability of FailedPermanent state at time t.

### `results/files/hmm_forward_posterior.csv`
- `post_Healthy`: forward posterior probability of Healthy.
- `post_Degraded`: forward posterior probability of Degraded.
- `post_MaintenanceRequired`: forward posterior probability of MaintenanceRequired.
- `post_FailedRecoverable`: forward posterior probability of FailedRecoverable.
- `post_FailedPermanent`: forward posterior probability of FailedPermanent.

### `results/files/hmm_true_vs_viterbi.csv`
- `step`: timestep index.
- `true_state_idx`: latent state index from generated trajectory.
- `viterbi_state_idx`: decoded state index from Viterbi.

### `results/files/rl_returns_reference.csv`
- `episode`: episode index.
- `q_learning_return`: episodic return under Q-learning.
- `sarsa_return`: episodic return under SARSA.
- `r_learning_avg_reward`: per-episode average reward under R-learning.

### `results/files/mdp_policy_table.csv`
- `state`: state name.
- `action_idx`: selected optimal action index for state.
- `action_name`: selected optimal action name for state.

### `results/files/mdp_value_function.csv`
- `state`: state name.
- `value`: optimal value function at state.

### `results/files/mdp_q_table.csv`
- `Unnamed: 0`: action name (CSV index column).
- `Healthy`: Q-value for action at Healthy state.
- `Degraded`: Q-value for action at Degraded state.
- `MaintenanceRequired`: Q-value for action at MaintenanceRequired state.
- `FailedRecoverable`: Q-value for action at FailedRecoverable state.
- `FailedPermanent`: Q-value for action at FailedPermanent state.

### `results/files/non_markov_risk_multiplier.csv`
- `step`: timestep index.
- `risk_multiplier`: history-based risk inflation factor.

### `results/files/sensitivity_grid.csv`
- `gamma`: discount factor used in sensitivity grid.
- `failure_scale`: multiplier for failure-related cost intensity.
- `policy_score`: aggregate control score under parameter pair.
- `expected_uptime`: expected uptime proxy under parameter pair.

### `results/files/sensitivity_policy_score_grid.csv`
- `failure_scale`: row key for failure scale.
- `0.85`, `0.9`, `0.95`, `0.98`: policy score at corresponding gamma values.

Generated folders:
- `data/simulated_raw/` (multi-run traces)
- `data/simulated_processed/` (aggregated CSVs)
- `results/mandatory_plots/` (final report/demo figures)
- `results/files/` (supporting numerical tables)
- `results/sensitivity.csv`

## Notes
- The code is designed for pedagogical clarity and reproducibility.
- The simulated data generator is calibrated with assumptions consistent with published cluster reliability patterns.
- The original typo folder `reseacrh ` has been normalized to `research/`.
