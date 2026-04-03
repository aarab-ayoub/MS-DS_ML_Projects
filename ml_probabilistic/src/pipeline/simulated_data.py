from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from ..core.config import (
    ACTION_EFFECTS,
    ACTION_TO_IDX,
    ACTIONS,
    BASE_GENERATOR,
    BASE_TRANSITION,
    EMISSION_PARAMS,
    ProjectConfig,
    STATE_TO_IDX,
    STATES,
    XID_CODES,
    build_generator,
)


PolicyFn = Callable[[int, Dict[str, float], int], int]


def _normalize_row(row: np.ndarray) -> np.ndarray:
	row = np.clip(row, 1e-12, None)
	return row / row.sum()


def apply_action_effects(base_p: np.ndarray, action_name: str) -> np.ndarray:
	p = base_p.copy().astype(float)
	effects = ACTION_EFFECTS.get(action_name, {})
	for (i, j), multiplier in effects.items():
		p[i, j] *= multiplier
	for i in range(p.shape[0]):
		p[i] = _normalize_row(p[i])
	return p


def sample_observation(state_idx: int, rng: np.random.Generator) -> Dict[str, float]:
	params = EMISSION_PARAMS[STATES[state_idx]]
	temp_mu, temp_sigma = params["temperature"]
	util_mu, util_sigma = params["utilization"]
	p_mu, p_sigma = params["power_usage"]

	xid = int(rng.choice(XID_CODES, p=params["xid_probs"]))
	obs = {
		"temperature": float(rng.normal(temp_mu, temp_sigma)),
		"ecc_count": int(rng.poisson(params["ecc_lambda"])),
		"xid_code": xid,
		"utilization": float(np.clip(rng.normal(util_mu, util_sigma), 0.0, 100.0)),
		"power_usage": float(np.clip(rng.normal(p_mu, p_sigma), 0.0, None)),
		"retired_pages": int(rng.poisson(params["retired_pages_lambda"])),
	}
	return obs


def compute_reward(state_idx: int, action_idx: int, obs: Dict[str, float]) -> float:
	uptime = 1.0 if state_idx in (STATE_TO_IDX["Healthy"], STATE_TO_IDX["Degraded"]) else 0.3
	failure_penalty = 1.0 if state_idx == STATE_TO_IDX["FailedPermanent"] else 0.0
	recovery_penalty = 1.0 if state_idx == STATE_TO_IDX["FailedRecoverable"] else 0.0
	action_cost = 0.0 if ACTIONS[action_idx] == "keep_running" else 0.3
	thermal_penalty = max(0.0, (obs["temperature"] - 78.0) / 10.0)
	return 1.2 * uptime - 15.0 * failure_penalty - 3.0 * recovery_penalty - action_cost - thermal_penalty


def default_policy(state_idx: int, obs: Dict[str, float], t: int) -> int:
	if state_idx == STATE_TO_IDX["FailedRecoverable"]:
		return ACTION_TO_IDX["reboot"]
	if obs["ecc_count"] >= 4 or obs["temperature"] >= 78:
		return ACTION_TO_IDX["power_cap"]
	if state_idx == STATE_TO_IDX["MaintenanceRequired"]:
		return ACTION_TO_IDX["isolate_maintain"]
	return ACTION_TO_IDX["keep_running"]


def generate_dtmc_trace(
	cfg: ProjectConfig,
	start_state: str = "Healthy",
	policy_fn: Optional[PolicyFn] = None,
) -> pd.DataFrame:
	rng = np.random.default_rng(cfg.seed)
	policy_fn = policy_fn or default_policy

	cur = STATE_TO_IDX[start_state]
	rows = []
	for t in range(cfg.horizon_steps):
		obs = sample_observation(cur, rng)
		action = policy_fn(cur, obs, t)
		p_action = apply_action_effects(BASE_TRANSITION, ACTIONS[action])
		nxt = int(rng.choice(len(STATES), p=p_action[cur]))
		reward = compute_reward(nxt, action, obs)

		rows.append(
			{
				"step": t,
				"time_hours": t * cfg.dt_hours,
				"state_idx": cur,
				"state": STATES[cur],
				"action_idx": action,
				"action": ACTIONS[action],
				"next_state_idx": nxt,
				"next_state": STATES[nxt],
				"reward": reward,
				**obs,
			}
		)
		cur = nxt
	return pd.DataFrame(rows)


def generate_ctmc_trace(
	cfg: ProjectConfig,
	start_state: str = "Healthy",
	policy_fn: Optional[PolicyFn] = None,
) -> pd.DataFrame:
	rng = np.random.default_rng(cfg.seed + 1)
	policy_fn = policy_fn or default_policy

	off_diag = BASE_GENERATOR.copy()
	base_a = build_generator(off_diag)

	t = 0.0
	k = 0
	cur = STATE_TO_IDX[start_state]
	rows = []

	while t < cfg.ctmc_horizon_hours:
		obs = sample_observation(cur, rng)
		action = policy_fn(cur, obs, k)

		p_action = apply_action_effects(BASE_TRANSITION, ACTIONS[action])
		a = base_a.copy()
		for i in range(a.shape[0]):
			exit_rate = -a[i, i]
			if exit_rate <= 0:
				continue
			probs = p_action[i].copy()
			probs[i] = 0.0
			if probs.sum() <= 0:
				a[i, :] = 0.0
				continue
			probs /= probs.sum()
			for j in range(a.shape[1]):
				if i != j:
					a[i, j] = exit_rate * probs[j]
			a[i, i] = -a[i, :].sum() + a[i, i]

		total_rate = -a[cur, cur]
		if total_rate <= 1e-12:
			break

		holding = float(rng.exponential(1.0 / total_rate))
		probs_next = a[cur].copy()
		probs_next[cur] = 0.0
		probs_next = _normalize_row(probs_next)
		nxt = int(rng.choice(len(STATES), p=probs_next))
		reward = compute_reward(nxt, action, obs)

		rows.append(
			{
				"event_idx": k,
				"time_hours": t,
				"holding_hours": holding,
				"state_idx": cur,
				"state": STATES[cur],
				"action_idx": action,
				"action": ACTIONS[action],
				"next_state_idx": nxt,
				"next_state": STATES[nxt],
				"reward": reward,
				**obs,
			}
		)
		t += holding
		cur = nxt
		k += 1

	return pd.DataFrame(rows)


if __name__ == "__main__":
	cfg = ProjectConfig()
	df_dt = generate_dtmc_trace(cfg)
	df_ct = generate_ctmc_trace(cfg)
	print("DTMC trace shape:", df_dt.shape)
	print("CTMC trace shape:", df_ct.shape)
	print(df_dt.head(3).to_string(index=False))
