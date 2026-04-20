# D4 - GP1 Triage Workspace

This folder is reserved for Group D4 work:
- GP1 triage logic
- GRPO + LoRA training setup
- Reward triage design and experiments

Suggested local structure:
- agents/
- rl/
- configs/
- notebooks/
- checkpoints/ (ignored by git)
- outputs/ (ignored by git)

Use this directory as the main working area for D4 implementation.

## Current training entrypoint

- `train_gp1.py`: config-driven GP1 training runner with dry-run and local-data modes.
- `agents/triage_agent.py`: strict tag parser and route normalizer.
- `rl/reward_triage.py`: compositional triage reward engine.
- `data/triage_dataset.py`: local D1/D2 adapter with Colab-path remapping.
- `configs/gp1_triage.yaml`: baseline D4 hyperparameters.

## Quick start

From `healthcare_agents/D4`:

```bash
../.venv/bin/python train_gp1.py
```

This writes:
- `outputs/gp1_summary_real.json` (or `outputs/gp1_summary_dryrun.json` in dry-run mode)
- `outputs/dispatcher_kpi_report.json`
- `outputs/dispatcher_eval_rows.csv`

Default config now uses local-data mode with `model.backend: heuristic`.
If you want to use D3 model inference, set `model.backend: d3` and keep `paths.d3_model_path` valid.
