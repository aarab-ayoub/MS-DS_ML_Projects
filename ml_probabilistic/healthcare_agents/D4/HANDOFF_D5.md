# D4 to D5 Handoff Guide

## 1) What D4 Delivers
D4 provides GP1 triage training/evaluation for routing medical VQA cases to one specialist:
- Radiologue
- Pathologiste

The D4 runner consumes local D1 and D2 data, computes reward-driven rollout metrics, and writes operational artifacts that D5 can consume directly.

## 2) Single Command to Reproduce
Run from the `healthcare_agents/D4` directory:

```bash
../.venv/bin/python train_gp1.py
```

Alternative config:

```bash
../.venv/bin/python train_gp1.py --config configs/gp1_triage.yaml
```

## 3) Runtime Backends
Configured in `configs/gp1_triage.yaml`:
- `model.backend: heuristic` (default, stable baseline)
- `model.backend: d3` (uses D3 encoder + GGUF model)

When `d3` is enabled, outputs are normalized into the GP1 tag contract when possible.

## 4) Output Artifacts (Contract)
The run writes files to `paths.output_dir` (default: `./outputs`):

1. `gp1_summary_real.json` (or `gp1_summary_dryrun.json`)
- High-level run metadata (mode, backend, steps, average reward).

2. `dispatcher_kpi_report.json`
- `overall`: parse and accuracy rates.
- `confusion_matrix`: gold route vs predicted route (including `UNPARSEABLE`).
- `per_source`: metrics split by dataset source (`D1_VQA_RAD`, `D2_PathVQA`).
- `per_difficulty`: metrics split by difficulty bucket.
- `reward_stats`: average reward and route stability.

3. `dispatcher_eval_rows.csv`
- Step-level rows for auditability and downstream analytics.

4. `d5_handoff_manifest.json`
- Machine-readable handoff payload combining summary, KPI overview, contract metadata, and artifact paths.

## 5) CSV Schema for D5
`dispatcher_eval_rows.csv` columns:
- `step`
- `sample_id`
- `source`
- `difficulty`
- `gold_route`
- `pred_route`
- `route_source`
- `parse_ok`
- `confidence`
- `route_stability`
- `reward_total`
- `r_format`
- `r_route`
- `r_stability`
- `r_risk`

`route_source` values:
- `heuristic`: produced by heuristic backend.
- `d3_tagged`: D3 output already matched GP1 tags.
- `d3_normalized`: D3 free-text route was normalized into GP1 tags.
- `heuristic_fallback`: D3 output had no route signal; fallback logic used.

Heuristic note:
- The heuristic router applies a light source prior for ambiguous questions: D1 leans Radiologue and D2 leans Pathologiste.

## 6) Recommended D5 Consumption Path
1. Read `d5_handoff_manifest.json` first.
2. Use `dispatcher_kpi_report.json` to detect weak slices by source/difficulty.
3. Use `dispatcher_eval_rows.csv` for sample-level routing QA and confidence filters.
4. Prioritize rows with `parse_ok=true` and track `route_source` drift over time.

## 7) Known Constraints
- Current loop is a GRPO-style scaffold with reward logging; it is not a full optimizer + LoRA fine-tuning pipeline yet.
- D3 backend quality depends on encoder output consistency; monitor `route_source` distribution for fallback dominance.
- Heuristic accuracy includes source-aware disambiguation, so compare fairly when benchmarking against D3.

## 8) Delivery Checklist
- [x] Local D1/D2 ingestion wired
- [x] Colab-to-local path remapping implemented
- [x] KPI report + confusion matrix generated
- [x] Step-level CSV generated
- [x] D5 machine-readable manifest generated
- [x] Handoff documentation added
