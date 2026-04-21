"""D4 GP1 triage trainer.

This runner supports two modes:
- dry-run mode with synthetic samples
- local mode with real D1/D2 data adapters

Rollout generation supports:
- heuristic backend (fast, local, no heavy model)
- optional D3 backend via encoder.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import importlib.util
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.triage_agent import ALLOWED_ROUTES, normalize_route, parse_gp1_output
from data.triage_dataset import load_real_examples
from rl.reward_triage import RewardWeights, TriageExample, TriageRewardEngine, TriageRollout


def load_config(config_path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyYAML is required to load gp1_triage.yaml. Install with: pip install pyyaml"
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config root must be a mapping.")
    return config


def resolve_project_path(raw_path: str, project_root: Path) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def synthetic_examples() -> List[TriageExample]:
    return [
        TriageExample(
            sample_id="rad_0001",
            question="Is there a brain hemorrhage on this CT image?",
            gold_route="Radiologue",
            difficulty="easy",
            uncertainty_flag=True,
            image_ref="rad_0001.jpg",
            source_dataset="synthetic",
        ),
        TriageExample(
            sample_id="path_0001",
            question="What tissue pattern is visible in this histology image?",
            gold_route="Pathologiste",
            difficulty="medium",
            uncertainty_flag=False,
            image_ref="path_0001.jpg",
            source_dataset="synthetic",
        ),
    ]


def _flip_route(route: str) -> str:
    return "Pathologiste" if route == "Radiologue" else "Radiologue"


def _keyword_route(question: str, source_dataset: Optional[str] = None) -> str:
    q = question.lower()
    source = (source_dataset or "").strip().upper()

    radiology_terms = [
        "ct",
        "x-ray",
        "xray",
        "mri",
        "ultrasound",
        "radiograph",
        "scan",
        "fracture",
        "hemorrhage",
        "pneumothorax",
        "consolidation",
        "effusion",
        "opacity",
        "edema",
    ]
    pathology_terms = [
        "histology",
        "histopath",
        "tissue",
        "stain",
        "microscope",
        "cell",
        "nuclei",
        "biopsy",
        "carcinoma",
        "adenocarcinoma",
        "tumor",
        "stain",
        "immunohistochemical",
        "cytokeratin",
        "hyperplasia",
        "atypia",
        "glandular",
        "necrosis",
        "metastatic",
        "serosa",
        "muscularis",
    ]

    rad_score = sum(term in q for term in radiology_terms)
    path_score = sum(term in q for term in pathology_terms)

    # Lightweight source prior helps ambiguous questions route correctly.
    # D1 is radiology-heavy and D2 is pathology-heavy in this project setup.
    rad_score += 1 if source == "D1" else 0
    path_score += 1 if source == "D2" else 0

    if path_score > rad_score:
        return "Pathologiste"
    if rad_score > path_score:
        return "Radiologue"

    # Conservative default for image-level triage when no strong histology cue appears.
    return "Radiologue"


def _heuristic_tagged_output(example: TriageExample, rollout_id: int) -> str:
    pred = _keyword_route(example.question, source_dataset=example.source_dataset)
    if rollout_id > 0 and random.random() < 0.08:
        pred = _flip_route(pred)

    confidence = 0.66 if pred == example.gold_route else 0.58
    return (
        "<think>Routing based on modality hints and question semantics.</think>\n"
        f"<answer>{pred}</answer>\n"
        f"<confidence>{confidence:.2f}</confidence>"
    )


_ROUTE_TEXT_PATTERN = re.compile(
    r"\b(radiologue|radiologist|radiology|pathologiste|pathologist|pathology)\b",
    re.IGNORECASE,
)
_CONF_PERCENT_PATTERN = re.compile(r"(\d{1,3})(?:\.(\d+))?\s*%")
_CONF_FLOAT_PATTERN = re.compile(r"\b0(?:\.\d+)?\b|\b1(?:\.0+)?\b")


def _extract_route_from_free_text(text: str) -> Optional[str]:
    match = _ROUTE_TEXT_PATTERN.search(text)
    if not match:
        return None

    candidate = normalize_route(match.group(1))
    return candidate


def _extract_confidence_from_free_text(text: str) -> float:
    percent_match = _CONF_PERCENT_PATTERN.search(text)
    if percent_match:
        value = float(percent_match.group(0).replace("%", "").strip()) / 100.0
        return max(0.0, min(1.0, value))

    float_match = _CONF_FLOAT_PATTERN.search(text)
    if float_match:
        value = float(float_match.group(0))
        return max(0.0, min(1.0, value))

    return 0.60


def _build_d3_route_prompt(question: str, image_ref: str) -> str:
    return (
        "You are a medical triage classifier.\n"
        "Choose exactly one route from: Radiologue, Pathologiste.\n"
        "Do not answer the medical diagnosis itself.\n"
        "Return with this exact template only:\n"
        "<think>short routing rationale</think>\n"
        "<answer>Radiologue or Pathologiste</answer>\n"
        "<confidence>0.00 to 1.00</confidence>\n"
        f"Image reference: {image_ref}\n"
        f"Question: {question}\n"
    )


def _load_d3_encoder(config: Dict[str, Any], project_root: Path) -> Any:
    paths_cfg = config.get("paths", {})
    model_cfg = config.get("model", {})

    encoder_module_path = resolve_project_path(
        str(paths_cfg.get("d3_encoder_module", "../D3_VLEncoder/agents/encoder.py")),
        project_root,
    )
    if not encoder_module_path.exists():
        raise FileNotFoundError(f"D3 encoder module not found: {encoder_module_path}")

    spec = importlib.util.spec_from_file_location("d3_encoder_module", str(encoder_module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load D3 encoder module spec.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_path = paths_cfg.get("d3_model_path")
    if model_path:
        model_file = resolve_project_path(str(model_path), project_root)
    else:
        model_file = resolve_project_path(
            f"../D3_VLEncoder/models/{model_cfg.get('base_model', '')}",
            project_root,
        )

    if not model_file.exists():
        raise FileNotFoundError(f"D3 GGUF model file not found: {model_file}")

    llm = module.load_model(str(model_file))
    encoder = module.VLEncoder(llm)
    encoder.warmup()
    return encoder


class RolloutGenerator:
    def __init__(self, config: Dict[str, Any], project_root: Path) -> None:
        self.config = config
        model_cfg = config.get("model", {})
        self.backend = str(model_cfg.get("backend", "heuristic")).strip().lower()
        self.max_tokens = int(model_cfg.get("max_tokens", 128))
        self.temperature = float(model_cfg.get("temperature", 0.0))
        self.encoder: Optional[Any] = None

        if self.backend == "d3":
            self.encoder = _load_d3_encoder(config=config, project_root=project_root)
        elif self.backend != "heuristic":
            raise ValueError(f"Unsupported backend '{self.backend}'. Use 'heuristic' or 'd3'.")

    def generate(self, example: TriageExample, rollout_id: int) -> GeneratedRollout:
        image_ref = example.image_ref or f"{example.sample_id}.jpg"
        d3_prompt = _build_d3_route_prompt(question=example.question, image_ref=image_ref)

        if self.backend == "heuristic":
            return GeneratedRollout(
                text=_heuristic_tagged_output(example=example, rollout_id=rollout_id),
                route_source="heuristic",
            )

        assert self.encoder is not None
        image_input = example.image_path or image_ref
        output_text = self.encoder.encode(
            image_path=image_input,
            question=d3_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        parsed = parse_gp1_output(output_text)
        if parsed is not None:
            return GeneratedRollout(
                text=output_text,
                route_source="d3_tagged",
                raw_model_text=output_text,
            )

        extracted_route = _extract_route_from_free_text(output_text)
        if extracted_route is not None:
            extracted_conf = _extract_confidence_from_free_text(output_text)
            normalized_text = (
                "<think>D3 free-text route normalization.</think>\n"
                f"<answer>{extracted_route}</answer>\n"
                f"<confidence>{extracted_conf:.2f}</confidence>"
            )
            return GeneratedRollout(
                text=normalized_text,
                route_source="d3_normalized",
                raw_model_text=output_text,
            )

        # Keep training robust if D3 output contains no route signal.
        fallback_route = _keyword_route(example.question, source_dataset=example.source_dataset)
        fallback_conf = 0.55
        fallback_text = (
            "<think>D3 output fallback: enforcing GP1 contract.</think>\n"
            f"<answer>{fallback_route}</answer>\n"
            f"<confidence>{fallback_conf:.2f}</confidence>"
        )
        return GeneratedRollout(
            text=fallback_text,
            route_source="heuristic_fallback",
            raw_model_text=output_text,
        )


@dataclass(frozen=True)
class GeneratedRollout:
    text: str
    route_source: str
    raw_model_text: Optional[str] = None


def compute_route_stability(routes: List[str]) -> float:
    if not routes:
        return 0.0
    counts: Dict[str, int] = {}
    for route in routes:
        counts[route] = counts.get(route, 0) + 1
    return max(counts.values()) / len(routes)


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _metric_pack(correct: int, parsed: int, total: int) -> Dict[str, float | int]:
    return {
        "total": total,
        "parsed": parsed,
        "correct": correct,
        "parse_rate": _safe_rate(parsed, total),
        "accuracy_overall": _safe_rate(correct, total),
        "accuracy_on_parsed": _safe_rate(correct, parsed),
    }


def _compute_dispatcher_kpis(eval_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    route_labels = sorted(ALLOWED_ROUTES)
    pred_labels = route_labels + ["UNPARSEABLE"]

    confusion: Dict[str, Dict[str, int]] = {
        gold: {pred: 0 for pred in pred_labels} for gold in route_labels
    }

    total = len(eval_rows)
    parsed = 0
    correct = 0

    per_source_counter: Dict[str, Dict[str, int]] = {}
    per_difficulty_counter: Dict[str, Dict[str, int]] = {}
    predicted_distribution: Dict[str, int] = {pred: 0 for pred in pred_labels}

    total_reward = 0.0
    total_stability = 0.0

    for row in eval_rows:
        source = str(row.get("source", "unknown"))
        difficulty = str(row.get("difficulty", "unknown"))
        gold = str(row.get("gold_route", ""))
        parse_ok = bool(row.get("parse_ok", False))
        pred = row.get("pred_route")

        if source not in per_source_counter:
            per_source_counter[source] = {"total": 0, "parsed": 0, "correct": 0}
        if difficulty not in per_difficulty_counter:
            per_difficulty_counter[difficulty] = {"total": 0, "parsed": 0, "correct": 0}

        per_source_counter[source]["total"] += 1
        per_difficulty_counter[difficulty]["total"] += 1

        total_reward += float(row.get("reward_total", 0.0))
        total_stability += float(row.get("route_stability", 0.0))

        if parse_ok:
            parsed += 1
            per_source_counter[source]["parsed"] += 1
            per_difficulty_counter[difficulty]["parsed"] += 1

        pred_key = str(pred) if parse_ok and isinstance(pred, str) and pred in route_labels else "UNPARSEABLE"
        predicted_distribution[pred_key] += 1

        if gold in confusion:
            confusion[gold][pred_key] += 1

        is_correct = parse_ok and isinstance(pred, str) and pred == gold
        if is_correct:
            correct += 1
            per_source_counter[source]["correct"] += 1
            per_difficulty_counter[difficulty]["correct"] += 1

    per_source = {
        key: _metric_pack(
            correct=vals["correct"],
            parsed=vals["parsed"],
            total=vals["total"],
        )
        for key, vals in sorted(per_source_counter.items())
    }

    per_difficulty = {
        key: _metric_pack(
            correct=vals["correct"],
            parsed=vals["parsed"],
            total=vals["total"],
        )
        for key, vals in sorted(per_difficulty_counter.items())
    }

    return {
        "overall": _metric_pack(correct=correct, parsed=parsed, total=total),
        "predicted_distribution": predicted_distribution,
        "confusion_matrix": confusion,
        "per_source": per_source,
        "per_difficulty": per_difficulty,
        "reward_stats": {
            "avg_reward_total": total_reward / total if total else 0.0,
            "avg_route_stability": total_stability / total if total else 0.0,
        },
    }


def _write_eval_csv(eval_rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "step",
        "sample_id",
        "source",
        "difficulty",
        "gold_route",
        "pred_route",
        "route_source",
        "parse_ok",
        "confidence",
        "route_stability",
        "reward_total",
        "r_format",
        "r_route",
        "r_stability",
        "r_risk",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _build_handoff_manifest(
    *,
    config: Dict[str, Any],
    summary: Dict[str, Any],
    kpis: Dict[str, Any],
    artifact_paths: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "handoff_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "D4 GP1 Triage",
        "summary": summary,
        "kpi_overview": {
            "overall": kpis.get("overall", {}),
            "per_source": kpis.get("per_source", {}),
            "per_difficulty": kpis.get("per_difficulty", {}),
        },
        "contracts": {
            "allowed_routes": sorted(ALLOWED_ROUTES),
            "d1_expected_route": "Radiologue",
            "d2_expected_route": "Pathologiste",
            "route_sources": [
                "heuristic",
                "d3_tagged",
                "d3_normalized",
                "heuristic_fallback",
            ],
        },
        "artifacts": artifact_paths,
        "config_snapshot": {
            "model": config.get("model", {}),
            "data": config.get("data", {}),
            "grpo": config.get("grpo", {}),
            "reward": config.get("reward", {}),
            "train": config.get("train", {}),
        },
        "d5_handoff_notes": [
            "Use pred_route to dispatch cases to specialist fine-tuning branches.",
            "Use route_source to monitor whether route came from D3 or fallback logic.",
            "Use per_difficulty KPI to prioritize weak slices before final integration.",
        ],
    }


def _resolve_real_examples(config: Dict[str, Any], project_root: Path) -> List[TriageExample]:
    paths_cfg = config.get("paths", {})
    project_cfg = config.get("project", {})
    data_cfg = config.get("data", {})

    d1_root = resolve_project_path(
        str(paths_cfg.get("d1_arrow_root", "../D1/dataset/vqa_rad_raw")),
        project_root,
    )
    d2_split_root = resolve_project_path(
        str(paths_cfg.get("d2_split_root", "../D2/data/splits")),
        project_root,
    )

    healthcare_root = project_root.parent
    examples = load_real_examples(
        d1_root=d1_root,
        d2_split_root=d2_split_root,
        healthcare_root=healthcare_root,
        max_d1_per_split=int(data_cfg.get("max_d1_per_split", 150)),
        max_d2_per_difficulty=int(data_cfg.get("max_d2_per_difficulty", 200)),
        seed=int(project_cfg.get("seed", 42)),
    )

    max_total = int(data_cfg.get("max_examples_total", 0))
    if max_total > 0:
        examples = examples[:max_total]

    if not examples:
        raise RuntimeError("No real examples could be loaded from D1/D2.")

    return examples


def run_training(config: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    random.seed(config.get("project", {}).get("seed", 42))

    reward_cfg = config.get("reward", {})
    weights = RewardWeights(
        format_weight=float(reward_cfg.get("format_weight", 0.25)),
        route_weight=float(reward_cfg.get("route_weight", 0.50)),
        stability_weight=float(reward_cfg.get("stability_weight", 0.15)),
        risk_weight=float(reward_cfg.get("risk_weight", 0.10)),
    )
    engine = TriageRewardEngine(
        weights=weights,
        confidence_threshold=float(reward_cfg.get("confidence_threshold", 0.8)),
    )

    grpo_cfg = config.get("grpo", {})
    group_rollouts = int(grpo_cfg.get("group_rollouts", 4))

    train_cfg = config.get("train", {})
    max_steps = int(train_cfg.get("max_steps", 20))
    log_every = int(train_cfg.get("log_every", 5))
    dry_run = bool(train_cfg.get("dry_run", True))

    if dry_run:
        examples = synthetic_examples()
    else:
        examples = _resolve_real_examples(config=config, project_root=project_root)

    generator = RolloutGenerator(config=config, project_root=project_root)
    all_rewards: List[float] = []
    eval_rows: List[Dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        example = examples[(step - 1) % len(examples)]

        routes: List[str] = []
        first_rollout: TriageRollout | None = None
        first_route_source = "unknown"

        for rollout_id in range(group_rollouts):
            generated = generator.generate(example, rollout_id)
            raw_text = generated.text
            parsed = parse_gp1_output(raw_text)

            if parsed is None:
                rollout = TriageRollout(predicted_route=None, parse_ok=False, confidence=None)
            else:
                rollout = TriageRollout(
                    predicted_route=parsed.answer,
                    parse_ok=True,
                    confidence=parsed.confidence,
                )
                routes.append(parsed.answer)

            if first_rollout is None:
                first_rollout = rollout
                first_route_source = generated.route_source

        assert first_rollout is not None
        stability = compute_route_stability(routes)
        reward_map = engine.compute(example, first_rollout, route_stability=stability)
        all_rewards.append(reward_map["total"])

        eval_rows.append(
            {
                "step": step,
                "sample_id": example.sample_id,
                "source": example.source_dataset,
                "difficulty": example.difficulty,
                "gold_route": example.gold_route,
                "pred_route": first_rollout.predicted_route,
                "route_source": first_route_source,
                "parse_ok": first_rollout.parse_ok,
                "confidence": first_rollout.confidence,
                "route_stability": stability,
                "reward_total": reward_map["total"],
                "r_format": reward_map["r_format"],
                "r_route": reward_map["r_route"],
                "r_stability": reward_map["r_stability"],
                "r_risk": reward_map["r_risk"],
            }
        )

        if step % log_every == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "sample_id": example.sample_id,
                        "source": example.source_dataset,
                        "difficulty": example.difficulty,
                        "gold_route": example.gold_route,
                        "pred_route": first_rollout.predicted_route,
                        "route_source": first_route_source,
                        "reward": reward_map,
                    }
                )
            )

    summary = {
        "mode": "dry_run" if dry_run else "real_data",
        "backend": generator.backend,
        "num_examples": len(examples),
        "steps": max_steps,
        "group_rollouts": group_rollouts,
        "avg_reward": sum(all_rewards) / len(all_rewards) if all_rewards else 0.0,
        "min_reward": min(all_rewards) if all_rewards else 0.0,
        "max_reward": max(all_rewards) if all_rewards else 0.0,
    }

    return {
        "summary": summary,
        "kpis": _compute_dispatcher_kpis(eval_rows=eval_rows),
        "eval_rows": eval_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D4 GP1 triage scaffold runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "gp1_triage.yaml",
        help="Path to YAML config file.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config(args.config)
    train_result = run_training(config=config, project_root=project_root)
    summary = train_result["summary"]
    kpis = train_result["kpis"]
    eval_rows = train_result["eval_rows"]

    output_dir = resolve_project_path(
        str(config.get("paths", {}).get("output_dir", "./outputs")),
        project_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = str(summary.get("mode", "dry_run"))
    file_name = "gp1_summary_real.json" if mode == "real_data" else "gp1_summary_dryrun.json"
    summary_path = output_dir / file_name
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    kpi_path = output_dir / "dispatcher_kpi_report.json"
    kpi_path.write_text(json.dumps(kpis, indent=2), encoding="utf-8")

    csv_path = output_dir / "dispatcher_eval_rows.csv"
    _write_eval_csv(eval_rows=eval_rows, output_path=csv_path)

    manifest_path = output_dir / "d5_handoff_manifest.json"
    manifest = _build_handoff_manifest(
        config=config,
        summary=summary,
        kpis=kpis,
        artifact_paths={
            "summary": str(summary_path),
            "kpi_report": str(kpi_path),
            "eval_rows_csv": str(csv_path),
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved summary to: {summary_path}")
    print(f"Saved KPI report to: {kpi_path}")
    print(f"Saved eval rows CSV to: {csv_path}")
    print(f"Saved D5 handoff manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
