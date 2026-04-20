"""Reward components for GP1 triage in D4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class RewardWeights:
    format_weight: float = 0.25
    route_weight: float = 0.50
    stability_weight: float = 0.15
    risk_weight: float = 0.10


@dataclass(frozen=True)
class TriageExample:
    sample_id: str
    question: str
    gold_route: str
    difficulty: str = "easy"
    uncertainty_flag: bool = False
    image_ref: Optional[str] = None
    image_path: Optional[str] = None
    source_dataset: str = "unknown"


@dataclass(frozen=True)
class TriageRollout:
    predicted_route: Optional[str]
    parse_ok: bool
    confidence: Optional[float]


class TriageRewardEngine:
    """Computes a weighted reward with interpretable components."""

    def __init__(self, weights: RewardWeights, confidence_threshold: float = 0.8) -> None:
        self.weights = weights
        self.confidence_threshold = confidence_threshold

    def compute(
        self,
        example: TriageExample,
        rollout: TriageRollout,
        route_stability: float,
    ) -> Dict[str, float]:
        """Returns reward components and total reward."""
        r_format = self._format_reward(rollout)
        r_route = self._route_reward(example, rollout)
        r_stability = self._stability_reward(route_stability)
        r_risk = self._risk_reward(example, rollout)

        total = (
            self.weights.format_weight * r_format
            + self.weights.route_weight * r_route
            + self.weights.stability_weight * r_stability
            + self.weights.risk_weight * r_risk
        )

        return {
            "r_format": r_format,
            "r_route": r_route,
            "r_stability": r_stability,
            "r_risk": r_risk,
            "total": total,
        }

    @staticmethod
    def _format_reward(rollout: TriageRollout) -> float:
        return 1.0 if rollout.parse_ok else -1.0

    @staticmethod
    def _route_reward(example: TriageExample, rollout: TriageRollout) -> float:
        if not rollout.parse_ok or rollout.predicted_route is None:
            return -1.0
        return 1.0 if rollout.predicted_route == example.gold_route else -1.0

    @staticmethod
    def _stability_reward(route_stability: float) -> float:
        clipped = max(0.0, min(1.0, route_stability))
        return (2.0 * clipped) - 1.0

    def _risk_reward(self, example: TriageExample, rollout: TriageRollout) -> float:
        """Penalizes high-confidence wrong routing on uncertain cases."""
        if not rollout.parse_ok or rollout.predicted_route is None:
            return -1.0

        is_wrong = rollout.predicted_route != example.gold_route
        conf = rollout.confidence if rollout.confidence is not None else 0.5

        if is_wrong and example.uncertainty_flag and conf >= self.confidence_threshold:
            return -1.0
        if is_wrong:
            return -0.5
        return 1.0
