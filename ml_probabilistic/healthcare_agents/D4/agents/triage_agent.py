"""Helpers for GP1 triage prompt formatting and output parsing."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

ALLOWED_ROUTES = {"Radiologue", "Pathologiste"}

_ROUTE_ALIASES = {
    "radiologue": "Radiologue",
    "radiology": "Radiologue",
    "radiologist": "Radiologue",
    "pathologiste": "Pathologiste",
    "pathology": "Pathologiste",
    "pathologist": "Pathologiste",
}


@dataclass(frozen=True)
class GP1TriageOutput:
    think: str
    answer: str
    confidence: Optional[float]
    raw_text: str


_TAG_THINK = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
_TAG_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_TAG_CONFIDENCE = re.compile(r"<(?:confidence|confiance)>(.*?)</(?:confidence|confiance)>", re.IGNORECASE | re.DOTALL)


def normalize_route(route: str) -> Optional[str]:
    cleaned = route.strip().lower()
    if cleaned in _ROUTE_ALIASES:
        return _ROUTE_ALIASES[cleaned]
    if route.strip() in ALLOWED_ROUTES:
        return route.strip()
    return None


def build_gp1_prompt(question: str, image_ref: str) -> str:
    """Builds a strict GP1 triage prompt that requires tagged output."""
    return (
        "You are GP1 in a medical multi-agent system. "
        "Route the case to exactly one specialist.\n"
        "Allowed specialists: Radiologue, Pathologiste.\n"
        "Output exactly with tags <think>, <answer>, and optional <confidence>.\n"
        f"Image reference: {image_ref}\n"
        f"Question: {question}\n"
    )


def parse_gp1_output(text: str) -> Optional[GP1TriageOutput]:
    """Parses model output and returns a normalized triage object if valid."""
    think_match = _TAG_THINK.search(text)
    answer_match = _TAG_ANSWER.search(text)
    if not think_match or not answer_match:
        return None

    raw_route = answer_match.group(1).strip()
    normalized_route = normalize_route(raw_route)
    if normalized_route is None:
        return None

    confidence_match = _TAG_CONFIDENCE.search(text)
    confidence: Optional[float] = None
    if confidence_match:
        raw_conf = confidence_match.group(1).strip()
        try:
            parsed = float(raw_conf)
            confidence = max(0.0, min(1.0, parsed))
        except ValueError:
            confidence = None

    return GP1TriageOutput(
        think=think_match.group(1).strip(),
        answer=normalized_route,
        confidence=confidence,
        raw_text=text,
    )
