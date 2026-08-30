"""
Task classifier — fast deterministic heuristic classification.

No LLM call. Latency matters. Uses keyword matching and pattern analysis.
"""

from __future__ import annotations

import enum
import re
from typing import Optional

from harness_core.classifier.types import (
    TaskRequirementProfile,
    TASK_PROFILES,
)


class TaskType(enum.Enum):
    """High-level task categories."""

    IMPLEMENTATION = "implementation"
    BUG_FIX = "bug_fix"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TESTING = "testing"
    RESEARCH = "research"
    REPOSITORY_ANALYSIS = "repository_analysis"
    DOCUMENTATION = "documentation"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


# ── Keyword patterns for each task type ───────────────────────────────────

_TASK_PATTERNS: dict[TaskType, list[str]] = {
    TaskType.BUG_FIX: [
        r"\bfix\b.*\b(bug|error|fail|issue|problem|crash)\b",
        r"\b(bug|error|fail|issue|crash)\b.*\bfix\b",
        r"\b(broken|failing|fails|failed)\b.*\b(test|case|assertion)\b",
        r"\btest(s)?\s+(fail|broken|failing)\b",
        r"\bresolve\b.*\b(error|issue|bug)\b",
    ],
    TaskType.DEBUGGING: [
        r"\bdebug\b",
        r"\bdiagnos(e|is)\b",
        r"\btrace\b.*\b(error|crash|failure)\b",
        r"\broot\s*cause\b",
        r"\binvestigat(e|ing)\b.*\b(fail|error|issue)\b",
    ],
    TaskType.IMPLEMENTATION: [
        r"\bimplement\b",
        r"\bcreate\b.*\b(function|class|module|file|endpoint|api|route)\b",
        r"\badd\b.*\b(feature|function|class|method|support)\b",
        r"\bbuild\b.*\b(component|service|system|module)\b",
        r"\bwrite\b.*\b(code|function|class|module)\b",
        r"\bnew\b.*\b(function|class|module|file)\b",
        r"\b(enhance|improve|upgrade|upgrade|redesign)\b.*\b(this|the|my|a|an)\b",
        r"\badd\s+(dark\s*mode|light\s*mode|theme|responsive|animation)\b",
        r"\bmake\b.*\b(better|faster|nicer|responsive|modern)\b",
        r"\b(push|deploy|upload)\b.*\b(to|this)\b.*\b(git|github|repo)\b",
    ],
    TaskType.REFACTORING: [
        r"\brefactor\b",
        r"\brestructur(e|ing)\b",
        r"\bclean\s*up\b",
        r"\bsimplif(y|ying)\b",
        r"\breorganiz(e|ing)\b",
        r"\bextract\b.*\b(function|class|module)\b",
        r"\bmove\b.*\b(to|into)\b.*\b(module|file|class)\b",
    ],
    TaskType.TESTING: [
        r"\bwrite\b.*\btest(s)?\b",
        r"\badd\b.*\btest(s)?\b",
        r"\btest(s)?\s+(for|coverage|suite)\b",
        r"\bunit\s+test\b",
        r"\bintegration\s+test\b",
        r"\bcoverage\b",
        r"\bmock\b.*\b(test|assertion)\b",
    ],
    TaskType.RESEARCH: [
        r"\bexplain\b",
        r"\bhow\s+(does|do|is|are|would)\b",
        r"\bwhat\s+(is|are|does|do)\b",
        r"\bwh(y|en|ere)\b.*\b(does|do|is|are|would)\b",
        r"\b(analyze|analyse|research|investigate)\b",
        r"\bcompar(e|ing|ison)\b",
        r"\bunderstand\b",
    ],
    TaskType.REPOSITORY_ANALYSIS: [
        r"\binspect\b.*\b(repo|repository|code|codebase)\b",
        r"\b(analyze|analyse)\b.*\b(repo|repository|code|codebase|project)\b",
        r"\bwhat\s+does\s+this\s+(project|repo|codebase)\b",
        r"\bexplain\s+the\s+(architecture|structure|design)\b",
        r"\blist\b.*\b(files|functions|classes|modules)\b",
        r"\bfind\b.*\b(where|how)\b.*\b(implement|defined|used)\b",
    ],
    TaskType.DOCUMENTATION: [
        r"\b(document|documentation|readme|docstring)\b",
        r"\bwrite\b.*\b(doc(s)?|readme|guide|tutorial)\b",
        r"\bupdate\b.*\b(doc(s)?|readme|comment)\b",
        r"\badd\b.*\b(comment|docstring|documentation)\b",
    ],
    TaskType.SECURITY: [
        r"\bsecurity\b",
        r"\b(vulnerability|vulnerabilities|CVE)\b",
        r"\bauth\b",
        r"\bpermission\b",
        r"\bencrypt\b",
        r"\bauthenticat(e|ion)\b",
        r"\bauthoriz(e|ation)\b",
        r"\bhardening\b",
    ],
    TaskType.PERFORMANCE: [
        r"\bperformance\b",
        r"\boptimiz(e|e|ing)\b",
        r"\bslow\b",
        r"\bfast(er|est)?\b",
        r"\blatency\b",
        r"\bcach(e|ing)\b",
        r"\bbenchmark\b",
        r"\bprofil(e|ing)\b",
    ],
}


class TaskClassifier:
    """Fast deterministic task classifier.

    Uses keyword/pattern matching. No LLM calls.
    Supports custom patterns for extensibility.
    """

    def __init__(self) -> None:
        self._patterns: dict[TaskType, list[re.Pattern]] = {}
        for task_type, patterns in _TASK_PATTERNS.items():
            self._patterns[task_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def classify(self, task: str) -> TaskType:
        """Classify a task into a category.

        Returns the best-matching TaskType.
        """
        scores: dict[TaskType, int] = {}

        for task_type, compiled_patterns in self._patterns.items():
            score = 0
            for pattern in compiled_patterns:
                matches = pattern.findall(task)
                score += len(matches)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return TaskType.UNKNOWN

        # Return highest scoring type
        return max(scores, key=scores.get)  # type: ignore[arg-type]

    def get_profile(self, task: str) -> TaskRequirementProfile:
        """Classify a task and return its requirement profile."""
        task_type = self.classify(task)
        profile_fn = TASK_PROFILES.get(task_type.value)
        if profile_fn:
            return profile_fn()

        # Default profile for unknown tasks
        return TaskRequirementProfile(task_type="unknown")

    def classify_with_confidence(self, task: str) -> tuple[TaskType, float]:
        """Classify with a confidence score (0.0-1.0).

        Confidence is based on how many patterns matched relative to
        the total possible matches.
        """
        scores: dict[TaskType, int] = {}

        for task_type, compiled_patterns in self._patterns.items():
            score = 0
            for pattern in compiled_patterns:
                matches = pattern.findall(task)
                score += len(matches)
            if score > 0:
                scores[task_type] = score

        if not scores:
            return TaskType.UNKNOWN, 0.0

        best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_score = scores[best_type]
        total_score = sum(scores.values())

        # Confidence: how dominant is the best match?
        confidence = best_score / total_score if total_score > 0 else 0.0
        # Cap at reasonable levels
        confidence = min(1.0, confidence * 1.2)

        return best_type, confidence
