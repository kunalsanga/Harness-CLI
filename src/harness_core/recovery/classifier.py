"""
Failure classification for autonomous recovery.
"""

import enum
from dataclasses import dataclass, field
from typing import Any, Optional

from harness_core.agents.domain import AgentResult, AgentStatus, SubTask


class FailureCategory(enum.Enum):
    """Categories of task failures."""
    TEST_FAILURE = "test_failure"
    BUILD_FAILURE = "build_failure"
    TYPE_ERROR = "type_error"
    LINT_FAILURE = "lint_failure"
    RUNTIME_ERROR = "runtime_error"
    TOOL_FAILURE = "tool_failure"
    MODEL_FAILURE = "model_failure"
    TIMEOUT = "timeout"
    PERMISSION_FAILURE = "permission_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    ENVIRONMENT_FAILURE = "environment_failure"
    UNKNOWN = "unknown"


@dataclass
class FailureClassification:
    """Structured evidence of a failure."""
    category: FailureCategory
    summary: str
    evidence: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    failed_tests: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category.value,
            "summary": self.summary,
            "evidence": self.evidence,
            "affected_files": self.affected_files,
            "failed_tests": self.failed_tests,
            "confidence": self.confidence,
        }


class FailureClassifier:
    """Analyzes task output to classify failures."""

    @staticmethod
    def classify(task: SubTask, result: AgentResult) -> FailureClassification | None:
        """Classify a failure based on task and result.
        Returns None if there is no failure.
        """
        # If agent crashed, timed out, or explicitly failed
        if result.status == AgentStatus.FAILED:
            errors_str = " ".join(result.errors).lower()
            
            if "timeout" in errors_str:
                return FailureClassification(
                    category=FailureCategory.TIMEOUT,
                    summary="Agent timed out.",
                    evidence=result.errors
                )
                
            if "permission" in errors_str or "unauthorized" in errors_str:
                return FailureClassification(
                    category=FailureCategory.PERMISSION_FAILURE,
                    summary="Agent lacked permissions.",
                    evidence=result.errors
                )

            return FailureClassification(
                category=FailureCategory.UNKNOWN,
                summary="Agent execution failed.",
                evidence=result.errors
            )

        # Agent succeeded, but maybe tests failed
        if result.tests_total > 0 and result.tests_passed < result.tests_total:
            # Extract failed tests from findings if available
            failed_tests = []
            for finding in result.findings:
                if finding.get("type") == "test_failure":
                    failed_tests.append(finding.get("test_name", "unknown"))

            # Determine if this was a build failure masked as test failure
            summary_lower = result.summary.lower()
            if "build" in summary_lower and "fail" in summary_lower:
                return FailureClassification(
                    category=FailureCategory.BUILD_FAILURE,
                    summary=f"Build failed, tests could not pass.",
                    evidence=[result.summary] + result.errors,
                    affected_files=result.files_changed,
                    confidence=0.9
                )

            return FailureClassification(
                category=FailureCategory.TEST_FAILURE,
                summary=f"{result.tests_total - result.tests_passed} tests failed.",
                evidence=[result.summary] + result.errors,
                affected_files=result.files_changed,
                failed_tests=failed_tests,
                confidence=0.95
            )

        # Look for explicit failure findings even if tests_total is 0
        for finding in result.findings:
            if finding.get("type") == "lint_failure":
                return FailureClassification(
                    category=FailureCategory.LINT_FAILURE,
                    summary="Linting failed.",
                    evidence=[finding.get("message", "")]
                )
            if finding.get("type") == "build_failure":
                return FailureClassification(
                    category=FailureCategory.BUILD_FAILURE,
                    summary="Build failed.",
                    evidence=[finding.get("message", "")]
                )

        # Look for errors in the result that didn't trigger AgentStatus.FAILED
        if result.errors:
            err_str = " ".join(result.errors).lower()
            if "build" in err_str:
                cat = FailureCategory.BUILD_FAILURE
            elif "type" in err_str:
                cat = FailureCategory.TYPE_ERROR
            elif "permission" in err_str:
                cat = FailureCategory.PERMISSION_FAILURE
            elif "model" in err_str:
                cat = FailureCategory.MODEL_FAILURE
            else:
                cat = FailureCategory.UNKNOWN
                
            return FailureClassification(
                category=cat,
                summary="Errors reported during execution.",
                evidence=result.errors
            )

        # No failure detected
        return None
