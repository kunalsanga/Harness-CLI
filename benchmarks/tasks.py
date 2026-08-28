"""Benchmark task definitions for Harness coding evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskCategory(str, Enum):
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    TESTING = "testing"
    DEBUGGING = "debugging"
    SECURITY = "security"


class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TaskLanguage(str, Enum):
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    RUST = "rust"


@dataclass
class BenchmarkTask:
    """A single benchmark task."""

    task_id: str
    description: str
    fixture_path: str  # relative to benchmark-fixtures/
    category: TaskCategory
    difficulty: TaskDifficulty
    language: TaskLanguage
    verification_command: str  # command to verify success
    expected_files_modified: list[str] = field(default_factory=list)
    setup_command: str = ""  # pre-task setup
    pre_test_command: str = ""  # test to run BEFORE task (should fail)
    post_test_command: str = ""  # test to run AFTER task (should pass)

    def get_fixture_dir(self) -> Path:
        """Get absolute path to fixture directory."""
        return Path(__file__).parent.parent / "benchmark-fixtures" / self.fixture_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "fixture": self.fixture_path,
            "category": self.category.value,
            "difficulty": self.difficulty.value,
            "language": self.language.value,
            "verification": self.verification_command,
        }


# === PYTHON FIXTURE TASKS ===

PYTHON_TASKS = [
    BenchmarkTask(
        task_id="PY-BUG-001",
        description="Fix the failing test in test_calculator.py. The add function returns incorrect results for negative numbers.",
        fixture_path="python-app",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/ -q",
        expected_files_modified=["src/calculator.py"],
        pre_test_command="cd {fixture} && python -m pytest tests/ -q",
    ),
    BenchmarkTask(
        task_id="PY-BUG-002",
        description="Fix the broken string utility functions. The capitalize_words function fails on empty strings and the truncate function doesn't handle strings shorter than the limit.",
        fixture_path="python-app",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.MEDIUM,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/test_string_utils.py -q",
        expected_files_modified=["src/string_utils.py"],
        pre_test_command="cd {fixture} && python -m pytest tests/test_string_utils.py -q",
    ),
    BenchmarkTask(
        task_id="PY-FEAT-001",
        description="Add a multiply function to the Calculator class and a factorial function. Both need tests.",
        fixture_path="python-app",
        category=TaskCategory.FEATURE,
        difficulty=TaskDifficulty.EASY,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/test_calculator.py -q",
        expected_files_modified=["src/calculator.py", "tests/test_calculator.py"],
    ),
    BenchmarkTask(
        task_id="PY-REFACTOR-001",
        description="Refactor the duplicated validation logic in user_service.py and order_service.py into a shared validators module. All existing tests must pass.",
        fixture_path="python-app",
        category=TaskCategory.REFACTOR,
        difficulty=TaskDifficulty.MEDIUM,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/ -q",
        expected_files_modified=["src/validators.py", "src/user_service.py", "src/order_service.py"],
    ),
    BenchmarkTask(
        task_id="PY-TEST-001",
        description="Add missing test cases for the DataStore class. The class has get, set, delete, and list methods but only basic tests exist. Add edge case tests for empty keys, non-existent keys, and concurrent-like operations.",
        fixture_path="python-app",
        category=TaskCategory.TESTING,
        difficulty=TaskDifficulty.MEDIUM,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/test_datastore.py -v",
        expected_files_modified=["tests/test_datastore.py"],
    ),
    BenchmarkTask(
        task_id="PY-DEBUG-001",
        description="The API server has a race condition in the rate limiter. When two requests arrive simultaneously, the counter can exceed the limit. Find and fix the issue.",
        fixture_path="python-app",
        category=TaskCategory.DEBUGGING,
        difficulty=TaskDifficulty.HARD,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/test_rate_limiter.py -q",
        expected_files_modified=["src/rate_limiter.py"],
        pre_test_command="cd {fixture} && python -m pytest tests/test_rate_limiter.py -q",
    ),
    BenchmarkTask(
        task_id="PY-SEC-001",
        description="The input validation in the API handlers is vulnerable to SQL injection through the search endpoint. Fix the validation to properly sanitize user input.",
        fixture_path="python-app",
        category=TaskCategory.SECURITY,
        difficulty=TaskDifficulty.MEDIUM,
        language=TaskLanguage.PYTHON,
        verification_command="cd {fixture} && python -m pytest tests/test_validation.py -q",
        expected_files_modified=["src/validators.py"],
        pre_test_command="cd {fixture} && python -m pytest tests/test_validation.py -q",
    ),
]

# === NODE/TYPESCRIPT FIXTURE TASKS ===

NODE_TASKS = [
    BenchmarkTask(
        task_id="NODE-BUG-001",
        description="Fix the failing test in math.test.ts. The fibonacci function returns wrong values for n=0 and n=1.",
        fixture_path="node-app",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.EASY,
        language=TaskLanguage.TYPESCRIPT,
        verification_command="cd {fixture} && npm test",
        expected_files_modified=["src/math.ts"],
        pre_test_command="cd {fixture} && npm test",
    ),
    BenchmarkTask(
        task_id="NODE-FEAT-001",
        description="Add a binarySearch function to the math module and write tests for it.",
        fixture_path="node-app",
        category=TaskCategory.FEATURE,
        difficulty=TaskDifficulty.EASY,
        language=TaskLanguage.TYPESCRIPT,
        verification_command="cd {fixture} && npm test",
        expected_files_modified=["src/math.ts", "tests/math.test.ts"],
    ),
    BenchmarkTask(
        task_id="NODE-BUG-002",
        description="Fix the URL parser. It incorrectly handles URLs with ports and URLs with authentication strings. Tests are failing.",
        fixture_path="node-app",
        category=TaskCategory.BUG_FIX,
        difficulty=TaskDifficulty.MEDIUM,
        language=TaskLanguage.TYPESCRIPT,
        verification_command="cd {fixture} && npm test",
        expected_files_modified=["src/url-utils.ts"],
        pre_test_command="cd {fixture} && npm test",
    ),
]

# === ALL TASKS ===

ALL_TASKS = PYTHON_TASKS + NODE_TASKS
