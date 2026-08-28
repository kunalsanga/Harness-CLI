"""Debugging benchmark tasks."""

from harness_core.benchmarks.types import BenchmarkCategory, BenchmarkTask

DEBUGGING_TASKS = [
    BenchmarkTask(
        name="fix_broken_function",
        category=BenchmarkCategory.BUG_FIXING,
        description="Fix the bug in the divide function",
        setup_files={
            "math_utils.py": "def divide(a, b):\n    return a / b  # Bug: no zero division check\n",
            "test_math_utils.py": "from math_utils import divide\n\ndef test_divide_normal():\n    assert divide(10, 2) == 5.0\n\ndef test_divide_by_zero():\    result = divide(10, 0)\n    assert result is None  # Should handle gracefully\n",
        },
        instructions="The divide function crashes when dividing by zero. Fix it so it returns None for zero division. Make sure the tests pass.",
        expected_files=["math_utils.py"],
        expected_tests_pass=True,
        timeout_seconds=90,
        max_iterations=8,
    ),
    BenchmarkTask(
        name="fix_failing_test",
        category=BenchmarkCategory.BUG_FIXING,
        description="Fix the code so the test passes",
        setup_files={
            "greeting.py": "def greet(name):\n    return f'Hello {name}'  # Missing exclamation mark\n",
            "test_greeting.py": "from greeting import greet\n\ndef test_greet_world():\n    assert greet('World') == 'Hello, World!'\n",
        },
        instructions="The test test_greet_world is failing. Fix the greet function so the test passes.",
        expected_files=["greeting.py"],
        expected_tests_pass=True,
        timeout_seconds=60,
        max_iterations=6,
    ),
]
