"""Code generation benchmark tasks."""

from harness_core.benchmarks.types import BenchmarkCategory, BenchmarkTask

CODING_TASKS = [
    BenchmarkTask(
        name="implement_function",
        category=BenchmarkCategory.CODE_GENERATION,
        description="Implement a function that reverses a string",
        setup_files={},
        instructions="Create a file called 'string_utils.py' with a function called 'reverse_string' that takes a string and returns it reversed. Also create 'test_string_utils.py' with at least 3 test cases.",
        expected_files=["string_utils.py", "test_string_utils.py"],
        timeout_seconds=60,
        max_iterations=5,
    ),
    BenchmarkTask(
        name="implement_class",
        category=BenchmarkCategory.CODE_GENERATION,
        description="Implement a Stack class with push, pop, peek, and is_empty methods",
        setup_files={},
        instructions="Create a file called 'stack.py' with a Stack class that supports push(item), pop(), peek(), is_empty(), and size(). Create 'test_stack.py' with tests.",
        expected_files=["stack.py", "test_stack.py"],
        timeout_seconds=90,
        max_iterations=8,
    ),
]
