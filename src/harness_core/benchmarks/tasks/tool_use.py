"""Tool use benchmark tasks."""

from harness_core.benchmarks.types import BenchmarkCategory, BenchmarkTask

TOOL_USE_TASKS = [
    BenchmarkTask(
        name="read_existing_file",
        category=BenchmarkCategory.TOOL_USE,
        description="Read the contents of main.py",
        setup_files={
            "main.py": "def greet(name):\n    return f'Hello, {name}!'\n\nprint(greet('World'))\n",
        },
        instructions="Use the read_file tool to read main.py and return its contents.",
        timeout_seconds=30,
        max_iterations=3,
    ),
    BenchmarkTask(
        name="write_new_file",
        category=BenchmarkCategory.TOOL_USE,
        description="Create a new file called utils.py with a helper function",
        setup_files={},
        instructions="Use the write_file tool to create utils.py with a function called 'add' that takes two numbers and returns their sum.",
        expected_files=["utils.py"],
        timeout_seconds=30,
        max_iterations=3,
    ),
    BenchmarkTask(
        name="list_directory",
        category=BenchmarkCategory.TOOL_USE,
        description="List files in the project directory",
        setup_files={
            "app.py": "print('app')\n",
            "config.py": "DEBUG = True\n",
            "README.md": "# Project\n",
        },
        instructions="Use the list_files tool to see what files exist in the current directory.",
        timeout_seconds=30,
        max_iterations=3,
    ),
]
