"""Repository navigation benchmark tasks."""

from harness_core.benchmarks.types import BenchmarkCategory, BenchmarkTask

NAVIGATION_TASKS = [
    BenchmarkTask(
        name="find_function_definition",
        category=BenchmarkCategory.REPOSITORY_NAVIGATION,
        description="Find where the 'calculate_total' function is defined",
        setup_files={
            "src/calculator.py": "def calculate_total(items):\n    return sum(item.price for item in items)\n\ndef calculate_tax(amount, rate=0.1):\n    return amount * rate\n",
            "src/models.py": "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price\n",
            "tests/test_calculator.py": "from src.calculator import calculate_total\n",
        },
        instructions="Find the file and line where 'calculate_total' is defined. Report the file path and function signature.",
        timeout_seconds=60,
        max_iterations=5,
    ),
    BenchmarkTask(
        name="find_class_usage",
        category=BenchmarkCategory.REPOSITORY_NAVIGATION,
        description="Find all files that import or use the 'Item' class",
        setup_files={
            "src/models.py": "class Item:\n    def __init__(self, name, price):\n        self.name = name\n        self.price = price\n",
            "src/store.py": "from src.models import Item\n\ndef create_item():\n    return Item('Widget', 9.99)\n",
            "src/api.py": "from src.models import Item\nfrom src.store import create_item\n",
            "tests/test_models.py": "from src.models import Item\n",
        },
        instructions="Find all files that import or reference the 'Item' class. List each file path.",
        timeout_seconds=60,
        max_iterations=5,
    ),
]
