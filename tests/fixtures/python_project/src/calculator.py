"""Calculator module with intentional bugs for testing."""


def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b. Raises ValueError if b is zero."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b  # noqa: RUF100


def power(a: float, b: float) -> float:
    """Raise a to the power of b."""
    return a ** b


def factorial(n: int) -> int:
    """Calculate factorial. BUG: does not handle negative numbers."""
    if n == 0:
        return 1
    return n * factorial(n - 1)
