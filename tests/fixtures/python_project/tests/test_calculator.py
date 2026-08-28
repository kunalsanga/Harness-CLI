"""Tests for calculator — contains intentional failures."""

import pytest
from src.calculator import add, subtract, multiply, divide, power, factorial


class TestAdd:
    def test_add_positive(self):
        assert add(2, 3) == 5

    def test_add_negative(self):
        assert add(-1, -2) == -3

    def test_add_zero(self):
        assert add(0, 0) == 0


class TestSubtract:
    def test_subtract_basic(self):
        assert subtract(5, 3) == 2

    def test_subtract_negative_result(self):
        assert subtract(3, 5) == -2


class TestMultiply:
    def test_multiply_basic(self):
        assert multiply(3, 4) == 12

    def test_multiply_by_zero(self):
        assert multiply(5, 0) == 0


class TestDivide:
    def test_divide_basic(self):
        assert divide(10, 2) == 5.0

    def test_divide_by_zero(self):
        """This test should pass after the agent fixes divide() to handle ZeroDivisionError."""
        with pytest.raises(ZeroDivisionError):
            divide(10, 0)

    def test_divide_negative(self):
        assert divide(-10, 2) == -5.0


class TestPower:
    def test_power_basic(self):
        assert power(2, 3) == 8

    def test_power_zero(self):
        assert power(5, 0) == 1


class TestFactorial:
    def test_factorial_zero(self):
        assert factorial(0) == 1

    def test_factorial_positive(self):
        assert factorial(5) == 120

    def test_factorial_negative(self):
        """This test should pass after the agent fixes factorial() to handle negatives."""
        with pytest.raises(ValueError):
            factorial(-1)
