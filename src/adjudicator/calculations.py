"""Deterministic arithmetic with unit algebra, exposed to the model as a tool.

Policy-specific values (limits, waiting periods) are *read from the policy* by the model
and passed in with a unit; this module only guarantees the arithmetic and refuses
dimensionally meaningless combinations (hours plus dollars). Every result carries a unit
and an id so validation can confirm amounts were computed rather than asserted.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from enum import StrEnum
from typing import Literal

type Unit = Literal["usd", "hours", "days", "weeks", "months", "years", "percent", "count", "other"]
type Operand = tuple[float, Unit]

UNITS: tuple[str, ...] = (
    "usd",
    "hours",
    "days",
    "weeks",
    "months",
    "years",
    "percent",
    "count",
    "other",
)


class CalcOperation(StrEnum):
    DAYS_BETWEEN = "days_between"  # datetimes[1] - datetimes[0], in days
    HOURS_BETWEEN = "hours_between"  # datetimes[1] - datetimes[0], in hours
    SUM = "sum"
    SUBTRACT = "subtract"  # operands[0] - operands[1]
    MULTIPLY = "multiply"  # at most one operand may carry a unit other than "count"
    MIN = "min"
    MAX = "max"
    FLOOR_DIVIDE = "floor_divide"  # floor(a / b); both operands share a unit; result is a count


class CalculationError(ValueError):
    pass


def evaluate(
    operation: CalcOperation, operands: list[Operand], datetimes: list[str]
) -> tuple[float, Unit]:
    """Return ``(result, unit)``; raise ``CalculationError`` for invalid or mixed-unit input."""
    values = [value for value, _ in operands]
    match operation:
        case CalcOperation.DAYS_BETWEEN | CalcOperation.HOURS_BETWEEN:
            start, end = _two(datetimes, operation)
            delta = _parse_moment(end) - _parse_moment(start)
            if operation is CalcOperation.DAYS_BETWEEN:
                return round(delta.total_seconds() / 86_400, 4), "days"
            return round(delta.total_seconds() / 3_600, 4), "hours"
        case CalcOperation.SUM:
            unit = _shared_unit(operands, 1, operation)
            return round(math.fsum(values), 2), unit
        case CalcOperation.SUBTRACT:
            _two(operands, operation)
            unit = _shared_unit(operands, 2, operation)
            return round(values[0] - values[1], 2), unit
        case CalcOperation.MULTIPLY:
            _at_least(operands, 2, operation)
            carried = {unit for _, unit in operands if unit != "count"}
            if len(carried) > 1:
                raise CalculationError(f"multiply cannot combine units {sorted(carried)}")
            unit = next(iter(carried), "count")
            return round(math.prod(values), 2), unit  # pyright: ignore[reportReturnType]
        case CalcOperation.MIN:
            unit = _shared_unit(operands, 1, operation)
            return min(values), unit
        case CalcOperation.MAX:
            unit = _shared_unit(operands, 1, operation)
            return max(values), unit
        case CalcOperation.FLOOR_DIVIDE:
            _two(operands, operation)
            _shared_unit(operands, 2, operation)
            if values[1] == 0:
                raise CalculationError("floor_divide by zero")
            return float(math.floor(values[0] / values[1])), "count"


def _parse_moment(value: str) -> datetime:
    try:
        if len(value) == 10:
            return datetime.combine(date.fromisoformat(value), datetime.min.time())
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise CalculationError(f"not an ISO-8601 date/datetime: {value!r}") from exc


def _shared_unit(operands: list[Operand], minimum: int, operation: CalcOperation) -> Unit:
    _at_least(operands, minimum, operation)
    units = {unit for _, unit in operands}
    if len(units) != 1:
        raise CalculationError(f"{operation.value} needs one shared unit, got {sorted(units)}")
    return operands[0][1]


def _two[T](values: list[T], operation: CalcOperation) -> tuple[T, T]:
    if len(values) != 2:
        raise CalculationError(f"{operation.value} needs exactly 2 values, got {len(values)}")
    return values[0], values[1]


def _at_least[T](values: list[T], n: int, operation: CalcOperation) -> list[T]:
    if len(values) < n:
        raise CalculationError(f"{operation.value} needs at least {n} values, got {len(values)}")
    return values
