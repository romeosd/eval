import pytest

from adjudicator.calculations import CalcOperation, CalculationError, evaluate


def test_hours_between_datetimes_returns_hours():
    result = evaluate(CalcOperation.HOURS_BETWEEN, [], ["2026-07-02T08:00", "2026-07-03T04:00"])
    assert result == (20, "hours")


def test_days_between_dates_returns_days():
    assert evaluate(CalcOperation.DAYS_BETWEEN, [], ["2026-06-01", "2026-06-06"]) == (5, "days")


def test_floor_divide_of_matching_units_is_a_count():
    assert evaluate(CalcOperation.FLOOR_DIVIDE, [(20, "hours"), (12, "hours")], []) == (1, "count")


def test_min_keeps_the_shared_unit():
    assert evaluate(CalcOperation.MIN, [(1200, "usd"), (500, "usd")], []) == (500, "usd")


def test_multiply_a_rate_by_a_count_keeps_the_rate_unit():
    assert evaluate(CalcOperation.MULTIPLY, [(150, "usd"), (3, "count")], []) == (450, "usd")


@pytest.mark.parametrize(
    ("operation", "operands", "datetimes"),
    [
        (CalcOperation.SUM, [(5000, "usd"), (24, "hours")], []),  # dollars plus hours
        (CalcOperation.MULTIPLY, [(150, "usd"), (12, "hours")], []),
        (CalcOperation.FLOOR_DIVIDE, [(20, "hours"), (12, "usd")], []),
        (CalcOperation.SUBTRACT, [(1, "usd")], []),
        (CalcOperation.FLOOR_DIVIDE, [(1, "usd"), (0, "usd")], []),
        (CalcOperation.DAYS_BETWEEN, [], ["not-a-date", "2026-01-01"]),
        (CalcOperation.SUM, [], []),
    ],
)
def test_invalid_or_mixed_unit_input_raises(operation, operands, datetimes):
    with pytest.raises(CalculationError):
        evaluate(operation, operands, datetimes)
