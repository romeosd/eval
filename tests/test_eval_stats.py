import pytest

from adjudicator.evaluation.stats import paired, sign_test_p, single_case_swing, wilson


def test_wilson_interval_is_wide_for_small_samples():
    perfect = wilson(8, 8)
    assert perfect.rate == 1.0 and perfect.high == 1.0
    assert perfect.low == pytest.approx(0.676, abs=0.002)  # 8/8 does not prove reliability
    none = wilson(0, 8)
    assert none.low == 0.0 and none.high == pytest.approx(0.324, abs=0.002)


def test_wilson_handles_empty_input():
    empty = wilson(0, 0)
    assert empty.rate is None and empty.text() == "n/a"


def test_interval_text_shows_counts_and_bounds():
    assert wilson(4, 5).text() == "80% (4/5; 95% CI 38%-96%)"


@pytest.mark.parametrize(
    ("only_a", "only_b", "expected"),
    [(3, 0, 0.25), (5, 1, 0.21875), (0, 0, 1.0), (2, 2, 1.0), (10, 0, 2 / 1024)],
)
def test_exact_sign_test(only_a, only_b, expected):
    assert sign_test_p(only_a, only_b) == pytest.approx(expected)


def test_paired_counts_and_requires_the_same_cases():
    a = {"c1": True, "c2": True, "c3": False, "c4": False}
    b = {"c1": True, "c2": False, "c3": True, "c4": False}
    result = paired(a, b)
    assert (result.both, result.only_a, result.only_b, result.neither) == (1, 1, 1, 1)
    assert result.p_value == 1.0
    with pytest.raises(ValueError, match="same cases"):
        paired(a, {"c1": True})


def test_single_case_swing():
    assert single_case_swing(8) == 0.125 and single_case_swing(0) is None
