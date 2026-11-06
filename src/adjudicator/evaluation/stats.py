"""Small-sample statistics for evaluation reports.

With a handful of labelled cases a single case moves accuracy by several points, so every
headline rate is reported with a Wilson score interval and every comparison between two
configurations with an exact sign test. Nothing here is clever on purpose: the point is to
stop small differences being presented as findings.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

Z_95 = 1.959964


class Interval(BaseModel):
    """A proportion with its 95% Wilson score interval."""

    model_config = ConfigDict(extra="forbid")
    k: int
    n: int
    rate: float | None
    low: float | None
    high: float | None

    def text(self) -> str:
        if self.rate is None or self.low is None or self.high is None:
            return "n/a"
        return f"{self.rate:.0%} ({self.k}/{self.n}; 95% CI {self.low:.0%}-{self.high:.0%})"


def wilson(k: int, n: int, z: float = Z_95) -> Interval:
    if n <= 0:
        return Interval(k=k, n=n, rate=None, low=None, high=None)
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return Interval(k=k, n=n, rate=p, low=max(0.0, centre - margin), high=min(1.0, centre + margin))


def sign_test_p(only_a: int, only_b: int) -> float:
    """Exact two-sided sign test on discordant pairs (cases where exactly one config is right)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


class Paired(BaseModel):
    model_config = ConfigDict(extra="forbid")
    both: int
    only_a: int
    only_b: int
    neither: int
    p_value: float


def paired(correct_a: dict[str, bool], correct_b: dict[str, bool]) -> Paired:
    """Compare two configurations on the same cases (keys must match)."""
    if correct_a.keys() != correct_b.keys():
        raise ValueError("paired comparison needs the same cases on both sides")
    only_a = sum(correct_a[c] and not correct_b[c] for c in correct_a)
    only_b = sum(correct_b[c] and not correct_a[c] for c in correct_a)
    both = sum(correct_a[c] and correct_b[c] for c in correct_a)
    return Paired(
        both=both,
        only_a=only_a,
        only_b=only_b,
        neither=len(correct_a) - both - only_a - only_b,
        p_value=sign_test_p(only_a, only_b),
    )


def single_case_swing(n: int) -> float | None:
    """How far accuracy moves when one case flips."""
    return 1 / n if n else None
