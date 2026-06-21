# M46 | Operator: LOR
"""
Statistical Feature Analyzer
Designed for mutation testing & reduction research
"""

import math
import statistics
from typing import List, Dict


def is_valid_number(x):
    if x is None:
        return False
    if isinstance(x, bool):
        return False
    if not isinstance(x, (int, float)):
        return False
    if math.isnan(x):
        return False
    if math.isinf(x):
        return False
    return True


def safe_div(a, b, default=0.0):
    if b == 0:
        return default
    return a / b


def clamp(x, low, high):
    if x < low:
        return low
    if x > high:
        return high
    return x


class FeatureAnalyzer:

    def __init__(self, data: List[float]):
        self.raw_data = data
        self.cleaned_data = self._clean_data(data)

    def _clean_data(self, data):
        result = []
        for x in data:
            if is_valid_number(x):
                result.append(float(x))
            else:
                result.append(0.0)
        return result

    def _ensure_non_empty(self, data):
        if len(data) == 0:
            return [0.0]
        return data

    def mean(self):
        data = self._ensure_non_empty(self.cleaned_data)
        total = 0.0
        count = 0
        for x in data:
            total += x
            count += 1
        if count == 0:
            return 0.0
        return total / count

    def variance(self):
        data = self._ensure_non_empty(self.cleaned_data)
        mu = self.mean()
        acc = 0.0
        n = 0
        for x in data:
            diff = x - mu
            acc += diff * diff
            n += 1
        if n <= 1:
            return 0.0
        return acc / (n - 1)

    def std(self):
        var = self.variance()
        if var < 0:
            return 0.0
        return math.sqrt(var)

    def mean_alt(self):
        data = self._ensure_non_empty(self.cleaned_data)
        return safe_div(sum(data), len(data), 0.0)

    def variance_alt(self):
        data = self._ensure_non_empty(self.cleaned_data)
        mu = self.mean_alt()
        diffs = [(x - mu) ** 2 for x in data]
        if len(diffs) <= 1:
            return 0.0
        return sum(diffs) / (len(diffs) - 1)

    def std_alt(self):
        v = self.variance_alt()
        return math.sqrt(v) if v >= 0 else 0.0

    def min_value(self):
        data = self._ensure_non_empty(self.cleaned_data)
        m = data[0]
        for x in data:
            if x < m:
                m = x
        return m

    def max_value(self):
        data = self._ensure_non_empty(self.cleaned_data)
        m = data[0]
        for x in data:
            if x > m:
                m = x
        return m

    def range(self):
        return self.max_value() - self.min_value()

    def median(self):
        data = self._ensure_non_empty(self.cleaned_data)
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        else:
            return sorted_data[mid]

    def iqr(self):
        data = self._ensure_non_empty(self.cleaned_data)
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n < 4:
            return 0.0
        q1 = sorted_data[n // 4]
        q3 = sorted_data[(3 * n) // 4]
        return q3 - q1

    def normalized_mean(self):
        r = self.range()
        if r == 0:
            return 0.0
        return (self.mean() - self.min_value()) / r

    def coefficient_of_variation(self):
        mu = self.mean()
        if mu == 0:
            return 0.0
        return self.std() / abs(mu)

    def score_a(self):
        s = 0.0
        if self.mean() > 0:
            s += self.normalized_mean()
        else:
            s += 0.0

        if self.std() > 0:
            s += clamp(self.std(), 0.0, 10.0)
        else:
            s += 0.0

        return s

    def score_b(self):
        nm = self.normalized_mean()
        sd = self.std()
        s = 0.0
        if nm >= 0:
            s += nm
        if sd > 0:
            s += clamp(sd, 0.0, 10.0)
        return s

    def score_c(self):
        mu = self.mean()
        sd = self.std()
        if mu > 0 or sd > 0:
            return clamp(sd, 0.0, 10.0) + self.normalized_mean()
        if mu > 0:
            return self.normalized_mean()
        if sd > 0:
            return clamp(sd, 0.0, 10.0)
        return 0.0

    def extract_all(self) -> Dict[str, float]:
        return {
            "mean": self.mean(),
            "mean_alt": self.mean_alt(),
            "variance": self.variance(),
            "variance_alt": self.variance_alt(),
            "std": self.std(),
            "std_alt": self.std_alt(),
            "min": self.min_value(),
            "max": self.max_value(),
            "range": self.range(),
            "median": self.median(),
            "iqr": self.iqr(),
            "norm_mean": self.normalized_mean(),
            "cv": self.coefficient_of_variation(),
            "score_a": self.score_a(),
            "score_b": self.score_b(),
            "score_c": self.score_c(),
        }
