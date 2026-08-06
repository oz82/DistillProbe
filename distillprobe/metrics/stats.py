from __future__ import annotations

from typing import NamedTuple

import numpy as np


class CIResult(NamedTuple):
    mean: float
    low: float
    high: float
    n: int


def bootstrap_ci(
    values, n: int = 10_000, alpha: float = 0.05, seed: int = 42, statistic=np.mean
) -> CIResult:
    """Bootstrap confidence interval for a 1-D sample.

    Default statistic is the sample mean. Returns the (1 - alpha) percentile CI.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("bootstrap_ci expects a 1-D sample")
    if arr.size == 0:
        return CIResult(mean=float("nan"), low=float("nan"), high=float("nan"), n=0)
    if arr.size == 1:
        v = float(arr[0])
        return CIResult(mean=v, low=v, high=v, n=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    boot = statistic(arr[idx], axis=1)
    low = float(np.quantile(boot, alpha / 2))
    high = float(np.quantile(boot, 1 - alpha / 2))
    return CIResult(mean=float(statistic(arr)), low=low, high=high, n=int(arr.size))


def paired_wilcoxon(x, y):
    """Two-sided paired Wilcoxon signed-rank test. Returns scipy WilcoxonResult."""
    from scipy.stats import wilcoxon

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    return wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")


def mannwhitneyu(x, y, alternative: str = "two-sided"):
    """Independent-samples Mann–Whitney U test. Returns scipy result object."""
    from scipy.stats import mannwhitneyu as _mw

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return _mw(x, y, alternative=alternative)


def cliffs_delta(x, y) -> float:
    """Cliff's delta non-parametric effect size in [-1, 1].

    delta > 0 means x tends to be larger than y.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = int((x[:, None] > y[None, :]).sum())
    less = int((x[:, None] < y[None, :]).sum())
    return float((greater - less) / (x.size * y.size))


# ---------------------------------------------------------------------------
# Multiple-comparison correction (PLOS ONE statistical-rigor requirement).
# Dependency-free implementations of Holm-Bonferroni (FWER) and
# Benjamini-Hochberg (FDR). Both return adjusted p-values in input order.
# ---------------------------------------------------------------------------

def holm_bonferroni(pvals) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (controls FWER).

    Returns an array of adjusted p-values aligned with the input order.
    """
    p = np.asarray(list(pvals), dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)  # enforce monotone non-decreasing
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values / q-values (controls FDR).

    Returns an array of adjusted p-values aligned with the input order.
    """
    p = np.asarray(list(pvals), dtype=np.float64)
    m = p.size
    if m == 0:
        return p
    order = np.argsort(p)
    adj_sorted = np.empty(m, dtype=np.float64)
    running = 1.0
    # step-up: iterate from the largest p-value down
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * m / (rank + 1)
        running = min(running, val)  # enforce monotone non-increasing going down
        adj_sorted[rank] = min(running, 1.0)
    adj = np.empty(m, dtype=np.float64)
    adj[order] = adj_sorted
    return adj


# ---------------------------------------------------------------------------
# Binomial proportion confidence intervals (for hit-rate / success-count
# metrics such as negation polarity flips and retrieval P@1).
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> CIResult:
    """Wilson score confidence interval for a binomial proportion k/n.

    Preferred over the normal-approximation interval for small n or
    extreme proportions (e.g. 0/98 negation flips).
    """
    from scipy.stats import norm

    if n == 0:
        return CIResult(mean=float("nan"), low=float("nan"), high=float("nan"), n=0)
    z = float(norm.ppf(1 - alpha / 2))
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return CIResult(mean=float(phat), low=float(max(0.0, center - half)),
                    high=float(min(1.0, center + half)), n=int(n))


def rule_of_three(n: int, conf: float = 0.95) -> float:
    """Upper bound on a proportion when zero successes are observed in n trials.

    For the standard 95% case this is the classic 3/n rule; for general
    confidence it is -ln(1-conf)/n.
    """
    if n <= 0:
        return float("nan")
    return float(-np.log(1.0 - conf) / n)


# ---------------------------------------------------------------------------
# Permutation test (for the typology x distillation interaction, where the
# unit of analysis is the per-language gap and n is small / non-normal).
# ---------------------------------------------------------------------------

def permutation_test(x, y, n: int = 10_000, seed: int = 42,
                     statistic=None, alternative: str = "two-sided"):
    """Two-sample permutation test on a difference statistic.

    Default statistic is the difference in means (mean(x) - mean(y)).
    Returns a NamedTuple-like dict with observed statistic and p-value.
    Suitable for small samples where parametric assumptions fail.
    """
    x = np.asarray(list(x), dtype=np.float64)
    y = np.asarray(list(y), dtype=np.float64)
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a) - np.mean(b))
    observed = statistic(x, y)
    pooled = np.concatenate([x, y])
    nx = x.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        stat = statistic(pooled[:nx], pooled[nx:])
        if alternative == "two-sided":
            if abs(stat) >= abs(observed) - 1e-12:
                count += 1
        elif alternative == "greater":
            if stat >= observed - 1e-12:
                count += 1
        else:  # "less"
            if stat <= observed + 1e-12:
                count += 1
    # add-one correction so p is never exactly 0
    pval = (count + 1) / (n + 1)
    return {"observed": float(observed), "pvalue": float(pval), "n_perm": int(n),
            "alternative": alternative}
