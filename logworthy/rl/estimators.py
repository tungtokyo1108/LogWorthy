"""Off-policy value estimators and their influence functions.

The audit certifies an *improvement*, so what matters is not the estimated value
of a policy but the uncertainty of the difference between that value and a
baseline. Getting the influence function right is therefore the whole job.

For a target policy pi evaluated against logged data, the doubly robust estimate
of the improvement over the logging policy is the mean of

    phi_i = q(s_i, pi(s_i)) + (1{a_i = pi(s_i)} / pi_b(a_i | s_i)) (y_i - q(s_i, a_i)) - y_i

and its standard error is the standard deviation of **phi**, not of the first two
terms. The ``- y_i`` belongs inside the influence function because the logged
outcome is itself estimated from the same rows; computing the standard error
without it treats ``mean(y)`` as a known constant and inflates the standard error
by a factor that was measured at 1.42 on one of the paper's corpora. The point
estimate is unaffected, which is exactly why the error is easy to miss.

The class :class:`ImprovementEstimate` therefore always carries the influence
function it was computed from, so the mistake cannot be repeated downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "ImprovementEstimate",
    "doubly_robust",
    "inverse_propensity",
    "direct_method",
    "paired_increment",
]


@dataclass(frozen=True)
class ImprovementEstimate:
    """An estimated improvement with the influence function behind it.

    Attributes:
        estimate: Mean improvement over the baseline, in outcome units.
        se: Standard error, computed from ``influence``.
        ci: Normal-approximation confidence interval.
        influence: Per-row influence function. Keep it: cluster-robust standard
            errors and paired comparisons are computed from this, not from the
            point estimate.
        n: Rows entering the estimate.
        estimator: Which estimator produced it.
    """

    estimate: float
    se: float
    ci: tuple
    influence: np.ndarray
    n: int
    estimator: str

    @property
    def excludes_zero(self) -> bool:
        """True when the interval lies entirely above zero.

        This is the per-seed certification event: the deployed learner showed an
        improvement that the data can distinguish from none.
        """
        return self.ci[0] > 0

    def cluster_se(self, clusters: np.ndarray) -> float:
        """Cluster-robust standard error of the same estimate.

        Rows within a cluster are dependent, so the independent-row standard
        error understates uncertainty. Use this whenever the log has a cluster
        structure, which is almost always.

        Args:
            clusters: Cluster label per row.

        Returns:
            The cluster-robust standard error.
        """
        codes = np.unique(np.asarray(clusters), return_inverse=True)[1]
        n_clusters = codes.max() + 1
        totals = np.bincount(codes, weights=self.influence, minlength=n_clusters)
        return float(np.sqrt((totals**2).sum()) / self.n)

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.estimator}: {self.estimate:+.5f} "
            f"[{self.ci[0]:+.5f},{self.ci[1]:+.5f}] (SE {self.se:.5f}, n={self.n:,})"
        )


def _finish(influence: np.ndarray, estimator: str, z: float) -> ImprovementEstimate:
    """Package an influence function into an estimate."""
    n = influence.size
    estimate = float(influence.mean())
    se = float(influence.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    return ImprovementEstimate(
        estimate=estimate,
        se=se,
        ci=(estimate - z * se, estimate + z * se),
        influence=influence,
        n=int(n),
        estimator=estimator,
    )


def doubly_robust(
    y: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    q_values: np.ndarray,
    target_actions: np.ndarray,
    *,
    baseline: Optional[np.ndarray] = None,
    z: float = 1.96,
    clip_propensity: float = 1e-6,
) -> ImprovementEstimate:
    """Doubly robust estimate of a policy's improvement over a baseline.

    Doubly robust means consistent if *either* the outcome model ``q_values`` or
    the propensity model is correct. Neither needs to be right, which is why it
    is the estimator the audit certifies on.

    Args:
        y: Logged outcome, shape ``(n,)``.
        actions: Logged action index, shape ``(n,)``.
        propensities: ``pi_b(a_i | s_i)`` for the action actually taken, shape
            ``(n,)``. Must be estimated on data disjoint from the evaluation
            fold, or the certificate is not honest.
        q_values: Outcome model predictions, shape ``(n, K)``.
        target_actions: Action the target policy takes per row, shape ``(n,)``.
        baseline: Per-row baseline value. Defaults to the logged outcome ``y``,
            giving the improvement over the logging policy. For the increment
            rule pass the fixed rule's doubly robust values instead, or use
            :func:`paired_increment`.
        z: Normal quantile for the interval.
        clip_propensity: Floor applied to propensities before inverting, to
            stop a single near-zero propensity from dominating.

    Returns:
        An :class:`ImprovementEstimate`.

    Raises:
        ValueError: If shapes disagree.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(actions)
    pi = np.asarray(propensities, dtype=float)
    q = np.asarray(q_values, dtype=float)
    tgt = np.asarray(target_actions)
    n = y.size
    if q.ndim != 2 or q.shape[0] != n:
        raise ValueError(f"q_values must have shape (n, K); got {q.shape} for n={n}")
    if not (a.size == pi.size == tgt.size == n):
        raise ValueError("y, actions, propensities and target_actions must agree in length")

    rows = np.arange(n)
    pi_safe = np.maximum(pi, clip_propensity)
    direct = q[rows, tgt]
    correction = (a == tgt) / pi_safe * (y - q[rows, a])
    base = y if baseline is None else np.asarray(baseline, dtype=float)
    # The baseline is subtracted inside the influence function: see module docstring.
    influence = direct + correction - base
    return _finish(influence, "doubly robust", z)


def inverse_propensity(
    y: np.ndarray,
    actions: np.ndarray,
    propensities: np.ndarray,
    target_actions: np.ndarray,
    *,
    baseline: Optional[np.ndarray] = None,
    z: float = 1.96,
    clip_propensity: float = 1e-6,
) -> ImprovementEstimate:
    """Inverse propensity score estimate of an improvement.

    Unbiased when the propensity model is correct, but high variance under poor
    overlap. Provided for comparison; the audit certifies on
    :func:`doubly_robust`.
    """
    y = np.asarray(y, dtype=float)
    a = np.asarray(actions)
    pi = np.maximum(np.asarray(propensities, dtype=float), clip_propensity)
    tgt = np.asarray(target_actions)
    base = y if baseline is None else np.asarray(baseline, dtype=float)
    influence = (a == tgt) / pi * y - base
    return _finish(influence, "inverse propensity", z)


def direct_method(
    q_values: np.ndarray,
    target_actions: np.ndarray,
    *,
    baseline: np.ndarray,
    z: float = 1.96,
) -> ImprovementEstimate:
    """Direct-method estimate: the outcome model's own prediction.

    Carries no correction term, so it inherits every bias of the outcome model.
    A learner that selects on its own fitting error will look good here; that is
    the failure mode the coupling test exists to catch.
    """
    q = np.asarray(q_values, dtype=float)
    tgt = np.asarray(target_actions)
    rows = np.arange(q.shape[0])
    influence = q[rows, tgt] - np.asarray(baseline, dtype=float)
    return _finish(influence, "direct method", z)


def paired_increment(
    contextual: ImprovementEstimate,
    fixed: ImprovementEstimate,
    *,
    z: float = 1.96,
    clusters: Optional[np.ndarray] = None,
) -> ImprovementEstimate:
    """The increment rule: what personalisation adds over the best fixed rule.

    Because the fixed-rule class is contained in the contextual class, certifying
    a contextual policy against the *logging* policy states nothing that the
    fixed-rule verdict does not already imply. The quantity that carries
    information is the paired difference, evaluated on the same rows so that the
    shared variation cancels:

        phi_increment = phi_contextual - phi_fixed

    This is why both estimates must carry their influence functions, and why
    they must come from the same evaluation fold.

    Args:
        contextual: Estimate for the contextual policy.
        fixed: Estimate for the certified fixed rule, same rows and order.
        z: Normal quantile for the interval.
        clusters: Optional cluster labels; when given, the standard error is
            cluster-robust.

    Returns:
        An :class:`ImprovementEstimate` for the increment.

    Raises:
        ValueError: If the two estimates do not come from the same rows.
    """
    if contextual.n != fixed.n:
        raise ValueError(
            "the paired increment needs both policies evaluated on the same rows; "
            f"got n={contextual.n} and n={fixed.n}"
        )
    influence = contextual.influence - fixed.influence
    est = _finish(influence, "paired increment (DR)", z)
    if clusters is None:
        return est
    se = est.cluster_se(clusters)
    return ImprovementEstimate(
        estimate=est.estimate,
        se=se,
        ci=(est.estimate - z * se, est.estimate + z * se),
        influence=influence,
        n=est.n,
        estimator="paired increment (DR, cluster-robust)",
    )
