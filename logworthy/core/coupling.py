"""Action-outcome coupling: the weighted, bias-corrected estimator and its test.

The quantity audited throughout is the partial coupling of the action block,

    omega^2_{a|s} = (SS_0 - SS_1 - df1 * MSE_1) / (SS_0 + MSE_1),

the bias-corrected (Hays 1963) share of outcome variance that the action block
explains *beyond* the state model M0. Two properties of this estimator drive
every design decision below.

Weights.  Logs are observational, so rows carry importance weights
w = rho(a) / pi_b(a | s). Every sum of squares is weighted; an unweighted ANOVA
answers a different question (the logged mix, not the reference design rho).

Clusters.  Rows are not independent. Orders share a customer state, impressions
share a user stratum. All uncertainty is therefore computed by resampling
*clusters*, never rows. A row-level bootstrap on clustered data understates the
standard error by the square root of the design effect, which on the corpora in
the paper is a factor of 2-3. This module does not provide a row-level bootstrap,
because on the data this package is for, it would be wrong.

The bias correction can return a negative omega^2 when the observed between-action
variation is smaller than chance alone would produce. That is evidence *for* the
null, not a defect, and it is never clipped: see `partial_omega2`.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass

import numpy as np

__all__ = [
    "CouplingEstimate",
    "partial_omega2",
    "wild_cluster_bootstrap",
    "design_effect",
]


@dataclass(frozen=True)
class CouplingEstimate:
    """A coupling measurement together with the uncertainty that qualifies it.

    Attributes:
        omega2: Point estimate of the partial coupling of the action block.
        ci: Confidence interval for ``omega2``, from the cluster bootstrap.
        p_value: Bootstrap p-value against the decoupled null, or ``None`` if
            the interval was produced without a null distribution.
        df1: Degrees of freedom of the action block (the contrast audited).
        n: Rows entering the estimate.
        n_clusters: Independent clusters. This, not ``n``, sets the precision.
        n_eff: Kish effective sample size of the weights.
    """

    omega2: float
    ci: tuple[float, float]
    p_value: Optional[float]
    df1: int
    n: int
    n_clusters: int
    n_eff: float

    @property
    def lower(self) -> float:
        """Lower confidence limit, written ``L-hat`` in the paper."""
        return self.ci[0]

    @property
    def upper(self) -> float:
        """Upper confidence limit, written ``U-hat`` in the paper."""
        return self.ci[1]

    def __str__(self) -> str:  # pragma: no cover - display only
        p = "" if self.p_value is None else f" p={self.p_value:.4f}"
        return (
            f"omega2={self.omega2:.4e} "
            f"95% CI=[{self.ci[0]:.4e},{self.ci[1]:.4e}]{p} "
            f"(df1={self.df1}, G={self.n_clusters}, n_eff={self.n_eff:.0f})"
        )


def _wls_residuals(design: np.ndarray, y: np.ndarray, sqrt_w: np.ndarray):
    """Weighted least squares. Returns (residuals, coefficients)."""
    beta, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    return y - design @ beta, beta


def partial_omega2(
    m0: np.ndarray,
    m1: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    df1: int,
    *,
    clip: bool = True,
) -> float:
    """Bias-corrected partial coupling of the action block.

    Args:
        m0: State-only design matrix, shape (n, k0). Includes the intercept.
        m1: Full design matrix with the action block appended, shape (n, k1).
        y: Outcome.
        weights: Importance weights ``rho(a) / pi_b(a | s)``.
        df1: Degrees of freedom of the action block, ``k1 - k0``.
        clip: If True (the default) return ``max(estimate, 0)``, which is what
            the audit's interval machinery expects, since a negative point
            estimate and a zero one lead to the same verdict. Pass False to see
            the raw bias-corrected value, which may be negative and is then
            positive evidence of no coupling.

    Returns:
        The partial omega^2 of the action block.
    """
    n = len(y)
    sqrt_w = np.sqrt(weights)
    resid1, _ = _wls_residuals(m1, y, sqrt_w)
    resid0, _ = _wls_residuals(m0, y, sqrt_w)
    ss1 = float((weights * resid1**2).sum())
    ss0 = float((weights * resid0**2).sum())
    mse1 = ss1 / max(n - m1.shape[1], 1)
    denom = ss0 + mse1
    if denom <= 0:
        return 0.0
    estimate = (ss0 - ss1 - df1 * mse1) / denom
    return float(max(estimate, 0.0)) if clip else float(estimate)


def wild_cluster_bootstrap(
    m0: np.ndarray,
    m1: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    df1: int,
    *,
    n_boot: int = 999,
    alpha: float = 0.05,
    weights_type: str = "rademacher",
    seed: int = 0,
) -> CouplingEstimate:
    """Coupling with a restricted wild cluster bootstrap interval and p-value.

    The bootstrap is *restricted*: the null model M0 is fitted once, and each
    resample perturbs its residuals by a cluster-level random sign. This imposes
    the decoupled null exactly, which is what the audit tests, and is the
    recommended form at few clusters (Cameron, Gelbach and Miller, 2008).

    The interval is obtained by inverting that null distribution rather than by
    resampling the estimate, so interval and threshold rest on one test. This
    matters: the threshold returned by `logworthy.audit.calibrate_threshold` is
    calibrated on this same bootstrap, and mixing footings was the defect that
    an earlier version of this analysis had to correct.

    Args:
        m0: State-only design matrix.
        m1: Full design matrix including the action block.
        y: Outcome.
        weights: Importance weights.
        clusters: Cluster label per row. Any hashable labels; they are
            factorised internally.
        df1: Degrees of freedom of the action block.
        n_boot: Bootstrap replications. 999 is the default; use at least 399
            for a 5% test, and more when the p-value is near the boundary.
        alpha: One minus the interval's coverage.
        weights_type: ``"rademacher"`` (two-point, the default) or ``"webb"``
            (six-point). Webb weights are preferred below roughly 12 clusters,
            where the Rademacher distribution has too few distinct values.
        seed: Seed for reproducibility.

    Returns:
        A :class:`CouplingEstimate`.

    Raises:
        ValueError: If inputs disagree in length, or fewer than two clusters.
    """
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    n = len(y)
    if not (len(m0) == len(m1) == n == len(weights) == len(clusters)):
        raise ValueError(
            "y, m0, m1, weights and clusters must have the same length; got "
            f"{n}, {len(m0)}, {len(m1)}, {len(weights)}, {len(clusters)}"
        )
    codes, uniques = _factorise(clusters)
    n_clusters = len(uniques)
    if n_clusters < 2:
        raise ValueError(f"need at least 2 clusters, got {n_clusters}")

    observed = partial_omega2(m0, m1, y, weights, df1)
    sqrt_w = np.sqrt(weights)
    resid0, beta0 = _wls_residuals(m0, y, sqrt_w)
    fitted0 = m0 @ beta0

    rng = np.random.default_rng(seed)
    draw = _weight_sampler(weights_type)
    null = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        signs = draw(rng, n_clusters)[codes]
        null[b] = partial_omega2(m0, m1, fitted0 + resid0 * signs, weights, df1)

    # p-value: the usual (1 + #{null >= observed}) / (1 + B), which is valid
    # for any B and never returns exactly zero.
    p_value = float((1 + (null >= observed).sum()) / (1 + n_boot))

    # Interval by inverting the null distribution. The lower limit is the
    # observed estimate shifted down by the null's upper tail; a coupling the
    # null routinely reaches has a lower limit at zero.
    hi_null = float(np.quantile(null, 1 - alpha / 2))
    lo_null = float(np.quantile(null, alpha / 2))
    lower = max(observed - hi_null, 0.0)
    upper = max(observed + hi_null - lo_null, lower)

    return CouplingEstimate(
        omega2=observed,
        ci=(lower, upper),
        p_value=p_value,
        df1=int(df1),
        n=int(n),
        n_clusters=int(n_clusters),
        n_eff=kish_n_eff(weights),
    )


def _factorise(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map arbitrary cluster labels to 0..G-1. Mirrors pandas.factorize."""
    uniques, codes = np.unique(np.asarray(labels), return_inverse=True)
    return codes, uniques


def _weight_sampler(kind: str):
    """Return a function drawing auxiliary bootstrap weights."""
    kind = kind.lower()
    if kind == "rademacher":
        return lambda rng, size: rng.choice(np.array([-1.0, 1.0]), size=size)
    if kind == "webb":
        # Webb's six-point distribution, each value with probability 1/6.
        points = np.array(
            [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)]
        )
        return lambda rng, size: rng.choice(points, size=size)
    raise ValueError(f"unknown weights_type {kind!r}; use 'rademacher' or 'webb'")


def kish_n_eff(weights: np.ndarray) -> float:
    """Kish effective sample size, ``(sum w)^2 / sum w^2``."""
    w = np.asarray(weights, dtype=float)
    denom = float((w**2).sum())
    return float((w.sum() ** 2) / denom) if denom > 0 else 0.0


def design_effect(
    m0: np.ndarray,
    m1: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    df1: int,
) -> float:
    """Generalised design effect of the action block (Rao-Scott).

    Defined as ``tr(V0^-1 V_CR) / df1``, where ``V0`` is the model-based
    covariance of the action-block coefficients and ``V_CR`` its cluster-robust
    counterpart. It converts rows into independent-row equivalents for *this
    block*, which is what the detectability threshold needs.

    This is deliberately not the Kish mean-based factor. With uneven clusters
    the two differ by several fold, and the mean-based factor answers a question
    about cluster sizes rather than about the contrast being tested.

    Returns:
        The design effect. Values below 1 are returned as 1.0, since a block
        cannot carry more independent information than it has rows.
    """
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    codes, uniques = _factorise(clusters)
    sqrt_w = np.sqrt(weights)
    k0 = m1.shape[1] - df1

    xw = m1 * sqrt_w[:, None]
    xtx = xw.T @ xw
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)

    resid, _ = _wls_residuals(m1, y, sqrt_w)
    meat = np.zeros_like(xtx)
    for g in range(len(uniques)):
        rows = codes == g
        sg = (m1[rows] * weights[rows, None]).T @ resid[rows]
        meat += np.outer(sg, sg)

    sigma2 = float((weights * resid**2).sum()) / max(len(y) - m1.shape[1], 1)
    v_model = sigma2 * xtx_inv
    v_cr = xtx_inv @ meat @ xtx_inv

    block = slice(k0, m1.shape[1])
    try:
        ratio = np.linalg.solve(v_model[block, block], v_cr[block, block])
    except np.linalg.LinAlgError:
        ratio = np.linalg.pinv(v_model[block, block]) @ v_cr[block, block]
    deff = float(np.trace(ratio) / df1)
    return max(deff, 1.0)
