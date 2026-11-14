"""The detectability threshold and the three-branch rule.

A coupling estimate means nothing on its own. The question the audit asks is
whether the coupling is large enough that *this design*, with its clusters,
weights and contrast, could have detected it. That reference point is the
detectability threshold ``eta2_min``.

Two ways to obtain it are provided, and they are not interchangeable.

`analytic_threshold` inverts the non-central F power function, treating the
information count as independent rows. It is fast, closed-form, and correct when
rows really are independent.

`calibrate_threshold` is Algorithm 2 of the paper: it injects a rescaled copy of
the fitted action block into null data and finds the smallest coupling the
audit's *own* test rejects with the stated power. This is the footing the audit
requires, because interval and threshold then come from one test. Prefer it
whenever the interval came from the cluster bootstrap.

A cluster-robust correction such as CR2 with Satterthwaite degrees of freedom
diagnoses how little independent information few clusters supply, but it cannot
set a threshold: its reference distribution F(q, eta-q+1) can leave barely one
denominator degree of freedom, implying thresholds orders of magnitude above any
attainable coupling. The package therefore does not offer it as a threshold.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats

from .coupling import partial_omega2

__all__ = [
    "Threshold",
    "Branch",
    "analytic_threshold",
    "calibrate_threshold",
    "three_branch",
]


@dataclass(frozen=True)
class Threshold:
    """The smallest coupling a design can detect, with its uncertainty.

    Attributes:
        eta2_min: Point estimate of the threshold.
        band: ``(low, high)`` Monte Carlo band. The three-branch rule compares
            the coupling interval against this band, not against the point.
        power: Power the threshold is calibrated at, conventionally 0.8.
        alpha: Test size the threshold is calibrated at.
        method: ``"analytic"`` or ``"calibrated"``.
    """

    eta2_min: float
    band: tuple[float, float]
    power: float
    alpha: float
    method: str

    @property
    def low(self) -> float:
        """Lower edge of the band, written ``eta2_min,lo``."""
        return self.band[0]

    @property
    def high(self) -> float:
        """Upper edge of the band, written ``eta2_min,hi``."""
        return self.band[1]

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"eta2_min={self.eta2_min:.4e} "
            f"band=[{self.band[0]:.4e},{self.band[1]:.4e}] "
            f"({self.method}, power={self.power})"
        )


class Branch:
    """Outcomes of the three-branch rule."""

    BELOW = "below detectability"
    DETECTABLE = "robustly detectable"
    INDETERMINATE = "indeterminate"


def analytic_threshold(
    n_info: float,
    df1: int,
    *,
    power: float = 0.80,
    alpha: float = 0.05,
    band_frac: float = 0.0,
) -> Threshold:
    """Smallest detectable coupling, by inverting the non-central F power function.

    Solves for the ``eta2`` whose non-centrality ``lambda = n_info * eta2 / (1 - eta2)``
    gives the stated power against an F test with ``df1`` numerator and
    ``n_info - df1 - 1`` denominator degrees of freedom.

    Args:
        n_info: Information count: independent-row equivalents available to the
            contrast. With clustered data pass ``n_eff / design_effect``, not
            the raw row count.
        df1: Degrees of freedom of the action block.
        power: Target power, conventionally 0.8.
        alpha: Test size.
        band_frac: Optional relative half-width for the reported band, e.g.
            ``0.1`` for +/-10%. Defaults to a point band, since the analytic
            threshold carries no Monte Carlo error of its own.

    Returns:
        A :class:`Threshold` with ``method="analytic"``.

    Raises:
        ValueError: If ``n_info`` leaves no denominator degrees of freedom.
    """
    df2 = n_info - df1 - 1
    if df2 <= 0:
        raise ValueError(
            f"n_info={n_info:.1f} with df1={df1} leaves no denominator degrees "
            "of freedom; the contrast is not estimable at this information count"
        )
    crit = stats.f.ppf(1 - alpha, df1, df2)

    def shortfall(lam: float) -> float:
        # scipy's non-central F survival function overflows for very large
        # non-centralities, where power is numerically 1 anyway.
        with np.errstate(over="ignore", invalid="ignore"):
            attained = float(stats.ncf.sf(crit, df1, df2, lam))
        if not np.isfinite(attained):
            attained = 1.0
        return attained - power

    lam = optimize.brentq(shortfall, 1e-9, 1e7, xtol=1e-10)
    eta2 = lam / (n_info + lam)
    band = (eta2 * (1 - band_frac), eta2 * (1 + band_frac))
    return Threshold(float(eta2), band, power, alpha, "analytic")


def calibrate_threshold(
    m0: np.ndarray,
    m1: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    df1: int,
    *,
    grid: Sequence[float],
    power: float = 0.80,
    alpha: float = 0.05,
    n_sim: int = 60,
    n_boot: int = 49,
    weights_type: str = "rademacher",
    seed: int = 0,
    progress: Optional[Callable[[float, float], None]] = None,
) -> Threshold:
    """Algorithm 2: calibrate the threshold on the audit's own test.

    For each candidate coupling on ``grid``, data are generated under the null
    (fitted M0 plus cluster-sign-flipped residuals) with a rescaled copy of the
    fitted action block injected at that coupling. The audit's wild cluster
    bootstrap test is then run, and the rejection rate estimated. The threshold
    is the smallest coupling whose rejection rate reaches ``power``, obtained by
    linear interpolation between the bracketing grid points.

    The band comes from the Monte Carlo uncertainty of the power curve: the
    Wilson interval for each rejection rate is interpolated separately, so the
    band widens where the curve is estimated from few replications.

    This is expensive by construction: cost is roughly
    ``len(grid) * n_sim * n_boot`` weighted least squares fits. The defaults
    reproduce the paper's settings. Raise ``n_sim`` before ``n_boot`` if the
    curve looks noisy.

    Args:
        m0: State-only design matrix.
        m1: Full design matrix including the action block.
        y: Outcome, used to fit the null model and the injected block.
        weights: Importance weights.
        clusters: Cluster labels.
        df1: Degrees of freedom of the action block.
        grid: Candidate couplings, ascending. Should bracket the threshold;
            the search stops early once power exceeds 0.97.
        power: Target power.
        alpha: Test size.
        n_sim: Replications per grid point.
        n_boot: Bootstrap replications inside each test.
        weights_type: ``"rademacher"`` or ``"webb"``.
        seed: Seed for reproducibility.
        progress: Optional callback receiving ``(eta, power)`` per grid point.

    Returns:
        A :class:`Threshold` with ``method="calibrated"``.

    Raises:
        ValueError: If no grid point reaches the target power. Widen the grid.
    """
    from .coupling import _factorise, _weight_sampler, _wls_residuals

    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    codes, uniques = _factorise(clusters)
    n_clusters = len(uniques)
    sqrt_w = np.sqrt(weights)
    draw = _weight_sampler(weights_type)

    # The action block as fitted, centred: the shape we inject.
    _, beta1 = _wls_residuals(m1, y, sqrt_w)
    k0 = m1.shape[1] - df1
    block = m1[:, k0:] @ beta1[k0:]
    block = block - block.mean()
    var_block = float((weights * block**2).sum() / weights.sum())

    resid0, beta0 = _wls_residuals(m0, y, sqrt_w)
    fitted0 = m0 @ beta0
    wmean = float((weights * y).sum() / weights.sum())
    var_y = float((weights * (y - wmean) ** 2).sum() / weights.sum())

    rng = np.random.default_rng(seed)
    curve: list[tuple[float, float, float, float]] = []
    for eta in grid:
        if var_block <= 0:
            raise ValueError(
                "the fitted action block has no variance; the contrast cannot "
                "be injected, so no threshold can be calibrated"
            )
        theta = np.sqrt(eta / (1 - eta) * var_y / var_block)
        hits = 0
        for _ in range(n_sim):
            signs = draw(rng, n_clusters)[codes]
            y_star = fitted0 + resid0 * signs + theta * block
            if _rejects(m0, m1, y_star, weights, codes, n_clusters, df1,
                        alpha, n_boot, draw, rng):
                hits += 1
        rate = hits / n_sim
        lo, hi = _wilson(rate, n_sim)
        curve.append((float(eta), rate, lo, hi))
        if progress is not None:
            progress(float(eta), rate)
        if rate >= 0.97:
            break

    point = _interpolate(curve, 1, power)
    if point is None:
        reached = max(c[1] for c in curve)
        raise ValueError(
            f"no grid point reached power {power}; highest was {reached:.2f}. "
            "Extend `grid` upward."
        )
    # A higher rejection rate corresponds to a lower threshold, so the band's
    # low edge comes from the optimistic (upper) power limit.
    low = _interpolate(curve, 3, power) or point
    high = _interpolate(curve, 2, power) or point
    return Threshold(point, (min(low, high), max(low, high)), power, alpha, "calibrated")


def _rejects(m0, m1, y, weights, codes, n_clusters, df1, alpha, n_boot, draw, rng) -> bool:
    """One run of the audit's restricted wild cluster bootstrap test."""
    from .coupling import _wls_residuals

    sqrt_w = np.sqrt(weights)
    observed = partial_omega2(m0, m1, y, weights, df1)
    resid0, beta0 = _wls_residuals(m0, y, sqrt_w)
    fitted0 = m0 @ beta0
    null = np.empty(n_boot)
    for b in range(n_boot):
        signs = draw(rng, n_clusters)[codes]
        null[b] = partial_omega2(m0, m1, fitted0 + resid0 * signs, weights, df1)
    return bool(observed > np.quantile(null, 1 - alpha))


def _wilson(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion; stable at 0 and 1."""
    denom = 1 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denom
    half = z * np.sqrt(rate * (1 - rate) / n + z**2 / (4 * n * n)) / denom
    return float(centre - half), float(centre + half)


def _interpolate(curve, column: int, target: float) -> Optional[float]:
    """Coupling at which `column` of the power curve first crosses `target`."""
    for j in range(1, len(curve)):
        lo_pt, hi_pt = curve[j - 1], curve[j]
        if hi_pt[column] >= target >= lo_pt[column]:
            span = hi_pt[column] - lo_pt[column]
            if abs(span) < 1e-12:
                return float(lo_pt[0])
            frac = (target - lo_pt[column]) / span
            return float(lo_pt[0] + frac * (hi_pt[0] - lo_pt[0]))
    if curve and curve[0][column] >= target:
        return float(curve[0][0])
    return None


def three_branch(coupling, threshold: Threshold) -> str:
    """Apply the three-branch rule of the paper.

    The comparison is deliberately asymmetric and uses the *band*, not the point
    threshold, so that a verdict is only issued when it survives the threshold's
    own uncertainty:

        below detectability   if U-hat < eta2_min,lo
        robustly detectable   if L-hat > eta2_min,hi
        indeterminate         otherwise

    Args:
        coupling: A :class:`~logworthy.core.coupling.CouplingEstimate`, or any
            object exposing ``lower`` and ``upper``.
        threshold: The design's detectability threshold.

    Returns:
        One of the :class:`Branch` constants.
    """
    if coupling.upper < threshold.low:
        return Branch.BELOW
    if coupling.lower > threshold.high:
        return Branch.DETECTABLE
    return Branch.INDETERMINATE
