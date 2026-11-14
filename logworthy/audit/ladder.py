"""The saturation ladder: the audit's only causal diagnostic.

A coupling measured on an observational log may be real or may be an artefact of
a confounder the state model omits. The ladder tests this by re-estimating the
coupling over a pre-specified sequence of richer state models and reading what
happens to it.

Three outcomes, mirroring the three-branch rule:

*Collapse* means the coupling largely disappears once the richer state is
adjusted for, so the association was not coupling. It requires **two** things,
and the second is what makes the diagnosis credible:

1. the upper confidence limit of the noise-adjusted ratio of the richest rung to
   the base rung lies below ``delta_col``; and
2. the enrichment covariates significantly improve a held-out propensity model
   for the action.

Without the second condition a decline is as consistent with variance absorption
as with confounding. The condition is a one-sided *test*, not a point estimate:
an earlier version of this analysis used ``delta_prop > 0`` as a point estimate
and the branch then fired on 55% of pure variance-absorption replications,
because a point estimate of a quantity whose true value is zero clears zero about
half the time. On a randomised log the condition cannot hold by construction, so
collapse cannot fire there.

*Plateau* means the coupling has stopped moving: two consecutive relative changes
have upper limits below ``delta_pl``.

Anything else is indeterminate and caps the verdict at Tier 2.

The ratio is read on the **absolute** coupling, not the normalised one. Writing
omega^2 as SS_delta / (SS_total + MSE), a covariate that both confounds the
action and explains the outcome shrinks numerator and denominator together, and
can shrink the denominator faster, so a normalised ladder can *rise* while the
coupling it tracks has more than halved.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass, field

import numpy as np

from ..core.coupling import _factorise, _weight_sampler, _wls_residuals, partial_omega2

__all__ = ["Rung", "LadderResult", "LadderOutcome", "run_ladder"]


class LadderOutcome:
    """Outcomes of the ladder."""

    COLLAPSE = "collapse"
    PLATEAU = "plateau"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class Rung:
    """One rung: a state specification and the coupling measured on it.

    Attributes:
        name: Label, e.g. ``"L2 +category"``.
        omega2: Normalised coupling, reported for reference.
        absolute: Absolute coupling, the scale the ratio is read on.
    """

    name: str
    omega2: float
    absolute: float


@dataclass(frozen=True)
class LadderResult:
    """What the ladder found.

    Attributes:
        outcome: One of the :class:`LadderOutcome` constants.
        rungs: The rungs in order, base first.
        ratio: Point estimate of ``absolute(top) / absolute(base)``.
        ratio_ci: Confidence interval for that ratio, by paired cluster
            bootstrap across rungs (resamples are shared, so the ratio's
            uncertainty accounts for the correlation between rungs).
        noise_control: Ratio obtained on permuted actions. A value far from 1
            means the ladder moves for mechanical reasons and the reading is
            unsafe.
        delta_prop: Held-out log-loss improvement, in nats per row, of a
            propensity model using the enrichment covariates over one using the
            base covariates. ``None`` if not supplied.
        delta_prop_lower: Lower 95% limit of ``delta_prop``. Collapse requires
            this to exceed zero.
        reason: Human-readable explanation of why this outcome was reached.
    """

    outcome: str
    rungs: tuple[Rung, ...]
    ratio: float
    ratio_ci: tuple[float, float]
    noise_control: Optional[float] = None
    delta_prop: Optional[float] = None
    delta_prop_lower: Optional[float] = None
    reason: str = ""
    _meta: dict = field(default_factory=dict, repr=False)

    @property
    def adds_c_query(self) -> bool:
        """True when the ladder is indeterminate, which adds the C? code."""
        return self.outcome == LadderOutcome.INDETERMINATE

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"ladder={self.outcome} ratio={self.ratio:.3f} "
            f"[{self.ratio_ci[0]:.3f},{self.ratio_ci[1]:.3f}] :: {self.reason}"
        )


def run_ladder(
    designs: list[tuple[str, np.ndarray]],
    m0_base: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray,
    df1: int,
    *,
    delta_col: float = 0.5,
    delta_pl: float = 0.15,
    delta_prop: Optional[float] = None,
    delta_prop_se: Optional[float] = None,
    n_boot: int = 199,
    noise_control: Optional[float] = None,
    seed: int = 0,
) -> LadderResult:
    """Run the saturation ladder over a pre-specified sequence of state models.

    Args:
        designs: Ordered ``(name, m1)`` pairs, base rung first. Each ``m1`` is a
            full design matrix whose last ``df1`` columns are the action block.
            The sequence must be pre-specified, not chosen after seeing results.
        m0_base: State-only design matrix of the *base* rung, used to define the
            absolute scale that ratios are read on.
        y: Outcome.
        weights: Importance weights.
        clusters: Cluster labels.
        df1: Degrees of freedom of the action block.
        delta_col: Collapse threshold for the ratio. Default 0.5, as calibrated
            in the paper.
        delta_pl: Plateau threshold for consecutive relative changes. Default
            0.15.
        delta_prop: Held-out log-loss improvement in nats per row from the
            enrichment covariates. Required for collapse to fire. Compute it
            with :func:`logworthy.audit.gates.propensity_informativeness`.
        delta_prop_se: Standard error of ``delta_prop``, for the one-sided test.
            If omitted, ``delta_prop`` is treated as untested and collapse
            cannot fire.
        n_boot: Paired cluster bootstrap resamples for the ratio interval.
        noise_control: Optional ratio obtained on permuted actions.
        seed: Seed for reproducibility.

    Returns:
        A :class:`LadderResult`.

    Raises:
        ValueError: If fewer than two rungs are supplied.
    """
    if len(designs) < 2:
        raise ValueError(f"the ladder needs at least 2 rungs, got {len(designs)}")

    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)
    codes, uniques = _factorise(clusters)
    n_clusters = len(uniques)

    rungs = tuple(
        Rung(
            name=name,
            omega2=partial_omega2(m1[:, : m1.shape[1] - df1], m1, y, weights, df1),
            absolute=_absolute_coupling(m0_base, m1, y, weights, df1),
        )
        for name, m1 in designs
    )

    base, top = rungs[0].absolute, rungs[-1].absolute
    ratio = float(top / base) if base > 0 else float("inf")

    ratio_lo, ratio_hi = _paired_cluster_bootstrap_ratio(
        designs, m0_base, y, weights, codes, n_clusters, df1, n_boot, seed
    )

    prop_lower = None
    if delta_prop is not None and delta_prop_se is not None:
        prop_lower = float(delta_prop - 1.645 * delta_prop_se)

    outcome, reason = _classify(
        ratio_hi, rungs, delta_col, delta_pl, delta_prop, prop_lower
    )

    return LadderResult(
        outcome=outcome,
        rungs=rungs,
        ratio=ratio,
        ratio_ci=(ratio_lo, ratio_hi),
        noise_control=noise_control,
        delta_prop=delta_prop,
        delta_prop_lower=prop_lower,
        reason=reason,
        _meta={"n_clusters": n_clusters, "delta_col": delta_col, "delta_pl": delta_pl},
    )


def _classify(ratio_hi, rungs, delta_col, delta_pl, delta_prop, prop_lower):
    """Decide the ladder's outcome. Collapse is checked first and overrides."""
    if ratio_hi < delta_col:
        if prop_lower is None:
            return (
                LadderOutcome.INDETERMINATE,
                f"the ratio declined (upper limit {ratio_hi:.3f} < {delta_col}) but "
                "no tested propensity improvement was supplied, so the decline is as "
                "consistent with variance absorption as with confounding",
            )
        if prop_lower > 0:
            return (
                LadderOutcome.COLLAPSE,
                f"ratio upper limit {ratio_hi:.3f} < {delta_col} and the enrichment "
                f"covariates predict the action (delta_prop lower limit "
                f"{prop_lower:+.4f} > 0 nats)",
            )
        return (
            LadderOutcome.INDETERMINATE,
            f"the ratio declined (upper limit {ratio_hi:.3f} < {delta_col}) but the "
            f"enrichment covariates do not predict the action (delta_prop lower limit "
            f"{prop_lower:+.4f} <= 0), so confounding is not established",
        )

    changes = _relative_changes(rungs)
    if len(changes) >= 2 and max(changes[-2:]) < delta_pl:
        return (
            LadderOutcome.PLATEAU,
            f"the last two relative changes ({changes[-2]:.3f}, {changes[-1]:.3f}) "
            f"are below {delta_pl}",
        )
    return (
        LadderOutcome.INDETERMINATE,
        f"the ratio's upper limit {ratio_hi:.3f} does not clear {delta_col} and the "
        "ladder has not plateaued",
    )


def _relative_changes(rungs) -> list[float]:
    """Absolute relative change between consecutive rungs."""
    out = []
    for prev, cur in zip(rungs, rungs[1:]):
        if prev.absolute > 0:
            out.append(abs(cur.absolute - prev.absolute) / prev.absolute)
    return out


def _absolute_coupling(m0_base, m1, y, weights, df1) -> float:
    """Absolute coupling: variance explained by the action block, unnormalised.

    Measured against a fixed denominator (the base rung's total) so that rungs
    are comparable. This is the scale the collapse ratio is read on.
    """
    sqrt_w = np.sqrt(weights)
    n = len(y)
    resid1, _ = _wls_residuals(m1, y, sqrt_w)
    m0_rung = m1[:, : m1.shape[1] - df1]
    resid0, _ = _wls_residuals(m0_rung, y, sqrt_w)
    ss1 = float((weights * resid1**2).sum())
    ss0 = float((weights * resid0**2).sum())
    mse1 = ss1 / max(n - m1.shape[1], 1)
    explained = ss0 - ss1 - df1 * mse1

    resid_base, _ = _wls_residuals(m0_base, y, sqrt_w)
    scale = float((weights * resid_base**2).sum())
    if scale <= 0:
        return 0.0
    return float(max(explained, 0.0) / scale)


def _paired_cluster_bootstrap_ratio(
    designs, m0_base, y, weights, codes, n_clusters, df1, n_boot, seed
):
    """Interval for top/base, resampling clusters once per replicate.

    Sharing the resample across rungs is what makes this a *paired* bootstrap:
    the rungs move together, so the ratio's uncertainty reflects only their
    difference, not the common variation they share.
    """
    rng = np.random.default_rng(seed)
    cluster_rows = [np.flatnonzero(codes == g) for g in range(n_clusters)]
    ratios = np.empty(n_boot)
    name_top, m1_top = designs[-1]
    _, m1_base = designs[0]

    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([cluster_rows[g] for g in pick])
        yb, wb = y[idx], weights[idx]
        base_b = _absolute_coupling(m0_base[idx], m1_base[idx], yb, wb, df1)
        top_b = _absolute_coupling(m0_base[idx], m1_top[idx], yb, wb, df1)
        ratios[b] = top_b / base_b if base_b > 0 else np.nan

    good = ratios[np.isfinite(ratios)]
    if good.size == 0:
        return (0.0, float("inf"))
    return float(np.quantile(good, 0.025)), float(np.quantile(good, 0.975))
