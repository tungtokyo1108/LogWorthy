"""Stage-0 gates: the checks that run before any coupling is measured.

These answer questions that no amount of data can fix later. A target that is an
accounting identity of the action (C1) cannot be audited for coupling, because
Q(s, a) is not meaningful. A log with no overlap (C4a) is silent, not negative.
Running the coupling test first and these gates afterwards would produce
confident numbers about a quantity that does not exist.

Each gate returns a :class:`GateResult` that carries the measurement, the
threshold it was compared against, and whether it passed, so a refusal can
always be traced to a number.
"""

from __future__ import annotations

from typing import Optional

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GateResult",
    "definitional_target",
    "overlap",
    "propensity_calibration",
    "propensity_informativeness",
    "effective_overlap",
]


@dataclass(frozen=True)
class GateResult:
    """Outcome of one feasibility gate.

    Attributes:
        code: The condition's code, e.g. ``"C1"``.
        passed: Whether the condition is satisfied.
        statistic: The measured value.
        threshold: What it was compared against, or ``None`` when the gate is a
            report rather than a test.
        detail: One sentence explaining the result.
    """

    code: str
    passed: bool
    statistic: float
    threshold: Optional[float]
    detail: str

    def __str__(self) -> str:  # pragma: no cover - display only
        mark = "pass" if self.passed else "FAIL"
        return f"{self.code} {mark}: {self.detail}"


def definitional_target(
    r2_out_of_sample: Optional[float] = None,
    *,
    recovery_rate: Optional[float] = None,
    threshold: float = 0.95,
) -> GateResult:
    """C1: the target must not be an accounting identity of the action.

    When the outcome is a deterministic function of the action, an offline audit
    will measure a large, perfectly real coupling that licenses nothing: the
    policy cannot change the outcome, it only relabels it.

    Two statistics are accepted, and which one is right depends on the outcome.
    For a continuous outcome, pass ``r2_out_of_sample``. For a **binary or
    categorical** outcome, pass ``recovery_rate``: the share of rows a model
    recovers exactly. R-squared understates recoverability badly on a binary
    target — DataCo's on-time indicator is reproduced in 97.55% of rows by a
    rule on the scheduled and realised shipping days, yet scores an R-squared
    near 0.90, which would pass a threshold the target plainly fails.

    This screen flags, it does not decide. A high score on a genuinely
    predictable outcome is possible, so the paper pairs the screen with human
    review of how the target is constructed.

    Args:
        r2_out_of_sample: Held-out R-squared of ``y`` on ``(s, a)``, for a
            continuous outcome.
        recovery_rate: Share of rows whose outcome is recovered exactly, for a
            discrete outcome.
        threshold: Value above which the target is flagged. Default 0.95,
            which refuses DataCo's on-time indicator at its measured 0.9755
            recovery rate. Raise it for a target that is legitimately almost
            deterministic, and record the choice in the analysis plan.

    Returns:
        A :class:`GateResult` for C1.

    Raises:
        ValueError: If neither statistic, or both, are supplied.
    """
    if (r2_out_of_sample is None) == (recovery_rate is None):
        raise ValueError(
            "pass exactly one of r2_out_of_sample (continuous outcome) or "
            "recovery_rate (discrete outcome)"
        )
    if recovery_rate is not None:
        value, label = float(recovery_rate), "exact recovery rate of y from (s,a)"
    else:
        value, label = float(r2_out_of_sample), "out-of-sample R^2 of y on (s,a)"

    passed = value < threshold
    detail = f"{label} is {value:.4f}" + (
        f", at or above {threshold}: the target is recoverable from the action "
        "and should be redefined or replaced before any coupling verdict"
        if not passed
        else f", below {threshold}"
    )
    return GateResult("C1", passed, value, threshold, detail)


def overlap(
    propensities: np.ndarray, *, trim: float = 0.01, max_share_below: float = 0.05
) -> GateResult:
    """C4a: local overlap. Enough mass must lie on the supported region.

    Args:
        propensities: Estimated ``pi_b(a | s)`` for the action actually taken.
        trim: Pre-specified trimming floor. Must be fixed in the analysis plan,
            not chosen after seeing verdicts.
        max_share_below: Share of rows below ``trim`` above which the gate fails
            and the log is routed to ``0 (I)`` — randomise.

    Returns:
        A :class:`GateResult` for C4a.
    """
    pi = np.asarray(propensities, dtype=float)
    share = float((pi < trim).mean())
    passed = share <= max_share_below
    detail = (
        f"{share:.2%} of rows have pi_b below the pre-specified trim {trim}"
        + (
            f", above the {max_share_below:.0%} allowance: the data are silent on "
            "those states, not negative, so exploration is indicated"
            if not passed
            else ""
        )
    )
    return GateResult("C4a", passed, share, max_share_below, detail)


def propensity_calibration(
    predicted: np.ndarray,
    actions: np.ndarray,
    *,
    n_bins: int = 10,
    threshold: float = 0.05,
) -> GateResult:
    """C5a: the propensity model must be calibrated.

    Weights are only as trustworthy as the model producing them. An uncalibrated
    propensity model yields weights that are wrong in a direction the audit
    cannot see.

    The statistic is the class-wise expected calibration error. For each action
    ``k`` the predicted probabilities are binned by equal count, and within each
    bin the mean predicted probability is compared against the realised
    frequency of that action. The per-class errors are averaged. This needs the
    *full* predicted distribution, not just the probability of the action taken:
    the probability of the realised action alone cannot be checked against
    anything, since its realised indicator is identically one.

    Args:
        predicted: Predicted distribution over actions, shape ``(n, K)``. Rows
            should sum to 1.
        actions: Integer action index actually taken, shape ``(n,)``, with
            values in ``0..K-1``.
        n_bins: Number of equal-count bins per class.
        threshold: ECE above which the gate fails. Default 0.05.

    Returns:
        A :class:`GateResult` for C5a.

    Raises:
        ValueError: If ``predicted`` is not two-dimensional or disagrees with
            ``actions`` in length.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actions)
    if p.ndim != 2:
        raise ValueError(
            "predicted must have shape (n, K): the full distribution over "
            f"actions, got shape {p.shape}. The probability of the realised "
            "action alone is not checkable."
        )
    if len(p) != len(a):
        raise ValueError(f"predicted and actions length mismatch: {len(p)} vs {len(a)}")
    if p.size == 0:
        return GateResult("C5a", False, float("nan"), threshold, "no predictions supplied")

    n, n_actions = p.shape
    per_class = []
    for k in range(n_actions):
        pk = p[:, k]
        realised = (a == k).astype(float)
        order = np.argsort(pk)
        bins = np.array_split(order, min(n_bins, max(n, 1)))
        err = 0.0
        for b in bins:
            if b.size == 0:
                continue
            err += (b.size / n) * abs(pk[b].mean() - realised[b].mean())
        per_class.append(err)
    ece = float(np.mean(per_class))

    passed = ece < threshold
    detail = f"class-wise expected calibration error {ece:.4f}" + (
        f", at or above {threshold}: the weights cannot be trusted"
        if not passed
        else f", below {threshold}"
    )
    return GateResult("C5a", passed, ece, threshold, detail)


def propensity_informativeness(
    logloss_base: np.ndarray,
    logloss_enriched: np.ndarray,
    clusters: np.ndarray,
    *,
    threshold: float = 0.01,
    n_boot: int = 999,
    seed: int = 0,
) -> tuple[GateResult, float, float]:
    """C5b: do the covariates predict the action at all?

    Compares held-out log-loss of a propensity model using enrichment covariates
    against one using only the base covariates. A positive improvement means the
    recorded state carries information about how actions were chosen.

    This same quantity is the ladder's second collapse condition, which is why
    the function returns its standard error: the ladder needs a *test*, not a
    point estimate.

    Args:
        logloss_base: Per-row held-out log loss of the base model.
        logloss_enriched: Per-row held-out log loss of the enriched model.
        clusters: Cluster labels, for the cluster bootstrap standard error.
        threshold: Improvement in nats per row the gate asks for. Default 0.01.
        n_boot: Cluster bootstrap resamples for the standard error.
        seed: Seed for reproducibility.

    Returns:
        ``(gate, delta_prop, standard_error)``. ``delta_prop`` is positive when
        the enriched model is better.
    """
    base = np.asarray(logloss_base, dtype=float)
    rich = np.asarray(logloss_enriched, dtype=float)
    diff = base - rich  # positive: the enriched model predicts better
    delta = float(diff.mean())

    uniques, codes = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = len(uniques)
    rows = [np.flatnonzero(codes == g) for g in range(n_clusters)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([rows[g] for g in pick])
        boots[b] = diff[idx].mean()
    se = float(boots.std(ddof=1))

    passed = delta > threshold
    detail = (
        f"enrichment covariates improve held-out log loss by {delta:+.4f} nats/row "
        f"(SE {se:.4f})" + ("" if passed else f", not clearing {threshold}")
    )
    return GateResult("C5b", passed, delta, threshold, detail), delta, se


def effective_overlap(weights: np.ndarray, n_rows: Optional[int] = None) -> GateResult:
    """C4b: precision. Reports effective overlap rather than testing it.

    ``o_eff = n_eff / n`` is the share of nominal sample size the weights leave.
    A small value is not a failure in itself; it is reported so that a later
    refusal for power can be traced to its cause.

    Args:
        weights: Importance weights.
        n_rows: Row count, defaulting to ``len(weights)``.

    Returns:
        A :class:`GateResult` for C4b, always ``passed=True`` since this is a
        report, with ``o_eff`` as the statistic.
    """
    from ..core.coupling import kish_n_eff

    w = np.asarray(weights, dtype=float)
    n = int(n_rows if n_rows is not None else w.size)
    n_eff = kish_n_eff(w)
    o_eff = n_eff / n if n > 0 else 0.0
    return GateResult(
        "C4b",
        True,
        float(o_eff),
        None,
        f"effective overlap o_eff={o_eff:.4f} (n_eff={n_eff:,.0f} of {n:,} rows)",
    )
