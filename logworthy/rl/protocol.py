"""The certification protocol: Stage 2 of the audit.

Certification asks whether the learner a team would actually deploy shows an
improvement these data can distinguish from none. The protocol is fixed in
advance, and each of its parts answers a specific failure.

**Disjoint folds.** The learner is fitted on one fold and evaluated on another.
Under fold reuse the false-licence rate grows with the number of state cells
selected over, so reuse does not merely add noise, it adds bias that worsens as
the state grows richer.

**Many seeds.** One honest split is one draw. The protocol repeats over
``n_seeds`` splits and certifies when the proportion of seeds whose doubly
robust interval excludes zero exceeds ``q``. With 30 seeds the seed proportion
is itself uncertain to about +/- 0.09, which is why ``q`` is a proportion and
not "at least one seed".

**A permutation null.** A seed proportion means nothing without knowing what
proportion a decoupled log would produce. The null permutes the outcome within
strata and reruns the pipeline, giving a calibrated reference.

**The increment rule.** For a contextual class the certified quantity is the
*paired* uplift over the best fixed rule, not the uplift over the logging
policy. Since the fixed-rule class is contained in the contextual one, the
latter is implied by a fixed-rule certificate and carries no information.

**A certificate is not a licence.** `certify` returns what the learner achieved;
it does not decide the tier. A certificate on a coupling the design cannot
resolve is reported as unsupported, and the log is routed by its detectability
branch. That coherence rule lives in :func:`logworthy.verdict.route`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .estimators import ImprovementEstimate, doubly_robust, paired_increment

__all__ = ["CertificationResult", "certify", "certify_increment", "PolicyClass"]


class PolicyClass:
    """Named policy classes."""

    FIXED = "fixed rules"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True)
class CertificationResult:
    """What the learner achieved under the protocol.

    Attributes:
        certified: Whether the seed proportion exceeded ``q``. ``None`` when the
            result is indeterminate against the permutation null, which routes
            to ``1 (L?)``.
        seeds_certified: How many seeds showed an interval excluding zero.
        n_seeds: Seeds run.
        proportion: ``seeds_certified / n_seeds``.
        q: Threshold the proportion was held to.
        mean_uplift: Mean improvement across seeds, in outcome units.
        uplift_ci: Interval for the mean uplift, pooled across seeds.
        null_proportion: Proportion the permutation null reached at the same
            quantile, or ``None`` when no null was run.
        p_value: Permutation p-value for the observed seed proportion.
        learner: Name of the learner certified.
        policy_class: Which class was certified.
        per_seed: One estimate per seed, kept for inspection.
    """

    certified: Optional[bool]
    seeds_certified: int
    n_seeds: int
    proportion: float
    q: float
    mean_uplift: float
    uplift_ci: tuple
    null_proportion: Optional[float] = None
    p_value: Optional[float] = None
    learner: str = ""
    policy_class: str = ""
    per_seed: tuple = field(default_factory=tuple, repr=False)

    def __str__(self) -> str:  # pragma: no cover - display only
        verdict = {True: "certified", False: "not certified", None: "indeterminate"}[
            self.certified
        ]
        null = "" if self.null_proportion is None else f" null={self.null_proportion:.2f}"
        return (
            f"{self.learner} [{self.policy_class}]: {verdict} "
            f"{self.seeds_certified}/{self.n_seeds} seeds "
            f"(q={self.q}){null} uplift={self.mean_uplift:+.5f}"
        )


def certify(
    learner,
    y: np.ndarray,
    actions: np.ndarray,
    state: np.ndarray,
    propensities: np.ndarray,
    *,
    n_actions: Optional[int] = None,
    n_seeds: int = 30,
    q: float = 0.5,
    clusters: Optional[np.ndarray] = None,
    time_order: Optional[np.ndarray] = None,
    n_permutations: int = 0,
    strata: Optional[np.ndarray] = None,
    policy_class: str = PolicyClass.FIXED,
    baseline: str = "logging",
    seed: int = 0,
    progress: Optional[Callable[[int, bool], None]] = None,
) -> CertificationResult:
    """Certify a fixed learner on disjoint folds over many seeds.

    Args:
        learner: Any object satisfying :class:`~logworthy.rl.Learner`. It must
            be chosen *before* seeing results; certifying several and reporting
            the best is selection, and inflates the false-licence rate.
        y: Logged outcome.
        actions: Logged action index.
        state: Pre-decision state, ``(n, d)``.
        propensities: ``pi_b(a | s)`` for the logged action.
        n_actions: Number of actions. Inferred from ``actions`` when omitted.
        n_seeds: Honest splits to run. The paper uses 30.
        q: Proportion of certifying seeds required. The paper uses 0.5.
        clusters: Cluster labels. When given, each seed's interval uses a
            cluster-robust standard error, and splits are made by *cluster*, so
            that no cluster spans both folds.
        time_order: Optional timestamp per row. When given, splits use a rolling
            origin (train on the past, evaluate on the future) instead of random
            halves, which is the honest split for a time-ordered log.
        n_permutations: Permutation replications for the null. 0 skips the null,
            in which case ``certified`` is decided by ``q`` alone.
        strata: Strata within which to permute the outcome. Defaults to
            ``clusters``, then to a single stratum.
        policy_class: Name carried into the result.
        baseline: ``"logging"`` compares against the logged outcome. Use
            :func:`certify_increment` for the contextual increment.
        seed: Base seed.
        progress: Optional callback receiving ``(seed_index, certified)``.

    Returns:
        A :class:`CertificationResult`.

    Raises:
        ValueError: If inputs disagree in length, or the learner lacks the
            required interface.
    """
    y, a, s, pi, n_act = _validate(y, actions, state, propensities, n_actions, learner)
    rng = np.random.default_rng(seed)

    estimates: list[ImprovementEstimate] = []
    for i in range(n_seeds):
        train, test = _split(len(y), rng, clusters, time_order, i, n_seeds)
        if train.sum() < 2 * n_act or test.sum() < 2 * n_act:
            continue
        est = _evaluate_fold(
            learner, y, a, s, pi, n_act, train, test, clusters, baseline
        )
        estimates.append(est)
        if progress is not None:
            progress(i, est.excludes_zero)

    return _assemble(
        estimates, q, learner, policy_class, n_seeds,
        y, a, s, pi, n_act, clusters, time_order, strata, n_permutations, seed,
        baseline,
    )


def certify_increment(
    learner,
    y: np.ndarray,
    actions: np.ndarray,
    state: np.ndarray,
    propensities: np.ndarray,
    *,
    n_actions: Optional[int] = None,
    n_seeds: int = 30,
    q: float = 0.5,
    clusters: Optional[np.ndarray] = None,
    time_order: Optional[np.ndarray] = None,
    seed: int = 0,
    progress: Optional[Callable[[int, bool], None]] = None,
) -> CertificationResult:
    """Certify what personalisation adds over the best fixed rule.

    This is the quantity a contextual verdict must be decided on. On each fold
    the contextual policy (the learner's argmax) and the best fixed rule (the
    single action with the highest trained value) are evaluated on the *same*
    rows, and the paired difference is certified.

    Certifying a contextual policy against the logging policy instead would be
    implied by the fixed-rule certificate and carry no information, since the
    fixed-rule class is contained in the contextual one.

    Args:
        learner: The deployed learner, fixed in advance.
        y: Logged outcome.
        actions: Logged action index.
        state: Pre-decision state.
        propensities: ``pi_b(a | s)`` for the logged action.
        n_actions: Number of actions.
        n_seeds: Honest splits.
        q: Required proportion of certifying seeds.
        clusters: Cluster labels; splits respect cluster boundaries.
        time_order: Timestamps, for a rolling-origin split.
        seed: Base seed.
        progress: Optional callback.

    Returns:
        A :class:`CertificationResult` for the increment.
    """
    y, a, s, pi, n_act = _validate(y, actions, state, propensities, n_actions, learner)
    rng = np.random.default_rng(seed)

    estimates: list[ImprovementEstimate] = []
    for i in range(n_seeds):
        train, test = _split(len(y), rng, clusters, time_order, i, n_seeds)
        if train.sum() < 2 * n_act or test.sum() < 2 * n_act:
            continue
        fitted = _fresh(learner).fit(s[train], a[train], y[train], n_actions=n_act)
        q_test = fitted.q_values(s[test])

        contextual_actions = q_test.argmax(axis=1)
        # The best fixed rule is chosen on the TRAINING fold, so that the
        # comparison does not hand the baseline an advantage the policy lacks.
        q_train = fitted.q_values(s[train])
        best_fixed = int(q_train.mean(axis=0).argmax())
        fixed_actions = np.full(test.sum(), best_fixed)

        ctx = doubly_robust(
            y[test], a[test], pi[test], q_test, contextual_actions,
            baseline=np.zeros(test.sum()),
        )
        fix = doubly_robust(
            y[test], a[test], pi[test], q_test, fixed_actions,
            baseline=np.zeros(test.sum()),
        )
        cl = None if clusters is None else np.asarray(clusters)[test]
        est = paired_increment(ctx, fix, clusters=cl)
        estimates.append(est)
        if progress is not None:
            progress(i, est.excludes_zero)

    return _assemble(
        estimates, q, learner, PolicyClass.CONTEXTUAL, n_seeds,
        y, a, s, pi, n_act, clusters, time_order, None, 0, seed, "increment",
    )


def _validate(y, actions, state, propensities, n_actions, learner):
    y = np.asarray(y, dtype=float)
    a = np.asarray(actions)
    pi = np.asarray(propensities, dtype=float)
    s = np.asarray(state, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
    n = y.size
    if not (a.size == pi.size == s.shape[0] == n):
        raise ValueError(
            "y, actions, state and propensities must agree in length; got "
            f"{n}, {a.size}, {s.shape[0]}, {pi.size}"
        )
    if not (hasattr(learner, "fit") and hasattr(learner, "q_values")):
        raise ValueError(
            f"{type(learner).__name__} does not satisfy the Learner interface: "
            "it needs fit(state, actions, rewards, n_actions=...) and q_values(state)"
        )
    n_act = int(n_actions if n_actions is not None else a.max() + 1)
    return y, a, s, pi, n_act


def _split(n, rng, clusters, time_order, index, n_seeds):
    """One honest split. Respects clusters and time order when given."""
    if time_order is not None:
        # Rolling origin: train on everything before a moving cut, evaluate after.
        order = np.argsort(np.asarray(time_order))
        frac = 0.5 + 0.4 * (index + 1) / (n_seeds + 1)
        cut = int(n * min(frac, 0.9))
        train = np.zeros(n, dtype=bool)
        test = np.zeros(n, dtype=bool)
        train[order[:cut]] = True
        test[order[cut:]] = True
        return train, test
    if clusters is not None:
        codes = np.unique(np.asarray(clusters), return_inverse=True)[1]
        n_clusters = codes.max() + 1
        held = rng.random(n_clusters) < 0.5
        test = held[codes]
        return ~test, test
    test = rng.random(n) < 0.5
    return ~test, test


def _fresh(learner):
    """A clean copy of the learner for this seed.

    Learners hold fitted state, and reusing one object across seeds risks
    carrying information from one split into the next, which would quietly
    break the honesty of the split. Copying is cheap next to fitting, and a
    learner that cannot be copied is used as-is with its own ``fit`` expected
    to reset.
    """
    import copy

    try:
        return copy.deepcopy(learner)
    except Exception:  # pragma: no cover - exotic learners (open handles, GPU)
        return learner


def _evaluate_fold(learner, y, a, s, pi, n_act, train, test, clusters, baseline):
    """Fit on the training fold, estimate the improvement on the evaluation fold."""
    fitted = _fresh(learner).fit(s[train], a[train], y[train], n_actions=n_act)
    q_test = fitted.q_values(s[test])
    if baseline == "best_fixed":
        q_train = fitted.q_values(s[train])
        target = np.full(test.sum(), int(q_train.mean(axis=0).argmax()))
    else:
        target = q_test.argmax(axis=1)
    est = doubly_robust(y[test], a[test], pi[test], q_test, target)
    if clusters is None:
        return est
    se = est.cluster_se(np.asarray(clusters)[test])
    return ImprovementEstimate(
        estimate=est.estimate,
        se=se,
        ci=(est.estimate - 1.96 * se, est.estimate + 1.96 * se),
        influence=est.influence,
        n=est.n,
        estimator=est.estimator + ", cluster-robust",
    )


def _assemble(
    estimates, q, learner, policy_class, n_seeds,
    y, a, s, pi, n_act, clusters, time_order, strata, n_permutations, seed, baseline,
):
    """Turn per-seed estimates into a result, running the null if asked."""
    if not estimates:
        raise ValueError(
            "no seed produced a usable split; the log may be too small, or every "
            "split left a fold without all actions represented"
        )
    certified_seeds = sum(e.excludes_zero for e in estimates)
    proportion = certified_seeds / len(estimates)
    uplifts = np.array([e.estimate for e in estimates])
    mean_uplift = float(uplifts.mean())
    half = 1.96 * float(uplifts.std(ddof=1) / np.sqrt(uplifts.size)) if uplifts.size > 1 else float("inf")

    null_proportion = p_value = None
    certified: Optional[bool] = proportion > q

    if n_permutations > 0:
        null_props = _permutation_null(
            learner, y, a, s, pi, n_act, clusters, time_order, strata,
            len(estimates), n_permutations, seed, baseline,
        )
        null_proportion = float(np.quantile(null_props, 0.95))
        p_value = float((1 + (null_props >= proportion).sum()) / (1 + len(null_props)))
        if p_value > 0.05:
            # The seed proportion is not distinguishable from what a decoupled
            # log produces: indeterminate, not a refusal.
            certified = None if proportion > q else False

    return CertificationResult(
        certified=certified,
        seeds_certified=int(certified_seeds),
        n_seeds=len(estimates),
        proportion=float(proportion),
        q=float(q),
        mean_uplift=mean_uplift,
        uplift_ci=(mean_uplift - half, mean_uplift + half),
        null_proportion=null_proportion,
        p_value=p_value,
        learner=getattr(learner, "name", type(learner).__name__),
        policy_class=policy_class,
        per_seed=tuple(estimates),
    )


def _permutation_null(
    learner, y, a, s, pi, n_act, clusters, time_order, strata,
    n_seeds, n_permutations, seed, baseline,
):
    """Seed proportions a decoupled log would produce.

    The outcome is permuted within strata, which destroys any action-outcome
    relation while preserving the stratum structure. The whole pipeline is then
    rerun, so the null reflects the protocol, not just the estimator.
    """
    if strata is None:
        strata = clusters
    codes = (
        np.zeros(len(y), dtype=int)
        if strata is None
        else np.unique(np.asarray(strata), return_inverse=True)[1]
    )
    rng = np.random.default_rng(seed + 10_000)
    out = np.empty(n_permutations)
    for b in range(n_permutations):
        y_perm = y.copy()
        for g in np.unique(codes):
            rows = np.flatnonzero(codes == g)
            y_perm[rows] = y[rng.permutation(rows)]
        hits = 0
        used = 0
        for i in range(n_seeds):
            train, test = _split(len(y), rng, clusters, time_order, i, n_seeds)
            if train.sum() < 2 * n_act or test.sum() < 2 * n_act:
                continue
            est = _evaluate_fold(
                learner, y_perm, a, s, pi, n_act, train, test, clusters, baseline
            )
            hits += est.excludes_zero
            used += 1
        out[b] = hits / max(used, 1)
    return out
