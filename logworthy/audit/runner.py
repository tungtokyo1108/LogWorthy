"""The one-call entry point: `audit`.

This assembles the pieces in the order the routing rule requires and returns a
verdict. It is deliberately a thin layer: everything it does can be done by
calling the components directly, which is what you want when a step needs
customising (a different propensity model, a bespoke ladder, a learner's own
certification procedure).

The default is cautious. With no ladder supplied an observational log is treated
as having an unsettled state specification, and with no certification result no
verdict above Tier 1 is issued. Both can be overridden, but silence is never
read as evidence.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.coupling import design_effect, kish_n_eff, wild_cluster_bootstrap
from ..core.detectability import (
    analytic_threshold,
    calibrate_threshold,
    three_branch,
)
from ..verdict.verdict import Verdict, route
from .gates import GateResult, effective_overlap

__all__ = ["audit", "build_design"]


def build_design(
    state: np.ndarray, actions: np.ndarray, *, interactions: bool = False
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build ``(m0, m1, df1)`` from a state matrix and an action vector.

    The action block is dummy-coded against the first level, so ``df1`` is
    ``K - 1`` for a main-effect contrast and ``(K - 1) * (1 + d)`` when
    interactions with the ``d`` state columns are included.

    Args:
        state: State matrix, shape ``(n, d)``. Should not contain an intercept;
            one is added.
        actions: Integer action index per row, values ``0..K-1``.
        interactions: If True, append action-by-state interactions, giving the
            contrast the paper calls the interacted one.

    Returns:
        ``(m0, m1, df1)``.

    Raises:
        ValueError: If fewer than two distinct actions are present.
    """
    state = np.atleast_2d(np.asarray(state, dtype=float))
    if state.shape[0] != len(actions):
        state = state.T
    a = np.asarray(actions)
    levels = np.unique(a)
    n_actions = len(levels)
    if n_actions < 2:
        raise ValueError(f"need at least 2 action levels, got {n_actions}")

    n = len(a)
    block = np.zeros((n, n_actions - 1))
    for j, lv in enumerate(levels[1:]):
        block[:, j] = (a == lv).astype(float)

    m0 = np.hstack([np.ones((n, 1)), state])
    parts = [m0, block]
    if interactions:
        parts.extend(state * block[:, [j]] for j in range(n_actions - 1))
    m1 = np.hstack(parts)
    return m0, m1, m1.shape[1] - m0.shape[1]


def audit(
    y: np.ndarray,
    actions: np.ndarray,
    state: np.ndarray,
    clusters: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    interactions: bool = False,
    gates: Optional[List[GateResult]] = None,
    ladder=None,
    learner=None,
    certified: Optional[bool] = None,
    propensities: Optional[np.ndarray] = None,
    n_seeds: int = 30,
    q: float = 0.5,
    time_order: Optional[np.ndarray] = None,
    npv_positive: Optional[bool] = None,
    sequential_responsive: Optional[bool] = None,
    policy_class: str = "fixed rules",
    threshold_method: str = "analytic",
    threshold_grid=None,
    n_boot: int = 999,
    alpha: float = 0.05,
    weights_type: str = "rademacher",
    seed: int = 0,
) -> Verdict:
    """Audit whether a log can support a policy, and return a verdict.

    Args:
        y: Outcome, shape ``(n,)``.
        actions: Logged action index, shape ``(n,)``, values ``0..K-1``.
        state: Pre-decision state, shape ``(n, d)``. Must contain only variables
            known *before* the action; a post-decision variable here invalidates
            the audit (that is condition C2, which this function cannot check
            for you).
        clusters: Cluster label per row. This is the unit of independence —
            customer state, user stratum, store. Getting it wrong is the single
            most common way to overstate precision.
        weights: Importance weights ``rho(a) / pi_b(a | s)``. Defaults to
            uniform, which is correct only for a randomised log.
        interactions: Audit the interacted contrast rather than main effects.
        gates: Stage-0 gate results from :mod:`logworthy.audit.gates`. C4b is
            computed automatically and appended.
        ladder: A :class:`~logworthy.audit.ladder.LadderResult`, when one was
            run. Omitting it on an observational log leaves the causal reading
            uncorroborated, but does not by itself add ``C?``: pass the ladder
            if you want that question answered.
        learner: The deployed learner to certify, satisfying
            :class:`~logworthy.rl.Learner`. When given, Stage 2 runs here and
            ``certified`` is derived from it; the learner must be fixed in
            advance, since choosing among several is selection. For
            ``policy_class="contextual"`` the increment rule is applied
            automatically, certifying the paired uplift over the best fixed
            rule rather than over the logging policy.
        certified: Whether the deployed learner certified. Pass this instead of
            ``learner`` when certification was run elsewhere. ``None`` means
            indeterminate and caps the verdict at Tier 1.
        propensities: ``pi_b(a | s)`` for the logged action, required when
            ``learner`` is given. Must be fitted on data disjoint from the
            evaluation fold, or the certificate is not honest.
        n_seeds: Honest splits for certification.
        q: Proportion of certifying seeds required.
        time_order: Timestamps, for a rolling-origin split on a time-ordered log.
        npv_positive: Whether the payoff ceiling clears build cost.
        sequential_responsive: See :func:`logworthy.verdict.route`.
        policy_class: Name carried into the verdict, e.g. ``"contextual"``.
        threshold_method: ``"analytic"`` (fast, assumes the information count
            behaves like independent rows) or ``"calibrated"`` (Algorithm 2, run
            on the audit's own bootstrap). Use ``"calibrated"`` for anything
            you intend to publish.
        threshold_grid: Candidate couplings for the calibrated method.
        n_boot: Bootstrap replications for the coupling interval.
        alpha: One minus coverage.
        weights_type: ``"rademacher"`` or ``"webb"``. Prefer Webb below roughly
            12 clusters.
        seed: Seed for reproducibility.

    Returns:
        A :class:`~logworthy.verdict.Verdict`.

    Raises:
        ValueError: If ``threshold_method`` is ``"calibrated"`` without a grid,
            or inputs disagree in length.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    if not (len(actions) == len(clusters) == len(w) == n):
        raise ValueError(
            "y, actions, clusters and weights must have the same length; got "
            f"{n}, {len(actions)}, {len(clusters)}, {len(w)}"
        )

    m0, m1, df1 = build_design(state, actions, interactions=interactions)

    coupling = wild_cluster_bootstrap(
        m0, m1, y, w, clusters, df1,
        n_boot=n_boot, alpha=alpha, weights_type=weights_type, seed=seed,
    )

    if threshold_method == "calibrated":
        if threshold_grid is None:
            raise ValueError(
                "threshold_method='calibrated' needs threshold_grid: a list of "
                "candidate couplings bracketing the threshold"
            )
        thr = calibrate_threshold(
            m0, m1, y, w, clusters, df1,
            grid=threshold_grid, alpha=alpha,
            weights_type=weights_type, seed=seed,
        )
    elif threshold_method == "analytic":
        deff = design_effect(m0, m1, y, w, clusters, df1)
        thr = analytic_threshold(kish_n_eff(w) / deff, df1, alpha=alpha)
    else:
        raise ValueError(
            f"unknown threshold_method {threshold_method!r}; "
            "use 'analytic' or 'calibrated'"
        )

    all_gates = list(gates or [])
    if not any(g.code == "C4b" for g in all_gates):
        all_gates.append(effective_overlap(w))

    certification = None
    if learner is not None:
        if certified is not None:
            raise ValueError(
                "pass either `learner` (to run Stage 2 here) or `certified` "
                "(when it was run elsewhere), not both"
            )
        if propensities is None:
            raise ValueError(
                "certifying a learner needs `propensities`: pi_b(a | s) for the "
                "logged action, fitted on data disjoint from the evaluation fold"
            )
        from ..rl.protocol import PolicyClass, certify, certify_increment

        run = (
            certify_increment
            if policy_class == PolicyClass.CONTEXTUAL
            else certify
        )
        certification = run(
            learner, y, actions, state, propensities,
            n_seeds=n_seeds, q=q, clusters=clusters,
            time_order=time_order, seed=seed,
        )
        certified = certification.certified

    verdict = route(
        gates=all_gates,
        ladder=ladder,
        branch=three_branch(coupling, thr),
        certified=certified,
        npv_positive=npv_positive,
        sequential_responsive=sequential_responsive,
        policy_class=policy_class,
        coupling=coupling,
        threshold=thr,
    )
    if certification is not None:
        # Carry the certification into the verdict so a reader can see the
        # seeds behind it, including when the coherence rule overrode it.
        object.__setattr__(verdict, "_meta", {**verdict._meta,
                                              "certification": certification})
    return verdict
