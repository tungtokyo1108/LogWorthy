"""LogWorthy on the two public logistics logs the paper audits.

    python examples/quickstart.py

Everything here runs on real data bundled with the repository: 97,917 Olist
orders and 180,519 DataCo orders. Needs pandas and scikit-learn:

    pip install 'logworthy[examples]'
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

# Run straight from a clone, without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logworthy import (
    ConservativeQ,
    audit,
    audit_table,
    build_design,
    certify,
    certify_increment,
    definitional_target,
    effective_overlap,
    overlap,
    run_ladder,
    summary,
)
from logworthy.datasets import load_dataco, load_olist

warnings.filterwarnings("ignore")
RULE = "=" * 78


_PROPENSITY_CACHE: dict = {}


def fit_propensities(state, actions, n_actions, seed=0, key=None):
    """Cross-fitted propensities, so no row's weight is fitted on itself.

    Cached per dataset: this script audits the same logs from several angles,
    and refitting the propensity model each time would dominate its runtime
    without changing a single number.
    """
    if key is not None and key in _PROPENSITY_CACHE:
        return _PROPENSITY_CACHE[key]

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    model = HistGradientBoostingClassifier(max_iter=60, random_state=seed)
    probabilities = cross_val_predict(
        model, state, actions, cv=3, method="predict_proba"
    )
    result = (probabilities[np.arange(len(actions)), actions], probabilities)
    if key is not None:
        _PROPENSITY_CACHE[key] = result
    return result


def section_1_definitional_target():
    """DataCo's on-time reward: refused before any coupling is measured."""
    print(RULE)
    print("1. A REWARD THAT CANNOT BE AUDITED  (DataCo, on-time delivery)")
    print(RULE)
    log = load_dataco(reward="on_time")
    names = list(log.feature_names)
    scheduled = log.state[:, names.index("days_scheduled")]
    realised = log.state[:, names.index("days_real")]

    # On-time is defined as realised <= scheduled, and the scheduled days are
    # set with the shipping mode. So the "outcome" is a relabelling of the
    # decision, not a response to it.
    recovery = float(((realised <= scheduled).astype(float) == log.y).mean())
    gate = definitional_target(recovery_rate=recovery)

    verdict = audit(
        log.y, log.actions, log.state, log.clusters,
        gates=[gate], certified=True, seed=0, n_boot=199,
    )
    print(f"  {log}")
    print(f"  on-time is recovered as (realised <= scheduled) in {recovery:.2%} of rows")
    print(f"  VERDICT: {verdict.label}")
    print()
    print("  A coupling measured here would be real, large, and worth nothing:")
    print("  changing the shipping mode changes the schedule the outcome is")
    print("  measured against. No sample size fixes that, so the audit stops.")
    print()


def section_2_margin_reward():
    """DataCo's margin reward: a real question, refused for power."""
    print(RULE)
    print("2. A REAL QUESTION, TOO FEW CLUSTERS  (DataCo, profit per order)")
    print(RULE)
    log = load_dataco(reward="margin")
    propensities, _ = fit_propensities(
        log.state, log.actions, log.n_actions, key="dataco_margin"
    )
    weights = (1 / log.n_actions) / propensities

    verdict = audit(
        log.y, log.actions, log.state, log.clusters,
        weights=weights,
        gates=[overlap(propensities, trim=0.01), effective_overlap(weights)],
        certified=True, seed=0, n_boot=399,
    )
    print(summary(verdict))
    print()


def section_3_olist_both_windows():
    """Olist, on both registered windows."""
    print(RULE)
    print("3. THE SAME LOG, TWO WINDOWS  (Olist, review score)")
    print(RULE)
    verdicts = []
    for window in ("dev", "holdout"):
        log = load_olist(window=window)
        propensities, _ = fit_propensities(
            log.state, log.actions, log.n_actions, key=f"olist_{window}"
        )
        weights = (1 / log.n_actions) / propensities
        verdict = audit(
            log.y, log.actions, log.state, log.clusters,
            weights=weights, certified=True, seed=0, n_boot=399,
            policy_class=f"Olist {window}",
        )
        verdicts.append(verdict)
        print(f"  {log}")
        print(f"    coupling  [{verdict.coupling.lower:.3e}, {verdict.coupling.upper:.3e}]")
        print(f"    threshold [{verdict.threshold.low:.3e}, {verdict.threshold.high:.3e}]")
        print(f"    verdict   {verdict.label}")
    print()
    print(audit_table(verdicts))
    print()
    print("  97,917 orders, but only 27 customer states. Precision is set by")
    print("  the clusters, not the rows.")
    print()


def section_4_certification_and_increment():
    """What a learner certifies, and what that is worth."""
    print(RULE)
    print("4. CERTIFICATION, AND THE INCREMENT RULE  (Olist dev)")
    print(RULE)
    log = load_olist(window="dev")
    propensities, _ = fit_propensities(
        log.state, log.actions, log.n_actions, key="olist_dev"
    )

    fixed = certify(
        ConservativeQ(), log.y, log.actions, log.state, propensities,
        n_seeds=20, clusters=log.clusters, seed=0,
    )
    increment = certify_increment(
        ConservativeQ(), log.y, log.actions, log.state, propensities,
        n_seeds=20, clusters=log.clusters, seed=0,
    )
    print(f"  fixed rules : {fixed.seeds_certified:2d}/{fixed.n_seeds} seeds, "
          f"uplift {fixed.mean_uplift:+.4f} review points")
    print(f"  increment   : {increment.seeds_certified:2d}/{increment.n_seeds} seeds, "
          f"uplift {increment.mean_uplift:+.4f} review points")
    print()
    print("  The increment is what a contextual verdict must rest on. Since the")
    print("  fixed-rule class sits inside the contextual one, beating the")
    print("  logging policy is implied by the fixed-rule result and adds nothing.")
    print()

    verdict = audit(
        log.y, log.actions, log.state, log.clusters,
        weights=(1 / log.n_actions) / propensities,
        learner=ConservativeQ(), propensities=propensities,
        n_seeds=20, seed=0, n_boot=399,
    )
    certification = verdict._meta["certification"]
    print(f"  audit verdict: {verdict.label} "
          f"(learner certified {certification.seeds_certified}/{certification.n_seeds})")
    if certification.certified and verdict.tier == 0:
        print("  The certificate is reported as unsupported: the coupling does")
        print("  not clear what 27 clusters can detect.")
    print()


def section_5_ladder():
    """Is the Olist coupling confounded by geography?"""
    print(RULE)
    print("5. THE SATURATION LADDER  (Olist dev: is geography behind it?)")
    print(RULE)
    log = load_olist(window="dev")
    propensities, _ = fit_propensities(
        log.state, log.actions, log.n_actions, key="olist_dev"
    )
    weights = (1 / log.n_actions) / propensities

    # Enrich the state with seller geography and re-measure.
    base = log.state
    seller_state = np.unique(log.clusters, return_inverse=True)[1].astype(float)
    enriched = np.column_stack([base, seller_state])

    m0, m1_base, df1 = build_design(base, log.actions)
    _, m1_rich, _ = build_design(enriched, log.actions)

    delta_prop, delta_se = _propensity_improvement(
        base, enriched, log.actions, log.n_actions, log.clusters
    )
    ladder = run_ladder(
        [("L0 base", m1_base), ("L1 +geography", m1_rich)],
        m0, log.y, weights, log.clusters, df1,
        delta_prop=delta_prop, delta_prop_se=delta_se, n_boot=199, seed=0,
    )
    print(f"  ratio top/base = {ladder.ratio:.3f} "
          f"[{ladder.ratio_ci[0]:.3f}, {ladder.ratio_ci[1]:.3f}]")
    print(f"  delta_prop     = {delta_prop:+.4f} nats/row (SE {delta_se:.4f})")
    print(f"  outcome: {ladder.outcome}")
    print(f"    {ladder.reason}")
    print()
    print("  Collapse needs two things: the ratio must decline AND the enrichment")
    print("  covariates must predict the action. One without the other licenses")
    print("  no causal reading.")
    print()


def _propensity_improvement(base, enriched, actions, n_actions, clusters):
    """Held-out log-loss improvement from the enrichment covariates."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    from logworthy import propensity_informativeness

    def losses(features):
        probabilities = cross_val_predict(
            HistGradientBoostingClassifier(max_iter=50, random_state=0),
            features, actions, cv=3, method="predict_proba",
        )
        taken = probabilities[np.arange(len(actions)), actions]
        return -np.log(np.maximum(taken, 1e-12))

    _, delta, se = propensity_informativeness(
        losses(base), losses(enriched), clusters, n_boot=199
    )
    return delta, se


def main():
    section_1_definitional_target()
    section_2_margin_reward()
    section_3_olist_both_windows()
    section_4_certification_and_increment()
    section_5_ladder()
    print(RULE)
    print("Both logs are refused, for different reasons. That is the point:")
    print("the audit says which question the data cannot answer, and what")
    print("would have to change.")
    print(RULE)


if __name__ == "__main__":
    main()
