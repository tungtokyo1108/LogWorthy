"""Tests for the statistical core.

These check properties that must hold for the audit to mean anything, not just
that the code runs: that the estimator is unbiased under the null, that the test
has its nominal size, that clusters are respected, and that two independent
routes to the threshold agree.
"""

from __future__ import annotations

import numpy as np
import pytest

from logworthy.core import (
    Branch,
    analytic_threshold,
    calibrate_threshold,
    design_effect,
    kish_n_eff,
    partial_omega2,
    three_branch,
    wild_cluster_bootstrap,
)


def make_log(n=3000, n_clusters=25, n_actions=4, effect=0.0, cluster_sd=1.0,
             action_cluster_dependence=False, seed=0):
    """A synthetic log with a known coupling."""
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, n_clusters, n)
    state = rng.normal(size=(n, 2))
    if action_cluster_dependence:
        bias = rng.dirichlet(np.ones(n_actions) * 0.5, n_clusters)
        actions = np.array([rng.choice(n_actions, p=bias[g]) for g in clusters])
    else:
        actions = rng.integers(0, n_actions, n)
    y = (
        state @ [0.5, -0.3]
        + rng.normal(0, cluster_sd, n_clusters)[clusters]
        + effect * actions
        + rng.normal(size=n)
    )
    block = np.zeros((n, n_actions - 1))
    for k in range(1, n_actions):
        block[:, k - 1] = (actions == k)
    m0 = np.hstack([np.ones((n, 1)), state])
    m1 = np.hstack([m0, block])
    return dict(y=y, actions=actions, state=state, clusters=clusters,
                m0=m0, m1=m1, df1=n_actions - 1, weights=np.ones(n))


class TestPartialOmega2:
    def test_zero_when_no_coupling(self):
        d = make_log(effect=0.0, seed=1)
        est = partial_omega2(d["m0"], d["m1"], d["y"], d["weights"], d["df1"])
        assert est < 5e-3, "a decoupled log should show near-zero coupling"

    def test_grows_with_effect(self):
        prev = -1.0
        for effect in [0.0, 0.1, 0.3, 0.6]:
            d = make_log(effect=effect, seed=2)
            est = partial_omega2(d["m0"], d["m1"], d["y"], d["weights"], d["df1"])
            assert est > prev, "coupling must increase with the implanted effect"
            prev = est

    def test_negative_estimate_visible_when_unclipped(self):
        """The bias correction may go negative; that is evidence for the null."""
        seen_negative = False
        for seed in range(30):
            d = make_log(n=400, effect=0.0, seed=seed)
            raw = partial_omega2(
                d["m0"], d["m1"], d["y"], d["weights"], d["df1"], clip=False
            )
            if raw < 0:
                seen_negative = True
                clipped = partial_omega2(
                    d["m0"], d["m1"], d["y"], d["weights"], d["df1"]
                )
                assert clipped == 0.0
                break
        assert seen_negative, "expected at least one negative estimate under the null"

    def test_weights_change_the_estimand(self):
        d = make_log(effect=0.3, seed=4)
        rng = np.random.default_rng(0)
        w = rng.uniform(0.2, 3.0, len(d["y"]))
        unweighted = partial_omega2(d["m0"], d["m1"], d["y"], d["weights"], d["df1"])
        weighted = partial_omega2(d["m0"], d["m1"], d["y"], w, d["df1"])
        assert weighted != pytest.approx(unweighted, rel=1e-6)


class TestWildClusterBootstrap:
    def test_interval_contains_zero_under_null(self):
        d = make_log(effect=0.0, seed=5)
        est = wild_cluster_bootstrap(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
            n_boot=199, seed=0,
        )
        assert est.lower == 0.0, "a decoupled log should not exclude zero"

    def test_detects_a_real_effect(self):
        d = make_log(effect=0.8, seed=6)
        est = wild_cluster_bootstrap(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
            n_boot=199, seed=0,
        )
        assert est.p_value < 0.05
        assert est.lower > 0

    def test_size_is_near_nominal(self):
        """The decisive property: a 5% test should reject about 5% of the time."""
        rejections = 0
        trials = 40
        for seed in range(trials):
            d = make_log(n=1200, effect=0.0, seed=100 + seed)
            est = wild_cluster_bootstrap(
                d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
                n_boot=99, seed=seed,
            )
            rejections += est.p_value < 0.05
        rate = rejections / trials
        assert rate <= 0.20, f"test is oversized: rejected {rate:.0%} under the null"

    def test_rejects_mismatched_lengths(self):
        d = make_log()
        with pytest.raises(ValueError, match="same length"):
            wild_cluster_bootstrap(
                d["m0"], d["m1"], d["y"][:-1], d["weights"], d["clusters"], d["df1"]
            )

    def test_rejects_single_cluster(self):
        d = make_log()
        with pytest.raises(ValueError, match="at least 2 clusters"):
            wild_cluster_bootstrap(
                d["m0"], d["m1"], d["y"], d["weights"],
                np.zeros(len(d["y"])), d["df1"],
            )

    def test_webb_weights_accepted(self):
        d = make_log(n_clusters=8, seed=7)
        est = wild_cluster_bootstrap(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
            n_boot=99, weights_type="webb", seed=0,
        )
        assert est.n_clusters == 8

    def test_unknown_weight_type_rejected(self):
        d = make_log()
        with pytest.raises(ValueError, match="unknown weights_type"):
            wild_cluster_bootstrap(
                d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
                weights_type="gaussian",
            )


class TestDesignEffect:
    def test_near_one_when_action_independent_of_cluster(self):
        """Clustering in the outcome does not inflate a block it does not touch."""
        d = make_log(cluster_sd=2.0, action_cluster_dependence=False, seed=8)
        deff = design_effect(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"]
        )
        assert 1.0 <= deff < 3.0

    def test_large_when_action_tracks_cluster(self):
        d = make_log(cluster_sd=2.0, action_cluster_dependence=True, seed=9)
        deff = design_effect(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"]
        )
        assert deff > 3.0, "a cluster-correlated action block must inflate the deff"

    def test_never_below_one(self):
        d = make_log(cluster_sd=0.0, seed=10)
        deff = design_effect(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"]
        )
        assert deff >= 1.0


class TestThresholds:
    def test_analytic_decreases_with_information(self):
        small = analytic_threshold(500, 3).eta2_min
        large = analytic_threshold(5000, 3).eta2_min
        assert large < small, "more information must lower the threshold"

    def test_analytic_increases_with_df1(self):
        narrow = analytic_threshold(5000, 3).eta2_min
        wide = analytic_threshold(5000, 30).eta2_min
        assert wide > narrow, "a wider contrast costs power"

    def test_analytic_rejects_impossible_information(self):
        with pytest.raises(ValueError, match="denominator degrees of freedom"):
            analytic_threshold(3, 10)

    def test_calibrated_agrees_with_analytic(self):
        """Two independent routes to the same quantity should roughly agree.

        The analytic threshold inverts a power function; the calibrated one
        simulates the audit's own bootstrap. Agreement is evidence that both
        implementations are right.
        """
        d = make_log(n=3000, n_clusters=20, action_cluster_dependence=True, seed=7)
        deff = design_effect(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"]
        )
        analytic = analytic_threshold(kish_n_eff(d["weights"]) / deff, d["df1"])
        grid = np.array([0.5, 0.75, 1.0, 1.5, 2.0, 3.0]) * analytic.eta2_min
        calibrated = calibrate_threshold(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
            grid=grid, n_sim=40, n_boot=39, seed=2,
        )
        ratio = calibrated.eta2_min / analytic.eta2_min
        assert 0.4 < ratio < 2.5, (
            f"calibrated {calibrated.eta2_min:.3e} and analytic "
            f"{analytic.eta2_min:.3e} disagree by {ratio:.1f}x"
        )

    def test_calibrated_reports_a_band(self):
        d = make_log(n=1500, n_clusters=15, seed=11)
        thr = calibrate_threshold(
            d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
            grid=[0.005, 0.02, 0.05, 0.12, 0.3], n_sim=30, n_boot=29, seed=1,
        )
        assert thr.low <= thr.eta2_min <= thr.high
        assert thr.method == "calibrated"

    def test_calibrated_rejects_short_grid(self):
        d = make_log(n=1000, seed=12)
        with pytest.raises(ValueError, match="Extend `grid`"):
            calibrate_threshold(
                d["m0"], d["m1"], d["y"], d["weights"], d["clusters"], d["df1"],
                grid=[1e-9, 2e-9], n_sim=10, n_boot=19, seed=0,
            )


class TestThreeBranch:
    class _Interval:
        def __init__(self, lower, upper):
            self.lower, self.upper = lower, upper

    def _threshold(self, low, high):
        from logworthy.core import Threshold

        return Threshold((low + high) / 2, (low, high), 0.8, 0.05, "test")

    def test_below(self):
        branch = three_branch(self._Interval(0.0, 0.001), self._threshold(0.002, 0.003))
        assert branch == Branch.BELOW

    def test_detectable(self):
        branch = three_branch(self._Interval(0.01, 0.02), self._threshold(0.002, 0.003))
        assert branch == Branch.DETECTABLE

    def test_indeterminate_when_overlapping(self):
        branch = three_branch(self._Interval(0.001, 0.01), self._threshold(0.002, 0.003))
        assert branch == Branch.INDETERMINATE

    def test_band_not_point_decides(self):
        """Touching the band is indeterminate, not detectable."""
        branch = three_branch(
            self._Interval(0.0025, 0.01), self._threshold(0.002, 0.003)
        )
        assert branch == Branch.INDETERMINATE


class TestKishNEff:
    def test_uniform_weights_give_n(self):
        assert kish_n_eff(np.ones(100)) == pytest.approx(100.0)

    def test_unequal_weights_reduce_it(self):
        w = np.array([1.0] * 99 + [100.0])
        assert kish_n_eff(w) < 100

    def test_empty_is_zero(self):
        assert kish_n_eff(np.array([])) == 0.0
