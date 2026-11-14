"""Tests for the RL layer: estimators, learners, certification, C6.

The important tests here are the ones that would fail if a tempting shortcut
were taken: the influence function without its baseline term, the increment rule
replaced by a comparison against the logging policy, and C6 decided on the
sequential share instead of the downstream coupling.
"""

from __future__ import annotations

import numpy as np
import pytest

from logworthy.rl import (
    ConservativeQ,
    GradientBoostedQ,
    GreedyDR,
    Learner,
    available_learners,
    certify,
    certify_increment,
    direct_method,
    doubly_robust,
    inverse_propensity,
    paired_increment,
)
from logworthy.rl.sequential import decompose_coupling, fitted_q_evaluation, run_c6


def make_bandit(effect=0.0, contextual=0.0, n=4000, n_actions=3, seed=0):
    """A logged contextual bandit with a known structure."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 2))
    logits = np.column_stack(
        [np.zeros(n), 0.3 * state[:, 0], -0.2 * state[:, 1]]
    )
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    actions = np.array([rng.choice(n_actions, p=row) for row in p])
    propensities = p[np.arange(n), actions]
    high = state[:, 0] > 0
    y = (
        state @ [0.5, -0.3]
        + effect * (actions == 1)
        + contextual * ((actions == 1) & high)
        + contextual * ((actions == 2) & ~high)
        + rng.normal(size=n)
    )
    clusters = rng.integers(0, 20, n)
    return dict(
        y=y, actions=actions, state=state, propensities=propensities,
        clusters=clusters, n_actions=n_actions,
    )


class TestEstimators:
    def test_influence_function_includes_baseline(self):
        """The decisive test: the baseline must be inside phi, not subtracted after.

        If ``- y`` were applied to the mean rather than to each row, the point
        estimate would be unchanged but the standard error would be computed
        from the wrong quantity. We check that the influence function really is
        the per-row difference.
        """
        d = make_bandit(effect=0.4, seed=1)
        n = len(d["y"])
        q = np.tile(d["y"].mean(), (n, d["n_actions"]))
        target = np.zeros(n, dtype=int)
        est = doubly_robust(d["y"], d["actions"], d["propensities"], q, target)
        rows = np.arange(n)
        expected = (
            q[rows, target]
            + (d["actions"] == target) / np.maximum(d["propensities"], 1e-6)
            * (d["y"] - q[rows, d["actions"]])
            - d["y"]
        )
        assert np.allclose(est.influence, expected)
        assert est.estimate == pytest.approx(float(expected.mean()))
        assert est.se == pytest.approx(float(expected.std(ddof=1) / np.sqrt(n)))

    def test_se_differs_from_the_naive_version(self):
        """Dropping the baseline from phi changes the SE, which is the bug."""
        d = make_bandit(effect=0.4, seed=2)
        n = len(d["y"])
        rng = np.random.default_rng(0)
        q = rng.normal(size=(n, d["n_actions"]))
        target = q.argmax(axis=1)
        est = doubly_robust(d["y"], d["actions"], d["propensities"], q, target)
        rows = np.arange(n)
        naive = (
            q[rows, target]
            + (d["actions"] == target) / np.maximum(d["propensities"], 1e-6)
            * (d["y"] - q[rows, d["actions"]])
        )
        naive_se = float(naive.std(ddof=1) / np.sqrt(n))
        assert est.se != pytest.approx(naive_se, rel=1e-3)

    def test_cluster_se_exceeds_iid_se_under_clustering(self):
        d = make_bandit(effect=0.3, seed=3)
        n = len(d["y"])
        # Make the influence function strongly cluster-dependent.
        q = np.zeros((n, d["n_actions"]))
        est = doubly_robust(
            d["y"], d["actions"], d["propensities"], q, np.zeros(n, dtype=int)
        )
        clustered = est.cluster_se(d["clusters"])
        assert clustered > 0

    def test_estimators_agree_on_a_randomised_log(self):
        """DR, IPS and DM should broadly agree when the log is randomised."""
        rng = np.random.default_rng(4)
        n, k = 6000, 3
        state = rng.normal(size=(n, 1))
        actions = rng.integers(0, k, n)
        pi = np.full(n, 1 / k)
        y = 0.5 * (actions == 1) + rng.normal(size=n)
        q = np.column_stack([np.full(n, y[actions == j].mean()) for j in range(k)])
        target = np.ones(n, dtype=int)
        dr = doubly_robust(y, actions, pi, q, target)
        ips = inverse_propensity(y, actions, pi, target)
        dm = direct_method(q, target, baseline=y)
        assert abs(dr.estimate - dm.estimate) < 0.1
        assert abs(dr.estimate - ips.estimate) < 0.2

    def test_paired_increment_subtracts_influence_functions(self):
        d = make_bandit(effect=0.3, seed=5)
        n = len(d["y"])
        q = np.zeros((n, d["n_actions"]))
        a_est = doubly_robust(
            d["y"], d["actions"], d["propensities"], q,
            np.zeros(n, dtype=int), baseline=np.zeros(n),
        )
        b_est = doubly_robust(
            d["y"], d["actions"], d["propensities"], q,
            np.ones(n, dtype=int), baseline=np.zeros(n),
        )
        inc = paired_increment(a_est, b_est)
        assert np.allclose(inc.influence, a_est.influence - b_est.influence)

    def test_paired_increment_rejects_different_rows(self):
        d = make_bandit(seed=6)
        n = len(d["y"])
        q = np.zeros((n, d["n_actions"]))
        full = doubly_robust(
            d["y"], d["actions"], d["propensities"], q, np.zeros(n, dtype=int)
        )
        half = doubly_robust(
            d["y"][: n // 2], d["actions"][: n // 2], d["propensities"][: n // 2],
            q[: n // 2], np.zeros(n // 2, dtype=int),
        )
        with pytest.raises(ValueError, match="same rows"):
            paired_increment(full, half)

    def test_q_values_shape_is_checked(self):
        d = make_bandit(seed=7)
        n = len(d["y"])
        with pytest.raises(ValueError, match=r"shape \(n, K\)"):
            doubly_robust(
                d["y"], d["actions"], d["propensities"],
                np.zeros(n), np.zeros(n, dtype=int),
            )


class TestLearners:
    @pytest.mark.parametrize("cls", [ConservativeQ, GreedyDR])
    def test_satisfies_the_interface(self, cls):
        assert isinstance(cls(), Learner)

    @pytest.mark.parametrize("cls", [ConservativeQ, GreedyDR])
    def test_finds_the_best_action(self, cls):
        d = make_bandit(effect=0.8, seed=8)
        model = cls().fit(
            d["state"], d["actions"], d["y"], n_actions=d["n_actions"]
        )
        q = model.q_values(d["state"])
        assert q.shape == (len(d["y"]), d["n_actions"])
        assert q.argmax(axis=1).mean() > 0.5  # mostly picks action 1

    def test_conservative_shrinks_more_than_greedy(self):
        """Conservatism is the point: values must be pulled toward the mean."""
        d = make_bandit(effect=0.8, n=600, seed=9)
        cons = ConservativeQ(alpha=200.0).fit(
            d["state"], d["actions"], d["y"], n_actions=d["n_actions"]
        )
        greedy = GreedyDR().fit(
            d["state"], d["actions"], d["y"], n_actions=d["n_actions"]
        )
        spread_c = cons.q_values(d["state"]).std()
        spread_g = greedy.q_values(d["state"]).std()
        assert spread_c < spread_g

    def test_unfitted_learner_raises(self):
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            ConservativeQ().q_values(np.zeros((5, 2)))

    def test_one_dimensional_state_accepted(self):
        d = make_bandit(seed=10)
        model = ConservativeQ().fit(
            d["state"][:, 0], d["actions"], d["y"], n_actions=d["n_actions"]
        )
        assert model.q_values(d["state"][:, 0]).shape == (len(d["y"]), 3)

    def test_available_learners_reports_status(self):
        status = available_learners()
        assert status["ConservativeQ"] and status["GreedyDR"]
        assert set(status) >= {"ConservativeQ", "GreedyDR", "GradientBoostedQ",
                               "DiscreteCQL"}


class TestCertification:
    def test_decoupled_log_is_not_certified(self):
        d = make_bandit(effect=0.0, seed=11)
        result = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=10, clusters=d["clusters"], seed=0,
        )
        assert result.certified is False
        assert result.proportion <= 0.5

    def test_coupled_log_is_certified(self):
        d = make_bandit(effect=0.6, seed=12)
        result = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=10, clusters=d["clusters"], seed=0,
        )
        assert result.certified is True
        assert result.mean_uplift > 0

    def test_increment_is_zero_when_value_is_purely_fixed(self):
        """A log whose value is entirely non-contextual must not license Pi.

        This is the increment rule. Certifying the contextual policy against the
        logging policy would pass here, which is exactly why that comparison is
        not the one the verdict uses.
        """
        d = make_bandit(effect=0.5, contextual=0.0, seed=13)
        fixed = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=10, clusters=d["clusters"], seed=0,
        )
        increment = certify_increment(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=10, clusters=d["clusters"], seed=0,
        )
        assert fixed.certified is True
        assert increment.certified is False

    def test_increment_fires_when_personalisation_reorders_actions(self):
        d = make_bandit(effect=0.1, contextual=0.9, n=6000, seed=14)
        increment = certify_increment(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=10, clusters=d["clusters"], seed=0,
        )
        assert increment.certified is True

    def test_seeds_are_independent_of_each_other(self):
        """A stateful learner must not carry fitted state across seeds."""
        d = make_bandit(effect=0.4, seed=15)
        shared = ConservativeQ()
        first = certify(
            shared, d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=6, clusters=d["clusters"], seed=0,
        )
        second = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=6, clusters=d["clusters"], seed=0,
        )
        assert first.seeds_certified == second.seeds_certified

    def test_permutation_null_runs(self):
        d = make_bandit(effect=0.0, n=1500, seed=16)
        result = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=4, clusters=d["clusters"], n_permutations=3, seed=0,
        )
        assert result.null_proportion is not None
        assert result.p_value is not None

    def test_cluster_split_keeps_clusters_whole(self):
        d = make_bandit(effect=0.3, seed=17)
        result = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=3, clusters=d["clusters"], seed=0,
        )
        assert result.n_seeds == 3

    def test_rolling_origin_split_accepted(self):
        d = make_bandit(effect=0.4, seed=18)
        times = np.arange(len(d["y"]))
        result = certify(
            ConservativeQ(), d["y"], d["actions"], d["state"], d["propensities"],
            n_seeds=4, time_order=times, seed=0,
        )
        assert result.n_seeds == 4

    def test_bad_learner_rejected(self):
        d = make_bandit(seed=19)
        with pytest.raises(ValueError, match="Learner interface"):
            certify(
                object(), d["y"], d["actions"], d["state"], d["propensities"],
                n_seeds=2,
            )

    def test_length_mismatch_rejected(self):
        d = make_bandit(seed=20)
        with pytest.raises(ValueError, match="agree in length"):
            certify(
                ConservativeQ(), d["y"][:-1], d["actions"], d["state"],
                d["propensities"], n_seeds=2,
            )


def make_mdp(immediate=0.0, downstream=0.0, n=6000, n_actions=3, seed=0):
    """Transitions where the reward depends on the state, so states have value."""
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 1))
    actions = rng.integers(0, n_actions, n)
    rewards = (
        2.0 * state[:, 0] + immediate * (actions == 1) + rng.normal(0, 0.4, n)
    )
    next_state = (
        state + downstream * (actions == 1).reshape(-1, 1)
        + rng.normal(0, 0.3, size=(n, 1))
    )
    return dict(
        state=state, actions=actions, rewards=rewards, next_state=next_state,
        terminals=np.zeros(n), target_policy=np.ones(n, dtype=int),
        n_actions=n_actions,
    )


class TestSequential:
    def test_immediate_only_has_small_continuation(self):
        d = make_mdp(immediate=1.0, downstream=0.0, seed=21)
        res = run_c6(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            d["terminals"], d["target_policy"], n_actions=3, gamma=0.9, n_bins=6,
        )
        assert res.d1 > res.d_cont

    def test_downstream_only_is_detected(self):
        d = make_mdp(immediate=0.0, downstream=1.5, seed=21)
        res = run_c6(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            d["terminals"], d["target_policy"], n_actions=3, gamma=0.9, n_bins=6,
        )
        assert res.d_cont > res.d1
        assert res.chi_seq > 0.5

    def test_negative_share_with_positive_continuation(self):
        """The property the paper insists on: chi_seq is a two-sided diagnostic.

        A negative covariance between immediate reward and continuation value
        can push chi_seq below zero while downstream responsiveness is strong.
        C6 must therefore be decided on D_cont, not on chi_seq.
        """
        rng = np.random.default_rng(7)
        n, k = 10000, 3
        state = rng.normal(size=(n, 1))
        actions = rng.integers(0, k, n)
        rewards = 1.5 * state[:, 0] + 1.2 * (actions == 1) + rng.normal(0, 0.4, n)
        next_state = (
            state - 0.8 * (actions == 1).reshape(-1, 1)
            + rng.normal(0, 0.3, size=(n, 1))
        )
        res = run_c6(
            state, actions, rewards, next_state, np.zeros(n),
            np.ones(n, dtype=int), n_actions=k, gamma=0.9, n_bins=6,
        )
        assert res.chi_seq < 0, "fixture should produce a negative sequential share"
        assert res.d_cont > 1e-6, "yet downstream responsiveness is real"
        assert res.covariance_term < 0

    def test_minkowski_bounds_hold(self):
        for immediate, downstream in [(1.0, 0.0), (0.0, 1.5), (1.0, 1.5), (1.0, -1.5)]:
            d = make_mdp(immediate=immediate, downstream=downstream, seed=22)
            res = run_c6(
                d["state"], d["actions"], d["rewards"], d["next_state"],
                d["terminals"], d["target_policy"], n_actions=3, gamma=0.9,
                n_bins=6,
            )
            assert res.triangle_holds, (
                f"Proposition 1 bounds violated at ({immediate}, {downstream})"
            )

    def test_threshold_decides_responsiveness(self):
        d = make_mdp(immediate=0.0, downstream=1.5, seed=23)
        res = run_c6(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            d["terminals"], d["target_policy"], n_actions=3, gamma=0.9,
            n_bins=6, d_cont_threshold=1e-4, n_boot=49,
        )
        assert res.responsive is True

    def test_undecided_without_threshold(self):
        d = make_mdp(downstream=1.0, seed=24)
        res = run_c6(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            d["terminals"], d["target_policy"], n_actions=3, n_bins=6,
        )
        assert res.responsive is None

    def test_terminals_carry_no_continuation(self):
        d = make_mdp(downstream=1.5, seed=25)
        n = len(d["rewards"])
        all_terminal = run_c6(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            np.ones(n), d["target_policy"], n_actions=3, gamma=0.9, n_bins=6,
        )
        assert all_terminal.d_cont < 1e-6

    def test_length_mismatch_rejected(self):
        d = make_mdp(seed=26)
        with pytest.raises(ValueError, match="same number of rows"):
            fitted_q_evaluation(
                d["state"], d["actions"], d["rewards"][:-1], d["next_state"],
                d["terminals"], d["target_policy"],
            )

    def test_decompose_accepts_a_reference_distribution(self):
        d = make_mdp(immediate=1.0, seed=27)
        q, cell, _, reward = fitted_q_evaluation(
            d["state"], d["actions"], d["rewards"], d["next_state"],
            d["terminals"], d["target_policy"], n_actions=3, n_bins=6,
        )
        skewed = decompose_coupling(q, reward, cell, rho=np.array([0.8, 0.1, 0.1]))
        uniform = decompose_coupling(q, reward, cell)
        assert skewed.d1 != pytest.approx(uniform.d1)


class TestGradientBoosted:
    def test_runs_when_sklearn_present(self):
        pytest.importorskip("sklearn")
        d = make_bandit(effect=0.6, n=1500, seed=28)
        model = GradientBoostedQ(max_iter=20).fit(
            d["state"], d["actions"], d["y"], n_actions=d["n_actions"]
        )
        assert model.q_values(d["state"]).shape == (1500, 3)


class TestCoherenceRule:
    """A certificate the coupling test cannot support must not license.

    This is the paper's central empirical claim, reproduced on synthetic data:
    a learner can certify an improvement on a log whose coupling lies below what
    the design can detect. The audit routes such a log by its detectability
    branch and reports the certificate as unsupported, because selection on
    fitting error is the more plausible explanation.
    """

    def _niche_log(self, effect, n=6000, n_clusters=27, n_actions=3, seed=5):
        rng = np.random.default_rng(seed)
        clusters = rng.integers(0, n_clusters, n)
        state = rng.normal(size=(n, 2))
        actions = rng.integers(0, n_actions, n)
        propensities = np.full(n, 1 / n_actions)
        niche = state[:, 0] > 1.15
        y = (
            state @ [0.5, -0.3]
            + rng.normal(0, 1.0, n_clusters)[clusters]
            + effect * ((actions == 1) & niche)
            + rng.normal(size=n)
        )
        return y, actions, state, clusters, propensities

    def test_certificate_without_detectable_coupling_is_refused(self):
        from logworthy import audit

        y, actions, state, clusters, pi = self._niche_log(effect=1.1)
        verdict = audit(
            y, actions, state, clusters,
            learner=GreedyDR(), propensities=pi, n_seeds=30, seed=0,
        )
        certification = verdict._meta["certification"]
        assert certification.seeds_certified > 0, "fixture should produce a certificate"
        assert verdict.coupling.upper < verdict.threshold.low
        assert verdict.tier == 0, "an unsupported certificate must not license"
        assert "P" in verdict.reasons

    def test_learner_and_certified_are_mutually_exclusive(self):
        from logworthy import audit

        y, actions, state, clusters, pi = self._niche_log(effect=0.5)
        with pytest.raises(ValueError, match="not both"):
            audit(
                y, actions, state, clusters,
                learner=GreedyDR(), propensities=pi, certified=True,
            )

    def test_learner_requires_propensities(self):
        from logworthy import audit

        y, actions, state, clusters, _ = self._niche_log(effect=0.5)
        with pytest.raises(ValueError, match="needs `propensities`"):
            audit(y, actions, state, clusters, learner=GreedyDR())
