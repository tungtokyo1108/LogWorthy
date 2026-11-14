"""Tests for the gates, the ladder, and the routing rule.

The ladder tests are the important ones. Its collapse branch is the audit's only
causal diagnostic, and the property that makes it trustworthy is not that it
fires on confounding, but that it *refuses to fire* on variance absorption,
which looks identical on the ratio alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from logworthy import (
    Branch,
    LadderOutcome,
    Reason,
    Tier,
    audit,
    build_design,
    definitional_target,
    effective_overlap,
    overlap,
    propensity_calibration,
    propensity_informativeness,
    route,
    run_ladder,
)


class TestGates:
    def test_definitional_target_flags_identity(self):
        g = definitional_target(0.999)
        assert not g.passed and g.code == "C1"

    def test_definitional_target_passes_normal_outcome(self):
        assert definitional_target(0.42).passed

    def test_overlap_fails_when_mass_below_trim(self):
        pi = np.concatenate([np.full(80, 0.001), np.full(20, 0.5)])
        g = overlap(pi, trim=0.01)
        assert not g.passed and g.code == "C4a"

    def test_overlap_passes_with_support(self):
        pi = np.concatenate([np.full(2, 0.001), np.full(998, 0.4)])
        assert overlap(pi, trim=0.01).passed

    def test_effective_overlap_reports_ratio(self):
        g = effective_overlap(np.ones(500))
        assert g.passed and g.statistic == pytest.approx(1.0)

    def test_calibration_needs_full_distribution(self):
        with pytest.raises(ValueError, match=r"shape \(n, K\)"):
            propensity_calibration(np.full(100, 0.25), np.zeros(100, dtype=int))

    def test_calibration_passes_on_calibrated_model(self):
        rng = np.random.default_rng(0)
        n, k = 4000, 4
        p = np.full((n, k), 1 / k)
        a = rng.integers(0, k, n)
        g = propensity_calibration(p, a)
        assert g.passed, g.detail

    def test_calibration_fails_on_miscalibrated_model(self):
        rng = np.random.default_rng(0)
        n, k = 4000, 4
        p = np.zeros((n, k))
        p[:, 0] = 0.97
        p[:, 1:] = 0.01
        a = rng.integers(0, k, n)  # actions actually uniform
        g = propensity_calibration(p, a)
        assert not g.passed

    def test_informativeness_returns_tested_quantity(self):
        rng = np.random.default_rng(1)
        n = 3000
        clusters = rng.integers(0, 20, n)
        base = rng.normal(1.4, 0.2, n)
        rich = base - 0.05  # genuinely better
        gate, delta, se = propensity_informativeness(base, rich, clusters, n_boot=99)
        assert gate.passed and delta == pytest.approx(0.05, abs=1e-9) and se >= 0

    def test_informativeness_fails_when_no_improvement(self):
        rng = np.random.default_rng(2)
        n = 3000
        clusters = rng.integers(0, 20, n)
        base = rng.normal(1.4, 0.2, n)
        gate, delta, se = propensity_informativeness(
            base, base + 0.01, clusters, n_boot=99
        )
        assert not gate.passed and delta < 0


def _ladder_inputs(n=6000, n_clusters=25, confounded=True, seed=0):
    """Build a two-rung ladder where the truth is known.

    With ``confounded=True`` an unobserved U drives both the action and the
    outcome, and the proxy z stands in for U. With ``confounded=False`` the
    action is randomised, so z explains the outcome but predicts nothing about
    the action: a decline is then variance absorption, not confounding.
    """
    rng = np.random.default_rng(seed)
    clusters = rng.integers(0, n_clusters, n)
    u = rng.normal(size=n)
    z = u + rng.normal(0, 0.3, n)
    base_state = rng.normal(size=(n, 2))
    if confounded:
        p = 1 / (1 + np.exp(-u))
        actions = (rng.random(n) < p).astype(int) + (rng.random(n) < p * 0.4).astype(int)
        actions = np.clip(actions, 0, 2)
    else:
        actions = rng.integers(0, 3, n)
    y = 2.0 * u + rng.normal(size=n)  # outcome never responds to the action
    m0, m1_base, df1 = build_design(base_state, actions)
    _, m1_rich, _ = build_design(np.column_stack([base_state, z]), actions)
    return dict(
        designs=[("L0 base", m1_base), ("L1 +proxy", m1_rich)],
        m0=m0, y=y, weights=np.ones(n), clusters=clusters, df1=df1,
    )


class TestLadder:
    def test_collapse_fires_on_confounding(self):
        d = _ladder_inputs(confounded=True, seed=5)
        res = run_ladder(
            d["designs"], d["m0"], d["y"], d["weights"], d["clusters"], d["df1"],
            delta_prop=0.08, delta_prop_se=0.02, n_boot=99, seed=1,
        )
        assert res.outcome == LadderOutcome.COLLAPSE

    def test_collapse_blocked_without_propensity_evidence(self):
        """The decisive test: a decline alone must not license a causal claim.

        Same data that collapses in the test above — so the ratio does decline —
        but with enrichment covariates that do not predict the action. The ratio
        condition is met and the branch must still refuse to fire.
        """
        d = _ladder_inputs(confounded=True, seed=5)
        res = run_ladder(
            d["designs"], d["m0"], d["y"], d["weights"], d["clusters"], d["df1"],
            delta_prop=-0.001, delta_prop_se=0.02, n_boot=99, seed=1,
        )
        assert res.ratio_ci[1] < 0.5, "this fixture should show a declining ladder"
        assert res.outcome != LadderOutcome.COLLAPSE
        assert "do not predict the action" in res.reason

    def test_collapse_blocked_when_propensity_untested(self):
        d = _ladder_inputs(confounded=True, seed=5)
        res = run_ladder(
            d["designs"], d["m0"], d["y"], d["weights"], d["clusters"], d["df1"],
            delta_prop=0.08, delta_prop_se=None, n_boot=99, seed=1,
        )
        assert res.outcome == LadderOutcome.INDETERMINATE
        assert "no tested propensity improvement" in res.reason

    def test_point_estimate_alone_is_not_enough(self):
        """delta_prop > 0 as a point estimate, but not significantly so."""
        d = _ladder_inputs(confounded=True, seed=5)
        res = run_ladder(
            d["designs"], d["m0"], d["y"], d["weights"], d["clusters"], d["df1"],
            delta_prop=0.01, delta_prop_se=0.5, n_boot=99, seed=1,
        )
        assert res.outcome != LadderOutcome.COLLAPSE

    def test_requires_two_rungs(self):
        d = _ladder_inputs(seed=1)
        with pytest.raises(ValueError, match="at least 2 rungs"):
            run_ladder(
                d["designs"][:1], d["m0"], d["y"], d["weights"],
                d["clusters"], d["df1"],
            )

    def test_indeterminate_adds_c_query(self):
        d = _ladder_inputs(confounded=False, seed=3)
        res = run_ladder(
            d["designs"], d["m0"], d["y"], d["weights"], d["clusters"], d["df1"],
            delta_prop=-0.001, delta_prop_se=0.02, n_boot=99, seed=1,
        )
        if res.outcome == LadderOutcome.INDETERMINATE:
            assert res.adds_c_query


class TestRouting:
    def test_c1_failure_gives_g(self):
        v = route(gates=[definitional_target(0.999)], branch=Branch.DETECTABLE)
        assert v.tier == Tier.REFUSE and v.reasons == (Reason.G,)

    def test_overlap_failure_gives_i(self):
        pi = np.concatenate([np.full(80, 0.001), np.full(20, 0.5)])
        v = route(gates=[overlap(pi)], branch=Branch.DETECTABLE)
        assert v.tier == Tier.REFUSE and v.reasons == (Reason.I,)

    def test_below_gives_p(self):
        v = route(branch=Branch.BELOW)
        assert v.label.startswith("0 (P)")

    def test_indeterminate_gives_p_query(self):
        v = route(branch=Branch.INDETERMINATE)
        assert v.reasons == (Reason.P_QUERY,) and v.is_abstention

    def test_detectable_uncertified_gives_l(self):
        v = route(branch=Branch.DETECTABLE, certified=False)
        assert v.tier == Tier.PREDICT and v.reasons == (Reason.L,)

    def test_detectable_certified_gives_tier3(self):
        v = route(branch=Branch.DETECTABLE, certified=True)
        assert v.tier == Tier.DEPLOY and v.reasons == ()

    def test_unsettled_ladder_caps_at_shadow(self):
        from logworthy.audit.ladder import LadderResult

        lad = LadderResult(
            outcome=LadderOutcome.INDETERMINATE, rungs=(), ratio=1.0,
            ratio_ci=(0.5, 1.5), reason="not plateaued",
        )
        v = route(branch=Branch.DETECTABLE, certified=True, ladder=lad)
        assert v.tier == Tier.SHADOW and Reason.C_QUERY in v.reasons

    def test_collapse_overrides_detectability(self):
        from logworthy.audit.ladder import LadderResult

        lad = LadderResult(
            outcome=LadderOutcome.COLLAPSE, rungs=(), ratio=0.05,
            ratio_ci=(0.01, 0.2), reason="collapsed",
        )
        v = route(branch=Branch.DETECTABLE, certified=True, ladder=lad)
        assert v.tier == Tier.REFUSE and v.reasons == (Reason.C,)

    def test_econ_flag_does_not_change_tier(self):
        v = route(branch=Branch.DETECTABLE, certified=True, npv_positive=False)
        assert v.tier == Tier.DEPLOY and "[econ]" in v.flags

    def test_prov_flag_for_unchecked_sequential(self):
        v = route(branch=Branch.DETECTABLE, certified=True,
                  sequential_responsive=False)
        assert "[prov]" in v.flags

    def test_single_step_gets_no_prov_flag(self):
        v = route(branch=Branch.DETECTABLE, certified=True,
                  sequential_responsive=None)
        assert "[prov]" not in v.flags

    def test_missing_branch_refuses(self):
        v = route(branch=None)
        assert v.tier == Tier.REFUSE

    def test_responses_accompany_every_reason(self):
        v = route(branch=Branch.INDETERMINATE)
        assert len(v.responses) == len(v.reasons)


class TestAuditEndToEnd:
    def _log(self, effect, seed=0, n=3000):
        rng = np.random.default_rng(seed)
        clusters = rng.integers(0, 25, n)
        actions = rng.integers(0, 4, n)
        state = rng.normal(size=(n, 2))
        y = (
            state @ [0.5, -0.3]
            + rng.normal(0, 1, 25)[clusters]
            + effect * actions
            + rng.normal(size=n)
        )
        return y, actions, state, clusters

    def test_decoupled_log_is_refused(self):
        v = audit(*self._log(0.0, seed=1), certified=True, seed=0)
        assert v.tier == Tier.REFUSE and Reason.P in v.reasons

    def test_coupled_certified_log_reaches_tier3(self):
        v = audit(*self._log(0.5, seed=2), certified=True, seed=0)
        assert v.tier == Tier.DEPLOY

    def test_coupled_uncertified_log_stops_at_predict(self):
        v = audit(*self._log(0.5, seed=2), certified=False, seed=0)
        assert v.tier == Tier.PREDICT

    def test_no_certification_caps_at_tier1(self):
        v = audit(*self._log(0.5, seed=2), seed=0)
        assert v.tier <= Tier.PREDICT

    def test_interactions_widen_the_contrast(self):
        y, a, s, g = self._log(0.3, seed=3)
        main = audit(y, a, s, g, certified=True, seed=0)
        inter = audit(y, a, s, g, certified=True, interactions=True, seed=0)
        assert inter.coupling.df1 > main.coupling.df1

    def test_length_mismatch_rejected(self):
        y, a, s, g = self._log(0.0)
        with pytest.raises(ValueError, match="same length"):
            audit(y[:-1], a, s, g)

    def test_calibrated_needs_grid(self):
        y, a, s, g = self._log(0.0)
        with pytest.raises(ValueError, match="needs threshold_grid"):
            audit(y, a, s, g, threshold_method="calibrated")

    def test_unknown_threshold_method(self):
        y, a, s, g = self._log(0.0)
        with pytest.raises(ValueError, match="unknown threshold_method"):
            audit(y, a, s, g, threshold_method="cr2")

    def test_verdict_carries_its_evidence(self):
        v = audit(*self._log(0.0, seed=4), seed=0)
        assert v.coupling is not None and v.threshold is not None
        assert any(g.code == "C4b" for g in v.gates)
