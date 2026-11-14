"""LogWorthy: can your log support a policy?

An audit for offline reinforcement learning and off-policy evaluation. Before
asking which policy is best, it asks a prior question: does the outcome respond
to the action at all, by enough that *this* log could detect it?

The answer is a tier and a reason code, not a number:

    >>> import numpy as np
    >>> from logworthy import audit, summary
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> clusters = rng.integers(0, 25, n)
    >>> actions = rng.integers(0, 4, n)
    >>> state = rng.normal(size=(n, 2))
    >>> y = state @ [0.5, -0.3] + rng.normal(0, 1, 25)[clusters] + rng.normal(size=n)
    >>> v = audit(y, actions, state, clusters)   # no coupling implanted
    >>> v.label
    '0 (P)'

A ``0 (P)`` means the coupling's upper limit lies below what the design can
detect: the log is a negative control for this policy class, and the reported
threshold sizes the pilot that would resolve it.

"""

from .audit import (
    GateResult,
    LadderOutcome,
    LadderResult,
    Rung,
    audit,
    build_design,
    definitional_target,
    effective_overlap,
    overlap,
    propensity_calibration,
    propensity_informativeness,
    run_ladder,
)
from .core import (
    Branch,
    CouplingEstimate,
    Threshold,
    analytic_threshold,
    calibrate_threshold,
    design_effect,
    kish_n_eff,
    partial_omega2,
    three_branch,
    wild_cluster_bootstrap,
)
from .datasets import LoggedDecisions, load_dataco, load_olist
from .report import audit_table, summary
from .rl import (
    CertificationResult,
    ConservativeQ,
    DiscreteCQL,
    GradientBoostedQ,
    GreedyDR,
    ImprovementEstimate,
    Learner,
    PolicyClass,
    SequentialDecomposition,
    available_learners,
    certify,
    certify_increment,
    decompose_coupling,
    direct_method,
    doubly_robust,
    fitted_q_evaluation,
    inverse_propensity,
    paired_increment,
    run_c6,
)
from .verdict import Flag, Reason, Tier, Verdict, route

__version__ = "0.1.0"

__all__ = [
    # one-call entry point
    "audit", "summary", "audit_table",
    # verdicts
    "Verdict", "Tier", "Reason", "Flag", "route",
    # statistical core
    "CouplingEstimate", "partial_omega2", "wild_cluster_bootstrap",
    "design_effect", "kish_n_eff",
    "Threshold", "Branch", "analytic_threshold", "calibrate_threshold",
    "three_branch",
    # gates and ladder
    "GateResult", "definitional_target", "overlap", "propensity_calibration",
    "propensity_informativeness", "effective_overlap",
    "Rung", "LadderResult", "LadderOutcome", "run_ladder",
    "build_design",
    # offline RL: learners, estimators, certification, sequential C6
    "Learner", "ConservativeQ", "GreedyDR", "GradientBoostedQ", "DiscreteCQL",
    "available_learners",
    "ImprovementEstimate", "doubly_robust", "inverse_propensity",
    "direct_method", "paired_increment",
    "certify", "certify_increment", "CertificationResult", "PolicyClass",
    "fitted_q_evaluation", "decompose_coupling", "run_c6",
    "SequentialDecomposition",
    # bundled real datasets
    "LoggedDecisions", "load_olist", "load_dataco",
    "__version__",
]
