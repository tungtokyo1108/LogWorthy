"""Statistical core: the coupling estimator, its test, and the threshold."""

from .coupling import (
    CouplingEstimate,
    design_effect,
    kish_n_eff,
    partial_omega2,
    wild_cluster_bootstrap,
)
from .detectability import (
    Branch,
    Threshold,
    analytic_threshold,
    calibrate_threshold,
    three_branch,
)

__all__ = [
    "CouplingEstimate", "partial_omega2", "wild_cluster_bootstrap",
    "design_effect", "kish_n_eff",
    "Threshold", "Branch", "analytic_threshold", "calibrate_threshold",
    "three_branch",
]
