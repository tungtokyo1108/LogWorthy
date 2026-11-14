"""Offline RL: learners, off-policy estimators, certification, and C6.

The audit does not merely accompany a reinforcement learning pipeline; the
learner is part of what is audited. This subpackage provides the learners, the
off-policy estimators their improvement is measured with, the certification
protocol that decides whether that improvement is real, and condition C6 for
sequential objectives.
"""

from .estimators import (
    ImprovementEstimate,
    direct_method,
    doubly_robust,
    inverse_propensity,
    paired_increment,
)
from .learners import (
    ConservativeQ,
    DiscreteCQL,
    GradientBoostedQ,
    GreedyDR,
    Learner,
    available_learners,
)
from .protocol import CertificationResult, PolicyClass, certify, certify_increment
from .sequential import (
    SequentialDecomposition,
    decompose_coupling,
    fitted_q_evaluation,
    run_c6,
)

__all__ = [
    "Learner", "ConservativeQ", "GreedyDR", "GradientBoostedQ", "DiscreteCQL",
    "available_learners",
    "ImprovementEstimate", "doubly_robust", "inverse_propensity",
    "direct_method", "paired_increment",
    "certify", "certify_increment", "CertificationResult", "PolicyClass",
    "fitted_q_evaluation", "decompose_coupling", "run_c6",
    "SequentialDecomposition",
]
