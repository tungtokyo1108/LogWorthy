"""Feasibility gates, the saturation ladder, and the audit runner."""

from .gates import (
    GateResult, definitional_target, effective_overlap, overlap,
    propensity_calibration, propensity_informativeness,
)
from .ladder import LadderOutcome, LadderResult, Rung, run_ladder
from .runner import audit, build_design

__all__ = [
    "GateResult", "definitional_target", "overlap", "propensity_calibration",
    "propensity_informativeness", "effective_overlap",
    "Rung", "LadderResult", "LadderOutcome", "run_ladder",
    "audit", "build_design",
]
