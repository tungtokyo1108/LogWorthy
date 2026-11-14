"""Tiers, reason codes, and the routing rule that assigns them.

A verdict names what the data support, for a specific
``(dataset, reward, reference design, policy class)`` quadruple. It is not a
property of the dataset alone, and the same log can earn different verdicts for
fixed and contextual rules.

    Tier 0  Refuse   nothing for this quadruple           G, I, C, P, P?
    Tier 1  Predict  calibrated prediction only           L, L?
    Tier 2  Shadow   optimisation on the logged
                     distribution; shadow deployment      C?
    Tier 3  Deploy   offline policy optimisation          --

A trailing "?" has one meaning throughout: the rule abstained, and more data,
more seeds or a larger split would resolve it. Abstention is a first-class
answer here, not a failure to decide: it is reported together with the pilot
that would settle the question.

The order of checks matters and is not negotiable. Collapse is checked before
detectability, because a confounded association is not rescued by being
precisely measured.
"""

from __future__ import annotations

from typing import List, Optional

from dataclasses import dataclass, field

from ..audit.gates import GateResult
from ..audit.ladder import LadderOutcome, LadderResult
from ..core.detectability import Branch

__all__ = ["Tier", "Reason", "Flag", "Verdict", "route"]


class Tier:
    """What the data support."""

    REFUSE = 0
    PREDICT = 1
    SHADOW = 2
    DEPLOY = 3

    NAMES = {0: "Refuse", 1: "Predict", 2: "Shadow", 3: "Deploy"}

    SUPPORTS = {
        0: "nothing for this (dataset, reward, reference design, policy class)",
        1: "calibrated prediction and quoting with coverage; no policy verdict",
        2: "policy optimisation on the logged distribution; measured shadow "
           "deployment, not rollout",
        3: "offline policy optimisation within the named policy class",
    }


class Reason:
    """Why the audit stopped where it did."""

    G = "G"      # C1 or C2 failed
    I = "I"      # C4a failed: no overlap
    C = "C"      # ladder collapsed
    C_QUERY = "C?"   # ladder has not plateaued
    P = "P"      # coupling below detectability
    P_QUERY = "P?"   # detectability indeterminate
    L = "L"      # deployed learner does not certify
    L_QUERY = "L?"   # certification indeterminate

    RESPONSE = {
        "G": "redefine the outcome or audit another reward, then re-enter at Stage 0",
        "I": "randomise; exploration is most clearly indicated",
        "C": "do not treat the association as coupling; randomise, or add the "
             "pre-decision covariate the collapse implicates, then re-audit",
        "C?": "stabilise the state specification or confirm by randomisation "
              "before scale",
        "P": "use as a negative control; size a pilot from the reported threshold",
        "P?": "pilot sized to resolve the frontier; report which interval binds",
        "L": "more data or wider exploration: detectable but underpowered for policy",
        "L?": "more seeds or a larger honest split before deciding",
    }


class Flag:
    """Qualifiers that attach to any tier."""

    ECON = "[econ]"   # payoff ceiling below build cost
    PROV = "[prov]"   # sequential objective, single-step audit may understate


@dataclass(frozen=True)
class Verdict:
    """The audit's answer for one policy class.

    Attributes:
        tier: 0-3, see :class:`Tier`.
        reasons: Reason codes, in the order they fired.
        flags: Qualifying flags.
        policy_class: Name of the class audited, e.g. ``"fixed rules"``.
        explanation: Why this verdict was reached.
        gates: Stage-0 gate results, kept so a refusal can be traced.
        coupling: The coupling estimate, when one was computed.
        threshold: The detectability threshold, when one was computed.
        ladder: The ladder result, when one was run.
    """

    tier: int
    reasons: tuple[str, ...]
    flags: tuple[str, ...] = ()
    policy_class: str = ""
    explanation: str = ""
    gates: tuple[GateResult, ...] = ()
    coupling: Optional[object] = None
    threshold: Optional[object] = None
    ladder: Optional[LadderResult] = None
    _meta: dict = field(default_factory=dict, repr=False)

    @property
    def label(self) -> str:
        """Compact label, e.g. ``"0 (P?)"`` or ``"3"``."""
        core = str(self.tier)
        if self.reasons:
            core += " (" + ", ".join(self.reasons) + ")"
        if self.flags:
            core += " " + " ".join(self.flags)
        return core

    @property
    def name(self) -> str:
        """Tier name, e.g. ``"Refuse"``."""
        return Tier.NAMES[self.tier]

    @property
    def supports(self) -> str:
        """What this tier licenses, in words."""
        return Tier.SUPPORTS[self.tier]

    @property
    def is_abstention(self) -> bool:
        """True when the rule abstained rather than decided."""
        return any(r.endswith("?") for r in self.reasons)

    @property
    def responses(self) -> tuple[str, ...]:
        """Indicated next steps, one per reason code."""
        return tuple(Reason.RESPONSE[r] for r in self.reasons if r in Reason.RESPONSE)

    def __str__(self) -> str:  # pragma: no cover - display only
        head = f"{self.label} — {self.name}"
        if self.policy_class:
            head += f" [{self.policy_class}]"
        return head


def route(
    *,
    gates: Optional[List[GateResult]] = None,
    ladder: Optional[LadderResult] = None,
    branch: Optional[str] = None,
    certified: Optional[bool] = None,
    npv_positive: Optional[bool] = None,
    sequential_responsive: Optional[bool] = None,
    policy_class: str = "",
    coupling=None,
    threshold=None,
) -> Verdict:
    """Apply the routing rule and return a verdict.

    The order of checks is the rule, and it is deliberate:

    1. Stage-0 gates. C1/C2 failing gives ``0 (G)``; C4a failing gives ``0 (I)``.
    2. Ladder. Collapse gives ``0 (C)`` and **overrides detectability**.
    3. Detectability. Below gives ``0 (P)``; indeterminate gives ``0 (P?)``.
    4. Certification. Not certified gives ``1 (L)``; indeterminate ``1 (L?)``.
    5. Flags. ``[econ]`` when the payoff ceiling is below build cost;
       ``[prov]`` when a sequential objective was not or could not be checked.

    Args:
        gates: Stage-0 results. C1, C2 and C4a are consulted by code.
        ladder: Ladder result, if one was run. Observational logs should run it.
        branch: Outcome of the three-branch rule, from
            :func:`logworthy.core.three_branch`.
        certified: Whether the deployed learner certified on an honest split.
            ``None`` means certification was indeterminate.
        npv_positive: Whether the payoff ceiling clears build cost. ``False``
            adds ``[econ]``.
        sequential_responsive: Downstream responsiveness (C6). Pass ``None``
            (the default) for a single-step objective, where C6 does not apply.
            For a sequential objective pass ``True`` if responsiveness was
            established, or ``False`` if it was not or could not be run; both
            of the latter add ``[prov]``, because a one-step audit then
            understates or misstates the coupling.
        policy_class: Name of the class audited, carried into the verdict.
        coupling: Optional coupling estimate, carried into the verdict.
        threshold: Optional threshold, carried into the verdict.

    Returns:
        A :class:`Verdict`.
    """
    gates = list(gates or [])
    gate_tuple = tuple(gates)
    flags = _flags(npv_positive, sequential_responsive)

    def finish(tier, reasons, explanation):
        return Verdict(
            tier=tier,
            reasons=tuple(reasons),
            flags=flags,
            policy_class=policy_class,
            explanation=explanation,
            gates=gate_tuple,
            coupling=coupling,
            threshold=threshold,
            ladder=ladder,
        )

    # 1. Stage-0 gates.
    for code in ("C1", "C2"):
        failed = _find(gates, code)
        if failed is not None and not failed.passed:
            return finish(
                Tier.REFUSE, [Reason.G],
                f"{code} failed, so Q(s,a) is not meaningful for this reward: "
                f"{failed.detail}",
            )
    overlap_gate = _find(gates, "C4a")
    if overlap_gate is not None and not overlap_gate.passed:
        return finish(
            Tier.REFUSE, [Reason.I],
            f"C4a failed: {overlap_gate.detail}. The data are silent on the "
            "unsupported states, not negative.",
        )

    # 2. Ladder, before detectability.
    extra: list[str] = []
    if ladder is not None:
        if ladder.outcome == LadderOutcome.COLLAPSE:
            return finish(
                Tier.REFUSE, [Reason.C],
                f"the ladder collapsed: {ladder.reason}. A confounded "
                "association is not rescued by being precisely measured.",
            )
        if ladder.outcome == LadderOutcome.INDETERMINATE:
            extra.append(Reason.C_QUERY)

    # 3. Detectability.
    if branch == Branch.BELOW:
        return finish(
            Tier.REFUSE, [Reason.P] + extra,
            "the coupling's upper limit lies below the design's detectability "
            "threshold: this log is a negative control for this policy class.",
        )
    if branch == Branch.INDETERMINATE:
        return finish(
            Tier.REFUSE, [Reason.P_QUERY] + extra,
            "the coupling interval overlaps the detectability band, so the "
            "audit cannot separate the coupling from what the design detects.",
        )
    if branch is None:
        return finish(
            Tier.REFUSE, [Reason.P_QUERY] + extra,
            "no detectability branch was supplied, so no positive verdict can "
            "be issued.",
        )

    # 4. Certification.
    if certified is False:
        return finish(
            Tier.PREDICT, [Reason.L] + extra,
            "the coupling is detectable but the deployed learner did not "
            "certify: the data support prediction, not policy.",
        )
    if certified is None:
        return finish(
            Tier.PREDICT, [Reason.L_QUERY] + extra,
            "certification was indeterminate: more seeds or a larger honest "
            "split are needed before deciding.",
        )

    # 5. Certified. The ladder can still cap the tier at Shadow.
    if Reason.C_QUERY in extra:
        return finish(
            Tier.SHADOW, extra,
            "the learner certified, but the ladder has not plateaued, so the "
            "state specification is unsettled: shadow deployment, not rollout.",
        )
    return finish(
        Tier.DEPLOY, [],
        "gates passed, the coupling is robustly detectable, the ladder is "
        "settled and the learner certified on an honest split.",
    )


def _flags(npv_positive, sequential_responsive) -> tuple[str, ...]:
    """Flags qualify any tier and never change it.

    ``sequential_responsive is None`` means a single-step objective, where C6
    does not apply and no flag is warranted. ``False`` covers both "C6 ran and
    found no downstream responsiveness" and "C6 could not be run": the paper
    flags each, since a one-step audit then understates or misstates the
    coupling.
    """
    out = []
    if npv_positive is False:
        out.append(Flag.ECON)
    if sequential_responsive is False:
        out.append(Flag.PROV)
    return tuple(out)


def _find(gates, code) -> Optional[GateResult]:
    for g in gates:
        if g.code == code:
            return g
    return None
