"""Rendering a verdict so that a reader can check it.

A verdict that cannot be traced back to its numbers is an opinion. These
functions print the gates, the coupling, the threshold and the branch that
produced the answer, so a reviewer can see which quantity bound.
"""

from __future__ import annotations

from ..verdict.verdict import Verdict

__all__ = ["summary", "audit_table"]


def summary(verdict: Verdict, *, width: int = 78) -> str:
    """Render a verdict as a readable block of text.

    Args:
        verdict: The verdict to render.
        width: Line width for the rules.

    Returns:
        A multi-line string.
    """
    rule = "=" * width
    thin = "-" * width
    out = [rule, f"VERDICT  {verdict.label}  —  {verdict.name}"]
    if verdict.policy_class:
        out.append(f"policy class: {verdict.policy_class}")
    out += [rule, _wrap(f"Supports: {verdict.supports}", width), ""]
    out.append(_wrap(verdict.explanation, width))

    if verdict.coupling is not None:
        c = verdict.coupling
        out += ["", thin, "COUPLING"]
        out.append(
            f"  omega2 = {c.omega2:.4e}   95% CI = [{c.lower:.4e}, {c.upper:.4e}]"
        )
        if c.p_value is not None:
            out.append(f"  bootstrap p = {c.p_value:.4f}")
        out.append(
            f"  df1 = {c.df1}   clusters = {c.n_clusters}   "
            f"n = {c.n:,}   n_eff = {c.n_eff:,.0f}"
        )

    if verdict.threshold is not None:
        t = verdict.threshold
        out += ["", thin, "DETECTABILITY"]
        out.append(
            f"  eta2_min = {t.eta2_min:.4e}   "
            f"band = [{t.low:.4e}, {t.high:.4e}]"
        )
        out.append(f"  method = {t.method}   power = {t.power}   alpha = {t.alpha}")

    if verdict.ladder is not None:
        lad = verdict.ladder
        out += ["", thin, "SATURATION LADDER"]
        for r in lad.rungs:
            out.append(f"  {r.name:<24} omega2={r.omega2:.4e}  abs={r.absolute:.4e}")
        out.append(
            f"  ratio top/base = {lad.ratio:.3f} "
            f"[{lad.ratio_ci[0]:.3f}, {lad.ratio_ci[1]:.3f}]"
        )
        if lad.noise_control is not None:
            out.append(f"  noise control  = {lad.noise_control:.3f}")
        if lad.delta_prop is not None:
            lower = (
                f", lower limit {lad.delta_prop_lower:+.4f}"
                if lad.delta_prop_lower is not None
                else ""
            )
            out.append(f"  delta_prop     = {lad.delta_prop:+.4f} nats/row{lower}")
        out.append(f"  outcome: {lad.outcome} — {_wrap(lad.reason, width - 11, 11)}")

    cert = verdict._meta.get("certification") if verdict._meta else None
    if cert is not None:
        out += ["", thin, "CERTIFICATION (STAGE 2)"]
        out.append(f"  learner       = {cert.learner}  [{cert.policy_class}]")
        out.append(
            f"  seeds         = {cert.seeds_certified}/{cert.n_seeds} "
            f"({cert.proportion:.0%}, required > {cert.q:.0%})"
        )
        out.append(
            f"  mean uplift   = {cert.mean_uplift:+.5f} "
            f"[{cert.uplift_ci[0]:+.5f}, {cert.uplift_ci[1]:+.5f}]"
        )
        if cert.null_proportion is not None:
            out.append(
                f"  permutation   = null reaches {cert.null_proportion:.2f}, "
                f"p = {cert.p_value:.4f}"
            )
        if cert.certified and verdict.tier == 0:
            out.append(
                "  "
                + _wrap(
                    "NOTE: the learner certified, but the coupling does not clear "
                    "what this design can detect. The audit reports the "
                    "certificate as unsupported and routes the log by its "
                    "detectability branch: selection on fitting error is the "
                    "more plausible explanation.",
                    width - 2, 2,
                )
            )

    if verdict.gates:
        out += ["", thin, "STAGE-0 GATES"]
        for g in verdict.gates:
            mark = "pass" if g.passed else "FAIL"
            out.append(f"  [{mark}] {g.code}: {_wrap(g.detail, width - 14, 14)}")

    if verdict.responses:
        out += ["", thin, "INDICATED RESPONSE"]
        for code, resp in zip(verdict.reasons, verdict.responses):
            out.append(f"  {code}: {_wrap(resp, width - 7, 7)}")

    out.append(rule)
    return "\n".join(out)


def audit_table(verdicts: list[Verdict]) -> str:
    """Render several verdicts as one table, one row per policy class.

    Args:
        verdicts: Verdicts to tabulate.

    Returns:
        A multi-line string.
    """
    if not verdicts:
        return "(no verdicts)"
    head = f"{'policy class':<22}{'coupling [L,U]':<28}{'threshold':<24}verdict"
    lines = [head, "-" * len(head)]
    for v in verdicts:
        if v.coupling is not None:
            interval = f"[{v.coupling.lower:.3e},{v.coupling.upper:.3e}]"
        else:
            interval = "—"
        if v.threshold is not None:
            band = f"[{v.threshold.low:.3e},{v.threshold.high:.3e}]"
        else:
            band = "—"
        lines.append(f"{v.policy_class:<22}{interval:<28}{band:<24}{v.label}")
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: int = 0) -> str:
    """Wrap text to width, indenting continuation lines."""
    import textwrap

    if not text:
        return ""
    wrapped = textwrap.wrap(text, max(width, 20))
    pad = " " * indent
    return ("\n" + pad).join(wrapped)
