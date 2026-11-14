"""Condition C6: downstream responsiveness for sequential objectives.

A single-step audit asks whether the action moves the immediate reward. On a
sequential objective that is the wrong question, because an action can leave the
immediate reward untouched and still change everything that follows, or move the
immediate reward while the continuation cancels it out.

Proposition 1 of the paper separates the two. With ``PV(s, a)`` the expected
value of the next state,

    D_1    = E Var_a R(s, a)                     immediate coupling
    D_cont = gamma^2 E Var_a PV(s, a)            downstream coupling
    D_seq  = E Var_a Q(s, a)                     total coupling

and

    D_seq - D_1 = D_cont + 2 gamma E Cov_a[R(s,a), PV(s,a)].

Downstream responsiveness is ``D_cont > 0``, so **C6 is decided on D_cont**, not
on the sequential share

    chi_seq = (D_seq - D_1) / D_seq  in (-inf, 1].

That share is a two-sided diagnostic only. The covariance term can be negative,
so chi_seq can be negative while D_cont is strictly positive: a log can be
genuinely sequential and still show a negative share. Reporting chi_seq as if it
measured responsiveness is the error this module is arranged to prevent, which
is why :func:`run_c6` returns the flag from ``D_cont`` and carries chi_seq
alongside as a diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SequentialDecomposition",
    "fitted_q_evaluation",
    "decompose_coupling",
    "run_c6",
]


@dataclass(frozen=True)
class SequentialDecomposition:
    """The pieces of Proposition 1, measured.

    Attributes:
        d1: Immediate coupling ``E Var_a R(s, a)``.
        d_cont: Downstream coupling ``gamma^2 E Var_a PV(s, a)``. This is what
            C6 is decided on.
        d_seq: Total coupling ``E Var_a Q(s, a)``.
        chi_seq: Sequential share, a two-sided diagnostic. May be negative.
        covariance_term: ``2 gamma E Cov_a[R, PV]``, the term that makes
            chi_seq and ``d_cont`` disagree in sign.
        gamma: Discount factor used.
        responsive: Whether ``d_cont`` cleared its threshold.
        d_cont_threshold: What it was compared against.
    """

    d1: float
    d_cont: float
    d_seq: float
    chi_seq: float
    covariance_term: float
    gamma: float
    responsive: Optional[bool] = None
    d_cont_threshold: Optional[float] = None

    @property
    def triangle_holds(self) -> bool:
        """Whether the Minkowski bounds of Proposition 1 are satisfied.

        ``|sqrt(D1) - sqrt(D_cont)| <= sqrt(D_seq) <= sqrt(D1) + sqrt(D_cont)``.
        A violation means the three quantities were not estimated on one state
        distribution, which is a bug in the caller's inputs rather than a
        finding.
        """
        lo = abs(np.sqrt(self.d1) - np.sqrt(self.d_cont))
        hi = np.sqrt(self.d1) + np.sqrt(self.d_cont)
        root = np.sqrt(self.d_seq)
        tol = 1e-8 + 1e-6 * max(hi, 1.0)
        return bool(lo - tol <= root <= hi + tol)

    def __str__(self) -> str:  # pragma: no cover - display only
        verdict = {True: "responsive", False: "not responsive", None: "not decided"}[
            self.responsive
        ]
        return (
            f"D1={self.d1:.4e} D_cont={self.d_cont:.4e} D_seq={self.d_seq:.4e} "
            f"chi_seq={self.chi_seq:+.3f} ({verdict})"
        )


def fitted_q_evaluation(
    state: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_state: np.ndarray,
    terminals: np.ndarray,
    target_policy: np.ndarray,
    *,
    n_actions: Optional[int] = None,
    gamma: float = 0.99,
    n_iterations: int = 20,
    n_bins: int = 4,
    alpha: float = 10.0,
    tol: float = 1e-6,
) -> tuple:
    """Fitted-Q evaluation of a target policy on logged transitions.

    Iterates the Bellman evaluation operator on a tabular discretisation of the
    state until the value function stops moving:

        Q(s, a) <- R(s, a) + gamma * E[ Q(s', pi(s')) ]

    Values are shrunk toward the logged mean in proportion to cell support, so
    that a cell seen twice does not dominate the fixed point. This is evaluation
    of a *given* policy, not improvement over it, so no maximisation enters and
    the optimiser's curse does not apply.

    Args:
        state: Current state, ``(n, d)``.
        actions: Logged action index, ``(n,)``.
        rewards: Immediate reward, ``(n,)``.
        next_state: Successor state, ``(n, d)``.
        terminals: 1 where the episode ended, ``(n,)``. Terminal transitions
            contribute no continuation value.
        target_policy: Action the evaluated policy takes in ``next_state``,
            ``(n,)``.
        n_actions: Number of actions; inferred when omitted.
        gamma: Discount factor.
        n_iterations: Maximum Bellman iterations.
        n_bins: Quantile bins per state feature.
        alpha: Shrinkage pseudo-counts.
        tol: Stop when the largest value change falls below this.

    Returns:
        ``(q_table, cell_of_row, next_cell_of_row, reward_table)`` where
        ``q_table`` has shape ``(n_cells, K)``.

    Raises:
        ValueError: If inputs disagree in length.
    """
    s = _as_matrix(state)
    s_next = _as_matrix(next_state)
    a = np.asarray(actions)
    r = np.asarray(rewards, dtype=float)
    term = np.asarray(terminals, dtype=float)
    tgt = np.asarray(target_policy)
    n = r.size
    if not (s.shape[0] == s_next.shape[0] == a.size == term.size == tgt.size == n):
        raise ValueError(
            "state, next_state, actions, rewards, terminals and target_policy "
            "must all have the same number of rows"
        )
    n_act = int(n_actions if n_actions is not None else max(a.max(), tgt.max()) + 1)

    edges = _edges(s, n_bins)
    cell = _cells(s, edges, n_bins)
    cell_next = _cells(s_next, edges, n_bins)
    n_cells = int(max(cell.max(), cell_next.max())) + 1

    key = cell * n_act + a
    size = n_cells * n_act
    total = np.bincount(key, weights=r, minlength=size).reshape(n_cells, n_act)
    count = np.bincount(key, minlength=size).astype(float).reshape(n_cells, n_act)
    global_mean = float(r.mean())
    reward_table = (total + alpha * global_mean) / (count + alpha)

    q = reward_table.copy()
    for _ in range(n_iterations):
        # Continuation value of each logged transition under the target policy.
        continuation = (1.0 - term) * q[cell_next, tgt]
        backup_total = np.bincount(
            key, weights=r + gamma * continuation, minlength=size
        ).reshape(n_cells, n_act)
        q_new = (backup_total + alpha * global_mean) / (count + alpha)
        shift = float(np.abs(q_new - q).max())
        q = q_new
        if shift < tol:
            break
    return q, cell, cell_next, reward_table


def decompose_coupling(
    q_table: np.ndarray,
    reward_table: np.ndarray,
    cell: np.ndarray,
    *,
    rho: Optional[np.ndarray] = None,
    gamma: float = 0.99,
) -> SequentialDecomposition:
    """Split the coupling into immediate and downstream parts (Proposition 1).

    All three variances are taken over the *same* state distribution, the one
    induced by the rows supplied, and over the reference action distribution
    ``rho``. Using different state distributions for the three would break the
    Minkowski bounds, which :attr:`SequentialDecomposition.triangle_holds`
    checks.

    Args:
        q_table: Fitted ``Q(s, a)``, shape ``(n_cells, K)``.
        reward_table: Fitted ``R(s, a)``, same shape.
        cell: Cell index per row, so that states are weighted as they occur.
        rho: Reference action distribution, shape ``(K,)``. Defaults to uniform,
            which is the paper's convention.
        gamma: Discount factor, used to report the covariance term.

    Returns:
        A :class:`SequentialDecomposition`.
    """
    q = np.asarray(q_table, dtype=float)
    r = np.asarray(reward_table, dtype=float)
    n_act = q.shape[1]
    w = np.ones(n_act) / n_act if rho is None else np.asarray(rho, dtype=float)
    w = w / w.sum()

    rows = np.asarray(cell)
    qs, rs = q[rows], r[rows]
    pv = qs - rs  # gamma * E[V(s')], the continuation block

    d1 = float(_weighted_var(rs, w).mean())
    d_cont = float(_weighted_var(pv, w).mean())
    d_seq = float(_weighted_var(qs, w).mean())

    cov = float(
        (
            ((rs - (rs * w).sum(axis=1, keepdims=True))
             * (pv - (pv * w).sum(axis=1, keepdims=True))) * w
        ).sum(axis=1).mean()
    )
    chi = (d_seq - d1) / d_seq if d_seq > 0 else 0.0

    return SequentialDecomposition(
        d1=d1,
        d_cont=d_cont,
        d_seq=d_seq,
        chi_seq=float(chi),
        covariance_term=2.0 * cov,
        gamma=gamma,
    )


def run_c6(
    state: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_state: np.ndarray,
    terminals: np.ndarray,
    target_policy: np.ndarray,
    *,
    n_actions: Optional[int] = None,
    gamma: float = 0.99,
    d_cont_threshold: Optional[float] = None,
    clusters: Optional[np.ndarray] = None,
    n_boot: int = 199,
    seed: int = 0,
    **fqe_kwargs,
) -> SequentialDecomposition:
    """Run condition C6 end to end and decide downstream responsiveness.

    C6 is decided on the **lower confidence limit** of ``D_cont``, not on its
    point estimate, matching how detectability is decided elsewhere in the
    audit.

    Args:
        state: Current state.
        actions: Logged action.
        rewards: Immediate reward.
        next_state: Successor state.
        terminals: Episode-end indicator.
        target_policy: Action the evaluated policy takes in ``next_state``.
        n_actions: Number of actions.
        gamma: Discount factor.
        d_cont_threshold: Calibrated threshold for ``D_cont``. When omitted,
            responsiveness is left undecided (``responsive=None``), which the
            router treats as grounds for the ``[prov]`` flag.
        clusters: Cluster labels for the bootstrap interval.
        n_boot: Cluster bootstrap resamples.
        seed: Seed for reproducibility.
        **fqe_kwargs: Passed to :func:`fitted_q_evaluation`.

    Returns:
        A :class:`SequentialDecomposition` with ``responsive`` set when a
        threshold was supplied.
    """
    q, cell, cell_next, reward_table = fitted_q_evaluation(
        state, actions, rewards, next_state, terminals, target_policy,
        n_actions=n_actions, gamma=gamma, **fqe_kwargs,
    )
    point = decompose_coupling(q, reward_table, cell, gamma=gamma)

    if d_cont_threshold is None:
        return point

    rng = np.random.default_rng(seed)
    n = len(cell)
    codes = (
        np.zeros(n, dtype=int)
        if clusters is None
        else np.unique(np.asarray(clusters), return_inverse=True)[1]
    )
    groups = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        draws[b] = decompose_coupling(q, reward_table, cell[idx], gamma=gamma).d_cont
    lower = float(np.quantile(draws, 0.025))

    return SequentialDecomposition(
        d1=point.d1,
        d_cont=point.d_cont,
        d_seq=point.d_seq,
        chi_seq=point.chi_seq,
        covariance_term=point.covariance_term,
        gamma=gamma,
        responsive=bool(lower > d_cont_threshold),
        d_cont_threshold=float(d_cont_threshold),
    )


def _weighted_var(values: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Variance over actions under weights ``w``, per row."""
    mean = (values * w).sum(axis=1, keepdims=True)
    return ((values - mean) ** 2 * w).sum(axis=1)


def _as_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def _edges(s: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    if s.shape[1] == 0:
        return np.empty((0, 0))
    return np.column_stack([np.quantile(s[:, j], quantiles) for j in range(s.shape[1])])


def _cells(s: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    if edges.size == 0:
        return np.zeros(s.shape[0], dtype=int)
    cell = np.zeros(s.shape[0], dtype=int)
    for j in range(edges.shape[1]):
        cell = cell * n_bins + np.digitize(s[:, j], edges[:, j])
    return cell
