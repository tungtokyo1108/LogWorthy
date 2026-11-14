"""Offline learners, and the interface the audit holds them to.

A verdict is about a (log, reward, reference design, **policy class**) quadruple,
and the learner is part of what is audited, not a free choice made afterwards.
Two consequences shape this module.

First, the learner must be **fixed in advance**. Choosing, among several
learners, the one that certifies is selection, and the audit's own simulations
show it inflates the false-licence rate. :func:`certify` therefore takes one
learner, and the pre-registration helper records which.

Second, conservatism is not a detail. Near the detectability frontier a greedy
argmax selects the learner's own estimation noise, so a high-capacity learner is
*worse*, not better: on the paper's logistics corpus the conservative learner
certified in 29 of 30 seeds while the unregularised GBM managed 3. The three
learners here span that range deliberately, so that a user can see the spread on
their own data rather than take it on trust.

``ConservativeQ`` is a pure-NumPy shrinkage learner with the same conservative
character as CQL: it shrinks each state-action value toward the logged mean in
proportion to how little support that cell has. ``DiscreteCQL`` wraps d3rlpy's
real implementation and is used when it is installed. ``GreedyDR`` and
``GradientBoostedQ`` are the unregularised references.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Learner",
    "ConservativeQ",
    "GreedyDR",
    "GradientBoostedQ",
    "DiscreteCQL",
    "available_learners",
]


def _as_matrix(state: np.ndarray) -> np.ndarray:
    """Coerce a state argument to a 2-D ``(n, d)`` matrix.

    A 1-D array is read as one feature over ``n`` rows, which is the reading a
    caller passing a single state column intends.
    """
    s = np.asarray(state, dtype=float)
    if s.ndim == 1:
        return s.reshape(-1, 1)
    if s.ndim != 2:
        raise ValueError(f"state must be 1-D or 2-D, got shape {s.shape}")
    return s



@runtime_checkable
class Learner(Protocol):
    """What the audit needs from a learner.

    Any object with these two methods can be certified, so a team's own
    production learner can be audited by wrapping it to this interface.
    """

    name: str

    def fit(
        self,
        state: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        *,
        n_actions: int,
    ) -> "Learner":
        """Fit on the training fold. Must not see the evaluation fold."""
        ...

    def q_values(self, state: np.ndarray) -> np.ndarray:
        """Predicted value of every action, shape ``(n, K)``.

        The audit derives the policy as the row-wise argmax, and uses the same
        matrix as the outcome model in the doubly robust correction.
        """
        ...


class _TabularBase:
    """Shared tabular machinery: discretise the state, average the reward."""

    def __init__(self, n_bins: int = 4):
        self.n_bins = n_bins
        self._edges: Optional[np.ndarray] = None
        self._table: Optional[np.ndarray] = None
        self._counts: Optional[np.ndarray] = None
        self._global_mean: float = 0.0
        self.n_actions: int = 0

    def _discretise(self, state: np.ndarray, fit: bool = False) -> np.ndarray:
        """Map a state matrix to a single cell index per row.

        Each feature is cut at its training quantiles and the per-feature bins
        are combined into one index, so the table is over the product of bins.
        Bin edges are learned on the training fold only and reused at
        evaluation; recomputing them on the evaluation fold would leak.
        """
        s = _as_matrix(state)
        if fit:
            quantiles = np.linspace(0, 1, self.n_bins + 1)[1:-1]
            self._edges = (
                np.column_stack(
                    [np.quantile(s[:, j], quantiles) for j in range(s.shape[1])]
                )
                if s.shape[1]
                else np.empty((0, 0))
            )
        if self._edges is None or self._edges.size == 0:
            return np.zeros(s.shape[0], dtype=int)
        cell = np.zeros(s.shape[0], dtype=int)
        for j in range(self._edges.shape[1]):
            cell = cell * self.n_bins + np.digitize(s[:, j], self._edges[:, j])
        return cell

    def _accumulate(self, cell, actions, rewards, n_cells, n_actions):
        key = cell * n_actions + actions
        size = n_cells * n_actions
        total = np.bincount(key, weights=rewards, minlength=size)
        count = np.bincount(key, minlength=size).astype(float)
        return total.reshape(n_cells, n_actions), count.reshape(n_cells, n_actions)


class ConservativeQ(_TabularBase):
    """Shrinkage tabular Q-learner: conservative by construction.

    Each state-action value is shrunk toward the logged mean with weight
    ``alpha / (count + alpha)``, so cells the logging policy rarely visited are
    pulled back rather than trusted. This is the same pressure CQL applies by
    penalising out-of-distribution actions, in a form with no dependencies and
    no training loop.

    Args:
        alpha: Shrinkage strength in pseudo-counts. Larger is more conservative.
        n_bins: Quantile bins per state feature.
    """

    def __init__(self, alpha: float = 50.0, n_bins: int = 4):
        super().__init__(n_bins)
        self.alpha = alpha
        self.name = f"ConservativeQ(alpha={alpha:g})"

    def fit(self, state, actions, rewards, *, n_actions):
        rewards = np.asarray(rewards, dtype=float)
        actions = np.asarray(actions)
        self.n_actions = n_actions
        cell = self._discretise(state, fit=True)
        n_cells = int(cell.max()) + 1
        total, count = self._accumulate(cell, actions, rewards, n_cells, n_actions)
        self._global_mean = float(rewards.mean())
        self._table = (total + self.alpha * self._global_mean) / (count + self.alpha)
        self._counts = count
        return self

    def q_values(self, state):
        if self._table is None:
            raise RuntimeError("fit() must be called before q_values()")
        cell = np.clip(self._discretise(state), 0, self._table.shape[0] - 1)
        return self._table[cell]


class GreedyDR(_TabularBase):
    """Unshrunk tabular learner: the greedy reference.

    Takes each cell's sample mean at face value. On a decoupled log its argmax
    selects whichever action happened to look best, which is the mechanism
    behind spurious certificates: the paper's Theorem 2 bounds exactly this.
    Kept so users can see the contrast on their own data.
    """

    def __init__(self, n_bins: int = 4):
        super().__init__(n_bins)
        self.name = "GreedyDR"

    def fit(self, state, actions, rewards, *, n_actions):
        rewards = np.asarray(rewards, dtype=float)
        actions = np.asarray(actions)
        self.n_actions = n_actions
        cell = self._discretise(state, fit=True)
        n_cells = int(cell.max()) + 1
        total, count = self._accumulate(cell, actions, rewards, n_cells, n_actions)
        self._global_mean = float(rewards.mean())
        with np.errstate(invalid="ignore", divide="ignore"):
            table = np.where(count > 0, total / np.maximum(count, 1), self._global_mean)
        self._table = table
        self._counts = count
        return self

    def q_values(self, state):
        if self._table is None:
            raise RuntimeError("fit() must be called before q_values()")
        cell = np.clip(self._discretise(state), 0, self._table.shape[0] - 1)
        return self._table[cell]


class GradientBoostedQ:
    """Gradient-boosted outcome model, one regressor per action.

    High capacity and no conservatism, so it is the most exposed to the
    optimiser's curse near the frontier. Requires scikit-learn.

    Args:
        **kwargs: Passed to ``HistGradientBoostingRegressor``.
    """

    def __init__(self, **kwargs):
        self.kwargs = {"max_iter": 100, "random_state": 0, **kwargs}
        self.name = "GradientBoostedQ"
        self._models: list = []
        self.n_actions = 0
        self._global_mean = 0.0

    def fit(self, state, actions, rewards, *, n_actions):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "GradientBoostedQ needs scikit-learn: pip install 'logworthy[examples]'"
            ) from exc
        s = _as_matrix(state)
        rewards = np.asarray(rewards, dtype=float)
        actions = np.asarray(actions)
        self.n_actions = n_actions
        self._global_mean = float(rewards.mean())
        self._models = []
        for k in range(n_actions):
            rows = actions == k
            if rows.sum() < 10:
                self._models.append(None)
                continue
            model = HistGradientBoostingRegressor(**self.kwargs)
            model.fit(s[rows], rewards[rows])
            self._models.append(model)
        return self

    def q_values(self, state):
        s = _as_matrix(state)
        out = np.full((s.shape[0], self.n_actions), self._global_mean)
        for k, model in enumerate(self._models):
            if model is not None:
                out[:, k] = model.predict(s)
        return out


class DiscreteCQL:
    """Conservative Q-learning, wrapping d3rlpy's implementation.

    This is the learner the paper deploys. It is wrapped rather than
    reimplemented, so that a certificate obtained here is a certificate for the
    same algorithm a team would ship.

    d3rlpy and torch are imported lazily, so ``import logworthy`` does not pull
    them in. Install them with ``pip install 'logworthy[rl]'``.

    Args:
        n_epochs: Training epochs per fit. The audit refits once per seed, so
            keep this modest.
        alpha: Conservative weight; higher penalises out-of-distribution actions
            more strongly.
        batch_size: Minibatch size.
        learning_rate: Optimiser step size.
        use_gpu: Whether to train on GPU when one is available.
    """

    def __init__(
        self,
        n_epochs: int = 5,
        alpha: float = 1.0,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        use_gpu: bool = False,
    ):
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        self.name = f"DiscreteCQL(alpha={alpha:g})"
        self._algo = None
        self.n_actions = 0

    def fit(self, state, actions, rewards, *, n_actions):
        algo_module, dataset_module = _import_d3rlpy()
        s = _as_matrix(state).astype(np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int64)
        self.n_actions = n_actions

        # A single-step decision log: every transition terminates immediately.
        terminals = np.ones(len(rewards), dtype=np.float32)
        dataset = dataset_module.MDPDataset(
            observations=s,
            actions=actions.reshape(-1, 1),
            rewards=rewards.reshape(-1, 1),
            terminals=terminals,
        )
        self._algo = algo_module.DiscreteCQLConfig(
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        ).create(device="cuda:0" if self.use_gpu else "cpu")
        n_steps = max(1, self.n_epochs * len(rewards) // self.batch_size)
        self._algo.fit(
            dataset,
            n_steps=n_steps,
            n_steps_per_epoch=max(1, n_steps // max(self.n_epochs, 1)),
            show_progress=False,
            save_interval=10**9,
            logger_adapter=_null_logger(),
        )
        return self

    def q_values(self, state):
        if self._algo is None:
            raise RuntimeError("fit() must be called before q_values()")
        s = _as_matrix(state).astype(np.float32)
        n = s.shape[0]
        out = np.empty((n, self.n_actions), dtype=float)
        for k in range(self.n_actions):
            act = np.full((n, 1), k, dtype=np.int64)
            out[:, k] = self._algo.predict_value(s, act).reshape(-1)
        return out


def _import_d3rlpy():
    """Import d3rlpy lazily and quietly.

    Keeping this out of module import time means the audit runs without torch
    installed, which matters because most users audit a log without ever
    training a deep learner.
    """
    import logging
    import os

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        import d3rlpy
        from d3rlpy import dataset as dataset_module
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "DiscreteCQL needs d3rlpy and torch: pip install 'logworthy[rl]'. "
            "For a dependency-free conservative learner, use ConservativeQ."
        ) from exc
    logging.getLogger("d3rlpy").setLevel(logging.ERROR)
    _silence_structlog()
    return d3rlpy.algos, dataset_module


def _silence_structlog() -> None:
    """Drop d3rlpy's structlog output below ERROR.

    d3rlpy logs dataset signatures, parameters and per-epoch metrics through
    structlog, which bypasses the stdlib logging level set above. Certification
    refits once per seed, so at 30 seeds this is hundreds of lines of noise
    around the one number the user asked for.
    """
    try:
        import logging as _logging

        import structlog

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(_logging.ERROR)
        )
    except Exception:  # pragma: no cover - structlog absent or API changed
        pass


def _null_logger():
    """A d3rlpy logger adapter that writes nothing.

    The audit fits once per seed, so the default adapter would write tens of
    directories of checkpoints per run.
    """
    try:
        from d3rlpy.logging import NoopAdapterFactory

        return NoopAdapterFactory()
    except Exception:  # pragma: no cover - older d3rlpy
        return None


def available_learners() -> dict:
    """Which learners can run in this environment.

    Returns:
        Mapping from learner name to ``True`` when its dependencies are
        importable. Useful in a notebook before choosing one.
    """
    status = {"ConservativeQ": True, "GreedyDR": True}
    try:
        import sklearn  # noqa: F401

        status["GradientBoostedQ"] = True
    except ImportError:
        status["GradientBoostedQ"] = False
    try:
        import d3rlpy  # noqa: F401

        status["DiscreteCQL"] = True
    except ImportError:
        status["DiscreteCQL"] = False
    return status
