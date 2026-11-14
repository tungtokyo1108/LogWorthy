"""The two public logistics logs the paper audits, ready to audit.

Both are real published datasets, shipped as column subsets so that the examples
run after a clone. No value is altered; see ``examples/data/prepare_data.py`` for
the extraction, which can be rerun against the originals.

    Olist Brazilian E-Commerce (Kaggle, CC BY-NC-SA 4.0)
        Decision: the delivery promise, quartile-binned. Reward: the review
        score. Clusters: the 27 customer states.

    DataCo Smart Supply Chain (Mendeley Data, CC BY 4.0)
        Decision: the shipping mode. Reward: profit per order. Clusters: the
        order regions.

Each loader returns a :class:`LoggedDecisions`, which carries exactly what the
audit needs and nothing else, so that the shape of the problem is visible: an
outcome, an action, a pre-decision state, and a cluster label per row.

Loading needs pandas: ``pip install 'logworthy[examples]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["LoggedDecisions", "load_olist", "load_dataco", "data_dir"]


@dataclass(frozen=True)
class LoggedDecisions:
    """A log in the form the audit consumes.

    Attributes:
        y: Outcome per row.
        actions: Logged action index, ``0..K-1``.
        state: Pre-decision state, ``(n, d)``. Contains only variables known
            before the action was taken.
        clusters: Cluster label per row: the unit of independence.
        feature_names: Column names of ``state``.
        action_names: Human-readable action labels.
        name: Dataset name.
        reward_name: What the outcome measures.
        time_order: Timestamps where the log is time-ordered, else ``None``.
    """

    y: np.ndarray
    actions: np.ndarray
    state: np.ndarray
    clusters: np.ndarray
    feature_names: tuple
    action_names: tuple
    name: str
    reward_name: str
    time_order: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        """Rows."""
        return len(self.y)

    @property
    def n_actions(self) -> int:
        """Distinct actions."""
        return len(self.action_names)

    @property
    def n_clusters(self) -> int:
        """Independent clusters. This, not ``n``, sets the precision."""
        return len(np.unique(self.clusters))

    def __str__(self) -> str:  # pragma: no cover - display only
        return (
            f"{self.name}: {self.n:,} rows, {self.n_actions} actions, "
            f"{self.n_clusters} clusters, reward = {self.reward_name}"
        )


def data_dir() -> Path:
    """Directory holding the bundled extracts."""
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "examples" / "data", here / "data"):
        if candidate.is_dir():
            return candidate
    return here.parent / "examples" / "data"


def _read(filename: str):
    """Read a bundled extract, with a clear error when it is missing."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "loading the example datasets needs pandas: "
            "pip install 'logworthy[examples]'"
        ) from exc
    path = data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The bundled extracts live in examples/data/; "
            "rebuild them from the original datasets with "
            "examples/data/prepare_data.py"
        )
    return pd.read_csv(path, compression="gzip")


def load_olist(*, window: str = "all", drop_undelivered: bool = True) -> LoggedDecisions:
    """The Olist delivery-promise log.

    The decision is the delivery promise: how many days ahead the estimated
    delivery date is set at purchase, binned into quartiles. The reward is the
    customer's review score. Clusters are the 27 customer states, which is the
    binding constraint: 97,917 rows supply the information of a few thousand.

    The state holds only pre-decision variables: price, freight, and whether the
    customer and seller are in the same state. Realised delivery time is
    excluded, since it is decided after the promise and would leak.

    Args:
        window: ``"all"``, ``"dev"`` (through 2018-02-28) or ``"holdout"``
            (after it). The paper audits both windows separately, and the split
            is the one it registers.
        drop_undelivered: Drop orders without a delivery date, whose promise
            was never resolved.

    Returns:
        A :class:`LoggedDecisions`.

    Raises:
        ValueError: If ``window`` is not one of the three options.
    """
    import pandas as pd

    if window not in {"all", "dev", "holdout"}:
        raise ValueError(f"window must be 'all', 'dev' or 'holdout', got {window!r}")

    frame = _read("olist_extract.csv.gz")
    frame = frame[frame.order_status == "delivered"] if drop_undelivered else frame
    purchase = pd.to_datetime(frame.order_purchase_timestamp)
    estimated = pd.to_datetime(frame.order_estimated_delivery_date)
    frame = frame.assign(
        promise_days=(estimated - purchase).dt.total_seconds() / 86400.0,
        purchase_ts=purchase,
    ).dropna(subset=["promise_days", "review_score", "price"])

    cut = pd.Timestamp("2018-02-28 15:00:37")
    if window == "dev":
        frame = frame[frame.purchase_ts <= cut]
    elif window == "holdout":
        frame = frame[frame.purchase_ts > cut]
    frame = frame.reset_index(drop=True)

    actions = pd.qcut(frame.promise_days, 4, labels=False, duplicates="drop")
    keep = actions.notna()
    frame, actions = frame[keep].reset_index(drop=True), actions[keep].astype(int)

    same_state = (frame.customer_state == frame.seller_state).astype(float)
    state = np.column_stack([
        frame.price.to_numpy(float),
        frame.freight_value.to_numpy(float),
        same_state.to_numpy(),
    ])
    return LoggedDecisions(
        y=frame.review_score.to_numpy(float),
        actions=actions.to_numpy(),
        state=state,
        clusters=frame.customer_state.to_numpy(),
        feature_names=("price", "freight_value", "same_state"),
        action_names=("Q1 shortest", "Q2", "Q3", "Q4 longest"),
        name=f"Olist ({window})",
        reward_name="review score (1-5)",
        time_order=frame.purchase_ts.astype("int64").to_numpy(),
    )


def load_dataco(*, reward: str = "margin") -> LoggedDecisions:
    """The DataCo shipping-mode log.

    The decision is the shipping mode; the reward is profit per order, or the
    on-time indicator. Clusters are the order regions.

    The on-time reward is included because the paper's audit refuses it at the
    first gate: on-time delivery is an accounting identity of the shipping mode
    in 97.55% of rows, so no sample size can rescue it. Auditing it is the
    clearest demonstration of what condition C1 is for.

    Args:
        reward: ``"margin"`` for profit per order, or ``"on_time"`` for the
            delivery-risk indicator.

    Returns:
        A :class:`LoggedDecisions`.

    Raises:
        ValueError: If ``reward`` is not one of the two options.
    """
    if reward not in {"margin", "on_time"}:
        raise ValueError(f"reward must be 'margin' or 'on_time', got {reward!r}")

    frame = _read("dataco_extract.csv.gz")
    modes = {"Standard Class": 0, "First Class": 1, "Second Class": 2, "Same Day": 3}
    frame = frame[frame["Shipping Mode"].isin(modes)].reset_index(drop=True)

    categorical = ["Customer Segment", "Market", "Category Name"]
    codes = [
        frame[column].astype("category").cat.codes.to_numpy(float)
        for column in categorical
    ]
    columns = [
        frame["Order Item Quantity"].to_numpy(float),
        frame["Product Price"].to_numpy(float),
        *codes,
    ]
    names = ["quantity", "product_price",
             *[c.lower().replace(" ", "_") for c in categorical]]

    if reward == "margin":
        y = frame["Order Profit Per Order"].to_numpy(float)
        reward_name = "profit per order"
    else:
        y = 1.0 - frame["Late_delivery_risk"].to_numpy(float)
        reward_name = "on-time delivery (identity of the action)"
        # The scheduled shipping days are set with the mode, so they belong in
        # the state for this reward. Including them is what makes C1 fail: the
        # outcome is then recoverable as (real <= scheduled) in 97.55% of rows,
        # and the audit refuses the reward instead of measuring a coupling that
        # would be real, large, and worth nothing.
        columns.append(frame["Days for shipment (scheduled)"].to_numpy(float))
        columns.append(frame["Days for shipping (real)"].to_numpy(float))
        names += ["days_scheduled", "days_real"]

    state = np.column_stack(columns)
    return LoggedDecisions(
        y=y,
        actions=frame["Shipping Mode"].map(modes).to_numpy(),
        state=state,
        clusters=frame["Order Region"].to_numpy(),
        feature_names=tuple(names),
        action_names=tuple(modes),
        name="DataCo",
        reward_name=reward_name,
    )
