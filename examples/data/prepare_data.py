"""Rebuild the bundled example extracts from the original public datasets.

The extracts shipped in this directory are *subsets of columns*, not modified
data: every row and every value is as published. They exist so the examples run
straight after a clone, without a 210 MB download.

Run this only to regenerate them, or to verify that they match the sources:

    python examples/data/prepare_data.py --olist /path/to/olist \\
                                         --dataco /path/to/DataCoSupplyChainDataset.csv

Sources:
    Olist Brazilian E-Commerce (Kaggle, CC BY-NC-SA 4.0)
        https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
    DataCo Smart Supply Chain (Mendeley Data, CC BY 4.0)
        https://data.mendeley.com/datasets/8gx2fvg2k6/5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

DATACO_COLUMNS = [
    "Shipping Mode",
    "Order Profit Per Order",
    "Order Item Quantity",
    "Product Price",
    "Customer Segment",
    "Market",
    "Order Region",
    "Category Name",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
]
SHIPPING_MODES = ("Standard Class", "First Class", "Second Class", "Same Day")


def prepare_dataco(source: Path, out: Path) -> None:
    """Extract the columns the audit uses from the DataCo log."""
    frame = pd.read_csv(
        source, encoding="latin-1", low_memory=False, usecols=DATACO_COLUMNS
    )
    frame = frame[frame["Shipping Mode"].isin(SHIPPING_MODES)]
    frame.to_csv(out, index=False, compression="gzip")
    print(f"  DataCo: {len(frame):,} rows -> {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


def prepare_olist(source_dir: Path, out: Path) -> None:
    """Join the Olist tables the audit uses into one order-level extract."""
    orders = pd.read_csv(
        source_dir / "olist_orders_dataset.csv",
        usecols=[
            "order_id", "customer_id", "order_purchase_timestamp",
            "order_estimated_delivery_date", "order_delivered_customer_date",
            "order_status",
        ],
    )
    reviews = pd.read_csv(
        source_dir / "olist_order_reviews_dataset.csv",
        usecols=["order_id", "review_score"],
    ).groupby("order_id", as_index=False).review_score.mean()
    items = (
        pd.read_csv(
            source_dir / "olist_order_items_dataset.csv",
            usecols=["order_id", "seller_id", "price", "freight_value"],
        )
        .groupby("order_id", as_index=False)
        .agg(
            seller_id=("seller_id", "first"),
            price=("price", "sum"),
            freight_value=("freight_value", "sum"),
        )
    )
    customers = pd.read_csv(
        source_dir / "olist_customers_dataset.csv",
        usecols=["customer_id", "customer_state"],
    )
    sellers = pd.read_csv(
        source_dir / "olist_sellers_dataset.csv",
        usecols=["seller_id", "seller_state"],
    )
    merged = (
        orders.merge(reviews, on="order_id")
        .merge(items, on="order_id")
        .merge(customers, on="customer_id")
        .merge(sellers, on="seller_id")
        .drop(columns=["customer_id", "seller_id"])
    )
    merged.to_csv(out, index=False, compression="gzip")
    print(f"  Olist:  {len(merged):,} rows -> {out.name} "
          f"({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olist", type=Path, help="directory of Olist CSVs")
    parser.add_argument("--dataco", type=Path, help="DataCoSupplyChainDataset.csv")
    args = parser.parse_args()

    if not args.olist and not args.dataco:
        parser.error("give --olist, --dataco, or both")
    print("Rebuilding example extracts:")
    if args.dataco:
        prepare_dataco(args.dataco, HERE / "dataco_extract.csv.gz")
    if args.olist:
        prepare_olist(args.olist, HERE / "olist_extract.csv.gz")


if __name__ == "__main__":
    main()
