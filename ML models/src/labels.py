"""Label rules used as ground truth for the supervised models.

We have no expert-tagged labels, so labels are derived deterministically
from sales/stock statistics. Models then learn the multi-feature decision
boundary, which is useful for: (1) generalizing the rule to new products,
(2) producing calibrated probabilities, (3) ranking by confidence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEMAND_CLASSES = ["slow", "medium", "fast"]


def label_demand(feats: pd.DataFrame) -> pd.Series:
    """Quantile-based demand label on total_sales.

    Bottom 50% by sales -> slow, next 30% -> medium, top 20% -> fast.
    Products with zero sales are forced into 'slow'.
    """
    sales = feats["total_sales"].astype(float)
    nonzero = sales[sales > 0]
    if len(nonzero) < 10:
        return pd.Series(["slow"] * len(sales), index=feats.index)

    q50, q80 = np.quantile(nonzero, [0.5, 0.8])
    labels = np.where(sales <= 0, "slow",
              np.where(sales <= q50, "slow",
              np.where(sales <= q80, "medium", "fast")))
    return pd.Series(labels, index=feats.index)


def label_dead_stock(feats: pd.DataFrame) -> pd.Series:
    """Binary dead-stock label.

    A product is 'dead stock' when it carries inventory but has not sold:
      total_stock > 0 AND total_sales == 0
    OR when the stock-to-sales ratio is severely high (top 10%) AND
    sell_through is very low (< 5%).
    """
    no_sales_with_stock = (feats["total_stock"] > 0) & (feats["total_sales"] <= 0)

    ratio = feats["stock_to_sales_ratio"].replace([np.inf, -np.inf], np.nan)
    high_ratio_threshold = ratio.quantile(0.90) if ratio.notna().any() else np.inf
    sluggish = (ratio >= high_ratio_threshold) & (feats["sell_through"] < 0.05) & (feats["total_stock"] > 0)

    return (no_sales_with_stock | sluggish).astype(int)
