"""
Sector aggregation: group stocks by sector, compute sector-level stats.
"""
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def compute_sector_aggregates(metrics_df: pd.DataFrame,
                              tickers_df: pd.DataFrame,
                              index_return: Optional[float] = None) -> pd.DataFrame:
    """Compute sector-level aggregate statistics."""
    if "sector" not in tickers_df.columns:
        logger.warning("No 'sector' column in tickers data, skipping sector aggregation")
        return pd.DataFrame()

    # Merge sector info into metrics
    sector_map = tickers_df[["symbol", "sector"]].drop_duplicates()
    merged = metrics_df.merge(sector_map, on="symbol", how="left")

    # Drop rows without sector
    merged = merged.dropna(subset=["sector"])
    if merged.empty:
        logger.warning("No sector data available after merge")
        return pd.DataFrame()

    # Aggregate by sector
    agg_funcs = {}
    numeric_cols = {
        "total_return": ["mean", "median", "min", "max", "count"],
        "cagr": ["mean", "median"],
        "annualized_volatility": ["mean", "median"],
        "max_drawdown": ["mean", "min"],
        "sharpe_ratio": ["mean", "median"],
        "beta": ["mean", "median"],
        "correlation": ["mean"],
        "return_1m": ["mean"],
        "return_3m": ["mean"],
        "return_6m": ["mean"],
        "return_1y": ["mean"],
        "return_ytd": ["mean"],
        "real_cagr": ["mean"],
        "avg_daily_volume": ["sum", "mean"],
    }

    for col, funcs in numeric_cols.items():
        if col in merged.columns:
            agg_funcs[col] = funcs

    if not agg_funcs:
        logger.warning("No numeric columns to aggregate")
        return pd.DataFrame()

    sector_agg = merged.groupby("sector").agg(agg_funcs)

    # Flatten multi-level columns
    sector_agg.columns = ["_".join(col).strip("_") for col in sector_agg.columns]

    # Add company count
    if "total_return_count" in sector_agg.columns:
        sector_agg = sector_agg.rename(columns={"total_return_count": "num_companies"})

    # Compute relative strength vs index
    if index_return is not None and "total_return_mean" in sector_agg.columns:
        sector_agg["relative_strength"] = sector_agg["total_return_mean"] - index_return

    # Sort by average return descending
    if "total_return_mean" in sector_agg.columns:
        sector_agg = sector_agg.sort_values("total_return_mean", ascending=False)

    sector_agg.index.name = "sector"
    logger.info(f"Computed aggregates for {len(sector_agg)} sectors")
    return sector_agg


def get_top_bottom_performers(metrics_df: pd.DataFrame,
                              tickers_df: pd.DataFrame,
                              n: int = 3) -> dict:
    """Get top and bottom N performers per sector."""
    if "sector" not in tickers_df.columns:
        return {}

    sector_map = tickers_df[["symbol", "sector"]].drop_duplicates()
    merged = metrics_df.merge(sector_map, on="symbol", how="left")
    merged = merged.dropna(subset=["sector", "total_return"])

    result = {}
    for sector, group in merged.groupby("sector"):
        sorted_group = group.sort_values("total_return", ascending=False)
        top = sorted_group.head(n)[["symbol", "total_return", "cagr", "sharpe_ratio"]].to_dict("records")
        bottom = sorted_group.tail(n)[["symbol", "total_return", "cagr", "sharpe_ratio"]].to_dict("records")
        result[sector] = {"top": top, "bottom": bottom}

    return result


def get_sector_summary(metrics_df: pd.DataFrame,
                       tickers_df: pd.DataFrame) -> pd.DataFrame:
    """Get a clean sector summary table for display."""
    if "sector" not in tickers_df.columns:
        return pd.DataFrame()

    sector_map = tickers_df[["symbol", "sector"]].drop_duplicates()
    merged = metrics_df.merge(sector_map, on="symbol", how="left")
    merged = merged.dropna(subset=["sector"])

    summary_rows = []
    for sector, group in merged.groupby("sector"):
        row = {
            "sector": sector,
            "num_companies": len(group),
            "avg_total_return": group["total_return"].mean() if "total_return" in group else None,
            "avg_cagr": group["cagr"].mean() if "cagr" in group else None,
            "avg_volatility": group["annualized_volatility"].mean() if "annualized_volatility" in group else None,
            "avg_sharpe": group["sharpe_ratio"].mean() if "sharpe_ratio" in group else None,
            "avg_beta": group["beta"].mean() if "beta" in group else None,
            "avg_max_drawdown": group["max_drawdown"].mean() if "max_drawdown" in group else None,
            "best_performer": None,
            "worst_performer": None,
            "total_volume": group["avg_daily_volume"].sum() if "avg_daily_volume" in group else None,
        }

        if "total_return" in group.columns and group["total_return"].notna().any():
            best_idx = group["total_return"].idxmax()
            worst_idx = group["total_return"].idxmin()
            row["best_performer"] = group.loc[best_idx, "symbol"]
            row["worst_performer"] = group.loc[worst_idx, "symbol"]

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values("avg_total_return", ascending=False, na_position="last")
    return summary
