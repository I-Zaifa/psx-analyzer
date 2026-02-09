"""
CSV export module: generates master, sector, and per-symbol CSV files.
"""
import logging
from typing import Dict, Optional

import pandas as pd

from backend.config import CSV_DIR, SYMBOLS_CSV_DIR

logger = logging.getLogger(__name__)


def export_master_csv(metrics_df: pd.DataFrame,
                      fundamentals_df: Optional[pd.DataFrame] = None,
                      tickers_df: Optional[pd.DataFrame] = None) -> str:
    """Export master CSV with all companies and summary metrics."""
    df = metrics_df.copy()

    # Merge fundamentals
    if fundamentals_df is not None and not fundamentals_df.empty:
        fund_cols = [c for c in fundamentals_df.columns if c != "symbol" or c == "symbol"]
        # Avoid duplicate columns
        existing = set(df.columns) - {"symbol"}
        fund_only = fundamentals_df[[c for c in fund_cols if c not in existing or c == "symbol"]]
        if len(fund_only.columns) > 1:
            df = df.merge(fund_only, on="symbol", how="left")

    # Merge sector/company info
    if tickers_df is not None and not tickers_df.empty:
        info_cols = ["symbol"]
        for c in ["sector", "company", "name", "company_name", "listing_date", "status"]:
            if c in tickers_df.columns and c not in df.columns:
                info_cols.append(c)
        if len(info_cols) > 1:
            df = df.merge(tickers_df[info_cols].drop_duplicates(), on="symbol", how="left")

    # Reorder columns: symbol and sector first
    priority = ["symbol", "sector", "company", "name", "company_name"]
    first_cols = [c for c in priority if c in df.columns]
    other_cols = [c for c in df.columns if c not in first_cols]
    df = df[first_cols + other_cols]

    # Sort by symbol
    df = df.sort_values("symbol")

    path = CSV_DIR / "master.csv"
    df.to_csv(path, index=False, float_format="%.6f")
    logger.info(f"Exported master CSV: {path} ({len(df)} rows, {len(df.columns)} columns)")
    return str(path)


def export_sector_csv(sector_df: pd.DataFrame) -> str:
    """Export sector aggregates CSV."""
    path = CSV_DIR / "sectors.csv"
    sector_df.to_csv(path, index=False, float_format="%.6f")
    logger.info(f"Exported sector CSV: {path} ({len(sector_df)} sectors)")
    return str(path)


def export_symbol_csvs(stocks_data: Dict[str, pd.DataFrame]) -> int:
    """Export individual CSV files per symbol."""
    count = 0
    for symbol, df in stocks_data.items():
        try:
            path = SYMBOLS_CSV_DIR / f"{symbol}.csv"
            df.to_csv(path, float_format="%.4f")
            count += 1
        except Exception as e:
            logger.error(f"Failed to export {symbol} CSV: {e}")

    logger.info(f"Exported {count}/{len(stocks_data)} individual symbol CSVs to {SYMBOLS_CSV_DIR}")
    return count


def export_macro_csv(macro_data: dict) -> None:
    """Export macro data CSVs."""
    for name in ["kse100", "usdpkr", "cpi", "tbill"]:
        data = macro_data.get(name)
        if isinstance(data, pd.DataFrame) and not data.empty:
            path = CSV_DIR / f"macro_{name}.csv"
            data.to_csv(path, float_format="%.4f")
            logger.info(f"Exported macro CSV: {path}")


def export_all(metrics_df: pd.DataFrame,
               sector_df: pd.DataFrame,
               stocks_data: Dict[str, pd.DataFrame],
               macro_data: dict,
               fundamentals_df: Optional[pd.DataFrame] = None,
               tickers_df: Optional[pd.DataFrame] = None) -> dict:
    """Export all CSVs and return paths."""
    results = {}
    results["master"] = export_master_csv(metrics_df, fundamentals_df, tickers_df)
    results["sectors"] = export_sector_csv(sector_df)
    results["symbols_count"] = export_symbol_csvs(stocks_data)
    export_macro_csv(macro_data)
    return results
