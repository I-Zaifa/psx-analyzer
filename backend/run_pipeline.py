"""
Pipeline orchestrator: fetch → compute → export.
Ties all backend modules together for a complete run.
"""
import sys
import time
import logging
import datetime

from backend.config import PIPELINE_LOG_FILE, ERROR_LOG_FILE
from backend.fetcher import fetch_all_tickers, fetch_all_stocks
from backend.macro import fetch_all_macro
from backend.fundamentals import fetch_all_fundamentals
from backend.metrics import compute_metrics_for_all
from backend.sectors import compute_sector_aggregates, get_sector_summary
from backend.export import export_all

logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the pipeline."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                            datefmt="%H:%M:%S")
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # File handler
    fh = logging.FileHandler(PIPELINE_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)


def run_pipeline(skip_fundamentals: bool = False, max_symbols: int = None):
    """
    Run the complete data pipeline.

    Args:
        skip_fundamentals: Skip fundamentals scraping (faster run)
        max_symbols: Limit number of symbols to process (for testing)
    """
    setup_logging()
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("PSX Analyzer Pipeline Starting")
    logger.info(f"Time: {datetime.datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Step 1: Fetch all tickers
    logger.info("\n[1/6] Fetching listed companies...")
    tickers_df = fetch_all_tickers()
    if tickers_df.empty:
        logger.error("No tickers found. Aborting pipeline.")
        return False
    logger.info(f"Found {len(tickers_df)} listed companies")

    # Limit symbols if requested (for testing)
    if max_symbols and max_symbols < len(tickers_df):
        logger.info(f"Limiting to {max_symbols} symbols (test mode)")
        tickers_df = tickers_df.head(max_symbols)

    # Step 2: Fetch macro data
    logger.info("\n[2/6] Fetching macro data...")
    macro_data = fetch_all_macro()

    # Get index prices for beta/correlation
    kse100 = macro_data.get("kse100")
    index_prices = None
    if isinstance(kse100, dict):
        index_prices = None
    elif kse100 is not None and not kse100.empty and "close" in kse100.columns:
        index_prices = kse100["close"]

    risk_free_rate = macro_data.get("risk_free_rate", 0.10)
    cpi_data = macro_data.get("cpi")
    if isinstance(cpi_data, dict):
        cpi_data = None

    # Step 3: Fetch historical stock data
    logger.info("\n[3/6] Fetching historical stock data...")
    stocks_data = fetch_all_stocks(tickers_df)
    if not stocks_data:
        logger.error("No stock data fetched. Aborting pipeline.")
        return False
    logger.info(f"Fetched data for {len(stocks_data)} stocks")

    # Step 4: Compute metrics
    logger.info("\n[4/6] Computing metrics...")
    metrics_df = compute_metrics_for_all(
        stocks_data,
        index_prices=index_prices,
        risk_free_rate=risk_free_rate,
        cpi_data=cpi_data if cpi_data is not None else None
    )

    # Step 5: Fundamentals
    fundamentals_df = None
    if not skip_fundamentals:
        logger.info("\n[5/6] Fetching fundamentals...")
        symbols = list(stocks_data.keys())
        fundamentals_df = fetch_all_fundamentals(symbols)
    else:
        logger.info("\n[5/6] Skipping fundamentals (--skip-fundamentals)")

    # Step 6: Sector aggregation & export
    logger.info("\n[6/6] Computing sector aggregates and exporting...")

    # Compute index total return for relative strength
    index_return = None
    if index_prices is not None and len(index_prices) >= 2:
        index_return = (index_prices.iloc[-1] - index_prices.iloc[0]) / index_prices.iloc[0]

    sector_df = get_sector_summary(metrics_df, tickers_df)

    # Export everything
    export_results = export_all(
        metrics_df=metrics_df,
        sector_df=sector_df,
        stocks_data=stocks_data,
        macro_data=macro_data,
        fundamentals_df=fundamentals_df,
        tickers_df=tickers_df
    )

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info(f"Stocks processed: {len(stocks_data)}")
    logger.info(f"Sectors: {len(sector_df)}")
    logger.info(f"Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    logger.info(f"Master CSV: {export_results.get('master')}")
    logger.info(f"Sector CSV: {export_results.get('sectors')}")
    logger.info(f"Symbol CSVs: {export_results.get('symbols_count')}")
    logger.info("=" * 60)

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PSX Analyzer Pipeline")
    parser.add_argument("--skip-fundamentals", action="store_true",
                        help="Skip fundamentals scraping")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="Limit number of symbols (for testing)")
    args = parser.parse_args()
    run_pipeline(skip_fundamentals=args.skip_fundamentals, max_symbols=args.max_symbols)
