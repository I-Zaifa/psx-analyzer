"""
Macro data fetcher: KSE-100 index, USD/PKR, CPI/inflation, T-bill rates.
Uses yfinance and FRED with bundled CSV fallbacks.
"""
import datetime
import logging
import pickle
from typing import Optional

import pandas as pd
import numpy as np

from backend.config import (
    KSE100_TICKER, USDPKR_TICKER,
    FRED_CPI_SERIES, FRED_POLICY_RATE,
    HISTORY_START_DATE, CACHE_DIR, BUNDLED_DIR,
    DEFAULT_RISK_FREE_RATE
)

logger = logging.getLogger(__name__)


def _load_cached(name: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"macro_{name}.pkl"
    if cache_file.exists():
        try:
            cached = pickle.load(open(cache_file, "rb"))
            age = (datetime.datetime.now() - cached["time"]).total_seconds()
            if age < max_age_hours * 3600:
                return cached["data"]
        except Exception:
            pass
    return None


def _save_cache(name: str, df: pd.DataFrame):
    cache_file = CACHE_DIR / f"macro_{name}.pkl"
    try:
        pickle.dump({"data": df, "time": datetime.datetime.now()}, open(cache_file, "wb"))
    except Exception:
        pass


def _load_bundled(name: str) -> Optional[pd.DataFrame]:
    csv_file = BUNDLED_DIR / f"{name}.csv"
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file, parse_dates=["date"], index_col="date")
            logger.info(f"Loaded bundled {name} data ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Failed to load bundled {name}: {e}")
    return None


# ── KSE-100 Index ─────────────────────────────────────────────────────────────

def _parse_psx_timeseries(records: list) -> pd.DataFrame:
    """Parse PSX timeseries JSON records into a daily DataFrame."""
    rows = []
    for rec in records:
        if isinstance(rec, list) and len(rec) >= 2:
            ts = rec[0]
            close_val = float(rec[1])
            dt = datetime.datetime.fromtimestamp(ts)
            rows.append({"date": dt.date(), "close": close_val})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    # Keep only one row per day (last observation)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date").sort_index()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()
    return df


def fetch_kse100() -> pd.DataFrame:
    """Fetch KSE-100 index historical data.
    
    Sources:
    1. PSX Data Portal /timeseries/eod/KSE100 (daily close, 2021-present)
    2. yfinance ^KSE (daily OHLCV, 2010-2021, stale but useful for history)
    Both are combined to get the longest possible series.
    """
    cached = _load_cached("kse100")
    if cached is not None:
        return cached

    parts = []

    # Source 1: PSX Data Portal — /timeseries/eod/KSE100 (daily, 2021-present)
    try:
        import requests
        from backend.config import HEADERS
        resp = requests.get(
            "https://dps.psx.com.pk/timeseries/eod/KSE100",
            headers=HEADERS,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and data.get("status") == 1 and data.get("data"):
                psx_df = _parse_psx_timeseries(data["data"])
                if not psx_df.empty and len(psx_df) > 50:
                    logger.info(f"PSX timeseries/eod/KSE100: {len(psx_df)} daily rows, "
                                f"{psx_df.index.min().date()} to {psx_df.index.max().date()}")
                    parts.append(psx_df)
    except Exception as e:
        logger.warning(f"PSX KSE-100 timeseries/eod failed: {e}")

    # Source 2: yfinance ^KSE (stale, stops ~2021, but gives pre-2021 history)
    try:
        import yfinance as yf
        ticker = yf.Ticker(KSE100_TICKER)
        yf_df = ticker.history(start=HISTORY_START_DATE, end=datetime.date.today().strftime("%Y-%m-%d"))
        if yf_df is not None and not yf_df.empty and len(yf_df) > 100:
            yf_df.index = pd.to_datetime(yf_df.index).tz_localize(None)
            yf_df.index.name = "date"
            yf_df.columns = [c.lower().replace(" ", "_") for c in yf_df.columns]
            if "close" in yf_df.columns:
                yf_close = yf_df[["close"]].copy()
                logger.info(f"yfinance ^KSE: {len(yf_close)} rows, "
                            f"{yf_close.index.min().date()} to {yf_close.index.max().date()}")
                parts.append(yf_close)
    except Exception as e:
        logger.warning(f"yfinance KSE-100 failed: {e}")

    # Combine: yfinance (older) + PSX (newer), PSX takes precedence on overlap
    if parts:
        combined = pd.concat(parts)
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
        combined = combined.dropna(subset=["close"])
        if len(combined) > 50:
            _save_cache("kse100", combined)
            logger.info(f"KSE-100 combined: {len(combined)} rows, "
                        f"{combined.index.min().date()} to {combined.index.max().date()}")
            return combined

    # Fallback to bundled
    bundled = _load_bundled("kse100")
    if bundled is not None:
        return bundled

    logger.error("Could not fetch KSE-100 data from any source")
    return pd.DataFrame()


# ── USD/PKR Exchange Rate ──────────────────────────────────────────────────────

def _clean_fx_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean FX data: remove outliers where PKR/USD should be 50-500 range."""
    if "close" in df.columns:
        valid = (df["close"] > 50) & (df["close"] < 500)
        before = len(df)
        df = df[valid]
        removed = before - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} outlier FX rows (outside 50-500 range)")
    return df


def fetch_usdpkr() -> pd.DataFrame:
    """Fetch USD/PKR historical exchange rate."""
    cached = _load_cached("usdpkr")
    if cached is not None:
        return cached

    # Try yfinance with PKR=X (more reliable than USDPKR=X)
    for ticker_sym in [USDPKR_TICKER, "PKR=X"]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(ticker_sym)
            df = ticker.history(start=HISTORY_START_DATE, end=datetime.date.today().strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "date"
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                keep = [c for c in ["open", "high", "low", "close"] if c in df.columns]
                df = df[keep]
                df = _clean_fx_data(df)
                if not df.empty and len(df) > 100:
                    _save_cache("usdpkr", df)
                    logger.info(f"Fetched USD/PKR from yfinance ({ticker_sym}): {len(df)} rows")
                    return df
        except Exception as e:
            logger.warning(f"yfinance {ticker_sym} failed: {e}")

    # Fallback to bundled
    bundled = _load_bundled("usdpkr")
    if bundled is not None:
        return bundled

    logger.error("Could not fetch USD/PKR data from any source")
    return pd.DataFrame()


# ── Pakistan CPI / Inflation ──────────────────────────────────────────────────

def fetch_cpi() -> pd.DataFrame:
    """Fetch Pakistan CPI/inflation data."""
    cached = _load_cached("cpi")
    if cached is not None:
        return cached

    # Try World Bank API first (internationally standard annual CPI figures)
    try:
        wb_url = ("https://api.worldbank.org/v2/country/PAK/indicator/"
                  "FP.CPI.TOTL.ZG?format=json&per_page=500&date=2010:2026")
        import requests
        resp = requests.get(wb_url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if len(data) > 1 and data[1]:
                records = []
                for item in data[1]:
                    if item.get("value") is not None:
                        records.append({
                            "date": f"{item['date']}-01-01",
                            "cpi_yoy": float(item["value"])
                        })
                if records:
                    df = pd.DataFrame(records)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date").sort_index()
                    _save_cache("cpi", df)
                    logger.info(f"Fetched Pakistan CPI from World Bank: {len(df)} rows")
                    return df
    except Exception as e:
        logger.warning(f"World Bank CPI fetch failed: {e}")

    # Fallback to pandas-datareader with FRED
    try:
        import pandas_datareader.data as web
        df = web.DataReader(FRED_CPI_SERIES, "fred", start=HISTORY_START_DATE)
        if df is not None and not df.empty:
            df.index.name = "date"
            df.columns = ["cpi_yoy"]
            _save_cache("cpi", df)
            logger.info(f"Fetched Pakistan CPI from FRED: {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"FRED CPI fetch failed: {e}")

    # Fallback to bundled
    bundled = _load_bundled("cpi")
    if bundled is not None:
        return bundled

    logger.error("Could not fetch CPI data from any source")
    return pd.DataFrame()


# ── T-Bill / Policy Rate ──────────────────────────────────────────────────────

def fetch_tbill_rate() -> pd.DataFrame:
    """Fetch Pakistan T-bill / policy rate data."""
    cached = _load_cached("tbill")
    if cached is not None:
        return cached

    # Try pandas-datareader with FRED
    try:
        import pandas_datareader.data as web
        df = web.DataReader(FRED_POLICY_RATE, "fred", start=HISTORY_START_DATE)
        if df is not None and not df.empty:
            df.index.name = "date"
            df.columns = ["rate"]
            df["rate"] = df["rate"] / 100.0  # Convert to decimal
            _save_cache("tbill", df)
            logger.info(f"Fetched Pakistan policy rate from FRED: {len(df)} rows")
            return df
    except Exception as e:
        logger.warning(f"FRED policy rate fetch failed: {e}")

    # Fallback to bundled
    bundled = _load_bundled("tbill")
    if bundled is not None:
        return bundled

    logger.warning("Using default risk-free rate as fallback")
    return pd.DataFrame()


def get_avg_risk_free_rate() -> float:
    """Get average risk-free rate for Sharpe ratio calculation."""
    df = fetch_tbill_rate()
    if df is not None and not df.empty and "rate" in df.columns:
        return df["rate"].mean()
    return DEFAULT_RISK_FREE_RATE


# ── Fetch All Macro Data ──────────────────────────────────────────────────────

def fetch_all_macro() -> dict:
    """Fetch all macro data and return as dict of DataFrames."""
    logger.info("Fetching macro data...")
    macro = {
        "kse100": fetch_kse100(),
        "usdpkr": fetch_usdpkr(),
        "cpi": fetch_cpi(),
        "tbill": fetch_tbill_rate(),
        "risk_free_rate": get_avg_risk_free_rate(),
    }
    available = sum(1 for k, v in macro.items()
                    if k != "risk_free_rate" and isinstance(v, pd.DataFrame) and not v.empty)
    logger.info(f"Macro data: {available}/4 series available, "
                f"risk-free rate = {macro['risk_free_rate']:.2%}")
    return macro
