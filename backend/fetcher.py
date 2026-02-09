"""
Data fetcher for PSX stock data.
Primary: psx-data-reader library
Fallback: Direct scraping of dps.psx.com.pk
"""
import datetime
import time
import logging
import json
import pickle
from typing import Optional, Dict, List, Tuple

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from backend.config import (
    PSX_LISTINGS_URL, PSX_COMPANY_URL, PSX_HISTORICAL_URL,
    HEADERS, REQUEST_DELAY, MAX_RETRIES, BACKOFF_FACTOR,
    REQUEST_TIMEOUT, HISTORY_START_DATE, CACHE_DIR, ERROR_LOG_FILE
)

logger = logging.getLogger(__name__)


def _setup_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    return session


SESSION = _setup_session()


def _retry_request(url: str, method: str = "GET", **kwargs) -> Optional[requests.Response]:
    for attempt in range(MAX_RETRIES):
        try:
            kwargs.setdefault("timeout", REQUEST_TIMEOUT)
            resp = SESSION.request(method, url, **kwargs)
            if resp.status_code == 429:
                wait = BACKOFF_FACTOR ** (attempt + 1)
                logger.warning(f"Rate limited on {url}, waiting {wait}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except Exception as e:
            wait = BACKOFF_FACTOR ** attempt
            logger.warning(f"Attempt {attempt+1}/{MAX_RETRIES} failed for {url}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
    return None


# ── Fetch All Listed Companies ─────────────────────────────────────────────────

def fetch_tickers_psx_lib() -> Optional[pd.DataFrame]:
    """Try psx-data-reader library first."""
    try:
        from psx import tickers as psx_tickers
        df = psx_tickers()
        if df is not None and not df.empty:
            logger.info(f"psx-data-reader returned {len(df)} tickers")
            return df
    except Exception as e:
        logger.warning(f"psx-data-reader tickers() failed: {e}")
    return None


def fetch_tickers_scrape() -> Optional[pd.DataFrame]:
    """Fallback: scrape dps.psx.com.pk/listings."""
    try:
        resp = _retry_request(PSX_LISTINGS_URL)
        if resp is None:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table")
        if table is None:
            # Try parsing the page differently - PSX uses dynamic content
            # Look for script tags with JSON data
            scripts = soup.find_all("script")
            for script in scripts:
                if script.string and "listing" in script.string.lower():
                    logger.info("Found listing data in script tag")
                    break

            # Alternative: try the sector summary page for company list
            return _fetch_tickers_from_sectors()

        rows = []
        headers_row = table.find("thead")
        if headers_row:
            cols = [th.get_text(strip=True) for th in headers_row.find_all("th")]
        else:
            cols = ["Symbol", "Company", "Sector", "Status"]

        for tr in table.find_all("tr")[1:]:
            tds = tr.find_all("td")
            if len(tds) >= 2:
                row = {}
                for i, td in enumerate(tds):
                    if i < len(cols):
                        row[cols[i]] = td.get_text(strip=True)
                    # Also check for links containing symbol
                    a_tag = td.find("a")
                    if a_tag and "href" in a_tag.attrs:
                        href = a_tag["href"]
                        if "/company/" in href:
                            row["Symbol"] = href.split("/company/")[-1].strip("/")
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            logger.info(f"Scraped {len(df)} tickers from PSX listings")
            return df
    except Exception as e:
        logger.error(f"Scraping PSX listings failed: {e}")
    return None


def _fetch_tickers_from_sectors() -> Optional[pd.DataFrame]:
    """Try getting company list from sector summary page."""
    try:
        resp = _retry_request("https://dps.psx.com.pk/sector-summary")
        if resp is None:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Parse sector data
        rows = []
        tables = soup.find_all("table")
        for table in tables:
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if tds:
                    row_data = [td.get_text(strip=True) for td in tds]
                    if len(row_data) >= 2:
                        rows.append(row_data)
        if rows:
            logger.info(f"Found {len(rows)} entries from sector summary")
    except Exception as e:
        logger.warning(f"Sector summary scrape failed: {e}")
    return None


def fetch_all_tickers() -> pd.DataFrame:
    """Get all listed company tickers. Tries library first, then scraping."""
    cache_file = CACHE_DIR / "tickers.pkl"

    # Try cache first (valid for 24h)
    if cache_file.exists():
        try:
            cached = pickle.load(open(cache_file, "rb"))
            cache_time = cached.get("time")
            if cache_time and (datetime.datetime.now() - cache_time).total_seconds() < 86400:
                logger.info(f"Using cached tickers ({len(cached['data'])} symbols)")
                return cached["data"]
        except Exception:
            pass

    # Try psx-data-reader
    df = fetch_tickers_psx_lib()

    # Fallback to scraping
    if df is None or df.empty:
        df = fetch_tickers_scrape()

    if df is None or df.empty:
        logger.error("Could not fetch tickers from any source")
        return pd.DataFrame()

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure 'symbol' column exists
    symbol_cols = [c for c in df.columns if "symbol" in c or "ticker" in c]
    if symbol_cols:
        df = df.rename(columns={symbol_cols[0]: "symbol"})
    elif "name" in df.columns and df.columns[0] != "symbol":
        # If first column looks like symbols (short uppercase strings)
        first_col = df.columns[0]
        if df[first_col].str.match(r'^[A-Z]{2,10}$').mean() > 0.5:
            df = df.rename(columns={first_col: "symbol"})

    # Ensure sector column
    sector_cols = [c for c in df.columns if "sector" in c]
    if sector_cols and sector_cols[0] != "sector":
        df = df.rename(columns={sector_cols[0]: "sector"})

    # Clean symbols first
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
        df = df[df["symbol"].str.len() > 0]
        df = df.drop_duplicates(subset=["symbol"])

    total_before = len(df)

    # Filter out debt instruments
    if "isdebt" in df.columns:
        df = df[df["isdebt"] != True]
    # Filter out ETFs
    if "isetf" in df.columns:
        df = df[df["isetf"] != True]

    # Filter by name: remove rights, warrants, preference shares, TFCs, sukuks
    if "name" in df.columns:
        name_str = df["name"].astype(str)
        bad_name = (
            name_str.str.contains(r'\bRight\b', case=False, na=False) |
            name_str.str.contains(r'\bWarrant\b', case=False, na=False) |
            name_str.str.contains(r'\bPreference\b', case=False, na=False) |
            name_str.str.contains(r'\bTFC\b', case=False, na=False) |
            name_str.str.contains(r'\bSukuk\b', case=False, na=False) |
            name_str.str.contains(r'\bMorabaha\b', case=False, na=False) |
            name_str.str.contains(r'\bCertificate\b', case=False, na=False) |
            name_str.str.contains(r'\bBond\b', case=False, na=False)
        )
        df = df[~bad_name]

    # Filter by symbol pattern: remove rights (R1/R2), GEM board, preference shares
    if "symbol" in df.columns:
        sym_str = df["symbol"].astype(str)
        bad_sym = (
            sym_str.str.match(r'^GEM', na=False) |            # GEM board symbols
            sym_str.str.match(r'.*R\d+$', na=False) |         # Rights issues (ASCR1, FFLR1, etc.)
            sym_str.str.match(r'.*PPS$', na=False) |           # Preference shares
            sym_str.str.match(r'.*PPA$', na=False) |           # Preference shares alt
            sym_str.str.match(r'.*WR\d*$', na=False) |        # Warrants
            sym_str.str.match(r'.*TFC\d*$', na=False) |       # Term Finance Certificates
            sym_str.str.contains(r'NCPS', na=False) |          # Non-convertible preference shares
            sym_str.str.contains(r'SUKUK', na=False)           # Sukuks
        )
        df = df[~bad_sym]

    # Filter by sector: remove debt-related sectors
    if "sector" in df.columns:
        bad_sectors = df["sector"].astype(str).str.upper()
        sector_filter = (
            bad_sectors.str.contains("MODARABA", na=False) |
            bad_sectors.str.contains("LEASING", na=False)
        )
        # Don't filter modarabas/leasing - they're legitimate equity sectors
        # Only filter if sector name literally says "DEBT" or similar
        # (keeping this conservative)

    filtered = total_before - len(df)
    logger.info(f"Filtered out {filtered} non-equity instruments ({total_before} -> {len(df)})")

    # Cache
    try:
        pickle.dump({"data": df, "time": datetime.datetime.now()}, open(cache_file, "wb"))
    except Exception:
        pass

    logger.info(f"Fetched {len(df)} tickers total")
    return df


# ── Fetch Historical Data Per Symbol ───────────────────────────────────────────

def _fetch_history_post(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """
    Primary method: POST to dps.psx.com.pk/historical.
    Returns HTML table with DATE, OPEN, HIGH, LOW, CLOSE, VOLUME.
    Fetches month by month and concatenates.
    """
    start_dt = datetime.datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end, "%Y-%m-%d")
    all_rows = []

    current = start_dt
    while current <= end_dt:
        try:
            resp = SESSION.post(
                PSX_HISTORICAL_URL,
                data={"symbol": symbol, "month": str(current.month), "year": str(current.year)},
                timeout=REQUEST_TIMEOUT
            )
            if resp.status_code == 200 and len(resp.text) > 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                table = soup.find("table")
                if table:
                    for tr in table.find_all("tr")[1:]:  # Skip header
                        tds = tr.find_all("td")
                        if len(tds) >= 6:
                            row = {
                                "date": tds[0].get_text(strip=True),
                                "open": tds[1].get_text(strip=True).replace(",", ""),
                                "high": tds[2].get_text(strip=True).replace(",", ""),
                                "low": tds[3].get_text(strip=True).replace(",", ""),
                                "close": tds[4].get_text(strip=True).replace(",", ""),
                                "volume": tds[5].get_text(strip=True).replace(",", ""),
                            }
                            all_rows.append(row)
            time.sleep(0.15)  # Brief pause between month requests
        except Exception as e:
            logger.debug(f"POST historical {symbol} {current.year}-{current.month}: {e}")

        # Move to next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.set_index("date")
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_index()
    return df


def _fetch_history_timeseries(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fallback: GET /timeseries/eod/{symbol}.
    Returns JSON with data as list of [timestamp, close, volume, ?].
    Only provides close+volume (no OHLC), but better than nothing.
    """
    try:
        url = f"https://dps.psx.com.pk/timeseries/eod/{symbol}"
        resp = _retry_request(url)
        if resp is None:
            return None

        data = resp.json()
        if not data or data.get("status") != 1 or not data.get("data"):
            return None

        records = data["data"]
        rows = []
        for rec in records:
            if isinstance(rec, list) and len(rec) >= 3:
                ts = rec[0]
                close = rec[1]
                volume = rec[2]
                dt = datetime.datetime.fromtimestamp(ts)
                rows.append({"date": dt, "close": close, "volume": volume})

        if not rows:
            return None

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for c in ["close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.sort_index()
        return df

    except Exception as e:
        logger.debug(f"Timeseries EOD for {symbol} failed: {e}")
    return None


def fetch_stock_history(symbol: str,
                        start: str = HISTORY_START_DATE,
                        end: str = None) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV for a single symbol."""
    if end is None:
        end = datetime.date.today().strftime("%Y-%m-%d")

    cache_file = CACHE_DIR / f"{symbol}.pkl"

    # Check cache
    if cache_file.exists():
        try:
            cached = pickle.load(open(cache_file, "rb"))
            cache_time = cached.get("time")
            if cache_time and (datetime.datetime.now() - cache_time).total_seconds() < 86400:
                return cached["data"]
        except Exception:
            pass

    # Primary: /timeseries/eod (fast, single request, close + volume)
    df = _fetch_history_timeseries(symbol)

    # NOTE: POST /historical fallback disabled - too slow (180+ requests/symbol).
    # timeseries/eod provides close+volume which is sufficient for all metrics.

    if df is None or df.empty:
        return None

    # Normalize columns
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Ensure date index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df.index.name = "date"

    # Keep only OHLCV columns that exist
    ohlcv = ["open", "high", "low", "close", "volume"]
    existing = [c for c in ohlcv if c in df.columns]
    df = df[existing]

    # Convert to numeric
    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows with no close price
    if "close" in df.columns:
        df = df.dropna(subset=["close"])

    # Sort by date
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Drop NaT index
    df = df[df.index.notna()]

    # Cache
    try:
        pickle.dump({"data": df, "time": datetime.datetime.now()}, open(cache_file, "wb"))
    except Exception:
        pass

    return df


def fetch_all_stocks(tickers_df: pd.DataFrame,
                     start: str = HISTORY_START_DATE) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols with progress bar."""
    if "symbol" not in tickers_df.columns:
        logger.error("No 'symbol' column in tickers DataFrame")
        return {}

    symbols = tickers_df["symbol"].tolist()
    results = {}
    failed = []
    end = datetime.date.today().strftime("%Y-%m-%d")

    logger.info(f"Fetching historical data for {len(symbols)} symbols...")

    for symbol in tqdm(symbols, desc="Fetching stocks", unit="sym"):
        try:
            df = fetch_stock_history(symbol, start, end)
            if df is not None and not df.empty and len(df) >= 5:
                results[symbol] = df
            else:
                failed.append((symbol, "No data or too few rows"))
        except Exception as e:
            failed.append((symbol, str(e)))
            logger.error(f"Failed to fetch {symbol}: {e}")

        time.sleep(REQUEST_DELAY)

    # Log failures
    if failed:
        with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Fetch run: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Failed symbols ({len(failed)}/{len(symbols)}):\n")
            for sym, reason in failed:
                f.write(f"  {sym}: {reason}\n")

    logger.info(f"Successfully fetched {len(results)}/{len(symbols)} symbols "
                f"({len(failed)} failed)")
    return results
