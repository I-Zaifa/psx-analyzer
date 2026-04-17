"""
Fundamentals scraper: PE, PB, dividend yield, EPS from dps.psx.com.pk.
"""
import time
import logging
from typing import Optional, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from backend.config import (
    PSX_COMPANY_URL, HEADERS, REQUEST_DELAY,
    MAX_RETRIES, BACKOFF_FACTOR, REQUEST_TIMEOUT,
    ERROR_LOG_FILE
)

logger = logging.getLogger(__name__)


def _parse_number(text: str) -> Optional[float]:
    """Parse a number string, handling commas, parentheses (negative), etc."""
    if not text or text.strip() in ("-", "N/A", "n/a", "--", ""):
        return None
    text = text.strip().replace(",", "")
    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    try:
        val = float(text)
        return -val if negative else val
    except ValueError:
        return None


def fetch_company_fundamentals(symbol: str) -> Dict:
    """Scrape fundamental data for a single company from PSX data portal."""
    url = PSX_COMPANY_URL.format(symbol=symbol)
    result = {
        "symbol": symbol,
        "pe_ratio": None,
        "pb_ratio": None,
        "dividend_yield": None,
        "eps": None,
        "market_cap": None,
        "face_value": None,
        "shares_outstanding": None,
        "last_close": None,
        "52w_high": None,
        "52w_low": None,
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 429:
                time.sleep(BACKOFF_FACTOR ** (attempt + 1))
                continue
            if resp.status_code != 200:
                return result

            soup = BeautifulSoup(resp.text, "html.parser")

            # PSX data portal company pages have key stats in various formats
            # Look for data in tables, definition lists, or stat cards
            
            # Method 1: Look for stat cards / key-value pairs
            stats_divs = soup.find_all(["div", "span", "td", "dd", "li"])
            text_pairs = {}
            
            for el in stats_divs:
                text = el.get_text(strip=True)
                # Look for label: value patterns
                if ":" in text and len(text) < 100:
                    parts = text.split(":", 1)
                    if len(parts) == 2:
                        key = parts[0].strip().lower()
                        val = parts[1].strip()
                        text_pairs[key] = val

            # Method 2: Look for tables with financial data
            tables = soup.find_all("table")
            for table in tables:
                for tr in table.find_all("tr"):
                    tds = tr.find_all(["td", "th"])
                    if len(tds) >= 2:
                        key = tds[0].get_text(strip=True).lower()
                        val = tds[-1].get_text(strip=True)
                        text_pairs[key] = val

            # Method 3: Look for specific class patterns
            for div in soup.find_all(["div", "section"]):
                cls = " ".join(div.get("class", []))
                if any(k in cls.lower() for k in ["stat", "info", "detail", "summary", "quote"]):
                    labels = div.find_all(["label", "span", "dt", "th"])
                    values = div.find_all(["span", "dd", "td"])
                    for label in labels:
                        label_text = label.get_text(strip=True).lower()
                        next_el = label.find_next_sibling()
                        if next_el:
                            text_pairs[label_text] = next_el.get_text(strip=True)

            # Map found data to our fields
            for key, val in text_pairs.items():
                kl = key.lower()
                if any(x in kl for x in ["p/e", "pe ratio", "price/earning", "price earning", "pe"]):
                    result["pe_ratio"] = _parse_number(val)
                elif any(x in kl for x in ["p/b", "pb ratio", "price/book", "price book"]):
                    result["pb_ratio"] = _parse_number(val)
                elif any(x in kl for x in ["dividend yield", "div yield", "dy"]):
                    parsed = _parse_number(val.replace("%", ""))
                    if parsed is not None:
                        result["dividend_yield"] = parsed
                elif any(x in kl for x in ["eps", "earning per share", "earnings per share"]):
                    result["eps"] = _parse_number(val)
                elif any(x in kl for x in ["market cap", "mkt cap", "mcap"]):
                    result["market_cap"] = _parse_number(val)
                elif any(x in kl for x in ["face value", "par value"]):
                    result["face_value"] = _parse_number(val)
                elif any(x in kl for x in ["shares", "outstanding"]):
                    result["shares_outstanding"] = _parse_number(val)
                elif any(x in kl for x in ["last", "close", "current"]) and "52" not in kl:
                    if result["last_close"] is None:
                        result["last_close"] = _parse_number(val)
                elif "52" in kl and "high" in kl:
                    result["52w_high"] = _parse_number(val)
                elif "52" in kl and "low" in kl:
                    result["52w_low"] = _parse_number(val)

            return result

        except Exception as e:
            logger.debug(f"Fundamentals fetch attempt {attempt+1} for {symbol}: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)

    return result


def fetch_all_fundamentals(symbols: list) -> pd.DataFrame:
    """Fetch fundamentals for all symbols."""
    logger.info(f"Fetching fundamentals for {len(symbols)} symbols...")
    results = []
    failed = 0

    for symbol in tqdm(symbols, desc="Fetching fundamentals", unit="sym"):
        try:
            data = fetch_company_fundamentals(symbol)
            results.append(data)
        except Exception as e:
            logger.error(f"Fundamentals failed for {symbol}: {e}")
            results.append({"symbol": symbol})
            failed += 1
        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(results)
    has_data = df.drop(columns=["symbol"], errors="ignore").notna().any(axis=1).sum()
    logger.info(f"Fundamentals: {has_data}/{len(symbols)} symbols have data ({failed} errors)")
    return df
