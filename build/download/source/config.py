import os
from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
CSV_DIR = DATA_DIR / "csv"
SYMBOLS_CSV_DIR = CSV_DIR / "symbols"
BUNDLED_DIR = DATA_DIR / "bundled"
LOG_DIR = DATA_DIR / "logs"

for d in [CACHE_DIR, CSV_DIR, SYMBOLS_CSV_DIR, BUNDLED_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Fetching ──────────────────────────────────────────────────────────────
REQUEST_DELAY = 0.5          # seconds between requests (polite)
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT = 30         # seconds
HISTORY_START_DATE = "2010-01-01"

# ── PSX Endpoints ──────────────────────────────────────────────────────────────
PSX_LISTINGS_URL = "https://dps.psx.com.pk/listings"
PSX_COMPANY_URL = "https://dps.psx.com.pk/company/{symbol}"
PSX_HISTORICAL_URL = "https://dps.psx.com.pk/historical"
PSX_TIMESERIES_URL = "https://dps.psx.com.pk/timeseries/int/{date_from}/{date_to}"
PSX_SECTOR_URL = "https://dps.psx.com.pk/sector-summary"

# ── Yahoo Finance Tickers ─────────────────────────────────────────────────────
KSE100_TICKER = "^KSE"
USDPKR_TICKER = "USDPKR=X"

# ── FRED Series IDs ────────────────────────────────────────────────────────────
FRED_CPI_SERIES = "PAKPCPIPCHPT"       # Pakistan CPI % change
FRED_POLICY_RATE = "INTDSRPKM193N"     # Pakistan discount rate (proxy for T-bill)

# ── Risk-Free Rate (fallback) ─────────────────────────────────────────────────
DEFAULT_RISK_FREE_RATE = 0.10  # 10% annualized (approx avg Pakistan T-bill)

# ── Trading Days ───────────────────────────────────────────────────────────────
TRADING_DAYS_PER_YEAR = 245    # PSX trading days (approx)

# ── Logging ────────────────────────────────────────────────────────────────────
ERROR_LOG_FILE = LOG_DIR / "errors.log"
PIPELINE_LOG_FILE = LOG_DIR / "pipeline.log"

# ── User Agent ─────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
