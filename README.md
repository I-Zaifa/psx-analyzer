# PSX Analyzer
A complete analysis dashboard for the *Pakistan Stock Exchange*. It pulls real market data for 600+ listed companies, beep-boops the numbers, and gives you an interactive web dashboard you can browse locally or deploy to where ever the fuck.

---

## includes:
- **Every listed company on PSX** — not a sample, the whole exchange
- **Interactive charts** — price history with moving averages, drawdown, cumulative returns, volume analysis
- **Stock screener** — search, filter, sort by any metric across all 600+ companies
- **Sector analysis** — treemap visualization, performance rankings, best/worst performers per sector
- **Macro overlays** — KSE-100 index, USD/PKR exchange rate, CPI inflation, SBP policy rate
- **Side-by-side comparison** — pick up to 5 stocks, see them normalized on one chart
- **CSV downloads** — grab the raw data for your own analysis in Excel, Python, R, whatever
- **Full source code downloads** — every Python file available directly from the app
- **Methodology & disclosure** — data sources, metric definitions, and limitations documented on-site
- **Works offline** — it's a PWA, install it on your phone or desktop

## Data - Sources
| What | Source |
|------|--------|
| Company listings & sectors | [PSX Data Portal](https://dps.psx.com.pk) |
| Daily OHLCV prices | PSX Data Portal via `psx-data-reader` |
| KSE-100 Index | Yahoo Finance (`^KSE`) + PSX timeseries |
| USD/PKR Exchange Rate | Yahoo Finance (`USDPKR=X`) |
| CPI / Inflation | World Bank API → FRED fallback → bundled data |
| SBP Policy Rate | FRED → bundled data |
| Fundamentals (P/E, EPS, etc.) | PSX Data Portal company pages |

All macro data has multiple fallback sources so nothing breaks if one API is down.

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt
# 2. Run the full pipeline (fetches data + starts dashboard)
python run.py
# That's it. Open http://localhost:5000 in your browser.
```
If you just want to explore data that's already been fetched:
```bash
python run.py --server          # Skip pipeline, start dashboard with existing data
```

Other useful options:
```bash
python run.py --pipeline              # Fetch data only, no web server
python run.py --pipeline --max 10     # Quick test with just 10 stocks
python run.py --pipeline --skip-fund  # Skip scraping fundamentals (faster)
python run.py --port 8080             # Use a different port
python run.py --icons                 # Generate PWA icon files
```

The full pipeline takes about 15–30 minutes to pull data for all ~600 companies.
--
The `freeze.py` script pre-renders every page (dashboard, all 600+ stock pages, screener, sectors, downloads) into plain HTML with embedded Plotly charts. The Compare page works fully on the static site too: it fetches pre-built JSON data client-side and renders charts in the browser. No server, no database, just static files that load fast.

## Built With
- **Python** + Flask for the backend
- **Plotly.js** for interactive charts
- **pandas** + numpy + scipy for number crunching
- **psx-data-reader** + yfinance for market data
- **World Bank API** + FRED for macro data
- No heavy UI frameworks

## A Few Things to Know
- If a stock fails to fetch, the pipeline logs the error and continues with the rest.
- Macro data has triple fallback: live API → secondary API → bundled CSV.
- There's a 24-hour cache so re-running the pipeline doesn't hammer the APIs.
- All dates shown in the UI reflect the actual date range of the underlying data.


## License
MIT
