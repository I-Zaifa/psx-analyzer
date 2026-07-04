# PSX Analyzer
A complete analysis dashboard for the *Pakistan Stock Exchange* based on a static historical dataset. It analyzes 600+ listed companies and provides an interactive web dashboard for research and archival study.

## Made using Antigravity and updated/fixes done by copilot.

---

## Includes
- **Every listed company on PSX** — not a sample, the whole exchange
- **Interactive charts** — price history with moving averages, drawdown, cumulative returns, volume analysis
- **Stock screener** — search, filter, sort by key metrics across all 600+ companies
- **Sector analysis** — treemap visualization, performance rankings, best/worst performers per sector
- **Macro overlays** — KSE-100 index, USD/PKR exchange rate, CPI inflation, SBP policy rate
- **Side-by-side comparison** — pick up to 5 stocks, see them normalized on one chart
- **CSV downloads** — raw data for your own analysis in Excel, Python, R, etc.
- **Full source code downloads** — Python files available directly from the app
- **Methodology & disclosure** — sources, metric definitions, and limitations documented on-site
- **Works offline** — it's a PWA, installable on phone or desktop

## Data Update Policy (Important)
**Support ended in June 2026.** Automated live/delayed PSX data updates have been discontinued.

PSX does not permit delayed data usage in this context (including non-commercial and educational use). We believe this is a policy decision that should apply to live licensed feeds only, but we are complying with the current policy.

The site remains online with a static historical dataset for archival and analytical use. Scheduled GitHub Actions refresh jobs have been disabled.

## Data Sources
| What | Source |
|------|--------|
| Company listings & sectors | [PSX Data Portal](https://dps.psx.com.pk) |
| Daily OHLCV prices | PSX Data Portal (`/timeseries/eod/{symbol}` primary, `/historical` fallback) |
| KSE-100 Index | Yahoo Finance (`^KSE`) + PSX timeseries |
| USD/PKR Exchange Rate | Yahoo Finance (`USDPKR=X`) |
| CPI / Inflation | World Bank API → FRED fallback → bundled data |
| SBP Policy Rate | FRED → bundled data |
| Fundamentals (P/E, EPS, etc.) | PSX Data Portal company pages |

## Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt
# 2. Run the app with existing local data
python run.py --server
# Open http://localhost:5000 in your browser
```

If you need to run pipeline utilities manually for maintenance/testing:
```bash
python run.py --pipeline              # Data pipeline only
python run.py --pipeline --mode repair # Historical backfill/repair mode
python run.py --pipeline --max 10     # Quick test with 10 stocks
python run.py --pipeline --skip-fund  # Skip fundamentals scraping
python run.py --port 8080             # Alternate port
python run.py --icons                 # Generate PWA icons
```

The `freeze.py` script pre-renders every page (dashboard, stock pages, screener, sectors, downloads) into plain HTML with embedded Plotly charts. The Compare page works on the static site by fetching pre-built JSON and rendering in-browser.

## Deploy to Vercel
1. Import the repo in Vercel (GitHub integration).
2. Vercel reads `vercel.json` and:
   - installs with `python -m venv .venv && .venv/bin/pip install -r requirements-build.txt`
   - builds with `.venv/bin/python freeze.py`
   - publishes the `build/` directory
3. Enable auto-deploy on `main` for normal website deployments.
4. If prompted for a Python version, use **3.11**.

## Automation Status
The workflow file `.github/workflows/update-psx-data.yml` is now **manual-only**. Scheduled triggers are disabled.

## Built With
- **Python** + Flask
- **Plotly.js** for interactive charts
- **pandas** + numpy + scipy for analytics
- **PSX Data Portal endpoints** + yfinance for market data ingestion
- **World Bank API** + FRED for macro data

## A Few Things to Know
- This project is now maintained as a static historical analysis tool.
- Macro series include bundled fallback CSVs where applicable.
- All dates shown in the UI reflect the underlying dataset date range.

## License
MIT
