# PSX Analyzer

A complete analysis dashboard for the **Pakistan Stock Exchange**. It pulls real market data for 600+ listed companies, crunches the numbers, and gives you an interactive web dashboard you can browse locally or deploy to Netlify in one drag.

The data covers **January 2010 through February 2026** — daily closing prices, volumes, macro indicators, sector breakdowns, and more than a dozen financial metrics per stock.

---

## What You Get

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

## Where the Data Comes From

| What | Source |
|------|--------|
| Company listings & sectors | [PSX Data Portal](https://dps.psx.com.pk) |
| Daily OHLCV prices | PSX Data Portal via `psx-data-reader` |
| KSE-100 Index | Yahoo Finance (`^KSE`) + PSX timeseries |
| USD/PKR Exchange Rate | Yahoo Finance (`USDPKR=X`) |
| CPI / Inflation | World Bank API → FRED fallback → bundled data |
| SBP Policy Rate | FRED (`INTDSRPKM193N`) → bundled data |
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

## Deploy to Netlify

You can turn the whole thing into a static site and throw it on Netlify — no server needed.

### Option A: Git-based deploy (recommended)

Push your repo to GitHub/GitLab, connect it to Netlify, and it builds automatically.

1. Make sure the data pipeline has been run at least once (`python run.py --pipeline`)
2. Commit everything including `data/csv/` (the generated CSV data files)
3. Push to your repo
4. Connect to Netlify — it will use `netlify.toml` and run `freeze.py` on every push

Netlify installs only the minimal build dependencies (`requirements-build.txt`), not the full data-fetching stack.

### Option B: Drag-and-drop deploy

```bash
# 1. Make sure the data pipeline has been run at least once
# 2. Generate the static site
python freeze.py

# 3. Go to https://app.netlify.com/drop
#    Drag the 'build/' folder onto the page
#    Done. You've got a live URL.
```

### What gets generated

The `freeze.py` script pre-renders every page (dashboard, all 600+ stock pages, screener, sectors, downloads) into plain HTML with embedded Plotly charts. The Compare page works fully on the static site too — it fetches pre-built JSON data client-side and renders charts in the browser. No server, no database, just static files that load fast.

## Dashboard Pages

| Page | URL | What It Shows |
|------|-----|---------------|
| Dashboard | `/` | Market overview, KSE-100 chart, CPI, USD/PKR, top movers, return distribution |
| Screener | `/screener` | Sortable table of all companies — filter by sector, search by name |
| Compare | `/compare` | Normalized price chart + metrics table for up to 5 stocks |
| Sectors | `/sectors` | Treemap by company count, colored by return, plus sector summary table |
| Downloads & About | `/downloads` | Methodology, data sources, metric definitions, CSV downloads, Python source code |
| Stock Detail | `/stock/HBL` | Full analysis for any individual stock (replace HBL with any symbol) |

## Metrics Explained

Every stock gets these computed automatically:

| Metric | What It Means |
|--------|---------------|
| **Total Return** | How much the stock moved from its first to last close price |
| **CAGR** | Compound Annual Growth Rate — smoothed yearly return |
| **Volatility** | How much the price jumps around (annualized from daily log returns) |
| **Max Drawdown** | The worst peak-to-trough drop — how bad could it get? |
| **Sharpe Ratio** | Return per unit of risk (higher = better risk-adjusted performance) |
| **Beta** | How much the stock moves with the KSE-100 index |
| **Correlation** | How closely the stock tracks the index |
| **Real CAGR** | CAGR minus average inflation — your actual purchasing power gain |
| **Period Returns** | YTD, 1-month, 3-month, 6-month, 1-year returns |
| **52-Week High/Low** | Highest and lowest price in the last trading year |
| **Volume Trend** | Is trading volume increasing or decreasing recently? |

## Project Structure

```
psx-analyzer/
  app/                  # Flask web application
    app.py              # Routes, charts, data loading
    templates/          # Jinja2 HTML templates
    static/             # CSS, JS, icons
  backend/              # Data pipeline
    config.py           # Paths, API endpoints, settings
    fetcher.py          # Stock data fetching
    macro.py            # Macro data (KSE-100, USD/PKR, CPI, T-bill)
    metrics.py          # Financial calculations
    sectors.py          # Sector aggregation
    export.py           # CSV generation
    run_pipeline.py     # Pipeline orchestrator
  data/
    bundled/            # Fallback data (ships with repo)
    cache/              # Cached API responses (24h TTL)
    csv/                # Generated CSVs (pipeline output)
  freeze.py             # Static site generator for Netlify
  netlify.toml          # Netlify deployment config
  run.py                # Main entry point
  requirements.txt      # Python dependencies
```

## Built With

- **Python** + Flask for the backend
- **Plotly.js** for interactive charts
- **pandas** + numpy + scipy for number crunching
- **psx-data-reader** + yfinance for market data
- **World Bank API** + FRED for macro data
- Custom CSS with Inter font — no heavy UI frameworks

## A Few Things to Know

- If a stock fails to fetch, the pipeline logs the error and continues with the rest.
- Macro data has triple fallback: live API → secondary API → bundled CSV.
- There's a 24-hour cache so re-running the pipeline doesn't hammer the APIs.
- All dates shown in the UI reflect the actual date range of the underlying data.

## Disclaimer

This is a research and educational tool. It is **not financial advice**. The data may be delayed, incomplete, or contain errors. Always verify independently and consult a qualified financial advisor before making investment decisions.

## License

MIT
