"""
Static Site Generator for PSX Analyzer
=======================================
Generates a Netlify-ready static site in the build/ directory.

Usage:
    python freeze.py

Then deploy:
    Drag the 'build/' folder to https://app.netlify.com/drop
"""
import os
import sys
import shutil
import json
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from app.app import app, load_master, load_symbol_history
from backend.config import CSV_DIR, SYMBOLS_CSV_DIR

BUILD = Path("build")


def save_route(client, route, out_path):
    """Request a Flask route and save the response to a file."""
    resp = client.get(route)
    p = BUILD / out_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(resp.data)
    return resp.status_code


def freeze():
    """Generate static site from Flask app."""
    start = time.time()
    print("=" * 50)
    print("  PSX Analyzer — Static Site Generator")
    print("=" * 50)
    print()

    # Clean and create build directory
    if BUILD.exists():
        shutil.rmtree(BUILD)
    BUILD.mkdir(parents=True)

    client = app.test_client()

    # ── 1. Static pages ──────────────────────────────────────────────────
    pages = {
        "/": "index.html",
        "/screener": "screener/index.html",
        "/sectors": "sectors/index.html",
        "/downloads": "downloads/index.html",
        "/compare": "compare/index.html",
    }

    print("[1/6] Generating main pages...")
    for route, out in pages.items():
        status = save_route(client, route, out)
        print(f"  {route:20s} -> {out:30s} [{status}]")

    # 404 page (Netlify serves this for missing routes)
    p404 = Path("app/templates/404.html")
    if p404.exists():
        shutil.copy2(p404, BUILD / "404.html")
        print("  404.html copied to build root")

    # ── 2. Stock detail pages ────────────────────────────────────────────
    print("\n[2/6] Generating stock pages...")
    master = load_master()
    symbols = sorted(master["symbol"].dropna().unique()) if not master.empty else []
    total = len(symbols)
    errors = 0

    for i, sym in enumerate(symbols):
        try:
            save_route(client, f"/stock/{sym}", f"stock/{sym}/index.html")
        except Exception as e:
            errors += 1
            print(f"  WARNING: /stock/{sym} failed: {e}")
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  ...{i + 1}/{total} stock pages")

    if errors:
        print(f"  ({errors} errors)")

    # ── 3. Copy static assets ────────────────────────────────────────────
    print("\n[3/6] Copying static assets...")
    src_static = Path("app/static")
    if src_static.exists():
        shutil.copytree(src_static, BUILD / "static")
        print(f"  Copied app/static/ -> build/static/")

    # Root-level files (manifest.json, sw.js, robots.txt)
    for fname in ["manifest.json", "sw.js", "robots.txt"]:
        src = src_static / fname
        if src.exists():
            shutil.copy2(src, BUILD / fname)
            print(f"  Copied {fname} to build root")

    # ── 4. Copy CSV download files ───────────────────────────────────────
    print("\n[4/6] Copying download files...")
    dl = BUILD / "download"
    dl.mkdir(parents=True, exist_ok=True)

    csv_count = 0
    if CSV_DIR.exists():
        for f in CSV_DIR.glob("*.csv"):
            shutil.copy2(f, dl / f.name)
            csv_count += 1
    print(f"  Copied {csv_count} CSV files to build/download/")

    # Symbol CSV files
    sym_dl = dl / "symbol"
    sym_dl.mkdir(parents=True, exist_ok=True)
    sym_count = 0
    if SYMBOLS_CSV_DIR.exists():
        for f in SYMBOLS_CSV_DIR.glob("*.csv"):
            # Copy with .csv extension (direct file link)
            shutil.copy2(f, sym_dl / f.name)
            # Also copy WITHOUT extension (matches Flask route /download/symbol/HBL)
            shutil.copy2(f, sym_dl / f.stem)
            sym_count += 1
    print(f"  Copied {sym_count} symbol CSVs to build/download/symbol/")

    # ── 4b. Copy Python source files ──────────────────────────────────────────
    print("\n[4b/7] Copying source code files...")
    src_dl = dl / "source"
    src_dl.mkdir(parents=True, exist_ok=True)
    py_files = [
        ("backend/config.py", "config.py"),
        ("backend/fetcher.py", "fetcher.py"),
        ("backend/macro.py", "macro.py"),
        ("backend/metrics.py", "metrics.py"),
        ("backend/sectors.py", "sectors.py"),
        ("backend/fundamentals.py", "fundamentals.py"),
        ("backend/export.py", "export.py"),
        ("backend/run_pipeline.py", "run_pipeline.py"),
        ("app/app.py", "app.py"),
        ("freeze.py", "freeze.py"),
        ("run.py", "run.py"),
    ]
    py_count = 0
    for rel_path, out_name in py_files:
        src = Path(rel_path)
        if src.exists():
            shutil.copy2(src, src_dl / out_name)
            py_count += 1
    print(f"  Copied {py_count} Python source files to build/download/source/")

    # ── 5. API endpoints as static JSON ──────────────────────────────────────
    print("\n[5/7] Generating API endpoints...")
    api = BUILD / "api"
    api.mkdir(parents=True, exist_ok=True)

    resp = client.get("/api/symbols")
    (api / "symbols.json").write_bytes(resp.data)
    print("  /api/symbols -> api/symbols.json")

    # ── 6. Per-stock JSON for client-side compare ────────────────────────
    print("\n[6/7] Generating stock JSON data for compare...")
    prices_dir = api / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)

    # master.json — all metrics for the comparison table
    if not master.empty:
        master_clean = master.where(pd.notnull(master), None)
        master_dict = {}
        for _, row in master_clean.iterrows():
            sym = row.get("symbol")
            if sym:
                master_dict[sym] = {k: v for k, v in row.to_dict().items()}
        (api / "master.json").write_text(json.dumps(master_dict), encoding="utf-8")
        print(f"  api/master.json ({len(master_dict)} stocks)")

    # Per-stock price JSON — compact {dates:[], close:[]} for chart
    price_count = 0
    for i, sym in enumerate(symbols):
        try:
            df = load_symbol_history(sym)
            if not df.empty and "close" in df.columns:
                prices = df["close"].dropna()
                if len(prices) >= 2:
                    data = {
                        "dates": [d.strftime("%Y-%m-%d") for d in prices.index],
                        "close": [round(float(v), 2) for v in prices.values]
                    }
                    (prices_dir / f"{sym}.json").write_text(
                        json.dumps(data), encoding="utf-8"
                    )
                    price_count += 1
        except Exception:
            pass
        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  ...{i + 1}/{total} price files")
    print(f"  Generated {price_count} price JSON files")

    # ── 7. Netlify config files ──────────────────────────────────────────
    print("\n[7/7] Writing Netlify config files...")

    # _redirects: map API and download routes
    redirects = "\n".join([
        "# API redirects",
        "/api/symbols    /api/symbols.json    200",
        "/api/master     /api/master.json     200",
        "",
        "# Compare page: always serve the same HTML, JS handles query params",
        "/compare    /compare/index.html    200!",
        "/compare/*  /compare/index.html    200",
    ])
    (BUILD / "_redirects").write_text(redirects + "\n")
    print("  Created _redirects")

    # _headers: set correct content type for CSV and Python downloads
    headers = "\n".join([
        "/download/*.csv",
        "  Content-Type: text/csv; charset=utf-8",
        "  Content-Disposition: attachment",
        "",
        "/download/symbol/*",
        "  Content-Type: text/csv; charset=utf-8",
        "  Content-Disposition: attachment",
        "",
        "/download/source/*.py",
        "  Content-Type: text/x-python; charset=utf-8",
        "  Content-Disposition: attachment",
    ])
    (BUILD / "_headers").write_text(headers + "\n")
    print("  Created _headers")

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - start
    total_pages = len(pages) + len(symbols)
    print()
    print("=" * 50)
    print(f"  Build complete!")
    print(f"  Pages:    {total_pages} ({len(pages)} main + {len(symbols)} stocks)")
    print(f"  CSVs:     {csv_count + sym_count} download files")
    print(f"  Time:     {elapsed:.1f}s")
    print(f"  Output:   {BUILD.resolve()}")
    print("=" * 50)
    print()
    print("Deploy to Netlify:")
    print("  1. Go to https://app.netlify.com/drop")
    print(f"  2. Drag the '{BUILD}/' folder onto the page")
    print("  3. Done!")


if __name__ == "__main__":
    freeze()
