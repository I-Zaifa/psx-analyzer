"""
Flask application for PSX Analyzer interactive dashboard.
Loads generated CSVs and serves interactive tables + Plotly charts.
"""
import os
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
from flask import Flask, render_template, request, jsonify, send_from_directory, abort

from backend.config import CSV_DIR, SYMBOLS_CSV_DIR, DATA_DIR

logger = logging.getLogger(__name__)

app = Flask(__name__,
            static_folder="static",
            template_folder="templates")
app.config["SECRET_KEY"] = os.urandom(24).hex()


# ── Data Loading Helpers ───────────────────────────────────────────────────────

def load_master() -> pd.DataFrame:
    path = CSV_DIR / "master.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_sectors() -> pd.DataFrame:
    path = CSV_DIR / "sectors.csv"
    if path.exists():
        df = pd.read_csv(path)
        if "sector" in df.columns and not df.empty:
            df = df.dropna(subset=["sector"])
            df["sector"] = df["sector"].astype(str).str.strip()
            df = df[df["sector"].str.len() > 0]
            df = df[~df["sector"].str.lower().isin(["nan", "none"])]
        return df
    return pd.DataFrame()


def load_symbol_history(symbol: str) -> pd.DataFrame:
    path = SYMBOLS_CSV_DIR / f"{symbol}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    return pd.DataFrame()


def load_macro(name: str) -> pd.DataFrame:
    path = CSV_DIR / f"macro_{name}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"], index_col="date")
    return pd.DataFrame()


def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.2f}%"


def fmt_num(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    if abs(val) >= 1e9:
        return f"{val/1e9:.{decimals}f}B"
    if abs(val) >= 1e6:
        return f"{val/1e6:.{decimals}f}M"
    if abs(val) >= 1e3:
        return f"{val/1e3:.{decimals}f}K"
    return f"{val:.{decimals}f}"


app.jinja_env.globals.update(fmt_pct=fmt_pct, fmt_num=fmt_num)


@app.context_processor
def inject_data_dates():
    """Inject data date range into all templates for clarity."""
    dates = {"data_as_of": "", "data_from": "", "data_range": ""}
    try:
        kse = load_macro("kse100")
        if not kse.empty and "close" in kse.columns:
            last_date = kse.index.max()
            first_date = kse.index.min()
            dates["data_as_of"] = last_date.strftime("%b %d, %Y")
            dates["data_from"] = first_date.strftime("%b %d, %Y")
            dates["data_range"] = f"{first_date.strftime('%b %Y')} \u2013 {last_date.strftime('%b %Y')}"
    except Exception:
        pass
    return dates


# ── Chart Helpers ──────────────────────────────────────────────────────────────

GOLD = "#CFB53B"
GOLD_DARK = "#b89f2e"
SILVER = "#c0c0c0"
RED = "#c62828"
GREEN = "#2e7d32"
SILVER_DARK = "#a0a0a0"
SLATE = "#999999"

CHART_COLORS = [GOLD, "#5B8DEF", RED, GREEN, SILVER_DARK, "#E67E22", "#8E44AD", "#1ABC9C"]


def _base_layout(**overrides):
    """Base Plotly layout with consistent styling."""
    layout = dict(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="#fafafa",
        font=dict(color="#3a3a3a", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=60, r=30, t=55, b=50),
        xaxis=dict(gridcolor="#eeeeee", zerolinecolor="#c0c0c0", showgrid=True,
                   linecolor="#c0c0c0", linewidth=1, mirror=False),
        yaxis=dict(gridcolor="#eeeeee", zerolinecolor="#c0c0c0", showgrid=True,
                   separatethousands=True, linecolor="#c0c0c0", linewidth=1, mirror=False),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#c0c0c0", borderwidth=1,
                    font=dict(size=11)),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#1a1a1a", bordercolor="#c0c0c0",
                        font_size=12),
        hovermode="x unified",
    )
    layout.update(overrides)
    return layout


def _fig_to_json(fig) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _add_range_buttons(fig):
    """Add date range selector buttons to a time-series chart."""
    fig.update_xaxes(rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(count=3, label="3Y", step="year", stepmode="backward"),
            dict(step="all", label="All"),
        ],
        bgcolor="#f0f0f0", activecolor=GOLD, bordercolor="#c0c0c0",
        font=dict(size=10),
    ))


# ── Stock-Level Charts ────────────────────────────────────────────────────────

def make_price_chart(df: pd.DataFrame, symbol: str) -> str:
    """Price chart with 50-day and 200-day moving averages."""
    if df.empty or "close" not in df.columns:
        return ""
    prices = df["close"].dropna()
    if len(prices) < 5:
        return ""
    last_price = prices.iloc[-1]
    first_price = prices.iloc[0]
    total_ret = (last_price - first_price) / first_price * 100

    fig = go.Figure()

    # Price line
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="Close Price",
        line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(207,181,59,0.06)",
        hovertemplate="PKR %{y:,.2f}<extra></extra>"
    ))

    # 50-day MA
    if len(prices) >= 50:
        ma50 = prices.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=ma50.index, y=ma50.values,
            mode="lines", name="50-Day MA",
            line=dict(color="#5B8DEF", width=1.2, dash="dot"),
            hovertemplate="MA50: %{y:,.2f}<extra></extra>"
        ))

    # 200-day MA
    if len(prices) >= 200:
        ma200 = prices.rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=ma200.index, y=ma200.values,
            mode="lines", name="200-Day MA",
            line=dict(color=RED, width=1.2, dash="dash"),
            hovertemplate="MA200: %{y:,.2f}<extra></extra>"
        ))

    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> — PKR {last_price:,.2f}  "
                        f"<span style='color:{GREEN if total_ret>=0 else RED}'>"
                        f"({total_ret:+.1f}%)</span>",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Price (PKR)",
        yaxis_tickformat=",.0f",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


def make_volume_chart(df: pd.DataFrame, symbol: str) -> str:
    """Volume bar chart with 30-day moving average overlay."""
    if df.empty or "volume" not in df.columns:
        return ""
    vol = df["volume"].dropna()
    if vol.empty or vol.sum() == 0:
        return ""

    avg_vol = vol.iloc[-90:].mean() if len(vol) >= 90 else vol.mean()

    # Color bars by above/below 30d avg
    ma30 = vol.rolling(30, min_periods=1).mean()
    colors = [GREEN if v > m else SILVER for v, m in zip(vol.values, ma30.values)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vol.index, y=vol.values,
        name="Volume", marker_color=colors, opacity=0.75,
        hovertemplate="%{y:,.0f} shares<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=ma30.index, y=ma30.values,
        mode="lines", name="30-Day Avg",
        line=dict(color=GOLD, width=1.5),
        hovertemplate="30d avg: %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> Trading Volume — 90d Avg: {avg_vol:,.0f}",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Shares Traded",
        yaxis_tickformat=",.0s", barmode="overlay",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


def make_drawdown_chart(df: pd.DataFrame, symbol: str) -> str:
    """Drawdown chart showing peak-to-trough declines."""
    if df.empty or "close" not in df.columns:
        return ""
    prices = df["close"].dropna()
    if len(prices) < 5:
        return ""
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        mode="lines", name="Drawdown",
        line=dict(color=RED, width=1.5),
        fill="tozeroy", fillcolor="rgba(198,40,40,0.08)",
        hovertemplate="%{y:.1%}<extra></extra>"
    ))

    # Annotate maximum drawdown
    fig.add_annotation(
        x=max_dd_date, y=max_dd,
        text=f"Max: {max_dd:.1%}", showarrow=True,
        arrowhead=2, arrowsize=1, arrowwidth=1.5,
        arrowcolor=RED, font=dict(size=11, color=RED),
        ax=40, ay=-30
    )

    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> Drawdown — Worst: {max_dd:.1%} "
                        f"({max_dd_date.strftime('%b %Y')})",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Drawdown from Peak",
        yaxis_tickformat=".0%",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


def make_returns_chart(df: pd.DataFrame, symbol: str) -> str:
    """Cumulative return chart vs KSE-100 index."""
    if df.empty or "close" not in df.columns:
        return ""
    prices = df["close"].dropna()
    if len(prices) < 5:
        return ""
    cum_ret = (prices / prices.iloc[0] - 1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_ret.index, y=cum_ret.values,
        mode="lines", name=symbol,
        line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(207,181,59,0.05)",
        hovertemplate="%{y:.1%}<extra></extra>"
    ))

    # Overlay KSE-100 returns for context
    kse_df = load_macro("kse100")
    if not kse_df.empty and "close" in kse_df.columns:
        kse_prices = kse_df["close"].dropna()
        # Align to stock's first date
        kse_from = kse_prices[kse_prices.index >= prices.index[0]]
        if len(kse_from) > 5:
            kse_ret = kse_from / kse_from.iloc[0] - 1
            fig.add_trace(go.Scatter(
                x=kse_ret.index, y=kse_ret.values,
                mode="lines", name="KSE-100",
                line=dict(color=SILVER_DARK, width=1.5, dash="dot"),
                hovertemplate="KSE-100: %{y:.1%}<extra></extra>"
            ))

    final_ret = cum_ret.iloc[-1]
    fig.update_layout(
        title=dict(text=f"<b>{symbol}</b> Cumulative Return: "
                        f"<span style='color:{GREEN if final_ret>=0 else RED}'>"
                        f"{final_ret:.1%}</span> vs KSE-100",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Cumulative Return",
        yaxis_tickformat=".0%",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


# ── Dashboard / Macro Charts ─────────────────────────────────────────────────

def make_kse100_chart() -> str:
    """KSE-100 index chart with YTD return annotation. Source: PSX + yfinance."""
    df = load_macro("kse100")
    if df.empty or "close" not in df.columns:
        return ""
    prices = df["close"].dropna()
    if len(prices) < 5:
        return ""
    last_val = prices.iloc[-1]
    last_date = prices.index[-1]

    # Compute YTD return
    ytd_prices = prices[prices.index.year == last_date.year]
    ytd_ret = None
    if len(ytd_prices) >= 2:
        ytd_ret = (ytd_prices.iloc[-1] / ytd_prices.iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="KSE-100",
        line=dict(color=GOLD, width=2),
        fill="tozeroy", fillcolor="rgba(207,181,59,0.06)",
        hovertemplate="%{y:,.0f} pts<extra></extra>"
    ))

    # 200-day MA
    if len(prices) >= 200:
        ma200 = prices.rolling(200).mean()
        fig.add_trace(go.Scatter(
            x=ma200.index, y=ma200.values,
            mode="lines", name="200-Day MA",
            line=dict(color=SILVER_DARK, width=1, dash="dot"),
            hoverinfo="skip"
        ))

    title_parts = [f"<b>KSE-100 Index</b> — {last_val:,.0f}"]
    if ytd_ret is not None:
        color = GREEN if ytd_ret >= 0 else RED
        title_parts.append(f"<span style='color:{color}'>YTD {ytd_ret:+.1f}%</span>")
    title_parts.append(f"<span style='font-size:11px;color:#999'>({last_date.strftime('%d %b %Y')})</span>")

    fig.update_layout(
        title=dict(text="  ".join(title_parts), font=dict(size=14)),
        xaxis_title="", yaxis_title="Index Points",
        yaxis_tickformat=",.0f",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


def make_usdpkr_chart() -> str:
    """USD/PKR exchange rate chart. Source: Yahoo Finance (PKR=X)."""
    df = load_macro("usdpkr")
    if df.empty or "close" not in df.columns:
        return ""
    prices = df["close"].dropna()
    if len(prices) < 5:
        return ""
    last_val = prices.iloc[-1]
    last_date = prices.index[-1]

    # 1-year change
    yr_ago_idx = prices.index[prices.index <= last_date - pd.Timedelta(days=365)]
    yr_change = None
    if len(yr_ago_idx) > 0:
        yr_change = (last_val / prices.loc[yr_ago_idx[-1]] - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prices.index, y=prices.values,
        mode="lines", name="USD/PKR",
        line=dict(color="#5B8DEF", width=2),
        fill="tozeroy", fillcolor="rgba(91,141,239,0.06)",
        hovertemplate="PKR %{y:.2f}<extra></extra>"
    ))

    title_parts = [f"<b>USD/PKR</b> — {last_val:.2f}"]
    if yr_change is not None:
        color = RED if yr_change >= 0 else GREEN
        title_parts.append(f"<span style='color:{color}'>1Y {yr_change:+.1f}%</span>")
    title_parts.append(f"<span style='font-size:11px;color:#999'>({last_date.strftime('%d %b %Y')})</span>")

    fig.update_layout(
        title=dict(text="  ".join(title_parts), font=dict(size=14)),
        xaxis_title="", yaxis_title="PKR per 1 USD",
        yaxis_tickformat=",.0f",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


def make_cpi_chart() -> str:
    """CPI inflation chart. Source: World Bank API (FP.CPI.TOTL.ZG)."""
    df = load_macro("cpi")
    if df.empty or "cpi_yoy" not in df.columns:
        return ""
    cpi = df["cpi_yoy"].dropna()
    if cpi.empty:
        return ""

    colors = [RED if v > 10 else (GOLD if v > 5 else GREEN) for v in cpi.values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cpi.index, y=cpi.values,
        name="CPI YoY %", marker_color=colors,
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))
    # SBP target reference line
    fig.add_hline(y=5, line_dash="dash", line_color=SILVER_DARK, line_width=1,
                  annotation_text="SBP ~5% Target", annotation_position="bottom right",
                  annotation_font_size=10, annotation_font_color=SILVER_DARK)

    last_val = cpi.iloc[-1]
    last_year = cpi.index[-1].year
    fig.update_layout(
        title=dict(text=f"<b>Pakistan CPI Inflation</b> — {last_val:.1f}% ({last_year})"
                        f"  <span style='font-size:11px;color:#999'>Source: World Bank</span>",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="YoY Inflation (%)",
        **_base_layout()
    )
    return _fig_to_json(fig)


def make_tbill_chart() -> str:
    """T-bill / SBP policy rate chart. Source: SBP monetary policy decisions."""
    df = load_macro("tbill")
    if df.empty or "rate" not in df.columns:
        return ""
    rates = df["rate"].dropna()
    if rates.empty:
        return ""

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rates.index, y=rates.values * 100,
        mode="lines+markers", name="Policy Rate",
        line=dict(color=GOLD, width=2.5),
        marker=dict(size=6, color=GOLD_DARK),
        hovertemplate="%{y:.1f}%<extra></extra>"
    ))

    last_val = rates.iloc[-1] * 100
    last_year = rates.index[-1].year
    fig.update_layout(
        title=dict(text=f"<b>SBP Policy Rate</b> — {last_val:.1f}% ({last_year})"
                        f"  <span style='font-size:11px;color:#999'>Source: SBP</span>",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Rate (%)",
        yaxis_ticksuffix="%",
        **_base_layout()
    )
    return _fig_to_json(fig)


def make_sector_performance_chart(sectors_df: pd.DataFrame) -> str:
    """Horizontal bar chart of sector avg returns, sorted."""
    if sectors_df.empty or "avg_total_return" not in sectors_df.columns:
        return ""
    df = sectors_df.dropna(subset=["avg_total_return"]).copy()
    df = df[df["sector"].astype(str).str.strip().str.len() > 0]
    if df.empty:
        return ""
    df["return_pct"] = df["avg_total_return"] * 100
    df = df.sort_values("return_pct", ascending=True)

    colors = [GREEN if v >= 0 else RED for v in df["return_pct"].values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["sector"], x=df["return_pct"],
        orientation="h", marker_color=colors,
        text=[f"{v:+.0f}%" for v in df["return_pct"]],
        textposition="outside", textfont=dict(size=10),
        hovertemplate="<b>%{y}</b><br>Avg Return: %{x:.1f}%<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text="<b>Sector Performance</b> — Average Total Return",
                   font=dict(size=14)),
        xaxis_title="Avg Total Return (%)", yaxis_title="",
        height=max(400, len(df) * 22 + 100),
        **_base_layout(margin=dict(l=200, r=60, t=55, b=50))
    )
    return _fig_to_json(fig)


def make_return_distribution_chart(master_df: pd.DataFrame) -> str:
    """Histogram of stock total returns across all companies."""
    if master_df.empty or "total_return" not in master_df.columns:
        return ""
    rets = master_df["total_return"].dropna()
    if rets.empty:
        return ""
    # Clip extreme outliers for readability
    rets_clipped = rets.clip(-1, 10) * 100

    positive = (rets > 0).sum()
    negative = (rets < 0).sum()
    zero = (rets == 0).sum()
    median_ret = rets.median() * 100

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=rets_clipped, nbinsx=60, name="Stocks",
        marker_color=GOLD, opacity=0.85,
        hovertemplate="Return: %{x:.0f}%<br>Count: %{y}<extra></extra>"
    ))
    fig.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1.5)
    fig.add_vline(x=median_ret, line_dash="dash", line_color="#5B8DEF", line_width=1.5,
                  annotation_text=f"Median: {median_ret:.0f}%",
                  annotation_font_size=11, annotation_font_color="#5B8DEF")

    fig.update_layout(
        title=dict(text=f"<b>Return Distribution</b> — "
                        f"<span style='color:{GREEN}'>{positive} Gainers</span> · "
                        f"<span style='color:{RED}'>{negative} Losers</span> · "
                        f"{zero} Flat",
                   font=dict(size=14)),
        xaxis_title="Total Return (%)", yaxis_title="Number of Stocks",
        **_base_layout()
    )
    return _fig_to_json(fig)


def make_sector_heatmap(sectors_df: pd.DataFrame) -> str:
    """Sector heatmap showing percentile ranks across key metrics."""
    if sectors_df.empty:
        return ""
    df = sectors_df.dropna(subset=["sector"]).copy()
    if df.empty:
        return ""
    df["sector"] = df["sector"].astype(str).str.strip()
    df = df[df["sector"].str.len() > 0]
    df = df[~df["sector"].str.lower().isin(["nan", "none"])]
    if df.empty:
        return ""

    metric_defs = [
        {"key": "avg_total_return", "label": "Avg Return", "short": "Return", "fmt": "pct", "higher_better": True},
        {"key": "avg_cagr", "label": "Avg CAGR", "short": "CAGR", "fmt": "pct", "higher_better": True},
        {"key": "avg_volatility", "label": "Volatility", "short": "Vol", "fmt": "pct", "higher_better": False},
        {"key": "avg_sharpe", "label": "Sharpe", "short": "Sharpe", "fmt": "num", "higher_better": True},
        {"key": "avg_beta", "label": "Beta", "short": "Beta", "fmt": "num", "higher_better": False},
        {"key": "avg_max_drawdown", "label": "Max Drawdown", "short": "Max DD", "fmt": "pct", "higher_better": False},
    ]
    metric_defs = [m for m in metric_defs if m["key"] in df.columns]
    if not metric_defs:
        return ""

    if "avg_total_return" in df.columns:
        df = df.sort_values("avg_total_return", ascending=False)
    else:
        df = df.sort_values("sector")

    def fmt_value(value, fmt):
        if pd.isna(value):
            return "N/A"
        if fmt == "pct":
            return f"{value * 100:.1f}%"
        return f"{value:.2f}"

    percentiles = {}
    for meta in metric_defs:
        series = df[meta["key"]].astype(float)
        pct = series.rank(pct=True)
        if not meta["higher_better"]:
            pct = 1 - pct
        percentiles[meta["key"]] = pct.fillna(0.5)

    sectors = df["sector"].tolist()
    metric_labels = [m["short"] for m in metric_defs]
    short_sectors = [s if len(s) <= 26 else f"{s[:23]}…" for s in sectors]
    z_matrix = []
    hover_matrix = []
    for idx, sector in enumerate(sectors):
        row_z = []
        row_hover = []
        for meta in metric_defs:
            raw = df.iloc[idx][meta["key"]]
            pct = float(percentiles[meta["key"]].iloc[idx])
            raw_display = fmt_value(raw, meta["fmt"])
            pct_display = f"{pct * 100:.0f}%" if raw_display != "N/A" else "N/A"
            row_z.append(pct)
            row_hover.append(
                f"<b>{sector}</b><br>{meta['label']}: {raw_display}<br>Percentile: {pct_display}"
            )
        z_matrix.append(row_z)
        hover_matrix.append(row_hover)

    fig = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=metric_labels,
        y=short_sectors,
        customdata=hover_matrix,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale=[[0, RED], [0.5, "#f5f5f5"], [1, GREEN]],
        zmin=0,
        zmax=1,
        colorbar=dict(title="Percentile", tickformat=".0%"),
    ))

    height = max(520, 24 * len(sectors) + 160)
    fig.update_layout(
        title=dict(text="<b>Sector Heatmap</b> — Percentile Rank by Metric (green = better)",
                   font=dict(size=14)),
        height=height,
        **_base_layout(margin=dict(l=180, r=40, t=70, b=60), hovermode="closest")
    )
    fig.update_xaxes(side="top", tickangle=-25, showgrid=False, zeroline=False,
                     tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", showgrid=False,
                     tickfont=dict(size=10), automargin=True)
    return _fig_to_json(fig)


def make_comparison_chart(symbols: list) -> str:
    """Normalized price comparison (rebased to 100)."""
    fig = go.Figure()
    for i, sym in enumerate(symbols[:5]):
        df = load_symbol_history(sym)
        if df.empty or "close" not in df.columns:
            continue
        prices = df["close"].dropna()
        if len(prices) < 5:
            continue
        normalized = prices / prices.iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=normalized.index, y=normalized.values,
            mode="lines", name=sym,
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            hovertemplate=f"{sym}: %{{y:.1f}}<extra></extra>"
        ))

    # Add baseline
    fig.add_hline(y=100, line_dash="dot", line_color=SILVER, line_width=1)

    fig.update_layout(
        title=dict(text="<b>Price Comparison</b> — Normalized to 100 at First Date",
                   font=dict(size=14)),
        xaxis_title="", yaxis_title="Normalized Price (Base = 100)",
        **_base_layout()
    )
    _add_range_buttons(fig)
    return _fig_to_json(fig)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    master = load_master()
    sectors = load_sectors()

    stats = {}
    if not master.empty:
        stats["total_companies"] = len(master)
        stats["total_sectors"] = sectors["sector"].nunique() if not sectors.empty and "sector" in sectors.columns else 0
        stats["avg_return"] = master["total_return"].mean() if "total_return" in master.columns else None
        stats["median_return"] = master["total_return"].median() if "total_return" in master.columns else None
        stats["avg_cagr"] = master["cagr"].mean() if "cagr" in master.columns else None
        stats["avg_volatility"] = master["annualized_volatility"].mean() if "annualized_volatility" in master.columns else None
        stats["avg_sharpe"] = master["sharpe_ratio"].mean() if "sharpe_ratio" in master.columns else None
        stats["positive_count"] = int((master["total_return"] > 0).sum()) if "total_return" in master.columns else 0
        stats["negative_count"] = int((master["total_return"] < 0).sum()) if "total_return" in master.columns else 0

        # Top gainers / losers (1-year returns are more meaningful than all-time)
        if "return_1y" in master.columns:
            valid_1y = master.dropna(subset=["return_1y"])
            stats["top_gainers_1y"] = valid_1y.nlargest(5, "return_1y")[["symbol", "return_1y", "sector"]].to_dict("records")
            stats["top_losers_1y"] = valid_1y.nsmallest(5, "return_1y")[["symbol", "return_1y", "sector"]].to_dict("records")

        if "total_return" in master.columns:
            valid = master.dropna(subset=["total_return"])
            stats["top_gainers"] = valid.nlargest(5, "total_return")[["symbol", "total_return", "sector"]].to_dict("records")
            stats["top_losers"] = valid.nsmallest(5, "total_return")[["symbol", "total_return", "sector"]].to_dict("records")

    # Charts
    kse_chart = make_kse100_chart()
    usdpkr_chart = make_usdpkr_chart()
    cpi_chart = make_cpi_chart()
    tbill_chart = make_tbill_chart()
    dist_chart = make_return_distribution_chart(master)
    sector_perf_chart = make_sector_performance_chart(sectors) if not sectors.empty else ""

    return render_template("index.html",
                           stats=stats,
                           kse_chart=kse_chart,
                           usdpkr_chart=usdpkr_chart,
                           cpi_chart=cpi_chart,
                           tbill_chart=tbill_chart,
                           dist_chart=dist_chart,
                           sector_perf_chart=sector_perf_chart)


@app.route("/screener")
def screener():
    master = load_master()
    sectors = []
    if not master.empty and "sector" in master.columns:
        sectors = sorted(master["sector"].dropna().unique().tolist())
    return render_template("screener.html", data=master, sectors=sectors)


@app.route("/api/screener")
def api_screener():
    master = load_master()
    if master.empty:
        return jsonify({"data": []})

    # Apply filters
    sector = request.args.get("sector")
    min_return = request.args.get("min_return", type=float)
    max_return = request.args.get("max_return", type=float)
    min_sharpe = request.args.get("min_sharpe", type=float)
    min_volume = request.args.get("min_volume", type=float)
    search = request.args.get("search", "").strip().upper()

    df = master.copy()
    if sector and sector != "all" and "sector" in df.columns:
        df = df[df["sector"] == sector]
    if min_return is not None and "total_return" in df.columns:
        df = df[df["total_return"] >= min_return]
    if max_return is not None and "total_return" in df.columns:
        df = df[df["total_return"] <= max_return]
    if min_sharpe is not None and "sharpe_ratio" in df.columns:
        df = df[df["sharpe_ratio"] >= min_sharpe]
    if min_volume is not None and "avg_daily_volume" in df.columns:
        df = df[df["avg_daily_volume"] >= min_volume]
    if search:
        mask = df["symbol"].str.contains(search, na=False)
        if "sector" in df.columns:
            mask = mask | df["sector"].str.upper().str.contains(search, na=False)
        if "company" in df.columns:
            mask = mask | df["company"].str.upper().str.contains(search, na=False)
        if "name" in df.columns:
            mask = mask | df["name"].astype(str).str.upper().str.contains(search, na=False)
        df = df[mask]

    # Replace NaN with None for JSON
    df = df.where(pd.notnull(df), None)
    return jsonify({"data": df.to_dict("records"), "total": len(df)})


@app.route("/stock/<symbol>")
def stock_detail(symbol):
    symbol = symbol.upper()
    master = load_master()
    df = load_symbol_history(symbol)

    stock_info = {}
    if not master.empty:
        row = master[master["symbol"] == symbol]
        if not row.empty:
            stock_info = row.iloc[0].to_dict()
            # Replace NaN with None
            stock_info = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                          for k, v in stock_info.items()}

    price_chart = make_price_chart(df, symbol)
    volume_chart = make_volume_chart(df, symbol)
    drawdown_chart = make_drawdown_chart(df, symbol)
    returns_chart = make_returns_chart(df, symbol)

    return render_template("stock.html",
                           symbol=symbol,
                           info=stock_info,
                           price_chart=price_chart,
                           volume_chart=volume_chart,
                           drawdown_chart=drawdown_chart,
                           returns_chart=returns_chart,
                           has_data=not df.empty)


@app.route("/compare")
def compare():
    master = load_master()
    symbols = []
    if not master.empty:
        symbols = sorted(master["symbol"].dropna().unique().tolist())
    selected = request.args.getlist("symbols")
    if not selected:
        selected = request.args.get("s", "").split(",")
        selected = [s.strip().upper() for s in selected if s.strip()]

    chart = ""
    compare_data = []
    if selected:
        chart = make_comparison_chart(selected)
        for sym in selected:
            row = master[master["symbol"] == sym]
            if not row.empty:
                d = row.iloc[0].to_dict()
                d = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in d.items()}
                compare_data.append(d)

    return render_template("compare.html",
                           symbols=symbols,
                           selected=selected,
                           chart=chart,
                           compare_data=compare_data)


@app.route("/sectors")
def sectors_page():
    sectors_df = load_sectors()
    master = load_master()
    heatmap = make_sector_heatmap(sectors_df)
    return render_template("sector.html",
                           sectors=sectors_df,
                           heatmap=heatmap,
                           master=master)


@app.route("/downloads")
def downloads():
    files = []
    # Master CSV
    master_path = CSV_DIR / "master.csv"
    if master_path.exists():
        size = master_path.stat().st_size
        files.append({"name": "master.csv", "path": "master.csv", "size": size, "type": "master"})

    # Sectors CSV
    sectors_path = CSV_DIR / "sectors.csv"
    if sectors_path.exists():
        size = sectors_path.stat().st_size
        files.append({"name": "sectors.csv", "path": "sectors.csv", "size": size, "type": "sectors"})

    # Macro CSVs
    for name in ["kse100", "usdpkr", "cpi", "tbill"]:
        p = CSV_DIR / f"macro_{name}.csv"
        if p.exists():
            files.append({"name": f"macro_{name}.csv", "path": f"macro_{name}.csv",
                          "size": p.stat().st_size, "type": "macro"})

    # Symbol CSVs count
    sym_count = len(list(SYMBOLS_CSV_DIR.glob("*.csv"))) if SYMBOLS_CSV_DIR.exists() else 0

    # Source code files (.py)
    base_dir = Path(__file__).resolve().parent.parent
    source_files = []
    py_files = [
        ("backend/config.py", "Configuration, paths, API endpoints, constants"),
        ("backend/fetcher.py", "Stock data fetching from PSX Data Portal"),
        ("backend/macro.py", "Macro data: KSE-100, USD/PKR, CPI, T-bill"),
        ("backend/metrics.py", "Per-stock metric computation (CAGR, Sharpe, beta, etc.)"),
        ("backend/sectors.py", "Sector-level aggregation"),
        ("backend/fundamentals.py", "Fundamentals scraping (P/E, EPS, dividend yield)"),
        ("backend/export.py", "CSV export logic"),
        ("backend/run_pipeline.py", "Pipeline orchestrator"),
        ("app/app.py", "Flask application, routes, chart generation"),
        ("freeze.py", "Static site generator for Netlify deployment"),
        ("run.py", "Main entry point (CLI)"),
    ]
    for rel_path, desc in py_files:
        p = base_dir / rel_path
        if p.exists():
            source_files.append({
                "name": Path(rel_path).name,
                "path": f"source/{Path(rel_path).name}",
                "size": p.stat().st_size,
                "desc": desc,
            })

    return render_template("downloads.html", files=files, sym_count=sym_count,
                           source_files=source_files)


@app.route("/download/<path:filename>")
def download_file(filename):
    # Check in CSV_DIR first
    filepath = CSV_DIR / filename
    if filepath.exists():
        return send_from_directory(CSV_DIR, filename, as_attachment=True)
    # Check in symbols dir
    filepath = SYMBOLS_CSV_DIR / filename
    if filepath.exists():
        return send_from_directory(SYMBOLS_CSV_DIR, filename, as_attachment=True)
    # Check source files
    if filename.startswith("source/"):
        base_dir = Path(__file__).resolve().parent.parent
        pyname = filename.replace("source/", "")
        # Map filenames back to their actual paths
        source_map = {
            "config.py": base_dir / "backend" / "config.py",
            "fetcher.py": base_dir / "backend" / "fetcher.py",
            "macro.py": base_dir / "backend" / "macro.py",
            "metrics.py": base_dir / "backend" / "metrics.py",
            "sectors.py": base_dir / "backend" / "sectors.py",
            "fundamentals.py": base_dir / "backend" / "fundamentals.py",
            "export.py": base_dir / "backend" / "export.py",
            "run_pipeline.py": base_dir / "backend" / "run_pipeline.py",
            "app.py": base_dir / "app" / "app.py",
            "freeze.py": base_dir / "freeze.py",
            "run.py": base_dir / "run.py",
        }
        src = source_map.get(pyname)
        if src and src.exists():
            return send_from_directory(src.parent, src.name, as_attachment=True)
    abort(404)


@app.route("/download/symbol/<symbol>")
def download_symbol(symbol):
    symbol = symbol.upper()
    filename = f"{symbol}.csv"
    filepath = SYMBOLS_CSV_DIR / filename
    if filepath.exists():
        return send_from_directory(SYMBOLS_CSV_DIR, filename, as_attachment=True)
    abort(404)


@app.route("/api/symbols")
def api_symbols():
    master = load_master()
    if master.empty:
        return jsonify([])
    symbols = sorted(master["symbol"].dropna().unique().tolist())
    return jsonify(symbols)


@app.route("/manifest.json")
def manifest():
    return send_from_directory(app.static_folder, "manifest.json")


@app.route("/sw.js")
def service_worker():
    return send_from_directory(app.static_folder, "sw.js",
                               mimetype="application/javascript")


def create_app():
    return app


if __name__ == "__main__":
    app.run(debug=True, port=5000)
