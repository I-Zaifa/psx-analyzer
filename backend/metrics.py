"""
Per-stock metrics computation:
Total return, CAGR, volatility, max drawdown, Sharpe, beta, correlation, etc.
"""
import logging
from typing import Optional, Dict

import pandas as pd
import numpy as np

from backend.config import TRADING_DAYS_PER_YEAR, DEFAULT_RISK_FREE_RATE

logger = logging.getLogger(__name__)


def compute_total_return(prices: pd.Series) -> Optional[float]:
    """Total return from first to last close."""
    if len(prices) < 2:
        return None
    first = prices.iloc[0]
    last = prices.iloc[-1]
    if first <= 0:
        return None
    return (last - first) / first


def compute_cagr(prices: pd.Series) -> Optional[float]:
    """Compound Annual Growth Rate."""
    if len(prices) < 2:
        return None
    first = prices.iloc[0]
    last = prices.iloc[-1]
    if first <= 0:
        return None
    days = (prices.index[-1] - prices.index[0]).days
    if days <= 0:
        return None
    years = days / 365.25
    if years < 0.1:
        return None
    return (last / first) ** (1 / years) - 1


def compute_annualized_volatility(prices: pd.Series) -> Optional[float]:
    """Annualized volatility from daily log returns."""
    if len(prices) < 20:
        return None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    if len(log_returns) < 10:
        return None
    return log_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_max_drawdown(prices: pd.Series) -> Optional[float]:
    """Maximum peak-to-trough drawdown."""
    if len(prices) < 2:
        return None
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown.min()


def compute_sharpe_ratio(prices: pd.Series, risk_free_rate: float = DEFAULT_RISK_FREE_RATE) -> Optional[float]:
    """Annualized Sharpe ratio."""
    cagr = compute_cagr(prices)
    vol = compute_annualized_volatility(prices)
    if cagr is None or vol is None or vol == 0:
        return None
    return (cagr - risk_free_rate) / vol


def compute_beta(stock_prices: pd.Series, index_prices: pd.Series) -> Optional[float]:
    """Beta relative to index."""
    if len(stock_prices) < 20 or len(index_prices) < 20:
        return None

    # Normalize dates to midnight for alignment
    sp = stock_prices.copy()
    ip = index_prices.copy()
    sp.index = sp.index.normalize()
    ip.index = ip.index.normalize()
    sp = sp[~sp.index.duplicated(keep='last')]
    ip = ip[~ip.index.duplicated(keep='last')]

    combined = pd.DataFrame({
        "stock": sp,
        "index": ip
    }).dropna()

    if len(combined) < 20:
        return None

    stock_returns = np.log(combined["stock"] / combined["stock"].shift(1)).dropna()
    index_returns = np.log(combined["index"] / combined["index"].shift(1)).dropna()

    # Align after computing returns
    aligned = pd.DataFrame({"stock": stock_returns, "index": index_returns}).dropna()
    if len(aligned) < 20:
        return None

    cov = aligned["stock"].cov(aligned["index"])
    var = aligned["index"].var()
    if var == 0:
        return None
    return cov / var


def compute_correlation(stock_prices: pd.Series, index_prices: pd.Series) -> Optional[float]:
    """Correlation with index."""
    if len(stock_prices) < 20 or len(index_prices) < 20:
        return None

    sp = stock_prices.copy()
    ip = index_prices.copy()
    sp.index = sp.index.normalize()
    ip.index = ip.index.normalize()
    sp = sp[~sp.index.duplicated(keep='last')]
    ip = ip[~ip.index.duplicated(keep='last')]

    combined = pd.DataFrame({
        "stock": sp,
        "index": ip
    }).dropna()

    if len(combined) < 20:
        return None

    stock_returns = np.log(combined["stock"] / combined["stock"].shift(1)).dropna()
    index_returns = np.log(combined["index"] / combined["index"].shift(1)).dropna()

    aligned = pd.DataFrame({"stock": stock_returns, "index": index_returns}).dropna()
    if len(aligned) < 20:
        return None

    return aligned["stock"].corr(aligned["index"])


def compute_volume_trend(df: pd.DataFrame) -> Optional[float]:
    """Recent volume trend: 30-day avg / 90-day avg ratio."""
    if "volume" not in df.columns or len(df) < 90:
        return None
    vol = df["volume"].dropna()
    if len(vol) < 90:
        return None
    avg_30 = vol.iloc[-30:].mean()
    avg_90 = vol.iloc[-90:].mean()
    if avg_90 == 0:
        return None
    return avg_30 / avg_90


def compute_52w_high_low(prices: pd.Series) -> Dict:
    """52-week high and low."""
    if len(prices) < 10:
        return {"52w_high": None, "52w_low": None, "pct_from_52w_high": None}
    last_year = prices.iloc[-min(TRADING_DAYS_PER_YEAR, len(prices)):]
    high = last_year.max()
    low = last_year.min()
    current = prices.iloc[-1]
    pct_from_high = (current - high) / high if high > 0 else None
    return {"52w_high": high, "52w_low": low, "pct_from_52w_high": pct_from_high}


def compute_period_returns(prices: pd.Series) -> Dict:
    """1M, 3M, 6M, 1Y, YTD returns."""
    result = {"return_1m": None, "return_3m": None, "return_6m": None,
              "return_1y": None, "return_ytd": None}
    if len(prices) < 2:
        return result

    current = prices.iloc[-1]
    if current <= 0:
        return result

    # Period returns
    periods = {
        "return_1m": 21,
        "return_3m": 63,
        "return_6m": 126,
        "return_1y": TRADING_DAYS_PER_YEAR,
    }
    for key, days in periods.items():
        if len(prices) > days:
            past = prices.iloc[-days - 1]
            if past > 0:
                result[key] = (current - past) / past

    # YTD
    current_year = prices.index[-1].year
    ytd_prices = prices[prices.index.year == current_year]
    if len(ytd_prices) >= 2:
        first_of_year = ytd_prices.iloc[0]
        if first_of_year > 0:
            result["return_ytd"] = (current - first_of_year) / first_of_year

    return result


def compute_all_metrics(symbol: str,
                        df: pd.DataFrame,
                        index_prices: Optional[pd.Series] = None,
                        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                        cpi_data: Optional[pd.DataFrame] = None) -> Dict:
    """Compute all metrics for a single stock."""
    result = {"symbol": symbol}

    if df is None or df.empty or "close" not in df.columns:
        return result

    prices = df["close"].dropna()
    if len(prices) < 5:
        return result

    # Basic info
    result["first_date"] = prices.index[0].strftime("%Y-%m-%d")
    result["last_date"] = prices.index[-1].strftime("%Y-%m-%d")
    result["data_points"] = len(prices)
    result["last_close"] = prices.iloc[-1]

    # Core metrics
    result["total_return"] = compute_total_return(prices)
    result["cagr"] = compute_cagr(prices)
    result["annualized_volatility"] = compute_annualized_volatility(prices)
    result["max_drawdown"] = compute_max_drawdown(prices)
    result["sharpe_ratio"] = compute_sharpe_ratio(prices, risk_free_rate)

    # Market-relative metrics
    if index_prices is not None and not index_prices.empty:
        result["beta"] = compute_beta(prices, index_prices)
        result["correlation"] = compute_correlation(prices, index_prices)

    # Volume trend
    result["volume_trend"] = compute_volume_trend(df)

    # 52-week
    hw = compute_52w_high_low(prices)
    result.update(hw)

    # Period returns
    pr = compute_period_returns(prices)
    result.update(pr)

    # Inflation-adjusted CAGR
    if cpi_data is not None and not cpi_data.empty and result.get("cagr") is not None:
        try:
            avg_inflation = cpi_data["cpi_yoy"].mean() / 100.0
            result["real_cagr"] = result["cagr"] - avg_inflation
        except Exception:
            result["real_cagr"] = None
    else:
        result["real_cagr"] = None

    # Average daily volume
    if "volume" in df.columns:
        result["avg_daily_volume"] = df["volume"].mean()
        result["last_volume"] = df["volume"].iloc[-1] if not pd.isna(df["volume"].iloc[-1]) else None

    return result


def compute_metrics_for_all(stocks_data: Dict[str, pd.DataFrame],
                            index_prices: Optional[pd.Series] = None,
                            risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
                            cpi_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute metrics for all stocks and return as DataFrame."""
    logger.info(f"Computing metrics for {len(stocks_data)} stocks...")
    results = []

    for symbol, df in stocks_data.items():
        try:
            metrics = compute_all_metrics(symbol, df, index_prices, risk_free_rate, cpi_data)
            results.append(metrics)
        except Exception as e:
            logger.error(f"Metrics computation failed for {symbol}: {e}")
            results.append({"symbol": symbol})

    metrics_df = pd.DataFrame(results)
    logger.info(f"Computed metrics for {len(metrics_df)} stocks")
    return metrics_df
