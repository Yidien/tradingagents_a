"""akshare-based A-share stock data, indicators, and financial statements.

Drop-in replacement for y_finance.py when ``data_vendors`` is set to
``akshare``. Follows the same function signatures so ``interface.py``
can route to this module without any other changes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Annotated, Optional

import akshare as ak
import pandas as pd
from dateutil.relativedelta import relativedelta

from .stockstats_utils import (
    StockstatsUtils,
    _clean_dataframe,
    filter_financials_by_date,
)
from .config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A-share ticker helpers
# ---------------------------------------------------------------------------

def _normalise_ticker(symbol: str) -> str:
    """Strip exchange suffix so we get a clean 6-digit A-share code.

    ``600519.SH`` → ``600519``, ``000001.SZ`` → ``000001``.
    Non-A-share tickers are returned unchanged.
    """
    upper = symbol.upper().strip()
    if upper.endswith(".SH") or upper.endswith(".SZ"):
        return upper[:6]
    return upper


def _is_a_share(symbol: str) -> bool:
    """Return True if *symbol* looks like an A-share ticker."""
    code = _normalise_ticker(symbol)
    return code.isdigit() and len(code) == 6


def _ak_retry(func, max_retries=2, base_delay=3.0):
    """Simple retry wrapper for akshare calls (network can be flaky)."""
    import time
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "akshare call failed, retrying in %.0fs (attempt %d/%d)",
                    delay, attempt + 1, max_retries,
                )
                time.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# OHLCV stock data
# ---------------------------------------------------------------------------

def get_akshare_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch A-share OHLCV data from akshare.

    Signature mirrors :func:`y_finance.get_YFin_data_online` exactly so
    ``interface.py`` can route here via ``VENDOR_METHODS``.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    code = _normalise_ticker(symbol)

    try:
        df = _ak_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",  # forward-adjusted (前复权)
            )
        )
    except Exception as e:
        return (
            f"Error fetching stock data for '{symbol}' from akshare: {e}"
        )

    if df is None or df.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # akshare columns: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
    col_map = {
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    for needed in ("Open", "High", "Low", "Close", "Volume"):
        if needed not in df.columns:
            df[needed] = pd.NA

    # "Adj Close" is already the adjusted close in qfq mode
    df["Adj Close"] = df["Close"]

    # Set Date as index for CSV output
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    # Round numeric columns
    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_string = df.to_csv()
    header = (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare (前复权)\n\n"
    )
    return header + csv_string


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def get_stock_stats_indicators_akshare(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    """Compute technical indicators for A-shares.

    Reuses the same indicator descriptions and stockstats computation
    as yfinance but feeds akshare-derived OHLCV into :mod:`stockstats`.
    """
    indicator_descriptions = {
        "close_50_sma": "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.",
        "close_200_sma": "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.",
        "close_10_ema": "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.",
        "macd": "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.",
        "macds": "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.",
        "macdh": "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.",
        "rsi": "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.",
        "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.",
        "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.",
        "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.",
        "atr": "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.",
        "vwma": "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.",
        "mfi": "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals.",
    }

    if indicator not in indicator_descriptions:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Please choose from: {list(indicator_descriptions.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    try:
        indicator_data = _get_akshare_stock_stats_bulk(symbol, indicator, curr_date)

        current_dt = curr_date_dt
        date_values = []
        while current_dt >= before:
            date_str = current_dt.strftime("%Y-%m-%d")
            if date_str in indicator_data:
                date_values.append((date_str, indicator_data[date_str]))
            else:
                date_values.append((date_str, "N/A: Not a trading day (weekend or holiday)"))
            current_dt = current_dt - relativedelta(days=1)

        ind_string = ""
        for d, v in date_values:
            ind_string += f"{d}: {v}\n"
    except Exception as e:
        logger.warning("Bulk stockstats with akshare failed: %s", e)
        # Fallback — try individual lookups
        ind_string = ""
        cur = datetime.strptime(curr_date, "%Y-%m-%d")
        while cur >= before:
            try:
                v = StockstatsUtils.get_stock_stats(symbol, indicator, cur.strftime("%Y-%m-%d"))
            except Exception:
                v = "N/A"
            ind_string += f"{cur.strftime('%Y-%m-%d')}: {v}\n"
            cur = cur - relativedelta(days=1)

    desc = indicator_descriptions.get(indicator, "No description available.")
    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n" + desc
    )


def _get_akshare_stock_stats_bulk(symbol: str, indicator: str, curr_date: str) -> dict:
    """Calculate indicator for all available dates using akshare-sourced OHLCV."""
    from stockstats import wrap
    data = _load_ohclv_akshare(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    df[indicator]  # triggers stockstats computation

    result = {}
    for _, row in df.iterrows():
        d = row["Date"]
        v = row[indicator]
        result[d] = str(v) if not pd.isna(v) else "N/A"
    return result


def _load_ohclv_akshare(symbol: str, curr_date: str) -> pd.DataFrame:
    """Load OHLCV from akshare cache (or fetch if missing), filtered to curr_date.

    Mirrors :func:`stockstats_utils.load_ohlcv` but uses akshare.
    """
    from .utils import safe_ticker_component

    code = _normalise_ticker(symbol)
    safe_symbol = safe_ticker_component(code)
    config = get_config()
    cache_dir = config.get("data_cache_dir", os.path.join(os.path.expanduser("~"), ".tradingagents", "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"akshare_ohlcv_{safe_symbol}.csv.zip")

    curr_date_dt = pd.to_datetime(curr_date)

    # Try cache first
    if os.path.exists(cache_path):
        try:
            data = pd.read_csv(cache_path, compression="zip")
            data = _clean_dataframe(data)
            data = data[data["Date"] <= curr_date_dt]
            if not data.empty:
                return data
        except Exception:
            pass

    # Fetch from akshare
    try:
        end_str = curr_date_dt.strftime("%Y%m%d")
        start_str = (curr_date_dt - relativedelta(years=15)).strftime("%Y%m%d")
        df = _ak_retry(
            lambda: ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_str, end_date=end_str, adjust="qfq",
            )
        )
    except Exception:
        # Try reduced range if 15 years fails
        try:
            start_str = (curr_date_dt - relativedelta(years=5)).strftime("%Y%m%d")
            df = _ak_retry(
                lambda: ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_str, end_date=end_str, adjust="qfq",
                )
            )
        except Exception as e:
            raise RuntimeError(f"akshare OHLCV fetch failed for {symbol}: {e}")

    # Normalise column names to match stockstats expectations
    col_map = {
        "日期": "Date",
        "开盘": "Open",
        "最高": "High",
        "最低": "Low",
        "收盘": "Close",
        "成交量": "Volume",
        "振幅": "Amplitude",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            df[col] = pd.NA

    data = _clean_dataframe(df)
    data = data[data["Date"] <= curr_date_dt]

    # Cache
    try:
        data.to_csv(cache_path, index=False, compression="zip")
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------------
# Fundamentals
# ---------------------------------------------------------------------------

def get_fundamentals_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for akshare)"] = None,
) -> str:
    """Get A-share company fundamentals from akshare East Money data."""
    code = _normalise_ticker(ticker)
    try:
        info = _ak_retry(lambda: ak.stock_individual_info_em(symbol=code))
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker} from akshare: {e}"

    if info is None or info.empty:
        return f"No fundamentals data found for symbol '{ticker}'"

    # akshare returns a DataFrame with columns: item, value
    info_dict = dict(zip(info["item"], info["value"])) if "item" in info.columns else {}

    # Map common fields
    field_map = {
        "总市值": "Market Cap",
        "流通市值": "Circulating Market Cap",
        "行业": "Industry",
        "上市时间": "Listed Date",
        "股票简称": "Name",
        "总股本": "Total Shares",
        "流通股": "Circulating Shares",
        "市盈率-动态": "PE (TTM)",
    }

    lines = [f"# Company Fundamentals for {ticker} (akshare)\n"]
    for cn_key, en_label in field_map.items():
        val = info_dict.get(cn_key)
        if val is not None:
            lines.append(f"{en_label}: {val}")

    # Also include any other items
    for _, row in info.iterrows():
        item = row.get("item", "")
        val = row.get("value", "")
        if item not in field_map and val is not None:
            lines.append(f"{item}: {val}")

    return "\n".join(lines)


def get_balance_sheet_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share balance sheet from akshare."""
    code = _normalise_ticker(ticker)
    try:
        df = _ak_retry(lambda: ak.stock_balance_sheet_by_report_em(symbol=code))
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {e}"

    if df is None or df.empty:
        return f"No balance sheet data found for '{ticker}'"

    return _format_financial("Balance Sheet", ticker, df, curr_date)


def get_cashflow_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share cash flow from akshare."""
    code = _normalise_ticker(ticker)
    try:
        df = _ak_retry(lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=code))
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {e}"

    if df is None or df.empty:
        return f"No cash flow data found for '{ticker}'"

    return _format_financial("Cash Flow", ticker, df, curr_date)


def get_income_statement_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Get A-share income statement from akshare."""
    code = _normalise_ticker(ticker)
    try:
        df = _ak_retry(lambda: ak.stock_profit_sheet_by_report_em(symbol=code))
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {e}"

    if df is None or df.empty:
        return f"No income statement data found for '{ticker}'"

    return _format_financial("Income Statement", ticker, df, curr_date)


def _format_financial(sheet_type: str, ticker: str, df: pd.DataFrame, curr_date: str = None) -> str:
    """Format a financial statement DataFrame for prompt injection."""
    # Filter by date if provided
    if curr_date and "报告日期" in df.columns:
        df["报告日期"] = pd.to_datetime(df["报告日期"])
        curr = pd.to_datetime(curr_date)
        df = df[df["报告日期"] <= curr]

    if df.empty:
        return f"No {sheet_type} data found for '{ticker}' up to {curr_date}"

    csv_string = df.to_csv(index=False)
    header = (
        f"# {sheet_type} data for {ticker}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"# Source: akshare\n\n"
    )
    return header + csv_string


def get_insider_transactions_akshare(
    ticker: Annotated[str, "ticker symbol of the company"],
) -> str:
    """Insider transactions are not available via akshare for A-shares.

    Returns a clear placeholder so agents know the data is unavailable
    rather than silently missing.
    """
    return (
        f"# Insider Transactions for {ticker}\n"
        "Insider transaction data is not available for A-share stocks via akshare. "
        "A-share listed companies disclose major shareholder changes through "
        "periodic filings on cninfo.com.cn rather than real-time insider trade feeds.\n"
    )


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------
import os  # noqa: E402 (used in _load_ohclv_akshare)
