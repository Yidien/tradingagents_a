"""akshare-based news fetching for A-share stocks.

Drop-in replacement for :mod:`yfinance_news` when ``data_vendors`` is
set to ``akshare``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import akshare as ak
from dateutil.relativedelta import relativedelta

from .config import get_config
from .akshare_source import _ak_retry, _normalise_ticker

logger = logging.getLogger(__name__)


def get_news_akshare(
    ticker: str,
    start_date: str,
    end_date: str,
) -> str:
    """Retrieve A-share stock news from East Money via akshare.

    Signature matches :func:`yfinance_news.get_news_yfinance`.
    """
    code = _normalise_ticker(ticker)

    try:
        raw_df = _ak_retry(lambda: ak.stock_news_em(symbol=code))
    except Exception as e:
        return f"Error fetching news for {ticker} from akshare: {e}"

    if raw_df is None or raw_df.empty:
        return f"No news found for {ticker}"

    # Columns typically: 关键词, 新闻标题, 新闻内容, 发布时间, 文章来源, 新闻链接
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    lines = []
    count = 0
    limit = get_config().get("news_article_limit", 20)

    for _, row in raw_df.iterrows():
        title = str(row.get("新闻标题", "No title"))
        # Skip non-strings
        if title in ("nan", "None", ""):
            title = "No title"
        content = str(row.get("新闻内容", ""))
        if content in ("nan", "None"):
            content = ""
        pub_time = row.get("发布时间", "")
        source = str(row.get("文章来源", "unknown"))

        # Parse publish time
        if pub_time and pub_time not in ("nan", "None"):
            try:
                pub_dt = datetime.strptime(str(pub_time)[:19], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    pub_dt = datetime.strptime(str(pub_time)[:10], "%Y-%m-%d")
                except ValueError:
                    pub_dt = None
            if pub_dt and not (start_dt <= pub_dt <= end_dt + relativedelta(days=1)):
                continue
        else:
            # If no time, include it (best effort)
            pass

        lines.append(f"### {title} (source: {source})")
        if content:
            lines.append(content[:500])
        lines.append("")
        count += 1
        if count >= limit:
            break

    if count == 0:
        return f"No news found for {ticker} between {start_date} and {end_date}"

    return (
        f"## {ticker} A-Share News, from {start_date} to {end_date}:\n\n"
        + "\n".join(lines)
    )


def get_global_news_akshare(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """Retrieve Chinese macro/market news via akshare.

    Falls back to yfinance for global/macro news since akshare's news
    scope is primarily China domestic. This keeps international context
    available for A-share analysis (global macro still matters).
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("global_news_article_limit", 10)
    queries = config.get("global_news_queries", [])

    all_news = []
    seen = set()

    try:
        for query in queries:
            try:
                df = _ak_retry(lambda: ak.stock_news_main_cx(symbol=query))
            except Exception:
                logger.debug("akshare news_main_cx failed for query '%s', skipping", query)
                continue

            if df is None or df.empty:
                continue

            for _, row in df.iterrows():
                title = str(row.get("标题", row.get("title", "")))
                if not title or title in ("nan", "None"):
                    continue
                if title in seen:
                    continue
                seen.add(title)
                content = str(row.get("内容", row.get("content", "")))
                if content in ("nan", "None"):
                    content = ""
                source = str(row.get("来源", row.get("source", "unknown")))
                all_news.append({
                    "title": title,
                    "content": content[:500],
                    "source": source,
                })
                if len(all_news) >= limit:
                    break
            if len(all_news) >= limit:
                break

    except Exception as e:
        logger.warning("akshare global news fetch failed: %s", e)

    if not all_news:
        return (
            f"No macro news found for {curr_date}. "
            "Try setting 'global_news_queries' to Chinese keywords "
            "(e.g. ['央行 利率', 'A股 政策', 'GDP 经济 数据'])."
        )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=look_back_days)
    start_str = start_dt.strftime("%Y-%m-%d")

    lines = [
        f"## China Market Macro News, from {start_str} to {curr_date}:\n"
    ]
    for item in all_news[:limit]:
        lines.append(f"### {item['title']} (source: {item['source']})")
        if item["content"]:
            lines.append(item["content"])
        lines.append("")

    return "\n".join(lines)
