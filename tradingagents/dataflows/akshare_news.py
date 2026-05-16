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

    akshare returns the latest ~100 news per ticker. We first try to filter
    by date range; if no matches, we return the most recent news with a
    note that they may fall outside the window.
    """
    code = _normalise_ticker(ticker)

    try:
        raw_df = _ak_retry(lambda: ak.stock_news_em(symbol=code))
    except Exception as e:
        return f"Error fetching news for {ticker} from akshare: {e}"

    if raw_df is None or raw_df.empty:
        return f"No news found for {ticker}"

    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    limit = get_config().get("news_article_limit", 20)

    lines = []
    count = 0

    for _, row in raw_df.iterrows():
        title = str(row.get("新闻标题", "No title"))
        if title in ("nan", "None", ""):
            title = "No title"
        content = str(row.get("新闻内容", ""))
        if content in ("nan", "None"):
            content = ""
        pub_time = row.get("发布时间", "")
        source = str(row.get("文章来源", "unknown"))

        pub_dt = None
        if pub_time and pub_time not in ("nan", "None"):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    pub_dt = datetime.strptime(str(pub_time)[:19], fmt)
                    break
                except ValueError:
                    continue

        if pub_dt and not (start_dt <= pub_dt <= end_dt + relativedelta(days=1)):
            continue

        lines.append(f"### {title} (source: {source})")
        if content:
            lines.append(content[:500])
        lines.append("")
        count += 1
        if count >= limit:
            break

    # If no news in date range, return latest available
    if count == 0:
        lines = []
        count = 0
        for _, row in raw_df.iterrows():
            title = str(row.get("新闻标题", ""))
            if title in ("nan", "None", ""):
                continue
            pub_time = str(row.get("发布时间", ""))
            source = str(row.get("文章来源", "unknown"))
            lines.append(f"### {title} (source: {source}, time: {pub_time})")
            content = str(row.get("新闻内容", ""))
            if content and content not in ("nan", "None"):
                lines.append(content[:500])
            lines.append("")
            count += 1
            if count >= limit:
                break

        if count == 0:
            return f"No news found for {ticker}"

        return (
            f"## {ticker} Latest News (no articles in exact date range {start_date}~{end_date}):\n\n"
            + "\n".join(lines)
        )

    return (
        f"## {ticker} A-Share News, from {start_date} to {end_date}:\n\n"
        + "\n".join(lines)
    )


def get_global_news_akshare(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """Retrieve macro news from Baidu Economic Calendar + CCTV news.

    Combines two non-East Money sources:
    1. ``news_economic_baidu`` — global economic events with actual/forecast values
    2. ``news_cctv`` — Chinese policy/headlines from national TV news

    Falls back gracefully if either source fails.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("global_news_article_limit", 10)

    blocks = []

    # --- Source 1: CCTV news (policy headlines) ---
    try:
        cctv_block = _fetch_cctv_news(curr_date)
        if cctv_block:
            blocks.append(cctv_block)
    except Exception as e:
        logger.warning("CCTV news fetch failed: %s", e)

    # --- Source 2: Baidu economic calendar ---
    try:
        baidu_block = _fetch_baidu_economic(curr_date, look_back_days, limit // 2)
        if baidu_block:
            blocks.append(baidu_block)
    except Exception as e:
        logger.warning("Baidu economic news fetch failed: %s", e)

    if not blocks:
        return (
            f"# Global Macro News\n"
            f"No macro news available around {curr_date}.\n"
        )

    return "\n\n".join(blocks)


def _fetch_cctv_news(curr_date: str) -> str:
    """Fetch CCTV (新闻联播) headlines for one day."""
    df = _ak_retry(lambda: ak.news_cctv(date=curr_date.replace("-", "")))
    if df is None or df.empty:
        return ""

    lines = [f"## 央视新闻联播 — {curr_date}:\n"]
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        content = str(row.get("content", ""))
        if not title or title in ("nan", "None"):
            continue
        lines.append(f"### {title}")
        if content and content not in ("nan", "None"):
            lines.append(content[:400])
        lines.append("")
    return "\n".join(lines) if len(lines) > 1 else ""


def _fetch_baidu_economic(curr_date: str, look_back_days: int, limit: int) -> str:
    """Fetch global economic events from Baidu calendar."""
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    all_rows = []

    for offset in range(look_back_days):
        d = curr_dt - relativedelta(days=offset)
        date_str = d.strftime("%Y%m%d")
        try:
            df = _ak_retry(lambda: ak.news_economic_baidu(date=date_str))
        except Exception:
            continue
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                all_rows.append(row.to_dict())
                if len(all_rows) >= limit * 2:
                    break
        if len(all_rows) >= limit * 2:
            break

    if not all_rows:
        return ""

    # Deduplicate by event
    seen = set()
    unique = []
    for r in all_rows:
        event = str(r.get("事件", ""))
        if event in seen or event in ("nan", "None", ""):
            continue
        seen.add(event)
        unique.append(r)

    lines = [f"## 全球经济日历 — around {curr_date}:\n"]
    count = 0
    for r in unique:
        date_val = r.get("日期", "")
        time_val = r.get("时间", "")
        country = r.get("国家", r.get("country", ""))
        event = r.get("事件", "")
        actual = r.get("今值", "")
        forecast = r.get("预测", "")
        previous = r.get("前值", "")

        parts = [f"### {event}"]
        meta_parts = [f"{country} · {date_val} {time_val}"]
        if actual not in (None, "nan", ""):
            meta_parts.append(f"今值: {actual}")
        if forecast not in (None, "nan", ""):
            meta_parts.append(f"预测: {forecast}")
        if previous not in (None, "nan", ""):
            meta_parts.append(f"前值: {previous}")
        parts.append(" · ".join(meta_parts))
        lines.extend(parts)
        lines.append("")
        count += 1
        if count >= limit:
            break

    return "\n".join(lines) if count > 0 else ""
