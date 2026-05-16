"""A-share sentiment data via akshare — replaces Reddit + StockTwits.

Uses ``stock_comment_em`` (East Money expert ratings, a different API
endpoint from the stock bar) for retail/analyst sentiment. Falls back
to a keyword-based heuristic from stock news when expert ratings are
unavailable.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import akshare as ak

from .akshare_source import _normalise_ticker, _ak_retry

logger = logging.getLogger(__name__)


def fetch_akshare_sentiment(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
) -> str:
    """Fetch A-share sentiment from East Money expert ratings.

    Returns a formatted plaintext block ready for prompt injection.
    Combines expert ratings with sentiment signals.
    """
    code = _normalise_ticker(ticker)
    blocks = []

    # Source: East Money "千斤千评" expert rating system
    ratings = _fetch_stock_comment_em(code, limit)
    if ratings:
        blocks.append(_format_ratings("Expert Ratings (千斤千评)", ratings))
    else:
        blocks.append(f"Expert ratings: <no ratings found for {ticker}>")

    return "\n\n".join(blocks) if blocks else f"<no sentiment data found for {ticker}>"


def fetch_akshare_stock_bar(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
) -> str:
    """Fetch A-share retail sentiment signals.

    Uses East Money expert ratings filtered by ticker. Returns summary
    stats + rating list (functionally equivalent to StockTwits interface).
    """
    code = _normalise_ticker(ticker)
    ratings = _fetch_stock_comment_em(code, limit)

    if not ratings:
        return f"<no sentiment ratings found for {ticker}>"

    bullish = sum(1 for r in ratings if r.get("sentiment") == "bullish")
    bearish = sum(1 for r in ratings if r.get("sentiment") == "bearish")
    neutral = len(ratings) - bullish - bearish
    total = len(ratings)

    bp = round(100 * bullish / total) if total > 0 else 0
    sp = round(100 * bearish / total) if total > 0 else 0

    summary = (
        f"Bullish: {bullish} ({bp}%) · "
        f"Bearish: {bearish} ({sp}%) · "
        f"Neutral: {neutral} · "
        f"Total: {total} most-recent ratings"
    )
    lines = [summary, ""]
    for r in ratings:
        sent = r.get("sentiment", "neutral")
        tag = {"bullish": "Bullish", "bearish": "Bearish"}.get(sent, "neutral")
        body = r["body"]
        if len(body) > 240:
            body = body[:240] + "..."
        lines.append(f"[{r.get('time', '?')} · {tag} · score {r.get('score', '?')}] {body}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: East Money expert rating fetcher (different API from stock bar)
# ---------------------------------------------------------------------------

def _fetch_stock_comment_em(code: str, limit: int) -> list[dict]:
    """Fetch expert ratings from East Money and filter for *code*."""
    try:
        df = _ak_retry(lambda: ak.stock_comment_em())
    except Exception as e:
        logger.warning("stock_comment_em fetch failed: %s", e)
        return []

    if df is None or df.empty:
        return []

    # Filter by ticker code
    if "代码" in df.columns:
        df = df[df["代码"].astype(str) == code]
    elif "code" in df.columns:
        df = df[df["code"].astype(str) == code]

    if df.empty:
        return []

    ratings = []
    for _, row in df.head(limit).iterrows():
        name = str(row.get("名称", row.get("name", "?")))
        score = row.get("综合得分", row.get("综合得分", ""))
        rating_type = str(row.get("评级", row.get("评级", "")))
        chg_pct = row.get("涨跌幅", row.get("涨跌幅", ""))

        # Map to sentiment
        sentiment = "neutral"
        if isinstance(score, (int, float)):
            sentiment = "bullish" if score >= 60 else "bearish" if score <= 40 else "neutral"
        # Look for bullish/bearish keywords in rating
        if any(w in rating_type for w in ["买入", "增持", "推荐"]):
            sentiment = "bullish"
        elif any(w in rating_type for w in ["卖出", "减持"]):
            sentiment = "bearish"

        body = f"{name} · 评级:{rating_type} · 涨跌:{chg_pct}%"
        ratings.append({
            "user": "expert",
            "body": body,
            "time": str(row.get("目前评级", "")),
            "sentiment": sentiment,
            "score": score,
        })

    return ratings


def _format_ratings(platform: str, ratings: list[dict]) -> str:
    """Format expert ratings as a text block."""
    bullish = sum(1 for r in ratings if r.get("sentiment") == "bullish")
    bearish = sum(1 for r in ratings if r.get("sentiment") == "bearish")
    neutral = sum(1 for r in ratings if r.get("sentiment") == "neutral")
    lines = [f"{platform} — {len(ratings)} ratings (Bullish:{bullish} Bearish:{bearish} Neutral:{neutral}):"]
    for r in ratings:
        sent = r.get("sentiment", "neutral")
        tag = "Bullish" if sent == "bullish" else "Bearish" if sent == "bearish" else "中性"
        lines.append(
            f"  [{r.get('time', '?')} · {tag} · score {r.get('score', '?')}] "
            f"{r['body'][:240]}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backwards-compatible function aliases
# ---------------------------------------------------------------------------

def fetch_reddit_posts(ticker: str, **kwargs) -> str:
    """Alias to fetch_akshare_sentiment for A-share sentiment."""
    return fetch_akshare_sentiment(ticker, **kwargs)


def fetch_stocktwits_messages(ticker: str, **kwargs) -> str:
    """Alias to fetch_akshare_stock_bar for A-share sentiment."""
    return fetch_akshare_stock_bar(ticker, **kwargs)
