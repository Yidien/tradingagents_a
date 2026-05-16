"""A-share sentiment data via akshare — replaces Reddit + StockTwits sources.

Provides Chinese investor sentiment from East Money stock bar (股吧) and
Xueqiu (雪球) platform. These sources are functionally equivalent to
StockTwits for US stocks: retail-trader posts with sentiment signals.

Exports ``fetch_akshare_sentiment`` and ``fetch_akshare_stock_bar``
which the sentiment analyst can import instead of the US community
sources.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

from .akshare_source import _normalise_ticker

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def fetch_akshare_sentiment(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
) -> str:
    """Fetch retail sentiment from East Money stock bar (股吧) and Xueqiu.

    Returns a formatted plaintext block ready for prompt injection. Same
    return semantics as :func:`reddit.fetch_reddit_posts` and
    :func:`stocktwits.fetch_stocktwits_messages`.

    Contains:
    1. East Money stock bar posts — with like/comment counts, user sentiment
    2. If available, Xueqiu hot posts for the ticker
    """
    code = _normalise_ticker(ticker)
    blocks = []

    # East Money stock bar
    em_block = _fetch_eastmoney_bar(code, limit, timeout)
    if em_block:
        blocks.append(_format_stock_bar("East Money Stock Bar (东方财富股吧)", em_block))
    else:
        blocks.append(
            f"East Money Stock Bar (东方财富股吧): "
            f"<no posts found for {ticker} in the past 7 days>"
        )

    # Xueqiu (optional, may fail without cookies)
    xq_block = _fetch_xueqiu(code, min(limit, 10), timeout)
    if xq_block:
        blocks.append(_format_xueqiu("Xueqiu (雪球)", xq_block))

    return "\n\n".join(blocks) if blocks else f"<no sentiment data found for {ticker}>"


def fetch_akshare_stock_bar(
    ticker: str,
    limit: int = 30,
    timeout: float = 10.0,
) -> str:
    """Fetch East Money stock bar posts for *ticker*.

    Functionally equivalent to :func:`stocktwits.fetch_stocktwits_messages`.
    Returns a string with summary stats + post list.
    """
    code = _normalise_ticker(ticker)
    posts = _fetch_eastmoney_bar(code, limit, timeout)
    if not posts:
        return f"<no East Money stock bar posts found for {ticker}>"

    bullish = sum(1 for p in posts if p.get("sentiment") == "bullish")
    bearish = sum(1 for p in posts if p.get("sentiment") == "bearish")
    neutral = len(posts) - bullish - bearish
    total = len(posts)

    if total > 0:
        bp = round(100 * bullish / total)
        sp = round(100 * bearish / total)
    else:
        bp = sp = 0

    summary = (
        f"Bullish: {bullish} ({bp}%) · "
        f"Bearish: {bearish} ({sp}%) · "
        f"Neutral: {neutral} · "
        f"Total: {total} most-recent posts"
    )
    lines = [summary, ""]
    for p in posts:
        sent = p.get("sentiment", "neutral")
        tag = {"bullish": "Bullish", "bearish": "Bearish"}.get(sent, "no-label")
        body = p["body"]
        if len(body) > 240:
            body = body[:240] + "…"
        lines.append(f"[{p.get('time', '?')} · {p.get('user', '?')} · {tag}] {body}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal: East Money stock bar scraper
# ---------------------------------------------------------------------------

def _fetch_eastmoney_bar(code: str, limit: int, timeout: float) -> list[dict]:
    """Fetch recent posts from East Money stock bar for *code*."""
    try:
        url = (
            f"https://guba.eastmoney.com/list,{code},f_{1}.html"
        )
        # Use East Money's JSONP API (undocumented but public)
        api_url = (
            f"https://guba.eastmoney.com/interface/GetData.aspx?"
            f"code={code}&ps={limit}&type=1&f=1&path=reply/api/Article/NewArticleList"
        )
        headers = {"User-Agent": _UA, "Referer": url}
        resp = requests.get(api_url, headers=headers, timeout=timeout)
        resp.raise_for_status()

        raw = resp.text
        # Strip JSONP wrapper
        if raw.startswith("jQuery"):
            raw = raw[raw.index("(") + 1 : raw.rindex(")")]
        data = json.loads(raw)

        posts = []
        items = data.get("re", []) if isinstance(data, dict) else []
        for item in items:
            title = (item.get("post_title") or "").strip()
            body = (item.get("post_content") or "").strip()
            user = item.get("post_user", {}).get("user_nickname", "anonymous")
            post_time = item.get("post_publish_time", "")
            like_count = item.get("post_click_count", 0)
            comment_count = item.get("post_comment_count", 0)

            # Simple heuristic sentiment based on keywords
            text = (title + " " + body).lower()
            sentiment = "neutral"
            if any(w in text for w in ["暴涨", "涨停", "买入", "看多", "利好", "买进", "起飞", "加仓"]):
                sentiment = "bullish"
            elif any(w in text for w in ["暴跌", "跌停", "卖出", "看空", "利空", "清仓", "割肉", "跳水"]):
                sentiment = "bearish"

            posts.append({
                "user": user,
                "body": title + (" — " + body if body else ""),
                "time": post_time,
                "sentiment": sentiment,
                "likes": like_count,
                "comments": comment_count,
            })
            if len(posts) >= limit:
                break

        return posts
    except Exception as e:
        logger.warning("East Money stock bar fetch failed for %s: %s", code, e)
        return []


# ---------------------------------------------------------------------------
# Internal: Xueqiu scraper
# ---------------------------------------------------------------------------

def _fetch_xueqiu(code: str, limit: int, timeout: float) -> list[dict]:
    """Fetch hot posts from Xueqiu for *code*."""
    try:
        url = f"https://xueqiu.com/query/v1/symbol/search/status.json"
        headers = {
            "User-Agent": _UA,
            "Referer": "https://xueqiu.com/",
        }
        session = requests.Session()
        # Get cookies first
        session.get("https://xueqiu.com/", headers=headers, timeout=timeout)
        params = {
            "symbol": f"SH{code}" if code.startswith("6") else f"SZ{code}",
            "count": limit,
            "page": 1,
        }
        resp = session.get(url, headers=headers, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        posts = []
        items = data.get("list", []) if isinstance(data, dict) else []
        for item in items:
            title = (item.get("title") or item.get("text") or "").strip()
            body = (item.get("description") or item.get("text") or "").strip()
            user = item.get("user", {}).get("screen_name", "anonymous")
            created = item.get("created_at", "")
            like_count = item.get("like_count", 0)
            reply_count = item.get("reply_count", 0)

            posts.append({
                "user": user,
                "body": title,
                "full_body": body,
                "time": _ts_to_str(created),
                "likes": like_count,
                "comments": reply_count,
            })
            if len(posts) >= limit:
                break
        return posts
    except Exception as e:
        logger.warning("Xueqiu fetch failed for %s: %s", code, e)
        return []


def _ts_to_str(ts: int) -> str:
    """Convert Unix timestamp (ms) to date string."""
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts / 1000))
    except Exception:
        return "?"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_stock_bar(platform: str, posts: list[dict]) -> str:
    """Format East Money stock bar posts as a text block."""
    lines = [f"{platform} — {len(posts)} recent posts:"]
    for p in posts:
        likes = p.get("likes", 0)
        comments = p.get("comments", 0)
        sent = p.get("sentiment", "neutral")
        tag = {"bullish": "Bullish", "bearish": "Bearish"}.get(sent, "中性")
        lines.append(
            f"  [{p.get('time', '?')} · 👍{likes} · 💬{comments} · {tag}] "
            f"{p['body'][:240]}"
        )
    return "\n".join(lines)


def _format_xueqiu(platform: str, posts: list[dict]) -> str:
    """Format Xueqiu posts as a text block."""
    lines = [f"{platform} — {len(posts)} recent posts:"]
    for p in posts:
        likes = p.get("likes", 0)
        comments = p.get("comments", 0)
        lines.append(
            f"  [{p.get('time', '?')} · 👍{likes} · 💬{comments}] "
            f"{p['body'][:240]}"
        )
    return "\n".join(lines)
