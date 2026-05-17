"""A-share TradingAgents runner — 一键配置 A 股数据源并启动分析。

数据源: 东方财富(OHLCV/财报/新闻), 财联社(快讯), 同花顺(预测/新闻),
       雪球(热度排行), 新闻联播(政策), 百度(经济日历)
"""
import os

# === 缓存切到 D 盘 ===
os.environ["TRADINGAGENTS_CACHE_DIR"] = "D:/tradingagents_cache"
os.environ["TRADINGAGENTS_RESULTS_DIR"] = "D:/tradingagents_logs"

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.copy()
config.update({
    # === 数据源全部走 akshare ===
    "data_vendors": {
        "core_stock_apis": "akshare",       # 东财日K线
        "technical_indicators": "akshare",   # 东财日K → stockstats
        "fundamental_data": "akshare",       # 东财财报 + 同花顺盈利预测
        "news_data": "akshare",              # 东财个股新闻
    },
    # === 宏观新闻多源配置 ===
    "global_news_sources": [
        "cctv",         # 新闻联播 — 政策定调
        "cls",          # 财联社电报 — 实时快讯
        "eastmoney",    # 东方财富全球财经 — 覆盖面广
        "tonghuashun",  # 同花顺全球财经 — 互补视角
        "baidu",        # 百度经济日历 — 宏观数据
    ],
    # === A股宏观新闻关键词（中文） ===
    "global_news_queries": [
        "央行 利率 货币政策",
        "A股 政策 证监会",
        "GDP 经济 数据 PMI",
        "贸易 关税 进出口",
        "房地产 基建 投资",
    ],
    # === LLM 配置（从 .env 读取，这里设默认） ===
    "output_language": "Chinese",
    "max_debate_rounds": 1,
})

# 👇 修改这里分析你关注的 A 股
TICKER = "600519.SH"  # 贵州茅台（带交易所后缀，基准自动映射到沪深300）
DATE = "2025-05-10"  # 分析日期

ta = TradingAgentsGraph(debug=True, config=config)
_, decision = ta.propagate(TICKER, DATE)
print("\n" + "=" * 60)
print(decision)
