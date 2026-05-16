"""A-share TradingAgents runner — 一键配置 A 股数据源并启动分析。"""
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
        "core_stock_apis": "akshare",
        "technical_indicators": "akshare",
        "fundamental_data": "akshare",
        "news_data": "akshare",
    },
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
