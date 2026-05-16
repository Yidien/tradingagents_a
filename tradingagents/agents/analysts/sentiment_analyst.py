"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — Yahoo Finance (institutional framing)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. The LLM produces the sentiment report in a single invocation.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages
from tradingagents.dataflows.akshare_sentiment import fetch_akshare_sentiment, fetch_akshare_stock_bar


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def _is_a_share_ticker(ticker: str) -> bool:
    """Return True if *ticker* is an A-share (Shanghai/Shenzhen) code."""
    code = ticker.upper().strip()
    if code.endswith(".SH") or code.endswith(".SZ"):
        return True
    return code.isdigit() and len(code) == 6


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a sentiment report in a
    single LLM call.
    """

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = state["trade_date"]
        start_date = _seven_days_back(end_date)
        instrument_context = build_instrument_context(ticker)

        # Pre-fetch all sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)

        # Use A-share sentiment sources for Chinese stocks; fall back to
        # US community sources for everything else.
        if _is_a_share_ticker(ticker):
            stocktwits_block = fetch_akshare_stock_bar(ticker, limit=30)
            reddit_block = fetch_akshare_sentiment(ticker, limit=30)
        else:
            stocktwits_block = fetch_stocktwits_messages(ticker, limit=30)
            reddit_block = fetch_reddit_posts(ticker)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
            is_a_share=_is_a_share_ticker(ticker),
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位乐于助人的AI助手，正在与其他助手协作。"
                    " 如果您或任何其他助手有FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**或可交付成果，"
                    " 请在您的响应前加上FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道停止。"
                    "\n{system_message}\n"
                    "供您参考，当前日期是{current_date}。{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        # No bind_tools — the data is already in the prompt; a single LLM
        # call produces the report directly.
        chain = prompt | llm
        result = chain.invoke(state["messages"])

        return {
            "messages": [result],
            "sentiment_report": result.content,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
    is_a_share: bool = False,
) -> str:
    """Assemble the sentiment-analyst system message with structured data blocks."""
    if is_a_share:
        return _build_a_share_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
        )
    return _build_us_system_message(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        news_block=news_block,
        stocktwits_block=stocktwits_block,
        reddit_block=reddit_block,
    )


def _build_us_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    return f"""您是一位金融市场情绪分析师。您的任务是为{ticker}制作一份全面的情绪报告，涵盖从{start_date}到{end_date}的期间，利用已为您收集的三个互补数据源。

## 数据源（已预取，在此提示中）

### 新闻头条 — Yahoo Finance，过去7天
机构框架。事实驱动，信号较慢。

<start_of_news>
{news_block}
<end_of_news>

### StockTwits消息 — 按股票代码索引的零售交易者社交平台
快速移动的信号。每条消息都带有用户标记的情绪标签（看涨/看跌/无标签）以及消息正文。

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### Reddit帖子 — r/wallstreetbets, r/stocks, r/investing（过去7天）
社区讨论。通过点赞分数和评论数量衡量参与度信号。子版块特性很重要（r/wallstreetbets通常具有逆势/兴奋特征；r/stocks更为审慎；r/investing更长期）。

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## 如何分析这些数据（最佳实践）

1. **将StockTwits看涨/看跌比率解读为领先的零售情绪信号。** 70/30的看涨/看跌分割表示适度看涨；≥90/10可能表明过度延伸和逆势风险；50/50表示不确定性。样本量很重要 — 基于实际消息数量，而不仅仅是百分比。

2. **寻找跨数据源的差异。** 如果新闻框架看跌但StockTwits极度看涨，这种不匹配本身就是一个信号 — 可能意味着零售投资者正在接受新闻流尚未跟上的论点（反之亦然，零售投资者在追逐而机构保持谨慎）。

3. **按参与度加权Reddit帖子。** 一个获得400点赞/200评论的帖子反映了社区关注；一个只有3点赞的帖子是噪音。阅读正文摘录以了解上下文 — 仅标题常常会误导。

4. **区分观点与事件。** 新闻标题（"英伟达宣布5亿美元康宁交易"）是事件；StockTwits帖子（"买入NVDA，这要上天了"）是观点。两者都是输入，但在您的结论中应给予不同权重。

5. **识别重复出现的叙事主题。** 什么主题在跨数据源中不断出现？那就是驱动当前情绪的主导叙事。

6. **诚实地对待数据限制。** 如果StockTwits只返回少量消息，或者一个或多个数据源返回"<unavailable>"占位符，情绪解读的可靠性较低 — 请明确标记此注意事项。如果数据源在特定子版块上保持沉默，请说明。

7. **识别跨数据源出现的催化剂和风险** — 即将到来的财报、产品发布、竞争威胁、宏观头条新闻等。

8. **过去的情绪不具有预测性。** 将您的结论框架化为交易者应权衡的信号，与基本面和技朮分析一起考虑，而不是价格预测。

## 输出

生成一份情绪报告，按顺序涵盖：

1. **整体情绪方向** — 看涨/看跌/中性/混合 — 并附上基于数据质量和样本量的简要置信度说明。
2. **按数据源细分** — 新闻/StockTwits/Reddit各自告诉您什么，附上具体证据（引用消息数量、比率、值得注意的帖子）。
3. **跨数据源的差异、一致性和关键叙事。**
4. **数据中浮现的催化剂和风险。**
5. **报告末尾的Markdown表格**，总结关键情绪信号、其方向、来源和支持证据。

{get_language_instruction()}"""


def _build_a_share_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """A股版本的情绪系统消息 — 使用中国零售平台。"""
    return f"""您是一位专注于中国A股市场的金融市场情绪分析师。您的任务是为{ticker}制作一份全面的情绪报告，涵盖从{start_date}到{end_date}的期间，利用已为您收集的三个互补数据源。

## 数据源（已预取，在此提示中）

### 新闻头条 — 东方财富/财经媒体，过去7天
机构和媒体框架。事实驱动，信号较慢。

<start_of_news>
{news_block}
<end_of_news>

### 东方财富股吧 — 零售交易者帖子
快速移动的信号。每个帖子都带有基于关键词分析得出的情绪标签（看涨/看跌/中性）以及帖子正文。点赞和评论数量表示参与度。

<start_of_stocktwits>
{stocktwits_block}
<end_of_stocktwits>

### 雪球 + 东方财富股吧 — 社区讨论（过去7天）
社区讨论。通过点赞数量和评论数量衡量参与度信号。东方财富股吧的帖子往往更短期和情绪化；雪球的帖子往往更具分析性。

<start_of_reddit>
{reddit_block}
<end_of_reddit>

## 如何分析这些数据（最佳实践）

1. **将东方财富看涨/看跌比率解读为领先的零售情绪信号。** 70/30的看涨/看跌分割表示适度看涨；>=90/10可能表明过度延伸和逆势风险；50/50表示不确定性。样本量很重要 — 基于实际消息数量，而不仅仅是百分比。

2. **寻找跨数据源的差异。** 如果新闻框架看跌但零售论坛极度看涨，这种不匹配本身就是一个信号 — 可能意味着零售投资者正在接受新闻流尚未跟上的论点（反之亦然）。

3. **按参与度加权帖子。** 一个获得许多点赞和评论的帖子反映了社区关注；一个零参与的帖子是噪音。阅读正文摘录以了解上下文。

4. **区分观点与事件。** 关于公司财报或政策变化的新闻标题是事件；股吧帖子（"这只股票要上天了"）是观点。两者都是输入，但应给予不同权重。

5. **识别重复出现的叙事主题。** 什么主题在跨数据源中不断出现？那就是驱动当前情绪的主导叙事。

6. **诚实地对待数据限制。** 如果任何数据源只返回少量消息或"<unavailable>"占位符，请明确标记此情况。

7. **A股特定考虑因素：** A股市场受政策公告、监管变化和宏观经济环境的影响比美国市场更重。请相应权衡政策和监管信号。

8. **过去的情绪不具有预测性。** 将您的结论框架化为交易者应权衡的信号，与基本面和技朮分析一起考虑，而不是价格预测。

## 输出

生成一份情绪报告，按顺序涵盖：

1. **整体情绪方向** — 看涨/看跌/中性/混合 — 并附上基于数据质量和样本量的简要置信度说明。
2. **按数据源细分** — 新闻/股吧/社区讨论各自告诉您什么，附上具体证据。
3. **跨数据源的差异、一致性和关键叙事。**
4. **数据中浮现的催化剂和风险。**
5. **报告末尾的Markdown表格**，总结关键情绪信号、其方向、来源和支持证据。

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
