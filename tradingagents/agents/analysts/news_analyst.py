from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_global_news,
    get_language_instruction,
    get_news,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
        ]

        system_message = (
            "您是一位新闻研究员，负责分析过去一周的近期新闻和趋势。请撰写一份关于当前世界状况的全面报告，该报告与交易和宏观经济学相关。使用可用工具：get_news(query, start_date, end_date)用于公司特定或针对性新闻搜索，以及get_global_news(curr_date, look_back_days, limit)用于更广泛的宏观经济新闻。提供具体、可操作的见解并附上支持证据，以帮助交易者做出明智决策。"
            + """ 确保在报告末尾附加一个Markdown表格，以整理报告中的关键点，使其组织有序且易于阅读。"""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "您是一位乐于助人的AI助手，正在与其他助手协作。"
                    " 使用提供的工具来推进问题的解答。"
                    " 如果您无法完全回答，没关系；其他拥有不同工具的助手"
                    " 将在您离开的地方提供帮助。尽您所能执行以取得进展。"
                    " 如果您或任何其他助手有FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**或可交付成果，"
                    " 请在您的响应前加上FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**，以便团队知道停止。"
                    " 您可以访问以下工具：{tool_names}。\n{system_message}"
                    "供您参考，当前日期是{current_date}。{instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
