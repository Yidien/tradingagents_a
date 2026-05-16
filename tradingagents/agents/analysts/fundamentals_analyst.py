from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "您是一位研究员，负责分析过去一周关于公司的基本面信息。请撰写一份关于公司基本面信息的全面报告，例如财务文件、公司简介、基本公司财务数据和公司财务历史，以获得对公司基本面信息的完整视图，为交易者提供信息。确保包含尽可能多的细节。提供具体、可操作的见解并附上支持证据，以帮助交易者做出明智决策。"
            + " 确保在报告末尾附加一个Markdown表格，以整理报告中的关键点，使其组织有序且易于阅读。"
            + " 使用可用工具：`get_fundamentals`用于全面公司分析，`get_balance_sheet`、`get_cashflow`和`get_income_statement`用于特定财务报表。"
            + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
