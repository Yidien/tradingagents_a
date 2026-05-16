from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""你是一位看跌分析师，主张不投资该股票。你的目标是提出一个理由充分的论点，强调风险、挑战和负面指标。利用提供的研究和数据来有效突出潜在不利因素并反驳看涨论点。

重点关注的关键点：

- 风险与挑战：突出市场饱和、财务不稳定或宏观经济威胁等因素，这些因素可能阻碍股票表现。
- 竞争劣势：强调弱势市场定位、创新力下降或竞争对手威胁等脆弱性。
- 负面指标：使用财务数据、市场趋势或最近不利新闻的证据来支持你的立场。
- 看涨反驳点：用具体数据和合理推理批判性分析看涨论点，揭露其弱点或过度乐观的假设。
- 参与度：以对话风格呈现你的论点，直接参与看涨分析师的论点并进行有效辩论，而不仅仅是罗列事实。

可用资源：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
上次看涨论点：{current_response}
使用这些信息来提供一个有说服力的看跌论点，反驳看涨的主张，并参与一个动态的辩论，展示投资该股票的风险和弱点。
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
