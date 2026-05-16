from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_bull_researcher(llm):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        prompt = f"""你是一位看涨分析师，主张投资该股票。你的任务是建立一个强有力的、基于证据的案例，强调增长潜力、竞争优势和积极的市场指标。利用提供的研究和数据来有效解决疑虑并反驳看跌论点。

重点关注的关键点：
- 增长潜力：突出公司的市场机会、收入预测和可扩展性。
- 竞争优势：强调独特产品、强大品牌或主导市场地位等因素。
- 积极指标：使用财务健康状况、行业趋势和最近的积极新闻作为证据。
- 看跌反驳点：用具体数据和合理推理批判性分析看跌论点，彻底解决疑虑，并展示为什么看涨观点更具优势。
- 参与度：以对话风格呈现你的论点，直接参与看跌分析师的论点并进行有效辩论，而不仅仅是罗列数据。

可用资源：
市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务新闻：{news_report}
公司基本面报告：{fundamentals_report}
辩论对话历史：{history}
上次看跌论点：{current_response}
使用这些信息来提供一个有说服力的看涨论点，反驳看跌的担忧，并参与一个动态的辩论，展示看涨立场的优势。
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bull Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bull_history": bull_history + "\n" + argument,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
