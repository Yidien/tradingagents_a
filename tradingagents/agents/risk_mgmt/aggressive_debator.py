from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_aggressive_debator(llm):
    def aggressive_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        aggressive_history = risk_debate_state.get("aggressive_history", "")

        current_conservative_response = risk_debate_state.get("current_conservative_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为激进风险分析师，你的角色是积极倡导高回报、高风险的机会，强调大胆策略和竞争优势。在评估交易员的决策或计划时，要密切关注潜在的上行空间、增长潜力和创新收益——即使这些伴随着更高的风险。利用提供的市场数据和情绪分析来加强你的论点并挑战对立观点。具体来说，直接回应保守派和中性分析师的每个观点，用数据驱动的反驳和有说服力的推理进行反击。强调他们的谨慎可能会错过关键机会，或者他们的假设可能过于保守。以下是交易员的决策：

{trader_decision}

你的任务是通过质疑和批评保守派和中性立场来为交易员的决策创建一个有说服力的案例，展示为什么你的高回报视角提供了最佳前进路径。将以下来源的见解融入你的论点中：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务报告：{news_report}
公司基本面报告：{fundamentals_report}
这是当前的对话历史：{history} 这是保守派分析师的最新论点：{current_conservative_response} 这是中性分析师的最新论点：{current_neutral_response}。如果其他观点尚未回应，请根据可用数据提出你自己的论点。

积极参与，解决任何具体提出的担忧，反驳他们逻辑中的弱点，并主张承担风险以超越市场常规的好处。保持辩论和说服的焦点，而不仅仅是呈现数据。挑战每个对立观点，以强调为什么高风险方法是最优的。以对话方式输出，就像你在说话一样，不要使用特殊格式。""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Aggressive Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": aggressive_history + "\n" + argument,
            "conservative_history": risk_debate_state.get("conservative_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Aggressive",
            "current_aggressive_response": argument,
            "current_conservative_response": risk_debate_state.get("current_conservative_response", ""),
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return aggressive_node
