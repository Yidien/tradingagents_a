from tradingagents.agents.utils.agent_utils import get_language_instruction


def create_conservative_debator(llm):
    def conservative_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        conservative_history = risk_debate_state.get("conservative_history", "")

        current_aggressive_response = risk_debate_state.get("current_aggressive_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        trader_decision = state["trader_investment_plan"]

        prompt = f"""作为保守风险分析师，你的主要目标是保护资产、最小化波动性并确保稳定可靠的成长。你优先考虑稳定性、安全性和风险缓解，仔细评估潜在损失、经济衰退和市场波动。在评估交易员的决策或计划时，批判性地检查高风险元素，指出决策可能在哪些方面使公司面临不必要的风险，以及更谨慎的替代方案如何能够确保长期收益。以下是交易员的决策：

{trader_decision}

你的任务是积极反驳激进派和中性分析师的论点，强调他们的观点可能在哪些方面忽略了潜在威胁或未能优先考虑可持续性。直接回应他们的观点，利用以下数据来源为交易员的决策构建一个有说服力的低风险调整方案：

市场研究报告：{market_research_report}
社交媒体情绪报告：{sentiment_report}
最新世界事务报告：{news_report}
公司基本面报告：{fundamentals_report}
这是当前的对话历史：{history} 这是激进派分析师的最新回应：{current_aggressive_response} 这是中性分析师的最新回应：{current_neutral_response}。如果其他观点尚未回应，请根据可用数据提出你自己的论点。

通过质疑他们的乐观态度并强调他们可能忽略的潜在缺点来参与。解决他们的每个对立观点，展示为什么保守立场最终是公司资产最安全的路径。专注于辩论和批评他们的论点，以展示低风险策略相对于他们方法的优势。以对话方式输出，就像你在说话一样，不要使用特殊格式。""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Conservative Analyst: {response.content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument,
            "aggressive_history": risk_debate_state.get("aggressive_history", ""),
            "conservative_history": conservative_history + "\n" + argument,
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Conservative",
            "current_aggressive_response": risk_debate_state.get(
                "current_aggressive_response", ""
            ),
            "current_conservative_response": argument,
            "current_neutral_response": risk_debate_state.get(
                "current_neutral_response", ""
            ),
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return conservative_node
