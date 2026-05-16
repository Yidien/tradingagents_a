"""Portfolio Manager: 将风险分析师辩论综合为最终决策。

使用 LangChain 的 ``with_structured_output``，使 LLM 直接生成类型化的
``PortfolioDecision``，只需一次调用。结果被渲染回 markdown 格式存储在
``final_trade_decision`` 中，以便记忆日志、CLI 显示和保存的报告继续使用
与今天相同的格式。当提供商不提供结构化输出时，代理会优雅地回退到自由文本生成。
"""

from __future__ import annotations

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""作为投资组合经理，综合风险分析师的辩论并做出最终交易决策。

{instrument_context}

---

**评级标准** (仅使用一个)：
- **Buy**: 强烈确信进入或增加仓位
- **Overweight**: 前景良好，逐步增加敞口
- **Hold**: 维持当前仓位，无需操作
- **Underweight**: 减少敞口，部分获利了结
- **Sell**: 退出仓位或避免进入

**上下文:**
- 研究经理的投资计划：**{research_plan}**
- 交易员的交易提案：**{trader_plan}**
{lessons_line}
**风险分析师辩论历史:**
{history}

---

要果断，并将每个结论都基于分析师的具体证据。{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_pm_decision,
            "Portfolio Manager",
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
