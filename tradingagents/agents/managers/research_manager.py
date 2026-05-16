"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = build_instrument_context(state["company_of_interest"])
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""作为研究经理和辩论主持人，你的角色是批判性评估本轮辩论，并为交易员提供一个清晰、可操作的投资计划。

{instrument_context}

---

**评级标准**（请准确使用其中一个）：
- **Buy**: 对看涨论点有强烈信心；建议建立或增加仓位
- **Overweight**: 建设性观点；建议逐步增加敞口
- **Hold**: 平衡观点；建议维持当前仓位
- **Underweight**: 谨慎观点；建议减少敞口
- **Sell**: 对看跌论点有强烈信心；建议退出或避免仓位

当辩论中最有力的论点支持时，请明确表态；仅当双方证据真正平衡时才使用Hold评级。

---

**辩论历史：**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
