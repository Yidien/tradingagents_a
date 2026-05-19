"""Research Manager: turns the bull/bear debate (or analyst reports) into a structured investment plan for the trader."""

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

        # When debate is skipped (empty history), summarise analyst reports instead
        if not history or not history.strip():
            prompt = _build_analyst_summary_prompt(state, instrument_context)
        else:
            prompt = _build_debate_prompt(history, instrument_context)

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        investment_debate_state = state["investment_debate_state"]
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


def _build_debate_prompt(history: str, instrument_context: str) -> str:
    """Normal debate-based prompt."""
    return f"""作为研究经理和辩论主持人，你的角色是批判性评估本轮辩论，并为交易员提供一个清晰、可操作的投资计划。

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


def _build_analyst_summary_prompt(state: dict, instrument_context: str) -> str:
    """Analyst-report-based summary prompt (used when debate is skipped)."""
    reports = []
    for key, label in [
        ("market_report", "市场/技术分析报告"),
        ("sentiment_report", "情绪/社区分析报告"),
        ("news_report", "新闻/宏观分析报告"),
        ("fundamentals_report", "基本面分析报告"),
    ]:
        content = state.get(key, "")
        if content and content.strip():
            reports.append(f"### {label}\n\n{content[:3000]}")

    if not reports:
        return _build_debate_prompt("", instrument_context)

    all_reports = "\n\n---\n\n".join(reports)

    return f"""你是一位资深投资研究经理。由于本轮分析跳过了牛熊辩论环节，你的任务是基于分析师团队的报告直接给出综合投资建议。

{instrument_context}

---

**评级标准**（请准确使用其中一个）：
- **Buy**: 多维度分析均强烈看多，建议买入或加仓
- **Overweight**: 整体偏积极，建议逐步增持
- **Hold**: 多空因素相当或不确定性较大，建议维持现有仓位
- **Underweight**: 整体偏谨慎，建议逐步减仓
- **Sell**: 多维度分析均强烈看空，建议卖出或清仓

---

**分析师团队报告：**

{all_reports}

---

请综合上述所有分析报告，给出你的投资评级、核心理由（列出多空关键因素权衡）、以及具体可操作的战略行动建议。请用中文输出。""" + get_language_instruction()
