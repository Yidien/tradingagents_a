"""用于生成结构化输出的代理的 Pydantic 模式。

框架的主要产物仍然是文本：每个代理的自然语言推理是用户阅读的保存的 markdown 报告，
也是下游代理读取的上下文。结构化输出被分层到三个决策代理（研究经理、交易员、投资组合经理）上，
以便：

- 它们的输出在运行和提供商之间遵循一致的章节标题
- 使用每个提供商的原生结构化输出模式（OpenAI/xAI 的 json_schema，
  Gemini 的 response_schema，Anthropic 的 tool-use）
- 模式字段描述成为模型的输出指令，使提示主体能够专注于上下文和评级标准指导
- 渲染助手将解析后的 Pydantic 实例转换回系统其他部分已经使用的相同 markdown 格式，
  因此显示、记忆日志和保存的报告继续正常工作
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """研究经理和投资组合经理使用的 5 级评级。"""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """交易员使用的 3 级交易方向。

    交易员的工作是将研究经理的投资计划转化为具体的交易提案：
    交易台应执行买入、卖出还是本轮持有。仓位规模和细微的
    增持/减持决策稍后由投资组合经理处理。
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """研究经理生成的结构化投资计划。

    交接给交易员：推荐确定方向性观点，
    理由说明多空辩论中哪一方占主导，
    战略行动将其转化为交易员可以执行的具体指令。
    """

    recommendation: PortfolioRating = Field(
        description=(
            "投资建议。必须是 Buy / Overweight / Hold / Underweight / Sell 中的一个。"
            "仅当双方证据真正平衡时才使用 Hold；否则应支持论证更强的一方。"
        ),
    )
    rationale: str = Field(
        description=(
            "辩论双方关键点的对话式总结，最后说明哪些论证导致了该建议。"
            "自然地表达，就像对队友说话一样。"
        ),
    )
    strategic_actions: str = Field(
        description=(
            "交易员实施建议的具体步骤，"
            "包括与评级一致的仓位规模指导。"
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """将 ResearchPlan 渲染为 markdown 格式，用于存储和交易员的提示上下文。"""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """交易员生成的结构化交易提案。

    交易员阅读研究经理的投资计划和分析师报告，
    然后将其转化为具体交易：采取什么行动、
    证明其合理性的推理，以及入场、止损和规模的实际水平。
    """

    action: TraderAction = Field(
        description="交易方向。必须是 Buy / Hold / Sell 中的一个。",
    )
    reasoning: str = Field(
        description=(
            "此行动的理由，基于分析师报告和"
            "研究计划。两到四句话。"
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="可选的入场价格目标，以标的货币报价。",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="可选的止损价格，以标的货币报价。",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="可选的规模指导，例如'投资组合的5%'。",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """将 TraderProposal 渲染为 markdown。

    保留尾部的 ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` 行，
    以向后兼容分析师停止信号文本和任何搜索该文本的外部代码。
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """投资组合经理生成的结构化输出。

    模型在主要的 LLM 调用中填充每个字段；不需要单独的提取过程。
    字段描述兼作模型的输出指令，因此提示主体只需传达上下文和评级标准指导。
    """

    rating: PortfolioRating = Field(
        description=(
            "最终仓位评级。必须是 Buy / Overweight / Hold / Underweight / Sell 中的一个，"
            "基于分析师的辩论选择。"
        ),
    )
    executive_summary: str = Field(
        description=(
            "简洁的行动计划，涵盖入场策略、仓位规模、"
            "关键风险水平和时间范围。两到四句话。"
        ),
    )
    investment_thesis: str = Field(
        description=(
            "基于分析师辩论中具体证据的详细推理。"
            "如果提示上下文中引用了先前的经验教训，请纳入其中；"
            "否则仅依赖当前分析。"
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="可选的目标价格，以标的货币报价。",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="可推荐的持有期限，例如'3-6个月'。",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """将 PortfolioDecision 渲染回系统其他部分期望的 markdown 格式。

    记忆日志、CLI 显示和保存的报告文件都读取此 markdown，
    因此渲染的输出保留确切的章节标题（``**Rating**``、
    ``**Executive Summary**``、``**Investment Thesis**``），
    下游解析器和报告编写器已经处理这些标题。
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)
