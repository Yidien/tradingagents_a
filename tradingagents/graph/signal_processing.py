"""从投资组合经理的决策中提取 5 级投资组合评级。

投资组合经理通过结构化输出生成类型化的 ``PortfolioDecision``，
并将其渲染为始终包含 ``**Rating**: X`` 标题的 markdown
（参见 :func:`tradingagents.agents.schemas.render_pm_decision`）。
:mod:`tradingagents.agents.utils.rating` 中的确定性启发式方法
足以提取该评级；不需要额外的 LLM 调用。

此模块的存在是为了向后兼容期望 ``SignalProcessor.process_signal(text)`` 接口的调用者。
"""

from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.rating import parse_rating


class SignalProcessor:
    """从投资组合经理决策中读取 5 级评级。"""

    def __init__(self, quick_thinking_llm: Any = None):
        # 接受 LLM 参数是为了向后兼容，但不再使用：
        # PM 的结构化输出保证可以从渲染的 markdown 中解析评级，无需第二次 LLM 调用。
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """返回 Buy / Overweight / Hold / Underweight / Sell 中的一个。"""
        return parse_rating(full_signal)
