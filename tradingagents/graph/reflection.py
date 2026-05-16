# TradingAgents/graph/reflection.py

from typing import Any


class Reflector:
    """处理交易决策的反思。"""

    def __init__(self, quick_thinking_llm: Any):
        """使用 LLM 初始化反思器。"""
        self.quick_thinking_llm = quick_thinking_llm
        self.log_reflection_prompt = self._get_log_reflection_prompt()

    def _get_log_reflection_prompt(self) -> str:
        """用于 reflect_on_final_decision 的简洁提示（阶段 B 日志条目）。

        生成 2-4 句纯文本 — 足够紧凑，可以重新注入到未来代理提示中，
        而不会使上下文窗口膨胀。
        """
        return (
            "你是一名交易分析师，现在已知结果，正在回顾自己过去的决策。\n"
            "请准确写出 2-4 句纯文本（不要项目符号、标题或 markdown）。\n\n"
            "按顺序涵盖：\n"
            "1. 方向性判断是否正确？（引用 alpha 数据）\n"
            "2. 投资论点的哪部分成立或失败？\n"
            "3. 一个适用于下一次类似分析的具体经验教训。\n\n"
            "要具体且简洁。你的输出将逐字存储在决策日志中，"
            "并被未来的分析师重新阅读，因此每个词都必须有其价值。"
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
    ) -> str:
        """在结果上下文下对最终交易决策进行单次反思调用。

        用于阶段 B 的延迟反思。final_trade_decision 已经综合了所有分析师的见解，
        因此不需要单独的市场上下文。
        ``benchmark_name`` 是 alpha 线的标签（例如美国股票代码用 ``"SPY"``，
        ``.T`` 上市股票用 ``"^N225"``）；对于尚未更新以传递基准的调用者，默认为 SPY。
        """
        messages = [
            ("system", self.log_reflection_prompt),
                (
                    "human",
                    (
                        f"原始回报：{raw_return:+.1%}\n"
                        f"Alpha 相对于 {benchmark_name}：{alpha_return:+.1%}\n\n"
                        f"最终决策：\n{final_decision}"
                    ),
                ),
        ]
        return self.quick_thinking_llm.invoke(messages).content
