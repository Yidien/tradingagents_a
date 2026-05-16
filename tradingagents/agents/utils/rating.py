"""共享的 5 级评级词汇表和确定性启发式解析器。

相同的五级标准（Buy, Overweight, Hold, Underweight, Sell）被以下使用：
- 研究经理（投资计划建议）
- 投资组合经理（最终仓位决策）
- 信号处理器（为下游消费者提取评级）
- 记忆日志（与每个决策条目一起存储的评级标签）

在此集中管理可避免这些调用点之间的漂移。
"""

from __future__ import annotations

import re
from typing import Tuple


# 规范的、有序的 5 级标准（从最看涨到最看跌）。
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)

_RATING_SET = {r.lower() for r in RATINGS_5_TIER}

# 匹配 "Rating: X" / "rating - X" / "Rating: **X**" — 容忍 markdown
# 粗体包装和冒号或连字符分隔符。
_RATING_LABEL_RE = re.compile(r"rating.*?[:\-][\s*]*(\w+)", re.IGNORECASE)


def parse_rating(text: str, default: str = "Hold") -> str:
    """从文本中启发式提取 5 级评级。

    两阶段策略：
    1. 查找显式的 "Rating: X" 标签（容忍 markdown 粗体）。
    2. 回退到文本中任意位置找到的第一个 5 级评级词。

    返回首字母大写的评级字符串，如果未找到评级词则返回 ``default``。
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        if m and m.group(1).lower() in _RATING_SET:
            return m.group(1).capitalize()

    for line in text.splitlines():
        for word in line.lower().split():
            clean = word.strip("*:.,")
            if clean in _RATING_SET:
                return clean.capitalize()

    return default
