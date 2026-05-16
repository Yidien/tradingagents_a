from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

# DEFAULT_CONFIG 已经应用了 TRADINGAGENTS_* 环境变量覆盖
# (llm_provider, deep_think_llm, quick_think_llm, backend_url 等)，
# 因此用户可以通过 .env 文件切换模型或端点，而无需
# 编辑此脚本。仅当你想要忽略环境的硬编码值时，
# 才在此处覆盖单个键。
config = DEFAULT_CONFIG.copy()

# 使用自定义配置初始化
ta = TradingAgentsGraph(debug=True, config=config)

# 前向传播
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# 记忆错误并反思
# ta.reflect_and_remember(1000) # 参数是仓位回报
