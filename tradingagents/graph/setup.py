# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """处理代理图的设置和配置。"""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
        skip_researcher_debate: bool = False,
        skip_trader: bool = False,
        skip_risk_mgmt: bool = False,
        skip_portfolio_manager: bool = False,
    ):
        """使用必需组件进行初始化。"""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        self.skip_researcher_debate = skip_researcher_debate
        self.skip_trader = skip_trader
        self.skip_risk_mgmt = skip_risk_mgmt
        self.skip_portfolio_manager = skip_portfolio_manager

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """设置并编译代理工作流图。

        参数：
            selected_analysts (list): 要包含的分析师类型列表。选项包括：
                - "market": 市场分析师
                - "social": 社交媒体分析师
                - "news": 新闻分析师
                - "fundamentals": 基本面分析师
        """
        if len(selected_analysts) == 0:
            raise ValueError("交易代理图设置错误：未选择分析师！")

        # 创建分析师节点
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            # 保留 "social" 选择器键以向后兼容现有用户配置；
            # 底层代理已重命名为 sentiment_analyst（旧名称宣传了社交媒体数据，
            # 但代理从未访问过 — 参见问题 #557）。
            analyst_nodes["social"] = create_sentiment_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        # 创建研究员和经理节点
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # 创建风险分析节点
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # 创建工作流
        workflow = StateGraph(AgentState)

        # 将分析师节点添加到图中
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # 添加其他节点（条件添加）
        if not self.skip_researcher_debate:
            workflow.add_node("Bull Researcher", bull_researcher_node)
            workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        if not self.skip_trader:
            workflow.add_node("Trader", trader_node)
        if not self.skip_risk_mgmt:
            workflow.add_node("Aggressive Analyst", aggressive_analyst)
            workflow.add_node("Neutral Analyst", neutral_analyst)
            workflow.add_node("Conservative Analyst", conservative_analyst)
        if not self.skip_portfolio_manager:
            workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # 定义边
        # 从第一个分析师开始
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # 按顺序连接分析师
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # 为当前分析师添加条件边
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # 连接到下一个节点
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                if self.skip_researcher_debate:
                    # Skip debate: analysts → Research Manager directly
                    workflow.add_edge(current_clear, "Research Manager")
                else:
                    workflow.add_edge(current_clear, "Bull Researcher")

        # 添加辩论边（仅在未跳过时）
        if not self.skip_researcher_debate:
            workflow.add_conditional_edges(
                "Bull Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bear Researcher": "Bear Researcher",
                    "Research Manager": "Research Manager",
                },
            )
            workflow.add_conditional_edges(
                "Bear Researcher",
                self.conditional_logic.should_continue_debate,
                {
                    "Bull Researcher": "Bull Researcher",
                    "Research Manager": "Research Manager",
                },
            )

        # --- Post-Research-Manager routing ---
        if self.skip_trader and self.skip_risk_mgmt:
            # Everything after Research Manager is skipped → analyst summary only
            workflow.add_edge("Research Manager", END)
        elif self.skip_trader:
            # Skip trader, go directly to risk management
            workflow.add_edge("Research Manager", "Aggressive Analyst")
        else:
            workflow.add_edge("Research Manager", "Trader")

        # --- Post-Trader routing ---
        if not self.skip_trader:
            if self.skip_risk_mgmt:
                workflow.add_edge("Trader", END)
            else:
                workflow.add_edge("Trader", "Aggressive Analyst")

        # --- Risk management debate (仅在未跳过时) ---
        if not self.skip_risk_mgmt:
            # When PM is skipped, risk debaters route directly to END
            risk_next = "Portfolio Manager" if not self.skip_portfolio_manager else END
            workflow.add_conditional_edges(
                "Aggressive Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Conservative Analyst": "Conservative Analyst",
                    risk_next: risk_next,
                },
            )
            workflow.add_conditional_edges(
                "Conservative Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Neutral Analyst": "Neutral Analyst",
                    risk_next: risk_next,
                },
            )
            workflow.add_conditional_edges(
                "Neutral Analyst",
                self.conditional_logic.should_continue_risk_analysis,
                {
                    "Aggressive Analyst": "Aggressive Analyst",
                    risk_next: risk_next,
                },
            )

        # --- Portfolio Manager → END ---
        if not self.skip_portfolio_manager:
            workflow.add_edge("Portfolio Manager", END)

        return workflow
