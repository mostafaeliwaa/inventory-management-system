from langgraph.graph import StateGraph, END, START
from langgraph.graph import MessagesState
from .nodes import State, GraphNodes

class GraphBuilder:
    
    def __init__(self, nodes: GraphNodes):
        self.nodes = nodes
        self.graph = StateGraph(State)
    
    def build(self) -> StateGraph:
        
        self._add_nodes()
        self._add_edges()
        return self.graph.compile()
    
    def _add_nodes(self) -> None:
        
        self.graph.add_node("llm", self.nodes.process_llm_node)
        self.graph.add_node("tool", self.nodes.process_tool_node)
        self.graph.add_node("format", self.nodes.process_format_node)
        self.graph.add_node("general", self.nodes.process_general_node)
    
    def _add_edges(self) -> None:
        
        self.graph.set_entry_point("llm")
        
        self.graph.add_conditional_edges(
            "llm",
            lambda state: state["type"],
            {
                "tool": "tool",
                "general": "general"
            }
        )
        
        self.graph.add_edge("tool", "format")
        self.graph.add_edge("format", END)
        self.graph.add_edge("general", END)