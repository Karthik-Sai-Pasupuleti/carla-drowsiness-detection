from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, Annotated, List
import operator
from tools import voice_alert, vibrate_steering_wheel
from pprint import pprint


# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


# Bind tools to the model with descriptions
model = ChatOllama(model="llama3.1:8b", temperature=0.0)
model_with_tools = model.bind_tools([vibrate_steering_wheel, voice_alert])

# Create the tool node
tool_node = ToolNode([vibrate_steering_wheel, voice_alert])


def call_model(state: AgentState):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        print("Continuing to tools...")
        return "tools"
    return "end"


# Define the graph
builder = StateGraph(AgentState)
builder.add_node("call_model", call_model)
builder.add_node("tools", tool_node)

builder.add_edge(START, "call_model")

builder.add_conditional_edges(
    "call_model", should_continue, {"tools": "tools", "end": END}
)
builder.add_edge("tools", "call_model")
# builder.add_edge("tools", END)

# Compile the graph
graph = builder.compile()

if __name__ == "__main__":
    initial_message = HumanMessage(
        content=(
            "The driver is drowsy. Use the available tools "
            "(voice_alert and vibrate_steering_wheel) to alert them. "
            "Do not just give advice in text."
        )
    )

    inputs = {"messages": [initial_message]}

    # Run the graph and print the final state
    final_state = graph.invoke(inputs)
    pprint(final_state)
