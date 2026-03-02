"""
LangGraph Persistent Conversation with Checkpointing and Recovery

Rewrites the langchain-tool-handling-with-multiple-tools.py run_agent() loop as
a LangGraph graph so that the entire session is one continuous conversation:

  - Messages accumulate in state across all turns (no fresh start per call)
  - Nodes handle input, model call, tool execution, output, and history trim
  - MemorySaver checkpoints state after every completed node
  - SqliteSaver (commented) enables cross-process recovery

Graph structure:
  input ──(exit)──> END
    │  └──(history)──> input (self-loop)
    └──(normal)──> call_model
                        │
              ┌─(tool calls?)─┐
              ▼               ▼
            tools           output
              │               │
              └──> call_model trim_history ──> input
"""

import json
import math
from typing import Annotated, Sequence, Literal, TypedDict

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# For persistent cross-session recovery, replace MemorySaver with SqliteSaver:
# pip install langgraph-checkpoint-sqlite
# from langgraph.checkpoint.sqlite import SqliteSaver


# ============================================================================
# STATE
# ============================================================================

class ConversationState(TypedDict):
    """
    Persistent conversation state.

    messages: Accumulated conversation history. add_messages merges new
              messages into the list rather than replacing it, so every
              node only needs to return the *new* messages it produces.
    command:  Signal from input_node for routing ("exit", "history", or None).
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    command: str | None


# ============================================================================
# TOOLS  (same set as langchain-tool-handling-with-multiple-tools.py)
# ============================================================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F",
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculate(expression: str) -> str:
    """
    Calculate mathematical expressions including geometric and trigonometric functions.

    Supports: +, -, *, /, ** (power), sin, cos, tan, asin, acos, atan, atan2,
              sinh, cosh, tanh, exp, log, log10, log2, sqrt, abs, ceil, floor,
              round, degrees, radians, pi, e

    Note: Trigonometric functions expect angles in radians.

    Args:
        expression: Mathematical expression string to evaluate.
    """
    try:
        safe_dict = {
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "atan2": math.atan2, "sinh": math.sinh, "cosh": math.cosh,
            "tanh": math.tanh, "exp": math.exp, "log": math.log,
            "log10": math.log10, "log2": math.log2, "sqrt": math.sqrt,
            "pow": math.pow, "ceil": math.ceil, "floor": math.floor,
            "round": round, "abs": abs, "degrees": math.degrees,
            "radians": math.radians, "pi": math.pi, "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        return json.dumps({"expression": expression, "result": result, "success": True})
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "Division by zero", "success": False})
    except Exception as ex:
        return json.dumps({"expression": expression, "error": str(ex), "success": False})


@tool
def count_letter(text: str, letter: str) -> str:
    """
    Count the number of occurrences of a specific letter in a piece of text.

    Case-insensitive. Letter must be a single character.

    Args:
        text:   The text to search in.
        letter: The single letter to count.
    """
    try:
        if len(letter) != 1:
            return json.dumps({"error": "letter must be a single character", "success": False})
        text_lower = text.lower()
        letter_lower = letter.lower()
        count = text_lower.count(letter_lower)
        positions = [i for i, c in enumerate(text_lower) if c == letter_lower]
        return json.dumps({
            "text": text, "letter": letter,
            "count": count, "positions": positions, "success": True,
        })
    except Exception as ex:
        return json.dumps({"text": text, "letter": letter, "error": str(ex), "success": False})


tools = [get_weather, calculate, count_letter]
tool_map = {t.name: t for t in tools}


# ============================================================================
# LLM
# ============================================================================

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to weather lookup, a math calculator, "
    "and a letter-counting tool. Always use a tool when it can answer the question, "
    "even for simple arithmetic. For trig functions, convert degrees to radians first."
)

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


# ============================================================================
# NODES
# ============================================================================

def input_node(state: ConversationState) -> dict:
    """
    Prompt the user and add their message to the conversation.

    Special commands:
      exit / quit  →  command = "exit"  →  routes to END
      history      →  command = "history"  →  prints history, loops back
      (anything else)  →  appends HumanMessage, command = None
    """
    user_input = input("\nYou: ").strip()

    if user_input.lower() in ("exit", "quit", "q"):
        return {"command": "exit"}

    if user_input.lower() == "history":
        _print_history(state.get("messages", []))
        return {"command": "history"}

    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_model(state: ConversationState) -> dict:
    """
    Call the LLM with the full conversation history.

    Prepends the system prompt if it is not already the first message.
    Returns the model's response (which may contain tool_calls).
    """
    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def output_node(state: ConversationState) -> dict:
    """Print the last AI text response to the user."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"\nAssistant: {msg.content}")
            break
    return {}


def trim_history(state: ConversationState) -> dict:
    """
    Keep the conversation history from growing without bound.

    Preserves the system message and the 49 most recent other messages
    (≈ 24–25 conversation turns).
    """
    messages = list(state["messages"])
    max_messages = 50
    if len(messages) <= max_messages:
        return {}

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    other_msgs  = [m for m in messages if not isinstance(m, SystemMessage)]
    keep = max_messages - len(system_msgs)
    trimmed = system_msgs + other_msgs[-keep:]
    print(f"[History trimmed: {len(messages)} → {len(trimmed)} messages]")
    # Replace all messages with the trimmed list. Because add_messages deduplicates
    # by message id, we cannot simply return a shorter list — we need to signal a
    # full replacement by returning the trimmed list as the new state.
    return {"messages": trimmed}


# ============================================================================
# HELPERS
# ============================================================================

def _print_history(messages: Sequence[BaseMessage]) -> None:
    """Pretty-print the conversation history (skips system and tool messages)."""
    turns = [(m.type, m.content) for m in messages
             if isinstance(m, (HumanMessage, AIMessage)) and m.content]
    if not turns:
        print("[No conversation history yet]")
        return
    print(f"\n--- History ({len(turns)} messages) ---")
    for kind, content in turns:
        label = "You" if kind == "human" else "Assistant"
        snippet = content[:100].replace("\n", " ")
        suffix = "…" if len(content) > 100 else ""
        print(f"  {label}: {snippet}{suffix}")
    print("--- End ---")


# ============================================================================
# ROUTING
# ============================================================================

def route_after_input(state: ConversationState) -> Literal["call_model", "input", "end"]:
    """Route based on the command field set by input_node."""
    cmd = state.get("command")
    if cmd == "exit":
        return "end"
    if cmd == "history":
        return "input"          # self-loop: show history then prompt again
    return "call_model"


def route_after_model(state: ConversationState) -> Literal["tools", "output"]:
    """If the model requested tool calls, execute them; otherwise display the answer."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "output"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph(checkpointer):
    """
    Compile the conversation graph.

    Edges:
      input ──(exit)──────────────────────────────────> END
      input ──(history)───────────────────────────────> input   (self-loop)
      input ──(normal)────────────────────────────────> call_model
      call_model ──(tool calls)───────────────────────> tools
      tools ──────────────────────────────────────────> call_model
      call_model ──(no tool calls)────────────────────> output
      output ─────────────────────────────────────────> trim_history
      trim_history ───────────────────────────────────> input   (conversation loop)

    The checkpointer snapshots state after every completed node, enabling
    recovery: restarting the script with the same thread_id (and a persistent
    checkpointer like SqliteSaver) resumes the conversation from the last
    completed node.
    """
    workflow = StateGraph(ConversationState)

    # ToolNode executes all tools requested in a single model response.
    # For async tools it runs them concurrently via asyncio.gather().
    # These tools are sync, so they run sequentially inside ToolNode.
    tool_node = ToolNode(tools)

    workflow.add_node("input",        input_node)
    workflow.add_node("call_model",   call_model)
    workflow.add_node("tools",        tool_node)
    workflow.add_node("output",       output_node)
    workflow.add_node("trim_history", trim_history)

    workflow.set_entry_point("input")

    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {"call_model": "call_model", "input": "input", "end": END},
    )
    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {"tools": "tools", "output": "output"},
    )
    workflow.add_edge("tools",        "call_model")     # tool result → model
    workflow.add_edge("output",       "trim_history")
    workflow.add_edge("trim_history", "input")          # conversation loop

    return workflow.compile(checkpointer=checkpointer)


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("LangGraph Persistent Conversation  (topic 3, task 5)")
    print("=" * 70)
    print("Tools : get_weather · calculate · count_letter")
    print("Commands: 'history' (show conversation so far), 'exit' / 'quit'")
    print()

    # -----------------------------------------------------------------------
    # Checkpointer
    #
    # MemorySaver  — in-process only; history is lost when the script exits.
    #                Suitable for demonstrating within-session recovery.
    #
    # SqliteSaver  — persists to disk; history survives process restarts.
    #                To enable:
    #                  pip install langgraph-checkpoint-sqlite
    #                  from langgraph.checkpoint.sqlite import SqliteSaver
    #                  checkpointer = SqliteSaver.from_conn_string("conversation.db")
    # -----------------------------------------------------------------------
    checkpointer = MemorySaver()

    app = build_graph(checkpointer)

    # Print the Mermaid source so the diagram can be copied into the README.
    try:
        print(app.get_graph().draw_mermaid())
    except Exception:
        pass

    # Save PNG if graphviz / pillow is available.
    try:
        png = app.get_graph().draw_mermaid_png()
        with open("langgraph_conversation.png", "wb") as f:
            f.write(png)
        print("[Graph saved to langgraph_conversation.png]\n")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Thread ID — identifies this conversation in the checkpointer.
    #
    # Reusing the same thread_id on a subsequent run (with SqliteSaver)
    # causes LangGraph to load the saved state and resume the conversation
    # rather than starting fresh.  With MemorySaver the same effect can
    # be demonstrated by calling app.invoke() a second time below without
    # creating a new checkpointer object.
    # ------------------------------------------------------------------
    thread_id = "topic3-conversation"
    config = {"configurable": {"thread_id": thread_id}}

    # Show whether an existing conversation is being resumed.
    snapshot = checkpointer.get(config)
    if snapshot:
        prior_msgs = snapshot.channel_values.get("messages", [])
        n_turns = sum(1 for m in prior_msgs if isinstance(m, HumanMessage))
        print(f"[Resuming existing conversation — {n_turns} prior user turn(s)]\n")
    else:
        print(f"[Starting new conversation  thread_id={thread_id!r}]\n")

    try:
        app.invoke(
            {"messages": [], "command": None},
            config=config,
        )
    except KeyboardInterrupt:
        print(
            f"\n[Interrupted — state saved under thread_id={thread_id!r}]"
            "\n[Restart the script (with SqliteSaver) to resume this conversation]"
        )

    print("\n[Goodbye!]")


if __name__ == "__main__":
    main()
