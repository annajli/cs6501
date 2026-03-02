"""
LangGraph Multi-Agent System with Manual ToolNode Implementation

This program demonstrates a LangGraph application with manual tool calling:
- A single persistent conversation across multiple turns
- Manual tool calling loop with ToolNode (no create_react_agent)
- Graph-based looping (no Python loops or checkpointing)
- Automatic conversation history management (trimming after 100 messages)
- Verbose debugging output
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# ============================================================================
# STATE DEFINITION
# ============================================================================

class ConversationState(TypedDict):
    """
    State schema for the conversation.
    
    Attributes:
        messages: Full conversation history with automatic message merging
        verbose: Controls detailed tracing output
        command: Special command from user (exit, verbose, quiet, or None)
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    verbose: bool
    command: str  # "exit", "verbose", "quiet", or None


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@tool
async def get_weather(location: str) -> str:
    """
    Get current weather information for a specified location.
    
    Args:
        location: City name or location string
        
    Returns:
        Weather description string
    """
    # Simulate API call delay
    await asyncio.sleep(0.5)
    return f"Weather in {location}: Sunny, 72°F with light winds"


@tool
async def get_population(city: str) -> str:
    """
    Get population information for a specified city.
    
    Args:
        city: City name
        
    Returns:
        Population information string
    """
    # Simulate API call delay
    await asyncio.sleep(0.5)
    return f"Population of {city}: Approximately 1 million people"


@tool
async def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 2")
        
    Returns:
        Result of the calculation
    """
    try:
        # Safe evaluation of simple math expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


# List of all available tools
tools = [get_weather, get_population, calculate]


# ============================================================================
# NODE FUNCTIONS
# ============================================================================

def input_node(state: ConversationState) -> ConversationState:
    """
    Get input from the user and add it to the conversation.
    
    This node:
    - Prompts the user for input
    - Handles special commands (quit, exit, verbose, quiet)
    - Adds user message to conversation history (for real messages only)
    - Sets command field for special commands
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with new user message or command
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: input_node")
        print("="*80)
    
    # Get user input
    user_input = input("\nYou: ").strip()
    
    # Handle exit commands
    if user_input.lower() in ["quit", "exit"]:
        if state.get("verbose", True):
            print("[DEBUG] Exit command received")
        # Set command field, don't add to messages
        return {"command": "exit"}
    
    # Handle verbose toggle
    if user_input.lower() == "verbose":
        print("[SYSTEM] Verbose mode enabled")
        # Set command field and update verbose flag
        return {"command": "verbose", "verbose": True}
    
    if user_input.lower() == "quiet":
        print("[SYSTEM] Verbose mode disabled")
        # Set command field and update verbose flag
        return {"command": "quiet", "verbose": False}
    
    # Add user message to conversation history
    if state.get("verbose", True):
        print(f"[DEBUG] User input: {user_input}")
    
    # Clear command field and add message
    return {"command": None, "messages": [HumanMessage(content=user_input)]}


def call_model(state: ConversationState) -> ConversationState:
    """
    Call the LLM with tools bound.
    
    This node:
    - Prepends system message if not already present
    - Invokes the model with tool bindings
    - Returns the model's response (may include tool_calls)
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with model response
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: call_model")
        print("="*80)
        print(f"[DEBUG] Calling model with {len(state['messages'])} messages")
    
    messages = list(state["messages"])
    
    # Add system message if not present
    # Check if first message is a SystemMessage
    system_added = False
    if not messages or not isinstance(messages[0], SystemMessage):
        system_prompt = SystemMessage(
            content="You are a helpful assistant. "
                   "If a tool is able to solve a problem you are working on then "
                   "always use it, even if you are able to solve it without using a tool."
        )
        messages = [system_prompt] + messages
        system_added = True
        if state.get("verbose", True):
            print("[DEBUG] Added system prompt")
    
    # Initialize model with tools
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7
    )
    model_with_tools = model.bind_tools(tools)
    
    # Invoke the model
    response = model_with_tools.invoke(messages)
    
    if state.get("verbose", True):
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"[DEBUG] Model requested {len(response.tool_calls)} tool call(s):")
            for tc in response.tool_calls:
                print(f"  - {tc['name']}({tc['args']})")
        else:
            print(f"[DEBUG] Model response (no tools): {response.content[:100]}...")
    
    # Return the system message if we added it, plus the response
    # This ensures the system message is persisted in the conversation state
    if system_added:
        return {"messages": [messages[0], response]}  # system prompt + model response
    else:
        return {"messages": [response]}


def output_node(state: ConversationState) -> ConversationState:
    """
    Display the assistant's final response to the user.
    
    This node:
    - Extracts the last AI message from the conversation
    - Prints it to the console
    - Returns empty dict (no state changes)
    
    Args:
        state: Current conversation state
        
    Returns:
        Empty dict (no state modifications)
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: output_node")
        print("="*80)
    
    # Find the last AI message in the conversation
    # (there may be tool messages mixed in)
    last_ai_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            last_ai_message = msg
            break
    
    if last_ai_message:
        print(f"\nAssistant: {last_ai_message.content}")
    else:
        print("\n[WARNING] No assistant response found")
    
    return {}


def trim_history(state: ConversationState) -> ConversationState:
    """
    Manage conversation history length to prevent unlimited growth.
    
    Strategy:
    - Keep the system message (if present)
    - Keep the most recent 100 messages
    - This allows ~50 conversation turns (user + assistant pairs)
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with trimmed message history (if needed)
    """
    messages = state["messages"]
    max_messages = 100
    
    # Only trim if we've exceeded the limit
    if len(messages) > max_messages:
        if state.get("verbose", True):
            print(f"\n[DEBUG] History length: {len(messages)} messages")
            print(f"[DEBUG] Trimming to most recent {max_messages} messages")
        
        # Preserve system message if it exists at the start
        if messages and isinstance(messages[0], SystemMessage):
            # Keep system message + last (max_messages - 1) messages
            trimmed = [messages[0]] + list(messages[-(max_messages - 1):])
            if state.get("verbose", True):
                print(f"[DEBUG] Preserved system message + {max_messages - 1} recent messages")
        else:
            # Just keep the last max_messages
            trimmed = list(messages[-max_messages:])
            if state.get("verbose", True):
                print(f"[DEBUG] Kept {max_messages} most recent messages")
        
        return {"messages": trimmed}
    
    # No trimming needed
    return {}


# ============================================================================
# ROUTING LOGIC
# ============================================================================

def route_after_input(state: ConversationState) -> Literal["call_model", "end", "input"]:
    """
    Determine where to route after input based on command field.
    
    Logic:
    - If command is "exit", route to END
    - If command is "verbose" or "quiet", route back to input
    - Otherwise (command is None), route to call_model
    
    Args:
        state: Current conversation state
        
    Returns:
        "end" to terminate, "input" for verbose toggle, "call_model" to continue
    """
    command = state.get("command")
    
    # Check for exit command
    if command == "exit":
        if state.get("verbose", True):
            print("[DEBUG] Routing to END (exit requested)")
        return "end"
    
    # Check for verbose toggle commands - route back to input
    if command in ["verbose", "quiet"]:
        if state.get("verbose", True):
            print("[DEBUG] Routing back to input (verbose toggle)")
        return "input"
    
    # Normal message - route to model
    if state.get("verbose", True):
        print("[DEBUG] Routing to call_model")
    return "call_model"


def route_after_model(state: ConversationState) -> Literal["tools", "output"]:
    """
    Route after model call based on whether tools were requested.
    
    Logic:
    - If the model's response includes tool_calls, route to tools
    - Otherwise, route to output to display the response
    
    Args:
        state: Current conversation state
        
    Returns:
        "tools" if tools requested, "output" otherwise
    """
    last_message = state["messages"][-1]
    
    # Check if the last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if state.get("verbose", True):
            print("[DEBUG] Routing to tools")
        return "tools"
    
    if state.get("verbose", True):
        print("[DEBUG] Routing to output")
    return "output"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_conversation_graph():
    """
    Build the conversation graph with manual tool calling using ToolNode.
    
    Graph structure (single conversation with looping):
    
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        ▼                                                      │
      input_node ──(check command)──> call_model              │
          ▲                              │                     │
          │                              ├──(has tools)──> tools
          │                              │                    │
          │                              │                    │
          │                              └──(no tools)──> output_node
          │                                                    │
          │                                                    ▼
          └───(verbose/quiet)                           trim_history ──┘
          
          └─────(exit)──> END
                                         
    Key features:
    - Manual tool calling with ToolNode (no create_react_agent)
    - Command field used for special commands (no sentinel messages!)
    - Single conversation maintained in state.messages
    - Graph loops back to input_node after each turn
    - Tools route back to call_model for continued reasoning
    - History automatically trimmed when it grows too long
    
    Returns:
        Compiled LangGraph application
    """
    
    # ========================================================================
    # Create the Conversation Graph
    # ========================================================================
    
    workflow = StateGraph(ConversationState)
    
    # Create ToolNode to handle tool execution
    # ToolNode automatically executes tools in parallel when possible
    tool_node = ToolNode(tools)
    
    # Add all nodes
    workflow.add_node("input", input_node)
    workflow.add_node("call_model", call_model)
    workflow.add_node("tools", tool_node)  # ToolNode handles tool execution
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)
    
    # Set entry point - conversation always starts at input
    workflow.set_entry_point("input")
    
    # Add conditional edge from input based on command field
    # Check if user wants to exit, toggle verbose, or continue
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_model": "call_model",
            "input": "input",  # Loop back for verbose/quiet
            "end": END
        }
    )
    
    # Add conditional edge from call_model
    # Check if tools were requested or if we have final response
    workflow.add_conditional_edges(
        "call_model",
        route_after_model,
        {
            "tools": "tools",
            "output": "output"
        }
    )
    
    # After tools execute, loop back to call_model
    # This allows the model to see tool results and decide next action
    workflow.add_edge("tools", "call_model")
    
    # After output, trim history and loop back to input
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")  # This creates the conversation loop!
    
    # Compile the graph
    print("[SYSTEM] Conversation graph created successfully (using manual ToolNode)")
    return workflow.compile()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graph(app):
    """
    Generate Mermaid diagram of the conversation graph.
    
    Creates:
    - langchain_manual_tool_graph.png: The conversation loop with manual tool calling
    
    Args:
        app: Compiled conversation graph
    """
    try:
        graph_png = app.get_graph().draw_mermaid_png()
        with open("langchain_manual_tool_graph.png", "wb") as f:
            f.write(graph_png)
        print("[SYSTEM] Graph visualization saved to 'langchain_manual_tool_graph.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate graph visualization: {e}")
        print("You may need to install: pip install pygraphviz or pip install grandalf")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """
    Main execution function.
    
    This function:
    1. Creates the conversation graph
    2. Visualizes the graph structure
    3. Initializes the conversation state
    4. Invokes the graph ONCE
    
    The graph then runs indefinitely via internal looping (trim_history -> input)
    until the user types 'quit' or 'exit'.
    """
    print("="*80)
    print("LangGraph Manual Tool Calling - Persistent Multi-Turn Conversation")
    print("="*80)
    print("\nThis system uses manual tool calling with ToolNode:")
    print("  - call_model: Invokes LLM with tool bindings")
    print("  - ToolNode: Executes requested tools in parallel")
    print("  - Loop: tools -> call_model (until no more tools needed)")
    print("  - Single persistent conversation across all turns")
    print("  - History managed automatically (trimmed after 100 messages)")
    print("\nCommands:")
    print("  - Type 'quit' or 'exit' to end the conversation")
    print("  - Type 'verbose' to enable detailed tracing")
    print("  - Type 'quiet' to disable detailed tracing")
    print("\nAvailable tools:")
    print("  - get_weather(location): Get weather information")
    print("  - get_population(city): Get population data")
    print("  - calculate(expression): Evaluate math expressions")
    print("="*80)
    
    # Create the conversation graph
    app = create_conversation_graph()
    
    # Visualize the graph
    visualize_graph(app)
    
    # Initialize conversation state
    # This state persists across all turns via graph looping
    initial_state = {
        "messages": [],
        "verbose": True,
        "command": None
    }
    
    print("\n[SYSTEM] Starting conversation...\n")
    
    try:
        # Invoke the graph ONCE
        # The graph will loop internally until user exits
        # Each iteration: input -> call_model -> [tools -> call_model]* -> output -> trim -> input
        # Verbose commands: input -> input (direct loop!)
        await app.ainvoke(initial_state)
        
    except KeyboardInterrupt:
        print("\n\n[SYSTEM] Interrupted by user (Ctrl+C)")
    
    print("\n[SYSTEM] Conversation ended. Goodbye!\n")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # asyncio.run() executes main() exactly ONCE
    # The looping happens INSIDE the graph via edges
    asyncio.run(main())