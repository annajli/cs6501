"""
LangGraph ReAct Agent with Persistent Multi-Turn Conversation

This program demonstrates a LangGraph application using create_react_agent with:
- A single persistent conversation across multiple turns
- Graph-based looping (no Python loops or checkpointing)
- Automatic conversation history management (trimming after 100 messages)
- Verbose debugging output
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import time
from typing import TypedDict, Annotated, Sequence, Literal
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
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
def get_weather(location: str) -> str:
    """
    Get current weather information for a specified location.
    
    Args:
        location: City name or location string
        
    Returns:
        Weather description string
    """
    # Simulate API call delay
    time.sleep(0.5)
    return f"Weather in {location}: Sunny, 72°F with light winds"


@tool
def get_population(city: str) -> str:
    """
    Get population information for a specified city.
    
    Args:
        city: City name
        
    Returns:
        Population information string
    """
    # Simulate API call delay
    time.sleep(0.5)
    return f"Population of {city}: Approximately 1 million people"


@tool
def calculate(expression: str) -> str:
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


def call_react_agent(state: ConversationState) -> ConversationState:
    """
    Invoke the ReAct agent with the current conversation history.
    
    This node:
    - Takes the full conversation history from state
    - Invokes the ReAct agent (which handles tool calling internally)
    - Returns only the NEW messages generated by the agent
    
    Args:
        state: Current conversation state
        
    Returns:
        Updated state with agent's response messages
    """
    if state.get("verbose", True):
        print("\n" + "="*80)
        print("NODE: call_react_agent")
        print("="*80)
        print(f"[DEBUG] Invoking ReAct agent with {len(state['messages'])} messages in history")
    
    # Get the global react_agent
    global react_agent
    
    # Count messages before agent call
    messages_before = len(state["messages"])
    
    # Invoke the ReAct agent with full conversation history
    # The agent maintains context across all previous turns
    result = react_agent.invoke({"messages": state["messages"]})
    
    if state.get("verbose", True):
        messages_after = len(result["messages"])
        new_message_count = messages_after - messages_before
        print(f"[DEBUG] Agent generated {new_message_count} new messages")
        
        # Show what the agent did
        for msg in result["messages"][messages_before:]:
            if isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"[DEBUG] Tool calls: {[tc['name'] for tc in msg.tool_calls]}")
                elif msg.content:
                    print(f"[DEBUG] Response preview: {msg.content[:100]}...")
    
    # Return only the NEW messages (everything after what we sent)
    new_messages = result["messages"][messages_before:]
    return {"messages": new_messages}


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

def route_after_input(state: ConversationState) -> Literal["call_react_agent", "end", "input"]:
    """
    Determine where to route after input based on command field.
    
    Logic:
    - If command is "exit", route to END
    - If command is "verbose" or "quiet", route back to input
    - Otherwise (command is None), route to the ReAct agent
    
    Args:
        state: Current conversation state
        
    Returns:
        "end" to terminate, "input" for verbose toggle, "call_react_agent" to continue
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
    
    # Normal message - route to agent
    if state.get("verbose", True):
        print("[DEBUG] Routing to call_react_agent")
    return "call_react_agent"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Global variable to hold the ReAct agent
react_agent = None

def create_conversation_graph():
    """
    Build the conversation graph with persistent multi-turn capability.
    
    Graph structure (single conversation with looping):
    
        â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
        â”‚                                                      â”‚
        â–¼                                                      â”‚
      input_node â”€â”€(check command)â”€â”€> call_react_agent        â”‚
          â–²                              â”‚                     â”‚
          â”‚                              â–¼                     â”‚
          â”‚                         output_node                â”‚
          â”‚                              â”‚                     â”‚
          â”‚                              â–¼                     â”‚
          â””â”€â”€â”€(verbose/quiet)       trim_history â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
          
          â””â”€â”€â”€â”€â”€(exit)â”€â”€> END
                                         
    Key features:
    - Single conversation maintained in state.messages
    - Command field used for special commands (no sentinel messages!)
    - Graph loops back to input_node after each turn
    - Verbose/quiet commands route directly back to input
    - History automatically trimmed when it grows too long
    - No Python loops or checkpointing needed
    
    Returns:
        Compiled LangGraph application
    """
    global react_agent
    
    # ========================================================================
    # Create the ReAct Agent
    # ========================================================================
    
    model = ChatOpenAI(
        model="gpt-4o",
        temperature=0.7
    )
    
    # System message to encourage tool usage
    system_message = (
        "You are a helpful assistant. "
        "If a tool is able to solve a problem you are working on then "
        "always use it, even if you are able to solve it without using a tool."
    )
    
    # Create the ReAct agent using the built-in function
    # This agent handles the thought/action/observation loop internally
    react_agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=system_message
    )
    
    print("[SYSTEM] ReAct agent created successfully")
    
    # ========================================================================
    # Create the Conversation Wrapper Graph
    # ========================================================================
    
    workflow = StateGraph(ConversationState)
    
    # Add all nodes
    workflow.add_node("input", input_node)
    workflow.add_node("call_react_agent", call_react_agent)
    workflow.add_node("output", output_node)
    workflow.add_node("trim_history", trim_history)
    
    # Set entry point - conversation always starts at input
    workflow.set_entry_point("input")
    
    # Add conditional edge from input based on command field
    workflow.add_conditional_edges(
        "input",
        route_after_input,
        {
            "call_react_agent": "call_react_agent",
            "input": "input",  # Loop back for verbose/quiet
            "end": END
        }
    )
    
    # Add linear edges for the main conversation flow
    # Agent -> Output -> Trim -> Input (loops back!)
    workflow.add_edge("call_react_agent", "output")
    workflow.add_edge("output", "trim_history")
    workflow.add_edge("trim_history", "input")  # This creates the loop!
    
    # Compile the graph
    return workflow.compile()


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_graphs(wrapper_app):
    """
    Generate Mermaid diagrams for both graphs.
    
    Creates:
    - langchain_react_agent.png: Internal ReAct agent (thought/action/observation)
    - langchain_conversation_graph.png: Conversation loop wrapper
    
    Args:
        wrapper_app: Compiled conversation graph
    """
    global react_agent
    
    # Visualize the ReAct agent
    try:
        react_png = react_agent.get_graph().draw_mermaid_png()
        with open("langchain_react_agent.png", "wb") as f:
            f.write(react_png)
        print("[SYSTEM] ReAct agent graph saved to 'langchain_react_agent.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate ReAct agent visualization: {e}")
    
    # Visualize the conversation wrapper
    try:
        wrapper_png = wrapper_app.get_graph().draw_mermaid_png()
        with open("langchain_conversation_graph.png", "wb") as f:
            f.write(wrapper_png)
        print("[SYSTEM] Conversation graph saved to 'langchain_conversation_graph.png'")
    except Exception as e:
        print(f"[WARNING] Could not generate conversation graph visualization: {e}")


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
    print("LangGraph ReAct Agent - Persistent Multi-Turn Conversation")
    print("="*80)
    print("\nThis system uses create_react_agent with graph-based looping:")
    print("  - Single persistent conversation across all turns")
    print("  - History managed automatically (trimmed after 100 messages)")
    print("  - Loops via graph edges (no Python loops or checkpointing)")
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
    
    # Visualize both graphs
    visualize_graphs(app)
    
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
        # Each iteration: input -> agent -> output -> trim -> input (loop!)
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