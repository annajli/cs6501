# task5_chat_history_agent.py
# MODIFICATION to task4_routed_llm_agent.py:
# - Adds chat history using the LangChain Message API
#   State holds a messages list (SystemMessage, HumanMessage, AIMessage)
#   with the add_messages reducer, which appends rather than overwrites.
# - get_user_input appends a HumanMessage each turn.
# - call_llama formats the full message history into a prompt string,
#   invokes the LLM, extracts only the newly generated text, then appends
#   an AIMessage so the history grows across turns.
# - Qwen is disabled; only Llama-3.2-1B-Instruct is used.

import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated


SYSTEM_PROMPT = "You are a helpful assistant."


def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages:      Full conversation history as a list of LangChain message
                     objects (SystemMessage, HumanMessage, AIMessage).
                     Uses the add_messages reducer, which APPENDS new messages
                     rather than replacing the list on each update.
    - user_input:    Raw text from stdin (kept for routing logic).
    - should_exit:   True when the user types quit/exit/q.
    - is_command:    True for control commands (verbose/quiet) or empty input.
    - verbose:       True when per-node trace output is enabled.
    - llm_response:  The latest AI response text (for display in print_response).

    Graph:
        START -> get_user_input -+-> call_llama -> print_response -> get_user_input
                      ^          |                                         |
                      |          +-> get_user_input (is_command or empty) |
                      |          +-> END (should_exit)                    |
                      +----------------------------------------------------+
    """
    messages:     Annotated[list, add_messages]
    user_input:   str
    should_exit:  bool
    is_command:   bool
    verbose:      bool
    llm_response: str


# =============================================================================
# HELPERS
# =============================================================================

def format_prompt(messages: list) -> str:
    """
    Convert a LangChain message list into a single prompt string for the
    HuggingFacePipeline, preserving the full conversation history.

    SystemMessage  -> "System: ..."
    HumanMessage   -> "User: ..."
    AIMessage      -> "Assistant: ..."

    A bare "Assistant:" is appended at the end to prompt the model to continue.
    """
    parts = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            parts.append(f"System: {msg.content}")
        elif isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
    parts.append("Assistant:")
    return "\n".join(parts)


def extract_response(full_output: str, prompt: str) -> str:
    """
    HuggingFacePipeline text-generation returns the full text (prompt +
    generated tokens).  This helper strips the prompt prefix and then cuts
    off any self-generated "User:" continuation to avoid the multi-turn
    hallucination seen in earlier tasks.
    """
    generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
    return generated.split("\nUser:")[0].strip()


# =============================================================================
# LLM FACTORY
# =============================================================================

def create_llm(model_id: str, device: str) -> HuggingFacePipeline:
    """Load a causal LM from HuggingFace and wrap it as a LangChain LLM."""
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )
    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(f"Model loaded: {model_id}")
    return HuggingFacePipeline(pipeline=pipe)


# =============================================================================
# GRAPH
# =============================================================================

def create_graph(llama_llm: HuggingFacePipeline):
    """
    Build and compile the LangGraph state graph.

    Nodes:
      get_user_input -- reads stdin; appends HumanMessage to state.messages
      call_llama     -- formats full history into a prompt, calls LLM,
                        extracts generated text, appends AIMessage to history
      print_response -- prints the latest llm_response

    3-way routing after get_user_input:
      should_exit=True  --> END
      is_command=True   --> get_user_input  (command or empty input)
      otherwise         --> call_llama
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Read a line from stdin.

        For normal input, appends a HumanMessage to state.messages so the
        conversation history grows.  Control commands (verbose/quiet/quit/empty)
        set is_command=True and do NOT add a message to history.

        Reads state:  verbose
        Updates state: user_input, should_exit, is_command, verbose,
                       messages (HumanMessage appended for normal input)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input node")
            print(f"[TRACE] History length: {len(state.get('messages', []))} messages")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        # Empty input — do not pass to LLM
        if not user_input.strip():
            if state.get("verbose", False):
                print("[TRACE] Empty input detected, looping back to get_user_input")
            print("Please enter some text.")
            return {"user_input": "", "should_exit": False, "is_command": True}

        # verbose/quiet mode toggles
        if user_input.lower() == "verbose":
            if state.get("verbose", False):
                print("[TRACE] User entered 'verbose' command")
            print("[MODE] Verbose mode enabled - tracing information will be shown")
            return {"user_input": "", "should_exit": False, "verbose": True, "is_command": True}

        elif user_input.lower() == "quiet":
            if state.get("verbose", False):
                print("[TRACE] User entered 'quiet' command")
            print("[MODE] Quiet mode enabled - tracing information hidden")
            return {"user_input": "", "should_exit": False, "verbose": False, "is_command": True}

        # Quit
        if user_input.lower() in ["quit", "exit", "q"]:
            if state.get("verbose", False):
                print("[TRACE] User entered exit command")
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "is_command": True}

        # Normal input — append to history and route to LLM
        if state.get("verbose", False):
            print(f"[TRACE] User input captured: '{user_input}'")
            print("[TRACE] Appending HumanMessage to history, routing to call_llama")
        return {
            "user_input": user_input,
            "should_exit": False,
            "is_command": False,
            "messages": [HumanMessage(content=user_input)],  # add_messages appends
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Format the full message history into a prompt string and query Llama.
        Extracts only the newly generated text and appends it as an AIMessage
        so the history is available on the next turn.

        Reads state:  messages (full history), verbose
        Updates state: llm_response, messages (AIMessage appended)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama node")

        # Build prompt from the full conversation history
        prompt = format_prompt(state["messages"])

        if state.get("verbose", False):
            print(f"[TRACE] Prompt ({len(prompt)} chars):\n{prompt}")

        print("\nProcessing with Llama-3.2-1B-Instruct...")
        full_output = llama_llm.invoke(prompt)

        # Strip the prompt prefix and any self-generated User: continuations
        response = extract_response(full_output, prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Response extracted ({len(response)} chars): '{response[:80]}...'")
            print("[TRACE] Appending AIMessage to history, routing to print_response")

        return {
            "llm_response": response,
            "messages": [AIMessage(content=response)],  # add_messages appends
        }

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Print the latest LLM response.

        Reads state:  llm_response, verbose
        Updates state: nothing (returns empty dict)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering print_response node")

        print("\n" + "-" * 50)
        print("Llama-3.2-1B-Instruct Response:")
        print("-" * 50)
        print(state["llm_response"])

        if state.get("verbose", False):
            print("[TRACE] Response printed, looping back to get_user_input")
        return {}

    # =========================================================================
    # ROUTING FUNCTION  (3-way, same as task2)
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        3-way conditional branch after get_user_input:

          should_exit=True  --> END
          is_command=True   --> "get_user_input"  (command or empty)
          otherwise         --> "call_llama"
        """
        if state.get("verbose", False):
            print("[TRACE] Evaluating routing decision...")

        if state.get("should_exit", False):
            if state.get("verbose", False):
                print("[TRACE] Route decision: END (user exit)")
            return END

        if state.get("is_command", False):
            if state.get("verbose", False):
                print("[TRACE] Route decision: get_user_input (command or empty input)")
            return "get_user_input"

        if state.get("verbose", False):
            print("[TRACE] Route decision: call_llama")
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama",     call_llama)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama":      "call_llama",
            "get_user_input":  "get_user_input",
            END:               END,
        }
    )

    graph_builder.add_edge("call_llama",     "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()


def save_graph_image(graph, filename="lg_graph.png"):
    """Save a Mermaid PNG of the graph structure."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


def main():
    print("=" * 50)
    print("LangGraph Chat Agent with History (Task 5)")
    print("  Model: Llama-3.2-1B-Instruct")
    print("  History: full conversation retained across turns")
    print("=" * 50)
    print("  Type 'verbose' to enable per-node trace output.")
    print("  Type 'quiet'   to disable trace output.")
    print("  Type 'quit'    to exit.")
    print()

    device = get_device()
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Seed the message history with a system prompt.
    # add_messages will append all subsequent Human/AI messages to this list.
    initial_state: AgentState = {
        "messages":     [SystemMessage(content=SYSTEM_PROMPT)],
        "user_input":   "",
        "should_exit":  False,
        "is_command":   False,
        "verbose":      False,
        "llm_response": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
