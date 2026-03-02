# task4_routed_llm_agent.py
# MODIFICATION to task3_parallel_llm_agent.py:
# Instead of running both models in parallel, only one model runs per turn.
# The router decides which model based on the user's input:
#   - Starts with "Hey Qwen" (case-insensitive) --> Qwen2.5-0.5B-Instruct
#   - Anything else                              --> Llama-3.2-1B-Instruct
#
# The dispatch node is removed.  route_after_input becomes a 4-way branch:
#   should_exit=True  --> END
#   is_command=True   --> get_user_input
#   "hey qwen" prefix --> call_qwen
#   otherwise         --> call_llama
#
# Both call_llama and call_qwen write to the same llm_response field and set
# selected_model so print_response knows which model produced the output.

import warnings
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


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
    - user_input:     Text entered by the user (set by get_user_input).
    - should_exit:    True when the user types quit/exit/q.
    - is_command:     True for control commands (verbose/quiet) or empty input.
    - verbose:        True when per-node trace output is enabled.
    - selected_model: Name of the model chosen by the router for this turn.
                      Set by whichever LLM node runs (call_llama or call_qwen).
    - llm_response:   The response from whichever model was selected.

    Graph:
        START -> get_user_input -+-> call_llama --+-> print_response -> get_user_input
                      ^          |                |                           |
                      |          +-> call_qwen  --+                           |
                      |          +-> get_user_input (is_command or empty)     |
                      |          +-> END (should_exit)                        |
                      +------------------------------------------------------------+
    """
    user_input: str
    should_exit: bool
    is_command: bool
    verbose: bool
    selected_model: str
    llm_response: str


# =============================================================================
# LLM FACTORY
# =============================================================================

def create_llm(model_id: str, device: str) -> HuggingFacePipeline:
    """
    Load a causal LM from HuggingFace and wrap it as a LangChain LLM.

    Args:
        model_id: HuggingFace model identifier.
        device:   'cuda', 'mps', or 'cpu'.
    """
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

def create_graph(llama_llm: HuggingFacePipeline, qwen_llm: HuggingFacePipeline):
    """
    Build and compile the LangGraph state graph.

    Nodes:
      get_user_input -- reads stdin; handles verbose/quiet/quit/empty
      call_llama     -- queries Llama; only runs when router returns "call_llama"
      call_qwen      -- queries Qwen;  only runs when router returns "call_qwen"
      print_response -- prints whichever model's response was produced this turn

    4-way routing after get_user_input:
      should_exit=True            --> END
      is_command=True             --> get_user_input  (command or empty input)
      input starts with "hey qwen"--> call_qwen
      otherwise                   --> call_llama
    """

    # =========================================================================
    # NODE 1: get_user_input  (unchanged from task2/task3)
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Read a line from stdin.  Handles: empty input, verbose/quiet, quit,
        and normal text (routed to call_llama or call_qwen by the router).

        Reads state:  verbose
        Updates state: user_input, should_exit, is_command, verbose
        """
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input node")

        print("\n" + "=" * 50)
        print("Enter your text (prefix with 'Hey Qwen' to use Qwen, or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        # Empty input — do not pass to any LLM
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

        # Normal input — router will decide which model
        if state.get("verbose", False):
            print(f"[TRACE] User input captured: '{user_input}'")
            print("[TRACE] Routing to model selector")
        return {"user_input": user_input, "should_exit": False, "is_command": False}

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Query Llama-3.2-1B-Instruct with user_input.

        Reads state:  user_input, verbose
        Updates state: llm_response, selected_model
        """
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama node")

        prompt = f"User: {state['user_input']}\nAssistant:"
        if state.get("verbose", False):
            print(f"[TRACE] Llama prompt: '{prompt}'")

        print("\nProcessing with Llama-3.2-1B-Instruct...")
        response = llama_llm.invoke(prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Llama response received ({len(response)} chars)")
            print("[TRACE] Routing to print_response node")
        return {"llm_response": response, "selected_model": "Llama-3.2-1B-Instruct"}

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Query Qwen2.5-0.5B-Instruct with user_input (minus the "Hey Qwen" prefix).

        Strips the "Hey Qwen" prefix before sending to the model so that the
        model sees only the actual question.

        Reads state:  user_input, verbose
        Updates state: llm_response, selected_model
        """
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen node")

        # Strip the "Hey Qwen" prefix (any case) before passing to the model
        raw = state["user_input"]
        trimmed = raw[len("hey qwen"):].strip() if raw.lower().startswith("hey qwen") else raw

        prompt = f"User: {trimmed}\nAssistant:"
        if state.get("verbose", False):
            print(f"[TRACE] Qwen prompt (prefix stripped): '{prompt}'")

        print("\nProcessing with Qwen2.5-0.5B-Instruct...")
        response = qwen_llm.invoke(prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Qwen response received ({len(response)} chars)")
            print("[TRACE] Routing to print_response node")
        return {"llm_response": response, "selected_model": "Qwen2.5-0.5B-Instruct"}

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Print the response from whichever model ran this turn.

        Reads state:  llm_response, selected_model, verbose
        Updates state: nothing (returns empty dict)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering print_response node")

        print("\n" + "-" * 50)
        print(f"{state['selected_model']} Response:")
        print("-" * 50)
        print(state["llm_response"])

        if state.get("verbose", False):
            print("[TRACE] Response printed, looping back to get_user_input")
        return {}

    # =========================================================================
    # ROUTING FUNCTION  (4-way branch)
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        4-way conditional branch after get_user_input:

          should_exit=True             --> END
          is_command=True              --> "get_user_input"  (command or empty)
          user_input starts "hey qwen" --> "call_qwen"
          otherwise                    --> "call_llama"
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

        if state["user_input"].lower().startswith("hey qwen"):
            if state.get("verbose", False):
                print("[TRACE] Route decision: call_qwen ('Hey Qwen' prefix detected)")
            return "call_qwen"

        if state.get("verbose", False):
            print("[TRACE] Route decision: call_llama (default)")
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama",     call_llama)
    graph_builder.add_node("call_qwen",      call_qwen)
    graph_builder.add_node("print_response", print_response)

    # Entry point
    graph_builder.add_edge(START, "get_user_input")

    # 4-way conditional branch from get_user_input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama":      "call_llama",
            "call_qwen":       "call_qwen",
            "get_user_input":  "get_user_input",
            END:               END,
        }
    )

    # Both LLM nodes flow into the same print_response node
    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen",  "print_response")

    # Loop back for next input
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
    print("LangGraph Routed LLM Agent (Task 4)")
    print("  Default model: Llama-3.2-1B-Instruct")
    print("  Prefix 'Hey Qwen' to use Qwen2.5-0.5B-Instruct")
    print("=" * 50)
    print("  Type 'verbose' to enable per-node trace output.")
    print("  Type 'quiet'   to disable trace output.")
    print("  Type 'quit'    to exit.")
    print()

    device = get_device()

    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    qwen_llm  = create_llm("Qwen/Qwen2.5-0.5B-Instruct",       device)

    print("\nCreating LangGraph...")
    graph = create_graph(llama_llm, qwen_llm)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    initial_state: AgentState = {
        "user_input":     "",
        "should_exit":    False,
        "is_command":     False,
        "verbose":        False,
        "selected_model": "",
        "llm_response":   "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
