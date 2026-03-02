# task3_parallel_llm_agent.py
# MODIFICATION to task2_no_empty_input_agent.py:
# The single call_llm node is replaced with a fan-out/fan-in pattern:
#
#   dispatch node  -->  call_llama  --+
#                  -->  call_qwen   --+--> print_responses
#
# dispatch receives the user input from state and fans out to both LLM nodes
# in parallel (same LangGraph superstep). print_responses waits for both to
# complete, then prints the results side by side.
#
# Models:
#   Llama: meta-llama/Llama-3.2-1B-Instruct
#   Qwen:  Qwen/Qwen2.5-0.5B-Instruct

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
    - user_input:      Text entered by the user (set by get_user_input).
    - should_exit:     True when the user types quit/exit/q.
    - is_command:      True for control commands (verbose/quiet) or empty input;
                       the router loops back to get_user_input without calling any LLM.
    - verbose:         True when per-node trace output is enabled.
    - llama_response:  Response from Llama-3.2-1B-Instruct (set by call_llama).
    - qwen_response:   Response from Qwen2.5-0.5B-Instruct (set by call_qwen).

    Graph:
        START -> get_user_input -+-> dispatch -> call_llama --+-> print_responses -> get_user_input
                      ^          |           \-> call_qwen  --+                           |
                      |          +-> get_user_input (is_command or empty)                 |
                      |          +-> END (should_exit)                                    |
                      +--------------------------------------------------------------------------+
    """
    user_input: str
    should_exit: bool
    is_command: bool
    verbose: bool
    llama_response: str
    qwen_response: str


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
      get_user_input  -- reads stdin; handles verbose/quiet/quit/empty
      dispatch        -- pass-through fan-out node; triggers parallel LLM calls
      call_llama      -- queries Llama-3.2-1B-Instruct; writes llama_response
      call_qwen       -- queries Qwen2.5-0.5B-Instruct; writes qwen_response
      print_responses -- waits for both LLM nodes, then prints both responses

    Routing after get_user_input:
      should_exit=True  --> END
      is_command=True   --> get_user_input  (command or empty input)
      otherwise         --> dispatch        --> [call_llama || call_qwen]
    """

    # =========================================================================
    # NODE 1: get_user_input  (unchanged from task2)
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Read a line from stdin.  Handles: empty input, verbose/quiet commands,
        quit command, and normal text (routed to dispatch -> parallel LLMs).

        Reads state:  verbose
        Updates state: user_input, should_exit, is_command, verbose
        """
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input node")

        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        # Empty input — do not pass to LLMs
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

        # Normal input — dispatch to both LLMs
        if state.get("verbose", False):
            print(f"[TRACE] User input captured: '{user_input}'")
            print("[TRACE] Routing to dispatch node")
        return {"user_input": user_input, "should_exit": False, "is_command": False}

    # =========================================================================
    # NODE 2: dispatch  (new — fan-out)
    # =========================================================================
    # This node intentionally does no work on the state. Its sole purpose is
    # to be the source of two outgoing normal edges, which causes LangGraph to
    # activate call_llama and call_qwen in the same superstep (in parallel).
    def dispatch(state: AgentState) -> dict:
        """
        Fan-out node: passes user_input (already in state) to both LLM nodes.
        Makes no state changes — returns an empty dict.

        LangGraph activates all nodes that have an incoming edge satisfied in
        the same superstep, so call_llama and call_qwen will run in parallel.
        """
        if state.get("verbose", False):
            print("[TRACE] Entering dispatch node")
            print(f"[TRACE] Fanning out to call_llama and call_qwen in parallel")
        return {}

    # =========================================================================
    # NODE 3: call_llama  (replaces the old call_llm)
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Query Llama-3.2-1B-Instruct with user_input.

        Reads state:  user_input, verbose
        Updates state: llama_response
        """
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama node")

        prompt = f"User: {state['user_input']}\nAssistant:"
        if state.get("verbose", False):
            print(f"[TRACE] Llama prompt: '{prompt}'")

        response = llama_llm.invoke(prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Llama response received ({len(response)} chars)")
        return {"llama_response": response}

    # =========================================================================
    # NODE 4: call_qwen  (new — parallel LLM node)
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Query Qwen2.5-0.5B-Instruct with user_input.

        Reads state:  user_input, verbose
        Updates state: qwen_response
        """
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen node")

        prompt = f"User: {state['user_input']}\nAssistant:"
        if state.get("verbose", False):
            print(f"[TRACE] Qwen prompt: '{prompt}'")

        response = qwen_llm.invoke(prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Qwen response received ({len(response)} chars)")
        return {"qwen_response": response}

    # =========================================================================
    # NODE 5: print_responses  (replaces the old print_response)
    # =========================================================================
    # LangGraph only executes this node once BOTH call_llama and call_qwen have
    # completed (since both have edges leading here). By the time it runs,
    # both llama_response and qwen_response are populated in state.
    def print_responses(state: AgentState) -> dict:
        """
        Print the responses from both LLMs side by side.
        Runs only after both call_llama and call_qwen have completed.

        Reads state:  llama_response, qwen_response, verbose
        Updates state: nothing (returns empty dict)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering print_responses node (both LLMs have finished)")

        print("\n" + "=" * 50)
        print("Llama-3.2-1B-Instruct Response:")
        print("=" * 50)
        print(state["llama_response"])

        print("\n" + "=" * 50)
        print("Qwen2.5-0.5B-Instruct Response:")
        print("=" * 50)
        print(state["qwen_response"])

        if state.get("verbose", False):
            print("[TRACE] Responses printed, looping back to get_user_input")
        return {}

    # =========================================================================
    # ROUTING FUNCTION  (unchanged from task2, except target is now "dispatch")
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Three-way conditional branch after get_user_input:

          should_exit=True  --> END
          is_command=True   --> "get_user_input"  (command or empty; skip LLMs)
          otherwise         --> "dispatch"        (fan out to both LLMs)
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
            print("[TRACE] Route decision: dispatch (fan out to parallel LLMs)")
        return "dispatch"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("dispatch",        dispatch)
    graph_builder.add_node("call_llama",      call_llama)
    graph_builder.add_node("call_qwen",       call_qwen)
    graph_builder.add_node("print_responses", print_responses)

    # Entry point
    graph_builder.add_edge(START, "get_user_input")

    # 3-way conditional branch after get_user_input
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "dispatch":        "dispatch",
            "get_user_input":  "get_user_input",
            END:               END,
        }
    )

    # Fan-out: dispatch activates both LLM nodes in the same superstep
    graph_builder.add_edge("dispatch",   "call_llama")
    graph_builder.add_edge("dispatch",   "call_qwen")

    # Fan-in: print_responses waits for both LLM nodes to complete
    graph_builder.add_edge("call_llama", "print_responses")
    graph_builder.add_edge("call_qwen",  "print_responses")

    # Loop back for next input
    graph_builder.add_edge("print_responses", "get_user_input")

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
    print("LangGraph Parallel LLM Agent (Task 3)")
    print("Llama-3.2-1B-Instruct  vs  Qwen2.5-0.5B-Instruct")
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
        "llama_response": "",
        "qwen_response":  "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
