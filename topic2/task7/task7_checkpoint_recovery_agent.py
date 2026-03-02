# task7_checkpoint_recovery_agent.py
# MODIFICATION to task6_multiparty_chat_agent.py:
# Adds SQLite-backed checkpointing so that killing the program mid-conversation
# and restarting it resumes exactly where it left off — history intact, no turns lost.
#
# How LangGraph checkpointing works:
#   After every superstep (one node completes and returns its state update),
#   LangGraph serializes the full graph state and writes it to the checkpoint
#   store.  The checkpoint records both the state VALUES and the NEXT nodes
#   that should run.  On the next invoke() call for the same thread_id, the
#   graph resumes from those next nodes with the saved state already in place.
#
# Crash recovery logic (in main()):
#   1. Open a SqliteSaver backed by conversation.db (persists across restarts).
#   2. Call graph.get_state(config) to inspect the latest checkpoint.
#   3a. state.next is non-empty  -->  the graph was killed mid-run.
#         graph.invoke(None, config) resumes from the last saved superstep.
#         The conversation history is already in the restored state.
#   3b. state.next is empty      -->  no checkpoint, or previous run ended
#         normally (user typed "quit").
#         graph.invoke(initial_state, config) starts a fresh conversation.
#
# To wipe the checkpoint and start fresh, delete conversation.db.
#
# Routing (unchanged from task6):
#   "Hey Qwen" prefix --> call_qwen
#   anything else     --> call_llama
#   is_command/empty  --> get_user_input
#   quit              --> END

import os
import warnings
import torch
from operator import add
from typing import TypedDict, Annotated
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Suppress the spurious "Both max_new_tokens and max_length are set" warning.
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver


# =============================================================================
# SYSTEM PROMPTS  (one per model, each names the participants)
# =============================================================================

LLAMA_SYSTEM = (
    "You are Llama, an AI assistant participating in a three-way conversation "
    "with a Human user and another AI called Qwen. The conversation history labels "
    "each message with the speaker's name. Read the history, then give your own "
    "response. Respond only as Llama. Do not generate dialogue for the Human or "
    "Qwen, and do not prefix your reply with 'Llama:'."
)

QWEN_SYSTEM = (
    "You are Qwen, an AI assistant participating in a three-way conversation "
    "with a Human user and another AI called Llama. The conversation history labels "
    "each message with the speaker's name. Read the history, then give your own "
    "response. Respond only as Qwen. Do not generate dialogue for the Human or "
    "Llama, and do not prefix your reply with 'Qwen:'."
)


# =============================================================================
# STATE DEFINITION
# =============================================================================

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.
    SqliteSaver serializes this entire dict to disk after each superstep.

    Fields:
    - conversation:    Shared three-party history as a list of dicts:
                         {"speaker": "Human"/"Llama"/"Qwen", "content": str}
                       Uses operator.add as reducer (appends).
    - user_input:      Raw text from stdin (used for routing).
    - should_exit:     True when the user types quit/exit/q.
    - is_command:      True for verbose/quiet/empty — bypasses both LLMs.
    - verbose:         True when per-node trace output is enabled.
    - llm_response:    The latest model response (for print_response).
    - selected_model:  "Llama" or "Qwen" — whichever ran last.
    """
    conversation:   Annotated[list, add]
    user_input:     str
    should_exit:    bool
    is_command:     bool
    verbose:        bool
    llm_response:   str
    selected_model: str


# =============================================================================
# HELPERS
# =============================================================================

def build_prompt(conversation: list, model_name: str, system_prompt: str) -> str:
    """
    Build a single prompt string for `model_name` from the shared conversation.
    Own turns → "Assistant:", everyone else → "User: {speaker}: {content}".
    """
    lines = [f"System: {system_prompt}"]
    for msg in conversation:
        speaker = msg["speaker"]
        content = msg["content"]
        if speaker == model_name:
            lines.append(f"Assistant: {speaker}: {content}")
        else:
            lines.append(f"User: {speaker}: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def extract_response(full_output: str, prompt: str) -> str:
    """
    Strip the prompt prefix, then remove impersonation artifacts in two layers.

    Layer 1 — leading speaker prefix:
      Small models often prefix their reply with their own name
      (e.g. "Qwen: text", or even "Qwen: Qwen: text").
      Strip any such prefix repeatedly until the response starts with content.

    Layer 2 — newline speaker prefix:
      Cut at the earliest occurrence of a newline-prefixed speaker token
      (e.g. "\\nQwen:") to stop multi-turn hallucination.
      Space-prefixed patterns are excluded because they also match legitimate
      self-references ("I am Qwen: an AI...") and cause empty output.
    """
    generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output

    # Layer 1: strip repeated leading speaker prefixes ("Qwen: Qwen: ..." → "")
    PREFIXES = ["Qwen: ", "Llama: ", "Human: ", "User: ", "Assistant: "]
    changed = True
    while changed:
        changed = False
        for prefix in PREFIXES:
            if generated.startswith(prefix):
                generated = generated[len(prefix):]
                changed = True
                break

    # Layer 2: cut at the earliest newline-prefixed speaker token to stop
    # multi-turn hallucinations.  Space-prefixed patterns (e.g. " Qwen:") are
    # intentionally excluded — they also match legitimate self-references such
    # as "I am Qwen: an AI..." and would produce empty output.
    CUTOFFS = [
        "\nUser:", "\nHuman:", "\nLlama:", "\nQwen:", "\nAssistant:",
    ]
    first_cut = len(generated)
    for cutoff in CUTOFFS:
        idx = generated.find(cutoff)
        if 0 <= idx < first_cut:
            first_cut = idx
    generated = generated[:first_cut]

    return generated.strip()


# =============================================================================
# LLM FACTORY
# =============================================================================

def get_device() -> str:
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"


def create_llm(model_id: str, device: str) -> HuggingFacePipeline:
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

def create_graph(llama_llm: HuggingFacePipeline,
                 qwen_llm:  HuggingFacePipeline,
                 checkpointer):
    """
    Build and compile the LangGraph state graph with the given checkpointer.
    Passing a SqliteSaver as checkpointer makes every superstep persistent.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input node")
            print(f"[TRACE] Conversation length: {len(state.get('conversation', []))} turns")

        print("\n" + "=" * 50)
        print("Enter your text (prefix 'Hey Qwen' for Qwen, or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        if not user_input.strip():
            if state.get("verbose", False):
                print("[TRACE] Empty input detected, looping back")
            print("Please enter some text.")
            return {"user_input": "", "should_exit": False, "is_command": True}

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

        if user_input.lower() in ["quit", "exit", "q"]:
            if state.get("verbose", False):
                print("[TRACE] User entered exit command")
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True, "is_command": True}

        if state.get("verbose", False):
            print(f"[TRACE] User input: '{user_input}'")
            print("[TRACE] Appending Human turn to conversation, routing to model")
        return {
            "user_input": user_input,
            "should_exit": False,
            "is_command": False,
            "conversation": [{"speaker": "Human", "content": user_input}],
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_llama node")
            print(f"[TRACE] Building Llama's view of {len(state['conversation'])} turns")

        prompt = build_prompt(state["conversation"], "Llama", LLAMA_SYSTEM)

        if state.get("verbose", False):
            print(f"[TRACE] Prompt ({len(prompt)} chars):\n{prompt}\n")

        print("\nLlama is thinking...")
        full_output = llama_llm.invoke(prompt)
        response = extract_response(full_output, prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Llama response ({len(response)} chars): '{response[:80]}...'")

        return {
            "llm_response": response,
            "selected_model": "Llama",
            "conversation": [{"speaker": "Llama", "content": response}],
        }

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering call_qwen node")
            print(f"[TRACE] Building Qwen's view of {len(state['conversation'])} turns")

        prompt = build_prompt(state["conversation"], "Qwen", QWEN_SYSTEM)

        if state.get("verbose", False):
            print(f"[TRACE] Prompt ({len(prompt)} chars):\n{prompt}\n")

        print("\nQwen is thinking...")
        full_output = qwen_llm.invoke(prompt)
        response = extract_response(full_output, prompt)

        if state.get("verbose", False):
            print(f"[TRACE] Qwen response ({len(response)} chars): '{response[:80]}...'")

        return {
            "llm_response": response,
            "selected_model": "Qwen",
            "conversation": [{"speaker": "Qwen", "content": response}],
        }

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        if state.get("verbose", False):
            print("[TRACE] Entering print_response node")

        model = state.get("selected_model", "Model")
        print("\n" + "-" * 50)
        print(f"{model}:")
        print("-" * 50)
        print(state["llm_response"])

        if state.get("verbose", False):
            total = len(state.get("conversation", []))
            print(f"[TRACE] Conversation now has {total} turns")
            print("[TRACE] Looping back to get_user_input")
        return {}

    # =========================================================================
    # ROUTING FUNCTION  (4-way)
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        if state.get("verbose", False):
            print("[TRACE] Evaluating routing decision...")

        if state.get("should_exit", False):
            if state.get("verbose", False):
                print("[TRACE] Route decision: END")
            return END

        if state.get("is_command", False):
            if state.get("verbose", False):
                print("[TRACE] Route decision: get_user_input (command or empty)")
            return "get_user_input"

        if state["user_input"].lower().startswith("hey qwen"):
            if state.get("verbose", False):
                print("[TRACE] Route decision: call_qwen ('Hey Qwen' prefix)")
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

    graph_builder.add_edge(START, "get_user_input")

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

    graph_builder.add_edge("call_llama",     "print_response")
    graph_builder.add_edge("call_qwen",      "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile(checkpointer=checkpointer)


def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")


def main():
    print("=" * 50)
    print("LangGraph Three-Party Chat Agent (Task 7)")
    print("  Default: Llama-3.2-1B-Instruct")
    print("  Prefix 'Hey Qwen' to address Qwen2.5-0.5B-Instruct")
    print("=" * 50)
    print("  Type 'verbose' to enable per-node trace output.")
    print("  Type 'quiet'   to disable trace output.")
    print("  Type 'quit'    to exit.")
    print("  To start fresh, delete conversation.db and restart.")
    print()

    device = get_device()
    llama_llm = create_llm("meta-llama/Llama-3.2-1B-Instruct", device)
    qwen_llm  = create_llm("Qwen/Qwen2.5-0.5B-Instruct",       device)

    # SQLite checkpoint store — persists to disk next to this script.
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.db")

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        print("\nCreating LangGraph...")
        graph = create_graph(llama_llm, qwen_llm, checkpointer)
        print("Graph created successfully!")

        print("\nSaving graph visualization...")
        save_graph_image(graph)

        config = {"configurable": {"thread_id": "chat"}}

        initial_state: AgentState = {
            "conversation":   [],
            "user_input":     "",
            "should_exit":    False,
            "is_command":     False,
            "verbose":        False,
            "llm_response":   "",
            "selected_model": "",
        }

        # ---------------------------------------------------------------
        # Crash recovery: inspect the latest checkpoint for this thread.
        #
        # state.next is non-empty  -->  graph was killed mid-run
        #   The saved state already contains the conversation history.
        #   graph.invoke(None, config) resumes from the last superstep.
        #
        # state.next is empty      -->  no checkpoint or previous run ended
        #   normally (user typed "quit").
        #   graph.invoke(initial_state, config) starts a fresh conversation.
        # ---------------------------------------------------------------
        state = graph.get_state(config)

        if state.next:
            conversation = state.values.get("conversation", [])
            print(f"\n{'=' * 50}")
            print(f"Resuming interrupted conversation — {len(conversation)} turns saved.")
            if conversation:
                print("Recent history:")
                for turn in conversation[-4:]:
                    preview = turn["content"][:70].replace("\n", " ")
                    print(f"  [{turn['speaker']}] {preview}")
            print("=" * 50)
            graph.invoke(None, config=config)
        else:
            graph.invoke(initial_state, config=config)


if __name__ == "__main__":
    main()
