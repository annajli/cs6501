# task6_multiparty_chat_agent.py
# MODIFICATION to task5_chat_history_agent.py:
# Integrates chat history with the ability to switch between Llama and Qwen
# in a shared three-party conversation (Human, Llama, Qwen).
#
# The challenge: chat history only has roles user/assistant/system/tool, but
# there are three speakers.  Solution: each model sees itself as "assistant"
# and everyone else as "user", with speaker names embedded in message content.
#
# Shared history format (stored in state):
#   conversation: list of {speaker: "Human"/"Llama"/"Qwen", content: str}
#
# View built for Llama (Llama's own messages become assistant, others become user):
#   System: You are Llama...
#   User:   Human: What is the best ice cream flavor?
#   Asst:   Llama: There is no one best flavor...
#   User:   Qwen: No way, chocolate is the best!
#   User:   Human: I agree.
#   Asst:                          <-- model generates here
#
# View built for Qwen (symmetric, Qwen's messages become assistant):
#   System: You are Qwen...
#   User:   Human: What is the best ice cream flavor?
#   User:   Llama: There is no one best flavor...
#   Asst:                          <-- model generates here (no prior Qwen turns yet)
#
# Impersonation prevention (two layers):
#   1. System prompts explicitly instruct each model to respond ONLY as itself
#      and not to generate dialogue for any other participant.
#   2. extract_response() strips any continuation that begins with a known
#      speaker prefix (User:, Human:, Llama:, Qwen:, Assistant:) to catch
#      hallucinated turns that slip past the system prompt.
#
# Routing:
#   "Hey Qwen" prefix --> call_qwen
#   anything else     --> call_llama
#   is_command/empty  --> get_user_input
#   quit              --> END

import warnings
import torch
from operator import add
from typing import TypedDict, Annotated
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Suppress the spurious "Both max_new_tokens and max_length are set" warning
# that fires when a model's saved generation_config.json contains max_length
# and the pipeline is also given max_new_tokens.  max_new_tokens takes
# precedence (as the warning itself says), so this is safe to silence.
warnings.filterwarnings("ignore", message="Both `max_new_tokens`")
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END


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

    Fields:
    - conversation:    Shared three-party history as a list of dicts:
                         {"speaker": "Human"/"Llama"/"Qwen", "content": str}
                       Uses operator.add as reducer — nodes append by returning
                       a one-item list.  All three parties' turns accumulate here.
    - user_input:      Raw text from stdin (used for routing).
    - should_exit:     True when the user types quit/exit/q.
    - is_command:      True for verbose/quiet/empty — bypasses both LLMs.
    - verbose:         True when per-node trace output is enabled.
    - llm_response:    The latest model response (for print_response).
    - selected_model:  "Llama" or "Qwen" — whichever ran last (for display).
    """
    conversation:   Annotated[list, add]   # [{speaker, content}, ...]
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

    Each message is formatted as:
      - "Assistant: {content}"  if speaker == model_name   (the model's own turns)
      - "User: {speaker}: {content}"  for everyone else    (human and the other AI)

    Speaker names are embedded in the content so each model knows who said what.
    The prompt ends with a bare "Assistant:" to cue the model to generate its reply.
    """
    lines = [f"System: {system_prompt}"]
    for msg in conversation:
        speaker = msg["speaker"]
        content = msg["content"]
        if speaker == model_name:
            lines.append(f"Assistant: {content}")
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

    Layer 2 — speaker-token cutoff:
      Cut at the earliest occurrence of a speaker token prefixed by either
      a newline or a space (e.g. "\\nQwen:" or " Qwen:") to stop both
      multi-line hallucination and inline self-talk.  Leading speaker
      prefixes were already stripped in Layer 1, so any remaining matches
      indicate hallucinated continuation turns.
    """
    generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output

    # Strip leading/trailing whitespace so prefix detection works even when
    # the model output begins with a space or newline.
    generated = generated.strip()

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

    # Layer 2: cut at the earliest speaker token to stop both multi-line
    # hallucination and inline self-talk (e.g. "...for you. Qwen: Please
    # continue. Qwen: ...").  Both newline-prefixed and space-prefixed
    # patterns are checked.  Leading speaker prefixes were already stripped
    # in Layer 1, so any space-prefixed match here is a hallucinated turn,
    # not a legitimate self-reference.
    CUTOFFS = [
        "\nUser:", "\nHuman:", "\nLlama:", "\nQwen:", "\nAssistant:",
        " User:", " Human:", " Llama:", " Qwen:", " Assistant:",
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

def create_graph(llama_llm: HuggingFacePipeline, qwen_llm: HuggingFacePipeline):
    """
    Build and compile the LangGraph state graph.

    Nodes:
      get_user_input -- reads stdin; appends Human turn to conversation
      call_llama     -- builds Llama's view of history, calls LLM,
                        appends Llama turn to conversation
      call_qwen      -- builds Qwen's view of history, calls LLM,
                        appends Qwen turn to conversation
      print_response -- prints the latest model response with speaker label

    4-way routing after get_user_input:
      should_exit=True            --> END
      is_command=True             --> get_user_input
      "hey qwen" prefix           --> call_qwen
      otherwise                   --> call_llama
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        """
        Read a line from stdin.

        For normal text, appends {"speaker": "Human", "content": user_input}
        to conversation so all models see the human's turn in their history.

        Reads state:  verbose, conversation (for trace)
        Updates state: user_input, should_exit, is_command, verbose,
                       conversation (Human turn appended for normal input)
        """
        if state.get("verbose", False):
            print("[TRACE] Entering get_user_input node")
            print(f"[TRACE] Conversation length: {len(state.get('conversation', []))} turns")

        print("\n" + "=" * 50)
        print("Enter your text (prefix 'Hey Qwen' for Qwen, or 'quit' to exit):")
        print("=" * 50)
        print("\n> ", end="")
        user_input = input()

        # Empty input
        if not user_input.strip():
            if state.get("verbose", False):
                print("[TRACE] Empty input detected, looping back")
            print("Please enter some text.")
            return {"user_input": "", "should_exit": False, "is_command": True}

        # Verbose/quiet toggles
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

        # Normal input — add Human's turn to the shared history
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
        """
        Build Llama's view of the shared conversation history and call Llama.

        In Llama's view:
          - Llama's own previous turns   --> role "assistant"
          - Human's turns                --> role "user"
          - Qwen's turns                 --> role "user"
        All messages include the speaker name in their content so the model
        knows who said what.

        Reads state:  conversation (full), verbose
        Updates state: llm_response, selected_model,
                       conversation (Llama turn appended)
        """
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
        """
        Build Qwen's view of the shared conversation history and call Qwen.

        In Qwen's view:
          - Qwen's own previous turns    --> role "assistant"
          - Human's turns                --> role "user"
          - Llama's turns                --> role "user"

        Reads state:  conversation (full), verbose
        Updates state: llm_response, selected_model,
                       conversation (Qwen turn appended)
        """
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
        """
        Print the latest model response, labelled with the model's name.

        Reads state:  llm_response, selected_model, verbose
        Updates state: nothing
        """
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
        """
        4-way conditional branch after get_user_input:

          should_exit=True            --> END
          is_command=True             --> "get_user_input"
          input starts with "hey qwen"--> "call_qwen"
          otherwise                   --> "call_llama"
        """
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

    return graph_builder.compile()


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
    print("LangGraph Three-Party Chat Agent (Task 6)")
    print("  Default: Llama-3.2-1B-Instruct")
    print("  Prefix 'Hey Qwen' to address Qwen2.5-0.5B-Instruct")
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

    # conversation starts empty; system prompts are injected per-call in build_prompt
    initial_state: AgentState = {
        "conversation":   [],
        "user_input":     "",
        "should_exit":    False,
        "is_command":     False,
        "verbose":        False,
        "llm_response":   "",
        "selected_model": "",
    }

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
