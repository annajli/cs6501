"""
Chat Agent with Sliding Window Context Management

Extends simple_chat_agent.py with:
  1. Sliding window: keeps only the last N turns in the context sent to the
     model, preventing unbounded growth while preserving recent conversation.
  2. --no-history flag: stateless mode where each turn sees only the system
     prompt and the current message — no memory of previous turns at all.

Usage:
  python chat_agent.py                   # sliding window, default 10 turns
  python chat_agent.py --window 5        # sliding window, keep last 5 turns
  python chat_agent.py --no-history      # stateless (no memory)
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description="Chat agent with sliding window context management")
parser.add_argument(
    "--no-history",
    dest="no_history",
    action="store_true",
    default=False,
    help="Stateless mode: each turn gets only the system prompt + current "
         "message. The model has no memory of earlier turns."
)
parser.add_argument(
    "--window",
    type=int,
    default=10,
    metavar="N",
    help="Number of recent user+assistant pairs to keep in context "
         "(default: 10). Ignored when --no-history is set."
)
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME    = "meta-llama/Llama-3.2-1B-Instruct"
SYSTEM_PROMPT = "You are a helpful AI assistant. Be concise and friendly."
MAX_NEW_TOKENS = 512

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,   # FP16 for efficiency on GPU
    device_map="auto",           # auto-selects CUDA if available
    low_cpu_mem_usage=True
)
model.eval()
print(f"✓ Model loaded on: {next(model.parameters()).device}\n")

# ============================================================================
# CONTEXT MANAGEMENT: SLIDING WINDOW
# ============================================================================
#
# The core problem
# ─────────────────
# A language model has no built-in memory. To give it conversational memory
# we feed the entire chat history as input on every turn:
#
#   Turn 1 input:  [system, user1]
#   Turn 2 input:  [system, user1, asst1, user2]
#   Turn 3 input:  [system, user1, asst1, user2, asst2, user3]
#   ...
#
# This grows without bound, increasing both latency and memory.
# Llama 3.2-1B supports up to 128 K tokens, but even well before that limit,
# irrelevant early messages can distract the model from recent context.
#
# The sliding window fix
# ───────────────────────
# Always send the model:
#   • The system prompt  (permanent — sets the assistant's behaviour)
#   • The last N user+assistant pairs  (the "window")
#
# Older pairs are silently dropped from the context that is *sent to the model*.
# The full history is still stored in memory so we could inspect it or write
# it to a log, but the model never sees it after it leaves the window.
#
# Example with --window 2:
#
#   full_history after turn 4:
#     [system, user1, asst1, user2, asst2, user3, asst3, user4, asst4]
#
#   Context sent to model for turn 5:
#     [system, user3, asst3, user4, asst4, user5]   ← user1-2 dropped
#
# Trade-off: the model can no longer refer to anything that slid out of the
# window.  A larger window preserves more history but uses more tokens.
# For most conversations, the last 5–10 turns contain all relevant context.

def build_sliding_window(full_history, window_size):
    """Return the windowed context to send to the model.

    Args:
        full_history: complete list of message dicts, starting with system.
        window_size:  max number of user+assistant *pairs* to include.

    Returns:
        (context, n_dropped)
        context   — system prompt + last window_size pairs
        n_dropped — how many pairs were left out
    """
    system_msgs = [m for m in full_history if m["role"] == "system"]
    conv_msgs   = [m for m in full_history if m["role"] != "system"]

    # Each pair = 1 user message + 1 assistant message = 2 entries
    max_conv_msgs = window_size * 2
    if len(conv_msgs) > max_conv_msgs:
        n_dropped = (len(conv_msgs) - max_conv_msgs) // 2
        conv_msgs = conv_msgs[-max_conv_msgs:]
    else:
        n_dropped = 0

    return system_msgs + conv_msgs, n_dropped

# ============================================================================
# INITIALIZE CHAT HISTORY
# ============================================================================

# full_history holds every message for the entire session.
# The model only ever sees the sliding-window slice of it.
full_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ============================================================================
# CHAT LOOP
# ============================================================================

mode_label = (
    "no-history (stateless)" if args.no_history
    else f"sliding window — last {args.window} turns"
)
print("=" * 70)
print(f"Chat started  |  mode: {mode_label}")
print("Type 'quit' or 'exit' to end the conversation.")
print("=" * 70)
print()

turn = 0
while True:

    # -------------------------------------------------------------------------
    # Step 1: Get user input
    # -------------------------------------------------------------------------
    user_input = input("You: ").strip()

    if user_input.lower() in ["quit", "exit", "q"]:
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    turn += 1

    # -------------------------------------------------------------------------
    # Step 2: Add user message to the full history (always)
    # -------------------------------------------------------------------------
    full_history.append({"role": "user", "content": user_input})

    # -------------------------------------------------------------------------
    # Step 3: Build the context to send to the model
    #
    # --no-history (stateless):
    #   context = [system, current_user_message]
    #   The model sees no previous turns whatsoever.
    #   Every response is generated from scratch.
    #
    # sliding window (default):
    #   context = [system] + last N user+assistant pairs + current_user_message
    #   The model can refer to recent turns but not arbitrarily old ones.
    # -------------------------------------------------------------------------
    if args.no_history:
        context  = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_input},
        ]
        n_dropped = 0
    else:
        # full_history already contains the new user message (Step 2)
        context, n_dropped = build_sliding_window(full_history, args.window)

    # -------------------------------------------------------------------------
    # Step 4: Tokenize the context (text → token IDs)
    #
    # apply_chat_template formats the message list with Llama's special tokens
    # (<|start_header_id|>, <|end_header_id|>, etc.) then converts to integers.
    #
    # Note: newer transformers versions return a BatchEncoding rather than a
    # plain Tensor from apply_chat_template, so we handle both cases.
    # -------------------------------------------------------------------------
    _encoded = tokenizer.apply_chat_template(
        context,
        add_generation_prompt=True,   # appends the "assistant:" prompt
        return_tensors="pt"
    )
    input_ids = (_encoded if isinstance(_encoded, torch.Tensor)
                 else _encoded["input_ids"]).to(model.device)

    attention_mask = torch.ones_like(input_ids)

    # Show per-turn context info so the sliding window is observable
    info = f"turn {turn} | {input_ids.shape[1]} tokens in context"
    if not args.no_history and n_dropped > 0:
        info += f" | ⚠ {n_dropped} old turn(s) dropped from window"
    print(f"  [{info}]")

    # -------------------------------------------------------------------------
    # Step 5: Generate response
    # -------------------------------------------------------------------------
    print("Assistant: ", end="", flush=True)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    # -------------------------------------------------------------------------
    # Step 6: Decode only the newly generated tokens (text → response)
    # -------------------------------------------------------------------------
    new_tokens = outputs[0][input_ids.shape[1]:]
    response   = tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(response)
    print()

    # -------------------------------------------------------------------------
    # Step 7: Add assistant response to full history
    # -------------------------------------------------------------------------
    full_history.append({"role": "assistant", "content": response})

    # The loop repeats. On the next turn:
    # - sliding window: full_history has grown by 2 (user + assistant);
    #   build_sliding_window will drop oldest pair if window is exceeded.
    # - no_history: full_history grows but is never used for context.
