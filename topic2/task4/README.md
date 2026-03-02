## Task 4 — Input-Routed LLM Selection

**Script:** [task4_routed_llm_agent.py](task4_routed_llm_agent.py)
**Terminal output:** [task4_output.txt](task4_output.txt)

### Changes from task3

The parallel fan-out is replaced with **content-based routing**: only one model runs per turn, chosen by the router based on the user's input prefix.

| Input | Model selected |
|-------|---------------|
| Starts with "Hey Qwen" (any case) | Qwen2.5-0.5B-Instruct |
| Anything else | Llama-3.2-1B-Instruct (default) |

**`dispatch` node removed.** Its fan-out logic is replaced by two extra branches in `route_after_input`, which now has **4 ways**:

```
should_exit=True             -->  END
is_command=True              -->  "get_user_input"
input starts with "hey qwen" -->  "call_qwen"
otherwise                    -->  "call_llama"
```

**State simplification:** `llama_response` and `qwen_response` are merged back into a single `llm_response` field. A new `selected_model` field (set by whichever LLM node runs) tells `print_response` how to label the output.

**`call_qwen` strips the prefix** before sending to the model, so the model only sees the actual question:

```python
trimmed = raw[len("hey qwen"):].strip()
prompt  = f"User: {trimmed}\nAssistant:"
```

**Graph structure:**

```
                         +-> call_llama --+
get_user_input ----------+               +--> print_response --> get_user_input
                         +-> call_qwen  -+
                         +-> get_user_input (is_command or empty)
                         +-> END (should_exit)
```

**Router expansion:** route_after_input now has 4 branches (was 3):

```
should_exit  → END
is_command   → get_user_input
"hey qwen"   → call_qwen
otherwise    → call_llama
```
