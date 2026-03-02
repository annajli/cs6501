## Task 2 — Empty Input Handling via 3-Way Router

**Script:** [task2_no_empty_input_agent.py](task2_no_empty_input_agent.py)
**Terminal output:** [task2_output.txt](task2_output.txt)

### What happens on empty input

When the user presses Enter without typing anything, the prompt sent to the model is:

```
User:
Assistant:
```

With no context to ground it, the model hallucinates a plausible-but-random completion — in the two recorded runs it invented a research task about Indigenous Australian library books, then a journalism scenario about a struggling business. The two completions are completely unrelated, revealing a key property of small, less-sophisticated LLMs: **without a meaningful user turn to constrain the output, the model treats the blank prompt as a free-completion problem and generates whatever surface pattern its training data suggests is likely to follow `User:\nAssistant:`**. Larger, more instruction-tuned models would typically respond with something like "I didn't receive any input — please ask me a question," because they have been specifically trained to handle that case. Llama-3.2-1B-Instruct has not.

### Fix: catch empty input in the graph, not in a loop

Rather than wrapping `input()` in a `while` loop (an imperative pattern), the fix uses the existing 3-way conditional edge out of `get_user_input`. The router already has three branches:

```
should_exit = True   →  END
is_command  = True   →  "get_user_input"   (loop back without calling LLM)
otherwise            →  "call_llm"
```

Empty input is treated as another case where `is_command=True` should be set — the node prints `"Please enter some text."` and returns `is_command=True`, and the router loops back without any changes to the existing branch structure.

**Only change in `get_user_input`** — added before the other checks:

```python
if not user_input.strip():
    if state.get("verbose", False):
        print("[TRACE] Empty input detected, looping back to get_user_input")
    print("Please enter some text.")
    return {
        "user_input": "",
        "should_exit": False,
        "is_command": True  # router sends back to get_user_input
    }
```

The router and graph construction are **unchanged**.