## Task 1 — Simple LangGraph Agent with Verbose Tracing

**Script:** [task1_verbose_tracing_agent.py](task1_verbose_tracing_agent.py)
**Terminal output:** [task1_output.txt](task1_output.txt)

Starting from the provided `langgraph_simple_llama_agent.py`, the agent was modified to support runtime verbose tracing controlled by two new user commands:

| Command | Effect |
|---------|--------|
| `verbose` | Enables per-node trace output; each node prints `[TRACE]` lines to stdout |
| `quiet` | Suppresses trace output (default at startup) |

### How the LLM is wrapped

The Llama-3.2-1B-Instruct model is loaded from HuggingFace with `AutoModelForCausalLM` and `AutoTokenizer`. A `transformers` `pipeline` is created with generation parameters (temperature, top_p, max_new_tokens), then wrapped with `HuggingFacePipeline` from `langchain_huggingface`. This makes the HuggingFace model look like any other LangChain LLM — it can be called with `.invoke(prompt_string)` just like an OpenAI or Anthropic wrapper would be.

### Changes from the original

**State additions:**

| Field | Type | Purpose |
|-------|------|---------|
| `verbose` | `bool` | Persists the current tracing mode across every iteration of the loop |
| `is_command` | `bool` | Set `True` when input is `verbose`, `quiet`, or `quit` — tells the router the input was a control command, not text for the LLM |

**Nodes — what each one traces when `verbose=True`:**

- `get_user_input` — prints on entry; prints the captured input and routing intent before returning; prints a note when a control command is recognized
- `call_llm` — prints on entry, the raw `user_input`, the formatted prompt sent to the model, the response length, and a note about the next routing step
- `print_response` — prints on entry and confirms the loop-back to `get_user_input`
- `route_after_input` (router) — prints that it is evaluating the decision, then prints which branch it chose and why

**Router — `route_after_input` (three branches):**

```
should_exit = True   →  END
is_command  = True   →  "get_user_input"   (control command; bypass LLM)
otherwise            →  "call_llm"
```

**Conditional edges** updated with the new loop-back branch:

```python
graph_builder.add_conditional_edges(
    "get_user_input",
    route_after_input,
    {
        "call_llm":        "call_llm",           # normal input -> LLM
        "get_user_input":  "get_user_input",     # control command -> loop back
        END:               END,                  # quit -> terminate
    }
)
```

**Graph structure:**

```
START -> get_user_input --+-> call_llm -> print_response -+
              ^           |                                |
              |           +-> get_user_input (is_command) |
              +-----------+--------------------------------+
                          |
                          +-> END (should_exit)
```

### Sample verbose session

```
> verbose
[MODE] Verbose mode enabled - tracing information will be shown

==================================================
Enter your text (or 'quit' to exit):
==================================================

> hello
[TRACE] Entering get_user_input node
[TRACE] User input captured: 'hello'
[TRACE] Routing to call_llm node
[TRACE] Evaluating routing decision...
[TRACE] Route decision: call_llm (process input)
[TRACE] Entering call_llm node
[TRACE] Processing user input: 'hello'
[TRACE] Formatted prompt: 'User: hello\nAssistant:'
Processing your input...
[TRACE] LLM response received (length: 143 chars)
[TRACE] Routing to print_response node
[TRACE] Entering print_response node
--------------------------------------------------
LLM Response:
--------------------------------------------------
User: hello
Assistant: Hello! How can I help you today? ...
[TRACE] Response printed, looping back to get_user_input
```

### Setup

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
```

### Usage

```bash
python task1_verbose_tracing_agent.py
```