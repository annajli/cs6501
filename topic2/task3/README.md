## Task 3 — Parallel LLM Fan-Out (Llama + Qwen)

**Script:** [task3_parallel_llm_agent.py](task3_parallel_llm_agent.py)
**Terminal output:** [task3_output.txt](task3_output.txt)

### Changes from task2

The single `call_llm` node is replaced with a **fan-out / fan-in** pattern using two new nodes and an intermediate dispatch node:

```
get_user_input -> dispatch -> call_llama --+-> print_responses -> get_user_input
                          \-> call_qwen  --+
```

**New state fields:**

| Field | Type | Purpose |
|-------|------|---------|
| `llama_response` | `str` | Response from Llama-3.2-1B-Instruct (replaces `llm_response`) |
| `qwen_response` | `str` | Response from Qwen2.5-0.5B-Instruct |

**New nodes:**

- `dispatch` — makes no state changes; its sole purpose is to be the source of two outgoing normal edges, which causes LangGraph to run `call_llama` and `call_qwen` in the same superstep (in parallel)
- `call_llama` — queries Llama-3.2-1B-Instruct; writes `llama_response`
- `call_qwen` — queries Qwen2.5-0.5B-Instruct; writes `qwen_response`
- `print_responses` — LangGraph holds this node until both `call_llama` and `call_qwen` have completed; then prints both responses

**Router change:** the `"call_llm"` branch is renamed to `"dispatch"`:

```
should_exit=True  -->  END
is_command=True   -->  "get_user_input"
otherwise         -->  "dispatch"        ← was "call_llm"
```

**Graph edges for the parallel section:**

```python
# fan-out: both LLM nodes activate in the same superstep
graph_builder.add_edge("dispatch",   "call_llama")
graph_builder.add_edge("dispatch",   "call_qwen")

# fan-in: print_responses waits for both to complete
graph_builder.add_edge("call_llama", "print_responses")
graph_builder.add_edge("call_qwen",  "print_responses")
```

**Model loading:** `create_llm` is refactored to accept a `model_id` argument so both models can be loaded with the same function before the graph is built.