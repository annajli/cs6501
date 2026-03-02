## Task 5 — Chat History with Message API

**Script:** [task5_chat_history_agent.py](task5_chat_history_agent.py)
**Terminal output:** [task5_output.txt](task5_output.txt)

Qwen is disabled. Only Llama-3.2-1B-Instruct is used. The key addition is a persistent conversation history using the LangChain Message API.

### Message API concepts used

| Role | Class | When added |
|------|-------|-----------|
| `system` | `SystemMessage` | Once, in `initial_state` — sets the assistant's persona |
| `human` / `user` | `HumanMessage` | By `get_user_input` on every normal turn |
| `ai` / `assistant` | `AIMessage` | By `call_llama` after each model response |

### `add_messages` reducer

The `messages` field uses the `add_messages` reducer from `langgraph.graph.message`:

```python
from langgraph.graph.message import add_messages
from typing import Annotated

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    ...
```

Unlike the default reducer (which overwrites), `add_messages` **appends** new messages to the existing list. Returning `{"messages": [HumanMessage(content="hello")]}` from a node adds one message to the history rather than replacing the whole list. This is how state accumulates across the graph's loop.

### How the history flows through the graph

Each turn:

1. `get_user_input` returns `{"messages": [HumanMessage(content=user_input)]}` — appended to history
2. `call_llama` calls `format_prompt(state["messages"])` to build a single string from the full history, invokes the LLM, extracts the new text, then returns `{"messages": [AIMessage(content=response)]}` — also appended
3. On the next turn, `state["messages"]` contains the system prompt plus all prior human/AI pairs, so the model has full context

### Prompt formatting and response extraction

`HuggingFacePipeline` takes a string, so messages are converted with `format_prompt`:

```
System: You are a helpful assistant.
User: My name is Anna.
Assistant: Hello Anna! How can I help you?
User: What is my name?
Assistant:
```

`HuggingFacePipeline.text-generation` returns the full text (prompt + generated tokens), so `extract_response` strips the prompt prefix. It also cuts off any self-generated `\nUser:` continuation — the multi-turn hallucination bug identified in Task 1:

```python
def extract_response(full_output: str, prompt: str) -> str:
    generated = full_output[len(prompt):] if full_output.startswith(prompt) else full_output
    return generated.split("\nUser:")[0].strip()
```

### Observations from testing

The model correctly recalled information introduced earlier in the conversation (e.g., the cat's name, the weather). However, precision matters — questions phrased differently from the original input sometimes caused the model to hallucinate. For example, "What did we do today?" after mentioning napping produced an evasive non-answer, while "What is my favorite pasttime?" (matching the original wording) succeeded. This reflects the limitations of the 1B parameter model: it has the context in its prompt but inconsistently attends to it.
