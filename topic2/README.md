# Topic 2 — Agent Frameworks

Building and understanding agent workflows using LangGraph.

- **Hardware:** NVIDIA GeForce RTX 3080 (10 GB VRAM) / Windows 11

---

## Table of Contents

| # | Task | Files |
|---|------|-------|
| 0 | [LangGraph Graph API Overview](#task-0--langgraph-graph-api-overview) | *(reading / notes)* |
| 1 | [Simple LangGraph Agent with Verbose Tracing](#task-1--simple-langgraph-agent-with-verbose-tracing) | [task1_verbose_tracing_agent.py](task1/task1_verbose_tracing_agent.py) · [task1_output.txt](task1/task1_output.txt) |
| 2 | [Empty Input Handling via 3-Way Router](#task-2--empty-input-handling-via-3-way-router) | [task2_no_empty_input_agent.py](task2/task2_no_empty_input_agent.py) · [task2_output.txt](task2/task2_output.txt) |
| 3 | [Parallel LLM Fan-Out (Llama + Qwen)](#task-3--parallel-llm-fan-out-llama--qwen) | [task3_parallel_llm_agent.py](task3/task3_parallel_llm_agent.py) · [task3_output.txt](task3/task3_output.txt) |
| 4 | [Input-Routed LLM Selection](#task-4--input-routed-llm-selection) | [task4_routed_llm_agent.py](task4/task4_routed_llm_agent.py) · [task4_output.txt](task4/task4_output.txt) |
| 5 | [Chat History with Message API](#task-5--chat-history-with-message-api) | [task5_chat_history_agent.py](task5/task5_chat_history_agent.py) · [task5_output.txt](task5/task5_output.txt) |
| 6 | [Three-Party Chat (Human + Llama + Qwen)](#task-6--three-party-chat-human--llama--qwen) | [task6_multiparty_chat_agent.py](task6/task6_multiparty_chat_agent.py) · [task6_output.txt](task6/task6_output.txt) |
| 7 | [Checkpoint and Crash Recovery](#task-7--checkpoint-and-crash-recovery) | [task7_checkpoint_recovery_agent.py](task7/task7_checkpoint_recovery_agent.py) · [task7_output.txt](task7/task7_output.txt) |

---

## Task 0 — LangGraph Graph API Overview

Source: [LangChain Graph API Overview](https://docs.langchain.com/oss/python/langgraph/graph-api)

LangGraph models agent workflows as **directed graphs**. Behavior is defined by three key components: **state**, **nodes**, and **edges**.

---

### State

The state is a data structure (typically a `TypedDict` or Pydantic model) that flows through every node in the graph. It represents a complete snapshot of the application at any point in time.

- Contains all data that spans nodes — input, intermediate results, flags, conversation history, etc.
- Each node receives the full current state and returns a **partial update** (a dict of only the fields it changed).
- LangGraph merges updates back into the state using **reducer functions**. The default reducer simply overwrites the field with the new value. Custom reducers (e.g., `operator.add`) can instead append to a list, enabling patterns like maintaining a running conversation history with `add_messages`.

```python
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    llm_response: str
```

---

### Nodes

Nodes are Python functions that **do work**. Each node:

- Accepts the current state as its argument
- Performs a computation — calling an LLM, invoking a tool, or any other data processing
- Returns a dict containing only the state fields it wants to update

Nodes are added to a `StateGraph` and named so edges can reference them:

```python
graph_builder.add_node("call_llm", call_llm)
```

---

### Edges

Edges are connections between nodes that **define flow** — they tell the graph what to do next after a node finishes.

**Normal edges** always go from one specific node to another:

```python
graph_builder.add_edge("call_llm", "print_response")
```

**`START` and `END`** are special virtual nodes. `START` marks the entry point of the graph; `END` terminates execution.

**Conditional edges** choose from a set of potential next nodes based on the current state. They require a **router function** that inspects the state and returns the name of the next node:

```python
graph_builder.add_conditional_edges(
    "get_user_input",       # source node
    route_after_input,      # router function
    {
        "call_llm": "call_llm",
        "get_user_input": "get_user_input",
        END: END
    }
)
```

---

### Routers

A router is a plain Python function that **gates a conditional edge**. It takes the current state as input and returns a string (or list of strings) indicating which node(s) to go to next. Routers contain all the decision logic for branching.

```python
def route_after_input(state: AgentState) -> str:
    if state.get("should_exit", False):
        return END
    if state.get("skip_llm", False):
        return "get_user_input"
    return "call_llm"
```

---

### Parallelism

If a node has **multiple outgoing edges** (normal or conditional), all destination nodes execute in parallel in the same "superstep." This is how LangGraph supports fan-out patterns — for example, calling multiple tools or sub-agents simultaneously and then merging their results in a subsequent node.

---

### Checkpoints

Checkpoints are snapshots of the full graph state saved at every superstep. They are enabled by attaching a **checkpointer** (e.g., `MemorySaver`) when compiling the graph:

```python
from langgraph.checkpoint.memory import MemorySaver
graph = graph_builder.compile(checkpointer=MemorySaver())
```

Checkpoints enable:
- **Persistence** — state survives across separate `invoke` calls (e.g., multi-turn conversations)
- **Time travel** — replay or branch execution from any previous checkpoint
- **Human-in-the-loop** — pause the graph mid-run for human review, then resume

Each checkpoint is identified by a `thread_id` passed in the config, so multiple independent conversation threads can share the same compiled graph.

---

### Compilation and Execution

After all nodes and edges are defined, the graph must be **compiled** before it can run:

```python
graph = graph_builder.compile()
```

Compilation validates the graph structure and wires up runtime features (checkpointers, interrupts). The graph is then executed by calling `invoke` with an initial state:

```python
graph.invoke({"user_input": "", "should_exit": False, "llm_response": ""})
```

Execution proceeds in discrete **supersteps**: at each superstep all currently-active nodes run (in parallel if there are multiple), update the state, and the resulting edges determine which nodes activate next. The graph halts when no nodes remain active.

---

## Task 1 — Simple LangGraph Agent with Verbose Tracing

**Script:** [task1_verbose_tracing_agent.py](task1/task1_verbose_tracing_agent.py)
**Terminal output:** [task1_output.txt](task1/task1_output.txt)
**README:** [task1/README.md](task1/README.md)

Starting from the provided `langgraph_simple_llama_agent.py`, the agent was modified to support runtime verbose tracing controlled by two new user commands:

| Command | Effect |
|---------|--------|
| `verbose` | Enables per-node trace output; each node prints `[TRACE]` lines to stdout |
| `quiet` | Suppresses trace output (default at startup) |

Note: Looking at task1_output.txt, I was initially confused as to why the model will occasionally output multi-turn conversational data. But this is because Llama-3.2-1B-Instruct was trained on multi-turn conversation data, after generating the assistant's first response it sometimes predicts that more conversation follows, and starts generating User: and Assistant: turns on its own until it hits the max_new_tokens=256 budget. The do_sample=True with temperature=0.7 makes generation stochastic — sometimes it happens to generate an end-of-sequence token after the first reply, sometimes it doesn't.

A way to fix this is to post-process the response and take only the text up to the first \nUser: that the model generates:
``` python
response = llm.invoke(prompt)
# Strip any self-generated continuation turns
response = response.split("\nUser:")[0].strip()
```

---

## Task 2 — Empty Input Handling via 3-Way Router

**Script:** [task2_no_empty_input_agent.py](task2/task2_no_empty_input_agent.py)
**Terminal output:** [task2_output.txt](task2/task2_output.txt)

Problem: When user presses Enter without typing anything, Llama-3.2-1B-Instruct hallucinates random completions because it lacks training to handle empty prompts gracefully (unlike larger models that would say "Please provide input"). See terminal output for example of two competely unrelated, random (yet plausible) completions.

Fix: Treat empty input as a command (like "verbose"/"quiet") by setting is_command=True, which makes the router loop back to get_user_input without calling the LLM. Just add an empty-string check in get_user_input that prints "Please enter some text." and returns with is_command=True. No changes needed to router or graph structure.

---

## Task 3 — Parallel LLM Fan-Out (Llama + Qwen)

**Script:** [task3_parallel_llm_agent.py](task3/task3_parallel_llm_agent.py)
**Terminal output:** [task3_output.txt](task3/task3_output.txt)

The task2_no_empty_input_agent.py was modified to replace a single LLM call with parallel execution of two LLMs (Llama-3.2-1B and Qwen2.5-0.5B).

Pattern: Fan-out/fan-in using a dispatch node that triggers both call_llama and call_qwen simultaneously. LangGraph automatically waits for both to complete before proceeding to print_responses.

State updates: Split llm_response into llama_response and qwen_response.

Graph flow:
```
get_user_input → dispatch → call_llama ↘
                        ↘ call_qwen   → print_responses → get_user_input
```

Note: Qwen seemed to produce shorter outputs than Llama, which may have to do with the difference of size.
---

## Task 4 — Input-Routed LLM Selection

**Script:** [task4_routed_llm_agent.py](task4/task4_routed_llm_agent.py)
**Terminal output:** [task4_output.txt](task4/task4_output.txt)

The task3_parallel_llm_agent.py was modified to replace parallel execution with conditional routing — only one LLM runs per turn, selected based on input.

Routing logic:
* Input starts with "Hey Qwen" → call_qwen (prefix stripped before sending to model)
* Anything else → call_llama (default)

State changes:
* Merge llama_response and qwen_response back into single llm_response
* Add selected_model field so print_response knows which model answered

The dispatch node has been deleted — routing happens directly from get_user_input via conditional edges.

---

## Task 5 — Chat History with Message API

**Script:** [task5_chat_history_agent.py](task5/task5_chat_history_agent.py)
**Terminal output:** [task5_output.txt](task5/task5_output.txt)

The task4_routed_llm_agent.py was modified to add persistent conversation history using LangChain's Message API so the LLM now has full conversation context each turn instead of treating every input as isolated.

Implementation:
* Use add_messages reducer on messages field (appends instead of overwrites)
* get_user_input appends HumanMessage each turn
* call_llama appends AIMessage after generating response
* System message set once in initial_state to define persona

Note: I played out a few conversations with the chat agent, as seen in task5_output.txt. I found that at times I needed to be very specific,  otherwise the LLM would begin to hallucinate or not have an answer -- i.e., my questions to the LLM would need to match nearly exact wording of my previous inputs in order to receive the most accurate outputs.

---

## Task 6 — Three-Party Chat (Human + Llama + Qwen)

**Script:** [task6_multiparty_chat_agent.py](task6/task6_multiparty_chat_agent.py)
**Terminal output:** [task6_output.txt](task6/task6_output.txt)

Chat history only supports user/assistant roles, but there are three speakers (Human, Llama, Qwen). The solution is to have each model see itself as "assistant" and everyone else as "user", with speaker names embedded in message content. The state holds a shared history as a plain list of {speaker, content} dicts using operator.add as the reducer to append new turns. When building each model's prompt, build_prompt() transforms this shared history so the model sees its own messages formatted as Assistant: {model_name}: {content} and everyone else's messages as User: {speaker}: {content}.

To prevent models from impersonating other speakers, there are two layers of defense: system prompts explicitly instruct each model to "respond only as yourself", and extract_response() cuts the output at any occurrence of \nHuman:, \nLlama:, or \nQwen: to stop hallucinated multi-turn continuations.

Testing revealed two additional failure modes not caught by the original \nSpeaker: patterns. First, Llama would sometimes start its entire response with "Qwen: ..." — a leading prefix with no preceding newline. Second, Qwen would produce an inline self-loop: "...reply. Qwen: more. Qwen: even more..." using spaces rather than newlines between repetitions. See buggy_output.txt for recorded examples of both bugs.

The leading-prefix bug is fixed by a Layer 1 loop in extract_response() that strips any leading speaker prefix repeatedly (handles the "Qwen: Qwen: ..." double-prefix case too). The inline self-loop is an accepted limitation — space-prefixed cutoffs like " Qwen:" also match legitimate self-references such as "I am Qwen: an AI assistant", producing empty output, so they are excluded from Layer 2 which uses only \nSpeaker: patterns.

I am sure that there's a more elegant solution, and that'll be something that I revisit!

---

## Task 7 — Checkpoint and Crash Recovery

**Script:** [task7_checkpoint_recovery_agent.py](task7/task7_checkpoint_recovery_agent.py)
**Terminal output:** [task7_output.txt](task7/task7_output.txt)

The task6_multiparty_chat_agent.py was modified to add SQLite-backed checkpointing so that killing the program mid-conversation and restarting it resumes exactly where it left off.

LangGraph checkpointing works by serializing the full graph state (all `AgentState` fields, including the entire `conversation` list) to disk after every superstep. The checkpoint also records `next` — which node(s) should run after the saved state — so execution can resume precisely from the interrupted point.

Crash recovery uses `graph.get_state(config)` on startup:
* `state.next` is non-empty → graph was killed mid-run; `graph.invoke(None, config)` resumes it
* `state.next` is empty → no checkpoint or previous run ended normally; `graph.invoke(initial_state, config)` starts fresh

Key changes from task6:
* `SqliteSaver` from `langgraph.checkpoint.sqlite` replaces the default (no checkpointer)
* `create_graph` accepts a `checkpointer` parameter passed to `graph_builder.compile()`
* `main()` opens `conversation.db` via `SqliteSaver.from_conn_string()`, inspects the latest checkpoint, and either resumes or starts fresh
* On resume, the last few conversation turns are printed so the user knows where they are
* Deleting `conversation.db` starts a completely fresh conversation