## Task 7 — Checkpoint and Crash Recovery

**Script:** [task7_checkpoint_recovery_agent.py](task7_checkpoint_recovery_agent.py)
**Terminal output:** [task7_output.txt](task7_output.txt)

### What checkpointing adds

Without checkpointing, the entire conversation state lives only in Python memory. If the process is killed (Ctrl+C, Ctrl+Z, power loss, OS termination), everything is lost and the next run starts blank.

With the `SqliteSaver` checkpointer, LangGraph writes the full graph state to a SQLite database file (`conversation.db`) after every superstep. On restart, the program inspects the database and resumes exactly where it left off.

### How LangGraph checkpointing works

After each node completes and returns its state update, LangGraph:
1. Merges the update into the current state using the field reducers
2. Records `next` — the set of nodes that should run after this superstep
3. Writes `(state_values, next)` to the checkpoint store under the `thread_id`

When `graph.invoke` is called with a `thread_id` that has a checkpoint:
- The saved state is restored
- Execution resumes at the nodes listed in `next`

### Key implementation changes from task6

**New import:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
```
*(May require `pip install langgraph-checkpoint-sqlite` depending on your LangGraph version.)*

**`create_graph` now accepts a `checkpointer` parameter:**
```python
def create_graph(llama_llm, qwen_llm, checkpointer):
    ...
    return graph_builder.compile(checkpointer=checkpointer)
```

**`main()` opens the store and detects whether to resume or start fresh:**
```python
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conversation.db")

with SqliteSaver.from_conn_string(db_path) as checkpointer:
    graph = create_graph(llama_llm, qwen_llm, checkpointer)
    config = {"configurable": {"thread_id": "chat"}}

    state = graph.get_state(config)
    if state.next:
        # Crashed mid-run — resume from last checkpoint
        graph.invoke(None, config=config)
    else:
        # No checkpoint or previous run ended normally
        graph.invoke(initial_state, config=config)
```

### Recovery scenarios

| What happened | `state.next` | Behavior on restart |
|---------------|-------------|---------------------|
| Killed while waiting for input | `["get_user_input"]` | Asks for input again; history preserved |
| Killed while LLM was generating | `["call_llama"]` or `["call_qwen"]` | Re-runs the LLM call; history preserved |
| Killed during print_response | `["print_response"]` | Re-prints the response; history preserved |
| Typed "quit" (normal exit) | `[]` (empty — hit END) | Starts a fresh conversation |
| First ever run | `[]` (no DB yet) | Starts a fresh conversation |

### Starting a fresh conversation

Delete `conversation.db` and restart. The script also prints a reminder on startup:
```
To start fresh, delete conversation.db and restart.
```
