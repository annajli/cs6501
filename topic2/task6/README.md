## Task 6 — Three-Party Chat (Human + Llama + Qwen)

**Script:** [task6_multiparty_chat_agent.py](task6/task6_multiparty_chat_agent.py)
**Terminal output:** [task6_output.txt](task6/task6_output.txt)

### The problem: three speakers, two roles

Chat history only supports `system / user / assistant / tool` roles, but there are three speakers (Human, Llama, Qwen). The solution: **each model sees itself as `assistant` and everyone else as `user`**, with speaker names embedded in every message's content so the model can follow who said what.

### Shared conversation history

Instead of a LangChain `messages` list, the state holds a plain list of dicts using `operator.add` as the reducer (which concatenates lists, i.e. appends):

```python
from operator import add
conversation: Annotated[list, add]   # [{speaker, content}, ...]
```

Each turn appends one entry:
- `get_user_input` → `{"speaker": "Human", "content": user_input}`
- `call_llama`     → `{"speaker": "Llama", "content": response}`
- `call_qwen`      → `{"speaker": "Qwen",  "content": response}`

### Model-specific prompt views

`build_prompt(conversation, model_name, system_prompt)` constructs each model's prompt from the shared history:

| Message speaker | Role in model's prompt | Content format |
|-----------------|----------------------|----------------|
| This model | `Assistant:` | `{model_name}: {content}` |
| Anyone else | `User:` | `{speaker}: {content}` |

**Llama's view** of an example exchange:
```
System: You are Llama, ...
User:   Human: What is the best ice cream flavor?
Asst:   Llama: There is no one best flavor, but the most popular is vanilla.
User:   Qwen: No way, chocolate is the best!
User:   Human: I agree.
Asst:
```

**Qwen's view** at the same point (Qwen has no prior turns, so no `Asst:` lines yet):
```
System: You are Qwen, ...
User:   Human: What is the best ice cream flavor?
User:   Llama: There is no one best flavor, but the most popular is vanilla.
User:   Human: Hey Qwen, what do you think?
Asst:
```

### Impersonation prevention (two layers)

Keeping the three entities separate requires preventing each model from inventing speech for the other two participants.

**Layer 1 — System prompts** instruct each model to respond only as itself:

- **Llama**: `"Respond only as Llama. Do not generate dialogue for the Human or Qwen."`
- **Qwen**: `"Respond only as Qwen. Do not generate dialogue for the Human or Llama."`

**Layer 2 — `extract_response()` cutoff patterns** catch hallucinated continuations that slip past the system prompt. Testing revealed two failure modes:

| Failure mode | Example | Root cause | Fix |
|---|---|---|---|
| Leading speaker prefix | Llama outputs `"Qwen: ..."` as its entire response | Model learned to label turns; labels its own output wrong | Layer 1: strip leading `Speaker: ` prefix repeatedly (handles `"Qwen: Qwen: ..."`) |
| Inline self-loop | Qwen outputs `"...reply. Qwen: more. Qwen: even more..."` | Model generates continuations separated by spaces, not newlines | Accepted limitation — space-prefixed cutoffs (` Qwen:`) also fire on legitimate self-references like `"I am Qwen: an AI..."`, producing empty output, so they are excluded |

The `extract_response()` function uses two layers:

```python
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

# Layer 2: cut at the earliest newline-prefixed speaker token
# Space-prefixed patterns excluded — they match legitimate self-references
# and cause empty output.
CUTOFFS = [
    "\nUser:", "\nHuman:", "\nLlama:", "\nQwen:", "\nAssistant:",
]
first_cut = len(generated)
for cutoff in CUTOFFS:
    idx = generated.find(cutoff)
    if 0 <= idx < first_cut:
        first_cut = idx
generated = generated[:first_cut]
```

The "earliest match" strategy in Layer 2 ensures that whichever speaker cutoff appears first in the text wins, regardless of the order they are listed.

### Routing

Same 4-way branch as task4 — "Hey Qwen" prefix routes to `call_qwen`, everything else to `call_llama`.
