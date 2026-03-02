# CS 6501 — Agentic AI

## Table of Contents

| Topic | Description | Tasks |
|-------|-------------|-------|
| [Topic 1 — LLM Evaluation](topic1/README.md) | Evaluate small LLMs on MMLU across hardware and quantization configs | 8 tasks |
| [Topic 2 — Agent Frameworks](topic2/README.md) | Build and understand agent workflows using LangGraph | 7 tasks |
| [Topic 3 — Agent Tool Use](topic3/README.md) | Manual and LangChain tool handling; parallel Ollama inference | 6 tasks |
| [Topic 4 — LangGraph Tool Calling](topic4/README.md) | ToolNode vs. ReAct agent; YouTube educational video agent | 2 tasks |


Note to Professor Kautz: I am still working on Topics 5 & 6, but they will definitely be in my portfolio by the end of spring break. Please check back. Thank you!

---

### Topic 1 — LLM Evaluation

| Task | Description |
|------|-------------|
| [Task 1](topic1/README.md#task-1--python-environment-setup) | Python environment setup |
| [Task 2](topic1/README.md#task-2--hugging-face-authorization-for-llama-32-1b) | Hugging Face authorization for Llama 3.2-1B |
| [Task 3](topic1/README.md#task-3--verify-setup) | Verify setup with MMLU evaluation |
| [Task 4](topic1/README.md#task-4--timing-comparison) | Timing comparison across GPU/CPU and quantization levels |
| [Task 5](topic1/README.md#task-5--multi-model-evaluation-with-timing-and-question-output) | Multi-model evaluation (Llama, Qwen, OLMo) with detailed timing |
| [Task 6](topic1/README.md#task-6--results-and-graph-analysis) | Results and graph analysis |
| [Task 7](topic1/README.md#task-7--google-colab-with-larger-models) | Google Colab evaluation with larger models on A100 |
| [Task 8](topic1/README.md#task-8--chat-agent-with-context-management) | Chat agent with sliding-window context management |

---

### Topic 2 — Agent Frameworks

| Task | Description |
|------|-------------|
| [Task 0](topic2/README.md#task-0--langgraph-graph-api-overview) | LangGraph Graph API overview (state, nodes, edges, checkpoints) |
| [Task 1](topic2/README.md#task-1--simple-langgraph-agent-with-verbose-tracing) | Simple LangGraph agent with verbose tracing |
| [Task 2](topic2/README.md#task-2--empty-input-handling-via-3-way-router) | Empty input handling via 3-way router |
| [Task 3](topic2/README.md#task-3--parallel-llm-fan-out-llama--qwen) | Parallel LLM fan-out (Llama + Qwen) |
| [Task 4](topic2/README.md#task-4--input-routed-llm-selection) | Input-routed LLM selection |
| [Task 5](topic2/README.md#task-5--chat-history-with-message-api) | Chat history with Message API |
| [Task 6](topic2/README.md#task-6--three-party-chat-human--llama--qwen) | Three-party chat (Human + Llama + Qwen) |
| [Task 7](topic2/README.md#task-7--checkpoint-and-crash-recovery) | Checkpoint and crash recovery with SQLite |

---

### Topic 3 — Agent Tool Use

| Task | Description |
|------|-------------|
| [Task 1](topic3/README.md#task-1--sequential-vs-parallel-ollama-inference) | Sequential vs. parallel Ollama inference |
| [Task 3](topic3/README.md#task-3--manual-tool-handling-with-a-geometric-calculator) | Manual tool handling with a geometric calculator |
| [Task 4](topic3/README.md#task-4--langchain-tool-handling-with-multiple-tools) | LangChain tool handling with multiple tools (weather, calculator, letter count) |
| [Task 5](topic3/README.md#task-5--langgraph-persistent-conversation-with-checkpointing) | LangGraph persistent conversation with checkpointing |
| [Task 6](topic3/README.md#task-6--parallelization-opportunity) | Parallelization opportunity analysis |

---

### Topic 4 — LangGraph Tool Calling

| Task | Description |
|------|-------------|
| [Task 3](topic4/README.md#task-3-toolnode-vs-react-agent) | ToolNode vs. ReAct agent — parallel dispatch, graph structure, design trade-offs |
| [Task 5](topic4/README.md#task-5-youtube-educational-video-agent) | YouTube educational video agent (summary, key concepts, quiz) |
