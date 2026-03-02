# Topic 1 — LLM Evaluation

Evaluation of small language models on the MMLU benchmark across different hardware and quantization configurations.

- **Hardware:** NVIDIA GeForce RTX 3080 (10 GB VRAM) / Windows 11

---

## Task 1 — Python Environment Setup

A conda environment was created and the following packages installed via pip:

```bash
pip install transformers torch datasets accelerate tqdm huggingface_hub bitsandbytes
```

To enable GPU (CUDA) support, PyTorch must be installed with the appropriate CUDA build. For this machine (CUDA driver 13.0):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

---

## Task 2 — Hugging Face Authorization for Llama 3.2-1B

Followed the guide here to get set up: https://www.cs.virginia.edu/~rmw7my/Courses/AgenticAISpring2026/COMPLETE_HF_AUTH_GUIDE.html.

---

## Task 3 — Verify Setup

Ran `llama_mmlu_eval.py` on two MMLU subjects (`astronomy` and `business_ethics`) to confirm the model loads and produces valid predictions. The script reports per-subject accuracy and saves results to a JSON file.

---

## Task 4 — Timing Comparison

Each configuration was timed using the shell `time` command. The script was extended with `--use-gpu`/`--no-gpu` and `--quantization` CLI flags to switch setups without editing the source file.

### Inference time (evaluation loop only)

| Setup | Device | Quantization | Eval Time | vs GPU no-quant | Accuracy |
|---|---|---|---|---|---|
| GPU, no quant | CUDA | None (FP16) | 10.8 s | 1.0× (baseline) | 48.02% |
| GPU, 4-bit | CUDA | 4-bit NF4 | 16.6 s | 1.5× slower | 44.05% |
| GPU, 8-bit | CUDA | 8-bit | 27.9 s | 2.6× slower | 48.81% |
| CPU, no quant | CPU | None (FP32) | 410.9 s | 38× slower | 47.62% |
| CPU, 4-bit | CPU | 4-bit NF4 | 2336.4 s | 216× slower | 44.44% |

### Wall-clock time (includes model loading, measured with `time`)

| Setup | Wall-clock |
|---|---|
| GPU, no quant | 50.9 s |
| GPU, 4-bit | 29.0 s |
| GPU, 8-bit | 40.7 s |
| CPU, no quant | ~7 min |
| CPU, 4-bit | ~40 min |

### Run commands

```bash
# GPU, no quantization
time python llama_mmlu_eval.py --use-gpu --quantization none

# GPU, 4-bit quantization
time python llama_mmlu_eval.py --use-gpu --quantization 4

# GPU, 8-bit quantization
time python llama_mmlu_eval.py --use-gpu --quantization 8

# CPU, no quantization
time python llama_mmlu_eval.py --no-gpu --quantization none

# CPU, 4-bit quantization
time python llama_mmlu_eval.py --no-gpu --quantization 4
```

### Key Observations

- **GPU no-quant is the fastest for inference.** The model (1B params) fits comfortably in VRAM at FP16, so quantization overhead outweighs any memory bandwidth savings.
- **GPU 4-bit and 8-bit are slower than GPU no-quant.** Quantization adds dequantization overhead on every forward pass. With only 1 output token generated per question, that overhead dominates.
- **CPU 4-bit is extremely slow.** `bitsandbytes` quantization is CUDA-optimized; running it on CPU incurs far more overhead than plain FP32 CPU inference.
- **GPU vs CPU speedup** for full-precision inference: ~38×.
- **Accuracy is consistent across setups** (±5%), confirming quantization preserves model quality at this scale.

---

## Task 5 — Multi-Model Evaluation with Timing and Question Output

`multimodel_mmlu_eval.py` was extended (merging the former `multimodel_mmlu_eval_detailed.py`) to:

1. **Run on 10 MMLU subjects** using 3 models: Llama 3.2-1B-Instruct, Qwen 2.5-0.5B-Instruct, and OLMo-1B.
2. **Track detailed timing per model** — real time, CPU time, and GPU time are measured for every inference call using `time.time()`, `time.process_time()`, and CUDA events respectively.
3. **Print each question with its answer** via the `--print-questions` flag (off by default).
4. **Record per-question results** (question text, choices, correct answer, predicted answer, correct/wrong) in the output JSON, enabling downstream overlap analysis.

---

## Task 6 — Results and Graph Analysis

All graphs are saved to [`mmlu_analysis_plots/`](mmlu_analysis_plots/).

| Plot | Description |
|---|---|
| `1_overall_accuracy.png` | Bar chart of overall accuracy per model |
| `2_subject_comparison.png` | Grouped bar chart of accuracy by subject and model |
| `3_timing_comparison.png` | Avg real, CPU, and GPU time per question per model |
| `4_accuracy_vs_speed.png` | Scatter plot of accuracy vs inference speed |
| `5_performance_heatmap.png` | Heatmap of per-subject accuracy across all models |
| `6_correct_wrong_breakdown.png` | Pie charts showing correct vs wrong per model |
| `7_subject_rankings.png` | Per-subject model rankings |
| `8_question_overlap.png` | Per-subject stacked bar: all-correct / mixed / all-wrong |
| `9_pairwise_agreement.png` | Heatmap of answer agreement between model pairs |

### Subject difficulty

The hardest subjects (averaged across all models) were:
- **moral_scenarios** (27.1%) and **elementary_mathematics** (25.1%) — both near or below random chance (25%), suggesting the models have not learned these concepts at this scale.
- **college_computer_science** (31.3%) — slightly better but still weak.

The easiest were **international_law** (48.5%) and **human_sexuality** (45.3%), likely because these subjects reward pattern matching over reasoning.

### Patterns in mistakes

- **35.3% of questions were answered incorrectly by all three models** — a large shared failure set indicating these questions are genuinely hard for sub-1B models, not just noise.
- **Only 4.3% of questions were answered correctly by all models** — consistent successes are rare, suggesting the models are each relying on different surface patterns rather than shared knowledge.
- **Pairwise agreement** (plot 9) shows models agree on the same answer (right or wrong) at rates well above chance, meaning mistakes are structured, not random — all three models are drawn to the same wrong distractors on difficult questions.
- **Llama-3.2-1B dominated** 8 out of 10 subjects, suggesting better instruction tuning at the same parameter count compared to Qwen and OLMo.
- **OLMo was fastest but least accurate** (23.54% — near random chance), indicating its pretraining did not produce strong MMLU knowledge without further fine-tuning.

---

## Task 7 — Google Colab with Larger Models

**Notebook:** [`colab_files/topic1_colab.ipynb`](https://colab.research.google.com/drive/1xw0unwQP6ZRyLIiM6_yvoNOBGbt4hFfd?usp=sharing) 
**Hardware:** NVIDIA A100-SXM4-40 GB (Colab)
**Results:** [`colab_files/multimodel_mmlu_results_a100_20260302_000534.json`](colab_files/multimodel_mmlu_results_a100_20260302_000534.json)
**Plots:** [`colab_files/mmlu_analysis_plots/`](colab_files/mmlu_analysis_plots/)

The same evaluation pipeline was run on Google Colab using an A100 GPU, extending the model set from 3 small models to 5 by adding two medium-sized models.

### Models and results (10 subjects, 3135 questions, no quantization)

| Model | Size | Accuracy | Avg/question |
|---|---|---|---|
| allenai/OLMo-1B-0724-hf | 1B | 23.32% | 0.019 s |
| Qwen/Qwen2.5-0.5B-Instruct | 0.5B | 35.66% | 0.030 s |
| meta-llama/Llama-3.2-1B-Instruct | 1B | 42.39% | 0.021 s |
| meta-llama/Llama-3.2-3B-Instruct | 3B | 61.31% | 0.034 s |
| Qwen/Qwen2.5-7B-Instruct | 7B | **65.87%** | 0.035 s |

### Key takeaways

- **Larger models dramatically outperform smaller ones.** Going from 1B → 3B parameters (Llama) lifted accuracy from 42% to 61% — a +19 point jump. The 7B Qwen model reached 66%, a +23 point gain over its 0.5B sibling.
- **The 1B accuracy results are consistent with the local RTX 3080 run** (~42% for Llama-3.2-1B in both environments), confirming the evaluation is reproducible across hardware.
- **Speed scales modestly with size.** The 7B model is only ~1.7× slower per question than the 1B model on an A100, because the A100's high memory bandwidth keeps larger models nearly as fast as small ones.
- **OLMo-1B remains near random chance** (23%) regardless of hardware, confirming that its pretraining did not produce strong MMLU knowledge without further fine-tuning.

---

## Task 8 — Chat Agent with Context Management

**Script:** [`chat_agent.py`](chat_agent.py) (extends [`simple_chat_agent.py`](simple_chat_agent.py))

### Context management: sliding window

The base `simple_chat_agent.py` feeds the entire conversation history to the model on every turn, which grows without bound. `chat_agent.py` implements a **sliding window**: only the system prompt plus the last N user+assistant pairs are included in the context sent to the model. Older turns are dropped from the model's view (but kept in `full_history` for the transcript).

```
Window = 2, after turn 4:
  full_history : [system, u1, a1, u2, a2, u3, a3, u4, a4]
  sent to model: [system,         u3, a3, u4, a4, u5     ]
```

Each turn prints the current token count and warns when turns are dropped:
```
[turn 3 | 312 tokens in context | ⚠ 1 old turn(s) dropped from window]
```

### Usage

```bash
# With history — sliding window, keep last 10 turns (default)
python chat_agent.py

# With history — smaller window
python chat_agent.py --window 5

# Without history — stateless (each turn is independent)
python chat_agent.py --no-history
```

### With history vs. without history

| Aspect | With history (sliding window) | Without history (--no-history) |
|---|---|---|
| Context sent to model | System prompt + last N turns | System prompt + current message only |
| Remembers your name | Yes — can refer back to earlier turns | No — forgets immediately |
| Follows up on prior answers | Yes — "as I mentioned..." works | No — each turn is independent |
| Pronoun resolution ("it", "that") | Works within the window | Breaks — model has no referent |
| Token count growth | Bounded (window × avg tokens/turn) | Constant (only current message) |
| Risk of context overflow | Eliminated by window | None |
