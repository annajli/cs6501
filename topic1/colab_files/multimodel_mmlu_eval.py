
"""
Multi-Model MMLU Evaluation Script (Optimized for Google Colab A100)

This script evaluates multiple small/medium language models on the MMLU benchmark:
- Llama 3.2 1B-Instruct
- Qwen 2.5 0.5B-Instruct
- OLMo 2 1B
- Llama 3.2 3B-Instruct
- Qwen 2.5 7B-Instruct
- OLMo 3 Base 7B

Optimized for A100 GPU:
- bfloat16 precision (A100 has native BF16 tensor cores — faster & more stable than FP16)
- No quantization needed (A100 40GB easily fits all models: ~2 GB for 1B, ~14 GB for 7B)
- torch.compile() for additional inference speedup
- Models are evaluated sequentially and unloaded between runs to keep memory clean

Usage:
1. Open in Google Colab with A100 runtime (Runtime > Change runtime type > A100)
2. Install: pip install transformers torch datasets accelerate tqdm
3. Login: huggingface-cli login
4. Run: python multimodel_mmlu_eval.py

Set PRINT_QUESTIONS to True to see each question and answer.
Set USE_TORCH_COMPILE to False if you hit any compilation errors.
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
from tqdm.auto import tqdm
from datetime import datetime
import sys
import platform
import time
import gc

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# Models to evaluate (ordered small to large)
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "allenai/OLMo-1B-0724-hf",          # OLMo 2 1B
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "allenai/OLMo-3-7B"                  # OLMo 3 Base 7B
]

# A100 precision settings
# bfloat16 is the recommended dtype for A100:
#   - Native BF16 tensor cores -> faster than FP16
#   - Wider dynamic range than FP16 -> more numerically stable
#   - No quantization needed; A100 (40 GB) fits all models at full BF16 precision
TORCH_DTYPE = torch.bfloat16

# torch.compile() fuses ops into optimized CUDA kernels, giving a meaningful
# speedup on A100. Adds ~1-2 min compilation overhead the first time each model
# runs. Disable with --no-compile or by setting False below.
USE_TORCH_COMPILE = True

MAX_NEW_TOKENS = 1

# Print detailed question/answer information
PRINT_QUESTIONS = False  # Set to True to see each question, answer, and result

# 10 MMLU subjects to evaluate
MMLU_SUBJECTS = [
    "astronomy",
    "college_computer_science",
    "elementary_mathematics",
    "human_sexuality",
    "international_law",
    "logical_fallacies",
    "moral_scenarios",
    "philosophy",
    "professional_medicine",
    "professional_psychology"
]


# ============================================================================
# ENVIRONMENT CHECK
# ============================================================================

def check_environment():
    """Verify we are running on an A100 and that dependencies are in order."""
    print("=" * 70)
    print("Environment Check")
    print("=" * 70)

    # Colab detection
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except ImportError:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")

    # GPU check
    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected.")
        print("   This script is optimized for an A100. Please switch your Colab")
        print("   runtime: Runtime > Change runtime type > GPU > A100.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✓ GPU: {gpu_name}")
    print(f"✓ VRAM: {gpu_memory_gb:.1f} GB")

    if "A100" not in gpu_name:
        print(f"⚠️  Expected an A100 but found '{gpu_name}'.")
        print("   The script will still run, but settings are tuned for A100.")

    # BF16 support check
    global TORCH_DTYPE
    if torch.cuda.is_bf16_supported():
        print("✓ bfloat16 supported — using native A100 BF16 tensor cores")
    else:
        print("⚠️  bfloat16 not supported on this GPU. Falling back to float16.")
        TORCH_DTYPE = torch.float16

    # torch.compile check
    global USE_TORCH_COMPILE
    if USE_TORCH_COMPILE:
        if int(torch.__version__.split(".")[0]) >= 2:
            print(f"✓ torch.compile() available (PyTorch {torch.__version__})")
        else:
            print(f"⚠️  torch.compile() requires PyTorch >= 2.0 (found {torch.__version__}). Disabling.")
            USE_TORCH_COMPILE = False

    # HF authentication
    try:
        from huggingface_hub import HfFolder
        if HfFolder.get_token():
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token — run: huggingface-cli login")
    except Exception:
        print("⚠️  Could not check Hugging Face authentication")

    # Configuration summary
    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"Models to evaluate : {len(MODELS)}")
    for m in MODELS:
        print(f"  - {m}")
    print(f"Precision          : {TORCH_DTYPE}  (BF16 — optimal for A100)")
    print(f"Quantization       : None  (not needed with {gpu_memory_gb:.0f} GB VRAM)")
    print(f"torch.compile()    : {USE_TORCH_COMPILE}")
    print(f"Subjects           : {len(MMLU_SUBJECTS)}")
    print(f"Print questions    : {PRINT_QUESTIONS}")
    print()
    print("Approximate BF16 VRAM per model:")
    print("  0.5B -> ~1 GB  |  1B -> ~2 GB  |  3B -> ~6 GB  |  7B -> ~14 GB")
    print("=" * 70 + "\n")

    return in_colab


# ============================================================================
# MODEL LOADING / UNLOADING
# ============================================================================

def load_model_and_tokenizer(model_name: str):
    """Load a model in BF16 on CUDA with optional torch.compile()."""
    print(f"\nLoading {model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("  ✓ Tokenizer loaded")

    print("  Loading weights in BF16 (may take 1-3 min for 7B models)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=TORCH_DTYPE,    # BF16 — native A100 tensor-core format
        device_map="auto",          # Automatically places all layers on GPU
        low_cpu_mem_usage=True      # Reduces peak CPU RAM during loading
    )
    model.eval()

    if USE_TORCH_COMPILE:
        print("  Compiling model with torch.compile() ...")
        try:
            model = torch.compile(model)
            print("  ✓ torch.compile() applied")
        except Exception as e:
            print(f"  ⚠️  torch.compile() failed ({e}). Continuing without compilation.")

    mem_alloc = torch.cuda.memory_allocated(0) / 1e9
    mem_reserv = torch.cuda.memory_reserved(0) / 1e9
    print(f"  ✓ Ready  |  dtype: {TORCH_DTYPE}  |  "
          f"VRAM: {mem_alloc:.2f} GB allocated, {mem_reserv:.2f} GB reserved")

    return model, tokenizer


def unload_model(model, tokenizer):
    """Delete model and free GPU memory before loading the next one."""
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    mem = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ Model unloaded  |  VRAM now: {mem:.2f} GB allocated")


# ============================================================================
# INFERENCE UTILITIES
# ============================================================================

def format_mmlu_prompt(question: str, choices: list) -> str:
    """Format an MMLU question as a standard multiple-choice prompt."""
    prompt = f"{question}\n\n"
    for label, choice in zip(("A", "B", "C", "D"), choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


class TimingInfo:
    """Accumulate wall-clock, CPU, and GPU timing across inference calls."""

    def __init__(self):
        self.real_time = 0.0
        self.cpu_time = 0.0
        self.gpu_time = 0.0
        self.num_questions = 0

    def start(self):
        self._t_real = time.time()
        self._t_cpu = time.process_time()
        torch.cuda.synchronize()
        self._ev_start = torch.cuda.Event(enable_timing=True)
        self._ev_end = torch.cuda.Event(enable_timing=True)
        self._ev_start.record()

    def stop(self):
        self._ev_end.record()
        torch.cuda.synchronize()
        self.gpu_time += self._ev_start.elapsed_time(self._ev_end) / 1000.0  # ms -> s
        self.real_time += time.time() - self._t_real
        self.cpu_time += time.process_time() - self._t_cpu
        self.num_questions += 1

    def summary(self) -> dict:
        n = max(self.num_questions, 1)
        return {
            "total_real_time_seconds": self.real_time,
            "total_cpu_time_seconds": self.cpu_time,
            "total_gpu_time_seconds": self.gpu_time,
            "avg_real_time_per_question": self.real_time / n,
            "avg_cpu_time_per_question": self.cpu_time / n,
            "avg_gpu_time_per_question": self.gpu_time / n,
            "num_questions": self.num_questions,
        }


def get_model_prediction(model, tokenizer, prompt: str, timing: TimingInfo) -> str:
    """Run one greedy forward pass and return the predicted answer letter."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    timing.start()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0,
        )
    timing.stop()

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    # Extract first valid answer letter; fall back to "A" if none found
    answer = generated.strip()[:1].upper()
    if answer not in ("A", "B", "C", "D"):
        for ch in generated.upper():
            if ch in ("A", "B", "C", "D"):
                answer = ch
                break
        else:
            answer = "A"

    return answer


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_subject(model, tokenizer, subject: str, model_name: str):
    """Evaluate a model on one MMLU subject and return a results dict."""
    print(f"\n{'='*70}")
    print(f"Subject : {subject}")
    print(f"Model   : {model_name}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Could not load subject '{subject}': {e}")
        return None

    correct = 0
    timing = TimingInfo()
    question_results = []

    for idx, example in enumerate(tqdm(dataset, desc=subject, leave=True)):
        choices = example["choices"]
        correct_answer = ("A", "B", "C", "D")[example["answer"]]
        prompt = format_mmlu_prompt(example["question"], choices)
        predicted = get_model_prediction(model, tokenizer, prompt, timing)

        is_correct = predicted == correct_answer
        if is_correct:
            correct += 1

        question_results.append({
            "question_id": idx,
            "question": example["question"],
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted,
            "is_correct": is_correct,
        })

        if PRINT_QUESTIONS:
            print(f"\n--- Q{idx + 1} ---")
            print(f"  {example['question']}")
            for i, ch in enumerate(choices):
                lbl = ("A", "B", "C", "D")[i]
                marker = "  <--" if lbl == correct_answer else ""
                print(f"  {lbl}. {ch}{marker}")
            print(f"  Predicted: {predicted}  |  Correct: {correct_answer}  |  "
                  f"{'✓ CORRECT' if is_correct else '✗ WRONG'}")

    total = len(question_results)
    accuracy = (correct / total * 100) if total > 0 else 0.0
    t = timing.summary()

    print(f"  ✓ {correct}/{total} = {accuracy:.2f}%  "
          f"| wall: {t['total_real_time_seconds']:.1f}s  "
          f"| GPU: {t['total_gpu_time_seconds']:.1f}s  "
          f"| avg: {t['avg_gpu_time_per_question']*1000:.1f}ms/q")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "timing": t,
        "question_results": question_results,
    }


def evaluate_model(model_name: str) -> dict:
    """Load, evaluate, and unload one model across all MMLU subjects."""
    print(f"\n{'#'*70}")
    print(f"# EVALUATING: {model_name}")
    print(f"{'#'*70}\n")

    model, tokenizer = load_model_and_tokenizer(model_name)

    results = []
    total_correct = 0
    total_questions = 0
    agg = TimingInfo()

    wall_start = time.time()
    cpu_start = time.process_time()

    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nSubject {i}/{len(MMLU_SUBJECTS)}")
        result = evaluate_subject(model, tokenizer, subject, model_name)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]
            t = result["timing"]
            agg.real_time += t["total_real_time_seconds"]
            agg.cpu_time += t["total_cpu_time_seconds"]
            agg.gpu_time += t["total_gpu_time_seconds"]
            agg.num_questions += t["num_questions"]

    wall_elapsed = time.time() - wall_start
    cpu_elapsed = time.process_time() - cpu_start
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0

    unload_model(model, tokenizer)

    return {
        "model": model_name,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results,
        "timing": agg.summary(),
        "wall_clock_time": wall_elapsed,
        "total_cpu_time": cpu_elapsed,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("Multi-Model MMLU Evaluation  —  Google Colab A100")
    print("=" * 70 + "\n")

    in_colab = check_environment()
    all_results = []

    for model_name in MODELS:
        try:
            all_results.append(evaluate_model(model_name))
        except Exception as e:
            print(f"\n❌ Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Comparative summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("=" * 70)

    for r in all_results:
        print(f"\n{r['model']}")
        print("-" * 70)
        print(f"  Accuracy        : {r['overall_accuracy']:.2f}%  "
              f"({r['total_correct']}/{r['total_questions']})")
        print(f"  Wall-clock time : {r['wall_clock_time']:.1f}s  "
              f"({r['wall_clock_time']/60:.2f} min)")
        t = r["timing"]
        print(f"  GPU time (inf.) : {t['total_gpu_time_seconds']:.1f}s")
        print(f"  Avg / question  : {t['avg_real_time_per_question']*1000:.1f}ms wall  "
              f"| {t['avg_gpu_time_per_question']*1000:.1f}ms GPU")

    print("\n" + "=" * 70)
    print("ACCURACY RANKING")
    print("=" * 70)
    for i, r in enumerate(
        sorted(all_results, key=lambda x: x["overall_accuracy"], reverse=True), 1
    ):
        print(f"  {i}. {r['model']:<47} {r['overall_accuracy']:.2f}%")

    print("\n" + "=" * 70)
    print("SPEED RANKING  (avg GPU time per question)")
    print("=" * 70)
    for i, r in enumerate(
        sorted(all_results, key=lambda x: x["timing"]["avg_gpu_time_per_question"]), 1
    ):
        ms = r["timing"]["avg_gpu_time_per_question"] * 1000
        print(f"  {i}. {r['model']:<47} {ms:.1f}ms / question")

    # ── Save results ──────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"multimodel_mmlu_results_a100_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "device": torch.cuda.get_device_name(0),
                "torch_dtype": str(TORCH_DTYPE),
                "torch_compile": USE_TORCH_COMPILE,
                "quantization_bits": None,
                "subjects": MMLU_SUBJECTS,
                "num_subjects": len(MMLU_SUBJECTS),
                "results": all_results,
            },
            f,
            indent=2,
        )
    print(f"\n✓ Results saved -> {output_file}")

    # ── Subject breakdown for the best model ──────────────────────────────────
    if all_results:
        best = max(all_results, key=lambda x: x["overall_accuracy"])
        print(f"\n📊 Subject breakdown — best model: {best['model']}")
        ranked = sorted(best["subject_results"], key=lambda x: x["accuracy"], reverse=True)
        print("  Top 5 subjects:")
        for i, s in enumerate(ranked[:5], 1):
            print(f"    {i}. {s['subject']:<35} {s['accuracy']:.2f}%")
        print("  Bottom 5 subjects:")
        for i, s in enumerate(ranked[-5:], 1):
            print(f"    {i}. {s['subject']:<35} {s['accuracy']:.2f}%")

    if in_colab:
        print("\n" + "=" * 70)
        print("💾 Download results from Colab:")
        print("=" * 70)
        print("  from google.colab import files")
        print(f"  files.download('{output_file}')")

    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-model MMLU eval — A100 optimized")
    parser.add_argument("--no-compile", dest="compile", action="store_false",
                        help="Disable torch.compile() (faster startup, slightly slower inference)")
    parser.add_argument("--print-questions", action="store_true",
                        help="Print each question, model prediction, and result")
    args = parser.parse_args()

    USE_TORCH_COMPILE = args.compile
    PRINT_QUESTIONS = args.print_questions

    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
