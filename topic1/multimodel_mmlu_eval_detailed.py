"""
Multi-Model MMLU Evaluation Script with Question-Level Tracking

This version tracks individual question responses to analyze:
- Which questions all models get wrong
- Which questions all models get right
- Patterns in mistakes across models
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import time
import gc

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "allenai/OLMo-1B-0724-hf"
]

USE_GPU = True
MAX_NEW_TOKENS = 1
QUANTIZATION_BITS = None
PRINT_QUESTIONS = False

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
# HELPER FUNCTIONS (Same as before)
# ============================================================================

def detect_device():
    if not USE_GPU:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"
        if is_apple_arm:
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                sys.exit(1)
            return "mps"
    return "cpu"


def check_environment():
    global QUANTIZATION_BITS
    print("="*70)
    print("Environment Check")
    print("="*70)

    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    device = detect_device()

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
    else:
        print("⚠️  No GPU detected - running on CPU")

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")

    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models to evaluate: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}")
    print(f"Device: {device}")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")
    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    if QUANTIZATION_BITS is None:
        return None
    if QUANTIZATION_BITS == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif QUANTIZATION_BITS == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
    return None


def load_model_and_tokenizer(model_name, device):
    print(f"\nLoading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")

    quant_config = get_quantization_config()

    if quant_config is not None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif device == "mps":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            model = model.to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = model.to(device)

    model.eval()
    print("✓ Model loaded successfully!")
    return model, tokenizer


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Model unloaded and memory cleared")


def format_mmlu_prompt(question, choices):
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

    answer = generated_text.strip()[:1].upper()

    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"

    return answer


# ============================================================================
# EVALUATION WITH QUESTION TRACKING
# ============================================================================

def evaluate_subject_detailed(model, tokenizer, subject, model_name):
    """Evaluate model on a subject and track individual questions"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0
    question_results = []

    for idx, example in enumerate(tqdm(dataset, desc=f"Testing {subject}", leave=True)):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(model, tokenizer, prompt)

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Store question-level result
        question_results.append({
            "question_id": idx,
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "question_results": question_results
    }


def evaluate_model_detailed(model_name, device):
    """Evaluate a single model on all subjects with question tracking"""
    print(f"\n{'#'*70}")
    print(f"# EVALUATING MODEL: {model_name}")
    print(f"{'#'*70}\n")

    model, tokenizer = load_model_and_tokenizer(model_name, device)

    results = []
    total_correct = 0
    total_questions = 0

    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject_detailed(model, tokenizer, subject, model_name)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]

    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    unload_model(model, tokenizer)

    return {
        "model": model_name,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results
    }


def main():
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation (Detailed)")
    print("="*70 + "\n")

    in_colab, device = check_environment()

    # Evaluate all models
    all_results = []

    for model_name in MODELS:
        try:
            result = evaluate_model_detailed(model_name, device)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"multimodel_mmlu_detailed_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "device": str(device),
        "quantization_bits": QUANTIZATION_BITS,
        "subjects": MMLU_SUBJECTS,
        "results": all_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Detailed results saved to: {output_file}")

    # Analyze question overlap
    print("\n" + "="*70)
    print("ANALYZING QUESTION PATTERNS")
    print("="*70)

    analyze_question_overlap(all_results)

    return output_file


def analyze_question_overlap(all_results):
    """Analyze which questions all models get right/wrong"""

    if len(all_results) < 2:
        print("Need at least 2 models to analyze overlap")
        return

    for subject in MMLU_SUBJECTS:
        # Get question results for this subject from all models
        subject_results_by_model = []
        for model_result in all_results:
            for subj_result in model_result['subject_results']:
                if subj_result['subject'] == subject:
                    subject_results_by_model.append(subj_result)
                    break

        if len(subject_results_by_model) < len(all_results):
            continue

        num_questions = subject_results_by_model[0]['total']

        # Track questions all models get right/wrong
        all_correct = []
        all_wrong = []
        mixed = []

        for q_idx in range(num_questions):
            results = [sr['question_results'][q_idx]['is_correct']
                      for sr in subject_results_by_model]

            if all(results):
                all_correct.append(q_idx)
            elif not any(results):
                all_wrong.append(q_idx)
            else:
                mixed.append(q_idx)

        print(f"\n{subject}:")
        print(f"  Total questions: {num_questions}")
        print(f"  All models correct: {len(all_correct)} ({len(all_correct)/num_questions*100:.1f}%)")
        print(f"  All models wrong: {len(all_wrong)} ({len(all_wrong)/num_questions*100:.1f}%)")
        print(f"  Mixed results: {len(mixed)} ({len(mixed)/num_questions*100:.1f}%)")

        # Show example of question all models got wrong
        if all_wrong:
            q_idx = all_wrong[0]
            q_data = subject_results_by_model[0]['question_results'][q_idx]
            print(f"\n  Example question ALL models got wrong:")
            print(f"    Q: {q_data['question'][:100]}...")
            print(f"    Correct: {q_data['correct_answer']}")
            print(f"    Models answered:", end=" ")
            for sr in subject_results_by_model:
                print(sr['question_results'][q_idx]['predicted_answer'], end=" ")
            print()


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
