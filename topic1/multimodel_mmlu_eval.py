"""
Multi-Model MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates multiple small language models on the MMLU benchmark:
- Llama 3.2-1B-Instruct
- Qwen 2.5 0.5B-Instruct
- OLMo 2 1B

Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python multimodel_mmlu_eval.py

Set QUANTIZATION_BITS below to choose quantization level.
Set PRINT_QUESTIONS to True to see each question and answer.
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
# CONFIGURATION - Modify these settings
# ============================================================================

# Models to evaluate
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "allenai/OLMo-1B-0724-hf"  # Using OLMo 1B instead of the larger 7B model
]

# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = True  # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory (most memory efficient)
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory (balanced quality/memory)
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB memory (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled

QUANTIZATION_BITS = None  # Change to 4 or 8 to enable quantization

# Print detailed question/answer information
# If True, prints each question, the model's answer, and whether it's correct
PRINT_QUESTIONS = False  # Set to True to see detailed output

# Select 10 subjects for evaluation
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


def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"




def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")

    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")

    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠️  Could not check Hugging Face authentication")

    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models to evaluate: {len(MODELS)}")
    for model in MODELS:
        print(f"  - {model}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory: ~1.5 GB per model")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory: ~2.5 GB per model")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory: ~2.5 GB (FP16) per model")
        elif device == "mps":
            print(f"Expected memory: ~2.5 GB (FP16) per model")
        else:
            print(f"Expected memory: ~5 GB (FP32) per model")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")
    print(f"Print questions: {PRINT_QUESTIONS}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None

    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")

    return config


def load_model_and_tokenizer(model_name, device):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config()

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Non-quantized model loading
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
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Model license not accepted - Visit the model page on Hugging Face")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def unload_model(model, tokenizer):
    """Unload model to free memory"""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✓ Model unloaded and memory cleared")


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


class TimingInfo:
    """Track timing information for model evaluation"""
    def __init__(self):
        self.real_time = 0.0
        self.cpu_time = 0.0
        self.gpu_time = 0.0
        self.num_questions = 0

    def start(self):
        """Start timing"""
        self.start_real = time.time()
        self.start_cpu = time.process_time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_gpu_event = torch.cuda.Event(enable_timing=True)
            self.end_gpu_event = torch.cuda.Event(enable_timing=True)
            self.start_gpu_event.record()

    def stop(self):
        """Stop timing"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.end_gpu_event.record()
            torch.cuda.synchronize()
            self.gpu_time += self.start_gpu_event.elapsed_time(self.end_gpu_event) / 1000.0  # Convert ms to s

        self.real_time += time.time() - self.start_real
        self.cpu_time += time.process_time() - self.start_cpu
        self.num_questions += 1

    def get_summary(self):
        """Get timing summary"""
        return {
            "total_real_time_seconds": self.real_time,
            "total_cpu_time_seconds": self.cpu_time,
            "total_gpu_time_seconds": self.gpu_time if torch.cuda.is_available() else 0.0,
            "avg_real_time_per_question": self.real_time / self.num_questions if self.num_questions > 0 else 0.0,
            "avg_cpu_time_per_question": self.cpu_time / self.num_questions if self.num_questions > 0 else 0.0,
            "avg_gpu_time_per_question": (self.gpu_time / self.num_questions) if (self.num_questions > 0 and torch.cuda.is_available()) else 0.0,
            "num_questions": self.num_questions
        }


def get_model_prediction(model, tokenizer, prompt, timing_info):
    """Get model's prediction for multiple-choice question"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    timing_info.start()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )

    timing_info.stop()

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


def evaluate_subject(model, tokenizer, subject, model_name):
    """Evaluate model on a specific MMLU subject"""
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
    timing_info = TimingInfo()

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction(model, tokenizer, prompt, timing_info)

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Print question details if enabled
        if PRINT_QUESTIONS:
            print(f"\n--- Question {total} ---")
            print(f"Q: {question}")
            for i, choice in enumerate(choices):
                label = ["A", "B", "C", "D"][i]
                marker = " <--" if label == correct_answer else ""
                print(f"  {label}. {choice}{marker}")
            print(f"Model Answer: {predicted_answer}")
            print(f"Correct Answer: {correct_answer}")
            print(f"Result: {'✓ CORRECT' if is_correct else '✗ WRONG'}")

    accuracy = (correct / total * 100) if total > 0 else 0
    timing_summary = timing_info.get_summary()

    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")
    print(f"  Real time: {timing_summary['total_real_time_seconds']:.2f}s")
    print(f"  CPU time: {timing_summary['total_cpu_time_seconds']:.2f}s")
    if torch.cuda.is_available():
        print(f"  GPU time: {timing_summary['total_gpu_time_seconds']:.2f}s")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "timing": timing_summary
    }


def evaluate_model(model_name, device):
    """Evaluate a single model on all subjects"""
    print(f"\n{'#'*70}")
    print(f"# EVALUATING MODEL: {model_name}")
    print(f"{'#'*70}\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Evaluate on all subjects
    results = []
    total_correct = 0
    total_questions = 0
    overall_timing = TimingInfo()

    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")

    start_time = time.time()
    start_cpu_time = time.process_time()

    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject(model, tokenizer, subject, model_name)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]

            # Aggregate timing
            overall_timing.real_time += result["timing"]["total_real_time_seconds"]
            overall_timing.cpu_time += result["timing"]["total_cpu_time_seconds"]
            overall_timing.gpu_time += result["timing"]["total_gpu_time_seconds"]
            overall_timing.num_questions += result["timing"]["num_questions"]

    end_time = time.time()
    end_cpu_time = time.process_time()

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    # Unload model to free memory
    unload_model(model, tokenizer)

    return {
        "model": model_name,
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "subject_results": results,
        "timing": overall_timing.get_summary(),
        "wall_clock_time": end_time - start_time,
        "total_cpu_time": end_cpu_time - start_cpu_time
    }


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")

    # Check environment
    in_colab, device = check_environment()

    # Evaluate all models
    all_results = []

    for model_name in MODELS:
        try:
            result = evaluate_model(model_name, device)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Error evaluating model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparative summary
    print("\n" + "="*70)
    print("COMPARATIVE EVALUATION SUMMARY")
    print("="*70)

    for result in all_results:
        print(f"\n{result['model']}")
        print("-" * 70)
        print(f"  Overall Accuracy: {result['overall_accuracy']:.2f}%")
        print(f"  Total Correct: {result['total_correct']}/{result['total_questions']}")
        print(f"  Timing:")
        print(f"    Wall clock time: {result['wall_clock_time']:.2f}s ({result['wall_clock_time']/60:.2f} min)")
        print(f"    Total CPU time: {result['total_cpu_time']:.2f}s")
        print(f"    Total real time (inference): {result['timing']['total_real_time_seconds']:.2f}s")
        print(f"    Total CPU time (inference): {result['timing']['total_cpu_time_seconds']:.2f}s")
        if torch.cuda.is_available():
            print(f"    Total GPU time (inference): {result['timing']['total_gpu_time_seconds']:.2f}s")
        print(f"    Avg time per question: {result['timing']['avg_real_time_per_question']:.4f}s")

    print("\n" + "="*70)
    print("ACCURACY RANKING")
    print("="*70)
    sorted_results = sorted(all_results, key=lambda x: x["overall_accuracy"], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i}. {result['model']}: {result['overall_accuracy']:.2f}%")

    print("\n" + "="*70)
    print("SPEED RANKING (by average time per question)")
    print("="*70)
    sorted_by_speed = sorted(all_results, key=lambda x: x["timing"]["avg_real_time_per_question"])
    for i, result in enumerate(sorted_by_speed, 1):
        print(f"{i}. {result['model']}: {result['timing']['avg_real_time_per_question']:.4f}s per question")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
    output_file = f"multimodel_mmlu_results{quant_suffix}_{timestamp}.json"

    output_data = {
        "timestamp": timestamp,
        "device": str(device),
        "quantization_bits": QUANTIZATION_BITS,
        "subjects": MMLU_SUBJECTS,
        "num_subjects": len(MMLU_SUBJECTS),
        "results": all_results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Print per-subject breakdown for best model
    if len(all_results) > 0:
        best_model = sorted_results[0]
        print(f"\n📊 Subject Breakdown for Best Model ({best_model['model']}):")
        subject_results = sorted(best_model['subject_results'], key=lambda x: x["accuracy"], reverse=True)

        print("\nTop 5 Subjects:")
        for i, result in enumerate(subject_results[:5], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")

        print("\nBottom 5 Subjects:")
        for i, result in enumerate(subject_results[-5:], 1):
            print(f"  {i}. {result['subject']}: {result['accuracy']:.2f}%")

    # Colab-specific instructions
    if in_colab:
        print("\n" + "="*70)
        print("💾 To download results in Colab:")
        print("="*70)
        print(f"from google.colab import files")
        print(f"files.download('{output_file}')")

    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
