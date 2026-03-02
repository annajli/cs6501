"""
MMLU Evaluation - Business Ethics (Ollama Version)
"""

import requests
from datasets import load_dataset
from datetime import datetime


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer with only the letter (A, B, C, or D):"
    return prompt


def get_model_prediction_ollama(prompt):
    """Get model's prediction from Ollama"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 1
                }
            },
            timeout=30
        )

        answer = response.json()['response'].strip().upper()

        # Extract just the letter
        if answer not in ["A", "B", "C", "D"]:
            for char in answer:
                if char in ["A", "B", "C", "D"]:
                    answer = char
                    break
            else:
                answer = "A"  # Default fallback

        return answer

    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return "A"  # Default fallback


def evaluate_business_ethics():
    """Evaluate on business ethics questions from the MMLU dataset"""
    subject = "business_ethics"

    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")

    dataset = load_dataset("cais/mmlu", subject, split="test")

    correct = 0
    total = 0

    for i, example in enumerate(dataset, 1):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)
        predicted_answer = get_model_prediction_ollama(prompt)

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else "✗"
        print(f"Q{i}: {status} (Predicted: {predicted_answer}, Correct: {correct_answer})")

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"\nResult: {correct}/{total} correct = {accuracy:.2f}%")
    print(f"{'='*70}\n")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


if __name__ == "__main__":
    start_time = datetime.now()
    result = evaluate_business_ethics()
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"Duration: {duration:.2f} seconds")
