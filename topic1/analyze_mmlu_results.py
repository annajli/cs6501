"""
Analysis script for multi-model MMLU results.
Creates visualizations and analyzes patterns in model mistakes.

Usage:
    python analyze_mmlu_results.py <results_file.json>
"""

import sys
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ============================================================================
# Load results
# ============================================================================

if len(sys.argv) < 2:
    print("Usage: python analyze_mmlu_results.py <results_file.json>")
    sys.exit(1)

results_file = sys.argv[1]
with open(results_file, 'r') as f:
    data = json.load(f)

models = [r['model'].split('/')[-1] for r in data['results']]
subjects = data['subjects']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

has_timing = 'timing' in data['results'][0]
has_questions = (
    data['results'] and
    data['results'][0].get('subject_results') and
    'question_results' in data['results'][0]['subject_results'][0]
)

Path("mmlu_analysis_plots").mkdir(exist_ok=True)

# ============================================================================
# 1. Overall Accuracy Comparison
# ============================================================================
print("Creating accuracy comparison graph...")

overall_accuracies = [r['overall_accuracy'] for r in data['results']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, overall_accuracies, color=colors)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Overall MMLU Accuracy by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('mmlu_analysis_plots/1_overall_accuracy.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/1_overall_accuracy.png")

# ============================================================================
# 2. Subject-by-Subject Comparison
# ============================================================================
print("Creating subject comparison graph...")

subject_data = {}
for subject in subjects:
    subject_data[subject] = []
    for model_result in data['results']:
        for subj_result in model_result['subject_results']:
            if subj_result['subject'] == subject:
                subject_data[subject].append(subj_result['accuracy'])
                break

fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(subjects))
width = 0.25

for i, model in enumerate(models):
    accuracies = [subject_data[subj][i] for subj in subjects]
    offset = width * (i - 1)
    ax.bar(x + offset, accuracies, width, label=model, color=colors[i], alpha=0.8)

ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Subject', fontsize=12)
ax.set_title('Model Performance by Subject', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(subjects, rotation=45, ha='right')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('mmlu_analysis_plots/2_subject_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/2_subject_comparison.png")

# ============================================================================
# 3. Timing Comparison (skip if no timing data)
# ============================================================================
if has_timing:
    print("Creating timing comparison graphs...")

    avg_real_times = [r['timing']['avg_real_time_per_question'] for r in data['results']]
    avg_cpu_times  = [r['timing']['avg_cpu_time_per_question']  for r in data['results']]
    avg_gpu_times  = [r['timing'].get('avg_gpu_time_per_question', 0) for r in data['results']]
    has_gpu_time   = any(t > 0 for t in avg_gpu_times)

    ncols = 3 if has_gpu_time else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    def _bar(ax, values, title, fmt='.3f'):
        b = ax.bar(models, values, color=colors)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        for bar in b:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:{fmt}}s', ha='center', va='bottom', fontsize=9)
        ax.tick_params(axis='x', rotation=45)

    _bar(axes[0], avg_real_times, 'Avg Real Time per Question')
    _bar(axes[1], avg_cpu_times,  'Avg CPU Time per Question')
    if has_gpu_time:
        _bar(axes[2], avg_gpu_times, 'Avg GPU Time per Question')

    plt.tight_layout()
    plt.savefig('mmlu_analysis_plots/3_timing_comparison.png', dpi=300, bbox_inches='tight')
    print("  Saved: mmlu_analysis_plots/3_timing_comparison.png")

    # ============================================================================
    # 4. Accuracy vs Speed Tradeoff
    # ============================================================================
    print("Creating accuracy vs speed graph...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (model, acc, t) in enumerate(zip(models, overall_accuracies, avg_real_times)):
        ax.scatter(t, acc, s=500, alpha=0.6, color=colors[i],
                   edgecolors='black', linewidth=2, label=model)
        ax.annotate(model, (t, acc), xytext=(10, 10),
                    textcoords='offset points', fontsize=10, fontweight='bold')

    ax.set_xlabel('Average Time per Question (seconds)', fontsize=12)
    ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Speed Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig('mmlu_analysis_plots/4_accuracy_vs_speed.png', dpi=300, bbox_inches='tight')
    print("  Saved: mmlu_analysis_plots/4_accuracy_vs_speed.png")

# ============================================================================
# 5. Heatmap of Subject Performance
# ============================================================================
print("Creating performance heatmap...")

acc_matrix = np.zeros((len(models), len(subjects)))
for i, model_result in enumerate(data['results']):
    for j, subject in enumerate(subjects):
        for subj_result in model_result['subject_results']:
            if subj_result['subject'] == subject:
                acc_matrix[i, j] = subj_result['accuracy']
                break

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

ax.set_xticks(np.arange(len(subjects)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(subjects, rotation=45, ha='right')
ax.set_yticklabels(models)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)

for i in range(len(models)):
    for j in range(len(subjects)):
        ax.text(j, i, f'{acc_matrix[i, j]:.1f}',
                ha="center", va="center", color="black", fontsize=8)

ax.set_title('Subject Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mmlu_analysis_plots/5_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/5_performance_heatmap.png")

# ============================================================================
# 6. Correct/Wrong Breakdown (pie charts)
# ============================================================================
print("Creating correct/wrong breakdown...")

fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
if len(models) == 1:
    axes = [axes]

for i, (model_result, model, ax) in enumerate(zip(data['results'], models, axes)):
    correct = model_result['total_correct']
    wrong   = model_result['total_questions'] - correct
    wedges, texts, autotexts = ax.pie(
        [correct, wrong], labels=['Correct', 'Wrong'],
        autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    for at in autotexts:
        at.set_color('white')
        at.set_fontsize(11)
        at.set_fontweight('bold')
    ax.set_title(f'{model}\n{correct}/{model_result["total_questions"]} correct',
                 fontsize=11, fontweight='bold')

plt.suptitle('Correct vs Wrong Answers by Model', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('mmlu_analysis_plots/6_correct_wrong_breakdown.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/6_correct_wrong_breakdown.png")

# ============================================================================
# 7. Performance Rankings by Subject
# ============================================================================
print("Creating subject rankings...")

rankings = {subject: [] for subject in subjects}
for subject in subjects:
    subject_accs = subject_data[subject]
    sorted_indices = np.argsort(subject_accs)[::-1]
    for rank, idx in enumerate(sorted_indices):
        rankings[subject].append((models[idx], subject_accs[idx], rank + 1))

fig, ax = plt.subplots(figsize=(12, 8))
y_pos = np.arange(len(subjects))
width = 0.25

for i, model in enumerate(models):
    ranks = []
    for subject in subjects:
        for model_name, acc, rank in rankings[subject]:
            if model_name == model:
                ranks.append(rank)
                break
    offset = width * (i - 1)
    bars = ax.barh(y_pos + offset, ranks, width, label=model, color=colors[i], alpha=0.8)
    for j, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(rank + 0.1, bar.get_y() + bar.get_height()/2,
                f'{rank}', va='center', fontsize=8, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(subjects)
ax.set_xlabel('Rank (1 = Best)', fontsize=12)
ax.set_title('Model Rankings by Subject', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0, 4)
ax.invert_xaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('mmlu_analysis_plots/7_subject_rankings.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/7_subject_rankings.png")

# ============================================================================
# Question-overlap plots (only when question_results are available)
# ============================================================================
if has_questions:

    # Build per-subject overlap counts
    overlap_data = {}
    for subject in subjects:
        subject_results_by_model = []
        for model_result in data['results']:
            for subj_result in model_result['subject_results']:
                if subj_result['subject'] == subject:
                    subject_results_by_model.append(subj_result)
                    break
        if len(subject_results_by_model) < len(data['results']):
            continue

        num_q = subject_results_by_model[0]['total']
        all_correct = all_wrong = mixed = 0
        for q_idx in range(num_q):
            results = [sr['question_results'][q_idx]['is_correct']
                       for sr in subject_results_by_model]
            if all(results):
                all_correct += 1
            elif not any(results):
                all_wrong += 1
            else:
                mixed += 1
        overlap_data[subject] = (all_correct, mixed, all_wrong, num_q)

    # ============================================================================
    # 8. Question Overlap Stacked Bar — how many questions do all models agree on?
    # ============================================================================
    print("Creating question overlap graph...")

    subj_labels  = list(overlap_data.keys())
    all_correct_pct = [overlap_data[s][0] / overlap_data[s][3] * 100 for s in subj_labels]
    mixed_pct       = [overlap_data[s][1] / overlap_data[s][3] * 100 for s in subj_labels]
    all_wrong_pct   = [overlap_data[s][2] / overlap_data[s][3] * 100 for s in subj_labels]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(subj_labels))
    b1 = ax.bar(x, all_correct_pct, label='All models correct', color='#4CAF50', alpha=0.85)
    b2 = ax.bar(x, mixed_pct,       bottom=all_correct_pct,     label='Mixed results',       color='#FFC107', alpha=0.85)
    b3 = ax.bar(x, all_wrong_pct,
                bottom=[a + b for a, b in zip(all_correct_pct, mixed_pct)],
                label='All models wrong', color='#F44336', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, rotation=45, ha='right')
    ax.set_ylabel('Percentage of Questions (%)', fontsize=12)
    ax.set_title('Question-Level Agreement Across All Models per Subject', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('mmlu_analysis_plots/8_question_overlap.png', dpi=300, bbox_inches='tight')
    print("  Saved: mmlu_analysis_plots/8_question_overlap.png")

    # ============================================================================
    # 9. Pairwise Model Agreement Heatmap
    # ============================================================================
    print("Creating pairwise agreement heatmap...")

    n = len(models)
    agreement_matrix = np.zeros((n, n))

    for i, res_i in enumerate(data['results']):
        for j, res_j in enumerate(data['results']):
            total_q = 0
            agree_q = 0
            for subj in subjects:
                sr_i = next((s for s in res_i['subject_results'] if s['subject'] == subj), None)
                sr_j = next((s for s in res_j['subject_results'] if s['subject'] == subj), None)
                if sr_i is None or sr_j is None:
                    continue
                for qi, qj in zip(sr_i['question_results'], sr_j['question_results']):
                    if qi['predicted_answer'] == qj['predicted_answer']:
                        agree_q += 1
                    total_q += 1
            agreement_matrix[i, j] = agree_q / total_q * 100 if total_q else 0

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(agreement_matrix, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_yticklabels(models)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Answer Agreement (%)', rotation=270, labelpad=20)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{agreement_matrix[i, j]:.1f}%',
                    ha='center', va='center', fontsize=10,
                    color='white' if agreement_matrix[i, j] > 60 else 'black')

    ax.set_title('Pairwise Model Answer Agreement\n(% questions with same answer choice)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mmlu_analysis_plots/9_pairwise_agreement.png', dpi=300, bbox_inches='tight')
    print("  Saved: mmlu_analysis_plots/9_pairwise_agreement.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

subject_avg_accs = [(s, np.mean(subject_data[s])) for s in subjects]
subject_avg_accs.sort(key=lambda x: x[1], reverse=True)

if has_timing:
    avg_real_times = [r['timing']['avg_real_time_per_question'] for r in data['results']]
    print(f"\nBest Overall Model:  {models[np.argmax(overall_accuracies)]} ({max(overall_accuracies):.2f}%)")
    print(f"Fastest Model:       {models[np.argmin(avg_real_times)]} ({min(avg_real_times):.3f}s per question)")

print("\nSubject Difficulty (average across all models):")
print("  Easiest:")
for subject, acc in subject_avg_accs[:3]:
    print(f"    {subject}: {acc:.2f}%")
print("  Hardest:")
for subject, acc in subject_avg_accs[-3:]:
    print(f"    {subject}: {acc:.2f}%")

print("\nModel Strengths (subjects where each model ranks #1):")
for i, model in enumerate(models):
    best = [s for s in subjects if subject_data[s] and subject_data[s][i] == max(subject_data[s])]
    if best:
        print(f"  {model}: {', '.join(best)}")

if has_questions:
    print("\nQuestion Overlap Summary:")
    total_all_wrong = sum(overlap_data[s][2] for s in overlap_data)
    total_all_q = sum(overlap_data[s][3] for s in overlap_data)
    print(f"  Questions ALL models got wrong: {total_all_wrong}/{total_all_q} "
          f"({total_all_wrong/total_all_q*100:.1f}%)")
    total_all_right = sum(overlap_data[s][0] for s in overlap_data)
    print(f"  Questions ALL models got right: {total_all_right}/{total_all_q} "
          f"({total_all_right/total_all_q*100:.1f}%)")

print("\n" + "="*70)
print("✓ All graphs saved to mmlu_analysis_plots/")
print("="*70)
