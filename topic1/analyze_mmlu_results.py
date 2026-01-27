"""
Analysis script for multi-model MMLU results
Creates visualizations and analyzes patterns in model mistakes
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_file = "multimodel_mmlu_results_full_20260120_110435.json"

with open(results_file, 'r') as f:
    data = json.load(f)

# Extract data
models = [r['model'].split('/')[-1] for r in data['results']]
subjects = data['subjects']

# Create figure directory
Path("mmlu_analysis_plots").mkdir(exist_ok=True)

# ============================================================================
# 1. Overall Accuracy Comparison
# ============================================================================
print("Creating accuracy comparison graph...")

overall_accuracies = [r['overall_accuracy'] for r in data['results']]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, overall_accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_xlabel('Model', fontsize=12)
ax.set_title('Overall MMLU Accuracy by Model', fontsize=14, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('mmlu_analysis_plots/1_overall_accuracy.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/1_overall_accuracy.png")

# ============================================================================
# 2. Subject-by-Subject Comparison
# ============================================================================
print("Creating subject comparison graph...")

# Extract subject accuracies for each model
subject_data = {}
for subject in subjects:
    subject_data[subject] = []
    for model_result in data['results']:
        for subj_result in model_result['subject_results']:
            if subj_result['subject'] == subject:
                subject_data[subject].append(subj_result['accuracy'])
                break

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(14, 8))
x = np.arange(len(subjects))
width = 0.25

for i, model in enumerate(models):
    accuracies = [subject_data[subj][i] for subj in subjects]
    offset = width * (i - 1)
    ax.bar(x + offset, accuracies, width, label=model,
           color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i], alpha=0.8)

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
# 3. Timing Comparison
# ============================================================================
print("Creating timing comparison graphs...")

# Average time per question
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

avg_real_times = [r['timing']['avg_real_time_per_question'] for r in data['results']]
avg_cpu_times = [r['timing']['avg_cpu_time_per_question'] for r in data['results']]

# Real time
bars1 = ax1.bar(models, avg_real_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_ylabel('Time (seconds)', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_title('Average Real Time per Question', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}s',
            ha='center', va='bottom', fontsize=9)
ax1.tick_params(axis='x', rotation=45)

# CPU time
bars2 = ax2.bar(models, avg_cpu_times, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_ylabel('Time (seconds)', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.set_title('Average CPU Time per Question', fontsize=12, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}s',
            ha='center', va='bottom', fontsize=9)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('mmlu_analysis_plots/3_timing_comparison.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/3_timing_comparison.png")

# ============================================================================
# 4. Accuracy vs Speed Tradeoff
# ============================================================================
print("Creating accuracy vs speed graph...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, (model, acc, time) in enumerate(zip(models, overall_accuracies, avg_real_times)):
    ax.scatter(time, acc, s=500, alpha=0.6, color=colors[i],
              edgecolors='black', linewidth=2, label=model)
    ax.annotate(model, (time, acc), xytext=(10, 10),
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

# Create matrix of accuracies
acc_matrix = np.zeros((len(models), len(subjects)))
for i, model_result in enumerate(data['results']):
    for j, subject in enumerate(subjects):
        for subj_result in model_result['subject_results']:
            if subj_result['subject'] == subject:
                acc_matrix[i, j] = subj_result['accuracy']
                break

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(acc_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

# Set ticks and labels
ax.set_xticks(np.arange(len(subjects)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(subjects, rotation=45, ha='right')
ax.set_yticklabels(models)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Accuracy (%)', rotation=270, labelpad=20)

# Add text annotations
for i in range(len(models)):
    for j in range(len(subjects)):
        text = ax.text(j, i, f'{acc_matrix[i, j]:.1f}',
                      ha="center", va="center", color="black", fontsize=8)

ax.set_title('Subject Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('mmlu_analysis_plots/5_performance_heatmap.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/5_performance_heatmap.png")

# ============================================================================
# 6. Total Questions Correct/Wrong
# ============================================================================
print("Creating correct/wrong breakdown...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (model_result, model, ax) in enumerate(zip(data['results'], models, axes)):
    correct = model_result['total_correct']
    wrong = model_result['total_questions'] - correct

    colors_pie = ['#4CAF50', '#F44336']
    wedges, texts, autotexts = ax.pie([correct, wrong],
                                        labels=['Correct', 'Wrong'],
                                        autopct='%1.1f%%',
                                        colors=colors_pie,
                                        startangle=90)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

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

fig, ax = plt.subplots(figsize=(12, 8))

# For each subject, rank the models
rankings = {subject: [] for subject in subjects}
for subject in subjects:
    subject_accs = subject_data[subject]
    # Get indices sorted by accuracy (descending)
    sorted_indices = np.argsort(subject_accs)[::-1]
    for rank, idx in enumerate(sorted_indices):
        rankings[subject].append((models[idx], subject_accs[idx], rank + 1))

# Create a visual ranking
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
    bars = ax.barh(y_pos + offset, ranks, width, label=model,
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'][i], alpha=0.8)

    # Add rank numbers
    for j, (bar, rank) in enumerate(zip(bars, ranks)):
        ax.text(rank + 0.1, bar.get_y() + bar.get_height()/2,
               f'{rank}', va='center', fontsize=8, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(subjects)
ax.set_xlabel('Rank (1 = Best)', fontsize=12)
ax.set_title('Model Rankings by Subject', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.set_xlim(0, 4)
ax.invert_xaxis()  # So rank 1 is on the left
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('mmlu_analysis_plots/7_subject_rankings.png', dpi=300, bbox_inches='tight')
print("  Saved: mmlu_analysis_plots/7_subject_rankings.png")

# ============================================================================
# Summary Statistics
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\nBest Overall Model: {models[np.argmax(overall_accuracies)]} ({max(overall_accuracies):.2f}%)")
print(f"Fastest Model: {models[np.argmin(avg_real_times)]} ({min(avg_real_times):.3f}s per question)")

print("\nSubject Difficulty (across all models):")
subject_avg_accs = []
for subject in subjects:
    avg_acc = np.mean(subject_data[subject])
    subject_avg_accs.append((subject, avg_acc))
subject_avg_accs.sort(key=lambda x: x[1], reverse=True)

print("\n  Easiest subjects:")
for subject, acc in subject_avg_accs[:3]:
    print(f"    {subject}: {acc:.2f}%")

print("\n  Hardest subjects:")
for subject, acc in subject_avg_accs[-3:]:
    print(f"    {subject}: {acc:.2f}%")

print("\nModel Strengths:")
for i, model in enumerate(models):
    best_subjects = []
    for subject in subjects:
        accs = subject_data[subject]
        if accs[i] == max(accs):
            best_subjects.append(subject)
    if best_subjects:
        print(f"  {model} excels at: {', '.join(best_subjects)}")

print("\n" + "="*70)
print("✓ All graphs saved to mmlu_analysis_plots/")
print("="*70)
