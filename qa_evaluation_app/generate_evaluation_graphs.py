"""
Generate evaluation graphs from the QA evaluation JSON file.

This script reads the evaluation results and creates:
1. Overall model performance comparison bar chart
2. Performance by QA type grouped bar chart
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Official model names mapping
OFFICIAL_MODEL_NAMES = {
    'claude_opus_4_5': 'Claude Opus 4.5',
    'claude_opus_45': 'Claude Opus 4.5',
    'gemini_3_pro_preview': 'Gemini 3',
    'openai_gpt_4_vision': 'GPT-4o',
    'grok_41_fast': 'Grok 4.1 Fast',
    'qwen3_vl_8b_instruct': 'Qwen3 VL 8B',
    'glm_46v': 'GLM-4.6V',
    'mistral_large_2512': 'Mistral Large 3',
    'openai_gpt_52': 'GPT-5.2',
    'amazon_nova_2_lite_v1': 'Amazon Nova 2 Lite',
    'cohere_command_a_vision': 'Cohere Command A',
    'nvidia_nemotron_nano_12b_v2_vl': 'Nvidia Nemotron 12B',
}

# Standard QA type names mapping
QA_TYPE_NAMES = {
    'ocr_qa': 'OCR',
    'spatial_qa': 'Spatial',
    'counting_qa': 'Counting',
    'comparison_qa': 'Comparison',
}

# Model display order
MODEL_DISPLAY_ORDER = [
    'gemini_3_pro_preview',
    'openai_gpt_52',
    'claude_opus_45',
    'mistral_large_2512',
    'grok_41_fast',
    'openai_gpt_4_vision',
    'qwen3_vl_8b_instruct',
    'glm_46v',
    'amazon_nova_2_lite_v1',
    'cohere_command_a_vision',
    'nvidia_nemotron_nano_12b_v2_vl',
]

# QA type order and colors
QA_TYPE_ORDER = ['spatial_qa', 'ocr_qa', 'comparison_qa', 'counting_qa']
QA_TYPE_COLORS = {
    'spatial_qa': '#E8956A',      # Orange
    'ocr_qa': '#6A9BE8',           # Blue
    'comparison_qa': '#6AE89B',    # Green
    'counting_qa': '#B86AE8',      # Purple
}


def load_evaluations(json_path: str) -> dict:
    """Load evaluation results from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_statistics(evaluations: list) -> dict:
    """
    Calculate accuracy statistics per model and QA type.

    Returns:
        Dictionary with overall and per-QA-type statistics
    """
    # Initialize counters
    model_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    model_qa_type_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    for eval_item in evaluations:
        qa_type = eval_item.get('qa_type', 'unknown')
        model_evaluations = eval_item.get('model_evaluations', {})

        for model_name, eval_data in model_evaluations.items():
            is_correct = eval_data.get('is_correct', False)

            # Overall stats
            model_stats[model_name]['total'] += 1
            if is_correct:
                model_stats[model_name]['correct'] += 1

            # Per QA type stats
            model_qa_type_stats[model_name][qa_type]['total'] += 1
            if is_correct:
                model_qa_type_stats[model_name][qa_type]['correct'] += 1

    # Calculate accuracies
    results = {
        'overall': {},
        'by_qa_type': {}
    }

    for model_name, stats in model_stats.items():
        if stats['total'] > 0:
            results['overall'][model_name] = stats['correct'] / stats['total']

    for model_name, qa_types in model_qa_type_stats.items():
        results['by_qa_type'][model_name] = {}
        for qa_type, stats in qa_types.items():
            if stats['total'] > 0:
                results['by_qa_type'][model_name][qa_type] = stats['correct'] / stats['total']

    return results


def sort_models(models: list) -> list:
    """Sort models according to MODEL_DISPLAY_ORDER."""
    def get_sort_key(model_name):
        try:
            return MODEL_DISPLAY_ORDER.index(model_name)
        except ValueError:
            return len(MODEL_DISPLAY_ORDER) + 1000

    return sorted(models, key=get_sort_key)


def create_overall_comparison_chart(stats: dict, output_path: str):
    """Create overall model performance comparison bar chart."""
    overall = stats['overall']

    # Sort models by display order
    models = sort_models(list(overall.keys()))
    scores = [overall[m] for m in models]
    display_names = [OFFICIAL_MODEL_NAMES.get(m, m.replace('_', ' ').title()) for m in models]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = range(len(models))
    width = 0.6

    bars = ax.bar(x, scores, width, color='steelblue')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison on Document QA', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    svg_path = output_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {svg_path}")


def create_qa_type_breakdown_chart(stats: dict, output_path: str):
    """Create performance by QA type grouped bar chart."""
    by_qa_type = stats['by_qa_type']

    # Get all models and sort by display order
    models = sort_models(list(by_qa_type.keys()))
    display_names = [OFFICIAL_MODEL_NAMES.get(m, m.replace('_', ' ').title()) for m in models]

    # Get available QA types in order
    all_qa_types = set()
    for model_data in by_qa_type.values():
        all_qa_types.update(model_data.keys())
    ordered_qa_types = [qt for qt in QA_TYPE_ORDER if qt in all_qa_types]

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(models))
    width = 0.8 / len(ordered_qa_types)

    for i, qa_type in enumerate(ordered_qa_types):
        scores = []
        for model in models:
            score = by_qa_type.get(model, {}).get(qa_type, 0.0)
            scores.append(score)

        offset = (i - len(ordered_qa_types)/2) * width + width/2
        qa_type_label = QA_TYPE_NAMES.get(qa_type, qa_type.replace('_', ' ').title())
        color = QA_TYPE_COLORS.get(qa_type, '#888888')
        ax.bar(x + offset, scores, width, label=qa_type_label, color=color)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance by QA Type', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylim([0, 1])
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    svg_path = output_path.replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Saved: {svg_path}")


def generate_summary_report(stats: dict, output_path: str):
    """Generate a text summary report."""
    overall = stats['overall']
    by_qa_type = stats['by_qa_type']

    # Sort by overall score (descending)
    sorted_models = sorted(overall.items(), key=lambda x: x[1], reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("QA EVALUATION SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write("MODEL RANKINGS (by Overall Accuracy)\n")
        f.write("-" * 80 + "\n")
        for rank, (model, score) in enumerate(sorted_models, 1):
            display_name = OFFICIAL_MODEL_NAMES.get(model, model.replace('_', ' ').title())
            f.write(f"{rank:2d}. {display_name:30s} | Accuracy: {score:.4f} ({score*100:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE BY QA TYPE\n")
        f.write("=" * 80 + "\n\n")

        for model, score in sorted_models:
            display_name = OFFICIAL_MODEL_NAMES.get(model, model.replace('_', ' ').title())
            f.write(f"\n{display_name}\n")
            f.write("-" * 40 + "\n")

            qa_scores = by_qa_type.get(model, {})
            for qa_type in QA_TYPE_ORDER:
                if qa_type in qa_scores:
                    qa_label = QA_TYPE_NAMES.get(qa_type, qa_type)
                    f.write(f"  {qa_label:20s}: {qa_scores[qa_type]:.4f} ({qa_scores[qa_type]*100:.1f}%)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")

    print(f"Saved: {output_path}")


def main():
    # Find the most recent evaluation JSON file
    script_dir = Path(__file__).parent
    evaluations_dir = script_dir / "evaluations"

    if not evaluations_dir.exists():
        print(f"Error: Evaluations directory not found: {evaluations_dir}")
        return

    # Find the latest JSON file
    json_files = list(evaluations_dir.glob("qa_evaluations_*.json"))
    if not json_files:
        print("Error: No evaluation JSON files found")
        return

    # Sort by modification time and get the latest
    latest_json = max(json_files, key=lambda f: f.stat().st_mtime)
    print(f"Loading evaluations from: {latest_json}")

    # Load and process evaluations
    data = load_evaluations(str(latest_json))
    evaluations = data.get('evaluations', [])
    print(f"Loaded {len(evaluations)} evaluation items")

    # Calculate statistics
    stats = calculate_statistics(evaluations)
    print(f"Found {len(stats['overall'])} models")

    # Create output directory for graphs
    output_dir = script_dir

    # Generate graphs
    create_overall_comparison_chart(
        stats,
        str(output_dir / "overall_comparison.png")
    )

    create_qa_type_breakdown_chart(
        stats,
        str(output_dir / "qa_type_breakdown.png")
    )

    # Generate summary report
    generate_summary_report(
        stats,
        str(output_dir / "evaluation_summary.txt")
    )

    print("\nGraph generation complete!")
    print(f"Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
