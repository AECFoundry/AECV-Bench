"""
Generate combined heatmaps for all benchmark results.

This script auto-discovers CSV files in the benchmark results directory
and generates combined accuracy and MAPE heatmaps for all models.
"""
import os
import glob
from src.benchmark.visualizer import plot_all_models_comparison


def extract_model_name(csv_path: str) -> str:
    """Extract a readable model name from CSV filename."""
    filename = os.path.splitext(os.path.basename(csv_path))[0]

    # Direct mapping for clean display names
    name_mapping = {
        "gemini_3_pro_preview": "Gemini 3 Pro",
        "openai_gpt_52": "OpenAI GPT 5.2",
        "claude_opus_45": "Claude Opus 4.5",
        "glm_46v": "GLM 4.6v",
        "qwen3_vl_8b_instruct": "Qwen3 VL 8B Instruct",
        "mistral_large_2512": "Mistral Large 3",
        "grok_41_fast": "Grok 4.1 Fast",
        "nvidia_nemotron_nano_12b_v2_vl": "NVIDIA Nemotron 12B VL",
        "amazon_nova_2_lite_v1": "Amazon Nova 2 Lite",
        "cohere_command_a_vision": "Cohere Command A Vision",
    }

    return name_mapping.get(filename, filename.replace("_", " ").title())


def main():
    """Main execution function."""
    # Configuration
    results_dir = "benchmark_result_object_counting"
    output_dir = "results/heatmap_outputs"

    # Only include models used for QA LLM judge results
    allowed_models = [
        "gemini_3_pro_preview.csv",
        "openai_gpt_52.csv",
        "claude_opus_45.csv",
        "glm_46v.csv",
        "qwen3_vl_8b_instruct.csv",
        "mistral_large_2512.csv",
        "nvidia_nemotron_nano_12b_v2_vl.csv",
        "grok_41_fast.csv",
        "amazon_nova_2_lite_v1.csv",
        "cohere_command_a_vision.csv",
    ]

    # Filter to only allowed models
    csv_files = [
        os.path.join(results_dir, model)
        for model in allowed_models
        if os.path.exists(os.path.join(results_dir, model))
    ]

    if not csv_files:
        print(f"No CSV files found in {results_dir}/")
        print("Run the benchmark first: python run_object_counting_benchmark.py")
        return

    # Extract model names from filenames
    model_names = [extract_model_name(f) for f in csv_files]

    print("=" * 60)
    print("GENERATING COMBINED HEATMAPS")
    print("=" * 60)
    print(f"Found {len(csv_files)} model results")
    print(f"Output directory: {output_dir}\n")

    # Verify CSV files and show status
    print("CSV files:")
    for csv_file, model_name in zip(csv_files, model_names):
        with open(csv_file, 'r') as f:
            row_count = sum(1 for line in f) - 1  # Subtract header
        print(f"  [x] {model_name}: {csv_file} ({row_count} rows)")
    print()

    # Generate combined visualizations
    plot_all_models_comparison(
        csv_files=csv_files,
        model_names=model_names,
        output_dir=output_dir,
        accuracy_filename="all_models_accuracy_heatmap.png",
        mape_filename="all_models_mape_heatmap.png"
    )

    print("\n[SUCCESS] Heatmaps generated!")
    print(f"  - all_models_accuracy_heatmap.png (PNG + SVG)")
    print(f"  - all_models_mape_heatmap.png (PNG + SVG)")


if __name__ == "__main__":
    main()
