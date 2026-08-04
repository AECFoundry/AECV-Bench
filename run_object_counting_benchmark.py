"""
Run object counting benchmark on vision-language models and generate combined visualization.

This unified script supports all VLM providers (OpenRouter, Cohere, Replicate) for
AEC drawings object counting tasks.

IMPORTANT: Only models that support vision/image input can be used here.
"""
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
from src.benchmark.processor import process_benchmark_floorplans
from src.analyzers.openrouter import analyze_floorplan, analyze_floorplan_prompt_based
from src.analyzers.cohere import analyze_floorplan_cohere
from src.analyzers.replicate import analyze_floorplan_replicate
from src.models.plan_elements import get_json_schema
from src.utils.config import require_api_key


@dataclass
class ModelConfig:
    """Configuration for a vision-language model to benchmark."""
    name: str
    model_id: str
    analyzer: callable
    provider: str = "openrouter"  # openrouter, cohere, replicate
    note: Optional[str] = None
    enabled: bool = False


class ModelRegistry:
    """Registry of available vision-language models for benchmarking."""

    def __init__(self):
        self.models = [
            # =================================================================
            # FLAGSHIP MULTIMODAL LLMS (OpenRouter)
            # =================================================================
            ModelConfig(
                name="Gemini 3 Pro Preview",
                model_id="google/gemini-3-pro-preview",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Latest flagship model, high-precision multimodal reasoning",
                enabled=False
            ),
            ModelConfig(
                name="Claude Opus 4.5",
                model_id="anthropic/claude-opus-4.5",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Most advanced Opus model with structured output support",
                enabled=False
            ),
            ModelConfig(
                name="Claude Sonnet 4.5",
                model_id="anthropic/claude-sonnet-4.5",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Most advanced Sonnet with structured output support",
                enabled=False
            ),
            ModelConfig(
                name="Claude Sonnet 4.6",
                model_id="anthropic/claude-sonnet-4.6",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Anthropic Claude Sonnet 4.6, latest Sonnet model",
                enabled=False
            ),
            ModelConfig(
                name="Claude Opus 4.6",
                model_id="anthropic/claude-opus-4.6",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Anthropic Claude Opus 4.6, most advanced Opus model",
                enabled=False
            ),
            ModelConfig(
                name="Gemini 3.1 Pro",
                model_id="google/gemini-3.1-pro-preview",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Gemini 3.1 Pro preview model",
                enabled=False
            ),
            ModelConfig(
                name="Qwen 3.5 Plus",
                model_id="qwen/qwen3.5-plus-02-15",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Qwen 3.5 Plus model from February 2025",
                enabled=False
            ),
            ModelConfig(
                name="Grok 4.1 Fast",
                model_id="x-ai/grok-4.1-fast",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Best agentic tool calling model, 2M context",
                enabled=False
            ),
            ModelConfig(
                name="Mistral Large 2512",
                model_id="mistralai/mistral-large-2512",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Mistral Large model from December 2025",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-5.1",
                model_id="openai/gpt-5.1",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="OpenAI GPT-5.1 model, balanced performance with dual modes",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-5.2",
                model_id="openai/gpt-5.2",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="OpenAI GPT-5.2 model, advanced multimodal capabilities",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-5.3",
                model_id="openai/gpt-5.3-chat",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="OpenAI GPT-5.3 Chat model",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-5.4",
                model_id="openai/gpt-5.4",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="OpenAI GPT-5.4 model",
                enabled=False
            ),
            ModelConfig(
                name="Amazon Nova 2 Lite v1",
                model_id="amazon/nova-2-lite-v1",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Amazon Nova 2 Lite multimodal language model",
                enabled=False
            ),

            # =================================================================
            # VISION-SPECIALIZED MODELS (OpenRouter)
            # =================================================================
            ModelConfig(
                name="Qwen3 VL 235B A22B Thinking",
                model_id="qwen/qwen3-vl-235b-a22b-thinking",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="235B Qwen3 thinking model - better reasoning, slower throughput",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-4 Vision",
                model_id="openai/gpt-4o",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="GPT-4o multimodal model (uses prompt-based JSON extraction)",
                enabled=False
            ),
            ModelConfig(
                name="Qwen3-VL 8B Instruct",
                model_id="qwen/qwen3-vl-8b-instruct",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="8B Qwen3 vision-language model - efficient and fast",
                enabled=False
            ),
            ModelConfig(
                name="Qwen3-VL 8B Thinking",
                model_id="qwen/qwen3-vl-8b-thinking",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="8B Qwen3 thinking model - better reasoning, slower throughput",
                enabled=False
            ),
            ModelConfig(
                name="GLM-4.5V",
                model_id="z-ai/glm-4.5v",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Z-AI vision-language model (uses prompt-based JSON extraction)",
                enabled=False
            ),
            ModelConfig(
                name="GLM-4.6V",
                model_id="z-ai/glm-4.6v",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Z-AI vision-language model - newer version",
                enabled=False
            ),
            ModelConfig(
                name="NVIDIA Nemotron Nano 12B V2 VL",
                model_id="nvidia/nemotron-nano-12b-v2-vl",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="NVIDIA Nemotron Nano 12B V2 vision-language model",
                enabled=False
            ),
            ModelConfig(
                name="Llama Nemotron Embed VL 1B V2",
                model_id="nvidia/llama-nemotron-embed-vl-1b-v2:free",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Nvidia Llama Nemotron Embed VL 1B V2 vision-language model (free)",
                enabled=False
            ),

            # =================================================================
            # NEW FRONTIER MODELS (added 2026-06, released since ~March 2026)
            # =================================================================
            ModelConfig(
                name="Claude Opus 4.8",
                model_id="anthropic/claude-opus-4.8",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Anthropic Claude Opus 4.8 - latest Opus flagship",
                enabled=True
            ),
            ModelConfig(
                name="Claude Opus 4.7",
                model_id="anthropic/claude-opus-4.7",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Anthropic Claude Opus 4.7",
                enabled=False
            ),
            ModelConfig(
                name="Claude Fable 5",
                model_id="anthropic/claude-fable-5",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Anthropic Claude Fable 5",
                enabled=False
            ),
            ModelConfig(
                name="OpenAI GPT-5.5",
                model_id="openai/gpt-5.5",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="OpenAI GPT-5.5 (practical frontier; pro variant excluded - too slow/expensive)",
                enabled=False
            ),
            ModelConfig(
                name="Gemini 3.5 Flash",
                model_id="google/gemini-3.5-flash",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Google Gemini 3.5 Flash - newest Gemini vision on OpenRouter",
                enabled=False
            ),
            ModelConfig(
                name="Grok 4.3",
                model_id="x-ai/grok-4.3",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="xAI Grok 4.3 - latest stable Grok",
                enabled=False
            ),
            ModelConfig(
                name="Kimi K2.6",
                model_id="moonshotai/kimi-k2.6",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Moonshot Kimi K2.6 - latest Kimi (prompt-based JSON)",
                enabled=False
            ),
            ModelConfig(
                name="MiniMax M3",
                model_id="minimax/minimax-m3",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="MiniMax M3 - latest MiniMax (prompt-based JSON)",
                enabled=False
            ),
            ModelConfig(
                name="Qwen 3.7 Plus",
                model_id="qwen/qwen3.7-plus",
                analyzer=analyze_floorplan,
                provider="openrouter",
                note="Qwen 3.7 Plus - newest Qwen vision",
                enabled=False
            ),
            ModelConfig(
                name="Gemma 4 31B IT",
                model_id="google/gemma-4-31b-it",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="Google Gemma 4 31B - open-weight (prompt-based JSON)",
                enabled=False
            ),
            ModelConfig(
                name="StepFun Step 3.7 Flash",
                model_id="stepfun/step-3.7-flash",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="StepFun Step 3.7 Flash - open VLM (prompt-based JSON)",
                enabled=False
            ),
            ModelConfig(
                name="NVIDIA Nemotron 3 Nano Omni 30B",
                model_id="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
                analyzer=analyze_floorplan_prompt_based,
                provider="openrouter",
                note="NVIDIA frontier VISION model on OpenRouter (Nemotron 3 Ultra is text-only)",
                enabled=False
            ),

            # =================================================================
            # COHERE MODELS
            # =================================================================
            ModelConfig(
                name="Cohere Command A Vision",
                model_id="command-a-vision-07-2025",
                analyzer=analyze_floorplan_cohere,
                provider="cohere",
                note="Cohere Command A Vision - multimodal model for document analysis",
                enabled=False
            ),

            # =================================================================
            # REPLICATE MODELS
            # =================================================================
            ModelConfig(
                name="DeepSeek VL2",
                model_id="deepseek-ai/deepseek-vl2:e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200",
                analyzer=analyze_floorplan_replicate,
                provider="replicate",
                note="DeepSeek VL2 vision-language model via Replicate API",
                enabled=False
            ),
        ]

    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models for benchmarking."""
        return [model for model in self.models if model.enabled]

    def enable_model(self, name: str):
        """Enable a model by name."""
        for model in self.models:
            if model.name == name:
                model.enabled = True
                break

    def disable_model(self, name: str):
        """Disable a model by name."""
        for model in self.models:
            if model.name == name:
                model.enabled = False
                break

    def list_models(self):
        """Print all available models."""
        print("\nAvailable models:")
        for model in self.models:
            status = "[x]" if model.enabled else "[ ]"
            print(f"  {status} {model.name} ({model.provider})")


class BenchmarkRunner:
    """Handles running vision model benchmarks with proper separation of concerns."""

    # Maps a provider to its (env var name, display name) for key loading.
    PROVIDER_KEY_ENV = {
        'openrouter': ('OPEN_ROUTER_API_KEY', 'OpenRouter'),
        'cohere': ('COHERE_API_KEY', 'Cohere'),
        'replicate': ('REPLICATE_API_TOKEN', 'Replicate'),
    }

    def __init__(self, benchmark_dir: str, output_dir: str, num_folders: int,
                 required_providers: Optional[Iterable[str]] = None,
                 max_workers: int = 35):
        self.benchmark_dir = benchmark_dir
        self.output_dir = output_dir
        self.num_folders = num_folders
        self.max_workers = max_workers

        # Load API keys only for the providers actually used by enabled models.
        # If required_providers is None, fall back to loading all providers.
        providers = set(required_providers) if required_providers is not None \
            else set(self.PROVIDER_KEY_ENV)
        self.api_keys = {
            provider: require_api_key(*self.PROVIDER_KEY_ENV[provider])
            for provider in providers
            if provider in self.PROVIDER_KEY_ENV
        }

        os.makedirs(output_dir, exist_ok=True)

    def _create_safe_filename(self, model_name: str) -> str:
        """Create filesystem-safe filename from model name."""
        return model_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")

    def _run_single_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Run benchmark for a single vision model."""
        safe_name = self._create_safe_filename(model_config.name)
        output_csv = os.path.join(self.output_dir, f"{safe_name}.csv")
        output_json_name = f"{safe_name}.json"

        print(f"  Model ID: {model_config.model_id}")
        print(f"  Provider: {model_config.provider}")
        if model_config.note:
            print(f"  Note: {model_config.note}")
        print(f"  Output: {output_csv}")
        print("-" * 60)

        # Build parameters based on provider
        base_params = {
            "benchmark_dir": self.benchmark_dir,
            "output_csv": output_csv,
            "output_json_name": output_json_name,
            "num_folders": self.num_folders,
            "model_name": model_config.model_id,
            "json_schema": get_json_schema(),
            "analyzer_func": model_config.analyzer,
            "max_workers": self.max_workers,
        }

        # Add provider-specific parameters
        if model_config.provider == "cohere":
            base_params["cohere_api_key"] = self.api_keys['cohere']
        elif model_config.provider == "replicate":
            base_params["replicate_api_token"] = self.api_keys['replicate']
        else:  # openrouter
            base_params.update({
                "open_router_api_key": self.api_keys['openrouter'],
                "url": "https://openrouter.ai/api/v1/chat/completions"
            })

        # Run benchmark
        process_benchmark_floorplans(**base_params)

        return {
            "name": model_config.name,
            "csv": output_csv,
            "status": "success"
        }

    def run_benchmark(self, models: List[ModelConfig]) -> List[Dict[str, Any]]:
        """Run benchmark for all specified models."""
        print("=" * 60)
        print("RUNNING OBJECT COUNTING BENCHMARK")
        print("=" * 60)
        print(f"Benchmark directory: {self.benchmark_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of models: {len(models)}\n")

        results_summary = []

        for i, model_config in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Processing: {model_config.name}")

            try:
                result = self._run_single_model(model_config)
                results_summary.append(result)
                print(f"[SUCCESS] {model_config.name} completed")
            except Exception as e:
                print(f"[ERROR] {model_config.name} failed: {e}")
                results_summary.append({
                    "name": model_config.name,
                    "csv": os.path.join(self.output_dir, f"{self._create_safe_filename(model_config.name)}.csv"),
                    "status": "failed",
                    "error": str(e)
                })

        return results_summary


def print_results_summary(results: List[Dict[str, Any]]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print("\nResults Summary:")
    for result in results:
        status_icon = "[OK]" if result["status"] == "success" else "[FAIL]"
        print(f"  {status_icon} {result['name']}: {result['csv']}")


def generate_visualization(results: List[Dict[str, Any]]):
    """Generate combined visualization from successful results."""
    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        print("\n[WARNING] No successful results to visualize")
        return

    print("\nGenerating combined visualization...")
    print("=" * 60)

    from src.benchmark.visualizer import plot_all_models_comparison

    plot_all_models_comparison(
        csv_files=[r["csv"] for r in successful_results],
        model_names=[r["name"] for r in successful_results],
        output_dir="results/heatmap_outputs"
    )


def main():
    """Main execution function."""
    # Configuration
    benchmark_dir = r"data/Use Case 1 - Object Counting/1 - Full Datasets"
    # Overridable via env for quick tests; defaults run the full benchmark.
    output_dir = os.getenv("BENCH_OUTPUT_DIR", "benchmark_result_object_counting")
    num_folders = int(os.getenv("BENCH_NUM_FOLDERS", "120"))
    max_workers = int(os.getenv("BENCH_MAX_WORKERS", "35"))

    # Initialize model registry and get enabled models first, so we only
    # require API keys for the providers those models actually use.
    model_registry = ModelRegistry()
    enabled_models = model_registry.get_enabled_models()

    if not enabled_models:
        print("No models are enabled for benchmarking.")
        print("Enable models by editing the ModelRegistry or using:")
        print("  registry.enable_model('Model Name')")
        model_registry.list_models()
        return

    required_providers = {model.provider for model in enabled_models}
    benchmark_runner = BenchmarkRunner(
        benchmark_dir, output_dir, num_folders,
        required_providers=required_providers,
        max_workers=max_workers,
    )

    # Run benchmark
    results = benchmark_runner.run_benchmark(enabled_models)

    # Print results summary
    print_results_summary(results)

    # Generate visualization (skippable for quick tests)
    if os.getenv("BENCH_SKIP_VIZ") == "1":
        print("\n[INFO] BENCH_SKIP_VIZ=1 set - skipping visualization.")
    else:
        generate_visualization(results)
        print("\n[SUCCESS] All done! Check 'results/' folder for visualizations.")


if __name__ == "__main__":
    main()
