"""
Benchmark processing pipeline for floor plan analysis.
"""
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from statistics import mean as _mean, median as _median
from typing import Callable, Dict, Optional, List, Any
from ..analyzers.openrouter import analyze_floorplan
from ..models.plan_elements import get_json_schema
from ..utils.pricing import compute_cost


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark processing."""
    benchmark_dir: str
    output_csv: str
    output_json_name: str
    num_folders: int
    max_workers: int = 35


@dataclass 
class ModelConfig:
    """Configuration for model parameters."""
    model_name: str
    json_schema: Optional[Dict] = None
    temperature: float = 0.0
    
    def __post_init__(self):
        if self.json_schema is None:
            self.json_schema = get_json_schema()


@dataclass
class AnalyzerConfig:
    """Configuration for analyzer function and its parameters."""
    analyzer_func: Optional[Callable] = None
    open_router_api_key: Optional[str] = None
    url: str = "https://openrouter.ai/api/v1/chat/completions"
    analyzer_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.analyzer_func is None:
            if self.open_router_api_key is None:
                raise ValueError("open_router_api_key is required when using default analyzer")
            self.analyzer_func = analyze_floorplan


@dataclass
class FolderInfo:
    """Information about a benchmark folder."""
    name: str
    path: str
    metadata_path: str
    image_path: Optional[str] = None
    output_json_path: Optional[str] = None


class DatasetScanner:
    """Handles scanning and validation of benchmark dataset folders."""
    
    def __init__(self):
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    
    def scan_folders(self, benchmark_dir: str, num_folders: int, output_json_name: str) -> List[FolderInfo]:
        """Scan benchmark directory and return valid folder information."""
        try:
            all_entries = sorted(os.listdir(benchmark_dir))
        except Exception as e:
            print(f"[ERROR] Could not list directory '{benchmark_dir}': {e}")
            return []

        folders = []
        processed = 0

        for entry in all_entries:
            if processed >= num_folders:
                break

            folder_path = os.path.join(benchmark_dir, entry)
            if not os.path.isdir(folder_path):
                continue

            folder_info = FolderInfo(
                name=entry,
                path=folder_path,
                metadata_path=os.path.join(folder_path, "metadata.json"),
                output_json_path=os.path.join(folder_path, output_json_name)
            )

            # Validate folder
            if not self._validate_folder(folder_info):
                processed += 1
                continue
            
            # Find image
            folder_info.image_path = self._find_image_in_folder(folder_info)
            if folder_info.image_path is None:
                print(f"[WARN] Skipping '{folder_info.name}': no image file found")
                processed += 1
                continue

            folders.append(folder_info)
            processed += 1

        return folders
    
    def _validate_folder(self, folder_info: FolderInfo) -> bool:
        """Validate that folder has required metadata.json."""
        if not os.path.isfile(folder_info.metadata_path):
            print(f"[WARN] Skipping '{folder_info.name}': missing metadata.json")
            return False
        return True
    
    def _find_image_in_folder(self, folder_info: FolderInfo) -> Optional[str]:
        """Find image file in the folder."""
        # First try to find image with same name as folder
        for ext in self.image_extensions:
            potential_path = os.path.join(folder_info.path, f"{folder_info.name}{ext}")
            if os.path.isfile(potential_path):
                return potential_path
        
        # If not found, look for any image file in the folder
        try:
            for file in os.listdir(folder_info.path):
                if any(file.lower().endswith(ext.lower()) for ext in self.image_extensions):
                    return os.path.join(folder_info.path, file)
        except OSError:
            pass
        
        return None


class AnalyzerRunner:
    """Handles running analysis on images using configured analyzer."""
    
    def run_analysis(self, image_path: str, model_config: ModelConfig, analyzer_config: AnalyzerConfig,
                     usage_out: Optional[Dict] = None) -> Dict:
        """Run analysis on image and return results.

        If ``usage_out`` is provided and the analyzer is an OpenRouter analyzer,
        it is passed through so the caller receives token usage.
        """
        if analyzer_config.analyzer_func == analyze_floorplan:
            # Default analyzer with standard parameters
            return analyzer_config.analyzer_func(
                image_path=image_path,
                model_name=model_config.model_name,
                json_schema=model_config.json_schema,
                open_router_api_key=analyzer_config.open_router_api_key,
                url=analyzer_config.url,
                temperature=model_config.temperature,
                usage_out=usage_out,
            )
        else:
            # Custom analyzer - build parameters dynamically
            kwargs = self._build_analyzer_kwargs(image_path, model_config, analyzer_config)
            # Only OpenRouter analyzers accept usage_out (cohere/replicate do not)
            is_openrouter = not ("cohere_api_key" in kwargs or "replicate_api_token" in kwargs)
            if usage_out is not None and is_openrouter:
                kwargs["usage_out"] = usage_out
            return analyzer_config.analyzer_func(**kwargs)
    
    def _build_analyzer_kwargs(self, image_path: str, model_config: ModelConfig, analyzer_config: AnalyzerConfig) -> Dict[str, Any]:
        """Build kwargs for custom analyzer functions."""
        kwargs = {
            "image_path": image_path,
            "model_name": model_config.model_name,
            "json_schema": model_config.json_schema,
            **analyzer_config.analyzer_kwargs
        }
        
        # Check analyzer type and add appropriate parameters
        is_cohere_analyzer = "cohere_api_key" in analyzer_config.analyzer_kwargs
        is_replicate_analyzer = "replicate_api_token" in analyzer_config.analyzer_kwargs
        
        if not is_cohere_analyzer and not is_replicate_analyzer:
            # For OpenRouter analyzers, add standard parameters
            if "open_router_api_key" not in kwargs and analyzer_config.open_router_api_key is not None:
                kwargs["open_router_api_key"] = analyzer_config.open_router_api_key
            if "url" not in kwargs:
                kwargs["url"] = analyzer_config.url
        
        if "temperature" not in kwargs:
            kwargs["temperature"] = model_config.temperature
            
        return kwargs


class ResultAggregator:
    """Handles saving and aggregating benchmark results."""
    
    def save_folder_result(self, folder_info: FolderInfo, result: Dict) -> bool:
        """Save result JSON to folder."""
        try:
            with open(folder_info.output_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return True
        except Exception as e:
            print(f"[ERROR] Could not write JSON for '{folder_info.name}': {e}")
            return False
    
    def load_folder_metadata(self, folder_info: FolderInfo) -> Optional[Dict]:
        """Load original metadata from folder."""
        try:
            with open(folder_info.metadata_path, "r", encoding="utf-8-sig") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read metadata.json for '{folder_info.name}': {e}")
            return None
    
    def create_csv_row(self, folder_info: FolderInfo, original_data: Dict, extracted_data: Dict,
                       metrics: Optional[Dict] = None) -> Optional[Dict]:
        """Create a CSV row from folder data, including per-call cost/latency metrics."""
        try:
            metrics = metrics or {}
            return {
                "name": folder_info.name,
                "original": json.dumps(original_data, separators=(",", ":")),
                "extracted": json.dumps(extracted_data, separators=(",", ":")),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "cost_usd": metrics.get("cost_usd"),
                "latency_s": metrics.get("latency_s"),
            }
        except Exception as e:
            print(f"[ERROR] Could not prepare CSV row for '{folder_info.name}': {e}")
            return None

    # Columns appended after the original metrics-free schema, so the evaluator
    # (which reads by column name) keeps working unchanged.
    CSV_FIELDNAMES = ["name", "original", "extracted",
                      "prompt_tokens", "completion_tokens", "cost_usd", "latency_s"]

    def write_csv(self, rows: List[Dict], output_csv: str) -> bool:
        """Write results to CSV file."""
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.CSV_FIELDNAMES)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            return True
        except Exception as e:
            print(f"[ERROR] Could not write CSV '{output_csv}': {e}")
            return False


class BenchmarkProcessor:
    """Main benchmark processor with clean separation of concerns."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset_scanner = DatasetScanner()
        self.analyzer_runner = AnalyzerRunner()
        self.result_aggregator = ResultAggregator()
    
    def process_benchmark(self, model_config: ModelConfig, analyzer_config: AnalyzerConfig) -> bool:
        """
        Process benchmark using clean, modular approach.
        
        Returns:
            True if processing completed successfully, False otherwise
        """
        # Step 1: Scan and validate dataset folders
        print(f"Scanning benchmark directory: {self.config.benchmark_dir}")
        folders = self.dataset_scanner.scan_folders(
            self.config.benchmark_dir, 
            self.config.num_folders, 
            self.config.output_json_name
        )
        
        if not folders:
            print("[ERROR] No valid folders found to process")
            return False
        
        total = len(folders)
        print(f"Found {total} valid folders to process")
        print(f"Running with up to {self.config.max_workers} concurrent requests")

        # Step 2: Process folders concurrently (each call is an independent I/O-bound request)
        wall_start = time.time()
        csv_rows: List[Dict] = []
        done = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_folder = {
                executor.submit(
                    self._process_single_folder, folder_info, model_config, analyzer_config
                ): folder_info
                for folder_info in folders
            }
            for future in as_completed(future_to_folder):
                done += 1
                row = future.result()
                if row is not None:
                    csv_rows.append(row)
                    print(f"[DONE {done}/{total}] '{row['name']}' "
                          f"in {row.get('latency_s')}s\n", flush=True)

        wall_elapsed = time.time() - wall_start
        processed_count = len(csv_rows)

        # Keep deterministic output ordering despite out-of-order completion
        csv_rows.sort(key=lambda r: r["name"])

        # Step 3: Write CSV results + run summary
        if csv_rows:
            success = self.result_aggregator.write_csv(csv_rows, self.config.output_csv)
            if success:
                self._write_run_summary(csv_rows, model_config, wall_elapsed, total)
                print(f"[SUCCESS] Completed. CSV saved to '{self.config.output_csv}'. "
                      f"Processed {processed_count}/{total} folders in {wall_elapsed:.1f}s wall-clock.")
                return True

        print(f"[WARNING] Processing completed but no results to save. Processed {processed_count}/{total} folders.")
        return False

    def _write_run_summary(self, csv_rows: List[Dict], model_config: ModelConfig,
                           wall_elapsed: float, total: int) -> None:
        """Aggregate per-call metrics into a run summary (printed + saved as JSON)."""
        latencies = [r["latency_s"] for r in csv_rows if r.get("latency_s") is not None]
        costs = [r["cost_usd"] for r in csv_rows if r.get("cost_usd") is not None]
        ptoks = [r["prompt_tokens"] for r in csv_rows if r.get("prompt_tokens") is not None]
        ctoks = [r["completion_tokens"] for r in csv_rows if r.get("completion_tokens") is not None]

        summary = {
            "model": model_config.model_name,
            "folders_total": total,
            "folders_succeeded": len(csv_rows),
            "wall_clock_s": round(wall_elapsed, 1),
            "max_workers": self.config.max_workers,
            "total_cost_usd": round(sum(costs), 4) if costs else None,
            "total_prompt_tokens": sum(ptoks) if ptoks else None,
            "total_completion_tokens": sum(ctoks) if ctoks else None,
            "mean_latency_s": round(_mean(latencies), 2) if latencies else None,
            "median_latency_s": round(_median(latencies), 2) if latencies else None,
            "max_latency_s": round(max(latencies), 2) if latencies else None,
        }
        summary_path = os.path.splitext(self.config.output_csv)[0] + "_summary.json"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            print(f"[WARN] Could not write summary JSON: {e}")

        cost_str = f"${summary['total_cost_usd']:.4f}" if summary["total_cost_usd"] is not None else "n/a (unpriced)"
        print("\n  --- Run summary ---")
        print(f"  Cost: {cost_str}   Tokens: {summary['total_prompt_tokens']} in / {summary['total_completion_tokens']} out")
        print(f"  Latency/call: mean {summary['mean_latency_s']}s, median {summary['median_latency_s']}s, max {summary['max_latency_s']}s")
        print(f"  Wall-clock: {summary['wall_clock_s']}s for {len(csv_rows)}/{total} folders")

    def _process_single_folder(
        self,
        folder_info: FolderInfo,
        model_config: ModelConfig,
        analyzer_config: AnalyzerConfig,
    ) -> Optional[Dict]:
        """Process a single folder; return its CSV row (with metrics) or None on failure.

        Thread-safe: returns a value rather than mutating shared state, so it can
        run inside a ThreadPoolExecutor.
        """
        print(
            f"[START] '{folder_info.name}' "
            f"(image='{os.path.basename(folder_info.image_path)}')",
            flush=True,
        )
        usage: Dict = {}
        start_ts = time.time()
        # Step 1: Run analysis
        try:
            result = self.analyzer_runner.run_analysis(
                folder_info.image_path, model_config, analyzer_config, usage_out=usage
            )

            # Parse JSON string if needed
            if isinstance(result, str):
                if not result.strip():
                    raise ValueError("Empty response from analyzer")
                try:
                    result = json.loads(result)
                except json.JSONDecodeError as json_err:
                    raise ValueError(f"Invalid JSON response: {json_err}. Response preview: {result[:200]}")

        except Exception as e:
            print(f"[ERROR] Analysis failed for '{folder_info.name}': {e}")
            return None

        latency_s = round(time.time() - start_ts, 2)

        # Step 2: Save folder result
        if not self.result_aggregator.save_folder_result(folder_info, result):
            return None

        # Step 3: Load metadata and create CSV row with metrics
        original_data = self.result_aggregator.load_folder_metadata(folder_info)
        if original_data is None:
            return None

        has_usage = usage.get("prompt_tokens") is not None or usage.get("completion_tokens") is not None
        cost = compute_cost(model_config.model_name,
                            usage.get("prompt_tokens"), usage.get("completion_tokens")) if has_usage else None
        metrics = {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cost_usd": round(cost, 6) if cost is not None else None,
            "latency_s": latency_s,
        }
        return self.result_aggregator.create_csv_row(folder_info, original_data, result, metrics)


def process_benchmark_floorplans(
    benchmark_dir: str,
    output_csv: str,
    output_json_name: str,
    num_folders: int,
    model_name: str,
    json_schema: Optional[Dict] = None,
    open_router_api_key: Optional[str] = None,
    url: str = "https://openrouter.ai/api/v1/chat/completions",
    temperature: float = 0.0,
    analyzer_func: Optional[Callable] = None,
    max_workers: int = 35,
    **analyzer_kwargs
):
    """
    Process benchmark folders by calling analyze_floorplan on each image, saving per-folder JSON,
    and aggregating results into a CSV.

    This function is a backward compatibility wrapper around the new BenchmarkProcessor class.

    Parameters:
    - benchmark_dir: path to directory with subfolders to process.
    - output_csv: path to CSV file to write results.
    - output_json_name: name of JSON file to save in each folder (e.g., "results.json").
    - num_folders: number of folders (in sorted order) to process.
    - model_name: model identifier for analyze_floorplan.
    - json_schema: dict of JSON schema for analyze_floorplan (defaults to PlanElements schema).
    - open_router_api_key: API key for OpenRouter (if using default analyzer).
    - url: endpoint URL passed to analyze_floorplan.
    - temperature: temperature for analyze_floorplan.
    - analyzer_func: optional custom analyzer function to use instead of default.
    - **analyzer_kwargs: additional keyword arguments to pass to analyzer function.
    """
    # Create configuration objects for the new architecture
    benchmark_config = BenchmarkConfig(
        benchmark_dir=benchmark_dir,
        output_csv=output_csv,
        output_json_name=output_json_name,
        num_folders=num_folders,
        max_workers=max_workers
    )
    
    model_config = ModelConfig(
        model_name=model_name,
        json_schema=json_schema,
        temperature=temperature
    )
    
    analyzer_config = AnalyzerConfig(
        analyzer_func=analyzer_func,
        open_router_api_key=open_router_api_key,
        url=url,
        analyzer_kwargs=analyzer_kwargs
    )
    
    # Use the new BenchmarkProcessor
    processor = BenchmarkProcessor(benchmark_config)
    success = processor.process_benchmark(model_config, analyzer_config)
    
    # The old function didn't return anything, so we maintain that behavior
    # But internally we now have proper error handling and return values

