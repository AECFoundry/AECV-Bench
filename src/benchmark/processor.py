"""
Benchmark processing pipeline for floor plan analysis.
"""
import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, List, Any
from ..analyzers.openrouter import analyze_floorplan
from ..models.plan_elements import get_json_schema


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark processing."""
    benchmark_dir: str
    output_csv: str
    output_json_name: str
    num_folders: int


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
    
    def run_analysis(self, image_path: str, model_config: ModelConfig, analyzer_config: AnalyzerConfig) -> Dict:
        """Run analysis on image and return results."""
        if analyzer_config.analyzer_func == analyze_floorplan:
            # Default analyzer with standard parameters
            return analyzer_config.analyzer_func(
                image_path=image_path,
                model_name=model_config.model_name,
                json_schema=model_config.json_schema,
                open_router_api_key=analyzer_config.open_router_api_key,
                url=analyzer_config.url,
                temperature=model_config.temperature
            )
        else:
            # Custom analyzer - build parameters dynamically
            kwargs = self._build_analyzer_kwargs(image_path, model_config, analyzer_config)
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
    
    def create_csv_row(self, folder_info: FolderInfo, original_data: Dict, extracted_data: Dict) -> Optional[Dict]:
        """Create a CSV row from folder data."""
        try:
            return {
                "name": folder_info.name,
                "original": json.dumps(original_data, separators=(",", ":")),
                "extracted": json.dumps(extracted_data, separators=(",", ":"))
            }
        except Exception as e:
            print(f"[ERROR] Could not prepare CSV row for '{folder_info.name}': {e}")
            return None
    
    def write_csv(self, rows: List[Dict], output_csv: str) -> bool:
        """Write results to CSV file."""
        try:
            with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["name", "original", "extracted"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
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
        
        print(f"Found {len(folders)} valid folders to process")
        
        # Step 2: Process each folder
        csv_rows = []
        processed_count = 0
        
        for i, folder_info in enumerate(folders, 1):
            success = self._process_single_folder(
                folder_info, model_config, analyzer_config, csv_rows, i, len(folders)
            )
            if success:
                processed_count += 1
            
            # Rate limiting
            if i < len(folders):
                time.sleep(0.5)
        
        # Step 3: Write CSV results
        if csv_rows:
            success = self.result_aggregator.write_csv(csv_rows, self.config.output_csv)
            if success:
                print(f"[SUCCESS] Completed. CSV saved to '{self.config.output_csv}'. Processed {processed_count}/{len(folders)} folders.")
                return True
        
        print(f"[WARNING] Processing completed but no results to save. Processed {processed_count}/{len(folders)} folders.")
        return False
    
    def _process_single_folder(
        self, 
        folder_info: FolderInfo, 
        model_config: ModelConfig, 
        analyzer_config: AnalyzerConfig, 
        csv_rows: List[Dict],
        current: int,
        total: int
    ) -> bool:
        """Process a single folder and add result to csv_rows if successful."""
        start_ts = time.time()
        print(
            f"[START] {current}/{total}: '{folder_info.name}' "
            f"(image='{os.path.basename(folder_info.image_path)}', metadata='{os.path.basename(folder_info.metadata_path)}')",
            flush=True,
        )
        # Step 1: Run analysis
        try:
            result = self.analyzer_runner.run_analysis(
                folder_info.image_path, model_config, analyzer_config
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
            return False
        
        # Step 2: Save folder result
        if not self.result_aggregator.save_folder_result(folder_info, result):
            return False
        
        # Step 3: Load metadata and create CSV row
        original_data = self.result_aggregator.load_folder_metadata(folder_info)
        if original_data is None:
            return False
        
        csv_row = self.result_aggregator.create_csv_row(folder_info, original_data, result)
        if csv_row is None:
            return False
        
        csv_rows.append(csv_row)
        duration = time.time() - start_ts
        print(f"[DONE] {current}/{total}: '{folder_info.name}' in {duration:.1f}s\n", flush=True)
        return True


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
        num_folders=num_folders
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

