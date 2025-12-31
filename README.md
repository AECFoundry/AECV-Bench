# AECV-Bench

A comprehensive benchmarking framework for evaluating Multimodal Large Language Models (MMLMs) and Vision-Language Models (VLMs) on Architecture, Engineering, and Construction (AEC) drawing understanding tasks.

## Overview

This project evaluates the performance of various multimodal and vision-language models on understanding AEC drawings. The framework supports multiple providers (OpenRouter, Cohere, Replicate) and generates detailed performance analysis with visualizations.

## Features

- **Multi-Provider Support**: OpenRouter, Cohere, Replicate APIs
- **Comprehensive Benchmark**: Object counting, OCR, drawing and spatial understanding
- **LLM-as-Judge Evaluation**: Quality assessment using GPT-4o
- **Performance Visualization**: Accuracy heatmaps and MAPE analysis
- **Clean Architecture**: Modular, extensible, and maintainable codebase

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <https://github.com/AECFoundry/AECV-Bench>
cd AECV-Bench

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your OpenRouter, Cohere, and Replicate API keys
```

### 2. API Keys Configuration

Create a `.env` file with your API keys:

```
OPEN_ROUTER_API_KEY=your_openrouter_key_here
COHERE_API_KEY=your_cohere_key_here
REPLICATE_API_TOKEN=your_replicate_token_here
```

### 3. Run Benchmarks

**Object Counting Benchmark:**
```bash
python run_object_counting_benchmark.py
```

**Drawing Understanding (QA) Benchmark:**
```bash
python run_qa_benchmark.py
```

**QA Evaluation with LLM-as-Judge:**
```bash
python run_qa_llm_judge_evaluation.py
```

**Generate Heatmaps:**
```bash
python generate_heatmaps.py
```

### 4. Manual QA Evaluation (Web App)

A Flask-based web application for manually evaluating MMLM & VLM responses on the QA benchmark. Displays floor plan images alongside questions and allows users to mark model responses as correct/incorrect.

```bash
cd qa_evaluation_app
python app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Side-by-side comparison of model answers with ground truth
- Navigation through all QA items with keyboard shortcuts
- Search functionality by image ID or question text
- Progress tracking and statistics
- Export evaluations as JSON

## Benchmarked Models

- Anthropic / Claude Opus 4.5
- Google / Gemini 3 Pro Preview
- OpenAI / GPT-5.2
- xAI / Grok 4.1 Fast
- Mistral AI / Mistral Large 2512
- Alibaba / Qwen3-VL 8B Instruct
- Zhipu AI / GLM-4.6V
- NVIDIA / Nemotron Nano 12B V2 VL
- Amazon / Nova 2 Lite v1
- Cohere / Command A Vision

## Benchmark Tasks

### 1 - Object Counting
- **Doors**: Count all doors in floor plan
- **Windows**: Count all windows in floor plan
- **Bedrooms**: Count bedrooms specifically
- **Toilets**: Count bathroom/toilet facilities

### 2 - Drawing Understanding (Document Understanding)
The QA benchmark evaluates models on four categories of question-answer tasks:

- **OCR (Text Extraction)**: Locate and read labels, values, notes, dimensions, and metadata
  - Example: "What is the drawing scale indicated in the title block?"

- **Spatial Reasoning**: Infer spatial relationships including adjacency, connectivity, containment, and access paths
  - Example: "Where is the General Notes section located in the drawing?"

- **Instance Counting**: Identify and enumerate architectural elements and symbols
  - Example: "How many section view callouts are present in the drawing?"

- **Comparative Reasoning**: Compare across multiple candidates to satisfy a criterion
  - Example: "Which quadrant of the building has more core access?"

### Evaluation Metrics
- **Accuracy**: Exact match percentage
- **MAPE**: Mean Absolute Percentage Error
- **Recall**: Per-category recall scores
- **LLM-as-Judge**: Quality assessment by GPT-4o

## Configuration

### Model Selection
Edit the model configurations in benchmark scripts to enable/disable models:

```python
# In run_object_counting_benchmark.py
# Set enabled=True for models you want to run
models = [
    {"name": "Claude Opus 4.5", "model_id": "anthropic/claude-opus-4.5", "enabled": True},
    {"name": "Gemini 3 Pro", "model_id": "google/gemini-3-pro-preview", "enabled": False},
]
```

### Dataset Configuration
Update dataset paths in benchmark scripts:

**1 - Object Counting Benchmark:**
```python
benchmark_dir = r"data/Use Case 1 - Object Counting/1 - Full Datasets"
num_folders = 120  # Number of test images
```

**2 - Drawing Understanding (QA) Benchmark:**
```python
images_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images"
labels_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels"
```

## Output Files

### Benchmark Results
- `benchmark_result_object_counting/*.csv` - Object counting results
- `benchmark_result_qa/*.csv` - QA task results

### Visualizations
- `results/heatmap_outputs/` - Accuracy and MAPE heatmaps
- `results/qa_llm_judge_results/` - LLM evaluation reports & Bar charts

## API Reference

### Core Classes

**OpenRouterAnalyzer**
```python
analyzer = OpenRouterAnalyzer(use_schema_format=True)
result = analyzer.analyze_floorplan(image_path, model_id, api_key, schema)
```

**CohereAnalyzer**
```python
analyzer = CohereAnalyzer()
result = analyzer.analyze_floorplan(image_path, model_id, api_key)
```

**ReplicateAnalyzer**
```python
analyzer = ReplicateAnalyzer()
result = analyzer.analyze_floorplan(image_path, model_id, api_token)
```

### Configuration Management
```python
from src.utils.config import require_api_key
api_key = require_api_key('OPEN_ROUTER_API_KEY', 'OpenRouter')
```


## Contributing

We welcome contributions! Here's how to get started:

### Adding a New Model

1. **OpenRouter models**: Add a new `ModelConfig` entry in `run_object_counting_benchmark.py`:
   ```python
   ModelConfig(
       name="Model Display Name",
       model_id="provider/model-id",
       analyzer=analyze_floorplan,  # or analyze_floorplan_prompt_based
       provider="openrouter",
       note="Brief description",
       enabled=True
   )
   ```

2. **Other providers**: Implement a new analyzer in `src/analyzers/` following the existing patterns.

### Analyzer Selection

- `analyze_floorplan`: For models supporting structured output via `response_format` (recommended)
- `analyze_floorplan_prompt_based`: For models requiring prompt-based JSON extraction

### Project Structure

```
src/
├── analyzers/       # API provider implementations
├── benchmark/       # Benchmark processing and evaluation
├── models/          # Pydantic data models
├── parsers/         # Response parsing utilities
└── utils/           # Configuration and helpers
```

### Development Guidelines

- Use environment variables for API keys (never commit keys)
- Follow existing code patterns for consistency
- Test with a small `num_folders` before running full benchmarks

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Make your changes
4. Test locally with a small dataset
5. Submit a pull request with a clear description

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{aecv-bench,
  title={AECV-Bench: Benchmarking Multimodal Models on Architectural and Engineering Drawing Understanding},
  author={Authors: Aleksei Kondratenko, PhD; Houssame E. Hsain; Mussie Birhane; Guido Maciocci},
  year={2025},
  url={https://github.com/AECFoundry/AECV-Bench}
}
```