"""
Script to test QA pairs from ground truth labels.

This script:
1. Reads label JSON files with QA pairs (qa_pairs, ocr_qa, spatial_qa, counting_qa, comparison_qa)
2. Sends questions to the model with the corresponding image
3. Saves results: question, model answer, ground truth answer, task type
"""
import json
import os
import csv
import re
import time
from pathlib import Path
from src.analyzers.openrouter import analyze_floorplan
from src.utils.image_utils import encode_image_to_base64
from src.utils.config import require_api_key
import requests

# API keys are loaded conditionally in __main__, based on which providers the
# enabled models actually use (so an OpenRouter-only run never needs a Cohere key).
open_router_api_key = None
cohere_api_key = None

# Configuration
images_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images"
labels_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels"
# Overridable via env for quick tests; default writes to the canonical results dir.
output_dir = os.getenv("QA_OUTPUT_DIR", "benchmark_result_qa")
os.makedirs(output_dir, exist_ok=True)
url = "https://openrouter.ai/api/v1/chat/completions"
temperature = 0.0

# Model configurations - based on run_all_models_benchmark.py
# Uncomment models you want to test
models = [
    # {
    #     "name": "Gemini 3 Pro Preview",
    #     "model_id": "google/gemini-3-pro-preview",
    #     "note": "Latest flagship model, high-precision multimodal reasoning"
    # },
    # {
    #     "name": "Claude Opus 4.5",
    #     "model_id": "anthropic/claude-opus-4.5",
    #     "note": "Most advanced Opus model, optimized for complex reasoning tasks"
    # },
    # {
    #     "name": "Claude Sonnet 4.5",
    #     "model_id": "anthropic/claude-sonnet-4.5",
    #     "note": "Most advanced Sonnet, optimized for real-world agents"
    # },
    # {
    #     "name": "Gemini 3.1 Pro",
    #     "model_id": "google/gemini-3.1-pro-preview",
    #     "note": "Gemini 3.1 Pro preview model"
    # },
    # {
    #     "name": "Claude Sonnet 4.6",
    #     "model_id": "anthropic/claude-sonnet-4.6",
    #     "note": "Anthropic Claude Sonnet 4.6, latest Sonnet model"
    # },
    # {
    #     "name": "Qwen 3.5 Plus",
    #     "model_id": "qwen/qwen3.5-plus-02-15",
    #     "note": "Qwen 3.5 Plus model from February 2025"
    # },
    # {
    #     "name": "Claude Opus 4.6",
    #     "model_id": "anthropic/claude-opus-4.6",
    #     "note": "Anthropic Claude Opus 4.6, most advanced Opus model"
    # },
    # {
    #     "name": "Qwen3-VL 8B Instruct",
    #     "model_id": "qwen/qwen3-vl-8b-instruct",
    #     "note": "8B Qwen3 vision-language model – efficient and fast"
    # },
    # {
    #     "name": "Qwen3-VL 8B Thinking",
    #     "model_id": "qwen/qwen3-vl-8b-thinking",
    #     "note": "8B Qwen3 thinking model – better reasoning, slower throughput"
    # },
    # {
    #     "name": "Mistral Large 2512",
    #     "model_id": "mistralai/mistral-large-2512",
    #     "note": "Mistral Large model from December 2025"
    # },
    # {
    #     "name": "OpenAI GPT-5.2",
    #     "model_id": "openai/gpt-5.2",
    #     "note": "OpenAI GPT-5.2 model"
    # },
    # {
    #     "name": "OpenAI GPT-5.3",
    #     "model_id": "openai/gpt-5.3-chat",
    #     "note": "OpenAI GPT-5.3 Chat model"
    # },
    # {
    #     "name": "OpenAI GPT-5.4",
    #     "model_id": "openai/gpt-5.4",
    #     "note": "OpenAI GPT-5.4 model"
    # },
    # ----- New frontier models (added 2026-06). Uncomment to run. -----
    {
        "name": "Claude Opus 4.8",
        "model_id": "anthropic/claude-opus-4.8",
        "note": "Anthropic Claude Opus 4.8 - latest Opus flagship"
    },
    # {
    #     "name": "Claude Opus 4.7",
    #     "model_id": "anthropic/claude-opus-4.7",
    #     "note": "Anthropic Claude Opus 4.7"
    # },
    # {
    #     "name": "Claude Fable 5",
    #     "model_id": "anthropic/claude-fable-5",
    #     "note": "Anthropic Claude Fable 5"
    # },
    # {
    #     "name": "OpenAI GPT-5.5",
    #     "model_id": "openai/gpt-5.5",
    #     "note": "OpenAI GPT-5.5 (pro variant excluded - too slow/expensive)"
    # },
    # {
    #     "name": "Gemini 3.5 Flash",
    #     "model_id": "google/gemini-3.5-flash",
    #     "note": "Google Gemini 3.5 Flash - newest Gemini vision on OpenRouter"
    # },
    # {
    #     "name": "Grok 4.3",
    #     "model_id": "x-ai/grok-4.3",
    #     "note": "xAI Grok 4.3 - latest stable Grok"
    # },
    # {
    #     "name": "Kimi K2.6",
    #     "model_id": "moonshotai/kimi-k2.6",
    #     "note": "Moonshot Kimi K2.6 - latest Kimi"
    # },
    # {
    #     "name": "MiniMax M3",
    #     "model_id": "minimax/minimax-m3",
    #     "note": "MiniMax M3 - latest MiniMax"
    # },
    # {
    #     "name": "Qwen 3.7 Plus",
    #     "model_id": "qwen/qwen3.7-plus",
    #     "note": "Qwen 3.7 Plus - newest Qwen vision"
    # },
    # {
    #     "name": "Gemma 4 31B IT",
    #     "model_id": "google/gemma-4-31b-it",
    #     "note": "Google Gemma 4 31B - open-weight"
    # },
    # {
    #     "name": "StepFun Step 3.7 Flash",
    #     "model_id": "stepfun/step-3.7-flash",
    #     "note": "StepFun Step 3.7 Flash - open VLM"
    # },
    # {
    #     "name": "NVIDIA Nemotron 3 Nano Omni 30B",
    #     "model_id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    #     "note": "NVIDIA frontier VISION model (Nemotron 3 Ultra is text-only)"
    # },
    # {
    #     "name": "Amazon Nova 2 Lite v1",
    #     "model_id": "amazon/nova-2-lite-v1",
    #     "note": "Amazon Nova 2 Lite vision-language model"
    # },
    # {
    #     "name": "Grok 4.1 Fast",
    #     "model_id": "x-ai/grok-4.1-fast",
    #     "note": "Best agentic tool calling model, 2M context"
    # },
    # {
    #     "name": "OpenAI GPT-4 Vision",
    #     "model_id": "openai/gpt-4o",  # GPT-4o provides the latest GPT-4 vision features
    #     "note": "GPT-4o multimodal model (uses prompt-based JSON extraction via OpenRouter)"
    # },
    # {
    #     "name": "Nvidia Nemotron Nano 12B V2 VL",
    #     "model_id": "nvidia/nemotron-nano-12b-v2-vl",
    #     "note": "Nvidia Nemotron Nano 12B V2 vision-language model"
    # },
    # {
    #     "name": "Llama Nemotron Embed VL 1B V2",
    #     "model_id": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
    #     "note": "Nvidia Llama Nemotron Embed VL 1B V2 vision-language model (free)"
    # },
    # {
    #     "name": "GLM-4.6V",
    #     "model_id": "z-ai/glm-4.6v",
    #     "note": "Z-AI vision-language model - newer version (uses prompt-based JSON extraction)"
    # },
    # {
    #     "name": "Cohere Command A Vision",
    #     "model_id": "command-a-vision-07-2025",  # Cohere's first commercial multimodal vision model
    #     "use_cohere_api": True,  # Set this flag to use Cohere API instead of OpenRouter
    #     "note": "Cohere Command A Vision - multimodal model for document analysis, chart interpretation, and OCR. 128K context, supports up to 20 images per request."
    # },
]


def ask_question_with_image(image_path: str, question: str, model_name: str, open_router_api_key: str, url: str, temperature: float = 0.0, usage_out: dict = None) -> str:
    """
    Send a question with an image to the model and get the answer.

    Args:
        image_path: Path to the image file
        question: The question to ask
        model_name: Model identifier
        open_router_api_key: API key
        url: API endpoint URL
        temperature: Sampling temperature
        usage_out: Optional mutable dict populated with token usage from the response

    Returns:
        The model's answer as a string
    """
    # Read and encode image
    base64_image = encode_image_to_base64(image_path)
    from src.utils.image_utils import get_image_mime_type
    mime_type = get_image_mime_type(image_path)
    data_url = f"data:{mime_type};base64,{base64_image}"

    # Prepare headers
    headers = {
        "Authorization": f"Bearer {open_router_api_key}",
        "Content-Type": "application/json"
    }
    
    # Build the message payload with instruction for short, precise answers
    prompt_text = f"Please analyze the engineering/architectural drawing attached and provide a short and precise answer to the following question. Avoid extended explanations.\n\n{question}"
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                }
            ]
        }
    ]
    
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }
    
    # Retry logic for network errors
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            # For HTTP errors, try to get more details from the response
            error_msg = f"{e}"
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.json() if e.response.content else {}
                    error_msg = f"{e}: {error_detail}"
                    print(f"[API ERROR] {error_msg}")
            except:
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"[API ERROR] {e.response.text[:500]}")
                except:
                    pass
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise requests.exceptions.HTTPError(f"{error_msg} for {image_path}") from e
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    resp_json = resp.json()
    try:
        content = resp_json["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise KeyError(f"Unexpected response format: {resp_json}") from e

    if usage_out is not None:
        usage = resp_json.get("usage", {}) or {}
        usage_out["prompt_tokens"] = usage.get("prompt_tokens")
        usage_out["completion_tokens"] = usage.get("completion_tokens")
        usage_out["total_tokens"] = usage.get("total_tokens")

    return content


def ask_question_with_image_cohere(image_path: str, question: str, model_name: str, cohere_api_key: str, url: str = "https://api.cohere.com/v2/chat", temperature: float = 0.0) -> str:
    """
    Send a question with an image to Cohere model and get the answer.
    Based on the existing cohere.py analyzer structure.
    
    Args:
        image_path: Path to the image file
        question: The question to ask
        model_name: Model identifier
        cohere_api_key: Cohere API key
        url: Cohere API endpoint URL
        temperature: Sampling temperature
        
    Returns:
        The model's answer as a string
    """
    if not cohere_api_key:
        raise ValueError("Cohere API key is required")
    
    # Read and encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare headers
    headers = {
        'accept': 'application/json',
        'content-type': 'application/json',
        'Authorization': f'bearer {cohere_api_key}'
    }
    
    # Build the message payload with instruction for short, precise answers
    prompt_text = f"Please analyze the engineering/architectural drawing attached and provide a short and precise answer to the following question. Avoid extended explanations.\n\n{question}"
    
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{get_image_mime_type(image_path)};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": temperature,
        "max_tokens": 2000
    }
    
    # Retry logic for network errors
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            error_msg = f"{e}"
            try:
                if hasattr(e, 'response') and e.response is not None:
                    error_detail = e.response.json() if e.response.content else {}
                    error_msg = f"{e}: {error_detail}"
                    print(f"[API ERROR] {error_msg}")
            except:
                try:
                    if hasattr(e, 'response') and e.response is not None:
                        print(f"[API ERROR] {e.response.text[:500]}")
                except:
                    pass
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise requests.exceptions.HTTPError(f"{error_msg} for {image_path}") from e
        except (requests.exceptions.ConnectionError, 
                requests.exceptions.Timeout,
                requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"[RETRY] Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    resp_json = resp.json()
    
    # Handle Cohere v2 API response structure (similar to cohere.py)
    content = ""
    try:
        # Try different v2 response structures
        if 'message' in resp_json:
            if 'content' in resp_json['message']:
                if isinstance(resp_json['message']['content'], list):
                    content = resp_json['message']['content'][0].get('text', '')
                else:
                    content = resp_json['message']['content']
        elif 'text' in resp_json:
            content = resp_json['text']
        elif 'choices' in resp_json:
            content = resp_json['choices'][0]['message']['content']
        else:
            content = str(resp_json)
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error extracting content: {e}")
        content = str(resp_json)
    
    return content


QA_CATEGORIES = ["qa_pairs", "ocr_qa", "spatial_qa", "counting_qa", "comparison_qa"]

# Provider error payloads can embed account identifiers (e.g. 'user_id': 'org_...').
# Results are published publicly, so record a short, non-identifying reason instead.
_ACCOUNT_ID_RE = re.compile(r"[,\s]*['\"]user_id['\"]\s*:\s*['\"][^'\"]+['\"]")


def sanitize_error(exc) -> str:
    """Build a public-safe error string for a failed model call."""
    msg = str(exc)
    if "429" in msg or "Too Many Requests" in msg:
        return "[ERROR: 429 Too Many Requests - provider rate limit]"
    msg = _ACCOUNT_ID_RE.sub("", msg)
    return f"[ERROR: {msg[:300]}]"


def _resolve_qa_image_path(label_data, label_file, images_dir):
    """Resolve the image path for a label file, trying common name/extension variants."""
    image_id = label_data.get("image_id", label_file.stem)
    image_path_from_label = label_data.get("image_path", "")
    possible_names = [
        f"{image_id}.png",
        f"{image_id}.jpg",
        f"{image_id}.jpeg",
        os.path.basename(image_path_from_label) if image_path_from_label else None,
    ]
    for name in possible_names:
        if name:
            potential_path = os.path.join(images_dir, name)
            if os.path.isfile(potential_path):
                return image_id, potential_path
    return image_id, None


def process_qa_benchmark(labels_dir: str, images_dir: str, output_csv: str, model_name: str,
                         open_router_api_key: str, url: str, temperature: float = 0.0,
                         use_cohere_api: bool = False, cohere_api_key: str = None,
                         max_workers: int = 35, limit_per_category: int = None,
                         limit_images: int = None):
    """
    Process QA pairs from label files and save results, including per-call
    cost/latency metrics. All (image, question) tasks are flattened into a single
    work-list and run concurrently with a thread pool (each call is an independent
    I/O-bound request).

    Args:
        labels_dir: Directory containing label JSON files
        images_dir: Directory containing image files
        output_csv: Path to output CSV file
        model_name: Model identifier
        open_router_api_key: OpenRouter API key (if not using Cohere)
        url: OpenRouter API endpoint URL (if not using Cohere)
        temperature: Sampling temperature
        use_cohere_api: If True, use Cohere API instead of OpenRouter
        cohere_api_key: Cohere API key (required if use_cohere_api is True)
        max_workers: Maximum number of concurrent requests
        limit_per_category: If set, only take the first N questions per category
                            (useful for quick smoke tests)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from statistics import mean as _mean, median as _median
    from src.utils.pricing import compute_cost

    label_files = sorted(Path(labels_dir).glob("*.json"))
    if not label_files:
        print(f"Error: No label files found in {labels_dir}")
        return
    if limit_images is not None:
        label_files = label_files[:limit_images]

    # Build a flat task list across all images and all QA categories
    tasks = []
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read {label_file}: {e}")
            continue

        image_id, image_path = _resolve_qa_image_path(label_data, label_file, images_dir)
        if not image_path:
            print(f"[WARN] Skipping {image_id}: image file not found")
            continue

        for qa_type in QA_CATEGORIES:
            items = label_data.get(qa_type, [])
            if not isinstance(items, list):
                continue
            if limit_per_category is not None:
                items = items[:limit_per_category]
            for idx, qa in enumerate(items, 1):
                question = qa.get("question", "")
                if not question:
                    continue
                tasks.append({
                    "image_id": image_id,
                    "image_path": image_path,
                    "qa_type": qa_type,
                    "task": qa.get("task", "unknown"),
                    "question": question,
                    "ground_truth": qa.get("answer", ""),
                    "qa_id": qa.get("id", f"{image_id}_{qa_type}_{idx}"),
                })

    total = len(tasks)
    print(f"Found {len(label_files)} label files -> {total} QA tasks")
    print(f"Model: {model_name}")
    print(f"Running with up to {max_workers} concurrent requests\n")

    def run_one(task):
        usage = {}
        start = time.time()
        try:
            if use_cohere_api:
                answer = ask_question_with_image_cohere(
                    image_path=task["image_path"], question=task["question"],
                    model_name=model_name, cohere_api_key=cohere_api_key, temperature=temperature)
            else:
                answer = ask_question_with_image(
                    image_path=task["image_path"], question=task["question"],
                    model_name=model_name, open_router_api_key=open_router_api_key,
                    url=url, temperature=temperature, usage_out=usage)
        except Exception as e:
            answer = sanitize_error(e)
        latency_s = round(time.time() - start, 2)
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        has_usage = pt is not None or ct is not None
        cost = compute_cost(model_name, pt, ct) if has_usage else None
        return {
            "image_id": task["image_id"], "qa_id": task["qa_id"], "qa_type": task["qa_type"],
            "task": task["task"], "question": task["question"], "ground_truth": task["ground_truth"],
            "model_answer": answer,
            "prompt_tokens": pt, "completion_tokens": ct,
            "cost_usd": round(cost, 6) if cost is not None else None, "latency_s": latency_s,
        }

    wall_start = time.time()
    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_one, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            row = future.result()
            results.append(row)
            err = " [ERROR]" if str(row["model_answer"]).startswith("[ERROR") else ""
            print(f"[DONE {done}/{total}] {row['qa_id']} ({row['latency_s']}s){err}", flush=True)
    wall_elapsed = time.time() - wall_start

    # Deterministic ordering despite out-of-order completion
    results.sort(key=lambda r: (r["image_id"], r["qa_type"], r["qa_id"]))

    if not results:
        print("\n[WARN] No results to save")
        return

    fieldnames = ["image_id", "qa_id", "qa_type", "task", "question", "ground_truth",
                  "model_answer", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s"]
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Run summary (cost / latency / tokens)
    latencies = [r["latency_s"] for r in results if r["latency_s"] is not None]
    costs = [r["cost_usd"] for r in results if r["cost_usd"] is not None]
    ptoks = [r["prompt_tokens"] for r in results if r["prompt_tokens"] is not None]
    ctoks = [r["completion_tokens"] for r in results if r["completion_tokens"] is not None]
    errors = sum(1 for r in results if str(r["model_answer"]).startswith("[ERROR"))
    summary = {
        "model": model_name, "qa_tasks_total": total, "errors": errors,
        "wall_clock_s": round(wall_elapsed, 1), "max_workers": max_workers,
        "total_cost_usd": round(sum(costs), 4) if costs else None,
        "total_prompt_tokens": sum(ptoks) if ptoks else None,
        "total_completion_tokens": sum(ctoks) if ctoks else None,
        "mean_latency_s": round(_mean(latencies), 2) if latencies else None,
        "median_latency_s": round(_median(latencies), 2) if latencies else None,
        "max_latency_s": round(max(latencies), 2) if latencies else None,
    }
    summary_path = os.path.splitext(output_csv)[0] + "_summary.json"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        print(f"[WARN] Could not write summary JSON: {e}")

    cost_str = f"${summary['total_cost_usd']:.4f}" if summary["total_cost_usd"] is not None else "n/a"
    print(f"\n[SUCCESS] Results saved to {output_csv}")
    print(f"  Tasks: {total - errors}/{total} ok ({errors} errors)")
    print(f"  Cost: {cost_str}   Tokens: {summary['total_prompt_tokens']} in / {summary['total_completion_tokens']} out")
    print(f"  Latency/call: mean {summary['mean_latency_s']}s, median {summary['median_latency_s']}s, max {summary['max_latency_s']}s")
    print(f"  Wall-clock: {summary['wall_clock_s']}s")


if __name__ == "__main__":
    print("="*60)
    print("QA BENCHMARK - Testing Question/Answer Pairs")
    print("="*60)
    print(f"Labels directory: {labels_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of models: {len(models)}\n")

    # Quick-test / parallelism config (overridable via env)
    max_workers = int(os.getenv("QA_MAX_WORKERS", "35"))
    _lpc = os.getenv("QA_LIMIT_PER_CATEGORY")
    limit_per_category = int(_lpc) if _lpc else None
    _li = os.getenv("QA_LIMIT_IMAGES")
    limit_images = int(_li) if _li else None

    # Load API keys only for the providers the enabled models actually use.
    open_router_api_key = require_api_key('OPEN_ROUTER_API_KEY', 'OpenRouter')
    if any(m.get("use_cohere_api") for m in models):
        cohere_api_key = require_api_key('COHERE_API_KEY', 'Cohere')

    # Run benchmark for each model
    results_summary = []
    for i, model_config in enumerate(models, 1):
        model_name = model_config["name"]
        model_id = model_config["model_id"]
        
        # Create output filename
        safe_name = model_name.lower().replace(" ", "_").replace(".", "").replace("-", "_")
        output_csv = os.path.join(output_dir, f"qa_results_{safe_name}.csv")
        
        print(f"\n[{i}/{len(models)}] Processing: {model_name}")
        print(f"  Model ID: {model_id}")
        if "note" in model_config:
            print(f"  Note: {model_config['note']}")
        print(f"  Output: {output_csv}")
        print("-" * 60)
        
        try:
            # Check if this is a Cohere model
            use_cohere = model_config.get("use_cohere_api", False)
            cohere_key = None
            if use_cohere:
                cohere_key = cohere_api_key
            
            process_qa_benchmark(
                labels_dir=labels_dir,
                images_dir=images_dir,
                output_csv=output_csv,
                model_name=model_id,
                open_router_api_key=open_router_api_key,
                url=url,
                temperature=temperature,
                use_cohere_api=use_cohere,
                cohere_api_key=cohere_key,
                max_workers=max_workers,
                limit_per_category=limit_per_category,
                limit_images=limit_images,
            )
            results_summary.append({
                "name": model_name,
                "csv": output_csv,
                "status": "success"
            })
            print(f"[SUCCESS] {model_name} completed")
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            results_summary.append({
                "name": model_name,
                "csv": output_csv,
                "status": "failed",
                "error": str(e)
            })
    
    print("\n" + "="*60)
    print("QA BENCHMARK COMPLETE")
    print("="*60)
    print("\nResults Summary:")
    for result in results_summary:
        status_icon = "[OK]" if result["status"] == "success" else "[FAIL]"
        print(f"  {status_icon} {result['name']}: {result['csv']}")
        if result["status"] == "failed" and "error" in result:
            print(f"      Error: {result['error'][:100]}")

