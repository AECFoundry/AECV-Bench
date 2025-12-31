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
import time
from pathlib import Path
from src.analyzers.openrouter import analyze_floorplan
from src.utils.image_utils import encode_image_to_base64
from src.utils.config import require_api_key
import requests

# API keys from environment variables
open_router_api_key = require_api_key('OPEN_ROUTER_API_KEY', 'OpenRouter')
cohere_api_key = require_api_key('COHERE_API_KEY', 'Cohere')

# Configuration
images_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images"
labels_dir = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels"
output_dir = "benchmark_result_qa"
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
    {
        "name": "Claude Opus 4.5",
        "model_id": "anthropic/claude-opus-4.5",
        "note": "Most advanced Opus model, optimized for complex reasoning tasks"
    },
    # {
    #     "name": "Claude Sonnet 4.5",
    #     "model_id": "anthropic/claude-sonnet-4.5",
    #     "note": "Most advanced Sonnet, optimized for real-world agents"
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


def ask_question_with_image(image_path: str, question: str, model_name: str, open_router_api_key: str, url: str, temperature: float = 0.0) -> str:
    """
    Send a question with an image to the model and get the answer.
    
    Args:
        image_path: Path to the image file
        question: The question to ask
        model_name: Model identifier
        open_router_api_key: API key
        url: API endpoint URL
        temperature: Sampling temperature
        
    Returns:
        The model's answer as a string
    """
    # Read and encode image
    base64_image = encode_image_to_base64(image_path)
    data_url = f"data:image/png;base64,{base64_image}"
    
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
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
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
                            "url": f"data:image/png;base64,{base64_image}"
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
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
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


def process_qa_benchmark(labels_dir: str, images_dir: str, output_csv: str, model_name: str, open_router_api_key: str, url: str, temperature: float = 0.0, use_cohere_api: bool = False, cohere_api_key: str = None):
    """
    Process QA pairs from label files and save results.
    
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
    """
    # Get all label files
    label_files = sorted(Path(labels_dir).glob("*.json"))
    
    if not label_files:
        print(f"Error: No label files found in {labels_dir}")
        return
    
    print(f"Found {len(label_files)} label files")
    print(f"Model: {model_name}")
    print(f"Processing QA pairs...\n")
    
    results = []
    total_questions = 0
    processed_questions = 0
    
    for label_file in label_files:
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)
        except Exception as e:
            print(f"[ERROR] Could not read {label_file}: {e}")
            continue
        
        image_id = label_data.get("image_id", label_file.stem)
        image_path_from_label = label_data.get("image_path", "")
        
        # Find the actual image file
        # Try different possible paths/extensions
        image_path = None
        possible_names = [
            f"{image_id}.png",
            f"{image_id}.jpg",
            f"{image_id}.jpeg",
            os.path.basename(image_path_from_label) if image_path_from_label else None
        ]
        
        for name in possible_names:
            if name:
                potential_path = os.path.join(images_dir, name)
                if os.path.isfile(potential_path):
                    image_path = potential_path
                    break
        
        if not image_path or not os.path.isfile(image_path):
            print(f"[WARN] Skipping {image_id}: image file not found")
            continue
        
        # Process qa_pairs (skip if not present)
        qa_pairs = label_data.get("qa_pairs", [])
        if not isinstance(qa_pairs, list):
            qa_pairs = []
        
        for qa in qa_pairs:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_q{total_questions}")
            
            if not question:
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature
                    )
                
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "qa_pairs",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                # Small delay to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "qa_pairs",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })
        
        # Process ocr_qa (skip if not present)
        ocr_qa = label_data.get("ocr_qa", [])
        if not isinstance(ocr_qa, list):
            ocr_qa = []
        
        for qa in ocr_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_ocr{total_questions}")
            
            if not question:
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature
                    )
                
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "ocr_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "ocr_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })
        
        # Process spatial_qa (skip if not present)
        spatial_qa = label_data.get("spatial_qa", [])
        if not isinstance(spatial_qa, list):
            spatial_qa = []
        
        for qa in spatial_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_sp{total_questions}")
            
            if not question:
                continue
            
            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature
                    )
                
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "spatial_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "spatial_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

        # Process counting_qa (skip if not present)
        counting_qa = label_data.get("counting_qa", [])
        if not isinstance(counting_qa, list):
            counting_qa = []

        for qa in counting_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_cnt{total_questions}")

            if not question:
                continue

            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature
                    )

                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "counting_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")

                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "counting_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

        # Process comparison_qa (skip if not present)
        comparison_qa = label_data.get("comparison_qa", [])
        if not isinstance(comparison_qa, list):
            comparison_qa = []

        for qa in comparison_qa:
            total_questions += 1
            question = qa.get("question", "")
            ground_truth = qa.get("answer", "")
            task = qa.get("task", "unknown")
            qa_id = qa.get("id", f"{image_id}_cmp{total_questions}")

            if not question:
                continue

            try:
                print(f"Processing {qa_id}: {question[:60]}...")
                if use_cohere_api:
                    model_answer = ask_question_with_image_cohere(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        cohere_api_key=cohere_api_key,
                        temperature=temperature
                    )
                else:
                    model_answer = ask_question_with_image(
                        image_path=image_path,
                        question=question,
                        model_name=model_name,
                        open_router_api_key=open_router_api_key,
                        url=url,
                        temperature=temperature
                    )

                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "comparison_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer
                })
                processed_questions += 1
                print(f"  Done ({processed_questions}/{total_questions})")

                time.sleep(0.5)

            except Exception as e:
                print(f"[ERROR] Failed to process {qa_id}: {e}")
                results.append({
                    "image_id": image_id,
                    "qa_id": qa_id,
                    "qa_type": "comparison_qa",
                    "task": task,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": f"[ERROR: {str(e)}]"
                })

    # Save results to CSV
    if results:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["image_id", "qa_id", "qa_type", "task", "question", "ground_truth", "model_answer"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"\n[SUCCESS] Results saved to {output_csv}")
        print(f"Total questions processed: {processed_questions}/{total_questions}")
    else:
        print("\n[WARN] No results to save")


if __name__ == "__main__":
    print("="*60)
    print("QA BENCHMARK - Testing Question/Answer Pairs")
    print("="*60)
    print(f"Labels directory: {labels_dir}")
    print(f"Images directory: {images_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Number of models: {len(models)}\n")
    
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
                cohere_api_key=cohere_key
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

