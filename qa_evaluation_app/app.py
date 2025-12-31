"""
QA Evaluation Web App

A Flask-based web application for manually evaluating VLM responses on the QA benchmark.
Displays floor plan images with questions and allows users to mark model responses as correct/incorrect.
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_file, abort

app = Flask(__name__)

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_QA_DIR = PROJECT_ROOT / "benchmark_result_qa"
IMAGES_DIR = PROJECT_ROOT / "data" / "Use Case 2 - Drawing Understanding" / "01 - Full Dataset" / "images"
EVALUATIONS_DIR = Path(__file__).parent / "evaluations"
EVALUATIONS_FILE = EVALUATIONS_DIR / "evaluations.json"

# Ensure evaluations directory exists
EVALUATIONS_DIR.mkdir(exist_ok=True)

# Global data storage
qa_data = []  # List of unique QA items with all model responses
models = []   # List of model names


def load_evaluations():
    """Load existing evaluations from JSON file."""
    if EVALUATIONS_FILE.exists():
        try:
            with open(EVALUATIONS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {"evaluations": {}, "metadata": {"last_updated": None, "total_evaluated": 0}}


def save_evaluations(data):
    """Save evaluations to JSON file."""
    data["metadata"]["last_updated"] = datetime.now().isoformat()
    with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_csv_files():
    """Load all CSV files from benchmark_result_qa and merge by (image_id, qa_id)."""
    global qa_data, models

    # Dictionary to collect data: key = (image_id, qa_id), value = dict with metadata + model answers
    qa_dict = {}
    models = []

    # Find all CSV files
    csv_files = list(BENCHMARK_QA_DIR.glob("qa_results_*.csv"))

    for csv_file in csv_files:
        # Extract model name from filename
        # Format: qa_results_<model_name>.csv
        model_name = csv_file.stem.replace("qa_results_", "")
        models.append(model_name)

        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = row.get("image_id", "")
                    qa_id = row.get("qa_id", "")
                    key = f"{image_id}_{qa_id}"

                    if key not in qa_dict:
                        qa_dict[key] = {
                            "key": key,
                            "image_id": image_id,
                            "qa_id": qa_id,
                            "qa_type": row.get("qa_type", ""),
                            "task": row.get("task", ""),
                            "question": row.get("question", ""),
                            "ground_truth": row.get("ground_truth", ""),
                            "model_answers": {}
                        }

                    # Add this model's answer
                    qa_dict[key]["model_answers"][model_name] = row.get("model_answer", "")

        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    # Convert to sorted list
    qa_data = sorted(qa_dict.values(), key=lambda x: (x["image_id"], x["qa_id"]))
    models = sorted(models)

    print(f"Loaded {len(qa_data)} QA items from {len(models)} models")


# Load data on startup
load_csv_files()


@app.route('/')
def index():
    """Render the main evaluation interface."""
    return render_template('index.html')


@app.route('/api/qa/<int:index>')
def get_qa(index):
    """Get QA item data by index."""
    if index < 0 or index >= len(qa_data):
        return jsonify({"error": "Index out of range"}), 404

    item = qa_data[index]
    evaluations = load_evaluations()

    # Get existing evaluations for this QA item
    item_evaluations = evaluations["evaluations"].get(item["key"], {})

    return jsonify({
        "index": index,
        "total": len(qa_data),
        "key": item["key"],
        "image_id": item["image_id"],
        "qa_id": item["qa_id"],
        "qa_type": item["qa_type"],
        "task": item["task"],
        "question": item["question"],
        "ground_truth": item["ground_truth"],
        "model_answers": item["model_answers"],
        "evaluations": item_evaluations
    })


@app.route('/api/image/<image_id>')
def get_image(image_id):
    """Serve floor plan image by image_id."""
    # Try different extensions
    for ext in ['.png', '.jpg', '.jpeg']:
        image_path = IMAGES_DIR / f"{image_id}{ext}"
        if image_path.exists():
            return send_file(image_path, mimetype=f'image/{ext[1:]}')

    abort(404, description=f"Image not found: {image_id}")


@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Save an evaluation for a model's response."""
    data = request.get_json()

    qa_key = data.get("qa_key")
    model_name = data.get("model_name")
    is_correct = data.get("is_correct")

    if not all([qa_key, model_name, is_correct is not None]):
        return jsonify({"error": "Missing required fields"}), 400

    evaluations = load_evaluations()

    # Initialize QA key if not exists
    if qa_key not in evaluations["evaluations"]:
        evaluations["evaluations"][qa_key] = {}

    # Save evaluation
    evaluations["evaluations"][qa_key][model_name] = {
        "is_correct": is_correct,
        "timestamp": datetime.now().isoformat()
    }

    # Update total count
    total = sum(len(v) for v in evaluations["evaluations"].values())
    evaluations["metadata"]["total_evaluated"] = total

    save_evaluations(evaluations)

    return jsonify({"success": True, "total_evaluated": total})


@app.route('/api/stats')
def get_stats():
    """Get evaluation progress statistics."""
    evaluations = load_evaluations()

    # Count evaluated items per model
    model_counts = {model: 0 for model in models}
    for qa_key, model_evals in evaluations["evaluations"].items():
        for model_name in model_evals:
            if model_name in model_counts:
                model_counts[model_name] += 1

    # Count fully evaluated QA items (all models evaluated)
    fully_evaluated = 0
    for item in qa_data:
        item_evals = evaluations["evaluations"].get(item["key"], {})
        if len(item_evals) == len(models):
            fully_evaluated += 1

    return jsonify({
        "total_qa_items": len(qa_data),
        "total_models": len(models),
        "fully_evaluated": fully_evaluated,
        "model_counts": model_counts,
        "total_evaluations": evaluations["metadata"].get("total_evaluated", 0)
    })


@app.route('/api/models')
def get_models():
    """Get list of available models."""
    return jsonify({"models": models})


@app.route('/api/export')
def export_evaluations():
    """Export evaluations as downloadable JSON."""
    evaluations = load_evaluations()

    # Enrich with question/answer data
    export_data = {
        "metadata": evaluations["metadata"],
        "evaluations": []
    }

    for item in qa_data:
        item_evals = evaluations["evaluations"].get(item["key"], {})
        if item_evals:
            export_data["evaluations"].append({
                "image_id": item["image_id"],
                "qa_id": item["qa_id"],
                "qa_type": item["qa_type"],
                "task": item["task"],
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "model_evaluations": item_evals
            })

    return jsonify(export_data)


@app.route('/api/search')
def search_qa():
    """Search QA items by image_id or question text."""
    query = request.args.get('q', '').lower()
    results = []

    for i, item in enumerate(qa_data):
        if (query in item["image_id"].lower() or
            query in item["question"].lower() or
            query in item["qa_id"].lower()):
            results.append({
                "index": i,
                "key": item["key"],
                "image_id": item["image_id"],
                "qa_id": item["qa_id"],
                "question": item["question"][:80] + "..." if len(item["question"]) > 80 else item["question"]
            })

    return jsonify({"results": results[:20]})  # Limit to 20 results


if __name__ == '__main__':
    print(f"\nQA Evaluation Web App")
    print(f"=" * 50)
    print(f"Loaded {len(qa_data)} QA items from {len(models)} models")
    print(f"Models: {', '.join(models)}")
    print(f"\nOpen http://localhost:5000 in your browser")
    print(f"=" * 50)
    app.run(debug=True, port=5000)
