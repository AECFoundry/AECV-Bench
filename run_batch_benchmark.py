"""
Batch runner: for each agreed model, run object counting (parallel) -> QA (parallel)
-> GPT-4o LLM judge (parallel). Saves results, cost, latency. Resumable.
"""
import csv, json, os, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean as _mean, median as _median
from dotenv import load_dotenv
load_dotenv(".env")

from src.benchmark.processor import process_benchmark_floorplans
from src.benchmark.evaluator import compute_recalls_and_completeness
from src.benchmark.qa_evaluator import QAEvaluator
from src.analyzers.openrouter import analyze_floorplan, analyze_floorplan_prompt_based
from src.models.plan_elements import get_json_schema
from src.utils.pricing import compute_cost
from run_qa_benchmark import process_qa_benchmark, ask_question_with_image, sanitize_error

KEY = os.getenv("OPEN_ROUTER_API_KEY")
URL = "https://openrouter.ai/api/v1/chat/completions"
OC_DIR = os.getenv("BATCH_OC_DIR", "benchmark_result_object_counting")
QA_DIR = os.getenv("BATCH_QA_DIR", "benchmark_result_qa")
JUDGE_DIR = os.getenv("BATCH_JUDGE_DIR", "results/qa_llm_judge_results")
QA_IMAGES = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/images"
QA_LABELS = "data/Use Case 2 - Drawing Understanding/01 - Full Dataset/labels"
OC_DATA = "data/Use Case 1 - Object Counting/1 - Full Datasets"
MASTER = os.getenv("BATCH_MASTER", "results/new_models_batch_results.json")
WORKERS = int(os.getenv("BATCH_WORKERS", "20"))
NUM_FOLDERS = int(os.getenv("BATCH_NUM_FOLDERS", "120"))
QA_LIMIT_IMAGES = int(os.getenv("BATCH_QA_LIMIT_IMAGES")) if os.getenv("BATCH_QA_LIMIT_IMAGES") else None
QA_LIMIT_PER_CAT = int(os.getenv("BATCH_QA_LIMIT_PER_CAT")) if os.getenv("BATCH_QA_LIMIT_PER_CAT") else None
ONLY = os.getenv("BATCH_ONLY")  # substring filter on model name

# (name, model_id, oc_uses_schema)
# Order matters: fast models first; ultra-slow streaming models (Kimi) run LAST
# so they don't block the rest.
MODELS = [
    ("Claude Opus 4.7", "anthropic/claude-opus-4.7", True),
    ("Claude Fable 5", "anthropic/claude-fable-5", True),
    ("OpenAI GPT-5.5", "openai/gpt-5.5", True),
    ("Gemini 3.5 Flash", "google/gemini-3.5-flash", True),
    ("Grok 4.3", "x-ai/grok-4.3", True),
    ("MiniMax M3", "minimax/minimax-m3", False),
    ("Qwen 3.7 Plus", "qwen/qwen3.7-plus", True),
    ("Gemma 4 31B IT", "google/gemma-4-31b-it", False),
    ("StepFun Step 3.7 Flash", "stepfun/step-3.7-flash", False),
    ("NVIDIA Nemotron 3 Nano Omni 30B", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", False),
    ("Kimi K2.6", "moonshotai/kimi-k2.6", False),  # very slow streamer -> last + high concurrency
]

# Per-model concurrency override (slow streamers get more workers to hide latency)
MODEL_WORKERS = {"Kimi K2.6": 50}


def safe(name):
    return name.lower().replace(" ", "_").replace(".", "").replace("-", "_")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_master():
    if os.path.exists(MASTER):
        return json.load(open(MASTER))
    return {}


def save_master(m):
    os.makedirs(os.path.dirname(MASTER), exist_ok=True)
    json.dump(m, open(MASTER, "w"), indent=2)


def resolve_qa_image(image_id):
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(QA_IMAGES, f"{image_id}{ext}")
        if os.path.isfile(p):
            return p
    return None


def retry_qa_failures(qa_csv, model_id):
    """Re-run [ERROR] rows once (keyed by unique (image_id, qa_id)), patch CSV in place."""
    rows = list(csv.DictReader(open(qa_csv)))
    fails = [r for r in rows if str(r["model_answer"]).startswith("[ERROR")]
    if not fails:
        return 0
    log(f"    retrying {len(fails)} QA failures at 8 workers...")

    def run(r):
        img = resolve_qa_image(r["image_id"])
        if not img:
            return r["image_id"], r["qa_id"], None
        usage = {}
        t = time.time()
        try:
            ans = ask_question_with_image(image_path=img, question=r["question"], model_name=model_id,
                                          open_router_api_key=KEY, url=URL, temperature=0.0, usage_out=usage)
        except Exception as e:
            ans = sanitize_error(e)
        lat = round(time.time() - t, 2)
        pt, ct = usage.get("prompt_tokens"), usage.get("completion_tokens")
        cost = compute_cost(model_id, pt, ct) if (pt is not None or ct is not None) else None
        return r["image_id"], r["qa_id"], {
            "model_answer": ans, "prompt_tokens": pt, "completion_tokens": ct,
            "cost_usd": round(cost, 6) if cost is not None else None, "latency_s": lat}

    patches = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(run, r) for r in fails]):
            iid, qid, upd = f.result()
            if upd and not str(upd["model_answer"]).startswith("[ERROR"):
                patches[(iid, qid)] = upd
    for r in rows:
        k = (r["image_id"], r["qa_id"])
        if k in patches:
            r.update(patches[k])
    fn = ["image_id", "qa_id", "qa_type", "task", "question", "ground_truth",
          "model_answer", "prompt_tokens", "completion_tokens", "cost_usd", "latency_s"]
    with open(qa_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fn})
    return len(patches)


def summarize_qa(qa_csv):
    rows = list(csv.DictReader(open(qa_csv)))
    def fl(v):
        try:
            return float(v) if v not in (None, "") else None
        except (TypeError, ValueError):
            return None
    lat = [fl(r["latency_s"]) for r in rows if fl(r["latency_s"]) is not None]
    cost = [fl(r["cost_usd"]) for r in rows if fl(r["cost_usd"]) is not None]
    errs = sum(1 for r in rows if str(r["model_answer"]).startswith("[ERROR"))
    return {"total": len(rows), "errors": errs,
            "cost_usd": round(sum(cost), 4) if cost else None,
            "mean_latency_s": round(_mean(lat), 2) if lat else None}


def run_object_counting(name, model_id, use_schema, workers):
    sn = safe(name)
    out_csv = os.path.join(OC_DIR, f"{sn}.csv")
    t0 = time.time()
    analyzer = analyze_floorplan if use_schema else analyze_floorplan_prompt_based
    process_benchmark_floorplans(
        benchmark_dir=OC_DATA, output_csv=out_csv, output_json_name=f"{sn}.json",
        num_folders=NUM_FOLDERS, model_name=model_id, json_schema=get_json_schema(),
        open_router_api_key=KEY, url=URL, analyzer_func=analyzer, max_workers=workers)
    # Fallback: if schema produced nothing, retry prompt-based
    if use_schema and (not os.path.exists(out_csv) or len(list(csv.DictReader(open(out_csv)))) == 0):
        log(f"    schema OC produced 0 rows; falling back to prompt-based")
        process_benchmark_floorplans(
            benchmark_dir=OC_DATA, output_csv=out_csv, output_json_name=f"{sn}.json",
            num_folders=NUM_FOLDERS, model_name=model_id, json_schema=get_json_schema(),
            open_router_api_key=KEY, url=URL, analyzer_func=analyze_floorplan_prompt_based,
            max_workers=workers)
    wall = round(time.time() - t0, 1)
    summary = json.load(open(os.path.splitext(out_csv)[0] + "_summary.json"))
    try:
        _, acc = compute_recalls_and_completeness(out_csv)
    except Exception as e:
        acc = None
        log(f"    OC accuracy compute failed: {e}")
    return {"accuracy": round(acc, 4) if acc is not None else None,
            "cost_usd": summary.get("total_cost_usd"),
            "wall_clock_s": summary.get("wall_clock_s"),
            "succeeded": summary.get("folders_succeeded"), "total": summary.get("folders_total"),
            "mean_latency_s": summary.get("mean_latency_s")}


def run_qa(name, model_id, workers):
    sn = safe(name)
    out_csv = os.path.join(QA_DIR, f"qa_results_{sn}.csv")
    t0 = time.time()
    process_qa_benchmark(labels_dir=QA_LABELS, images_dir=QA_IMAGES, output_csv=out_csv,
                         model_name=model_id, open_router_api_key=KEY, url=URL,
                         temperature=0.0, max_workers=workers,
                         limit_per_category=QA_LIMIT_PER_CAT, limit_images=QA_LIMIT_IMAGES)
    retry_qa_failures(out_csv, model_id)
    wall = round(time.time() - t0, 1)
    s = summarize_qa(out_csv)
    s["wall_clock_s"] = wall
    return out_csv, s


def run_judge(name, qa_csv):
    sn = safe(name)
    t0 = time.time()
    ev = QAEvaluator(judge_model="openai/gpt-4o", open_router_api_key=KEY, max_workers=WORKERS)
    res = ev.evaluate_csv(qa_csv)
    wall = round(time.time() - t0, 1)
    # Save evaluation CSV alongside the existing ones
    os.makedirs(JUDGE_DIR, exist_ok=True)
    out = os.path.join(JUDGE_DIR, f"{sn}_evaluation_results.csv")
    if res["results"]:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(res["results"][0].keys()))
            w.writeheader()
            w.writerows(res["results"])
    bt = {k: round(v["mean_score"], 4) for k, v in res["summary"].get("qa_type_breakdown", {}).items()}
    return {"score": round(res["summary"]["mean_overall_score"], 4),
            "by_type": bt, "wall_clock_s": wall}


def main():
    master = load_master()
    models = [m for m in MODELS if (not ONLY or ONLY.lower() in m[0].lower())]
    for i, (name, model_id, use_schema) in enumerate(models, 1):
        if name in master and master[name].get("done"):
            log(f"[{i}/{len(models)}] SKIP {name} (already done)")
            continue
        w = MODEL_WORKERS.get(name, WORKERS)
        log(f"[{i}/{len(models)}] ===== {name} ({model_id}) [workers={w}] =====")
        entry = {"model_id": model_id}
        try:
            log(f"  -> object counting (120, {w} workers)")
            entry["object_counting"] = run_object_counting(name, model_id, use_schema, w)
            log(f"     OC acc={entry['object_counting']['accuracy']} "
                f"cost=${entry['object_counting']['cost_usd']} wall={entry['object_counting']['wall_clock_s']}s "
                f"ok={entry['object_counting']['succeeded']}/{entry['object_counting']['total']}")

            log(f"  -> QA (192, {w} workers)")
            qa_csv, qa_s = run_qa(name, model_id, w)
            entry["qa"] = qa_s
            log(f"     QA cost=${qa_s['cost_usd']} wall={qa_s['wall_clock_s']}s errors={qa_s['errors']}")

            log(f"  -> LLM judge (GPT-4o, {WORKERS} workers)")
            entry["judge"] = run_judge(name, qa_csv)
            log(f"     QA score={entry['judge']['score']} ({entry['judge']['by_type']})")

            entry["done"] = True
            entry["total_cost_usd"] = round((entry["object_counting"].get("cost_usd") or 0)
                                            + (entry["qa"].get("cost_usd") or 0), 4)
        except Exception as e:
            import traceback
            entry["error"] = f"{e}"
            log(f"  !! FAILED {name}: {e}\n{traceback.format_exc()}")
        master[name] = entry
        save_master(master)
        log(f"  saved master ({sum(1 for v in master.values() if v.get('done'))}/{len(models)} done)")
    log("===== BATCH COMPLETE =====")
    # Print final table
    print("\n=== FINAL BATCH RESULTS ===")
    print(f"{'MODEL':<34}{'OC_acc':>8}{'QA_score':>9}{'$OC':>8}{'$QA':>8}{'OCwall':>8}{'QAwall':>8}")
    for name, _, _ in MODELS:
        e = master.get(name, {})
        if not e.get("done"):
            print(f"{name:<34}  (not done: {e.get('error','?')[:40]})")
            continue
        oc, qa, jd = e["object_counting"], e["qa"], e["judge"]
        print(f"{name:<34}{(oc['accuracy'] or 0)*100:>7.1f}%{(jd['score'] or 0)*100:>8.1f}%"
              f"{oc['cost_usd'] or 0:>8.2f}{qa['cost_usd'] or 0:>8.2f}"
              f"{oc['wall_clock_s'] or 0:>8.0f}{qa['wall_clock_s'] or 0:>8.0f}")


if __name__ == "__main__":
    main()
