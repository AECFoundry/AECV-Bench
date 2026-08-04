"""Generate best-model-per-provider graphs in the existing AECV-Bench style:
   - Object-counting per-field accuracy heatmap (reuses src visualizer for identical style)
   - QA performance bar chart (proprietary=blue / open-source=green), + QA-type breakdown.
"""
import glob, os, csv
import numpy as np
import matplotlib.pyplot as plt
from src.benchmark.evaluator import compute_recalls_and_completeness
from src.benchmark.visualizer import plot_all_models_comparison

OC_DIR = "benchmark_result_object_counting"
JUDGE_DIR = "results/qa_llm_judge_results"
OUT_OC = "results/heatmap_outputs"
OUT_QA = "results/qa_llm_judge_results"

# Pretty display names (fall back to title-case for unknowns)
DISPLAY = {
    "gemini_35_flash": "Gemini 3.5 Flash", "gemini_3_pro_preview": "Gemini 3 Pro",
    "gemini_31_pro": "Gemini 3.1 Pro", "gemma_4_31b_it": "Gemma 4 31B",
    "claude_fable_5": "Claude Fable 5", "claude_opus_47": "Claude Opus 4.7",
    "claude_opus_48": "Claude Opus 4.8", "claude_opus_46": "Claude Opus 4.6",
    "claude_opus_45": "Claude Opus 4.5", "claude_sonnet_46": "Claude Sonnet 4.6",
    "openai_gpt_55": "GPT-5.5", "openai_gpt_54": "GPT-5.4", "openai_gpt_53": "GPT-5.3",
    "openai_gpt_52": "GPT-5.2", "openai_gpt_4_vision": "GPT-4o",
    "grok_43": "Grok 4.3", "grok_41_fast": "Grok 4.1 Fast",
    "qwen_37_plus": "Qwen 3.7 Plus", "qwen_35_plus": "Qwen 3.5 Plus",
    "qwen3_vl_8b_instruct": "Qwen3-VL 8B", "kimi_k26": "Kimi K2.6",
    "minimax_m3": "MiniMax M3", "stepfun_step_37_flash": "StepFun Step 3.7",
    "nvidia_nemotron_3_nano_omni_30b": "Nemotron 3 Omni 30B",
    "nvidia_nemotron_nano_12b_v2_vl": "Nemotron Nano 12B",
    "glm_46v": "GLM-4.6V", "mistral_large_2512": "Mistral Large 3",
    "amazon_nova_2_lite_v1": "Nova 2 Lite", "cohere_command_a_vision": "Cohere Command A",
}
OPEN_PROVIDERS = {"Alibaba", "Moonshot", "MiniMax", "StepFun", "NVIDIA", "Z-AI", "Mistral", "DeepSeek"}

# Pin specific (newest) models as the representative for a provider, so the SAME
# model is shown across both charts instead of whichever scored highest.
PINNED = {
    "OpenAI": "openai_gpt_55",
    "Alibaba": "qwen_37_plus",
    "xAI": "grok_43",
    "NVIDIA": "nvidia_nemotron_nano_12b_v2_vl",  # Omni 30B failed object counting -> use the 12B everywhere
}


def provider(safe):
    s = safe.lower()
    if "gemini" in s or "gemma" in s: return "Google"
    if "gpt" in s or "openai" in s: return "OpenAI"
    if "claude" in s: return "Anthropic"
    if "grok" in s: return "xAI"
    if "qwen" in s: return "Alibaba"
    if "kimi" in s: return "Moonshot"
    if "minimax" in s: return "MiniMax"
    if "step" in s: return "StepFun"
    if "nemotron" in s or "nvidia" in s: return "NVIDIA"
    if "glm" in s: return "Z-AI"
    if "mistral" in s: return "Mistral"
    if "nova" in s or "amazon" in s: return "Amazon"
    if "cohere" in s or "command" in s: return "Cohere"
    if "deepseek" in s: return "DeepSeek"
    return "Other"


def disp(safe):
    return DISPLAY.get(safe, safe.replace("_", " ").title())


# ---------- 1. OBJECT COUNTING: best per provider ----------
oc = {}  # safe -> (acc, nrows, path)
for f in glob.glob(os.path.join(OC_DIR, "*.csv")):
    safe = os.path.basename(f).replace(".csv", "")
    try:
        nrows = len(list(csv.DictReader(open(f))))
        if nrows < 100:  # skip badly-incomplete runs (e.g. failed Nemotron Omni 26/120)
            continue
        _, acc = compute_recalls_and_completeness(f)
        oc[safe] = (acc, nrows, f)
    except Exception:
        pass

best_oc = {}  # provider -> (acc, safe, path)
for safe, (acc, nr, f) in oc.items():
    p = provider(safe)
    if p in PINNED:
        continue  # handled below
    if p not in best_oc or acc > best_oc[p][0]:
        best_oc[p] = (acc, safe, f)
for p, safe in PINNED.items():  # force pinned model if it has OC data
    if safe in oc:
        acc, nr, f = oc[safe]
        best_oc[p] = (acc, safe, f)

oc_sorted = sorted(best_oc.items(), key=lambda kv: -kv[1][0])
oc_files = [v[2] for _, v in oc_sorted]
oc_names = [disp(v[1]) for _, v in oc_sorted]
print("OC best-per-provider:", [(p, disp(v[1]), round(v[0], 3)) for p, v in oc_sorted])
plot_all_models_comparison(
    csv_files=oc_files, model_names=oc_names, output_dir=OUT_OC,
    accuracy_filename="best_per_provider_accuracy_heatmap.png",
    mape_filename="best_per_provider_mape_heatmap.png")

# ---------- 2. QA: best per provider ----------
qa = {}  # safe -> score
for f in glob.glob(os.path.join(JUDGE_DIR, "*_evaluation_results.csv")):
    safe = os.path.basename(f).replace("_evaluation_results.csv", "")
    if safe in ("complete", "detailed"):
        continue
    rr = list(csv.DictReader(open(f)))
    sc = [float(r["overall"]) for r in rr if r.get("overall") not in (None, "")]
    if sc:
        qa[safe] = np.mean(sc)

best_qa = {}
for safe, s in qa.items():
    p = provider(safe)
    if p in PINNED:
        continue  # handled below
    if p not in best_qa or s > best_qa[p][0]:
        best_qa[p] = (s, safe)
for p, safe in PINNED.items():  # force pinned model if it has QA data
    if safe in qa:
        best_qa[p] = (qa[safe], safe)

qa_sorted = sorted(best_qa.items(), key=lambda kv: -kv[1][0])
names = [disp(v[1]) for _, v in qa_sorted]
scores = [v[0] for _, v in qa_sorted]
colors = ["#70AD47" if p in OPEN_PROVIDERS else "#4472C4" for p, _ in qa_sorted]
print("QA best-per-provider:", [(p, disp(v[1]), round(v[0], 3)) for p, v in qa_sorted])

fig, ax = plt.subplots(figsize=(12, 6))
x = range(len(names))
bars = ax.bar(x, scores, 0.6, color=colors)
ax.set_xlabel("Model (best per provider)", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy Score", fontsize=12, fontweight="bold")
ax.set_title("Best Model per Provider — Document QA", fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylim([0, 1]); ax.grid(axis="y", alpha=0.3)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#4472C4", label="Proprietary"),
                   Patch(facecolor="#70AD47", label="Open-Source")], loc="upper right")
for b, s in zip(bars, scores):
    ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{s:.3f}", ha="center", va="bottom", fontsize=10)
plt.tight_layout()
for ext in ("png", "svg"):
    plt.savefig(os.path.join(OUT_QA, f"best_per_provider_qa_comparison.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_QA}/best_per_provider_qa_comparison.png")

# ---------- 3. QA-type breakdown for best-per-provider ----------
QA_TYPES = ["spatial_qa", "ocr_qa", "comparison_qa", "counting_qa"]
QA_LABELS = {"spatial_qa": "Spatial", "ocr_qa": "Text (OCR)", "comparison_qa": "Comparative", "counting_qa": "Counting"}
QA_COLORS = {"spatial_qa": "#E8956A", "ocr_qa": "#6A9BE8", "comparison_qa": "#6AE89B", "counting_qa": "#B86AE8"}


def type_scores(safe):
    f = os.path.join(JUDGE_DIR, f"{safe}_evaluation_results.csv")
    rr = list(csv.DictReader(open(f)))
    out = {}
    for t in QA_TYPES:
        v = [float(r["overall"]) for r in rr if r.get("qa_type") == t and r.get("overall") not in (None, "")]
        out[t] = np.mean(v) if v else 0.0
    return out

bd = {disp(v[1]): type_scores(v[1]) for _, v in qa_sorted}
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(names)); w = 0.8 / len(QA_TYPES)
for i, t in enumerate(QA_TYPES):
    vals = [bd[n][t] for n in names]
    ax.bar(x + (i - len(QA_TYPES)/2) * w + w/2, vals, w, label=QA_LABELS[t], color=QA_COLORS[t])
ax.set_xlabel("Model (best per provider)", fontsize=12, fontweight="bold")
ax.set_ylabel("Mean Score", fontsize=12, fontweight="bold")
ax.set_title("Best Model per Provider — QA Performance by Task Type", fontsize=14, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
ax.set_ylim([0, 1]); ax.legend(loc="upper right"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
for ext in ("png", "svg"):
    plt.savefig(os.path.join(OUT_QA, f"best_per_provider_qa_type_breakdown.{ext}"),
                dpi=300 if ext == "png" else None, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_QA}/best_per_provider_qa_type_breakdown.png")
