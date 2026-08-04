"""
Cost computation helpers for OpenRouter models.

Pricing is fetched once from the OpenRouter models API and cached in-process.
Costs are derived from token usage returned by the API on each call, so they
reflect what OpenRouter actually bills (input + output tokens).
"""
import threading
from typing import Dict, Optional

import requests

_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Module-level cache: model_id -> {"prompt": $/token, "completion": $/token}
_price_cache: Optional[Dict[str, Dict[str, float]]] = None
_lock = threading.Lock()


def _load_pricing() -> Dict[str, Dict[str, float]]:
    """Fetch and cache per-token pricing for all OpenRouter models (thread-safe)."""
    global _price_cache
    if _price_cache is not None:
        return _price_cache
    with _lock:
        if _price_cache is not None:
            return _price_cache
        prices: Dict[str, Dict[str, float]] = {}
        try:
            data = requests.get(_MODELS_URL, timeout=30).json().get("data", [])
            for m in data:
                p = m.get("pricing", {}) or {}
                prices[m["id"]] = {
                    "prompt": float(p.get("prompt", 0) or 0),
                    "completion": float(p.get("completion", 0) or 0),
                }
        except Exception as e:  # offline / API error -> empty cache, cost falls back to 0
            print(f"[WARN] Could not load OpenRouter pricing: {e}")
        _price_cache = prices
        return _price_cache


def compute_cost(model_id: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """
    Compute USD cost for a single call from token usage.

    Returns None if pricing for the model is unknown (so callers can distinguish
    "unpriced" from "$0.00").
    """
    prices = _load_pricing()
    rate = prices.get(model_id)
    if rate is None:
        return None
    return (prompt_tokens or 0) * rate["prompt"] + (completion_tokens or 0) * rate["completion"]
