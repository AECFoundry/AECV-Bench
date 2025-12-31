"""
Evaluation metrics computation for benchmark results.
"""
import csv
import json
from typing import Dict, Tuple


def compute_recalls_and_completeness(csv_path: str) -> Tuple[Dict[str, float], float]:
    """
    Compute recall and completeness metrics from benchmark CSV results.
    
    Works whether 'extracted' is:
      • a plain JSON object   -> {"Door":22,...}
      • or a doubly-encoded   -> "{\"Door\":22,...}"
    
    Args:
        csv_path: Path to the CSV file with benchmark results
        
    Returns:
        Tuple of (recalls_dict, mean_recall) where:
        - recalls_dict: Dictionary mapping field names to recall values
        - mean_recall: Average recall across all fields
    """
    total = 0
    exact = {f: 0 for f in ("Door", "Window", "Space", "Bedroom", "Toilet")}
    completeness = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        # Support both 'name' and 'namee' for backward compatibility
        name_col = 'name' if 'name' in reader.fieldnames else 'namee'
        
        for row in reader:
            total += 1
            orig = json.loads(row["original"])

            # 1-or-2 step decode for 'extracted'
            extracted_str = row["extracted"].replace('\\', '')  # Remove escape characters
            loaded = json.loads(extracted_str)
            ext = json.loads(loaded) if isinstance(loaded, str) else loaded

            all_match = True
            for field in exact:
                if ext.get(field, 0) == orig.get(field, 0):
                    exact[field] += 1
                else:
                    all_match = False
            completeness[row[name_col]] = 1 if all_match else 0

    if total == 0:
        raise ValueError("No valid rows found in CSV file")
    
    recalls = {f: exact[f] / total for f in exact}
    mean_recall = sum(recalls.values()) / len(recalls)
    avg_compl = sum(completeness.values()) / total
    return recalls, mean_recall

