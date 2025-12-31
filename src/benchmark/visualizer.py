"""
Visualization utilities for benchmark results.
"""
import csv
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional, Dict


def plot_benchmark_results(
    csv_path: str,
    output_dir: str = "results",
    excluded_names: Optional[List[str]] = None,
    model_name: Optional[str] = None
):
    """
    Generate visualization heatmaps from benchmark CSV results.
    
    Args:
        csv_path: Path to the CSV file with benchmark results
        output_dir: Directory to save visualization images
        excluded_names: List of sample names to exclude from analysis
        model_name: Optional model name to display (if None, uses filename)
    """
    if excluded_names is None:
        excluded_names = []
    
    fields = ["Door", "Window", "Space", "Bedroom", "Toilet"]
    # Fields to display in heatmap (excluding Space)
    heatmap_fields = [f for f in fields if f != "Space"]
    
    # Use model_name or derive from filename
    if model_name is None:
        model_name = os.path.basename(csv_path).replace('.csv', '')
    
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    # Compute metrics
    total = 0
    exact = {f: 0 for f in fields}
    mape_sums = {f: 0.0 for f in fields}
    mape_counts = {f: 0 for f in fields}
    processed_count = 0
    
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        name_col = 'name' if 'name' in reader.fieldnames else 'namee'
        if name_col not in reader.fieldnames:
            print(f"Error: '{csv_path}' does not have 'name' or 'namee' column")
            return
        
        for row in reader:
            # Check if the name should be excluded
            if row[name_col] in excluded_names:
                continue
            
            processed_count += 1
            orig = json.loads(row["original"])
            loaded = json.loads(row["extracted"])
            ext = json.loads(loaded) if isinstance(loaded, str) else loaded
            
            for field in fields:
                orig_val = orig.get(field, 0)
                ext_val = ext.get(field, 0)
                
                # Recall/accuracy count
                if ext_val == orig_val:
                    exact[field] += 1
                
                # MAPE calculation
                if orig_val == 0:
                    if ext_val == 0:
                        mape_sums[field] += 0.0
                        mape_counts[field] += 1
                    else:
                        # Infinite relative error: skip this entry
                        continue
                else:
                    err = abs(ext_val - orig_val) / abs(orig_val)
                    mape_sums[field] += err
                    mape_counts[field] += 1
    
    if processed_count == 0:
        print(f"No valid rows found in {csv_path}")
        return
    
    # Compute recalls
    recalls = {f: exact[f] / processed_count for f in fields}
    mean_accuracy = np.mean(list(recalls.values()))
    
    # Compute per-field MAPE
    mapes = {}
    for f in fields:
        if mape_counts[f] > 0:
            mapes[f] = mape_sums[f] / mape_counts[f]
        else:
            mapes[f] = np.nan
    
    # Mean MAPE across fields
    valid_mapes = [v for v in mapes.values() if not np.isnan(v)]
    mean_mape = np.mean(valid_mapes) if valid_mapes else np.nan
    
    # Create DataFrame
    results_data = {
        "model": [model_name],
        **{f"recall_{f}": [recalls[f]] for f in fields},
        **{f"mape_{f}": [mapes[f]] for f in fields},
        "mean_accuracy": [mean_accuracy],
        "mean_mape": [mean_mape]
    }
    
    df = pd.DataFrame(results_data).set_index("model")
    
    # Split into recall and MAPE DataFrames
    recall_cols = [f"recall_{f}" for f in fields] + ["mean_accuracy"]
    mape_cols = [f"mape_{f}" for f in fields] + ["mean_mape"]
    
    df_rec = df[recall_cols].copy()
    df_map = df[mape_cols].copy()
    
    # Rename columns
    recall_rename = {f"recall_{f}": f for f in fields}
    recall_rename["mean_accuracy"] = "Mean Accuracy"
    df_rec.rename(columns=recall_rename, inplace=True)
    
    map_rename = {f"mape_{f}": f for f in fields}
    map_rename["mean_mape"] = "Mean MAPE"
    df_map.rename(columns=map_rename, inplace=True)
    
    # Sort
    df_rec_sorted = df_rec.sort_values("Mean Accuracy", ascending=False)
    df_map_sorted = df_map.sort_values("Mean MAPE", ascending=True)
    df_map_sorted_pct = df_map_sorted * 100
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
    })
    
    def plot_heatmap(data, title, fmt, cmap='Blues', cbar_label="", figsize=(10, 6), save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data.values, aspect='auto', cmap=cmap)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=10)
        
        max_val = np.nanmax(data.values)
        threshold = max_val / 2 if not np.isnan(max_val) else 0.5
        
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if not np.isnan(val):
                    text = fmt.format(val)
                    color = 'white' if val > threshold else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        
        if save_path:
            # Save PNG version
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            # Save SVG version
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path}")
        else:
            plt.show()
        plt.close()

    # Plot recall heatmap (excluding Space)
    recall_path = os.path.join(output_dir, f"{model_name}_accuracy_heatmap.png")
    plot_heatmap(
        df_rec_sorted[["Mean Accuracy"] + heatmap_fields],
        title=f"Per-Field Accuracy: {model_name}",
        fmt="{:.2f}",
        cmap='Blues',
        cbar_label="Accuracy",
        save_path=recall_path
    )
    
    # Plot MAPE heatmap (excluding Space)
    mape_path = os.path.join(output_dir, f"{model_name}_mape_heatmap.png")
    plot_heatmap(
        df_map_sorted_pct[["Mean MAPE"] + heatmap_fields],
        title=f"Per-Field MAPE (%): {model_name}",
        fmt="{:.1f}%",
        cmap='Blues',
        cbar_label="MAPE (%)",
        save_path=mape_path
    )
    
    print(f"\nVisualizations saved to {output_dir}/")
    print(f"  - {model_name}_accuracy_heatmap.png")
    print(f"  - {model_name}_mape_heatmap.png")
    
    return {
        "recalls": recalls,
        "mean_accuracy": mean_accuracy,
        "mapes": mapes,
        "mean_mape": mean_mape
    }


def plot_all_models_comparison(
    csv_files: List[str],
    model_names: List[str],
    output_dir: str = "results",
    excluded_names: Optional[List[str]] = None,
    accuracy_filename: Optional[str] = None,
    mape_filename: Optional[str] = None
):
    """
    Generate combined visualization comparing multiple models.
    
    Args:
        csv_files: List of CSV file paths
        model_names: List of model names (must match csv_files order)
        output_dir: Directory to save visualization images
        excluded_names: List of sample names to exclude from analysis
        accuracy_filename: Optional custom filename for accuracy heatmap (default: "all_models_accuracy_heatmap.png")
        mape_filename: Optional custom filename for MAPE heatmap (default: "all_models_mape_heatmap.png")
    """
    if excluded_names is None:
        excluded_names = []
    
    fields = ["Door", "Window", "Space", "Bedroom", "Toilet"]
    # Fields to display in heatmap (excluding Space)
    heatmap_fields = [f for f in fields if f != "Space"]
    results = []
    
    # Process each CSV file
    for csv_path, model_name in zip(csv_files, model_names):
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping.")
            continue
        
        total = 0
        exact = {f: 0 for f in fields}
        mape_sums = {f: 0.0 for f in fields}
        mape_counts = {f: 0 for f in fields}
        processed_count = 0
        
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            name_col = 'name' if 'name' in reader.fieldnames else 'namee'
            if name_col not in reader.fieldnames:
                print(f"Warning: '{csv_path}' does not have 'name' or 'namee' column, skipping.")
                continue
            
            for row in reader:
                if row[name_col] in excluded_names:
                    continue
                
                processed_count += 1
                orig = json.loads(row["original"])
                loaded = json.loads(row["extracted"])
                ext = json.loads(loaded) if isinstance(loaded, str) else loaded
                
                for field in fields:
                    orig_val = orig.get(field, 0)
                    ext_val = ext.get(field, 0)
                    
                    if ext_val == orig_val:
                        exact[field] += 1
                    
                    if orig_val == 0:
                        if ext_val == 0:
                            mape_sums[field] += 0.0
                            mape_counts[field] += 1
                        else:
                            continue
                    else:
                        err = abs(ext_val - orig_val) / abs(orig_val)
                        mape_sums[field] += err
                        mape_counts[field] += 1
        
        if processed_count == 0:
            print(f"No non-excluded rows found in {csv_path}, skipping.")
            continue
        
        recalls = {f: exact[f] / processed_count for f in fields}
        mean_accuracy = np.mean(list(recalls.values()))
        
        mapes = {}
        for f in fields:
            if mape_counts[f] > 0:
                mapes[f] = mape_sums[f] / mape_counts[f]
            else:
                mapes[f] = np.nan
        
        valid_mapes = [v for v in mapes.values() if not np.isnan(v)]
        mean_mape = np.mean(valid_mapes) if valid_mapes else np.nan
        
        results.append({
            "model": model_name,
            **{f"recall_{f}": recalls[f] for f in fields},
            **{f"mape_{f}": mapes[f] for f in fields},
            "mean_accuracy": mean_accuracy,
            "mean_mape": mean_mape
        })
    
    if not results:
        print("No results to visualize")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results).set_index("model")
    
    # Split into recall and MAPE DataFrames
    recall_cols = [f"recall_{f}" for f in fields] + ["mean_accuracy"]
    mape_cols = [f"mape_{f}" for f in fields] + ["mean_mape"]
    
    df_rec = df[recall_cols].copy()
    df_map = df[mape_cols].copy()
    
    # Rename columns
    recall_rename = {f"recall_{f}": f for f in fields}
    recall_rename["mean_accuracy"] = "Mean Accuracy"
    df_rec.rename(columns=recall_rename, inplace=True)
    
    map_rename = {f"mape_{f}": f for f in fields}
    map_rename["mean_mape"] = "Mean MAPE"
    df_map.rename(columns=map_rename, inplace=True)
    
    # Sort by Mean Accuracy (descending)
    df_rec_sorted = df_rec.sort_values("Mean Accuracy", ascending=False)
    df_map_sorted = df_map.sort_values("Mean MAPE", ascending=True)
    df_map_sorted_pct = df_map_sorted * 100
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Styling
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
    })
    
    def plot_heatmap(data, title, fmt, cmap='Blues', cbar_label="", figsize=(10, 6), save_path=None):
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(data.values, aspect='auto', cmap=cmap)
        
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom", fontsize=10)
        
        ax.set_yticks(np.arange(len(data.index)))
        ax.set_yticklabels(data.index, fontsize=10)
        ax.set_xticks(np.arange(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right', fontsize=10)
        
        max_val = np.nanmax(data.values)
        threshold = max_val / 2 if not np.isnan(max_val) else 0.5
        
        for i in range(len(data.index)):
            for j in range(len(data.columns)):
                val = data.iloc[i, j]
                if not np.isnan(val):
                    text = fmt.format(val)
                    color = 'white' if val > threshold else 'black'
                    ax.text(j, i, text, ha='center', va='center', color=color, fontsize=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        
        if save_path:
            # Save PNG version
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
            # Save SVG version
            svg_path = save_path.replace('.png', '.svg')
            plt.savefig(svg_path, format='svg', bbox_inches='tight')
            print(f"Saved: {svg_path}")
        else:
            plt.show()
        plt.close()

    # Plot recall heatmap (excluding Space)
    if accuracy_filename is None:
        accuracy_filename = "all_models_accuracy_heatmap.png"
    recall_path = os.path.join(output_dir, accuracy_filename)
    accuracy_title = "Per-Field Accuracy by Model"
    if excluded_names:
        accuracy_title += f" (Excluding {len(excluded_names)} Sample(s))"
    plot_heatmap(
        df_rec_sorted[["Mean Accuracy"] + heatmap_fields],
        title=accuracy_title,
        fmt="{:.2f}",
        cmap='Blues',
        cbar_label="Recall",
        save_path=recall_path
    )
    
    # Plot MAPE heatmap (excluding Space)
    if mape_filename is None:
        mape_filename = "all_models_mape_heatmap.png"
    mape_path = os.path.join(output_dir, mape_filename)
    mape_title = "Per-Field MAPE (%) by model"
    if excluded_names:
        mape_title += f" (Excluding {len(excluded_names)} Sample(s))"
    plot_heatmap(
        df_map_sorted_pct[["Mean MAPE"] + heatmap_fields],
        title=mape_title,
        fmt="{:.1f}%",
        cmap='Blues',
        cbar_label="MAPE (%)",
        save_path=mape_path
    )
    
    print(f"\nCombined visualizations saved to {output_dir}/")
    print(f"  - {accuracy_filename}")
    print(f"  - {mape_filename}")

