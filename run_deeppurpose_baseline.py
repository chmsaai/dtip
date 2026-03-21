"""
Run DeepPurpose baseline on the same BindingDB 20k test split for fair comparison.

Usage (on local machine where DeepPurpose is installed):
  cd dtip_source
  python run_deeppurpose_baseline.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DeepPurpose"))

from DeepPurpose import DTI as models
from DeepPurpose import utils

from train_graph_dti import make_splits_stratified, seed_everything, count_parameters


def compute_metrics(pred, y):
    pred = np.array(pred)
    y = np.array(y)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson": corr}


def main():
    seed_everything(42)
    csv_path = Path("bindingdb_processed/bindingdb_20k_with_images.csv")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["smiles", "target_sequence", "affinity_value_nM"]).copy()
    df = df[df["affinity_value_nM"] > 0].copy()
    split_df = make_splits_stratified(df, seed=42)

    test_df = split_df[split_df["split"] == "test"].copy()
    print(f"Test set size: {len(test_df)}")

    test_smiles = test_df["smiles"].tolist()
    test_targets = test_df["target_sequence"].tolist()
    test_y = [math.log10(float(v)) for v in test_df["affinity_value_nM"].tolist()]

    configs = [
        ("MPNN_CNN_DAVIS", "MPNN", "CNN"),
        ("Morgan_CNN_DAVIS", "Morgan", "CNN"),
    ]

    results = []
    for model_name, drug_enc, target_enc in configs:
        print(f"\n--- {model_name} ({drug_enc} + {target_enc}) ---")
        try:
            model = models.model_pretrained(model=model_name)
            param_info = count_parameters(model)
            print(f"  Params: {param_info}")

            preds = []
            skipped = 0
            valid_y = []
            for smi, tgt, y_true in zip(test_smiles, test_targets, test_y):
                try:
                    pred_val = models.virtual_screening([smi], tgt, model, drug_enc, target_enc, verbose=False)
                    if pred_val is not None and len(pred_val) > 0:
                        preds.append(float(pred_val[0]))
                        valid_y.append(y_true)
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

            print(f"  Evaluated: {len(preds)}, Skipped: {skipped}")
            if len(preds) > 10:
                metrics = compute_metrics(preds, valid_y)
                print(f"  Metrics: {metrics}")
                results.append({
                    "model": model_name,
                    "drug_encoder": drug_enc,
                    "target_encoder": target_enc,
                    "total_params": param_info["total"],
                    "test_samples": len(preds),
                    **metrics,
                })
            else:
                print("  Too few valid predictions, skipping metrics.")
        except Exception as e:
            print(f"  Error: {e}")

    # Add our Graph Transformer results
    for seed in [42, 123, 3407]:
        metrics_path = Path(f"runs/graph_dti_seed{seed}/test_metrics.json")
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append({
                "model": f"GraphTransformer_seed{seed} (Ours)",
                "drug_encoder": "Graph Transformer",
                "target_encoder": "Protein Transformer",
                "total_params": m.get("total", 2515897),
                "test_samples": "~2000",
                "mae": m["mae"],
                "rmse": m["rmse"],
                "pearson": m["pearson"],
            })

    out = Path("runs/deeppurpose_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(comp_df.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
