import sys
import os

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DeepPurpose"))

print("[1/5] Importing dependencies...")
import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from train_graph_dti import make_splits_stratified, seed_everything

print("[2/5] Importing DeepPurpose...")
try:
    from DeepPurpose import DTI as models
    from DeepPurpose import utils
    print("  DeepPurpose imported OK")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

def compute_metrics(pred, y):
    pred = np.array(pred)
    y = np.array(y)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson": corr}

def main():
    seed_everything(42)

    csv_path = Path(__file__).resolve().parent / "bindingdb_processed" / "bindingdb_20k_with_images.csv"
    print(f"[3/5] Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["smiles", "target_sequence", "affinity_value_nM"]).copy()
    df = df[df["affinity_value_nM"] > 0].copy()
    split_df = make_splits_stratified(df, seed=42)

    test_df = split_df[split_df["split"] == "test"].copy()
    print(f"  Test set size: {len(test_df)}")

    test_smiles = test_df["smiles"].tolist()
    test_targets = test_df["target_sequence"].tolist()
    test_y = [math.log10(float(v)) for v in test_df["affinity_value_nM"].tolist()]

    configs = [
        ("MPNN_CNN_DAVIS", "MPNN", "CNN"),
    ]

    results = []
    for model_name, drug_enc, target_enc in configs:
        print(f"\n[4/5] Loading pretrained model: {model_name} ({drug_enc}+{target_enc})...")
        try:
            model = models.model_pretrained(model=model_name)
            total_params = sum(p.numel() for p in model.model_drug.parameters()) + \
                           sum(p.numel() for p in model.model_protein.parameters()) + \
                           sum(p.numel() for p in model.model.parameters())
            print(f"  Params: ~{total_params:,}")
        except Exception as e:
            print(f"  Failed to load model: {e}")
            continue

        print(f"[5/5] Evaluating on {len(test_df)} test samples (this may take a while)...")
        preds = []
        skipped = 0
        valid_y = []

        for idx, (smi, tgt, y_true) in enumerate(zip(test_smiles, test_targets, test_y)):
            if (idx + 1) % 200 == 0:
                print(f"  Progress: {idx+1}/{len(test_smiles)} ...")
            try:
                pred_val = models.virtual_screening(
                    [smi], tgt, model, drug_enc, target_enc, verbose=False
                )
                if pred_val is not None and len(pred_val) > 0:
                    p = float(pred_val[0])
                    if not (math.isnan(p) or math.isinf(p)):
                        preds.append(p)
                        valid_y.append(y_true)
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception:
                skipped += 1

        print(f"\n  Evaluated: {len(preds)}, Skipped: {skipped}")
        if len(preds) > 10:
            metrics = compute_metrics(preds, valid_y)
            print(f"  MAE:     {metrics['mae']:.4f}")
            print(f"  RMSE:    {metrics['rmse']:.4f}")
            print(f"  Pearson: {metrics['pearson']:.4f}")
            results.append({
                "model": model_name,
                "drug_encoder": drug_enc,
                "target_encoder": target_enc,
                "total_params": total_params,
                "test_samples": len(preds),
                **metrics,
            })
        else:
            print("  Too few valid predictions, skipping metrics.")

    for seed in [42, 123, 3407]:
        metrics_path = Path(__file__).resolve().parent / f"runs/graph_dti_seed{seed}/test_metrics.json"
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            results.append({
                "model": f"GraphTransformer_seed{seed} (Ours)",
                "drug_encoder": "Graph Transformer",
                "target_encoder": "Protein Transformer",
                "total_params": 2515897,
                "test_samples": "~2000",
                "mae": m["mae"],
                "rmse": m["rmse"],
                "pearson": m["pearson"],
            })

    out = Path(__file__).resolve().parent / "runs" / "deeppurpose_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame(results)
    comp_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n{'='*60}")
    print(comp_df.to_string(index=False))
    print(f"\nSaved to: {out}")


if __name__ == "__main__":
    main()
