"""
Retrain DeepPurpose MPNN+CNN on the same BindingDB 20k dataset for fair comparison.
Uses the same train/val/test split (seed=42, stratified) as Graph Transformer.
"""
import sys, os, math, json
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import builtins
_orig_print = builtins.print
def _flush_print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)
builtins.print = _flush_print

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "DeepPurpose"))

import numpy as np
import pandas as pd
from pathlib import Path

from train_graph_dti import make_splits_stratified, seed_everything

print("[1/6] Importing DeepPurpose...")
from DeepPurpose import DTI as models
from DeepPurpose import utils
print("  OK")


def main():
    seed_everything(42)

    csv_path = Path(__file__).resolve().parent / "bindingdb_processed" / "bindingdb_20k_with_images.csv"
    print(f"[2/6] Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["smiles", "target_sequence", "affinity_value_nM"]).copy()
    df = df[df["affinity_value_nM"] > 0].copy()

    split_df = make_splits_stratified(df, seed=42)

    train_df = split_df[split_df["split"] == "train"].copy()
    val_df   = split_df[split_df["split"] == "val"].copy()
    test_df  = split_df[split_df["split"] == "test"].copy()
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_smiles  = train_df["smiles"].tolist()
    train_targets = train_df["target_sequence"].tolist()
    train_y       = [math.log10(float(v)) for v in train_df["affinity_value_nM"].tolist()]

    val_smiles  = val_df["smiles"].tolist()
    val_targets = val_df["target_sequence"].tolist()
    val_y       = [math.log10(float(v)) for v in val_df["affinity_value_nM"].tolist()]

    test_smiles  = test_df["smiles"].tolist()
    test_targets = test_df["target_sequence"].tolist()
    test_y       = [math.log10(float(v)) for v in test_df["affinity_value_nM"].tolist()]

    drug_encoding  = "MPNN"
    target_encoding = "CNN"

    print(f"[3/6] Encoding data with {drug_encoding}+{target_encoding} ...")
    train_data = utils.data_process(
        X_drug=train_smiles, X_target=train_targets, y=train_y,
        drug_encoding=drug_encoding, target_encoding=target_encoding,
        split_method="no_split"
    )
    val_data = utils.data_process(
        X_drug=val_smiles, X_target=val_targets, y=val_y,
        drug_encoding=drug_encoding, target_encoding=target_encoding,
        split_method="no_split"
    )
    test_data = utils.data_process(
        X_drug=test_smiles, X_target=test_targets, y=test_y,
        drug_encoding=drug_encoding, target_encoding=target_encoding,
        split_method="no_split"
    )
    print("  Encoding done.")

    print("[4/6] Initializing model ...")
    config = utils.generate_config(
        drug_encoding=drug_encoding,
        target_encoding=target_encoding,
        result_folder="./runs/deeppurpose_mpnn_cnn/",
        cls_hidden_dims=[1024, 1024, 512],
        train_epoch=30,
        LR=1e-4,
        batch_size=64,
        mpnn_hidden_size=50,
        mpnn_depth=3,
        cnn_target_filters=[32, 64, 96],
        cnn_target_kernels=[4, 8, 12],
        num_workers=0,
    )
    model = models.model_initialize(**config)

    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"  Total params: {total_params:,}")

    print("[5/6] Training (30 epochs) ...")
    model.train(train_data, val_data, test_data, verbose=True)

    print("[6/6] Final evaluation on test set ...")
    y_pred = model.predict(test_data)
    y_true = np.array(test_y)
    y_pred = np.array(y_pred)

    mae  = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    pearson = float(np.corrcoef(y_pred, y_true)[0, 1]) if len(y_pred) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"DeepPurpose MPNN+CNN retrained on BindingDB 20k")
    print(f"  Total params: {total_params:,}")
    print(f"  MAE:     {mae:.4f}")
    print(f"  RMSE:    {rmse:.4f}")
    print(f"  Pearson: {pearson:.4f}")
    print(f"{'='*60}")

    results = {
        "model": f"DeepPurpose_{drug_encoding}_{target_encoding}_retrained",
        "total_params": total_params,
        "mae": mae,
        "rmse": rmse,
        "pearson": pearson,
    }
    out_path = Path("./runs/deeppurpose_mpnn_cnn/retrain_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
