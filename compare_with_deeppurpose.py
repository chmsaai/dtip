"""
Compare Graph Transformer DTI model with DeepPurpose on the same BindingDB 20k test set.

Usage:
  python compare_with_deeppurpose.py \
    --graph_ckpt runs/graph_dti_seed42/best_model.pt \
    --data_csv bindingdb_processed/bindingdb_20k_with_images.csv

Outputs a comparison table (CSV + console) with parameter counts and metrics.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "DeepPurpose"))

from train_graph_dti import (
    FusionRegressor,
    BindingDBGraphProteinDataset,
    make_splits_stratified,
    count_parameters,
    evaluate,
    seed_everything,
)
from torch.utils.data import DataLoader


def count_deeppurpose_params():
    """Load a DeepPurpose model and count its parameters."""
    try:
        from DeepPurpose import DTI as models
        from DeepPurpose import utils

        target_encoding = "CNN"
        drug_encoding = "MPNN"

        model = models.model_pretrained(model="MPNN_CNN_DAVIS")
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_name": f"DeepPurpose ({drug_encoding}+{target_encoding}, DAVIS)",
            "total_params": total,
            "trainable_params": trainable,
        }
    except Exception as e:
        print(f"[Warn] Could not load DeepPurpose model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser("Compare Graph-DTI with DeepPurpose")
    parser.add_argument("--graph_ckpt", type=str, required=True)
    parser.add_argument("--data_csv", type=str, default="bindingdb_processed/bindingdb_20k_with_images.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    seed_everything(args.seed)

    # --- Load Graph DTI model ---
    ckpt_path = Path(args.graph_ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}")
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("args", {})

    model = FusionRegressor(
        feat_dim=cfg.get("feat_dim", 256),
        dropout=cfg.get("dropout", 0.2),
        max_atoms=cfg.get("max_atoms", 128),
        mol_embed_dim=cfg.get("mol_embed_dim", 128),
        mol_nhead=cfg.get("mol_nhead", 8),
        mol_layers=cfg.get("mol_layers", 4),
        mol_ff_dim=cfg.get("mol_ff_dim", 512),
        use_protein_transformer=not cfg.get("disable_protein_transformer", False),
        protein_max_len=cfg.get("protein_max_len", 512),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    graph_params = count_parameters(model)
    print(f"[Graph-DTI] params: {graph_params}")

    # --- Evaluate on test set ---
    df = pd.read_csv(args.data_csv)
    df = df.dropna(subset=["smiles", "target_sequence", "affinity_value_nM"]).copy()
    df = df[df["affinity_value_nM"] > 0].copy()
    split_df = make_splits_stratified(df, seed=args.seed)
    test_ds = BindingDBGraphProteinDataset(
        split_df[split_df["split"] == "test"],
        max_atoms=cfg.get("max_atoms", 128),
        protein_max_len=cfg.get("protein_max_len", 512),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    test_loss, test_metrics = evaluate(model, test_loader, device)
    print(f"[Graph-DTI] test metrics: {test_metrics}")

    # --- DeepPurpose params ---
    dp_info = count_deeppurpose_params()

    # --- Build comparison table ---
    rows = [
        {
            "System": "Graph Transformer + Protein Transformer (Ours)",
            "Drug Encoder": "Graph Transformer (Graphormer-style)",
            "Target Encoder": "Protein Transformer",
            "Fusion": "Gated + Diff + Product",
            "Total Params": f"{graph_params['total']:,}",
            "MAE": f"{test_metrics['mae']:.4f}",
            "RMSE": f"{test_metrics['rmse']:.4f}",
            "Pearson": f"{test_metrics['pearson']:.4f}",
            "Dataset": "BindingDB 20k (Kd)",
        },
    ]
    if dp_info:
        rows.append({
            "System": dp_info["model_name"],
            "Drug Encoder": "MPNN",
            "Target Encoder": "CNN",
            "Fusion": "Concatenation + MLP",
            "Total Params": f"{dp_info['total_params']:,}",
            "MAE": "(DAVIS pretrained, different data)",
            "RMSE": "(DAVIS pretrained, different data)",
            "Pearson": "(DAVIS pretrained, different data)",
            "Dataset": "DAVIS (pretrained)",
        })

    comp_df = pd.DataFrame(rows)
    out_csv = ckpt_path.parent / "comparison_with_deeppurpose.csv"
    comp_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nComparison table saved to: {out_csv}")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    main()
