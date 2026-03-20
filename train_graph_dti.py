"""
Graph Transformer based Drug-Target Interaction prediction.

Molecular input:  SMILES -> RDKit molecular graph (atom features + adjacency matrix)
Protein input:    amino acid sequence -> token IDs
Output:           predicted log10(Kd)

Key design:
  - Atom-level features extracted from molecular graph via RDKit
  - Graph Transformer encoder with adjacency-biased multi-head attention
  - Gated fusion with protein sequence encoder
  - Joint regression + ranking loss
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("rdkit is required: pip install rdkit")

# ============================================================
# Constants
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}

ATOM_SYMBOLS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Se", "Si"]

HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

BOND_TYPES = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

# atom_type(13) + degree(6) + charge(5) + nHs(5) + aromatic(1) + ring(1) + hybrid(7) = 38
ATOM_FEAT_DIM = len(ATOM_SYMBOLS) + 1 + 6 + 5 + 5 + 1 + 1 + len(HYBRIDIZATIONS) + 1

# Edge types for attention bias:
#   0 = no connection, 1 = single, 2 = double, 3 = triple,
#   4 = aromatic, 5 = self-loop, 6 = CLS <-> atom
NUM_EDGE_TYPES = 7


# ============================================================
# Utility functions
# ============================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


# ============================================================
# Molecular graph construction
# ============================================================

def get_atom_features(atom) -> List[float]:
    """Extract a fixed-length feature vector from an RDKit atom."""
    features: List[float] = []

    sym = atom.GetSymbol()
    atom_type = [0.0] * (len(ATOM_SYMBOLS) + 1)
    if sym in ATOM_SYMBOLS:
        atom_type[ATOM_SYMBOLS.index(sym)] = 1.0
    else:
        atom_type[-1] = 1.0
    features.extend(atom_type)

    degree = [0.0] * 6
    degree[min(atom.GetDegree(), 5)] = 1.0
    features.extend(degree)

    charge = [0.0] * 5
    charge[max(-2, min(2, atom.GetFormalCharge())) + 2] = 1.0
    features.extend(charge)

    nhs = [0.0] * 5
    nhs[min(atom.GetTotalNumHs(), 4)] = 1.0
    features.extend(nhs)

    features.append(1.0 if atom.GetIsAromatic() else 0.0)
    features.append(1.0 if atom.IsInRing() else 0.0)

    hyb = [0.0] * (len(HYBRIDIZATIONS) + 1)
    h = atom.GetHybridization()
    if h in HYBRIDIZATIONS:
        hyb[HYBRIDIZATIONS.index(h)] = 1.0
    else:
        hyb[-1] = 1.0
    features.extend(hyb)

    return features


def smiles_to_graph(smiles: str, max_atoms: int = 128):
    """Convert SMILES to padded graph tensors.

    Returns (atom_feat, edge_type, num_atoms) or None on failure.
      atom_feat:  np.float32 [max_atoms, ATOM_FEAT_DIM]
      edge_type:  np.int64   [max_atoms, max_atoms]
      num_atoms:  int
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n = mol.GetNumAtoms()
    if n == 0 or n > max_atoms:
        return None

    feat = np.zeros((max_atoms, ATOM_FEAT_DIM), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        feat[i] = get_atom_features(atom)

    etype = np.zeros((max_atoms, max_atoms), dtype=np.int64)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TYPES.get(bond.GetBondType(), 1)
        etype[i, j] = bt
        etype[j, i] = bt
    for i in range(n):
        etype[i, i] = 5  # self-loop

    return feat, etype, n


# ============================================================
# Data splitting (reused from original pipeline)
# ============================================================

def make_splits_stratified(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_bins: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
        raise ValueError("train/val/test ratios must sum to 1.0")

    work = df.copy()
    work["log_kd"] = np.log10(work["affinity_value_nM"].astype(float))
    unique_vals = work["log_kd"].nunique()
    bins = min(n_bins, max(2, unique_vals))
    work["aff_bin"] = pd.qcut(work["log_kd"], q=bins, labels=False, duplicates="drop")
    work["split"] = "train"

    rng = np.random.default_rng(seed)
    val_idx_all: List[int] = []
    test_idx_all: List[int] = []

    for _, grp in work.groupby("aff_bin"):
        idx = np.array(grp.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        n = len(idx)
        n_test = max(1, int(round(n * test_ratio))) if n >= 10 else max(0, int(n * test_ratio))
        n_val = max(1, int(round(n * val_ratio))) if n >= 10 else max(0, int(n * val_ratio))
        if n_test + n_val >= n:
            n_test = max(0, int(n * test_ratio))
            n_val = max(0, int(n * val_ratio))
        test_idx = idx[:n_test]
        val_idx = idx[n_test : n_test + n_val]
        val_idx_all.extend(val_idx.tolist())
        test_idx_all.extend(test_idx.tolist())

    work.loc[val_idx_all, "split"] = "val"
    work.loc[test_idx_all, "split"] = "test"
    return work.drop(columns=["aff_bin"])


# ============================================================
# Dataset
# ============================================================

class BindingDBGraphProteinDataset(Dataset):
    """Converts SMILES on-the-fly to molecular graph + protein token IDs."""

    def __init__(
        self,
        df: pd.DataFrame,
        max_atoms: int = 128,
        protein_max_len: int = 512,
    ):
        self.max_atoms = max_atoms
        self.protein_max_len = protein_max_len

        atom_feats, edge_types, atom_masks = [], [], []
        protein_ids_list, y_list = [], []
        skipped = 0

        for i in range(len(df)):
            row = df.iloc[i]
            result = smiles_to_graph(str(row["smiles"]), max_atoms)
            if result is None:
                skipped += 1
                continue
            feat, etype, n_atoms = result

            mask = np.zeros(max_atoms, dtype=np.float32)
            mask[:n_atoms] = 1.0

            atom_feats.append(feat)
            edge_types.append(etype)
            atom_masks.append(mask)

            seq = str(row["target_sequence"])
            ids = [AA_TO_ID.get(ch, 0) for ch in seq[:protein_max_len]]
            if len(ids) < protein_max_len:
                ids.extend([0] * (protein_max_len - len(ids)))
            protein_ids_list.append(ids)

            y_list.append(math.log10(float(row["affinity_value_nM"])))

        self.atom_feats = torch.tensor(np.array(atom_feats), dtype=torch.float32)
        self.edge_types = torch.tensor(np.array(edge_types), dtype=torch.long)
        self.atom_masks = torch.tensor(np.array(atom_masks), dtype=torch.float32)
        self.protein_ids = torch.tensor(np.array(protein_ids_list), dtype=torch.long)
        self.y = torch.tensor(y_list, dtype=torch.float32)

        print(f"  [Dataset] valid={len(self.y)}, skipped={skipped}")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            self.atom_feats[idx],
            self.edge_types[idx],
            self.atom_masks[idx],
            self.protein_ids[idx],
            self.y[idx],
        )


# ============================================================
# Protein encoders (same as original)
# ============================================================

class ProteinEncoder(nn.Module):
    """BiGRU + attention pooling."""

    def __init__(
        self,
        vocab_size: int = len(AA_TO_ID) + 1,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, protein_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(protein_ids)
        h, _ = self.gru(x)
        score = self.attn(h).squeeze(-1)
        mask = protein_ids != 0
        neg_inf = torch.finfo(score.dtype).min
        score = score.masked_fill(~mask, neg_inf)
        w = torch.softmax(score.float(), dim=1).to(h.dtype).unsqueeze(-1)
        return self.proj((h * w).sum(dim=1))


class ProteinTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = len(AA_TO_ID) + 1,
        embed_dim: int = 128,
        out_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
        max_len: int = 1024,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.cls_pos = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def forward(self, protein_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = protein_ids.shape
        if seqlen > self.max_len:
            protein_ids = protein_ids[:, : self.max_len]
            seqlen = self.max_len
        x = self.token_embed(protein_ids) + self.pos_embed[:, :seqlen, :]
        cls = self.cls_token.expand(bsz, -1, -1) + self.cls_pos
        x = torch.cat([cls, x], dim=1)
        pad_mask = torch.zeros((bsz, seqlen + 1), dtype=torch.bool, device=protein_ids.device)
        pad_mask[:, 1:] = protein_ids == 0
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.proj(x[:, 0, :])


# ============================================================
# Graph Transformer encoder
# ============================================================

class GraphTransformerLayer(nn.Module):
    """Pre-norm Transformer layer that accepts a 3-D attention bias."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_bias, key_padding_mask=key_padding_mask)
        x = x + self.dropout(x2)
        x = x + self.ff(self.norm2(x))
        return x


class MoleculeGraphTransformerEncoder(nn.Module):
    """Graphormer-style encoder: atom features + edge-type attention bias + degree encoding."""

    def __init__(
        self,
        atom_feat_dim: int = ATOM_FEAT_DIM,
        embed_dim: int = 128,
        out_dim: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.2,
        max_atoms: int = 128,
    ):
        super().__init__()
        self.nhead = nhead
        self.max_atoms = max_atoms

        self.atom_proj = nn.Linear(atom_feat_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.degree_embed = nn.Embedding(16, embed_dim)
        self.edge_bias = nn.Embedding(NUM_EDGE_TYPES, nhead)

        self.layers = nn.ModuleList(
            [GraphTransformerLayer(embed_dim, nhead, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    def _build_attn_bias(self, edge_type: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Build [B*nhead, S, S] additive attention bias from edge type matrix.

        edge_type: [B, N, N] (0..5)  —  atom-level edge types
        atom_mask: [B, N]  float  (1=real, 0=pad)
        Returns:   [B*nhead, N+1, N+1]  float
        """
        B, N, _ = edge_type.shape
        device = edge_type.device
        S = N + 1

        full_etype = torch.zeros(B, S, S, dtype=torch.long, device=device)
        full_etype[:, 1:, 1:] = edge_type

        full_etype[:, 0, 0] = 5  # CLS self-loop
        bool_mask = atom_mask > 0.5
        full_etype[:, 0, 1:][bool_mask] = 6  # CLS -> atom
        full_etype[:, 1:, 0][bool_mask] = 6  # atom -> CLS

        bias = self.edge_bias(full_etype)  # [B, S, S, nhead]
        bias = bias.permute(0, 3, 1, 2).reshape(B * self.nhead, S, S)
        return bias

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_type: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        atom_features: [B, N, F]
        edge_type:     [B, N, N]  int (0..5)
        atom_mask:     [B, N]     float (1=real, 0=pad)
        """
        B, N, _ = atom_features.shape

        x = self.atom_proj(atom_features)
        degree = edge_type.clamp(min=0).bool().float().sum(dim=-1).long().clamp(max=15)
        x = x + self.degree_embed(degree)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, S, E]

        attn_bias = self._build_attn_bias(edge_type, atom_mask)

        cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_mask = torch.cat([cls_pad, atom_mask < 0.5], dim=1)  # True=ignore

        for layer in self.layers:
            x = layer(x, attn_bias=attn_bias, key_padding_mask=pad_mask)

        x = self.final_norm(x[:, 0, :])
        return self.proj(x)


# ============================================================
# Fusion regressor
# ============================================================

class FusionRegressor(nn.Module):
    def __init__(
        self,
        feat_dim: int = 256,
        dropout: float = 0.2,
        max_atoms: int = 128,
        mol_embed_dim: int = 128,
        mol_nhead: int = 8,
        mol_layers: int = 4,
        mol_ff_dim: int = 512,
        use_protein_transformer: bool = True,
        protein_max_len: int = 512,
    ):
        super().__init__()
        self.mol_encoder = MoleculeGraphTransformerEncoder(
            atom_feat_dim=ATOM_FEAT_DIM,
            embed_dim=mol_embed_dim,
            out_dim=feat_dim,
            nhead=mol_nhead,
            num_layers=mol_layers,
            ff_dim=mol_ff_dim,
            dropout=dropout,
            max_atoms=max_atoms,
        )

        if use_protein_transformer:
            self.protein_encoder = ProteinTransformerEncoder(
                out_dim=feat_dim, dropout=dropout, max_len=protein_max_len,
            )
        else:
            self.protein_encoder = ProteinEncoder(out_dim=feat_dim, dropout=dropout)

        self.gate = nn.Sequential(nn.Linear(feat_dim * 2, feat_dim), nn.Sigmoid())

        self.head = nn.Sequential(
            nn.Linear(feat_dim * 4, feat_dim * 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim, 1),
        )

    def forward(
        self,
        atom_features: torch.Tensor,
        edge_type: torch.Tensor,
        atom_mask: torch.Tensor,
        protein_ids: torch.Tensor,
    ) -> torch.Tensor:
        mol = self.mol_encoder(atom_features, edge_type, atom_mask)
        prot = self.protein_encoder(protein_ids)

        gate = self.gate(torch.cat([mol, prot], dim=-1))
        mol_g = mol * gate
        prot_g = prot * (1.0 - gate)

        fused = torch.cat([mol_g, prot_g, torch.abs(mol_g - prot_g), mol_g * prot_g], dim=-1)
        return self.head(fused).squeeze(-1)


# ============================================================
# Loss
# ============================================================

def ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    if pred.shape[0] < 2:
        return pred.new_tensor(0.0)
    perm = torch.randperm(pred.shape[0], device=pred.device)
    sign = torch.sign(target - target[perm])
    valid = sign != 0
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    diff = sign[valid] * (pred[valid] - pred[perm][valid])
    return F.relu(margin - diff).mean()


def compute_metrics(pred: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    corr = float(np.corrcoef(pred, y)[0, 1]) if len(pred) > 1 else 0.0
    return {"mae": mae, "rmse": rmse, "pearson": corr}


# ============================================================
# Evaluate
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool = False) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_fn = nn.SmoothL1Loss()
    losses, pred_all, y_all = [], [], []

    for atom_feat, etype, amask, prot_ids, y in loader:
        atom_feat = atom_feat.to(device)
        etype = etype.to(device)
        amask = amask.to(device)
        prot_ids = prot_ids.to(device)
        y = y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
            pred = model(atom_feat, etype, amask, prot_ids)
            loss = loss_fn(pred, y)

        losses.append(loss.item())
        pred_all.append(pred.cpu().numpy())
        y_all.append(y.cpu().numpy())

    pred_np = np.concatenate(pred_all) if pred_all else np.array([])
    y_np = np.concatenate(y_all) if y_all else np.array([])
    metrics = compute_metrics(pred_np, y_np) if len(pred_np) > 0 else {"mae": 0.0, "rmse": 0.0, "pearson": 0.0}
    return float(np.mean(losses) if losses else 0.0), metrics


# ============================================================
# Train
# ============================================================

def train(args):
    seed_everything(args.seed)
    data_csv = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    use_amp = (not args.no_amp) and torch.cuda.is_available()

    loader_workers = args.num_workers
    if args.windows_safe_loader and os.name == "nt" and loader_workers > 0:
        print(f"[Info] Windows safe mode: forcing num_workers=0")
        loader_workers = 0

    # --- data -------------------------------------------------
    df = pd.read_csv(data_csv)
    df = df.dropna(subset=["smiles", "target_sequence", "affinity_value_nM"]).copy()
    df = df[df["affinity_value_nM"] > 0].copy()

    split_csv = output_dir / "split.csv"
    if args.reuse_split and split_csv.exists():
        split_df = pd.read_csv(split_csv)
    else:
        split_df = make_splits_stratified(df, seed=args.seed)
        split_df.to_csv(split_csv, index=False, encoding="utf-8-sig")

    stats = {
        "total": int(len(split_df)),
        "train": int((split_df["split"] == "train").sum()),
        "val": int((split_df["split"] == "val").sum()),
        "test": int((split_df["split"] == "test").sum()),
    }
    (output_dir / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Split stats: {stats}")

    print("[Info] Building datasets (SMILES -> graph) ...")
    train_ds = BindingDBGraphProteinDataset(
        split_df[split_df["split"] == "train"], max_atoms=args.max_atoms, protein_max_len=args.protein_max_len,
    )
    val_ds = BindingDBGraphProteinDataset(
        split_df[split_df["split"] == "val"], max_atoms=args.max_atoms, protein_max_len=args.protein_max_len,
    )
    test_ds = BindingDBGraphProteinDataset(
        split_df[split_df["split"] == "test"], max_atoms=args.max_atoms, protein_max_len=args.protein_max_len,
    )

    loader_kw = {"num_workers": loader_workers, "pin_memory": torch.cuda.is_available()}
    if loader_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # --- device -----------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Runtime] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, device={device}")
    if torch.cuda.is_available():
        print(f"[Runtime] gpu={torch.cuda.get_device_name(0)}, count={torch.cuda.device_count()}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.require_cuda:
        raise RuntimeError("CUDA required but not available.")

    # --- model ------------------------------------------------
    model = FusionRegressor(
        feat_dim=args.feat_dim,
        dropout=args.dropout,
        max_atoms=args.max_atoms,
        mol_embed_dim=args.mol_embed_dim,
        mol_nhead=args.mol_nhead,
        mol_layers=args.mol_layers,
        mol_ff_dim=args.mol_ff_dim,
        use_protein_transformer=not args.disable_protein_transformer,
        protein_max_len=args.protein_max_len,
    ).to(device)

    param_info = count_parameters(model)
    print(f"[Model] total_params={param_info['total']:,}, trainable={param_info['trainable']:,}")
    (output_dir / "param_count.json").write_text(json.dumps(param_info, indent=2), encoding="utf-8")

    # --- optimizer / scheduler --------------------------------
    if device.type == "cuda":
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    reg_loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_rmse = float("inf")
    best_ckpt = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for atom_feat, etype, amask, prot_ids, y in train_loader:
            atom_feat = atom_feat.to(device, non_blocking=True)
            etype = etype.to(device, non_blocking=True)
            amask = amask.to(device, non_blocking=True)
            prot_ids = prot_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                pred = model(atom_feat, etype, amask, prot_ids)
                loss = reg_loss_fn(pred, y) + args.rank_weight * ranking_loss(pred, y, margin=args.rank_margin)

            optimizer.zero_grad()
            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        train_loss = float(np.mean(losses) if losses else 0.0)
        val_loss, val_metrics = evaluate(model, val_loader, device, use_amp=use_amp)

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"],
            "val_pearson": val_metrics["pearson"], "lr": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"val_r={val_metrics['pearson']:.4f}"
        )
        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_ckpt)
            print(f"  -> Saved best: {best_ckpt}")

    pd.DataFrame(history).to_csv(output_dir / "train_history.csv", index=False, encoding="utf-8-sig")

    # --- test -------------------------------------------------
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_loss, test_metrics = evaluate(model, test_loader, device, use_amp=use_amp)
    report = {"test_loss": test_loss, **test_metrics, **param_info}
    (output_dir / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test metrics: {report}")
    print(f"Artifacts saved to: {output_dir}")


# ============================================================
# CLI
# ============================================================

def build_argparser():
    p = argparse.ArgumentParser("Graph Transformer DTI on BindingDB 20k")
    p.add_argument("--data_csv", type=str, default="bindingdb_processed/bindingdb_20k_with_images.csv")
    p.add_argument("--output_dir", type=str, default="runs/graph_dti_seed42")
    p.add_argument("--reuse_split", action="store_true")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--strat_bins", type=int, default=10)

    p.add_argument("--max_atoms", type=int, default=128)
    p.add_argument("--protein_max_len", type=int, default=512)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--mol_embed_dim", type=int, default=128)
    p.add_argument("--mol_nhead", type=int, default=8)
    p.add_argument("--mol_layers", type=int, default=4)
    p.add_argument("--mol_ff_dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--disable_protein_transformer", action="store_true")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--rank_weight", type=float, default=0.1)
    p.add_argument("--rank_margin", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--prefetch_factor", type=int, default=4)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--windows_safe_loader", action="store_true")
    p.add_argument("--require_cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
