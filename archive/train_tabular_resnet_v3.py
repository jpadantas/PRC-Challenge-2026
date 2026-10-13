"""
Train Modernized Tabular ResNet 3.0 with Entity Embeddings and Aerodynamic Wind Features.
Optimized for NVIDIA GeForce RTX 4060 GPU with CUDA acceleration.
Saves model to models/tabular_resnet_v3.pt and predictions to scratch/resnet_v3_test_preds.npy.
Thermal-safe: batch_size=4096, mixed precision, and thermal checks.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import subprocess
from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SCRATCH_DIR = PROJECT_ROOT / "scratch"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

from src.features_fast import compute_target_encodings_clean, extract_features_seasonal

WINTER_MONTHS = ["2025-01", "2025-02", "2025-11", "2025-12"]
SUMMER_MONTHS = ["2025-05", "2025-06", "2025-07", "2025-08"]


def get_gpu_temp() -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        )
        parts = out.strip().split(",")
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


class FastTabularDataset(Dataset):
    def __init__(self, cat_tensor: torch.Tensor, num_tensor: torch.Tensor, y_tensor: torch.Tensor = None):
        self.cats = cat_tensor
        self.nums = num_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.nums)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.cats[idx], self.nums[idx], self.y[idx]
        return self.cats[idx], self.nums[idx]


class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.10):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return x + self.block(x)


class ModernTabularResNet(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], num_features: int, d_model: int = 256, n_blocks: int = 3, dropout_rate: float = 0.10):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = min(24, max(4, int(np.sqrt(card) * 2)))
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        self.bn_num = nn.BatchNorm1d(num_features)
        in_dim = total_emb_dim + num_features
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([ResNetBlock(d_model, dropout_rate) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        embs = [emb(cats[:, i]) for i, (_, emb) in enumerate(self.embeddings.items())]
        x_emb = torch.cat(embs, dim=1)
        x_num = self.bn_num(nums)
        x = torch.cat([x_emb, x_num], dim=1)
        z = self.input_proj(x)
        for block in self.blocks:
            z = block(z)
        return self.head(z).squeeze(-1)


def load_season_files(months: list[str]) -> pl.DataFrame:
    files = []
    for m in months:
        year, month = int(m[:4]), int(m[5:7])
        if month < 12:
            p = RAW_DIR / f"training_{m}-01_{year}-{month+1:02d}-01.parquet"
        else:
            p = RAW_DIR / f"training_{m}-01_{year+1}-01-01.parquet"
        files.append(p)
    return pl.concat([pl.read_parquet(f) for f in files])


def main():
    t_start = time.time()
    temp_init, pwr_init = get_gpu_temp()
    print("=" * 80)
    print(f"🧠 TRAINING NEXT-GEN TABULAR RESNET 3.0 (GPU Temp: {temp_init:.0f}°C, {pwr_init:.1f}W)")
    print("=" * 80)

    # 1. Load Dual-Season Data
    print("[1/5] Loading dual-season training data (Winter + Summer)...")
    df_win = load_season_files(WINTER_MONTHS)
    df_sum = load_season_files(SUMMER_MONTHS)
    df_train = pl.concat([df_win, df_sum])
    dep_train = df_train.filter(pl.col("PHASE_mvt") == "DEP")
    print(f"Loaded {len(df_train):,} movements ({len(dep_train):,} departures) in {time.time()-t_start:.1f}s")

    # 2. Extract Features
    t_feat = time.time()
    encs = compute_target_encodings_clean(dep_train)
    X_train_df, dep_train_out = extract_features_seasonal(df_train, encs)
    y_raw = dep_train_out["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
    has_aobt = X_train_df["has_aobt"].to_numpy() == 1
    da = X_train_df["da"].to_numpy()

    clean_mask = (y_raw >= 60) & (y_raw <= 7200) & has_aobt & (da >= 120) & (da <= 7200)
    X_tr = X_train_df[clean_mask].reset_index(drop=True)
    y_tr = y_raw[clean_mask].astype(np.float32)
    print(f"Extracted {X_tr.shape[1]} features in {time.time()-t_feat:.1f}s | Clean rows: {len(X_tr):,}")

    # Identify Categoricals vs Numerics
    cat_cols = list(X_tr.select_dtypes(include=["category", "object"]).columns)
    num_cols = [c for c in X_tr.columns if c not in cat_cols and c != "has_aobt"]

    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")
    print(f"Numerical features ({len(num_cols)}): {len(num_cols)} features (including headwind, crosswind)")

    # 3. Categorical Encodings & Numerical Standardization
    cat_maps = {}
    cat_cards = {}
    cat_arrays = []
    for c in cat_cols:
        vals = X_tr[c].astype(str).fillna("UNK").unique()
        mapping = {v: i + 1 for i, v in enumerate(vals) if v != "UNK"}
        mapping["UNK"] = 0
        cat_maps[c] = mapping
        cat_cards[c] = len(mapping)
        arr = X_tr[c].astype(str).fillna("UNK").map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
        cat_arrays.append(arr)
    cat_mat = np.column_stack(cat_arrays)

    num_means = {}
    num_stds = {}
    num_arrays = []
    for c in num_cols:
        vals = X_tr[c].to_numpy(dtype=np.float32)
        m = float(np.nanmean(vals))
        s = float(np.nanstd(vals))
        s = s if s > 1e-5 else 1.0
        num_means[c] = m
        num_stds[c] = s
        norm = np.nan_to_num((vals - m) / s, nan=0.0)
        num_arrays.append(norm)
    num_mat = np.column_stack(num_arrays).astype(np.float32)

    cat_tensor = torch.tensor(cat_mat, dtype=torch.long)
    num_tensor = torch.tensor(num_mat, dtype=torch.float32)
    y_tensor = torch.tensor(y_tr, dtype=torch.float32)

    # 4. DataLoader and Model Setup
    batch_size = 4096
    dataset = FastTabularDataset(cat_tensor, num_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModernTabularResNet(cat_cards, len(num_cols), d_model=256, n_blocks=3, dropout_rate=0.10).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    epochs = 8
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader))

    print(f"\n[4/5] Training ModernTabularResNet on {str(device).upper()} ({epochs} epochs, batch_size={batch_size})...")
    model.train()
    scaler = torch.amp.GradScaler("cuda")

    for ep in range(epochs):
        t_ep = time.time()
        running_loss = 0.0
        n_batches = 0
        for cats, nums, targets in loader:
            cats, nums, targets = cats.to(device, non_blocking=True), nums.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                preds = model(cats, nums)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            n_batches += 1

        rmse_ep = np.sqrt(running_loss / n_batches)
        temp, pwr = get_gpu_temp()
        t_ep_el = time.time() - t_ep
        print(f"   Epoch {ep+1:2d}/{epochs} ({t_ep_el:4.1f}s | {temp:.0f}°C, {pwr:.1f}W): Train RMSE = {rmse_ep:.2f}s")

    # 5. Predict on Test Set (ranking.parquet)
    print("\n[5/5] Generating predictions for test set (ranking.parquet)...")
    df_test = pl.read_parquet(RAW_DIR / "ranking.parquet")
    rank_dep = df_test.filter(pl.col("PHASE_mvt") == "DEP")
    X_test_all, dep_test_out = extract_features_seasonal(df_test, encs)

    # Encode test categoricals
    cat_te_arrays = []
    for c in cat_cols:
        mapping = cat_maps[c]
        arr = X_test_all[c].astype(str).fillna("UNK").map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
        cat_te_arrays.append(arr)
    cat_te_mat = np.column_stack(cat_te_arrays)

    num_te_arrays = []
    for c in num_cols:
        m = num_means[c]
        s = num_stds[c]
        vals = X_test_all[c].to_numpy(dtype=np.float32)
        norm = np.nan_to_num((vals - m) / s, nan=0.0)
        num_te_arrays.append(norm)
    num_te_mat = np.column_stack(num_te_arrays).astype(np.float32)

    cat_te_tensor = torch.tensor(cat_te_mat, dtype=torch.long)
    num_te_tensor = torch.tensor(num_te_mat, dtype=torch.float32)

    test_dataset = FastTabularDataset(cat_te_tensor, num_te_tensor)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for cats, nums in test_loader:
            cats, nums = cats.to(device), nums.to(device)
            with torch.amp.autocast("cuda"):
                p = model(cats, nums)
            all_preds.append(p.cpu().numpy())

    p_resnet_v3 = np.concatenate(all_preds).astype(np.float32)

    # Save model and predictions
    torch.save(model.state_dict(), MODELS_DIR / "tabular_resnet_v3.pt")
    out_npy = SCRATCH_DIR / "resnet_v3_test_preds.npy"
    np.save(out_npy, p_resnet_v3)

    temp_fin, _ = get_gpu_temp()
    print(f"\n💾 Saved ModernTabularResNet 3.0 model and test predictions to {out_npy}!")
    print(f"   Shape: {p_resnet_v3.shape}, Mean: {p_resnet_v3.mean():.2f}s, Std: {p_resnet_v3.std():.2f}s")
    print(f"   Total Runtime: {time.time()-t_start:.1f}s | Final GPU Temp: {temp_fin:.0f}°C")
    print("=" * 80)


if __name__ == "__main__":
    main()
