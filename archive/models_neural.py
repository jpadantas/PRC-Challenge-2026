"""
PyTorch Tabular Deep Neural Model with Entity Embeddings for Taxi-Out Prediction.
Optimized for NVIDIA GeForce RTX 4060 GPU with CUDA acceleration.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import time
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = [
    "adep", "runway", "stand", "stand_pier", "operator", 
    "aircraft_type", "airline_alliance", "market_segment", "recat_eu"
]

EMBEDDING_DIMS = {
    "adep": 12,
    "runway": 24,
    "stand": 32,
    "stand_pier": 8,
    "operator": 32,
    "aircraft_type": 20,
    "airline_alliance": 6,
    "market_segment": 8,
    "recat_eu": 6,
}


class TabularDataset(Dataset):
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


class TabularNeuralNet(nn.Module):
    def __init__(self, cat_cardinalities: dict[str, int], num_features: int, hidden_dims: list[int] = [512, 256, 128], dropout_rate: float = 0.15):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = EMBEDDING_DIMS.get(col, min(32, max(4, card // 2)))
            # +1 for unknown token (idx 0)
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        in_dim = total_emb_dim + num_features
        self.bn_num = nn.BatchNorm1d(num_features)

        # Main MLP with residual block
        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            curr_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(curr_dim, 1)

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        # Embeddings
        emb_outs = []
        for i, (col, emb_layer) in enumerate(self.embeddings.items()):
            emb_outs.append(emb_layer(cats[:, i]))
        x_emb = torch.cat(emb_outs, dim=1)
        x_num = self.bn_num(nums)
        x = torch.cat([x_emb, x_num], dim=1)
        h = self.mlp(x)
        out = self.head(h).squeeze(-1)
        return out


class TabularResBlock(nn.Module):
    """Residual Block for tabular deep learning with pre-activation LayerNorm & Dropout."""
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.drop(self.act(self.fc1(self.bn(x)))))


class TabularResNet(nn.Module):
    """
    Modern 4-Block Tabular ResNet with Entity Embeddings and residual skip connections.
    Directly aligns gradient descent for minimum squared error on continuous tabular data.
    """
    def __init__(
        self, 
        cat_cardinalities: dict[str, int], 
        num_features: int, 
        d_model: int = 512, 
        n_blocks: int = 4, 
        dropout: float = 0.15
    ):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for col, card in cat_cardinalities.items():
            emb_dim = EMBEDDING_DIMS.get(col, min(32, max(4, card // 2)))
            self.embeddings[col] = nn.Embedding(card + 1, emb_dim)
            total_emb_dim += emb_dim

        in_dim = total_emb_dim + num_features
        self.bn_num = nn.BatchNorm1d(num_features)
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.blocks = nn.ModuleList([TabularResBlock(d_model, dropout=dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, cats: torch.Tensor, nums: torch.Tensor) -> torch.Tensor:
        emb_outs = [self.embeddings[col](cats[:, i]) for i, col in enumerate(self.embeddings.keys())]
        x_emb = torch.cat(emb_outs, dim=1)
        x_num = self.bn_num(nums)
        x = torch.cat([x_emb, x_num], dim=1)
        z = self.input_proj(x)
        for b in self.blocks:
            z = b(z)
        out = self.head(z).squeeze(-1)
        return out


def prepare_tensors(
    df: pd.DataFrame, 
    cat_cols: list[str], 
    num_cols: list[str], 
    cat_maps: dict = None, 
    num_means: dict = None, 
    num_stds: dict = None, 
    y: np.ndarray = None
):
    """
    Encode categoricals to integer indices and standardize continuous features.
    """
    is_fit = cat_maps is None
    if is_fit:
        cat_maps = {}
        for c in cat_cols:
            vals = df[c].astype(str).fillna("UNK").unique()
            # 0 is reserved for unknown / padding
            mapping = {v: i + 1 for i, v in enumerate(vals) if v != "UNK"}
            mapping["UNK"] = 0
            cat_maps[c] = mapping

    cat_arrays = []
    for c in cat_cols:
        mapping = cat_maps[c]
        arr = df[c].astype(str).fillna("UNK").map(lambda x: mapping.get(x, 0)).to_numpy(dtype=np.int64)
        cat_arrays.append(arr)
    cat_matrix = np.column_stack(cat_arrays)

    # Numerical normalization
    if is_fit:
        num_means = {}
        num_stds = {}
        for c in num_cols:
            vals = df[c].to_numpy(dtype=np.float32)
            m = np.nanmean(vals)
            s = np.nanstd(vals)
            num_means[c] = float(m if not np.isnan(m) else 0.0)
            num_stds[c] = float(s if (not np.isnan(s) and s > 1e-5) else 1.0)

    num_arrays = []
    for c in num_cols:
        vals = df[c].to_numpy(dtype=np.float32)
        m = num_means[c]
        s = num_stds[c]
        clean_vals = np.nan_to_num(vals, nan=m)
        norm_vals = (clean_vals - m) / s
        num_arrays.append(norm_vals)
    num_matrix = np.column_stack(num_arrays).astype(np.float32)

    cat_tensor = torch.tensor(cat_matrix, dtype=torch.long)
    num_tensor = torch.tensor(num_matrix, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32) if y is not None else None

    return cat_tensor, num_tensor, y_tensor, cat_maps, num_means, num_stds


def train_neural_foundation(
    X_df: pd.DataFrame, 
    y: np.ndarray, 
    epochs: int = 12, 
    batch_size: int = 4096, 
    lr: float = 3e-3, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    print("=" * 75)
    print(f"🧠 TRAINING PYTORCH TABULAR NEURAL NETWORK ON {device.upper()}")
    print("=" * 75)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_df.columns]
    num_cols = [c for c in X_df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_df[c])]

    print(f"Features: {len(cat_cols)} Categorical Embeddings, {len(num_cols)} Continuous Dense.")

    valid_mask = (y >= 60) & (y <= 150000)
    X_train_df = X_df.iloc[valid_mask].reset_index(drop=True)
    y_train = y[valid_mask]

    cat_t, num_t, y_t, cat_maps, num_means, num_stds = prepare_tensors(
        X_train_df, cat_cols, num_cols, y=y_train
    )

    cardinalities = {c: max(cat_maps[c].values()) for c in cat_cols}
    dataset = TabularDataset(cat_t, num_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"), num_workers=0)

    model = TabularNeuralNet(cardinalities, len(num_cols)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)
    criterion = nn.SmoothL1Loss(beta=60.0)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for b_cats, b_nums, b_y in loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_cats, b_nums)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.2f} | Elapsed: {time.time() - t0:.1f}s")

    # Evaluate train RMSE
    model.eval()
    all_preds = []
    eval_loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=batch_size * 2, shuffle=False)
    with torch.no_grad():
        for b_cats, b_nums in eval_loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            p = model(b_cats, b_nums)
            all_preds.append(p.cpu().numpy())
    all_preds = np.clip(np.concatenate(all_preds), 60, 88500)
    final_rmse = np.sqrt(np.mean((y_train - all_preds) ** 2))
    print(f"\n✅ PyTorch Tabular Neural Net fitted in {time.time() - t0:.1f}s | Train RMSE: {final_rmse:.2f}s")

    # Save artifacts
    save_dict = {
        "state_dict": model.state_dict(),
        "cardinalities": cardinalities,
        "num_features": len(num_cols),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": cat_maps,
        "num_means": num_means,
        "num_stds": num_stds,
    }
    torch.save(save_dict, MODELS_DIR / "tabular_neural_foundation.pt")
    print(f"💾 Saved neural model to {MODELS_DIR / 'tabular_neural_foundation.pt'}")
    return model, save_dict


def predict_neural_foundation(model_dict: dict, X_test_df: pd.DataFrame, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> np.ndarray:
    cardinalities = model_dict["cardinalities"]
    num_features = model_dict["num_features"]
    cat_cols = model_dict["cat_cols"]
    num_cols = model_dict["num_cols"]
    cat_maps = model_dict["cat_maps"]
    num_means = model_dict["num_means"]
    num_stds = model_dict["num_stds"]

    cat_t, num_t, _, _, _, _ = prepare_tensors(
        X_test_df, cat_cols, num_cols, cat_maps=cat_maps, num_means=num_means, num_stds=num_stds
    )

    model = TabularNeuralNet(cardinalities, num_features).to(device)
    model.load_state_dict(model_dict["state_dict"])
    model.eval()

    loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=8192, shuffle=False)
    preds = []
    with torch.no_grad():
        for b_cats, b_nums in loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            p = model(b_cats, b_nums)
            preds.append(p.cpu().numpy())

    return np.clip(np.concatenate(preds), 60, 88500)


def train_tabular_resnet(
    X_df: pd.DataFrame, 
    y: np.ndarray, 
    epochs: int = 12, 
    batch_size: int = 4096, 
    lr: float = 2e-3, 
    d_model: int = 512,
    n_blocks: int = 4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    min_val: float = 60.0,
    max_val: float = 150000.0,
):
    print("=" * 80)
    print(f"🚀 TRAINING PYTORCH TABULAR RESNET 2.0 (MSE LOSS, RESIDUAL SKIPS) ON {device.upper()}")
    print("=" * 80)

    cat_cols = [c for c in CATEGORICAL_COLS if c in X_df.columns]
    num_cols = [c for c in X_df.columns if c not in cat_cols and pd.api.types.is_numeric_dtype(X_df[c])]

    print(f"Architecture: TabularResNet (d_model={d_model}, n_blocks={n_blocks})")
    print(f"Features: {len(cat_cols)} Embeddings, {len(num_cols)} Dense Inputs.")

    valid_mask = (y >= min_val) & (y <= max_val)
    X_train_df = X_df.iloc[valid_mask].reset_index(drop=True)
    y_raw = y[valid_mask]

    # Target Standardization (zero bias, optimal gradient scale)
    y_mean = float(np.mean(y_raw))
    y_std = float(np.std(y_raw))
    y_norm = (y_raw - y_mean) / y_std
    print(f"Target statistics: Mean = {y_mean:.2f}s, Std = {y_std:.2f}s (Normalized for MSE)")

    cat_t, num_t, y_t, cat_maps, num_means, num_stds = prepare_tensors(
        X_train_df, cat_cols, num_cols, y=y_norm
    )

    cardinalities = {c: max(cat_maps[c].values()) for c in cat_cols}
    dataset = TabularDataset(cat_t, num_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"), num_workers=0)

    model = TabularResNet(cardinalities, len(num_cols), d_model=d_model, n_blocks=n_blocks).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)
    criterion = nn.MSELoss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for b_cats, b_nums, b_y in loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            preds = model(b_cats, b_nums)
            loss = criterion(preds, b_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_mse = total_loss / n_batches
        epoch_rmse_sec = np.sqrt(avg_mse) * y_std
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] Norm MSE: {avg_mse:.4f} (approx RMSE: {epoch_rmse_sec:.2f}s) | Elapsed: {time.time() - t0:.1f}s")

    # Evaluate train RMSE on full training set
    model.eval()
    all_preds = []
    eval_loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=batch_size * 2, shuffle=False)
    with torch.no_grad():
        for b_cats, b_nums in eval_loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            p = model(b_cats, b_nums)
            all_preds.append(p.cpu().numpy())
    all_preds_norm = np.concatenate(all_preds)
    all_preds_sec = np.clip(all_preds_norm * y_std + y_mean, 60, 88500)
    final_rmse = np.sqrt(np.mean((y_raw - all_preds_sec) ** 2))
    print(f"\n✅ Tabular ResNet 2.0 fitted in {time.time() - t0:.1f}s | Train RMSE: {final_rmse:.2f}s")
    print(f"  Predicted Mean: {all_preds_sec.mean():.2f}s (Ground Truth: {y_mean:.2f}s, Bias: {all_preds_sec.mean() - y_mean:+.2f}s)")

    # Save artifacts
    save_dict = {
        "state_dict": model.state_dict(),
        "cardinalities": cardinalities,
        "num_features": len(num_cols),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "cat_maps": cat_maps,
        "num_means": num_means,
        "num_stds": num_stds,
        "y_mean": y_mean,
        "y_std": y_std,
        "d_model": d_model,
        "n_blocks": n_blocks,
    }
    torch.save(save_dict, MODELS_DIR / "tabular_resnet_foundation.pt")
    print(f"💾 Saved Tabular ResNet model to {MODELS_DIR / 'tabular_resnet_foundation.pt'}")
    return model, save_dict


def predict_tabular_resnet(
    model_dict: dict, 
    X_test_df: pd.DataFrame, 
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    min_clip: float = 60.0,
    max_clip: float = 88500.0,
) -> np.ndarray:
    cardinalities = model_dict["cardinalities"]
    num_features = model_dict["num_features"]
    cat_cols = model_dict["cat_cols"]
    num_cols = model_dict["num_cols"]
    cat_maps = model_dict["cat_maps"]
    num_means = model_dict["num_means"]
    num_stds = model_dict["num_stds"]
    y_mean = model_dict["y_mean"]
    y_std = model_dict["y_std"]
    d_model = model_dict.get("d_model", 512)
    n_blocks = model_dict.get("n_blocks", 4)

    cat_t, num_t, _, _, _, _ = prepare_tensors(
        X_test_df, cat_cols, num_cols, cat_maps=cat_maps, num_means=num_means, num_stds=num_stds
    )

    model = TabularResNet(cardinalities, num_features, d_model=d_model, n_blocks=n_blocks).to(device)
    model.load_state_dict(model_dict["state_dict"])
    model.eval()

    loader = DataLoader(TabularDataset(cat_t, num_t), batch_size=8192, shuffle=False)
    preds = []
    with torch.no_grad():
        for b_cats, b_nums in loader:
            b_cats = b_cats.to(device)
            b_nums = b_nums.to(device)
            p = model(b_cats, b_nums)
            preds.append(p.cpu().numpy())

    preds_norm = np.concatenate(preds)
    preds_sec = preds_norm * y_std + y_mean
    return np.clip(preds_sec, min_clip, max_clip)

