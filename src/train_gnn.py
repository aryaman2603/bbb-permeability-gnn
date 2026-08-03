"""Training loop for GAT and GIN models.

Implements:
  - Reproducible seeding (SEED=42)
  - Positive class weighting to handle BBB+/BBB- imbalance
  - Early stopping on validation ROC-AUC
  - Learning rate scheduling (ReduceLROnPlateau)
  - Gradient clipping for stability
  - Best-checkpoint saving
  - Final evaluation using the shared metrics module

Usage
-----
    uv run python -m src.train_gnn --model gat
    uv run python -m src.train_gnn --model gin

Results land in: results/person_b/{model}_val.json
                 results/person_b/{model}_test.json
                 results/person_b/{model}_best.pt
                 results/person_b/{model}_history.json
"""

import argparse
import json
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch_geometric.loader import DataLoader

from src.models.gat import GATClassifier
from src.models.gin import GINClassifier
from src.eval.metrics import evaluate_model, print_metrics, save_metrics

SEED = 42
RESULTS_DIR = Path("results/person_b")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model hyperparameters ──────────────────────────────────────────────────────
# These are sensible defaults for the BBBP dataset size (~1,400 training molecules).
# Feel free to tune hidden_channels, dropout, lr on the val set.
CONFIG: dict[str, dict] = {
    "gat": {
        "hidden_channels": 64,
        "num_heads": 4,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 150,
        "batch_size": 32,
        "patience": 25,   # stop if val AUC doesn't improve for 25 consecutive epochs
    },
    "gin": {
        "hidden_channels": 64,
        "num_layers": 3,
        "dropout": 0.3,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 150,
        "batch_size": 32,
        "patience": 25,
    },
}


def set_seed(seed: int) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Makes CUDA ops deterministic (small speed cost, big reproducibility gain)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_model(model_name: str, cfg: dict) -> nn.Module:
    """Instantiate the correct model with the given config."""
    if model_name == "gat":
        return GATClassifier(
            in_channels=22,
            hidden_channels=cfg["hidden_channels"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            edge_dim=7,
        )
    elif model_name == "gin":
        return GINClassifier(
            in_channels=22,
            hidden_channels=cfg["hidden_channels"],
            num_layers=cfg["num_layers"],
            dropout=cfg["dropout"],
            edge_dim=7,
        )
    else:
        raise ValueError(f"Unknown model: '{model_name}'. Choose 'gat' or 'gin'.")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """Run one full training epoch. Returns mean loss over all graphs."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze(-1))
        loss.backward()

        # Clip gradients to prevent exploding gradients in deep GNNs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference on a DataLoader. Returns (y_true, y_prob) numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        probs  = torch.sigmoid(logits).cpu().numpy()  # convert logits → [0,1]
        labels = batch.y.squeeze(-1).cpu().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_pos_weight(train_data: list) -> torch.Tensor:
    """Compute positive class weight to counter class imbalance.

    BCEWithLogitsLoss accepts a pos_weight that upweights the loss on
    positive (BBB+) examples. We set it to neg_count / pos_count,
    which balances gradient contributions from both classes.
    """
    labels = [int(d.y.item()) for d in train_data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    weight = n_neg / n_pos
    print(f"  Class balance — BBB+: {n_pos} ({n_pos/len(labels):.1%})  "
          f"BBB-: {n_neg} ({n_neg/len(labels):.1%})")
    print(f"  pos_weight set to: {weight:.3f}")
    return torch.tensor([weight], device=DEVICE)


def train(model_name: str) -> None:
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = CONFIG[model_name]

    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"  Device  : {DEVICE}")
    print(f"  Config  : {cfg}")
    print(f"{'='*60}")

    # ── 1. Load frozen scaffold-split dataset ──────────────────────────────
    print("\nLoading frozen dataset...")
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    val_data   = torch.load("data/processed/val.pt",   weights_only=False)
    test_data  = torch.load("data/processed/test.pt",  weights_only=False)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    # ── 2. Build DataLoaders ───────────────────────────────────────────────
    # Use a seeded Generator for shuffle so training order is reproducible
    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_data, batch_size=cfg["batch_size"],
        shuffle=True, generator=g
    )
    val_loader  = DataLoader(val_data,  batch_size=cfg["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=cfg["batch_size"], shuffle=False)

    # ── 3. Instantiate model, optimizer, scheduler, loss ──────────────────
    model = build_model(model_name, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    # Reduce LR by half when val AUC plateaus for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10
    )

    print("\nClass balance:")
    pos_weight = compute_pos_weight(train_data)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── 4. Training loop with early stopping ──────────────────────────────
    best_val_auc      = 0.0
    best_epoch        = 0
    patience_counter  = 0
    history           = []
    ckpt_path         = RESULTS_DIR / f"{model_name}_best.pt"

    print(f"\nTraining for up to {cfg['epochs']} epochs "
          f"(patience={cfg['patience']})...\n")

    for epoch in range(1, cfg["epochs"] + 1):
        # Forward + backward pass
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)

        # Validation evaluation
        val_y, val_prob = get_predictions(model, val_loader)
        val_metrics = evaluate_model(val_y, val_prob)
        val_auc = val_metrics["roc_auc"]

        # Update LR scheduler
        scheduler.step(val_auc)

        # Track history
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_auc": round(val_auc, 6),
            "val_f1": round(val_metrics["f1"], 6),
            "lr": optimizer.param_groups[0]["lr"],
        })

        # Save best model checkpoint
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch   = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        # Periodic logging
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{cfg['epochs']} | "
                f"loss={train_loss:.4f} | "
                f"val_auc={val_auc:.4f} | "
                f"best={best_val_auc:.4f} (ep {best_epoch}) | "
                f"lr={lr_now:.2e} | "
                f"patience={patience_counter}/{cfg['patience']}"
            )

        # Early stopping check
        if patience_counter >= cfg["patience"]:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print(f"No val AUC improvement for {cfg['patience']} consecutive epochs.")
            break

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f} at epoch {best_epoch}.")

    # ── 5. Final evaluation using best checkpoint ──────────────────────────
    print(f"\nLoading best checkpoint from epoch {best_epoch}...")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_prob = get_predictions(model, loader)
        metrics = evaluate_model(y_true, y_prob)
        print_metrics(metrics, model_name=model_name.upper(), split=split_name)
        save_metrics(metrics, model_name=model_name, split=split_name, out_dir=RESULTS_DIR)

    # ── 6. Persist training history ────────────────────────────────────────
    hist_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump({"model": model_name, "config": cfg, "history": history}, f, indent=2)
    print(f"\nTraining history saved → {hist_path}")
    print(f"Best checkpoint saved  → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GAT or GIN on the frozen BBBP scaffold-split dataset."
    )
    parser.add_argument(
        "--model",
        choices=["gat", "gin"],
        required=True,
        help="Which GNN architecture to train: 'gat' or 'gin'",
    )
    args = parser.parse_args()
    train(args.model)
