import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.models.gat import GATClassifier
from src.models.gin import GINClassifier
from src.eval.metrics import evaluate_model, print_metrics, save_metrics

SEED = 42
MAX_EPOCHS = 300
TRIAL_PATIENCE = 30          # early stop within a single trial
RESULTS_DIR = Path("results/person_b/optuna")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(train_data: list) -> torch.Tensor:
    labels = [int(d.y.item()) for d in train_data]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    return torch.tensor([n_neg / n_pos], device=DEVICE)


def build_model(model_name: str, params: dict) -> nn.Module:
    if model_name == "gat":
        return GATClassifier(
            in_channels=22,
            hidden_channels=params["hidden_channels"],
            num_heads=params["num_heads"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            edge_dim=7,
        )
    elif model_name == "gin":
        return GINClassifier(
            in_channels=22,
            hidden_channels=params["hidden_channels"],
            num_layers=params["num_layers"],
            dropout=params["dropout"],
            edge_dim=7,
        )
    raise ValueError(f"Unknown model: {model_name}")


def sample_params(trial: optuna.Trial, model_name: str) -> dict:
    params = {
        "hidden_channels": trial.suggest_categorical("hidden_channels", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }
    if model_name == "gat":
        params["num_heads"] = trial.suggest_categorical("num_heads", [2, 4, 8])
    return params


def train_one_epoch(model, loader, optimizer, criterion) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(logits, batch.y.squeeze(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def get_predictions(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(batch.y.squeeze(-1).cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def make_objective(model_name: str, train_data: list, val_data: list):
    def objective(trial: optuna.Trial) -> float:
        set_seed(SEED)  # fixed seed per trial -> isolates hyperparameter effect from init variance
        params = sample_params(trial, model_name)

        g = torch.Generator()
        g.manual_seed(SEED)
        train_loader = DataLoader(train_data, batch_size=params["batch_size"], shuffle=True, generator=g)
        val_loader = DataLoader(val_data, batch_size=params["batch_size"], shuffle=False)

        model = build_model(model_name, params).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_data))

        best_val_auc = 0.0
        patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            train_one_epoch(model, train_loader, optimizer, criterion)
            val_y, val_prob = get_predictions(model, val_loader)
            val_auc = evaluate_model(val_y, val_prob)["roc_auc"]

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1

            trial.report(val_auc, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if patience_counter >= TRIAL_PATIENCE:
                break

        return best_val_auc

    return objective


def retrain_best(model_name: str, best_params: dict, train_data, val_data, test_data) -> None:
    """Retrain with the winning hyperparameters for the full epoch budget and save results."""
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(train_data, batch_size=best_params["batch_size"], shuffle=True, generator=g)
    val_loader = DataLoader(val_data, batch_size=best_params["batch_size"], shuffle=False)
    test_loader = DataLoader(test_data, batch_size=best_params["batch_size"], shuffle=False)

    model = build_model(model_name, best_params).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10)
    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_data))

    best_val_auc, best_epoch, patience_counter = 0.0, 0, 0
    ckpt_path = RESULTS_DIR / f"{model_name}_optuna_best.pt"

    print(f"\nRetraining {model_name} with best params: {best_params}")
    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_y, val_prob = get_predictions(model, val_loader)
        val_auc = evaluate_model(val_y, val_prob)["roc_auc"]
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc, best_epoch, patience_counter = val_auc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{MAX_EPOCHS} | loss={train_loss:.4f} | val_auc={val_auc:.4f} | best={best_val_auc:.4f}")

        if patience_counter >= TRIAL_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        y_true, y_prob = get_predictions(model, loader)
        metrics = evaluate_model(y_true, y_prob)
        print_metrics(metrics, model_name=f"{model_name.upper()} (Optuna-tuned)", split=split_name)
        save_metrics(metrics, model_name=f"{model_name}_optuna", split=split_name, out_dir=RESULTS_DIR)


def run_study(model_name: str, timeout_seconds: int, train_data, val_data, test_data) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    storage_path = f"sqlite:///{RESULTS_DIR}/{model_name}_study.db"  # persists in Kaggle output, resumable

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        study_name=f"{model_name}_bbbp",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage_path,
        load_if_exists=True,
    )

    print(f"\n{'='*60}\nOptuna tuning: {model_name.upper()} | budget: {timeout_seconds/3600:.1f}h\n{'='*60}")
    start = time.time()
    study.optimize(make_objective(model_name, train_data, val_data), timeout=timeout_seconds)
    print(f"\nFinished in {(time.time()-start)/60:.1f} min | {len(study.trials)} trials")
    print(f"Best val AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    with open(RESULTS_DIR / f"{model_name}_best_params.json", "w") as f:
        json.dump({"model": model_name, "best_val_auc": study.best_value, "best_params": study.best_params}, f, indent=2)

    retrain_best(model_name, study.best_params, train_data, val_data, test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gat", "gin", "both"], default="both")
    parser.add_argument("--hours", type=float, default=3.0, help="Total time budget across selected model(s)")
    args = parser.parse_args()

    print("Loading frozen dataset...")
    train_data = torch.load("data/processed/train.pt", weights_only=False)
    val_data = torch.load("data/processed/val.pt", weights_only=False)
    test_data = torch.load("data/processed/test.pt", weights_only=False)
    print(f"  Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}")

    models = ["gat", "gin"] if args.model == "both" else [args.model]
    per_model_seconds = int((args.hours * 3600) / len(models))

    for m in models:
        run_study(m, per_model_seconds, train_data, val_data, test_data)