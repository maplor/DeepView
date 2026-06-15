"""
Main runner — reproduce `active_supContrast_iter20_visUnlabelData.ipynb`
========================================================================
End-to-end:
    1. load + preprocess `all_norm_data.pkl`
    2. 20-iteration active-learning loop (supervised contrastive warm-up +
       classifier training), evaluating train/test each iteration
    3. save per-iteration metrics (CSV), full results (pkl) and model weights
    4. (optional) reproduce the UMAP / label-propagation / time-series figures

Run:
    .venv/bin/python reproduce/run_reproduce.py            # metrics + figures
    .venv/bin/python reproduce/run_reproduce.py --no-viz   # metrics only
"""

import argparse
import os
import pickle
import random
import sys

# Let MPS (Apple Silicon GPU) fall back to CPU for unsupported ops instead of
# crashing. Must be set before torch is first used.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader

# allow running both as `python reproduce/run_reproduce.py` and from inside dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import load_data, build_splits, data_loader_umineko
from model import SimpleNN, SupContrastiveLoss
import torch.nn as nn
from train_eval import (
    train_model, evaluate_model, evaluate_supContrast_model,
    freeze_encoders, unfreeze_encoders, unfreeze_all, uncertainty_sampling,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PKL = os.path.normpath(
    os.path.join(HERE, "..", "analysis", "pape_results", "all_norm_data.pkl"))
RESULTS_DIR = os.path.join(HERE, "results")


def pick_device(name="auto"):
    """Resolve a torch.device. ``auto`` picks CUDA > MPS (Apple Silicon) > CPU."""
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(pkl_path=DEFAULT_PKL, device=None, max_iter=20, warmup=20,
         batch_size=4000, seed=2025, do_viz=True):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_random_seed(seed)

    if device is None:
        device = pick_device()
    elif isinstance(device, str):
        device = pick_device(device)
    print(f"torch {torch.__version__} | device {device} | seed {seed}")

    # ----- data --------------------------------------------------------- #
    data_b, label_b, vote_label = load_data(pkl_path)
    splits = build_splits(data_b, vote_label)
    X_labeled = splits["X_labeled"]; y_labeled = splits["y_labeled"]
    X_unlabeled = splits["X_unlabeled"]; y_unlabeled = splits["y_unlabeled"]
    X_test = splits["X_test"]; y_test = splits["y_test"]
    X_train_full = splits["X_train_full"]

    print("X_labeled shape:", X_labeled.shape)
    print("X_unlabeled shape:", X_unlabeled.shape)
    print("X_test shape:", X_test.shape)

    # ----- model -------------------------------------------------------- #
    model = SimpleNN().to(device)
    classify_criterion = nn.CrossEntropyLoss()
    supContrast_criterion = SupContrastiveLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    update_size = max(int(0.01 * len(X_train_full)), 2)

    metrics = []
    weight_list, loss_list, selected_labels_list = [], [], []
    test_pred_label_lists, test_truth_label_lists = [], []

    iteration = 1
    while len(X_unlabeled) > 0:
        print(f"==============Iteration {iteration}===============")
        print(f"Already labeled samples: {len(X_labeled)}")
        print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

        labeled_dataset = data_loader_umineko(
            X_labeled.astype(float), y_labeled.astype(int))
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size,
                                    shuffle=True, drop_last=False)

        # stage 1: supervised contrastive warm-up
        # (encoders + linear + projector trainable; classifier frozen)
        unfreeze_encoders(model)
        if iteration < warmup:
            model, _ = train_model(model, labeled_loader, supContrast_criterion,
                                   optimizer, epochs=10, device=device,
                                   if_contrast=True)

        # stage 2: classifier-only training
        freeze_encoders(model)
        model, _ = train_model(model, labeled_loader, classify_criterion,
                               optimizer, epochs=50, device=device,
                               if_contrast=False)
        # stage 3: unfreeze all, train classifier head
        unfreeze_all(model)
        model, avg_loss = train_model(model, labeled_loader, classify_criterion,
                                      optimizer, epochs=50, device=device,
                                      if_contrast=False)
        loss_list.append(np.average(avg_loss))

        # ----- evaluation ----------------------------------------------- #
        print("-------Training----------")
        tr_acc, tr_macro, tr_micro, _, _ = evaluate_model(
            model, labeled_loader, device)

        test_dataset = data_loader_umineko(
            X_test.astype(float), y_test.astype(int))
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, drop_last=False)
        print("-------Test----------")
        (te_acc, te_macro, te_micro, _, te_pred, te_truth) = \
            evaluate_supContrast_model(model, test_loader,
                                       cal_threshold=0.5, device=device)

        metrics.append({
            "iteration": iteration,
            "labeled_samples": len(X_labeled),
            "train_acc": tr_acc, "train_macro_f1": tr_macro,
            "train_micro_f1": tr_micro,
            "test_acc": te_acc, "test_macro_f1": te_macro,
            "test_micro_f1": te_micro,
        })
        test_pred_label_lists.append(te_pred)
        test_truth_label_lists.append(te_truth)
        weight_list.append({k: v.clone() for k, v in model.state_dict().items()})

        # ----- active-learning update ----------------------------------- #
        select_size = min(update_size, len(X_unlabeled))
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels = \
            uncertainty_sampling(X_labeled, y_labeled, X_unlabeled,
                                 y_unlabeled, model, select_size, device)
        selected_labels_list.append(selected_labels)

        iteration += 1
        if iteration == max_iter + 1:
            break

    print("All requested iterations complete!")

    # ----- save results ------------------------------------------------- #
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(RESULTS_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics -> {csv_path}")
    print(df.to_string(index=False))

    results = {
        "metrics": metrics,
        "loss_list": loss_list,
        "selected_labels_list": selected_labels_list,
        "test_pred_label_lists": test_pred_label_lists,
        "test_truth_label_lists": test_truth_label_lists,
        "weight_list": weight_list,
        "final_X_labeled": X_labeled, "final_y_labeled": y_labeled,
    }
    pkl_out = os.path.join(RESULTS_DIR, "results.pkl")
    with open(pkl_out, "wb") as f:
        pickle.dump(results, f)
    print(f"Saved full results -> {pkl_out}")

    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, "final_model.pth"))
    print(f"Saved final model weights -> {os.path.join(RESULTS_DIR, 'final_model.pth')}")

    # ----- visualisations ----------------------------------------------- #
    if do_viz:
        print("\n=============== Visualisations ===============")
        from visualize import run_visualizations
        run_visualizations(model, data_b, vote_label, X_labeled, y_labeled,
                           out_dir=RESULTS_DIR, max_iter=max_iter,
                           warmup=warmup, batch_size=batch_size, device=device)
        print(f"Saved figures (HTML + PDF) -> {RESULTS_DIR}")

    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", default=DEFAULT_PKL, help="path to all_norm_data.pkl")
    p.add_argument("--device", default="auto",
                   help="auto | cpu | cuda | mps (auto picks cuda>mps>cpu)")
    p.add_argument("--max-iter", type=int, default=20)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--no-viz", action="store_true", help="skip figures")
    a = p.parse_args()
    main(pkl_path=a.pkl, device=a.device, max_iter=a.max_iter,
         warmup=a.warmup, seed=a.seed, do_viz=not a.no_viz)
