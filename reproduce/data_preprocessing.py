"""
Data preprocessing module
==========================
Reproduces the "数据读取" (data reading) block of
`active_supContrast_iter20_visUnlabelData.ipynb`.

Pipeline
--------
1. Load the pre-normalised umineko dataset (`all_norm_data.pkl`).
   - data : (N, 4, 50)  -> 3 accelerometer channels + 1 pressure channel,
                            window length 50.
   - label: (N, 50)     -> per-timestep activity label inside each window.
2. Majority-vote each window's per-timestep labels into a single label.
3. Map foraging (5) -> -2 (treated as "unknown").
4. Drop unknown (-2) windows for the supervised split.
5. 80/20 stratified train/test split (random_state=42).
6. From the train pool, keep 1% as the initial labelled set, 99% unlabelled
   (this seeds the active-learning loop), random_state=42.

All numeric constants are copied verbatim from the notebook.
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# majority_value: copied from deepview/calculate_results/models/utils.py
# (kept local so this package is self-contained and does not import the
#  fragile utils.py, which reads a hard-coded Windows CSV path on import).
# --------------------------------------------------------------------------- #
def majority_value(arr):
    """Collapse a (batch, win_len) label array to (batch,) via majority vote.

    Idempotent on 1-D input (already-voted labels)."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr)
    majority = []
    for row in np.atleast_1d(arr):
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
    return np.array(majority).astype(int)


class data_loader_umineko(Dataset):
    """Minimal 2-argument Dataset matching the notebook's usage:
    ``data_loader_umineko(samples, labels)`` yielding ``(sample, label)``."""

    def __init__(self, samples, labels, device="cpu"):
        self.samples = torch.tensor(samples).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- #
# label dictionaries (verbatim from notebook)
# --------------------------------------------------------------------------- #
labeldict_findstr = {
    -2: "unknown",
    -1: "unknown",
    0: "ground_stationary",
    1: "stationary",
    2: "bathing",
    3: "flying_active",
    4: "flying_passive",
}
labeldict_wounknown_findstr = {
    0: "ground_stationary",
    1: "stationary",
    2: "bathing",
    3: "flying_active",
    4: "flying_passive",
}
label_colors = {
    0: "#5470C6", 1: "#91CC75", 2: "#FAC858", 3: "#EE6666", 4: "#73C0DE",
    5: "brown", 6: "pink", 7: "cyan", 8: "magenta", 9: "lime",
    10: "teal", 11: "violet", 12: "gold", 13: "coral", 14: "salmon",
}
act_color_dict = {
    "ground_stationary": "#5470C6",
    "stationary": "#91CC75",
    "bathing": "#FAC858",
    "flying_active": "#EE6666",
    "flying_passive": "#73C0DE",
}


def load_data(pkl_path):
    """Return (data_b, label_b, vote_label) from the norm-data pickle.

    vote_label is the majority-voted per-window label with foraging (5)
    remapped to -2 (unknown), exactly as in the notebook."""
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    data_b = np.asarray(d["data"])
    label_b = np.asarray(d["label"])
    vote_label = majority_value(label_b)
    vote_label[vote_label == 5] = -2  # remove foraging
    return data_b, label_b, vote_label


def build_splits(data_b, vote_label, test_size=0.2, labeled_frac=0.01,
                 random_state=42):
    """Reproduce the notebook's train/test and labelled/unlabelled splits.

    Returns a dict with X_train_full, X_test, y_*, X_labeled, X_unlabeled, ...
    """
    # drop unknown (-2) windows
    data_select = data_b[vote_label != -2]
    label_select = vote_label[vote_label != -2]

    # 80/20 stratified train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        data_select, label_select,
        test_size=test_size, stratify=label_select, random_state=random_state,
    )

    # 1% labelled, 99% unlabelled (note: test_size=1-labeled_frac in notebook)
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
        X_train_full, y_train_full,
        test_size=1.0 - labeled_frac, random_state=random_state,
    )

    return {
        "X_train_full": X_train_full, "y_train_full": y_train_full,
        "X_test": X_test, "y_test": y_test,
        "X_labeled": X_labeled, "y_labeled": y_labeled,
        "X_unlabeled": X_unlabeled, "y_unlabeled": y_unlabeled,
    }


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    default_pkl = os.path.normpath(
        os.path.join(here, "..", "analysis", "pape_results", "all_norm_data.pkl")
    )
    data_b, label_b, vote_label = load_data(default_pkl)
    print("data_b", data_b.shape, "label_b", label_b.shape)
    s = build_splits(data_b, vote_label)
    print("X_labeled", s["X_labeled"].shape)
    print("X_unlabeled", s["X_unlabeled"].shape)
    print("X_test", s["X_test"].shape)
