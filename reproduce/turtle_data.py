"""
Turtle data preprocessing module
=================================
Turtle counterpart of `data_preprocessing.py` (which targets the umineko
dataset).  The turtle dataset differs from umineko:

  * data : (N, 3, 200) -> 3 accelerometer channels (no pressure channel),
                          window length 200.
  * label: (N, 200)    -> per-timestep activity label inside each window.
  * 7 behaviour classes (0..6); -1 marks unknown / unlabelled timesteps.

Pipeline (mirrors the umineko reproduce, minus the foraging remap):
1. Load `data/turtle/data.npy` + `data/turtle/label.npy`.
2. Majority-vote each window's per-timestep labels into a single label.
3. Drop unknown (-1) windows for the supervised split.
4. 80/20 stratified train/test split (random_state=42).
5. From the train pool, keep 1% as the initial labelled set, 99% unlabelled
   (this seeds the active-learning loop), random_state=42.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# majority_value is identical to the umineko version; reuse it so behaviour
# matches exactly.
from data_preprocessing import majority_value


class data_loader_turtle(Dataset):
    """Minimal 2-argument Dataset matching the notebook's usage:
    ``data_loader_turtle(samples, labels)`` yielding ``(sample, label)``."""

    def __init__(self, samples, labels, device="cpu"):
        self.samples = torch.tensor(samples).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- #
# label dictionaries (turtle, from
# deepview/calculate_results/models/utils.py: labeldict_findstr_turtle)
# --------------------------------------------------------------------------- #
labeldict_findstr = {
    -1: "unknown",
    0: "Resting",
    1: "Swimming",
    2: "Stay in surface",
    3: "Gliding",
    4: "Feeding",
    5: "Scratching",
    6: "Breathing",
}
labeldict_wounknown_findstr = {
    0: "Resting",
    1: "Swimming",
    2: "Stay in surface",
    3: "Gliding",
    4: "Feeding",
    5: "Scratching",
    6: "Breathing",
}
label_colors = {
    0: "#5470C6", 1: "#91CC75", 2: "#FAC858", 3: "#EE6666", 4: "#73C0DE",
    5: "brown", 6: "pink", 7: "cyan", 8: "magenta", 9: "lime",
    10: "teal", 11: "violet", 12: "gold", 13: "coral", 14: "salmon",
}

NUM_CLASSES = 7


def load_data(data_path, label_path):
    """Return (data_b, label_b, vote_label) from the turtle npy files.

    vote_label is the majority-voted per-window label.  Unlike umineko there
    is no foraging remap; -1 already marks the unknown class."""
    data_b = np.asarray(np.load(data_path))
    label_b = np.asarray(np.load(label_path))
    vote_label = majority_value(label_b)
    return data_b, label_b, vote_label


def build_splits(data_b, vote_label, test_size=0.2, labeled_frac=0.01,
                 random_state=42):
    """Reproduce the umineko notebook's split logic for the turtle data.

    Returns a dict with X_train_full, X_test, y_*, X_labeled, X_unlabeled, ...
    """
    # drop unknown (-1) windows
    data_select = data_b[vote_label != -1]
    label_select = vote_label[vote_label != -1]

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
    dp = os.path.normpath(os.path.join(here, "..", "data", "turtle", "data.npy"))
    lp = os.path.normpath(os.path.join(here, "..", "data", "turtle", "label.npy"))
    data_b, label_b, vote_label = load_data(dp, lp)
    print("data_b", data_b.shape, "label_b", label_b.shape)
    s = build_splits(data_b, vote_label)
    print("X_labeled", s["X_labeled"].shape)
    print("X_unlabeled", s["X_unlabeled"].shape)
    print("X_test", s["X_test"].shape)
