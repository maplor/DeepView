"""
Omizunagidori (streaked shearwater) data preprocessing module
=============================================================
Omizunagidori counterpart of `bear_data.py` / `turtle_data.py` /
`data_preprocessing.py`.  The omizunagidori dataset is structured like bear
(raw npy arrays, single 3-channel accelerometer stream), but with its own
class set:

  * data : (N, 3, 50)  -> 3 accelerometer channels (no pressure channel),
                          window length 50 (same as bear / umineko).
  * label: (N, 50)     -> per-timestep activity label inside each window.
  * 7 behaviour classes (0..6); per-timestep labels are stored as floats but
    are integer-valued.

  Label id mapping (from the raw logbot `label` -> `label_id` column, see
  deepview/generate_training_dataset/utils.py and
  deepview/calculate_results/models/utils.py:labeldict_findstr_omizu_org):

        0 stationary      1 preening        2 bathing
        3 flight_take_off 4 flight_cruising 5 foraging_dive
        6 surface_seizing

  The windowed `label.npy` also contains a tiny residual value 7 (a leftover
  "unknown" / out-of-vocabulary marker: it appears in only 4 windows, majority
  of just 1).  Anything outside 0..6 is therefore treated as unknown and is
  dropped from the supervised split — this also keeps the stratified split well
  defined (a class with a single window cannot be stratified-split).

Pipeline (mirrors the bear / turtle reproduce):
1. Load `data/omizunagidori/data.npy` + `data/omizunagidori/label.npy`.
2. Majority-vote each window's per-timestep labels into a single label.
3. Drop unknown windows (majority vote outside 0..6) for the supervised split.
4. 80/20 stratified train/test split (random_state=42).
5. From the train pool, keep 1% as the initial labelled set, 99% unlabelled
   (this seeds the active-learning loop), random_state=42.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# majority_value is identical to the umineko/turtle/bear version; reuse it so
# behaviour matches exactly.
from data_preprocessing import majority_value


class data_loader_omizu(Dataset):
    """Minimal 2-argument Dataset matching the notebook's usage:
    ``data_loader_omizu(samples, labels)`` yielding ``(sample, label)``."""

    def __init__(self, samples, labels, device="cpu"):
        self.samples = torch.tensor(samples).to(device)
        self.labels = torch.tensor(labels).to(device)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


# --------------------------------------------------------------------------- #
# label dictionaries (omizunagidori, from
# deepview/calculate_results/models/utils.py: labeldict_findstr_omizu_org)
# --------------------------------------------------------------------------- #
labeldict_findstr = {
    -1: "unknown",
    0: "stationary",
    1: "preening",
    2: "bathing",
    3: "flight_take_off",
    4: "flight_cruising",
    5: "foraging_dive",
    6: "surface_seizing",
}
labeldict_wounknown_findstr = {
    0: "stationary",
    1: "preening",
    2: "bathing",
    3: "flight_take_off",
    4: "flight_cruising",
    5: "foraging_dive",
    6: "surface_seizing",
}
label_colors = {
    0: "#5470C6", 1: "#91CC75", 2: "#FAC858", 3: "#EE6666", 4: "#73C0DE",
    5: "brown", 6: "pink", 7: "cyan", 8: "magenta", 9: "lime",
    10: "teal", 11: "violet", 12: "gold", 13: "coral", 14: "salmon",
}

NUM_CLASSES = 7
VALID_LABELS = set(range(NUM_CLASSES))  # 0..6; anything else -> unknown


def load_data(data_path, label_path):
    """Return (data_b, label_b, vote_label) from the omizunagidori npy files.

    vote_label is the majority-voted per-window label.  Per-timestep labels
    are integer-valued floats; they are cast to int.  Values outside 0..6 are
    left as-is here (build_splits drops them); nothing is remapped."""
    data_b = np.asarray(np.load(data_path))
    label_b = np.asarray(np.load(label_path)).astype(int)
    vote_label = majority_value(label_b)
    return data_b, label_b, vote_label


def build_splits(data_b, vote_label, test_size=0.2, labeled_frac=0.01,
                 random_state=42):
    """Reproduce the bear/turtle/umineko notebook's split logic for omizu data.

    Unknown windows (majority vote outside 0..6, e.g. the stray label 7) are
    dropped before splitting.  Returns a dict with X_train_full, X_test, y_*,
    X_labeled, X_unlabeled, ...
    """
    # keep only windows whose majority vote is a valid behaviour class (0..6)
    keep = np.isin(vote_label, list(VALID_LABELS))
    data_select = data_b[keep]
    label_select = vote_label[keep]

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
    dp = os.path.normpath(os.path.join(here, "..", "data", "omizunagidori", "data.npy"))
    lp = os.path.normpath(os.path.join(here, "..", "data", "omizunagidori", "label.npy"))
    data_b, label_b, vote_label = load_data(dp, lp)
    print("data_b", data_b.shape, "label_b", label_b.shape)
    s = build_splits(data_b, vote_label)
    print("X_labeled", s["X_labeled"].shape)
    print("X_unlabeled", s["X_unlabeled"].shape)
    print("X_test", s["X_test"].shape)
