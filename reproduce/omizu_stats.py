"""
Omizunagidori data & label statistics / sanity check
=====================================================
Standalone script that validates `data/omizunagidori/data.npy` and
`label.npy` BEFORE any training, so you can confirm the data and labels are
correct (问题1: 先 check 数据和 label 是否准确).

It reports, with no model involved:

  1. shapes / dtypes / memory
  2. data integrity — NaN / Inf, per-channel min/max/mean/std (是否已归一化)
  3. per-timestep label distribution (with class names) + out-of-vocabulary
     label values (e.g. the stray 7 = unknown)
  4. per-window majority-vote label distribution (with class names)
  5. window purity — how many windows are single-label vs mixed
  6. how many windows survive the supervised split (after dropping unknown),
     and the train/test/labeled/unlabeled sizes + their class balance
  7. a PASS/WARN summary of automatic sanity checks

Run (from repo root, project venv):
    .venv/bin/python reproduce/omizu_stats.py
    .venv/bin/python reproduce/omizu_stats.py --csv reproduce/results/omizu_stats.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from omizu_data import (load_data, build_splits, labeldict_findstr,
                        VALID_LABELS, NUM_CLASSES)


def _name(lbl):
    return labeldict_findstr.get(int(lbl), f"<unknown id {int(lbl)}>")


def _dist_frame(values):
    u, c = np.unique(values, return_counts=True)
    total = c.sum()
    return pd.DataFrame({
        "label": u,
        "name": [_name(v) for v in u],
        "count": c,
        "pct": (100.0 * c / total).round(3),
    })


def main():
    ap = argparse.ArgumentParser(description="Omizunagidori data/label statistics")
    ap.add_argument("--data", default=os.path.normpath(
        os.path.join(HERE, "..", "data", "omizunagidori", "data.npy")))
    ap.add_argument("--label", default=os.path.normpath(
        os.path.join(HERE, "..", "data", "omizunagidori", "label.npy")))
    ap.add_argument("--csv", default=None,
                    help="optional path to dump the window-label distribution")
    args = ap.parse_args()

    checks = []  # (passed: bool, message: str)

    def check(cond, msg):
        checks.append((bool(cond), msg))

    print("=" * 70)
    print("OMIZUNAGIDORI DATA / LABEL STATISTICS")
    print("=" * 70)
    print(f"data : {args.data}")
    print(f"label: {args.label}")

    # ---- raw load (no remap / no drop yet) so we can see everything --------
    data_b = np.asarray(np.load(args.data))
    label_raw = np.asarray(np.load(args.label))

    # ---- 1. shapes / dtypes ------------------------------------------------
    print("\n[1] shapes / dtypes")
    print(f"  data : {data_b.shape}  {data_b.dtype}  "
          f"({data_b.nbytes / 1e6:.1f} MB)")
    print(f"  label: {label_raw.shape}  {label_raw.dtype}  "
          f"({label_raw.nbytes / 1e6:.1f} MB)")
    check(data_b.ndim == 3 and data_b.shape[1] == 3,
          f"data is (N, 3, win): got {data_b.shape}")
    check(label_raw.ndim == 2 and label_raw.shape[0] == data_b.shape[0]
          and label_raw.shape[1] == data_b.shape[2],
          f"label (N, win) aligns with data: {label_raw.shape} vs {data_b.shape}")

    # ---- 2. data integrity -------------------------------------------------
    print("\n[2] data integrity")
    n_nan = int(np.isnan(data_b).sum())
    n_inf = int(np.isinf(data_b).sum())
    print(f"  NaN: {n_nan}   Inf: {n_inf}")
    check(n_nan == 0, "no NaN in data")
    check(n_inf == 0, "no Inf in data")
    ch_stats = []
    for c in range(data_b.shape[1]):
        x = data_b[:, c, :]
        ch_stats.append(dict(channel=c, min=x.min(), max=x.max(),
                             mean=x.mean(), std=x.std()))
    ch_df = pd.DataFrame(ch_stats)
    print(ch_df.to_string(index=False,
                          float_format=lambda v: f"{v:8.4f}"))
    normalized = all(abs(r["mean"]) < 1e-2 and abs(r["std"] - 1.0) < 1e-2
                     for r in ch_stats)
    check(normalized, "per-channel mean~0 / std~1 (data appears z-normalized)")

    # ---- 3. per-timestep labels -------------------------------------------
    print("\n[3] per-timestep label distribution")
    lab_int = label_raw.astype(int)
    n_noninteger = int((label_raw != lab_int).sum())
    print(f"  non-integer label values: {n_noninteger}")
    check(n_noninteger == 0, "all label values are integer-valued")
    ts_df = _dist_frame(lab_int.ravel())
    print(ts_df.to_string(index=False))
    oov = sorted(set(np.unique(lab_int).tolist()) - VALID_LABELS)
    if oov:
        print(f"  ! out-of-vocabulary label ids (not in 0..{NUM_CLASSES - 1}): {oov}")
        print("    -> these timesteps are treated as 'unknown' and the windows "
              "they dominate are dropped from the supervised split.")
    check(len(oov) == 0 or oov == [7],
          f"only known classes 0..{NUM_CLASSES - 1} (+stray 7=unknown) present; "
          f"got extras {oov}")

    # ---- 4. window majority-vote labels -----------------------------------
    print("\n[4] per-window majority-vote label distribution")
    _, _, vote_label = load_data(args.data, args.label)
    win_df = _dist_frame(vote_label)
    print(win_df.to_string(index=False))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        win_df.to_csv(args.csv, index=False)
        print(f"  (window-label distribution written to {args.csv})")

    # ---- 5. window purity --------------------------------------------------
    print("\n[5] window purity (single-label vs mixed)")
    n_unique_per_win = np.array([len(np.unique(row)) for row in lab_int])
    pure = int((n_unique_per_win == 1).sum())
    print(f"  single-label windows: {pure} / {len(lab_int)} "
          f"({100.0 * pure / len(lab_int):.1f}%)")
    print(f"  mixed-label windows : {len(lab_int) - pure}")

    # ---- 6. supervised split sizes / balance ------------------------------
    print("\n[6] supervised split (drop unknown, 80/20 stratified, 1% labelled)")
    n_unknown_win = int((~np.isin(vote_label, list(VALID_LABELS))).sum())
    print(f"  windows dropped as unknown (majority vote outside 0..{NUM_CLASSES - 1}): "
          f"{n_unknown_win}")
    s = build_splits(data_b, vote_label)
    print(f"  X_train_full: {s['X_train_full'].shape}")
    print(f"  X_test      : {s['X_test'].shape}")
    print(f"  X_labeled   : {s['X_labeled'].shape}  (initial active-learning seed)")
    print(f"  X_unlabeled : {s['X_unlabeled'].shape}")
    print("\n  initial labelled-set class balance:")
    print(_dist_frame(s["y_labeled"]).to_string(index=False))
    # every test class should also appear in the training pool
    train_classes = set(np.unique(s["y_train_full"]).tolist())
    test_classes = set(np.unique(s["y_test"]).tolist())
    check(test_classes <= train_classes,
          "every test class is represented in the train pool")
    check(len(s["X_labeled"]) > 0 and len(s["X_unlabeled"]) > 0,
          "labelled and unlabelled seeds are both non-empty")

    # ---- 7. summary --------------------------------------------------------
    print("\n" + "=" * 70)
    print("SANITY-CHECK SUMMARY")
    print("=" * 70)
    n_pass = sum(1 for ok, _ in checks if ok)
    for ok, msg in checks:
        print(f"  [{'PASS' if ok else 'WARN'}] {msg}")
    print(f"\n{n_pass}/{len(checks)} checks passed.")
    if n_pass != len(checks):
        print("  -> review the WARN lines above before trusting the reproduce run.")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
