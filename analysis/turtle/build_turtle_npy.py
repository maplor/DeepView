"""
Turtle dataset windowed-npy generator (corrected).
===================================================
Reads ALL 16 sessions (h5 sensor + Behaviors_*.csv annotations) from the Dryad
green-turtle dataset (doi:10.5061/dryad.hhmgqnkd9) and produces windowed
`data.npy` / `label.npy`, keeping ONLY labelled data.

Correctness notes (vs the first draft in data_statistics.ipynb):
  * SAMPLE_RATE = 20 Hz.  The dataset README states acceleration/gyroscope are
    recorded at 20 Hz (depth at 1 Hz).  The earlier draft used 200 Hz, which
    overflows every h5 file and produces garbage labels (Swimming wrongly
    became the 65% majority; the correct majority is Resting ~51%).
  * Behaviour intervals use the `Start (s)` / `Stop (s)` columns, which the
    README says are synchronised to the h5 file with its first line as t=0.
  * Only the 3 accelerometer channels are kept (gyro/depth dropped), matching
    the TurtleNN encoder.

Usage:
    python build_turtle_npy.py                 # canonical: WIN=200 STEP=200
    python build_turtle_npy.py --win 40 --step 20 --out-suffix _w40
"""
import os
import glob
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import h5py

SAMPLE_RATE = 20  # Hz (README: acc/gyro @ 20 Hz)

# Raw 46-behaviour vocabulary -> 7 canonical classes.  Behaviours not listed
# here (U-turns, Shaking, Watching, Obstacle, ...) are treated as unlabelled.
BEHAVIOR_TO_LABEL = {
    'Resting': 0, 'Resting in flow': 0, 'Resting active': 0, 'Resting watching': 0,
    'Swimming horizontally': 1, 'Swimming ascent': 1, 'Swimming descent': 1,
    'Swimming on the bottom': 1, 'Swimming 1 horizontally': 1, 'Swimming 1 ascent': 1,
    'Swimming 1 descent': 1, 'Swimming fast horizontally': 1, 'Swimming fast ascent': 1,
    'Swimming fast descent': 1, 'Swimming in place': 1, 'Flipper beat': 1, 'Escape': 1,
    'Stay in surface': 2,
    'Gliding ascent': 3, 'Gliding descent': 3,
    'Catching': 4, 'Catching jellyfish': 4, 'Chewing jellyfish': 4,
    'Chewing on movement': 4, 'Chewing stationary': 4, 'Foraging': 4,
    'Grabbing on movement': 4, 'Grabbing stationary': 4, 'Hunting jellyfish': 4,
    'Regurgitating': 4, 'Prospection': 4, 'Pursuit': 4, 'Sand': 4,
    'Scratching': 5, 'Scratching camera': 5, 'Scratching head': 5,
    'Breathing': 6,
}
LABEL_NAMES = {0: 'Resting', 1: 'Swimming', 2: 'Stay_in_surface', 3: 'Gliding',
               4: 'Feeding', 5: 'Scratching', 6: 'Breathing'}


def session_labels(n_samples, csv_path):
    """Per-sample label array (-1 = unlabelled) for one session."""
    labels = np.full(n_samples, -1, dtype=int)
    beh = pd.read_csv(csv_path, sep=';')
    for _, row in beh.iterrows():
        lid = BEHAVIOR_TO_LABEL.get(str(row.get('Behavior', '')).strip(), -1)
        if lid < 0:
            continue
        try:
            s, e = float(row['Start (s)']), float(row['Stop (s)'])
        except (KeyError, ValueError):
            continue
        if s < 0 or e < 0:
            continue
        i0, i1 = max(0, int(s * SAMPLE_RATE)), min(n_samples, int(e * SAMPLE_RATE))
        if i0 < i1:
            labels[i0:i1] = lid
    return labels


def build(raw_dir, out_data, out_label, win=200, step=200, keep_frac=0.5):
    h5_files = sorted(glob.glob(os.path.join(raw_dir, '*.h5')))
    sensors, labels = [], []
    print(f'Reading {len(h5_files)} sessions @ {SAMPLE_RATE} Hz ...')
    for h5_path in h5_files:
        base = os.path.basename(h5_path)[:-3]
        csv_path = os.path.join(raw_dir, f'Behaviors_{base}.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f'missing CSV for {base}')
        with h5py.File(h5_path, 'r') as f:
            sensor = f['data'][:]            # (N, 7) acc3 + gyro3 + depth
        lab = session_labels(sensor.shape[0], csv_path)
        sensors.append(sensor)
        labels.append(lab)
        print(f'  {base}: {sensor.shape[0]:>7d} samples  '
              f'labelled={int((lab >= 0).sum()):>7d} ({(lab >= 0).mean()*100:4.1f}%)')

    sensor_all = np.concatenate(sensors, axis=0)
    labels_all = np.concatenate(labels, axis=0)
    n_total, n_lab = len(sensor_all), int((labels_all >= 0).sum())
    print(f'\nTotal {n_total:,} samples, labelled {n_lab:,} ({n_lab/n_total*100:.1f}%)')

    # z-score the 3 accelerometer channels over labelled samples (== gaussian_std)
    acc = sensor_all[:, :3].astype(float)
    m = labels_all >= 0
    mean_v, std_v = acc[m].mean(axis=0), np.maximum(acc[m].std(axis=0), 1e-5)
    acc = (acc - mean_v) / std_v

    # non/overlapping sliding window; keep only windows that are mostly labelled
    data_w, label_w, voted = [], [], []
    for i in range(0, len(acc) - win + 1, step):
        seg_lab = labels_all[i:i + win]
        known = seg_lab[seg_lab >= 0]
        if len(known) / win < keep_frac:
            continue
        maj = Counter(known).most_common(1)[0][0]
        data_w.append(acc[i:i + win].T)          # (3, win)
        label_w.append(seg_lab)                  # (win,) per-timestep, -1 kept
        voted.append(maj)

    data_b = np.asarray(data_w, dtype=np.float64)
    label_b = np.asarray(label_w, dtype=np.int64)
    voted = np.asarray(voted)
    np.save(out_data, data_b)
    np.save(out_label, label_b)

    print(f'\nSaved {out_data}  {data_b.shape}')
    print(f'Saved {out_label} {label_b.shape}')
    print('\nPer-class window distribution (majority vote):')
    u, c = np.unique(voted, return_counts=True)
    for k, n in zip(u, c):
        print(f'  {k} {LABEL_NAMES.get(int(k), "?"):16s}: {n:6d} ({n/len(voted)*100:4.1f}%)')
    return data_b, label_b


if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    default_raw = os.path.normpath(os.path.join(here, '..', '..', 'data', 'turtle', 'raw', 'doi_10_5061'))
    default_dir = os.path.normpath(os.path.join(here, '..', '..', 'data', 'turtle'))
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default=default_raw)
    ap.add_argument('--win', type=int, default=200)
    ap.add_argument('--step', type=int, default=200)
    ap.add_argument('--keep-frac', type=float, default=0.5)
    ap.add_argument('--out-suffix', default='')
    args = ap.parse_args()
    out_data = os.path.join(default_dir, f'data{args.out_suffix}.npy')
    out_label = os.path.join(default_dir, f'label{args.out_suffix}.npy')
    build(args.raw, out_data, out_label, args.win, args.step, args.keep_frac)
