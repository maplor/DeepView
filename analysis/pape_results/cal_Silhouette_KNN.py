#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cal_Silhouette_KNN.py

Read a saved '*_results.pkl' (produced by active_supContrast.py),
rebuild the model at each saved iteration, extract latent embeddings
via AE_eval_time_series, and compute:
  - Silhouette / Davies-Bouldin / Calinski-Harabasz
  - kNN (k-fold CV) accuracy & macro-F1

Outputs: <prefix>_sil_knn.csv and <prefix>_sil_knn.json

Example:
python cal_Silhouette_KNN.py --pkl AccelTemp__Contrast1_warm20_seed2025_new_epoch20_results.pkl \
  --sensors Accel,Temp --device cuda:0 --metric cosine --k 5 --folds 5
"""
import os, json, pickle, argparse, csv, sys
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader

# --- Project utilities (must exist in your environment) ---
from deepview.calculate_results.models.utils import (
    data_loader_umineko,
    AE_eval_time_series,
    majority_value,
    read_sensor_data,
    process_sensors,
    sliding_window,
    sensor_list_to_tag,
)
from deepview.calculate_results.data.umineko.model_func import get_model_for_sensors

# --- Metrics helpers ---
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score


def _concat(arr_list):
    return np.concatenate(arr_list, axis=0) if isinstance(arr_list, (list, tuple)) else arr_list

def _pool_time_if_needed(x, how='mean'):
    x = np.asarray(x)
    if x.ndim == 3:
        if how == 'mean':
            x = x.mean(axis=-1)
        elif how == 'max':
            x = x.max(axis=-1)
        else:
            raise ValueError(f"Unknown pool method: {how}")
    return x

def _majority_vote_rows(Y2d, ignore_label=None):
    Y2d = np.asarray(Y2d)
    if Y2d.ndim == 1:
        return Y2d
    out = np.empty((Y2d.shape[0],), dtype=int)
    for i, row in enumerate(Y2d):
        if ignore_label is not None:
            row = row[row != ignore_label]
        if row.size == 0:
            out[i] = -1
            continue
        offset = 0
        m = row.min()
        if m < 0:
            offset = -int(m)
        binc = np.bincount((row + offset).astype(int))
        out[i] = int(np.argmax(binc) - offset)
    return out

def prepare_X_y_from_AE_eval(representation_list, label_list, pool='mean', ignore_label=None, l2_normalize=False):
    X = _concat(representation_list)
    y = _concat(label_list)
    X = _pool_time_if_needed(X, how=pool)
    y = _majority_vote_rows(y, ignore_label=ignore_label)
    keep = y != -1
    X, y = X[keep], y[keep]
    if l2_normalize:
        denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = X / denom
    return X, y

def separability_metrics(X, y, silhouette_metric='euclidean'):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return {"silhouette": float("nan"), "DB": float("nan"), "CH": float("nan")}
    sil = silhouette_score(X, y, metric=silhouette_metric)
    db  = davies_bouldin_score(X, y)
    ch  = calinski_harabasz_score(X, y)
    return {"silhouette": float(sil), "DB": float(db), "CH": float(ch)}

def knn_cv_report(X, y, n_splits=5, k=5, weights='distance', random_state=0):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights, n_jobs=-1)
        knn.fit(Xtr, ytr)
        yhat = knn.predict(Xte)
        accs.append(accuracy_score(yte, yhat))
        f1s.append(f1_score(yte, yhat, average='macro'))
    accs, f1s = np.asarray(accs), np.asarray(f1s)
    return {
        "acc_mean": float(accs.mean()), "acc_std": float(accs.std(ddof=1) if len(accs)>1 else 0.0),
        "f1_mean":  float(f1s.mean()),  "f1_std":  float(f1s.std(ddof=1)  if len(f1s)>1 else 0.0),
        "k": int(k), "folds": int(n_splits)
    }


def read_pickle_results(path:str) -> Dict[str, Any]:
    """
    Training saved multiple dicts into one PKL:
        with open(..._results.pkl, 'wb') as f:
            pickle.dump({key: value}, f)  # repeated
    We must load until EOF and merge.
    """
    merged: Dict[str, Any] = {}
    with open(path, 'rb') as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                break
            if isinstance(obj, dict):
                merged.update(obj)
    return merged


def ensure_data_npz(tag:str):
    """
    Load <tag>_data.npz created by training. If missing, raise an error asking user
    to provide --data-npz or regenerate via the training script.
    """
    npz_path = f"{tag}_data.npz"
    if os.path.exists(npz_path):
        with np.load(npz_path, allow_pickle=False) as f:
            data_b = f["data"]
            label_b = f["label"]
        return data_b, label_b
    raise FileNotFoundError(
        f"Missing {npz_path}. Please run the training script to create it, or supply --data-npz."
    )

def ensure_data_pkl(tag: str):
    with open(f"{tag}_data.pkl", "rb") as f:
        obj = pickle.load(f)
    return obj["data"], obj["label"]

def make_plot_loader(data_b, label_b, device:str, batch_size:int):
    major_label_b = majority_value(label_b)
    plot_dataset = data_loader_umineko(data_b.astype(float),
                                       major_label_b.astype(int),
                                       label_b.astype(int),
                                       device=device)
    return DataLoader(plot_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def main(alg, argspkl):
    parser = argparse.ArgumentParser(description="Compute Silhouette and kNN on latent space from saved PKL checkpoints.")
    parser.add_argument("--pkl", type=str,
                        default=r'Accel__entropy_Contrast1_warm0_seed2025_epoch20_results.pkl',
                        # default=r'AccelGyroDepth__entropy_Contrast1_SimCLR_warm20_seed2025_epoch20_results.pkl',
                        # default=r'acc__Random_Contrast1_warm20_seed34_new_epoch20_results.pkl',
                        # default=r'acc_tmp__entropy_Contrast1_warm20_seed34_new_epoch20_results.pkl',
                        help="Path to '*_results.pkl' produced by training.")
    parser.add_argument("--sensors", type=str, default=["Accel"],
                        help="Comma-separated sensor names (e.g., 'Accel,Temp' or 'Accel').")
    parser.add_argument("--compute-gps", action="store_true",
                        default=True,
                        help="Match training: sensor_list_to_tag(..., compute_gps=True). Use --no-compute-gps to disable.")
    parser.add_argument("--no-compute-gps", dest="compute_gps", action="store_false")
    parser.add_argument("--data-npz", type=str,
                        default=None,
                        help="Optional path to a prebuilt '<tag>_data.npz'. If omitted, use tag-derived npz in CWD.")
    parser.add_argument("--batch-size", type=int, default=4000)
    parser.add_argument("--device", type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "max"], help="Time-pooling over [C,T] -> [C].")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine","euclidean"], help="Metric for silhouette.")
    parser.add_argument("--k", type=int, default=5, help="k for kNN.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified K-Fold for kNN.")
    parser.add_argument("--l2norm", action="store_true", default=True, help="L2-normalize embeddings before metrics.")
    parser.add_argument("--no-l2norm", dest="l2norm", action="store_false")
    # parser.add_argument("--last-only", 
    #                     action="store_true", 
    #                     help="Only evaluate the last checkpoint (fast).")
    # parser.add_argument("--out-prefix", type=str, default=None, help="Prefix for output CSV/JSON. Defaults to PKL basename.")
    args = parser.parse_args()
    # animal = 'omizunagidori'
    animal = 'bear'
    # root_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko'
    # root_path = r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori'
    # root_path = r'D:\code\DeepView\deepview\calculate_results\data\bear'
    root_path = f"D:\code\DeepView\deepview\calculate_results\data\{animal}"
    args.pkl = argspkl


    # sensor_list = [s.strip() for s in args.sensors.split(",") if s.strip()]
    tag = sensor_list_to_tag(args.sensors, compute_gps=args.compute_gps)
    print(f"Using tag: {tag}  | sensors={args.sensors}  | device={args.device}")

    # Load PKL
    res = read_pickle_results(os.path.join(root_path, args.pkl))
    if "weight_list" not in res:
        raise RuntimeError(f"'weight_list' not found in {os.path.join(root_path, args.pkl)}. Did the training script dump it?")
    weight_list: List[Dict[str, Any]] = res["weight_list"]
    label_data_list = res.get("label_data_list", [])
    unlabel_data_list = res.get("unlabel_data_list", [])

    # Load data
    if args.data_npz is not None:
        with np.load(os.path.join(root_path, args.data_npz), allow_pickle=False) as f:
            data_b = f["data"]
            label_b = f["label"]
    else:
        # data_b, label_b = ensure_data_npz(os.path.join(root_path, "Accel"))
        # data_b, label_b = ensure_data_npz(os.path.join(root_path, tag))
        # data_b, label_b = ensure_data_pkl(os.path.join(root_path, "AccelGyroDepth"))  # bear and turtle
        data_b, label_b = ensure_data_pkl(os.path.join(root_path, "Accel"))  # bear and turtle

    num_classes = int(np.max(label_b) + 1)
    plot_loader = make_plot_loader(data_b, label_b, device=args.device, batch_size=args.batch_size)

    # if args.out_prefix is None:
    #     base = os.path.splitext(os.path.basename(args.pkl))[0]
    #     out_prefix = base
    # else:
    #     out_prefix = args.out_prefix
    # out_csv = f"{out_prefix}_sil_knn.csv"
    # out_json = f"{out_prefix}_sil_knn.json"

    rows = []
    # iters = [len(weight_list)-1] if args.last_only else list(range(len(weight_list)))
    for i in range(20):
        # i = -1  # 最后一个
        state_dict = weight_list[i]
        model = get_model_for_sensors(args.sensors, number_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        model.to(args.device).eval()

        repres_list, sample_list, pred_list, label_list = AE_eval_time_series(plot_loader, model, args.device)
        X, y = prepare_X_y_from_AE_eval(repres_list, label_list, pool=args.pool, l2_normalize=args.l2norm)

        sep = separability_metrics(X, y, silhouette_metric=args.metric)
        knn_res = knn_cv_report(X, y, n_splits=args.folds, k=args.k, weights='distance', random_state=0)

        labeled_n = len(label_data_list[i][0]) if i < len(label_data_list) else None
        unlabeled_n = len(unlabel_data_list[i][0]) if i < len(unlabel_data_list) else None

        row = {
            "iteration": i+1,
            "labeled_n": labeled_n,
            "unlabeled_n": unlabeled_n,
            "silhouette": sep["silhouette"],
            "DB": sep["DB"],
            "CH": sep["CH"],
            "knn_acc_mean": knn_res["acc_mean"],
            "knn_acc_std": knn_res["acc_std"],
            "knn_f1_mean": knn_res["f1_mean"],
            "knn_f1_std": knn_res["f1_std"],
            "k": knn_res["k"],
            "folds": knn_res["folds"],
            "pool": args.pool,
            "metric": args.metric,
            "l2norm": args.l2norm,
        }
        rows.append(row)

        print(f"[Iter {i+1}] Silhouette={row['silhouette']:.3f} | DB={row['DB']:.3f} | CH={row['CH']:.1f} | "
              f"kNN k={row['k']} folds={row['folds']} Acc={row['knn_acc_mean']:.3f}±{row['knn_acc_std']:.3f} "
              f"| Macro-F1={row['knn_f1_mean']:.3f}±{row['knn_f1_std']:.3f} | labeled={labeled_n} unlabeled={unlabeled_n}")

    with open(f"{tag}_{animal}_{alg}.txt", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # with open(out_csv, "w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    #     writer.writeheader()
    #     for r in rows:
    #         writer.writerow(r)
    # 
    # with open(out_json, "w", encoding="utf-8") as f:
    #     json.dump(rows, f, ensure_ascii=False, indent=2)
    # 
    # print(f"\nSaved: {out_csv}")
    # print(f"Saved: {out_json}")


if __name__ == "__main__":
    algs = ['albe']
    # algs = ['random', 'albe', 'simclr']
    for alg in algs:
        if alg == 'random':
            argspkl = r'Accel__entropy_Contrast1_warm0_seed2025_epoch20_results.pkl'  # ; alg = 'random'
            main(alg, argspkl)
        elif alg == 'albe':
            # argspkl = r'Accel__entropy_Contrast1_warm20_seed2025_epoch20_results.pkl'  # ; alg = 'albe'
            argspkl = r'Accel__repreSamp_Contrast1_warm20_seed10_epoch2025_results.pkl'  # ; alg = 'albe'
            main(alg, argspkl)
        elif alg == 'simclr':
            argspkl = r'Accel__entropy_Contrast1_SimCLR_warm20_seed2025_epoch20_results.pkl'  # ; alg = 'simclr'
            main(alg, argspkl)
        else:
            print('not support alg:', alg)

    # main(argspkl)
    # try:
    #     main()
    # except Exception as e:
    #     print("ERROR:", e)
    #     sys.exit(1)
