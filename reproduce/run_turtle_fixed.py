"""Standalone full 20-iteration active-learning run on the corrected window-40
turtle data (mirrors the notebook loop).  Prints per-iteration timing + F1 and
writes metrics to results/turtle_fixed_metrics.csv."""
import os, sys, time, random
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from turtle_data import load_data, build_splits, data_loader_turtle, NUM_CLASSES
from turtle_model import TurtleNN, SupContrastiveLoss, freeze_encoders
from train_eval import (train_model, evaluate_model, evaluate_supContrast_model,
                        unfreeze_encoders, unfreeze_all, uncertainty_sampling)
from sklearn.metrics import f1_score

SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = 'cpu'
HERE = os.path.dirname(os.path.abspath(__file__))
data_b, label_b, vote_label = load_data(
    os.path.join(HERE, '..', 'data', 'turtle', 'data_w40.npy'),
    os.path.join(HERE, '..', 'data', 'turtle', 'label_w40.npy'))
s = build_splits(data_b, vote_label)
X_labeled, y_labeled = s['X_labeled'], s['y_labeled']
X_unlabeled, y_unlabeled = s['X_unlabeled'], s['y_unlabeled']
X_test, y_test = s['X_test'], s['y_test']
X_train_full = s['X_train_full']
print('labeled', X_labeled.shape, 'unlabeled', X_unlabeled.shape, 'test', X_test.shape, flush=True)

model = TurtleNN(number_classes=NUM_CLASSES, input_dim=128 * 5).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
supContrast_criterion = SupContrastiveLoss()
max_iter, warmup, batch_size = 20, 20, 4000
update_size = max(int(0.01 * len(X_train_full)), 2)


def bw(y, n=NUM_CLASSES):
    c = np.bincount(y.astype(int), minlength=n).astype(float); c[c == 0] = 1.0
    return torch.tensor(c.sum() / (n * c), dtype=torch.float, device=device)


metrics = []
iteration = 1
t0 = time.time()
while len(X_unlabeled) > 0:
    ti = time.time()
    classify_criterion = nn.CrossEntropyLoss(weight=bw(y_labeled))
    loader = DataLoader(data_loader_turtle(X_labeled.astype(float), y_labeled.astype(int)),
                        batch_size=batch_size, shuffle=True, drop_last=False)
    unfreeze_encoders(model)
    if iteration < warmup:
        model, _ = train_model(model, loader, supContrast_criterion, optimizer, 10, device, True)
    freeze_encoders(model)
    model, _ = train_model(model, loader, classify_criterion, optimizer, 50, device, False)
    unfreeze_all(model)
    model, _ = train_model(model, loader, classify_criterion, optimizer, 50, device, False)

    tr_acc, tr_macro, tr_micro, tr_pred, tr_gt = evaluate_model(model, loader, device)
    tr_w = f1_score(np.concatenate(tr_gt), np.concatenate(tr_pred), average='weighted')
    test_loader = DataLoader(data_loader_turtle(X_test.astype(float), y_test.astype(int)),
                             batch_size=batch_size, shuffle=False, drop_last=False)
    te_acc, te_macro, te_micro, _, te_pred, te_gt = evaluate_supContrast_model(
        model, test_loader, cal_threshold=0.5, device=device)
    te_w = f1_score(te_gt, te_pred, average='weighted')
    metrics.append(dict(iteration=iteration, labeled_samples=len(X_labeled),
                        train_acc=tr_acc, train_macro_f1=tr_macro, train_weighted_f1=tr_w,
                        test_acc=te_acc, test_macro_f1=te_macro, test_micro_f1=te_micro,
                        test_weighted_f1=te_w))
    print(f'iter {iteration:2d} | labeled={len(X_labeled):6d} | test acc={te_acc:.3f} '
          f'macroF1={te_macro:.3f} weightedF1={te_w:.3f} | {time.time()-ti:.0f}s', flush=True)

    sel = min(update_size, len(X_unlabeled))
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, _ = uncertainty_sampling(
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, sel, device)
    iteration += 1
    if iteration == max_iter + 1:
        break

df = pd.DataFrame(metrics)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
df.to_csv(os.path.join(HERE, 'results', 'turtle_fixed_metrics.csv'), index=False)
print(f'\nDONE in {time.time()-t0:.0f}s. Final 20% labels: '
      f"weightedF1={df.test_weighted_f1.iloc[-1]:.3f} macroF1={df.test_macro_f1.iloc[-1]:.3f}", flush=True)
