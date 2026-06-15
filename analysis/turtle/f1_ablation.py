"""
Why is turtle macro-F1 low?  Ablation on the CORRECTED data.
============================================================
Trains the supervised-contrastive method at a fixed 20% labelled fraction and
reports accuracy / macro-F1 / weighted-F1 / per-class-F1 for a sequence of
configs, isolating each suspected cause:

  A baseline      : as-is (classifier ends in Softmax + CrossEntropyLoss =>
                    double-softmax), no class weights, window 200
  B +logits       : remove the double-softmax (classifier outputs logits)
  C +classweight  : B + class-balanced CrossEntropy
  D +window40      : C but on 2 s windows (short behaviours survive)

Run:  python analysis/turtle/f1_ablation.py
"""
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

HERE = os.path.dirname(os.path.abspath(__file__))
REPRO = os.path.normpath(os.path.join(HERE, '..', '..', 'reproduce'))
DATADIR = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'turtle'))
sys.path.insert(0, REPRO)
from model_func import Encoder3d4          # noqa: E402
from model import SupContrastiveLoss        # noqa: E402

NAMES = ['Resting', 'Swimming', 'Stay_surf', 'Gliding', 'Feeding', 'Scratching', 'Breathing']
DEVICE = 'cpu'


def seed(s=2025):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class Net(nn.Module):
    """TurtleNN, but with a switch to emit classifier logits (no Softmax)."""
    def __init__(self, feat_dim, n_cls=7, softmax_head=False):
        super().__init__()
        self.acc_encoder = Encoder3d4()
        self.linear = nn.Linear(feat_dim, 32)
        self.projector = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 16))
        head = [nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, n_cls)]
        if softmax_head:
            head.append(nn.Softmax(dim=1))
        self.classifier = nn.Sequential(*head)

    def forward(self, x, if_contrast=True):
        f, _, _ = self.acc_encoder(x[:, :3, :])
        z = self.linear(f)
        return (self.projector(z) if if_contrast else self.classifier(z)), z


def set_grad(model, enc, proj, clf):
    for p in model.acc_encoder.parameters(): p.requires_grad = enc
    for p in model.linear.parameters():      p.requires_grad = enc
    for p in model.projector.parameters():   p.requires_grad = proj
    for p in model.classifier.parameters():  p.requires_grad = clf


def train(model, loader, crit, opt, epochs, contrast):
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            out, _ = model(xb.float(), if_contrast=contrast)
            loss = crit(out, yb.long())
            opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    out, _ = model(torch.tensor(X).float(), if_contrast=False)
    pred = out.argmax(1).numpy()
    return (accuracy_score(y, pred),
            f1_score(y, pred, average='macro'),
            f1_score(y, pred, average='weighted'),
            f1_score(y, pred, average=None, labels=list(range(7)), zero_division=0))


def run(tag, data_npy, label_npy, feat_dim, softmax_head, class_weight,
        supcon_ep=25, clf_ep=35, labeled_frac=0.20):
    seed()
    X = np.load(os.path.join(DATADIR, data_npy))
    L = np.load(os.path.join(DATADIR, label_npy))
    # per-window majority over labelled timesteps
    y = np.array([np.bincount(r[r >= 0]).argmax() if (r >= 0).any() else -1 for r in L])
    X, y = X[y >= 0], y[y >= 0]

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    Xl, _, yl, _ = train_test_split(Xtr, ytr, train_size=labeled_frac, stratify=ytr, random_state=42)

    model = Net(feat_dim, softmax_head=softmax_head).to(DEVICE)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(Xl), torch.tensor(yl)),
                        batch_size=2000, shuffle=True)

    if class_weight:
        w = compute_class_weight('balanced', classes=np.arange(7), y=yl)
        ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float))
    else:
        ce = nn.CrossEntropyLoss()
    supcon = SupContrastiveLoss()

    set_grad(model, True, True, False);  train(model, loader, supcon, opt, supcon_ep, True)
    set_grad(model, False, False, True); train(model, loader, ce, opt, clf_ep, False)
    set_grad(model, True, True, True);   train(model, loader, ce, opt, clf_ep, False)

    acc, mac, wf1, per = evaluate(model, Xte, yte)
    print(f'\n=== {tag} ===  labeled={len(Xl)}  test={len(Xte)}')
    print(f'  accuracy={acc:.3f}  macroF1={mac:.3f}  weightedF1={wf1:.3f}')
    print('  per-class F1: ' + '  '.join(f'{n}={v:.2f}' for n, v in zip(NAMES, per)))
    return dict(tag=tag, acc=acc, macro=mac, weighted=wf1, per=per)


if __name__ == '__main__':
    res = []
    res.append(run('A baseline (softmax+CE, no weight, win200)',
                   'data.npy', 'label.npy', 128 * 25, True, False))
    res.append(run('B +logits (fix double-softmax)',
                   'data.npy', 'label.npy', 128 * 25, False, False))
    res.append(run('C +class-weighted CE',
                   'data.npy', 'label.npy', 128 * 25, False, True))
    res.append(run('D +window40 (2s) +class-weight',
                   'data_w40.npy', 'label_w40.npy', 128 * 5, False, True))

    print('\n\n================ SUMMARY (20% labels) ================')
    print(f'{"config":<42}{"acc":>7}{"macroF1":>9}{"weightF1":>10}')
    for r in res:
        print(f'{r["tag"]:<42}{r["acc"]:>7.3f}{r["macro"]:>9.3f}{r["weighted"]:>10.3f}')
