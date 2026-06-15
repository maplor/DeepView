"""Generate `reproduce_walkthrough_bear.ipynb` — bear-dataset counterpart
of `reproduce_walkthrough.ipynb`.  Same step-by-step structure, keeping every
intermediate result inline, but targeting the bear dataset
(`data/bear/data.npy` + `label.npy`): 3 accel channels, window length 50,
10 behaviour classes."""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
def md(t): cells.append(new_markdown_cell(t))
def code(t): cells.append(new_code_cell(t))

md("""# 复现 walkthrough（bear 数据集）

本 notebook 与 `reproduce_walkthrough.ipynb`（umineko 数据集）、
`reproduce_walkthrough_turtle.ipynb`（turtle 数据集）结构完全一致，
但目标换成 **bear 数据集**，并**保留每一步的中间结果**：数据形状、标签分布、
数据切分、模型结构、20 轮主动学习的逐轮指标、指标曲线，以及 UMAP / 标签传播 /
时序标签等可视化。

bear 与 umineko / turtle 的差异：
- 数据 `(N, 3, 50)`：仅 3 轴加速度（无气压通道），窗口长度 50（turtle 为 200，
  与 umineko 的 50 相同）。
- 10 个行为类别（0..9）：dig / eat / groom / head_shake / pounce / rest / roll /
  run / swim / walk。数据**全部带标签**（无 -1 unknown），类别高度不均衡
  （rest 占 ~77%）。

代码复用 `reproduce/` 下的 bear 模块（`bear_data` / `bear_model`）与通用的
`train_eval`。""")

# ---- 0. setup ----
md("## 0. 环境与随机种子")
code("""import os, sys, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

HERE = os.path.abspath('.')
sys.path.insert(0, HERE)

from bear_data import (load_data, build_splits, data_loader_bear,
                       labeldict_findstr, label_colors,
                       labeldict_wounknown_findstr, NUM_CLASSES)
from bear_model import BearNN, SupContrastiveLoss, freeze_encoders
from data_preprocessing import majority_value
from train_eval import (train_model, evaluate_model, evaluate_supContrast_model,
                        unfreeze_encoders, unfreeze_all,
                        uncertainty_sampling, AE_eval_time_series)

def set_random_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    # GPU 优先：CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

SEED = 2025
set_random_seed(SEED)
device = get_device()
print('torch', torch.__version__, '| device', device, '| seed', SEED)""")

# ---- 1. data ----
md("""## 1. 数据读取与预处理

加载 `data/bear/data.npy` 与 `label.npy`，对每个窗口的逐帧标签做多数投票
（bear 无 foraging 重映射；数据全部带标签，无 unknown）。""")
code("""DATA = os.path.normpath(os.path.join(HERE, '..', 'data', 'bear', 'data.npy'))
LABEL = os.path.normpath(os.path.join(HERE, '..', 'data', 'bear', 'label.npy'))
data_b, label_b, vote_label = load_data(DATA, LABEL)
print('data_b :', data_b.shape, data_b.dtype, '  (N, channels=3accel, win_len=50)')
print('label_b:', label_b.shape, '  (per-timestep labels)')
print('vote_label:', vote_label.shape)""")

code("""# 投票后标签分布
uniq, cnt = np.unique(vote_label, return_counts=True)
dist = pd.DataFrame({'label': uniq,
                     'name': [labeldict_findstr.get(int(u), str(u)) for u in uniq],
                     'count': cnt})
dist""")

md("### 1.1 训练/测试切分 + labeled/unlabeled 切分")
code("""splits = build_splits(data_b, vote_label)
X_labeled, y_labeled   = splits['X_labeled'],   splits['y_labeled']
X_unlabeled, y_unlabeled = splits['X_unlabeled'], splits['y_unlabeled']
X_test, y_test         = splits['X_test'],      splits['y_test']
X_train_full           = splits['X_train_full']

print('X_labeled  :', X_labeled.shape)
print('X_unlabeled:', X_unlabeled.shape)
print('X_test     :', X_test.shape)""")

code("""# 初始 labeled 集合的类别分布
u, c = np.unique(y_labeled, return_counts=True)
pd.DataFrame({'label': u, 'name': [labeldict_findstr[int(i)] for i in u], 'count': c})""")

# ---- 2. model ----
md("""## 2. 模型

`BearNN`：3 轴加速度走单个 `Encoder3d4`，窗口 50 经三次 /2 池化得到 6，
特征维 128*6=768；Linear 到 32 维共享嵌入；再分出 projector(对比学习 16 维)
与 classifier(10 类)。""")
code("""model = BearNN(number_classes=NUM_CLASSES).to(device)
classify_criterion = nn.CrossEntropyLoss()
supContrast_criterion = SupContrastiveLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

n_params = sum(p.numel() for p in model.parameters())
print(f'总参数量: {n_params:,}')
print(model)""")

code("""# 单次前向，确认各阶段张量形状
model.eval()
with torch.no_grad():
    sample = torch.tensor(X_labeled[:4]).float().to(device)
    proj_out, feat32 = model(sample, if_contrast=True)
    clf_out, _       = model(sample, if_contrast=False)
print('共享32维特征 :', tuple(feat32.shape))
print('projector输出 :', tuple(proj_out.shape))
print('classifier输出:', tuple(clf_out.shape), '(logits over 10类)')""")

# ---- 3. active learning loop ----
md("""## 3. 主动学习训练循环（20 轮）

每轮三阶段：①监督对比 warm-up（解冻编码器，10 epoch）②冻结编码器训练分类器（50 epoch）
③全解冻再训分类头（50 epoch）。随后在测试集评估，并用**熵不确定性采样**新增 1% 样本。

下方每轮都会打印训练/测试的 Accuracy、Macro-F1、Micro-F1（中间结果全部保留）。

> bear 数据集体量较大（labeled 集从 ~1800 增长到 ~36000）。已自动选择 GPU（CUDA/MPS）；
> 若无可用 GPU 则回退 CPU，在 CPU 上该循环耗时较长。""")
code("""set_random_seed(SEED)  # 重置以保证可复现
model = BearNN(number_classes=NUM_CLASSES).to(device)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

max_iter, warmup, batch_size = 20, 20, 4000
update_size = max(int(0.01 * len(X_train_full)), 2)

metrics, loss_list, selected_labels_list = [], [], []
weight_list = []
iteration = 1
while len(X_unlabeled) > 0:
    print(f'============== Iteration {iteration} ==============')
    print(f'labeled={len(X_labeled)}  unlabeled={len(X_unlabeled)}')

    labeled_loader = DataLoader(
        data_loader_bear(X_labeled.astype(float), y_labeled.astype(int)),
        batch_size=batch_size, shuffle=True, drop_last=False)

    # stage 1: supervised contrastive warm-up
    unfreeze_encoders(model)
    if iteration < warmup:
        model, _ = train_model(model, labeled_loader, supContrast_criterion,
                               optimizer, epochs=10, device=device, if_contrast=True)
    # stage 2: classifier-only
    freeze_encoders(model)
    model, _ = train_model(model, labeled_loader, classify_criterion,
                           optimizer, epochs=50, device=device, if_contrast=False)
    # stage 3: unfreeze all
    unfreeze_all(model)
    model, avg_loss = train_model(model, labeled_loader, classify_criterion,
                                  optimizer, epochs=50, device=device, if_contrast=False)
    loss_list.append(np.average(avg_loss))

    print('-- Train --')
    tr_acc, tr_macro, tr_micro, _, _ = evaluate_model(model, labeled_loader, device)
    test_loader = DataLoader(
        data_loader_bear(X_test.astype(float), y_test.astype(int)),
        batch_size=batch_size, shuffle=False, drop_last=False)
    print('-- Test --')
    te_acc, te_macro, te_micro, _, _, _ = evaluate_supContrast_model(
        model, test_loader, cal_threshold=0.5, device=device)

    metrics.append(dict(iteration=iteration, labeled_samples=len(X_labeled),
                        train_acc=tr_acc, train_macro_f1=tr_macro, train_micro_f1=tr_micro,
                        test_acc=te_acc, test_macro_f1=te_macro, test_micro_f1=te_micro))
    weight_list.append({k: v.clone() for k, v in model.state_dict().items()})

    select_size = min(update_size, len(X_unlabeled))
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, sel = uncertainty_sampling(
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, model, select_size, device)
    selected_labels_list.append(sel)

    iteration += 1
    if iteration == max_iter + 1:
        break
print('训练循环完成')""")

# ---- 4. metrics ----
md("## 4. 逐轮指标汇总")
code("""df = pd.DataFrame(metrics)
df""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(df.iteration, df.train_acc, 'o-', label='train acc')
axes[0].plot(df.iteration, df.test_acc, 's-', label='test acc')
axes[0].set_title('Accuracy vs iteration'); axes[0].set_xlabel('iteration')
axes[0].legend(); axes[0].grid(alpha=.3)
axes[1].plot(df.iteration, df.train_macro_f1, 'o-', label='train macroF1')
axes[1].plot(df.iteration, df.test_macro_f1, 's-', label='test macroF1')
axes[1].plot(df.iteration, df.test_micro_f1, '^-', label='test microF1')
axes[1].set_title('F1 vs iteration'); axes[1].set_xlabel('iteration')
axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()""")

# ---- 5. visualization ----
md("""## 5. 可视化中间结果

构造 combined 集合（labeled 窗口 + 录音切片 [6000:9000]，该区间覆盖全部 10 类），
过模型取 32 维特征，UMAP 投影到 2 维；再用 `LabelSpreading` 做标签传播。下面把
UMAP 散点（真值 / 预测 / 传播标签）直接画在 notebook 中。""")
code("""from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
from umap import UMAP

startloc, endloc = 6000, 9000
combinedata = np.concatenate([X_labeled, data_b[startloc:endloc]])
combinelabel_true = np.concatenate([y_labeled, vote_label[startloc:endloc]])

loader = DataLoader(data_loader_bear(combinedata.astype(float), combinelabel_true.astype(int)),
                    batch_size=batch_size, shuffle=False, drop_last=False)
repres_list, _, pred_list, label_list = AE_eval_time_series(loader, model, device)

repre = np.concatenate(repres_list).reshape(len(combinedata), -1).astype(float)
proj = UMAP(n_components=2, random_state=SEED).fit_transform(repre)
all_labels = np.concatenate(label_list)
predictions = np.argmax(np.concatenate(pred_list), axis=1)
print('combined:', combinedata.shape, '| UMAP proj:', proj.shape)""")

code("""# 标签传播：把追加的录音切片标记为 -1(未知)，从已知 labeled 点扩散
combineunlabel = np.concatenate([y_labeled, np.full(vote_label[startloc:endloc].shape, -1)])
loader2 = DataLoader(data_loader_bear(combinedata.astype(float), combineunlabel.astype(int)),
                     batch_size=batch_size, shuffle=False, drop_last=False)
_, _, _, unlabel_list = AE_eval_time_series(loader2, model, device)
all_labels_before = np.concatenate(unlabel_list)

data_umap = StandardScaler().fit_transform(proj)
ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.2)
ls.fit(data_umap, all_labels_before)
propagated_labels = ls.transduction_
u, c = np.unique(propagated_labels, return_counts=True)
pd.DataFrame({'label': u, 'name': [labeldict_findstr.get(int(i), str(i)) for i in u], 'count': c})""")

code("""def scatter(ax, xy, labels, title):
    for lab in np.unique(labels):
        if lab < 0:
            m = labels == lab
            ax.scatter(xy[m,0], xy[m,1], s=6, c='lightgrey', alpha=0.3, label='unknown')
        else:
            m = labels == lab
            ax.scatter(xy[m,0], xy[m,1], s=6, alpha=0.6,
                       c=label_colors[int(lab)], label=labeldict_findstr.get(int(lab), str(lab)))
    ax.set_title(title); ax.set_xlabel('UMAP 0'); ax.set_ylabel('UMAP 1')
    ax.legend(markerscale=2, fontsize=7, loc='best')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
scatter(axes[0], data_umap, all_labels, 'Ground-truth (voted)')
scatter(axes[1], data_umap, predictions, 'Model prediction')
scatter(axes[2], data_umap, propagated_labels, 'Label propagation')
plt.tight_layout(); plt.show()""")

md("### 5.1 时序标签图：真值(散点) vs 传播标签 / 模型预测(背景色块)")
code("""import matplotlib.patches as patches
new_len = len(data_b[startloc:endloc])
true_new = all_labels[-new_len:]
prop_new = propagated_labels[-new_len:]
pred_new = predictions[-new_len:]

def ts_plot(ax, overlay, title):
    for lab, name in labeldict_findstr.items():
        if lab >= 0:
            idx = np.where(true_new == lab)[0]
            ax.scatter(idx, [lab]*len(idx), s=1, color=label_colors[lab], label=name)
    for start in range(0, new_len):
        for lab in np.unique(overlay[start:start+1]):
            if lab >= 0 and lab in label_colors:
                ax.add_patch(patches.Rectangle((start, lab-0.3), 1, 0.6, linewidth=0,
                             facecolor=label_colors[int(lab)], alpha=0.2))
    ax.set_title(title); ax.set_xlabel('Timestamp'); ax.set_ylabel('Activity')
    ax.set_yticks(list(labeldict_wounknown_findstr.keys()))
    ax.set_yticklabels(list(labeldict_wounknown_findstr.values()))
    ax.legend(markerscale=5, fontsize=7, loc='upper right')

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
ts_plot(axes[0], prop_new, 'Groundtruth (dots) vs Propagated labels (rectangles)')
ts_plot(axes[1], pred_new, 'Groundtruth (dots) vs Model-prediction labels (rectangles)')
plt.tight_layout(); plt.show()""")

code("""# 原始加速度三轴
plt.figure(figsize=(13,4))
acc = np.concatenate(data_b[startloc:endloc,:3,:].transpose(0,2,1), axis=0)
for i, name in enumerate(['axis_X','axis_Y','axis_Z']):
    plt.plot(acc[:,i], label=name)
plt.xlabel('Timestamp'); plt.ylabel('Acceleration [G]'); plt.title('Raw Acceleration Data')
plt.legend(loc='upper right'); plt.show()""")

md("""## 6. 小结

- 与 umineko / turtle 复现 notebook 结构一致，仅替换数据集与模型（单加速度编码器、
  10 类、窗口 50、特征维 768）。
- 所有中间结果（数据形状、分布、逐轮指标、特征/UMAP/标签传播）均已内联保留。
- bear 类别极度不均衡（rest 占 ~77%，pounce/run 等极少），
  Macro-F1 受稀有类影响明显，Accuracy / Micro-F1 受多数类 rest 主导。""")

nb = new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3'}
nb.metadata['language_info'] = {'name': 'python'}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reproduce_walkthrough_bear.ipynb')
with open(out, 'w') as f:
    nbf.write(nb, f)
print('wrote', out, '|', len(cells), 'cells')
