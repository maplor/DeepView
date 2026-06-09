
"""
Replot latent scatter from saved epoch weights (e.g., epoch=5).

功能:
1) 读取结果PKL(流式pickle dump)，抽出 weight_list -> 指定迭代的 state_dict
2) 重建模型并载入权重
3) 重新读取/处理传感器数据，跑 AE_eval_time_series 拿到特征
4) UMAP到2D，并用全新文件名写出交互式HTML散点

使用前请检查【参数区】里的路径/设备等变量。
"""
import copy
import io
# import os
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
# from umap import UMAP
import torch
# import torch.nn as nn
from torch.utils.data import DataLoader
# import os
# import math
import json
# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
# from collections import Counter, defaultdict
#
# from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    # normalized_mutual_info_score, adjusted_rand_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    precision_recall_fscore_support
)
# from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold, train_test_split

# import networkx as nx
import os
import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from collections import Counter, defaultdict

# ---------- Optional deps: UMAP & HDBSCAN ----------
_HAS_UMAP = True
try:
    from umap import UMAP
except Exception:
    _HAS_UMAP = False

_HAS_HDBSCAN = True
try:
    import hdbscan
except Exception:
    _HAS_HDBSCAN = False

# ======== 参数区（请按需修改） ========
# EPOCH_TO_USE       = 20               # 取第几次迭代保存的权重（你说是 5）
# # run viz_latent_subact.py 里生成的 pkl 文件路径(turtle)
# WEIGHT_PKL_PATH    = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm20_seed2025_epoch%d_results_Day1105.pkl"%EPOCH_TO_USE
# WOCL_WEIGHT_PKL_PATH    = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm0_seed2025_epoch%d_results_Day1105.pkl"%EPOCH_TO_USE
# # ↑ 上面只是示例命名：与你在 run() 里生成的 pkl 文件名一致即可
# #   例如：sensor_type+'_%s_epoch%s_results_Day1105.pkl' （iteration==5）

TURTLE_PKL_PATH    = r"D:\code\DeepView\deepview\calculate_results\data\turtle\turtle.pkl"   # 数据文件路径（与你之前一致）
SENSOR_TYPE        = "AccelGyroDepth"  # 备选: Accel / AccelDepth / GyroDepth / AccelGyro / AccelGyroDepth

DEVICE             = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE         = 4000            # 与原脚本一致的推理 batch
UMAP_N_NEIGHBORS   = 15
UMAP_MIN_DIST      = 0.1
UMAP_METRIC        = "euclidean"     # 可改 "cosine"

# 输出图设置（确保与原脚本不同名，不会覆盖）
OUTPUT_DIR         = r"D:\code\DeepView\deepview\calculate_results\data\turtle\figures"
OUTPUT_PREFIX      = "latent_replot"         # 自定义前缀
# OUTPUT_TAG         = f"{SENSOR_TYPE}_epoch{EPOCH_TO_USE}_v2"  # 自定义tag
# 最终 HTML 路径（确保不与 viz_latent_subact.py 的命名冲突）
# OUTPUT_HTML_PATH   = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}.html")
# OUTPUT_HTML_ORG_PATH   = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}_org.html")
# OUTPUT_HTML_PATH_wocl   = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}_wocl.html")
# OUTPUT_HTML_ORG_PATH_wocl   = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}_org_wocl.html")
#
#
# # 是否同时输出 PNG（需要安装 kaleido）
# SAVE_STATIC_PNG    = False
# OUTPUT_PNG_PATH    = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}.png")
# OUTPUT_PNG_PATH_wocl    = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}_{OUTPUT_TAG}_wocl.png")
PNG_WIDTH, PNG_HEIGHT, PNG_SCALE = 1400, 900, 2

# ========= 依赖于 deepview 的工具方法（与原文件保持一致的导入） ========
from deepview.calculate_results.models.utils import (
    data_loader_umineko,
    sliding_window,
    process_acc,
    process_acc_temperature,
    process_acc_gyr,
    process_acc_gyr_dpt,
    majority_value,
    AE_eval_time_series
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_1s,
    SimpleNN_13s,
    SimpleNN_33s,
    SimpleNN_331s,
)

# 图算法依赖（自动探测）
_HAS_IGRAPH = False
_HAS_LEIDEN = False
_HAS_LOUVAIN = False
# try:
#     import igraph as ig
#     _HAS_IGRAPH = True
#     try:
#         import leidenalg as la
#         _HAS_LEIDEN = True
#     except Exception:
#         _HAS_LEIDEN = False
# except Exception:
#     _HAS_IGRAPH = False
try:
    import networkx as nx
except Exception as e:
    raise RuntimeError("需要 networkx：pip install networkx") from e
# try:
#     import community as community_louvain  # python-louvain
#     _HAS_LOUVAIN = True
# except Exception:
#     _HAS_LOUVAIN = False


# ========= 标签映射 & 颜色（与原脚本一致） =========
label_colors = {
    0: "#5470C6",  # Resting
    1: "#91CC75",  # Swimming
    2: "#FAC858",  # Stay in surface
    3: "#73C0DE",  # Gliding
    4: "#EE6666",  # Feeding
    5: 'brown',    # Scratching
    6: 'pink',     # Breathing
}

label_dict = {
    'Resting': 0,
    'Swimming': 1,
    'Stay in surface': 2,
    'Gliding': 3,
    'Feeding': 4,
    'Scratching': 5,
    'Breathing': 6,
    'Unknown': -1,
}

# 将原 Label 文本合并成 coarse 类别
labelcategory_dict = {
    'Breathing': 'Breathing',
    'Catching': 'Feeding',
    'Chewing on movement': 'Feeding',
    'Chewing stationary': 'Feeding',
    'Escape': 'Unknown',
    'Flipper beat': 'Unknown',
    'Foraging': 'Unknown',
    'Gliding ascent': 'Gliding',
    'Gliding descent': 'Gliding',
    'Grabbing on movement': 'Feeding',
    'Grabbing stationary': 'Feeding',
    'Interaction': 'Unknown',
    'Left U-turn': 'Swimming',
    'New video': 'Unknown',
    'Obstacle': 'Unknown',
    'Prospection': 'Swimming',
    'Pursuit': 'Unknown',
    'Resting': 'Resting',
    'Resting active': 'Resting',
    'Resting in flow': 'Resting',
    'Right U-turn': 'Swimming',
    'Scratching': 'Scratching',
    'Shaking': 'Unknown',
    'Shaking head': 'Unknown',
    'Stay in surface': 'Stay in surface',
    'Swimming 1 ascent': 'Swimming',
    'Swimming 1 descent': 'Swimming',
    'Swimming 1 horizontally': 'Swimming',
    'Swimming ascent': 'Swimming',
    'Swimming descent': 'Swimming',
    'Swimming fast descent': 'Swimming',
    'Swimming fast horizontally': 'Swimming',
    'Swimming horizontally': 'Swimming',
    'Swimming in place': 'Swimming',
    'Swimming on the bottom': 'Swimming',
    'Watching': 'Swimming',
    'Catching jellyfish': 'Feeding',
    'Resting watching': 'Resting',
    'Sand': 'Unknown',
    'Scratching camera': 'Scratching',
    'Swimming fast ascent': 'Swimming',
    'Chewing jellyfish': 'Feeding',
    'Landing': 'Unknown',
    'Scratching head': 'Scratching',
    'Stepping back': 'Swimming',
    'Regurgitating': 'Unknown',
    'Time .': 'Unknown',
    'Grabbing the wall': 'Feeding',
    'Hunting jellyfish': 'Unknown',
    'Unknown': 'Unknown',
}

labeldict_findstr_large = {
    0: 'Resting',
    1: 'Swimming',
    2: 'Stay in surface',
    3: 'Gliding',
    4: 'Feeding',
    5: 'Scratching',
    6: 'Breathing',
}

labeldict_findstr_rst = {
    0: 'Resting',
    1: 'Swimming',
    2: 'Stay in surface',
    3: 'Gliding',
    4: 'Feeding',
    5: 'Scratching',
    6: 'Breathing',
    -1: 'Unknown',
    7: 'Resting active',
    8: 'Resting in flow',
    9: 'Resting watching',
}

# split resting labels
labelcategory_dict_rst = {
    'Breathing': 'Breathing',
    'Catching': 'Feeding',
    'Chewing on movement': 'Feeding',
    'Chewing stationary': 'Feeding',
    'Escape': 'Unknown',
    'Flipper beat': 'Unknown',
    'Foraging': 'Unknown',
    'Gliding ascent': 'Gliding',
    'Gliding descent': 'Gliding',
    'Grabbing on movement': 'Feeding',
    'Grabbing stationary': 'Feeding',
    'Interaction': 'Unknown',
    'Left U-turn': 'Swimming',
    'New video': 'Unknown', #??
    'Obstacle': 'Unknown',
    'Prospection': 'Swimming',
    'Pursuit': 'Unknown',
    'Resting': 'Resting',
    'Resting active': 'Resting active',
    'Resting in flow': 'Resting in flow',
    'Right U-turn': 'Swimming',
    'Scratching': 'Scratching',
    'Shaking': 'Unknown',
    'Shaking head': 'Unknown',
    'Stay in surface': 'Stay in surface',
    'Swimming 1 ascent': 'Swimming',
    'Swimming 1 descent': 'Swimming',
    'Swimming 1 horizontally': 'Swimming',
    'Swimming ascent': 'Swimming',
    'Swimming descent': 'Swimming',
    'Swimming fast descent': 'Swimming',
    'Swimming fast horizontally': 'Swimming',
    'Swimming horizontally': 'Swimming',
    'Swimming in place': 'Swimming',
    'Swimming on the bottom': 'Swimming',
    'Watching': 'Swimming',
    'Catching jellyfish': 'Feeding',
    'Resting watching': 'Resting watching',
    'Sand': 'Unknown',
    'Scratching camera': 'Scratching',
    'Swimming fast ascent': 'Swimming',
    'Chewing jellyfish': 'Feeding',
    'Landing': 'Unknown',
    'Scratching head': 'Scratching',
    'Stepping back': 'Swimming',
    'Regurgitating': 'Unknown',
    'Time .': 'Unknown', #'rest_passive',#??
    'Grabbing the wall': 'Feeding',
    'Hunting jellyfish': 'Unknown',
    'Unknown': 'Unknown',
}

label_dict_rst = {
    'Resting': 0,
    'Swimming': 1,
    'Stay in surface': 2,
    'Gliding': 3,
    'Feeding': 4,
    'Scratching': 5,
    'Breathing': 6,
    'Unknown': -1,
    'Resting active': 7,
    'Resting in flow': 8,
    'Resting watching': 9,
}

# -------------------- Utils --------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.stats import ttest_ind
import itertools

def plot_rest_sub_behavior_violin(
    sample_list,
    pred_list,
    label_list,
    rest_class=0,
    feature_name="Mean |x| over window",
    sub_label_prefix="Sub ",
    out_png=None,
):
    """
    根据 AE_eval_time_series 的输出结果，在 rest 类内部的 4 个 sub-behavior 上
    计算原始时间序列的统计特征，并画出 violin+box+显著性 的图。

    参数
    ----
    sample_list : List[np.ndarray]
        AE_eval_time_series 返回的 sample_list，每个元素形状类似 [B, T, D]。
    pred_list : List[np.ndarray]
        每个窗口的子类 / cluster 预测（例如 HDBSCAN/AE cluster），和 sample_list 对齐。
    label_list : List[np.ndarray]
        coarse label（包含 rest=0 这一类），和 sample_list 对齐。
    rest_class : int, 默认 0
        coarse label 中表示 resting 的类别 ID。
    feature_name : str
        y 轴上显示的特征名字。
    sub_label_prefix : str
        x 轴标签前缀，例如 "Sub " -> "Sub 0, Sub 1, ...".
    out_png : str 或 None
        如果给出路径，就保存成 PNG；否则只显示。

    返回
    ----
    fig, ax : matplotlib Figure 和 Axes 句柄
    """
    # -------- 1. 把 list 拼成平铺数组 --------
    samples = np.concatenate(sample_list, axis=0)   # [N, T, D] 或 [N, D, T]
    coarse_labels_concat = np.concatenate(label_list, axis=0)  # [N]
    coarse_labels = majority_value(coarse_labels_concat)
    sub_preds_concat = np.concatenate(pred_list, axis=0)       # [N]
    sub_preds = majority_value(sub_preds_concat)

    # assert len(samples) == len(coarse_labels) == len(sub_preds), \
    #     "samples / labels / preds 长度不一致，请检查 AE_eval_time_series 输出。"

    # -------- 2. 选出 resting 的所有窗口，并取其 sub-beh 标注 --------
    rest_idx = np.where(coarse_labels == rest_class)[0]
    rest_samples = samples[rest_idx]           # [N_rest, ...]
    rest_sub = sub_preds[rest_idx]            # [N_rest]

    # 假设 rest 内有 4 个子类（0,1,2,3），但这里写成泛化版本
    # sub_ids = np.unique(rest_sub)
    # sub_ids = np.sort(sub_ids)
    sub_ids = np.array([0,7,8,9])

    # -------- 3. 定义“特征”，用原始时间序列算一个标量 --------
    # 这里默认：先算每个时间点的 L2 范数，再在整个窗口取平均。
    def window_feature(x):
        x = np.asarray(x)
        # 把时间维放在前面（如果是 [D, T]，转置也不影响 L2 范数；这里直接按最后一维求）
        if x.ndim == 1:
            return float(np.abs(x).mean())
        mag = np.linalg.norm(x, axis=-1)  # [T]
        return float(mag.mean())

    # 对 rest 中的每一个窗口计算 feature
    all_feats = np.array([window_feature(w) for w in rest_samples])

    # 按 sub-behavior 分组
    groups = []
    for sid in sub_ids:
        vals = all_feats[rest_sub == sid]
        if len(vals) == 0:
            continue
        groups.append(vals)
    n_groups = len(groups)
    if n_groups == 0:
        raise ValueError("在 rest 类中没有有效的 sub-behavior 数据，无法作图。")

    # -------- 4. 绘图：violin + box + significance stars --------
    fig, ax = plt.subplots(figsize=(4.0, 4.8))

    positions = np.arange(1, n_groups + 1)

    # 4.1 Violin
    vparts = ax.violinplot(
        groups,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    # 上色：轻柔一点
    base_colors = ["#c5b3e6", "#d4a6c8", "#f4c9a4", "#a3d9c9", "#f2e2a2"]
    for i, body in enumerate(vparts["bodies"]):
        color = base_colors[i % len(base_colors)]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.8)

    # 4.2 Boxplot（叠在 violin 上）
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=0.18,
        patch_artist=False,
        showfliers=False,
        whis=1.5,
    )
    for element in ["boxes", "whiskers", "caps", "medians"]:
        for line in bp[element]:
            line.set_color("black")
            line.set_linewidth(1.0)

    # -------- 5. 显著性比较（两两 t-test） --------
    ymin = min(g.min() for g in groups)
    ymax = max(g.max() for g in groups)
    h = (ymax - ymin) * 0.04 if ymax > ymin else 0.05
    y = ymax + h * 2

    def p_to_stars(p):
        if p < 1e-4:
            return "****"
        elif p < 1e-3:
            return "***"
        elif p < 1e-2:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"

    for (i, j) in itertools.combinations(range(n_groups), 2):
        g1, g2 = groups[i], groups[j]
        stat, p = ttest_ind(g1, g2, equal_var=False)
        stars = p_to_stars(p)

        x1, x2 = positions[i], positions[j]
        ax.plot([x1, x1, x2, x2],
                [y,  y + h, y + h, y],
                color="black", linewidth=0.8)
        ax.text((x1 + x2) / 2.0, y + h * 1.1, stars,
                ha="center", va="bottom", fontsize=8)
        y += h * 1.7  # 每一对往上错一点，避免重叠

    # -------- 6. 轴 & 标签设置 --------
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{sub_label_prefix}{int(s)}" for s in sub_ids],
                       fontsize=9)
    ax.set_ylabel(feature_name, fontsize=10)

    ax.tick_params(axis="y", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()

    if out_png is not None:
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {out_png}")

    return fig, ax


def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return str(p)

def _scale(X):
    return StandardScaler().fit_transform(X)


def _to_similarity(dist: np.ndarray, metric: str) -> np.ndarray:
    """将距离转成相似度。"""
    if metric == 'cosine':
        sim = 1.0 - dist
        sim[sim < 0] = 0.0
        return sim
    else:
        # 通用：相似度 = 1 / (1 + 距离)
        return 1.0 / (1.0 + dist)

# def build_mutual_knn_graph(X: np.ndarray, k: int, metric: str = 'cosine') -> Tuple[nx.Graph, np.ndarray]:
#     """
#     用 mutual-kNN 构图：i<->j 互为近邻才连边；边权 = 相似度。
#     返回：G (networkx.Graph) 和 邻接矩阵的稀疏三元组 edges(list)
#     """
#     X = np.asarray(X, dtype=float)
#     n = X.shape[0]
#     k = min(max(1, k), max(1, n-1))
#
#     nbrs = NearestNeighbors(n_neighbors=k+1, metric=metric).fit(X)
#     dists, idxs = nbrs.kneighbors(X, return_distance=True)
#
#     # 去掉自环（第一个是自己）
#     dists = dists[:, 1:]
#     idxs  = idxs[:, 1:]
#
#     # i 的近邻清单
#     nbr_list = [set(idxs[i].tolist()) for i in range(n)]
#
#     # mutual 过滤 + 权重（相似度）
#     edges = []
#     for i in range(n):
#         for j_pos, j in enumerate(idxs[i]):
#             if i in nbr_list[j]:  # 互为近邻
#                 # w = _to_similarity(dists[i, j_pos], metric)
#                 w = _to_similarity(dists[i, j_pos], '')  # 只有一个值，不是余弦
#                 if w > 0:
#                     a, b = (i, int(j)) if i < j else (int(j), i)
#                     edges.append((a, b, float(w)))
#
#     # 去重
#     edges = list({(a, b): w for a, b, w in edges}.items())
#     edges = [(a, b, w) for (a, b), w in edges]
#
#     G = nx.Graph()
#     G.add_nodes_from(range(n))
#     for a, b, w in edges:
#         G.add_edge(a, b, weight=w)
#
#     return G, edges
#
# def modularity_and_communities(G: nx.Graph, resolutions: List[float]) -> Tuple[float, List[int], float]:
#     """
#     优先用 Leiden（ig/LA），其次 Louvain（python-louvain），否则 greedy modularity。
#     返回：最佳模块度Q、对应社群标签list、对应的分辨率值
#     """
#     # 转 igraph
#     if _HAS_IGRAPH and (G.number_of_edges() > 0):
#         # 建 igraph
#         mapping = {node: i for i, node in enumerate(G.nodes())}
#         inv_map = {i: node for node, i in mapping.items()}
#         edges = [(mapping[u], mapping[v]) for u, v in G.edges()]
#         weights = [G[u][v].get('weight', 1.0) for u, v in G.edges()]
#
#         g = ig.Graph()
#         g.add_vertices(G.number_of_nodes())
#         if edges:
#             g.add_edges(edges)
#             g.es['weight'] = weights
#
#         best_Q, best_labels, best_res = -1.0, None, None
#         for res in resolutions:
#             if _HAS_LEIDEN:
#                 part = la.find_partition(
#                     g, la.RBConfigurationVertexPartition,
#                     weights='weight',
#                     resolution_parameter=float(res)
#                 )
#                 Q = part.quality()
#                 labels = np.array(part.membership, dtype=int).tolist()
#             else:
#                 # 使用 igraph 的 louvain 等价
#                 part = g.community_multilevel(weights='weight')  # 无分辨率参数
#                 Q = part.modularity
#                 labels = np.zeros(g.vcount(), dtype=int)
#                 for cid, comm in enumerate(part):
#                     labels[list(comm)] = cid
#                 res = float('nan')
#
#             if Q > best_Q:
#                 best_Q, best_labels, best_res = Q, labels, res
#
#         # 还原到原节点次序
#         if best_labels is not None:
#             lbl_map = {inv_map[i]: lab for i, lab in enumerate(best_labels)}
#             labels_out = [lbl_map[n] for n in G.nodes()]
#             return float(best_Q), labels_out, float(best_res if best_res is not None else float('nan'))
#
#     # 尝试 python-louvain
#     if _HAS_LOUVAIN and (G.number_of_edges() > 0):
#         part = community_louvain.best_partition(G, weight='weight', random_state=0)
#         labels_out = [part[n] for n in G.nodes()]
#         Q = community_louvain.modularity(part, G, weight='weight')
#         return float(Q), labels_out, float('nan')
#
#     # 最后回退：greedy modularity communities（无权/无分辨率）
#     if G.number_of_edges() == 0:
#         # 完全无边图
#         labels_out = list(range(G.number_of_nodes()))
#         return 0.0, labels_out, float('nan')
#
#     comms = list(nx.algorithms.community.greedy_modularity_communities(G))
#     labels_out = np.zeros(G.number_of_nodes(), dtype=int)
#     for cid, comm in enumerate(comms):
#         for node in comm:
#             labels_out[node] = cid
#     # 估模块度
#     Q = nx.algorithms.community.quality.modularity(G, comms, weight='weight')
#     return float(Q), labels_out.tolist(), float('nan')
#
# def edge_homophily_cross_ratio(G: nx.Graph, y_true: np.ndarray) -> Tuple[float, float]:
#     """同类同配率 & 跨簇边占比。权重加权。"""
#     if G.number_of_edges() == 0:
#         return 0.0, 0.0
#     same_w, total_w = 0.0, 0.0
#     for u, v, d in G.edges(data=True):
#         w = d.get('weight', 1.0)
#         total_w += w
#         if y_true[u] == y_true[v]:
#             same_w += w
#     hom = same_w / total_w if total_w > 0 else 0.0
#     cross = 1.0 - hom
#     return float(hom), float(cross)
#
# def purity_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
#     """宏平均 Purity（社群→真标签）。"""
#     df = pd.DataFrame({"t": true_labels, "p": pred_labels})
#     purity_sum = 0
#     for c, sub in df.groupby("p"):
#         cnts = sub["t"].value_counts()
#         purity_sum += cnts.max()
#     return float(purity_sum / len(df)) if len(df) > 0 else 0.0
#
# def conductance_by_class(G: nx.Graph, y_true: np.ndarray) -> Dict[int, float]:
#     """
#     对每个“真类”计算电导：phi(S) = cut(S,~S)/min(vol(S),vol(~S))，权重图。
#     返回：{label: conductance}
#     """
#     if G.number_of_edges() == 0:
#         # 无边时电导为 1（完全不连通），也可返回 NaN
#         return {int(c): 1.0 for c in np.unique(y_true)}
#
#     deg = dict(G.degree(weight='weight'))
#     vol = lambda S: sum(deg[i] for i in S)
#
#     # 预先把边权做成查表
#     w_lookup = defaultdict(float)
#     for u, v, d in G.edges(data=True):
#         w = d.get('weight', 1.0)
#         a, b = (u, v) if u < v else (v, u)
#         w_lookup[(a, b)] += w
#
#     labels = np.asarray(y_true)
#     out = {}
#     for c in np.unique(labels):
#         S = np.where(labels == c)[0].tolist()
#         Sset = set(S)
#         if len(S) == 0 or len(S) == len(labels):
#             out[int(c)] = float('nan')
#             continue
#
#         # cut(S,~S)
#         cut_w = 0.0
#         for u in S:
#             for v in G.neighbors(u):
#                 if v not in Sset:
#                     a, b = (u, v) if u < v else (v, u)
#                     cut_w += w_lookup[(a, b)]
#         volS = vol(S)
#         volT = vol(set(G.nodes()) - Sset)
#         denom = min(volS, volT)
#         phi = cut_w / denom if denom > 0 else np.nan
#         out[int(c)] = float(phi)
#     return out

def _umap2d(X, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42):
    if _HAS_UMAP:
        um = UMAP(
            n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
            metric=metric, random_state=random_state, verbose=False
        )
        return um.fit_transform(X)
    else:
        pca = PCA(n_components=2, random_state=random_state)
        return pca.fit_transform(X)

def _hopkins(X, m=0.1, random_state=42):
    """Hopkins∈[0,1]，越接近1越有聚类倾向。"""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    n, d = X.shape
    if m < 1:
        m = max(10, int(m * n))
    m = min(m, n - 1)

    xmin, xmax = X.min(axis=0), X.max(axis=0)
    U = rng.random((m, d)) * (xmax - xmin) + xmin

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    u_dist, _ = nbrs.kneighbors(U, n_neighbors=1)
    idx = rng.choice(n, size=m, replace=False)
    w_dist_all, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
    w_dist = w_dist_all[:, 1]

    return float(u_dist.sum() / (u_dist.sum() + w_dist.sum() + 1e-12))

def _gmm_bic_best_k(X, k_min=2, k_max=20, random_state=42, n_init=2):
    bics = {}
    best_k, best_bic, best_model = None, np.inf, None
    for k in range(k_min, k_max + 1):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=random_state, n_init=n_init)
        gm.fit(X)
        bic = gm.bic(X)
        bics[k] = bic
        if bic < best_bic:
            best_bic, best_k, best_model = bic, k, gm
    return best_k, bics, best_model

def _cluster_full(
    X,
    prefer_hdbscan: bool = True,
    random_state: int = 42,
    # —— 新增：可选的 HDBSCAN 参数（若为 None，则用下面两个“分数式”默认值推导）——
    hdbscan_kwargs = None,
    min_cluster_size_frac: float = 0.01,   # 默认 mcs = max(5, int(frac * N))
    min_samples_mode: str = "same_as_mcs", # "same_as_mcs" 或 None
    # —— 新增：回退与 GMM-BIC 搜索范围控制 ——
    fallback_when_all_noise: bool = True,
    gmm_k_min: int = 2,
    gmm_k_max = None,
    kmeans_n_init: int = 10,
):
    """
    返回: pred_labels, algo_name, extra
      - pred_labels: np.ndarray[int]，HDBSCAN 的噪声为 -1
      - algo_name:   str
      - extra: dict，常见键：
          * auto_k:     估计簇数（对 HDBSCAN 为“非 -1 唯一簇数”）
          * noise_ratio:噪声占比（HDBSCAN）
          * bic_curve:  GMM-BIC 搜索得到的 BIC 曲线（KMeans 分支）
          * algo_params:本次 HDBSCAN 的实际参数（若使用）
          * probabilities: HDBSCAN 的 membership（若可用）

    行为要点：
      1) 若 prefer_hdbscan 且环境支持，则优先用 HDBSCAN；
         * k_est = 非 -1 唯一簇数（与上层 NMI/ARI 的“仅评估非噪声样本”口径一致）
         * 若全部为噪声且允许回退，则自动回退到 KMeans@GMM-BIC
      2) 否则使用 KMeans，其中 best_k 通过 GMM-BIC 选取。
    """
    extra = {}
    N = len(X)
    if N == 0:
        return np.array([], dtype=int), "NA(empty)", {"auto_k": 0, "noise_ratio": 1.0}

    # ---------------------------
    # 优先：HDBSCAN（若可用）
    # ---------------------------
    if prefer_hdbscan and '_HAS_HDBSCAN' in globals() and _HAS_HDBSCAN:
        # 组装/推导 HDBSCAN 参数
        if hdbscan_kwargs is None:
            mcs = max(5, int(min_cluster_size_frac * N))
            ms = (mcs if min_samples_mode == "same_as_mcs" else None)
            hdbscan_kwargs = dict(min_cluster_size=mcs, min_samples=ms)
        else:
            # 确保不会传入 None 的键值
            hdbscan_kwargs = {k: v for k, v in hdbscan_kwargs.items() if v is not None}

        clusterer = hdbscan.HDBSCAN(**hdbscan_kwargs)
        pred = clusterer.fit_predict(X)  # -1 = noise
        uniq = np.unique(pred[pred != -1])
        k_est = int(len(uniq))

        extra.update({
            "noise_ratio": float(np.mean(pred == -1)),
            "auto_k": k_est,
            "algo_params": {"hdbscan": dict(hdbscan_kwargs)},
        })
        if hasattr(clusterer, "probabilities_"):
            extra["probabilities"] = clusterer.probabilities_

        algo = f"HDBSCAN(min_cluster_size={hdbscan_kwargs.get('min_cluster_size')}, " \
               f"min_samples={hdbscan_kwargs.get('min_samples')})"

        # 若 HDBSCAN 全部判为噪声，按需回退
        if not (fallback_when_all_noise and k_est == 0):
            return pred, algo, extra
        # 否则继续回退到 KMeans（下面执行）

    # ---------------------------
    # 回退：KMeans(best_k via GMM-BIC)
    # ---------------------------
    k_max = gmm_k_max if gmm_k_max is not None else min(20, max(6, int(math.sqrt(max(N, 1)))))
    best_k, bics, _ = _gmm_bic_best_k(X, k_min=gmm_k_min, k_max=k_max, random_state=random_state)

    km = KMeans(n_clusters=int(best_k), random_state=random_state, n_init=kmeans_n_init)
    pred = km.fit_predict(X)

    algo = "KMeans(best_k via GMM-BIC)"
    extra.update({
        "noise_ratio": 0.0,
        "auto_k": int(best_k),
        "bic_curve": bics,
    })
    return pred, algo, extra

def _internal_cluster_metrics(X, y_pred):
    """内部聚类指标（去掉噪声=-1后计算）。"""
    mask = (y_pred != -1)
    Xv, yv = X[mask], y_pred[mask]
    res = {"silhouette": np.nan, "db": np.nan, "ch": np.nan}
    if len(Xv) > 20 and len(np.unique(yv)) >= 2:
        res["silhouette"] = float(silhouette_score(Xv, yv))
        res["db"] = float(davies_bouldin_score(Xv, yv))
        res["ch"] = float(calinski_harabasz_score(Xv, yv))
    return res

def _graph_metrics_knn(X, k=10):
    """构建kNN图，返回模块度和谱间隙λ2。"""
    k = min(k, max(2, len(X)-1))
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    dist, idx = nn.kneighbors(X)
    G = nx.Graph()
    G.add_nodes_from(range(len(X)))
    for i in range(len(X)):
        for j in idx[i, 1:]:
            if i != j:
                G.add_edge(int(i), int(j))

    try:
        comms = nx.algorithms.community.greedy_modularity_communities(G)
        modularity = nx.algorithms.community.modularity(G, comms)
    except Exception:
        modularity = np.nan

    try:
        L = nx.normalized_laplacian_matrix(G).A
        eigvals = np.linalg.eigvalsh(L)
        eigvals.sort()
        spectral_gap = float(eigvals[1]) if len(eigvals) >= 2 else np.nan
    except Exception:
        spectral_gap = np.nan

    return modularity, spectral_gap

def _make_true_labels_9class(labels_org_flat, resting_sub_labels, resting_class=0):
    """
    构造“9类真值标签”：resting(0)的4个子类 → {0,1,2,3}；
    其它四个粗类1..4 → {4,5,6,7,8}
    """
    y_coarse = np.asarray(labels_org_flat).astype(int)
    mask_rest = (y_coarse == resting_class)
    sub = np.asarray(resting_sub_labels).astype(int)
    assert sub.shape[0] == mask_rest.sum(), "resting_sub_labels长度必须等于resting样本数"

    y9 = np.empty_like(y_coarse)
    # 填其它粗类
    y9[~mask_rest] = 4 + (y_coarse[~mask_rest] - 1)
    # 填rest子类
    y9[mask_rest] = sub
    return y9

def _unknown_f1_from_clusters(y_pred, true_unknown_mask, tau=0.6, noise_as_unknown=False):
    y_pred=np.asarray(y_pred); tu=np.asarray(true_unknown_mask,bool)
    pred_unknown=np.zeros_like(y_pred,bool)
    for lb in np.unique(y_pred):
        idx=(y_pred==lb)
        if not idx.any(): continue
        pu=tu[idx].mean()
        if pu>=tau: pred_unknown[idx]=True
    prec, rec, f1, _ = precision_recall_fscore_support(tu, pred_unknown, average="binary", pos_label=True, zero_division=0)
    return float(prec), float(rec), float(f1)
# def _unknown_f1_from_clusters(y_pred, y_true_unknown_mask):
#     """
#     用“簇多数表决是否属于unknown(=resting)”来得到簇级别的unknown预测，
#     然后对样本级别计算 F1(unknown)。
#     """
#     y_pred = np.asarray(y_pred)
#     is_noise = (y_pred == -1)
#     # 噪声样本：归为“unknown预测”（合理选择）
#     pred_unknown = is_noise.copy()
#
#     # 对每个非噪声簇，多数表决
#     for c in np.unique(y_pred[y_pred >= 0]):
#         idx = np.where(y_pred == c)[0]
#         majority_unknown = (y_true_unknown_mask[idx].mean() >= 0.5)
#         pred_unknown[idx] = majority_unknown
#
#     y_true = y_true_unknown_mask.astype(int)
#     y_hat = pred_unknown.astype(int)
#     prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_hat, average="binary", zero_division=0)
#     return float(prec), float(rec), float(f1)

def _line_probe_macro_f1(X, y_sub, labels_per_class_list=(1,3,5,10,20), n_repeats=3, random_state=42):
    """
    线性探针：在resting四子类上做少样本训练，报告不同标注预算下的Macro-F1（平均n_repeats次）。
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X); y = np.asarray(y_sub).astype(int)
    classes = np.unique(y)
    scores = []

    for L in labels_per_class_list:
        reps = []
        for rep in range(n_repeats):
            train_idx = []
            test_idx = np.arange(len(y))
            # 每个子类抽L个做训练
            for c in classes:
                idx_c = np.where(y == c)[0]
                if len(idx_c) < L + 2:
                    # 类太小，跳过
                    continue
                choose = rng.choice(idx_c, size=L, replace=False)
                train_idx.extend(choose.tolist())
            train_idx = np.array(sorted(train_idx))
            mask = np.ones(len(y), dtype=bool)
            mask[train_idx] = False
            test_idx = np.where(mask)[0]

            if len(np.unique(y[train_idx])) < len(classes) or len(train_idx) < len(classes):
                continue

            clf = LogisticRegression(max_iter=200, multi_class="multinomial", n_jobs=None)
            clf.fit(X[train_idx], y[train_idx])
            yhat = clf.predict(X[test_idx])
            _, _, f1, _ = precision_recall_fscore_support(y[test_idx], yhat, average="macro", zero_division=0)
            reps.append(f1)
        scores.append(np.mean(reps) if len(reps) else np.nan)

    return np.array(labels_per_class_list), np.array(scores)


# -------------------- Panels --------------------
def panel_A_metrics(X_all, y_coarse, resting_idx, sub_rest, out_png, out_csv, random_state=42):
    """
    2.1：只在resting子集上做无监督聚类，计算聚类/可聚类性指标。
    输出：条形图（CE-only vs SupCon），以及csv。
    """
    def compute_one(X):
        # mask_rest = (y_coarse == resting_class)
        Xr = _scale(X)[resting_idx]
        # 聚类
        y_pred, algo, extra = _cluster_full(Xr, prefer_hdbscan=True, random_state=42)
        intern = _internal_cluster_metrics(Xr, y_pred)
        hop = _hopkins(Xr, m=0.1, random_state=42)

        # 与真子类的可比指标（如果提供）
        nmi = ari = np.nan
        if sub_rest is not None and len(sub_rest) == len(Xr):
            mask_eval = (y_pred != -1)
            yr_true = np.asarray(sub_rest)[mask_eval]
            yr_pred = y_pred[mask_eval]

            # # test plot
            # Xr_2d = _umap2d(Xr, random_state=42)
            # # ⭐ 关键：子集直接从 Xr_2d 里切，不用再跑 UMAP
            # Xr_2d_new = Xr_2d[mask_eval]
            #
            # plt.figure(figsize=(8, 4), dpi=150)
            # plt.scatter(Xr_2d[:, 0], Xr_2d[:, 1], c='lightgray', s=1, rasterized=True)
            # plt.show()
            #
            # plt.figure(figsize=(8, 4), dpi=150)
            # plt.scatter(Xr_2d_new[:, 0], Xr_2d_new[:, 1], c=yr_true, s=5, cmap='tab10', rasterized=True)
            # plt.show()

            if len(np.unique(yr_true)) >= 2 and len(np.unique(yr_pred)) >= 2:
                nmi = normalized_mutual_info_score(yr_true, yr_pred)
                ari = adjusted_rand_score(yr_true, yr_pred)

        result = {
            "algo": extra.get("auto_k", None),
            "auto_k": extra.get("auto_k", None),
            "noise_ratio": extra.get("noise_ratio", 0.0),
            "silhouette": intern["silhouette"],
            "db": intern["db"],
            "ch": intern["ch"],
            "hopkins": hop,
            "nmi_true": nmi,
            "ari_true": ari
        }
        return result

    # res_sup = compute_one(X_all)
    # res_ce  = compute_one(X_all["__CE__"]) if isinstance(X_all, dict) else None  # guard
    # 这里 X_all 传入的是 dict: {"SupCon": repr_flat, "__CE__": repr_flat_wo}
    sup = compute_one(X_all["SupCon"])
    ce  = compute_one(X_all["CEonly"])

    # 绘图（尺度化为正向指标），DB取 1/DB
    metrics = ["nmi_true", "ari_true", "silhouette", "1/DB", "CH (scaled)", "hopkins"]
    sup_vals = [
        _safe(sup["nmi_true"]), _safe(sup["ari_true"]),
        _safe(sup["silhouette"]),
        _inv_db(sup["db"]),
        _scale_01_pos(sup["ch"]),
        _safe(sup["hopkins"])
    ]
    ce_vals = [
        _safe(ce["nmi_true"]), _safe(ce["ari_true"]),
        _safe(ce["silhouette"]),
        _inv_db(ce["db"]),
        _scale_01_pos(ce["ch"]),
        _safe(ce["hopkins"])
    ]
    df = pd.DataFrame({"Metric": metrics, "CE-only": ce_vals, "CE+SupCon": sup_vals})
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4.8), dpi=150)
    x = np.arange(len(metrics))
    w = 0.35
    plt.bar(x - w/2, df["CE-only"], width=w, label="CE-only")
    plt.bar(x + w/2, df["CE+SupCon"], width=w, label="CE+SupCon")
    plt.xticks(x, metrics, rotation=15, ha="right")
    plt.ylabel("Score (normalized)")
    plt.title("A — A-class clustering quality & auto-K proxies")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"sup": sup, "ce": ce, "csv": out_csv, "png": out_png}

def panel_B_gcd_like_old(X_sup, X_ce, y_coarse, resting_sub, resting_class, out_png, out_csv):
    """
    2.2：GCD式端到端（简化实现）：
    - 在全体样本 embedding 上无监督聚类（HDBSCAN优先，否则KMeans@k=best by BIC）
    - 与“9类真值”比较：NMI/ARI
    - “未知类F1”：把预测簇按多数表决映射为 unknown(=resting) 或 known，其F1
    - “类数估计误差Scaled”：k_est=自动估计簇数（或BIC最佳k），与9的偏差缩放后取 1-err
    """
    y9 = _make_true_labels_9class(y_coarse, resting_sub, resting_class=resting_class)
    true_unknown_mask = (y_coarse == resting_class)

    def compute_one(X):
        Xs = _scale(X)
        y_pred, algo, extra = _cluster_full(Xs, prefer_hdbscan=True, random_state=42)
        # 设定估计簇数
        k_est = extra.get("auto_k", None)
        if k_est is None:
            # 兜底
            k_est, _, _ = _gmm_bic_best_k(Xs, k_min=2, k_max=20, random_state=42)
        # 9类 NMI/ARI
        nmi = normalized_mutual_info_score(y9[y_pred!=-1], y9[y_pred!=-1])  # placeholder逻辑（保证API统一）
        # 正确：与 y_pred（去噪）的一致性
        y9_eval = y9[y_pred!=-1]
        pred_eval = y_pred[y_pred!=-1]
        nmi = normalized_mutual_info_score(y9_eval, pred_eval)
        ari = adjusted_rand_score(y9_eval, pred_eval)

        # 未知F1
        prec_u, rec_u, f1_u = _unknown_f1_from_clusters(y_pred, true_unknown_mask)

        # k误差Scaled（越大越好）
        err_scaled = 1.0 - min(1.0, abs(int(k_est) - 9) / 9.0)

        return dict(
            nmi=nmi, ari=ari, unk_prec=prec_u, unk_rec=rec_u, unk_f1=f1_u,
            k_est=int(k_est), k_err_scaled=err_scaled
        )

    sup = compute_one(X_sup)
    ce  = compute_one(X_ce)

    df = pd.DataFrame({
        "Metric": ["NMI (9-class)", "ARI (9-class)", "Unknown F1", "|k_est-9|↓ (scaled)"],
        "CE-only": [ce["nmi"], ce["ari"], ce["unk_f1"], ce["k_err_scaled"]],
        "CE+SupCon": [sup["nmi"], sup["ari"], sup["unk_f1"], sup["k_err_scaled"]],
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4.8), dpi=150)
    x = np.arange(len(df))
    w = 0.35
    plt.bar(x - w/2, df["CE-only"], width=w, label="CE-only")
    plt.bar(x + w/2, df["CE+SupCon"], width=w, label="CE+SupCon")
    plt.xticks(x, df["Metric"], rotation=15, ha="right")
    plt.ylabel("Score (normalized)")
    plt.title("B — GCD-style discovery on mixed unlabeled set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"sup": sup, "ce": ce, "csv": out_csv, "png": out_png}


def panel_B_gcd_like(
    X_sup, X_ce, y_coarse, resting_sub, resting_class, out_png, out_csv,
    # --- 新增可调参数 ---
    k_true=9,
    prefer_hdbscan=True,
    min_cluster_size_frac=0.01,   # HDBSCAN: 簇的最小样本占比
    min_samples_mode="same_as_mcs",  # "same_as_mcs" 或 None
    unknown_tau=0.6,              # 判簇为未知的阈值（该簇内 true_unknown 占比）
    noise_as_unknown=True,        # 噪声点(-1)是否视为预测未知
    k_metric="error_scaled_down"  # "error_scaled_down" 或 "accuracy_scaled_up"
):
    """
    2.2：GCD式端到端（修订实现）
    - 在全体样本 embedding 上无监督聚类（HDBSCAN优先，否则KMeans@k=best by BIC）
    - 与“9类真值”比较：NMI/ARI（仅在被聚类到的样本上评估，与 k_est 口径一致）
    - “未知类F1”：簇内 true_unknown 占比 >= tau 判该簇为 Unknown；噪声(-1)按 noise_as_unknown 处理
    - “k类数估计”：HDBSCAN 用“非 -1 的唯一簇数”；否则用 BIC 选优的 k
    - k 误差指标：|k_est - k_true|/k_true（越低越好）或其互补（越高越好）
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, precision_recall_fscore_support

    # 你已有的工具函数
    # _make_true_labels_9class, _scale, _cluster_full, _gmm_bic_best_k

    y9 = _make_true_labels_9class(y_coarse, resting_sub, resting_class=resting_class)
    true_unknown_mask = (y_coarse == resting_class)

    def _unknown_f1_from_clusters_consistent(y_pred, true_unknown_mask, tau=0.6, noise_as_unknown=True):
        """统一的 Unknown F1 评测：以 '预测 Unknown' 为正类。"""
        y_pred = np.asarray(y_pred)
        tu = np.asarray(true_unknown_mask, dtype=bool)

        # 建簇 -> 是否判未知
        pred_unknown = np.zeros_like(y_pred, dtype=bool)
        labels = np.unique(y_pred)

        for lb in labels:
            if lb == -1:
                if noise_as_unknown:
                    pred_unknown[y_pred == -1] = True
                # 否则保持 False（丢弃噪声不在此处做，保持口径 A）
                continue
            idx = (y_pred == lb)
            if idx.sum() == 0:
                continue
            pu = tu[idx].mean()  # 该簇的 true_unknown 占比
            if pu >= tau:
                pred_unknown[idx] = True

        # 计算 F1（正类=Unknown）
        prec, rec, f1, _ = precision_recall_fscore_support(
            tu, pred_unknown, average="binary", pos_label=True, zero_division=0
        )
        return prec, rec, f1

    def compute_one(X):
        Xs = _scale(X)
        N = Xs.shape[0]

        # 聚类参数（HDBSCAN）
        mcs = max(5, int(min_cluster_size_frac * N))  # min_cluster_size
        ms = (mcs if min_samples_mode == "same_as_mcs" else None)

        y_pred, algo, extra = _cluster_full(
            Xs,
            prefer_hdbscan=prefer_hdbscan,
            random_state=42,
            # 确保两边口径一致的超参（如果 _cluster_full 支持）
            hdbscan_kwargs=dict(min_cluster_size=mcs, min_samples=ms) if prefer_hdbscan else None
        )

        # --- 估计簇数 ---
        if extra.get("auto_k", None) is not None:
            k_est = int(extra["auto_k"])
        else:
            if algo == "hdbscan":
                # 非 -1 唯一簇数
                uniq = np.unique(y_pred[y_pred != -1])
                k_est = int(len(uniq))
            else:
                # 与你原逻辑一致：用 GMM+BIC 搜索
                k_est, _, _ = _gmm_bic_best_k(Xs, k_min=2, k_max=20, random_state=42)

        # --- NMI/ARI（仅在被聚类到的样本上评估，与 k_est 的口径一致）---
        keep = (y_pred != -1)
        if keep.sum() == 0:
            nmi = 0.0
            ari = 0.0
        else:
            y9_eval = y9[keep]
            pred_eval = y_pred[keep]
            nmi = normalized_mutual_info_score(y9_eval, pred_eval)
            ari = adjusted_rand_score(y9_eval, pred_eval)

        # --- Unknown F1（显式约定口径）---
        unk_prec, unk_rec, unk_f1 = _unknown_f1_from_clusters_consistent(
            y_pred, true_unknown_mask, tau=unknown_tau, noise_as_unknown=noise_as_unknown
        )

        # --- k误差指标 ---
        err = min(1.0, abs(int(k_est) - int(k_true)) / float(k_true))
        if k_metric == "error_scaled_down":
            k_value = err                   # 越低越好
            k_label = "|k_est-{}|↓ (scaled)".format(int(k_true))
        else:
            k_value = 1.0 - err             # 越高越好
            k_label = "k-accuracy↑ (scaled vs {})".format(int(k_true))

        return dict(
            nmi=nmi, ari=ari,
            unk_prec=unk_prec, unk_rec=unk_rec, unk_f1=unk_f1,
            k_est=int(k_est), k_metric_value=k_value, k_label=k_label
        )

    sup = compute_one(X_sup)
    ce  = compute_one(X_ce)

    # 统一列名（随所选口径自动变化）
    k_col = sup["k_label"]  # 与上面选择一致

    df = pd.DataFrame({
        "Metric": ["NMI (9-class)", "ARI (9-class)", "Unknown F1", k_col],
        "CE-only": [ce["nmi"], ce["ari"], ce["unk_f1"], ce["k_metric_value"]],
        "CE+SupCon": [sup["nmi"], sup["ari"], sup["unk_f1"], sup["k_metric_value"]],
    })
    df.to_csv(out_csv, index=False)

    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.2, 4.6), dpi=150)
    x = np.arange(len(df))
    w = 0.36
    plt.bar(x - w/2, df["CE-only"], width=w, label="CE-only")
    plt.bar(x + w/2, df["CE+SupCon"], width=w, label="CE+SupCon")
    plt.xticks(x, df["Metric"], rotation=18, ha="right")
    plt.ylabel("Score")
    plt.title("B — GCD-style discovery on mixed unlabeled set")
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"sup": sup, "ce": ce, "csv": out_csv, "png": out_png}

def panel_C_linear_probe(X_sup, X_ce, y_coarse, resting_sub, resting_class, out_png, out_csv):
    """
    2.3：线性探针样本效率（仅在resting 4子类上）
    """
    mask = (y_coarse == resting_class)
    Xs_sup = _scale(X_sup)[mask]
    Xs_ce  = _scale(X_ce)[mask]
    y_sub  = np.asarray(resting_sub)

    labels_per = (1, 3, 5, 10, 20)
    Ls_sup, F1_sup = _line_probe_macro_f1(Xs_sup, y_sub, labels_per_class_list=labels_per, n_repeats=3)
    Ls_ce,  F1_ce  = _line_probe_macro_f1(Xs_ce,  y_sub, labels_per_class_list=labels_per, n_repeats=3)

    df = pd.DataFrame({
        "labels_per_subclass": Ls_sup,
        "CE-only_MacroF1": F1_ce,
        "CE+SupCon_MacroF1": F1_sup
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4.8), dpi=150)
    plt.plot(Ls_ce, F1_ce, marker="o", label="CE-only")
    plt.plot(Ls_sup, F1_sup, marker="o", label="CE+SupCon")
    plt.xlabel("Labels per sub-class")
    plt.ylabel("Macro-F1 (linear probe)")
    plt.title("C — Linear probe sample efficiency on 4 sub-classes")
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"csv": out_csv, "png": out_png}

def panel_D_graph(X_sup, X_ce, y_coarse, resting_class, out_png, out_csv):
    """
    2.4：图结构（resting子集的 kNN 图模块度与谱间隙）
    """
    mask = (y_coarse == resting_class)
    Xr_sup = _scale(X_sup)[mask]
    Xr_ce  = _scale(X_ce)[mask]

    mod_sup, gap_sup = _graph_metrics_knn(Xr_sup, k=min(10, max(5, int(0.01 * len(Xr_sup)))))
    mod_ce,  gap_ce  = _graph_metrics_knn(Xr_ce,  k=min(10, max(5, int(0.01 * len(Xr_ce)))))

    df = pd.DataFrame({
        "Metric": ["Modularity", "Spectral gap"],
        "CE-only": [mod_ce, gap_ce],
        "CE+SupCon": [mod_sup, gap_sup]
    })
    df.to_csv(out_csv, index=False)

    plt.figure(figsize=(8, 4.8), dpi=150)
    x = np.arange(2); w = 0.35
    plt.bar(x - w/2, df["CE-only"], width=w, label="CE-only")
    plt.bar(x + w/2, df["CE+SupCon"], width=w, label="CE+SupCon")
    plt.xticks(x, df["Metric"])
    plt.ylabel("Score")
    plt.title("D — Graph structure: modularity & spectral gap (A-class)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    return {"csv": out_csv, "png": out_png}


@dataclass
class PanelDConfig:
    k_list: List[int]
    sigma_list: List[float]
    resolutions: List[float]
    metric: str = 'cosine'          # 'cosine' or 'euclidean'
    focus_class: Optional[int] = None  # 仅评估某一类（如 resting）；None=全体
    out_prefix: str = 'panel_D'
    dpi: int = 160


# def _eval_one_setting(X: np.ndarray, y: np.ndarray, k: int, sigma: float,
#                       resolutions: List[float], metric: str) -> Dict[str, float]:
#     # 加噪声（在已标准化空间）
#     Xn = X.copy()
#     if sigma > 0:
#         Xn = Xn + np.random.normal(0, sigma, size=Xn.shape)
#
#     G, _ = build_mutual_knn_graph(Xn, k=k, metric=metric)
#     hom, cross = edge_homophily_cross_ratio(G, y)
#     Q, comm_labels, used_res = modularity_and_communities(G, resolutions=resolutions)
#
#     # 社群-标签一致性
#     nmi = normalized_mutual_info_score(y, comm_labels) if len(set(comm_labels)) > 1 else 0.0
#     ari = adjusted_rand_score(y, comm_labels) if len(set(comm_labels)) > 1 else 0.0
#     pur = purity_score(y, np.array(comm_labels))
#
#     # 电导（按真类）
#     cond_map = conductance_by_class(G, y)
#     cond_median = np.nanmedian(list(cond_map.values())) if len(cond_map) else np.nan
#
#     return {
#         "homophily": hom,
#         "cross_edge_ratio": cross,
#         "modularity_Q": Q,
#         "community_resolution": used_res,
#         "NMI": nmi,
#         "ARI": ari,
#         "purity": pur,
#         "conductance_median": float(cond_median),
#         "num_nodes": G.number_of_nodes(),
#         "num_edges": G.number_of_edges(),
#     }

def _pack_rows(tag: str, k: int, sigma: float, metrics: Dict[str, float]) -> Dict[str, float]:
    row = {"method": tag, "k": int(k), "sigma": float(sigma)}
    row.update(metrics)
    return row


# def panel_D_graph_v2(
#     X_sup: np.ndarray,
#     X_ce: np.ndarray,
#     y_coarse: np.ndarray,
#     focus_class: Optional[int] = None,
#     k_list: List[int] = (5,10,15),
#     sigma_list: List[float] = (0.0, 0.02),
#     resolutions: List[float] = (1.0,),
#     metric: str = 'cosine',
#     out_prefix: str = 'panel_D',
#     dpi: int = 160
# ) -> Dict[str, str]:
#     """
#     目标版 Panel D：
#     - mutual-kNN + Leiden（自动回退）
#     - 指标：homophily / cross_edge_ratio / modularity / NMI / ARI / purity / conductance
#     - 对 k 和 噪声 sigma 扫描，输出 CSV 和多张曲线图
#     """
#     cfg = PanelDConfig(
#         k_list=list(k_list),
#         sigma_list=list(sigma_list),
#         resolutions=list(resolutions),
#         metric=metric,
#         focus_class=focus_class,
#         out_prefix=out_prefix,
#         dpi=dpi,
#     )
#
#     # 选择子集（如 resting）
#     if cfg.focus_class is not None:
#         mask = (y_coarse == cfg.focus_class)
#     else:
#         mask = np.ones_like(y_coarse, dtype=bool)
#
#     X_sup_s = _scale(np.asarray(X_sup)[mask])
#     X_ce_s  = _scale(np.asarray(X_ce)[mask])
#     y_s     = np.asarray(y_coarse)[mask]
#
#     rows = []
#     rng = np.random.default_rng(1234)
#
#     for tag, X in [("CE-only", X_ce_s), ("CE+SupCon", X_sup_s)]:
#         for sigma in cfg.sigma_list:
#             for k in cfg.k_list:
#                 # 为了鲁棒性，可重复多次取均值（这里给1次；如需多次把 rep>1）
#                 metrics = _eval_one_setting(X, y_s, k, sigma, cfg.resolutions, cfg.metric)
#                 rows.append(_pack_rows(tag, k, sigma, metrics))
#
#     df = pd.DataFrame(rows)
#     csv_path = f"{cfg.out_prefix}_metrics.csv"
#     df.to_csv(csv_path, index=False)
#
#     # === 画曲线：随 k 的三类核心指标（对每个 sigma 分面）===
#     def _plot_metric_vs_k(metric_name: str, ylabel: str):
#         for sigma in cfg.sigma_list:
#             plt.figure()
#             sub = df[df['sigma'] == sigma]
#             for tag in ["CE-only", "CE+SupCon"]:
#                 dd = sub[sub['method'] == tag].sort_values('k')
#                 plt.plot(dd['k'], dd[metric_name], marker='o', label=tag)
#             plt.xlabel("k (mutual-kNN)")
#             plt.ylabel(ylabel)
#             plt.title(f"{metric_name} vs k  (sigma={sigma})")
#             plt.legend()
#             plt.tight_layout()
#             outp = f"{cfg.out_prefix}_{metric_name}_vs_k_sigma{sigma}.png".replace(" ", "_")
#             plt.savefig(outp, dpi=cfg.dpi)
#             plt.close()
#
#     _plot_metric_vs_k("homophily", "Edge homophily (weighted)")
#     _plot_metric_vs_k("cross_edge_ratio", "Cross-edge ratio (weighted)")
#     _plot_metric_vs_k("modularity_Q", "Modularity Q")
#     _plot_metric_vs_k("NMI", "NMI (community ↔ label)")
#     _plot_metric_vs_k("ARI", "ARI (community ↔ label)")
#     _plot_metric_vs_k("purity", "Purity (community ↔ label)")
#     _plot_metric_vs_k("conductance_median", "Median conductance (by true class)")
#
#     # === 画鲁棒性：随 sigma 的曲线（固定 k=列表的中位数）===
#     k0 = int(np.median(cfg.k_list))
#     def _plot_metric_vs_sigma(metric_name: str, ylabel: str):
#         plt.figure()
#         sub = df[df['k'] == k0]
#         for tag in ["CE-only", "CE+SupCon"]:
#             dd = sub[sub['method'] == tag].sort_values('sigma')
#             plt.plot(dd['sigma'], dd[metric_name], marker='o', label=tag)
#         plt.xlabel("Noise σ in feature space")
#         plt.ylabel(ylabel)
#         plt.title(f"{metric_name} vs sigma  (k={k0})")
#         plt.legend()
#         plt.tight_layout()
#         outp = f"{cfg.out_prefix}_{metric_name}_vs_sigma_k{k0}.png".replace(" ", "_")
#         plt.savefig(outp, dpi=cfg.dpi)
#         plt.close()
#
#     for m, ylab in [
#         ("homophily", "Edge homophily (weighted)"),
#         ("cross_edge_ratio", "Cross-edge ratio (weighted)"),
#         ("modularity_Q", "Modularity Q"),
#         ("NMI", "NMI (community ↔ label)"),
#         ("ARI", "ARI (community ↔ label)"),
#         ("purity", "Purity (community ↔ label)"),
#         ("conductance_median", "Median conductance (by true class)")
#     ]:
#         _plot_metric_vs_sigma(m, ylab)
#
#     return {
#         "csv": csv_path,
#         "figs_dir": os.path.abspath(os.path.dirname(csv_path)),
#     }

# ---- helpers for panel A scaling ----
def _safe(x):
    return np.nan if x is None else float(x) if not (isinstance(x, float) and math.isnan(x)) else np.nan

def _inv_db(db):
    return np.nan if (db is None or (isinstance(db, float) and math.isnan(db))) else 1.0 / max(db, 1e-6)

def _scale_01_pos(val):
    # 简单把CH做log后min-max归一（避免量纲影响）；若数据太少，此步仅作可视化尺度化
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return np.nan
    v = np.log(max(val, 1.0))
    return (v - 0.0) / (v + 5.0)  # 单调增的压缩映射

# -------------------- Main entry --------------------
# def generate_panels(
#     repr_flat: np.ndarray,
#     repr_flat_wo: np.ndarray,
#     labels_org_flat: np.ndarray,
#     resting_sub_labels: np.ndarray,
#     resting_class: int = 0,
#     outdir: str = "fig_supcon_vs_ce",
#     random_state: int = 42,
#     also_dump_umap: bool = True
# ):
#     """
#     主函数：生成 A–D 四面板 + 拼版图，导出CSV。
#     返回：各面板结果与文件路径。
#     """
#     _ensure_dir(outdir)
#     X_sup = np.asarray(repr_flat)
#     X_ce  = np.asarray(repr_flat_wo)
#     y     = np.asarray(labels_org_flat).astype(int)
#     sub   = np.asarray(resting_sub_labels).astype(int)
#
#     # # Panel A
#     # panelA_png = str(Path(outdir) / "panel_A_2.1_clustering_quality.png")
#     # panelA_csv = str(Path(outdir) / "panel_A_metrics.csv")
#     # Ares = panel_A_metrics(
#     #     X_all={"SupCon": X_sup, "CEonly": X_ce},
#     #     y_coarse=y, resting_class=resting_class, sub_rest=sub,
#     #     out_png=panelA_png, out_csv=panelA_csv, random_state=random_state
#     # )
#
#     # Panel B
#     panelB_png = str(Path(outdir) / "panel_B_2.2_gcd_end_to_end.png")
#     panelB_csv = str(Path(outdir) / "panel_B_metrics.csv")
#     Bres = panel_B_gcd_like(
#         X_sup=X_sup, X_ce=X_ce, y_coarse=y,
#         resting_sub=sub, resting_class=resting_class,
#         out_png=panelB_png, out_csv=panelB_csv
#     )
#
#     # Panel C
#     panelC_png = str(Path(outdir) / "panel_C_2.3_linear_probe.png")
#     panelC_csv = str(Path(outdir) / "panel_C_linear_probe.csv")
#     Cres = panel_C_linear_probe(
#         X_sup=X_sup, X_ce=X_ce, y_coarse=y,
#         resting_sub=sub, resting_class=resting_class,
#         out_png=panelC_png, out_csv=panelC_csv
#     )
#
#     # Panel D
#     # panelD_png = str(Path(outdir) / "panel_D_2.4_graph_structure.png")
#     # panelD_csv = str(Path(outdir) / "panel_D_metrics.csv")
#     # Dres = panel_D_graph(
#     #     X_sup=X_sup, X_ce=X_ce, y_coarse=y,
#     #     resting_class=resting_class,
#     #     out_png=panelD_png, out_csv=panelD_csv
#     # )
#     # Bout = panel_D_graph_v2(
#     #     X_sup, X_ce, y,
#     #     focus_class=resting_class,  # 也可不传，默认全体样本
#     #     k_list=[5, 10, 15, 20],
#     #     sigma_list=[0.0, 0.01, 0.02],  # 噪声强度（特征空间加高斯噪声）
#     #     resolutions=[1.0],  # Leiden/Louvain 分辨率；可给多值
#     #     metric='cosine',
#     #     out_prefix='panel_D_v2',  # 会生成 panel_D_v2_xxx.csv/png
#     # )
#
#     # 可选：全体/Resting的 UMAP（调试/附图）
#     extra_paths = {}
#     if also_dump_umap:
#         # 全体 UMAP：SupCon vs CE-only
#         for tag, X in [("SupCon", X_sup), ("CEonly", X_ce)]:
#             Xs = _scale(X)
#             P2 = _umap2d(Xs, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=random_state)
#             plt.figure(figsize=(7, 5), dpi=150)
#             for c in np.unique(y):
#                 m = (y == c)
#                 plt.scatter(P2[m, 0], P2[m, 1], s=6, alpha=0.85, label=f"class {int(c)}")
#             plt.legend(markerscale=2, fontsize=8, frameon=True)
#             plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
#             plt.title(f"UMAP (all) — {tag}")
#             p = str(Path(outdir) / f"UMAP_all_{tag}.png")
#             plt.tight_layout(); plt.savefig(p); plt.close()
#             extra_paths[f"UMAP_all_{tag}"] = p
#
#         # Resting子集的预测簇UMAP：SupCon vs CE-only（用于目视对比）
#         mask = (y == resting_class)
#         true_sub = sub
#         for tag, X in [("SupCon", X_sup), ("CEonly", X_ce)]:
#             Xr = _scale(X)[mask]
#             y_pred, algo, extra = _cluster_full(Xr, prefer_hdbscan=True, random_state=random_state)
#             P2 = _umap2d(Xr, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=random_state)
#             # 预测簇
#             plt.figure(figsize=(7, 5), dpi=150)
#             for c in np.unique(y_pred):
#                 m = (y_pred == c)
#                 lab = f"cluster {int(c)}" if c != -1 else "noise"
#                 plt.scatter(P2[m, 0], P2[m, 1], s=8, alpha=0.85, label=lab)
#             plt.legend(markerscale=2, fontsize=8, frameon=True)
#             plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
#             plt.title(f"Resting — Predicted clusters ({tag}, {algo})")
#             p = str(Path(outdir) / f"UMAP_rest_pred_{tag}.png")
#             plt.tight_layout(); plt.savefig(p); plt.close()
#             extra_paths[f"UMAP_rest_pred_{tag}"] = p
#
#             # 真子类
#             plt.figure(figsize=(7, 5), dpi=150)
#             for c in np.unique(true_sub):
#                 m = (true_sub == c)
#                 plt.scatter(P2[m, 0], P2[m, 1], s=8, alpha=0.85, label=f"true sub {int(c)}")
#             plt.legend(markerscale=2, fontsize=8, frameon=True)
#             plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
#             plt.title(f"Resting — True sub-classes ({tag})")
#             p = str(Path(outdir) / f"UMAP_rest_true_{tag}.png")
#             plt.tight_layout(); plt.savefig(p); plt.close()
#             extra_paths[f"UMAP_rest_true_{tag}"] = p
#
#     # 拼版（A–D）— 每张独立画布已保存，这里不强制拼接；如需拼图可复用PIL
#     result = {
#         # "panel_A": Ares,
#         "panel_B": Bres,
#         "panel_C": Cres,
#         # "panel_D": Dres,
#         # "panel_D_v2": Bout,
#         "extra_umap": extra_paths,
#         "outdir": outdir
#     }
#     # 也输出一个总json
#     with open(Path(outdir) / "summary.json", "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2, default=str)
#     return result

def _load_pickle_stream(path):
    """读取被多次 pickle.dump() 到同一个文件的流式结构，合并成一个 dict（无异常版）。"""
    out = {}
    with open(path, "rb") as raw:
        f = io.BufferedReader(raw)
        while f.peek(1):              # 还能“窥视”到至少1字节 => 还没到EOF
            obj = pickle.load(f)
            if isinstance(obj, dict):
                out.update(obj)
    return out

def read_sensor_data(TURTLE_PKL_PATH):
    # viz_latent_subact.py生成的category_org
    df_list = []
    with open(TURTLE_PKL_PATH, "rb") as raw:
        f = io.BufferedReader(raw)
        while f.peek(1):  # 还能窥视到至少1字节 => 还没到 EOF
            item = pickle.load(f)
            df_list.append(item)
        df_all = pd.concat(df_list, ignore_index=True)
        df_all['category'] = df_all['Label'].map(labelcategory_dict)
        df_all['label_id'] = df_all['category'].map(label_dict)
        df_all['label_id'] = df_all['label_id'].fillna(-2)

        # df_all['category_org'] = df_all['Label'].map(labelcategory_dict_org)
        df_all['category_org'] = df_all['Label'].map(labelcategory_dict_rst)
        # df_all['label_id_org'] = df_all['category_org'].map(label_dict_org)
        df_all['label_id_org'] = df_all['category_org'].map(label_dict_rst)
        df_all['label_id_org'] = df_all['label_id_org'].fillna(-2)

    return df_all

def _build_model(sensor_type: str, nclass: int):
    # 与原模型名称一一对应
    if sensor_type == "Accel":
        model = SimpleNN_1s(number_classes=nclass)
    elif sensor_type in ("AccelDepth", "GyroDepth"):
        model = SimpleNN_13s(number_classes=nclass)
    elif sensor_type == "AccelGyro":
        model = SimpleNN_33s(number_classes=nclass)
    elif sensor_type == "AccelGyroDepth":
        model = SimpleNN_331s(number_classes=nclass)
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}")
    return model

def _prepare_windowed_np(df, sensor_type: str, win_len=50, win_step=50):
    """对齐你原脚本的滑窗+通道选择"""
    import numpy as np
    if sensor_type == "Accel":
        cols = ["AccX", "AccY", "AccZ", "label_id"]
        arr = process_acc(df, cols)
        # arr: [N, 4], 后两列是label? (这里 process_* 函数内部已处理)
        # 统一走与原脚本一致的管线：
        tmp_b = sliding_window(arr, win_len, win_step)
        data_b   = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, C, T]
        label_b  = tmp_b[:, :, -1]
    elif sensor_type in ("AccelDepth", "GyroDepth"):
        cols = ["AccX", "AccY", "AccZ", "Depth", "label_id"]
        arr = process_acc_temperature(df, cols)
        tmp_b = sliding_window(arr, win_len, win_step)
        data_b   = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))
        label_b  = tmp_b[:, :, -1]
    elif sensor_type == "AccelGyro":
        cols = ["AccX","AccY","AccZ","GyrX","GyrY","GyrZ","label_id"]
        arr = process_acc_gyr(df, cols)
        tmp_b = sliding_window(arr, win_len, win_step)
        data_b   = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))
        label_b  = tmp_b[:, :, -1]
    elif sensor_type == "AccelGyroDepth":
        cols = ['AccX', 'AccY', 'AccZ',
                   'GyrX', 'GyrY', 'GyrZ',
                   'Depth', 'label_id', 'label_id_org']
        arr = process_acc_gyr_dpt(df, cols)
        tmp_b = sliding_window(arr, win_len, win_step)
        data_b   = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))
        label_org_b = tmp_b[:, :, -1]  # [B, Len]
        label_b = tmp_b[:, :, -2]  # [B, Len]
    else:
        raise ValueError(f"Unknown sensor_type: {sensor_type}")

    return data_b.astype(float), label_b, label_org_b

def _plot_umap_html(umap_2d, labels_int, labels_org_flat, html_path, html_org_path):
    # 将整数标签 -> 文本 & 颜色
    label_text = [labeldict_findstr_large.get(int(x), "Unknown") for x in labels_int]
    label_org_text = [labeldict_findstr_rst.get(int(x), "Unknown") for x in labels_org_flat]
    color_map = {
        'Resting':            label_colors[0],
        'Swimming':           label_colors[1],
        'Stay in surface':    label_colors[2],
        'Gliding':            label_colors[3],
        'Feeding':            label_colors[4],
        'Scratching':         label_colors[5],
        'Breathing':          label_colors[6],
        # 'Unknown':            "lightgray"
    }
    fig = px.scatter(
        x=umap_2d[:, 0],
        y=umap_2d[:, 1],
        color=label_text,
        color_discrete_map=color_map,
        labels={"color": "Activity", "x":"UMAP 1", "y":"UMAP 2"},
    )
    # 全局点样式
    for tr in fig.data:
        tr.marker.size = 5
        tr.marker.opacity = 0.8
        tr.marker.line = dict(width=0)

    os.makedirs(os.path.dirname(html_path), exist_ok=True)
    fig.write_html(html_path)

    fig_org = px.scatter(
        x=umap_2d[:, 0],
        y=umap_2d[:, 1],
        color=label_org_text,
        color_discrete_map=color_map,
        labels={"color": "Activity", "x":"UMAP 1", "y":"UMAP 2"},
    )
    # 全局点样式
    for tr in fig_org.data:
        tr.marker.size = 5
        tr.marker.opacity = 0.8
        tr.marker.line = dict(width=0)

    os.makedirs(os.path.dirname(html_org_path), exist_ok=True)
    fig_org.write_html(html_org_path)
    print(f"[OK] Saved interactive scatter to: {html_org_path}")

    return fig, fig_org


def _cluster_full_kmeans_bic(X, random_state=42):
    best_k, bics, _ = _gmm_bic_best_k(X, k_min=2, k_max=12, random_state=random_state)
    km = KMeans(n_clusters=int(best_k), random_state=random_state, n_init=10)
    pred = km.fit_predict(X)
    extra = {"noise_ratio":0.0, "auto_k":int(best_k), "bic_curve":bics}
    return pred, "KMeans(best_k via GMM-BIC)", extra

def panelB_metrics(X_sup,X_ce,y,y9,rc,k_true=9,tau=0.6,random_state=42):
    # y9=_make_true_labels_9class(y,sub,rc);
    tu=(y==rc)
    def compute_one(X):
        Xs=_scale(X); y_pred,algo,extra=_cluster_full_kmeans_bic(Xs,random_state=random_state)
        k_est=int(extra.get("auto_k",0))
        nmi=float(normalized_mutual_info_score(y9,y_pred))
        ari=float(adjusted_rand_score(y9,y_pred))
        _,_,unk_f1=_unknown_f1_from_clusters(y_pred,tu,tau=tau,noise_as_unknown=False)
        err=min(1.0, abs(k_est-int(k_true))/float(k_true))
        return dict(nmi=nmi,ari=ari,unk_f1=unk_f1,k_err_scaled=err,k_est=k_est,algo=algo)
    sup=compute_one(X_sup); ce=compute_one(X_ce)
    df=pd.DataFrame({"Metric":["NMI (9-class)","ARI (9-class)","Unknown F1","|k_est-9|↓ (scaled)"],
                     "CE-only":[ce["nmi"],ce["ari"],ce["unk_f1"],ce["k_err_scaled"]],
                     "CE+SupCon":[sup["nmi"],sup["ari"],sup["unk_f1"],sup["k_err_scaled"]]})
    return df,sup,ce

def main(EPOCH_TO_USE, WEIGHT_PKL_PATH, WOCL_WEIGHT_PKL_PATH):
    # 1) 读取权重结果PKL（流式dump）
    result = _load_pickle_stream(WEIGHT_PKL_PATH)
    wocl_result = _load_pickle_stream(WOCL_WEIGHT_PKL_PATH)
    if "weight_list" not in result:
        raise RuntimeError("weight_list not found in the given PKL. 请确认路径与文件是否正确。")
    weight_list = result["weight_list"]
    wocl_weight_list = wocl_result["weight_list"]
    if not isinstance(weight_list, list) or len(weight_list) == 0:
        raise RuntimeError("weight_list is empty or invalid.")

    # epoch(迭代)到索引：原脚本每一轮迭代 append 一次
    # 如果 EPOCH_TO_USE=5，通常使用 weight_list[4]
    idx = EPOCH_TO_USE - 1
    if idx < 0 or idx >= len(weight_list):
        raise IndexError(f"EPOCH_TO_USE={EPOCH_TO_USE} 超出可用范围 [1, {len(weight_list)}].")

    state_dict = weight_list[idx]
    wocl_state_dict = wocl_weight_list[idx]

    # 2) 重建模型并载入权重
    nclass = int(max(list(label_dict.values())) + 1)  # 7类（-1为Unknown不参与）
    model = _build_model(SENSOR_TYPE, nclass=nclass).to(DEVICE)
    wocl_model = copy.deepcopy(model).to(DEVICE)
    # 如果有些权重键不匹配，可适当使用 strict=False
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    wocl_missing, wocl_unexpected = wocl_model.load_state_dict(wocl_state_dict, strict=False)
    if missing:
        print("[Warn] Missing keys:", missing)
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)
    model.eval()
    wocl_model.eval()

    # 3) 读取并预处理数据（与原脚本一致）
    df_all = read_sensor_data(TURTLE_PKL_PATH)
    # 仅使用已标注（label_id > -1）
    selected_df = df_all[df_all["label_id"] > -1].copy()

    data_b, label_b, label_org_b = _prepare_windowed_np(selected_df, SENSOR_TYPE, win_len=50, win_step=50)
    # 用多数投票得到窗标签（B,T） -> (B,)
    y_major = majority_value(label_b)

    # 组装 DataLoader
    plot_dataset = data_loader_umineko(
        data_b.astype(float),
        y_major.astype(int),      # majority labels
        label_b.astype(int),      # raw per-step labels（实际 AE_eval_time_series 不使用或仅透传）
        device=DEVICE
    )
    plot_loader = DataLoader(plot_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    major_label_b = majority_value(label_org_b)
    plot_dataset = data_loader_umineko(data_b.astype(float),
                                       major_label_b.astype(int),
                                       label_org_b.astype(int),
                                       device=DEVICE)
    plot_loader_org = DataLoader(plot_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # 4) 前向抽特征
    with (torch.no_grad()):
        repres_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(plot_loader, model, DEVICE)
        repres_list_org, sample_list_org, pred_list_org, label_list_org = \
            AE_eval_time_series(plot_loader_org, model, DEVICE)
        # 比如 rest 的 coarse label 是 0
        fig, ax = plot_rest_sub_behavior_violin(
            sample_list=sample_list,
            pred_list=label_list_org,
            label_list=label_list,
            rest_class=0,
            feature_name="Mean accel magnitude",
            sub_label_prefix="Rest sub ",
            out_png="fig_supcon_vs_ce_ep%d" % EPOCH_TO_USE + "rest_sub_violin.png",
        )


        # 仅 CE-only 特征
        wocl_repres_list, wocl_sample_list, wocl_pred_list, wocl_label_list = \
            AE_eval_time_series(plot_loader, wocl_model, DEVICE)
        # wocl_repres_list_org, wocl_sample_list_org, wocl_pred_list_org, wocl_label_list_org = \
        #     AE_eval_time_series(plot_loader_org, wocl_model, DEVICE)

    # 5) 拼接 -> 高维->2D UMAP
    repr_concat = np.concatenate(repres_list)           # 形如 [B, C, T] 或 [B, D]
    repr_flat   = repr_concat.reshape(repr_concat.shape[0], -1).astype(float)
    labels_flat = majority_value(np.concatenate(label_list))  # (B,)
    labels_org_flat = majority_value(np.concatenate(label_list_org))  # (B,)
    # 仅 CE-only 特征
    wocl_repr_concat = np.concatenate(wocl_repres_list)  # 形如 [B, C, T] 或 [B, D]
    wocl_repr_flat = wocl_repr_concat.reshape(wocl_repr_concat.shape[0], -1).astype(float)
    # wocl_labels_flat = majority_value(np.concatenate(wocl_label_list))  # (B,)
    # wocl_labels_org_flat = majority_value(np.concatenate(wocl_label_list_org))  # (B,)


    mask = ((labels_org_flat == 0) |
            (labels_org_flat == 7) |
            (labels_org_flat == 8) |
            (labels_org_flat == 9))
    restidx = np.where(mask)[0]
    sub_rest = labels_org_flat[restidx]

    X_all = {
        "SupCon": repr_flat,
        "CEonly": wocl_repr_flat,
    }

    res = panel_A_metrics(
        X_all=X_all,
        y_coarse=labels_flat,  # not use
        resting_idx=restidx,
        sub_rest=sub_rest,
        out_png="fig_supcon_vs_ce_ep%d" % EPOCH_TO_USE + "/panelA_demo.png",
        out_csv="fig_supcon_vs_ce_ep%d" % EPOCH_TO_USE + "/panelA_demo.csv",
    )



    # # figure 4: pannel B
    # df, sup_stats, ce_stats = panelB_metrics(repr_flat, wocl_repr_flat,
    #                                          labels_flat, labels_org_flat,
    #                                          0, k_true=int(max(labels_org_flat)+1),
    #                                          tau=0.6, random_state=42)
    # df.to_csv("fig_supcon_vs_ce_ep%d"%EPOCH_TO_USE+"/panel_B_metrics.csv", index=False)
    # print('finish')


    # results = generate_panels(
    #     repr_flat=repr_flat,            # repr_supcon[N,D]
    #     repr_flat_wo=wocl_repr_flat,        # repr_ce_only[N,D]
    #     labels_org_flat=labels_flat,           # [N]
    #     resting_sub_labels=labels_org_flat_rest,      # [n_rest]：将resting子集的真子类标签传入，其他位置-1
    #     outdir="fig_supcon_vs_ce_ep%d"%EPOCH_TO_USE
    # )





if __name__ == "__main__":
    EPOCH_TO_USE = 20  # 取第几次迭代保存的权重（你说是 5）
    # run viz_latent_subact.py 里生成的 pkl 文件路径(turtle)
    WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm20_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    WOCL_WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm0_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    # ↑ 上面只是示例命名：与你在 run() 里生成的 pkl 文件名一致即可
    #   例如：sensor_type+'_%s_epoch%s_results_Day1105.pkl' （iteration==5）

    main(EPOCH_TO_USE, WEIGHT_PKL_PATH, WOCL_WEIGHT_PKL_PATH)
    #
    EPOCH_TO_USE = 15  # 取第几次迭代保存的权重（你说是 5）
    # run viz_latent_subact.py 里生成的 pkl 文件路径(turtle)
    WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm20_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    WOCL_WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm0_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    # ↑ 上面只是示例命名：与你在 run() 里生成的 pkl 文件名一致即可
    #   例如：sensor_type+'_%s_epoch%s_results_Day1105.pkl' （iteration==5）

    main(EPOCH_TO_USE, WEIGHT_PKL_PATH, WOCL_WEIGHT_PKL_PATH)
    #
    EPOCH_TO_USE = 10  # 取第几次迭代保存的权重（你说是 5）
    # run viz_latent_subact.py 里生成的 pkl 文件路径(turtle)
    WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm20_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    WOCL_WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm0_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    # ↑ 上面只是示例命名：与你在 run() 里生成的 pkl 文件名一致即可
    #   例如：sensor_type+'_%s_epoch%s_results_Day1105.pkl' （iteration==5）

    main(EPOCH_TO_USE, WEIGHT_PKL_PATH, WOCL_WEIGHT_PKL_PATH)

    EPOCH_TO_USE = 5  # 取第几次迭代保存的权重（你说是 5）
    # run viz_latent_subact.py 里生成的 pkl 文件路径(turtle)
    WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm20_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    WOCL_WEIGHT_PKL_PATH = r"D:\code\DeepView\deepview\calculate_results\data\turtle\AccelGyroDepth__entropy_rst_Contrast1_warm0_seed2025_epoch%d_results_Day1105.pkl" % EPOCH_TO_USE
    # ↑ 上面只是示例命名：与你在 run() 里生成的 pkl 文件名一致即可
    #   例如：sensor_type+'_%s_epoch%s_results_Day1105.pkl' （iteration==5）

    main(EPOCH_TO_USE, WEIGHT_PKL_PATH, WOCL_WEIGHT_PKL_PATH)