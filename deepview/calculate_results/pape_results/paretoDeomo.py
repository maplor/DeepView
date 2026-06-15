# 这个需要放到albe工具箱内
# 需要结合supervised_base.py 和 paretoFigure.ipynb两个函数整合
import pickle
import random
import torch.nn as nn
from torch import optim

from deepview.calculate_results.data.umineko.train_func import (
    train_model,
    # evaluate_model,
)

import numpy as np
import pandas as pd

# —— 复用你项目中的数据/模型工具（保持接口一致） ——
from deepview.calculate_results.models.utils import (
    sensor_list_to_tag,
    read_sensor_data,
)

import os
import copy
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

# —— 复用你项目中的数据/模型工具（保持接口一致） ——
from deepview.calculate_results.models.utils import (
    data_loader_umineko,
    majority_value,
    sliding_window,
    process_sensors,
    sensor_list_to_tag,
    # read_sensor_data,
)


# from deepview.calculate_results.data.omizunagidori.active_supContrast import read_sensor_data
from deepview.calculate_results.data.umineko.model_func import get_model_for_sensors
from deepview.calculate_results.data.umineko.train_func import evaluate_model
from itertools import combinations
# import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def set_random_seed(seed):
    # Set seed for Python's random module
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # If using CUDA, set seed for GPU as well
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


def accuracy_class(predict_labels, true_labels):
    # 1. 获取唯一类别
    unique_classes = np.unique(true_labels)

    # 2. 计算每个类别的准确率
    class_accuracies = {}
    for cls in unique_classes:
        # 筛选出属于当前类别的样本
        class_indices = true_labels == cls
        # 计算预测正确的数量和总数量
        correct_predictions = np.sum(predict_labels[class_indices] == true_labels[class_indices])
        total_samples = np.sum(class_indices)
        # 计算准确率
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0
        class_accuracies[cls] = accuracy

    # 输出每个类别的准确率
    for cls, acc in class_accuracies.items():
        print(f"Class {cls}: Accuracy = {acc:.2f}")


#-------------------------parameters---------------------------

device = 'cuda:1' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
len_sw = 50
batch_size = 2000
num_epochs = 50
all_sensor_list = [["Gyro"], ["Mag", "Illum", "Temp", "Press"],
    # , ["Mag", "Illum", "Temp"],
    #           ["Mag", "Illum", "Press"], ["Mag", "Temp", "Press"],
    #           ["Mag", "Illum"], ["Mag", "Temp"], ["Mag", "Press"],
    #           ["Accel", "Illum"],
                   ["Accel"]]  # 需要在完成syn lcb后确认
# all_sensor_list = [ ["Mag", "Illum", "Temp"], ["Gyro"]]

bird = 'umineko'  # 'umineko' or 'omizunagidori'
if bird=='umineko':
    from deepview.calculate_results.models.utils import read_sensor_data
    root_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko'
elif bird=='omizunagidori':
    from deepview.calculate_results.data.omizunagidori.active_supContrast import read_sensor_data
    root_path = r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori'
else:
    print('error, other animals')

sensor_specs = pd.DataFrame([
    {"sensor":"ACC",   "energy_mAh_per_day":0.35, "mass_g":0.135, "price_usd":2.5,  "volume_mm3":4.0},
    {"sensor":"GYRO",  "energy_mAh_per_day":20.0, "mass_g":0.098, "price_usd":4.5,  "volume_mm3":6.0},
    {"sensor":"MAG",   "energy_mAh_per_day":0.96, "mass_g":0.078, "price_usd":5.0,  "volume_mm3":4.0},
    {"sensor":"TEMP",  "energy_mAh_per_day":0.084,"mass_g":0.012, "price_usd":3.0,  "volume_mm3":3.0},
    {"sensor":"PRES",  "energy_mAh_per_day":0.077,"mass_g":0.015, "price_usd":3.1,  "volume_mm3":3.0},
    {"sensor":"ILLU", "energy_mAh_per_day":0.060,"mass_g":0.008, "price_usd":2.6,  "volume_mm3":2.6},
    {"sensor":"GPS",   "energy_mAh_per_day":168.0,"mass_g":0.500, "price_usd":22.0, "volume_mm3":245.0},
])

# n_rep = 5  # repeat 5 times
z_095 = 1.645  # 95% confidence interval z value

WEIGHTS_PATH = None                 # 或者指定模型权重文件 .pt/.pth；若设置了此项，将忽略 RESULTS_PKL
WEIGHT_IDX = -1                     # 从 results.pkl 的 weight_list 里取第几个（-1 表示最后一个）
# LEN_SW = 50                         # 滑窗长度，需与训练一致
TEST_SIZE = 0.2                     # 测试集比例
SEED = 42                           # train_test_split 的随机种子（与训练保持一致）
# DEVICE = "cuda:0"                   # 自动：'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


sensors = ["ACC","GYRO","MAG","TEMP","PRES","ILLU","GPS"]

tag2sensor = {
    'acc': 'ACC',
    'tmp': 'TEMP',
    'gpsbv': 'GPS',
    'prs': 'PRES',
    'gyr': 'GYRO',
    'mag': 'MAG',
    'illum': 'ILLU',

    'acc_tmp': 'ACC+TEMP',
    'acc_gpsbv': 'ACC+GPS',
    'acc_gyr': 'ACC+GYRO',
    'acc_prs': 'ACC+PRES',
    'acc_mag': 'ACC+MAG',
    'acc_illum': 'ACC+ILLU',

    'prs_tmp': 'PRES+TEMP',
    'prs_gpsbv': 'PRES+GPS',
    'prs_illum': 'PRES+ILLU',

    'gyr_gpsbv': 'GYRO+GPS',
    'gyr_mag': 'GYRO+MAG',
    'gyr_illum': 'GYRO+ILLU',
    'gyr_tmp': 'GYRO+TEMP',
    'gyr_prs': 'GYRO+PRES',

    'mag_illum': 'MAG+ILLU',
    'mag_prs': 'MAG+PRES',
    'mag_tmp': 'MAG+TEMP',
    'mag_gpsbv': 'MAG+GPS',

    'tmp_gpsbv': 'TEMP+GPS',
    'tmp_illum': 'TEMP+ILLU',

    'gpsbv_illum': 'GPS+ILLU',

    'prs_tmp_illum': 'PRES+TEMP+ILLU',
    'prs_tmp_gpsbv_illum': 'PRES+TEMP+GPS+ILLU',
    'prs_tmp_illum_gpsbv': 'PRES+TEMP+ILLU+GPS',

    'mag_prs_tmp_illum': 'MAG+PRES+TEMP+ILLU',
    'mag_prs_illum': 'MAG+PRES+ILLU',
    'mag_prs_tmp': 'MAG+PRES+TEMP',
    'mag_illum_tmp': 'MAG+ILLU+TEMP',
    'mag_illum_prs': 'MAG+ILLU+PRES',
    'mag_tmp_illum': 'MAG+TEMP+ILLU',
}
#-------------------------parameters---------------------------

def sum_costs(senl):
    E = M = P = V = 0.0
    for s in senl:
        E += cost_map[s]["energy_mAh_per_day"]
        M += cost_map[s]["mass_g"]
        P += cost_map[s]["price_usd"]
        V += cost_map[s]["volume_mm3"]
    return E, M, P, V

def build_dataset_for_sensors(sensor_type, len_sw=50, compute_gps=True):
    """
    与训练一致的数据构造：
    - 若缓存 {tag}_data.npz 存在则直接读；
    - 否则 read_sensor_data -> process_sensors -> sliding_window，并缓存。
    输出形状 [B, C, T]，label_b 为 [B, T]（与你训练脚本一致的转置约定）。
    """
    tag = sensor_list_to_tag(sensor_type, compute_gps=compute_gps)
    # cache_path = f"{tag}_data.npz"
    cache_path = os.path.join(root_path, f"{tag}_data.npz")

    # if os.path.exists(cache_path):
    if 0:  # todo, 因为read sensor data函数错误，重新计算
        with np.load(cache_path, allow_pickle=False) as f:
            data_b = f["data"]
            label_b = f["label"]
    else:
        all_df = read_sensor_data()
        selected_df = all_df[all_df.label_id != -2]
        selected_np, _ = process_sensors(selected_df, sensor_type)

        tmp_b = sliding_window(selected_np, len_sw, len_sw)
        data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
        label_b = tmp_b[:, :, -1]
        np.savez_compressed(cache_path, data=data_b.astype(np.float32), label=label_b)

    return data_b, label_b, tag


def stratified_train_test_split(data_b, label_b, test_size=TEST_SIZE, seed=SEED):
    """
    用多数投票标签 vote_label 做分层切分（与训练保持一致）。
    """
    vote_label = majority_value(label_b)
    X_train, X_test, y_train, y_test = train_test_split(
        data_b,
        label_b,
        test_size=test_size,
        stratify=vote_label,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


def build_model(sensor_type, num_classes, state_dict=None, weights_path=None, device=device):
    """
    构建并加载权重：
    - 优先 weights_path 的 .pt/.pth
    - 否则使用传入的 state_dict
    """
    model = get_model_for_sensors(sensor_type, number_classes=num_classes)
    model = model.to(device)

    if weights_path is not None:
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        else:
            raise ValueError("Unrecognized checkpoint format for weights_path.")
    elif state_dict is not None:
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Either 'state_dict' or 'weights_path' must be provided.")

    model.eval()
    return model


def build_loader(X, y, device=device, batch_size=4000):
    """
    与训练一致的数据封装到 data_loader_umineko，再用 DataLoader 包装。
    """
    major_y = majority_value(y)
    ds = data_loader_umineko(X.astype(float), major_y.astype(int), y.astype(int), device=device)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)



def get_supervised_results(len_sw, batch_size, num_epochs, all_sensor_list, device):

    sensor_result_dict = {}

    for sensor_type in all_sensor_list:
        if type(sensor_type) == str:
            sensor_type = [sensor_type]

        data_b, label_b, tag = build_dataset_for_sensors(sensor_type, len_sw, compute_gps=True)

        # ======= 5. 多次实验 ======= #
        test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
        test_pred_label_lists, test_truth_label_lists = [], []
        weight_list = []
        loss_list = []
        selected_labels_list = []

        for i in range(2):
        # for i in range(5):
            seed_value=30+i
            set_random_seed(seed_value)
            # name_label = '_supervisebase_seed%s' % str(seed_value)


            # 将数据分为8:2，其中2为测试集
            # todo，未来需要根据已有标签进行数据集切分
            vote_label = majority_value(label_b)
            X_train_full, X_test, y_train_full, y_test = train_test_split(data_b, label_b,
                                                                              test_size=0.2, stratify=vote_label,
                                                                              random_state=42)


            # 初始化模型和优化器
            model = get_model_for_sensors(sensor_type, number_classes=int(np.max(label_b)+1))

            model = model.to(device)  # 不需要读模型权重，模型encoder排序根据 all_sensor_list 来
            classify_criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


            # 创建数据加载器
            major_label_b = majority_value(y_train_full)
            train_dataset = data_loader_umineko(X_train_full.astype(float),
                                                major_label_b.astype(int),
                                                y_train_full.astype(int))
            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=False, drop_last=False)


            ## stage 1: train supervised model, 获得分类器新参数
            model.train()
            model, avg_loss = train_model(model, train_loader, classify_criterion, optimizer,
                                          epochs=num_epochs, device=device, if_contrast=False)
            loss_list.append(avg_loss)  # 记录50次epoch的平均loss

            # ======= 6. 模型测试 ======= #

            major_label_b = majority_value(y_test)
            test_dataset = data_loader_umineko(X_test.astype(float),
                                               major_label_b.astype(int),
                                               y_test.astype(int))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
            print('-------Test----------')
            print(sensor_type)
            (accuracy, macro_f1, micro_f1,
             pred_label_list, truth_label_list) = evaluate_model(model,
                                                                 test_loader,
                                                                 device)

            test_accuracy_list.append(accuracy)
            test_micro_f1_list.append(micro_f1)
            test_macro_f1_list.append(macro_f1)
            test_pred_label_lists.append(np.concatenate(pred_label_list, axis=0))
            test_truth_label_lists.append(np.concatenate(truth_label_list, axis=0))
            selected_labels_list.append(y_train_full)

            # 保存当前模型权重到列表
            weight_list.append(copy.deepcopy(model.state_dict()))


        # ======= 7. 保存结果 ======= #
        result_dict = {
            'test_accuracy_list': test_accuracy_list,
            'test_macro_f1_list': test_macro_f1_list,
            'test_micro_f1_list': test_micro_f1_list,
            'test_pred_label_lists': test_pred_label_lists,
            'test_truth_label_lists': test_truth_label_lists,
            'selected_labels_list': selected_labels_list,
            'weight_list': weight_list,
        }
        tag = sensor_list_to_tag(sensor_type, compute_gps=True)
        sensor_result_dict[tag] = result_dict

    return sensor_result_dict

def run_eval(m_weight, len_sw, device, SENSORS):
    # 1) 规范化传感器列表
    sensor_type = [s.strip() for s in SENSORS] if isinstance(SENSORS, (list, tuple)) else [str(SENSORS)]

    # 2) 构造数据
    data_b, label_b, tag = build_dataset_for_sensors(sensor_type, len_sw=len_sw)

    # 3) 切分（分层、随机种子与训练保持一致）
    X_train, X_test, y_train, y_test = stratified_train_test_split(data_b, label_b,
                                                                   test_size=TEST_SIZE,
                                                                   seed=SEED)

    # 4) 准备权重
    state_dict = copy.deepcopy(m_weight)

    # 5) 构建模型并加载权重
    num_classes = int(np.max(label_b) + 1)
    model = build_model(sensor_type, num_classes, state_dict=state_dict,
                        weights_path=WEIGHTS_PATH, device=device)

    # 6) 评估（与你训练中 evaluate_model 的返回保持一致）
    test_loader = build_loader(X_test, y_test, device=device, batch_size=4000)
    accuracy, macro_f1, micro_f1, pred_list, truth_list = evaluate_model(model, test_loader, device)

    return macro_f1, micro_f1




print('a')



#、、、、、、、、、、、## calculate multiple sensor combos (Syn_LCB)、、、、、、、、、、、、、、、、、

sensor_list=["Accel", "GPS", "Press", "Mag", "Gyro", "Illum", "Temp"]
pair_list = [list(p) for p in combinations(sensor_list, 2)]
sensor_list = [[s] for s in sensor_list]
sensor_combo_list = sensor_list + pair_list


sensor_result_dict = get_supervised_results(len_sw, batch_size,
                                            num_epochs,
                                            sensor_combo_list,
                                            device)

raw_map = {}   # sen_tag -> list of micro-F1
summary, summary_full = [], []   # [(sen_tag, mean, var, std), ...]
all_tags = []
means, stds = [], []
for Sen in sensor_combo_list:
    tag = sensor_list_to_tag(Sen, compute_gps=True)
    sen_result_dict = sensor_result_dict[tag]
    model_weight_list = sen_result_dict['weight_list']
    vals = []
    tag = sensor_list_to_tag(Sen, compute_gps=True)  # compute_gps 来自你的管线配置
    for i in range(len(model_weight_list)):
        m_weight = model_weight_list[i]
        macro_f1, micro_f1 = run_eval(m_weight, len_sw, device, Sen)
        vals.append(float(micro_f1))
    raw_map[tag] = vals

    m = float(np.mean(vals))
    v = float(np.var(vals, ddof=1))   # 样本方差
    s = float(np.std(vals, ddof=1))   # 样本标准差
    summary.append((tag, m, v, s))  # 作条形图
    summary_full.append((tag, vals))
    all_tags.append(tag)
    means.append(m)
    stds.append(s)


rows = []
count = 1
for s_info in summary_full:
    s = tag2sensor[s_info[0]]

    rows.append({
        "combo_id": str(count),
        "sensors": s,
        ** {f"F1_run{i + 1}": v for i, v in enumerate(s_info[1])}
    })
    count += 1

perf_df = pd.DataFrame(rows, columns=list(rows[0].keys()))

# -------------------------------------------------------
# 4) Compute costs by summing per-sensor specs per combo;
#    compute LCB95 from the (placeholder) 5 F1 runs.
# -------------------------------------------------------
# helper: sum costs
cost_map = sensor_specs.set_index("sensor")[["energy_mAh_per_day",
                                             "mass_g", "price_usd", "volume_mm3"]].to_dict(
    "index")

records = []
for _, r in perf_df.iterrows():
    senl = r["sensors"].split("+")
    E, M, P, V = sum_costs(senl)
    runs = np.array([value for key, value in r.items() if 'F1_run' in key], dtype=float)
    fbar = runs.mean()
    s = runs.std(ddof=1)
    lcb = float(fbar - z_095 * s / np.sqrt(len(runs)))
    records.append({
        "combo_id": r["combo_id"],
        "sensors": r["sensors"],
        "Energy_mAh_per_day": E,
        "Mass_g": M,
        "Price_usd": P,
        "Volume_mm3": V,
        "Fbar": fbar,
        "Std": s,
        "LCB95": lcb,
        # "n_rep": n_rep
    })
res_df = pd.DataFrame(records)

# ----------------------------
# 3) Compute Syn_LCB matrix
# ----------------------------
# 含不含 '+' 的布尔掩码
has_plus = res_df["sensors"].str.contains(r"\+", na=False)

# 拆成两个 DataFrame
single_dfu = res_df[~has_plus].copy()  # 没有 '+' 的（单体）
pair_dfu   = res_df[ has_plus].copy()  # 其余（包含 '+' 的组合）

# Align lookup tables
single_lcb = dict(zip(single_dfu["sensors"], single_dfu["LCB95"]))
pair_lcb = {tuple(sorted(p["sensors"].split("+"))): p["LCB95"] for _, p in pair_dfu.iterrows()}

syn_mat = np.full((len(sensors), len(sensors)), np.nan, dtype=float)
for i,a in enumerate(sensors):
    for j,b in enumerate(sensors):
        if i == j:
            syn_mat[i,j] = 0.0
        else:
            key = tuple(sorted([a,b]))
            lcb_ab = pair_lcb.get(key, np.nan)
            base_ab = max(single_lcb[a], single_lcb[b])
            syn = lcb_ab - base_ab if not np.isnan(lcb_ab) else np.nan
            syn_mat[i,j] = syn

syn_df = pd.DataFrame(syn_mat, index=sensors, columns=sensors)

# ----------------------------
# 4) Heatmap (matplotlib; no seaborn; one chart per figure)
# ----------------------------
fig1 = plt.figure(figsize=(9,7))
ax1 = fig1.add_subplot(111)
im1 = ax1.imshow(syn_mat, interpolation='nearest')
ax1.set_xticks(range(len(sensors)))
ax1.set_yticks(range(len(sensors)))
ax1.set_xticklabels(sensors, rotation=45, ha="right")
ax1.set_yticklabels(sensors)
ax1.set_title("Complementarity Heatmap: Syn_LCB(A,B)")
# fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
# 放大坐标轴刻度整体字号
ax1.tick_params(axis='both', labelsize=18)
# 放大colorbar刻度
cbar = fig1.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=18)
fig1.tight_layout()
plt.show()

#、、、、、、、、、、、## calculate multiple sensor combos (Syn_LCB)、、、、、、、、、、、、、、、、、


def plot_dot_panel(ax, tags, vals, means, stds, cis95, title=None,
                   point_alpha=0.7, seed_size=24, mean_size=90, grid=True,
                   xtick_bins=4, plot_xlabel=True):
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis='x', color='0.85', lw=0.8)
    y = np.arange(len(tags), dtype=float)

    # 5 个 seed 的小点（轻微竖向 jitter 防重叠）
    rng = np.random.default_rng(42)
    for i, v in enumerate(vals):
        jit = rng.uniform(-0.12, 0.12, size=len(v))
        ax.scatter(v, np.full_like(v, y[i]) + jit,
                   s=seed_size, alpha=point_alpha,
                   linewidths=0.5, edgecolor='white', color='#6ea8ff')

    # ±1 std 线 & 95%CI 带；均值黑点
    ax.hlines(y, means - stds,  means + stds,  lw=2.0, color='k', alpha=0.75)
    ax.hlines(y, means - cis95, means + cis95, lw=5.0, color='k', alpha=0.18)
    ax.scatter(means, y, s=mean_size, color='black', zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(tags, fontsize=10)
    if plot_xlabel:
        ax.set_xlabel("Micro-F1 score")
    if title:
        ax.set_title(title, fontsize=14)

    # x 轴范围留白
    xmin = min(min(v) for v in vals); xmax = max(max(v) for v in vals)
    pad = max(0.01, 0.03*(xmax - xmin))
    ax.set_xlim(xmin - pad, xmax + pad)

    # —— 关键：限制刻度数量 + 固定格式，防止重叠 ——
    ax.xaxis.set_major_locator(MaxNLocator(nbins=xtick_bins, prune='both'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='x', labelsize=10, pad=2)

    return ax

def plot_combo_with_inset(raw_map, all_combos,
                          inset_loc="lower right",
                          inset_size=("42%", "48%"),
                          figsize=(6.2, 7.4), dpi=180,
                          xtick_bins_main=4, xtick_bins_inset=4,
                          name_map=None):
    # 名称映射
    if name_map is None:
        name_map = {"Accel":"acc","GPS":"gpsbv","Press":"prs","Illum":"illum",
                    "Temp":"tmp","Gyro":"gyr","Mag":"mag"}
    def sensors_to_tag(sen_list, sep="_"):
        return sep.join([name_map[s] for s in sen_list])

    # 拆分单体/多组合
    single_tags = [sensors_to_tag(c) for c in all_combos if len(c) == 1]
    multi_tags  = [sensors_to_tag(c) for c in all_combos if len(c) > 1]

    # 整理数据（按均值从低到高）
    def prepare_panel(tags):
        items = []
        for t in tags:
            if t in raw_map and len(raw_map[t]) > 0:
                v = np.asarray(raw_map[t], dtype=float)
                v = v[~np.isnan(v)]
                if v.size > 0:
                    items.append((t, v))
        if not items:
            return [], [], [], [], []
        items.sort(key=lambda kv: kv[1].mean())
        tags_sorted = [k for k,_ in items]
        vals = [v for _,v in items]
        means = np.array([v.mean() for v in vals])
        stds  = np.array([v.std(ddof=1) if len(v)>1 else 0.0 for v in vals])
        ns    = np.array([len(v) for v in vals])
        cis95 = 1.96 * stds / np.sqrt(np.maximum(ns, 1))
        return tags_sorted, vals, means, stds, cis95

    tags_m, vals_m, means_m, stds_m, cis95_m = prepare_panel(multi_tags)
    tags_s, vals_s, means_s, stds_s, cis95_s = prepare_panel(single_tags)

    # 画布：更高更窄
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    plot_dot_panel(ax, tags_m, vals_m, means_m, stds_m, cis95_m,
                   title="Micro-F1 across 5 seeds (points) with mean ± std / 95% CI",
                   xtick_bins=xtick_bins_main)

    # 右下角 inset（只放单体）
    # ax_ins = inset_axes(ax, width=inset_size[0], height=inset_size[1],
    #                     loc=inset_loc, borderpad=1.0)
    ax_ins = inset_axes(
                        ax,
                        width="48%", height="48%",
                        loc="lower right",          # 仍然以右下角为参考
                        borderpad=0.8,              # 内缩一点点（可选）
                        bbox_to_anchor=(0, 0.05, 1, 1),  # 👈 往上挪动 0.06（axes 的 6% 高度）
                        bbox_transform=ax.transAxes
                        )
    plot_dot_panel(ax_ins, tags_s, vals_s, means_s, stds_s, cis95_s,
                   title=None, seed_size=20, mean_size=70, grid=True,
                   xtick_bins=xtick_bins_inset, plot_xlabel=False)
    ax_ins.set_ylabel("")
    for lbl in ax_ins.get_yticklabels():
        lbl.set_fontsize(9)
    for spine in ax_ins.spines.values():
        spine.set_alpha(0.4)

    # plt.tight_layout()
    return fig, ax, ax_ins

all_combos = [['Accel','Gyro','Mag','Press','Temp','GPS','Illum'],
              ['Press', 'Temp', 'Illum'],
              ["Press", "Temp", "GPS", "Illum"],
              ['Accel'],['GPS'],['Press'],['Illum'],['Temp'],['Gyro'],['Mag'],['Accel', 'Temp'],['Press', 'Temp']]
fig, ax, ax_ins = plot_combo_with_inset(raw_map, all_combos)
#-------------------------------pareto------------------------------------





# -----------------------------
# 工具：非支配前沿（若 df 无 ParetoFront 列则计算）
# -----------------------------

def pareto_front(df: pd.DataFrame, minimize=(), maximize=()):
    """
    返回布尔数组 is_front，表示 df 各行是否在 Pareto 前沿上。
    - minimize: 需要“越小越好”的列名序列
    - maximize: 需要“越大越好”的列名序列

    定义：j 支配 i <=> 对所有维度 x_j <= x_i，且至少一维 x_j < x_i（严格更优）。
    对 maximize 列，先取负以统一为“越小越好”。
    """
    if not (minimize or maximize):
        raise ValueError("minimize / maximize 至少提供一个。")

    cols = list(minimize) + list(maximize)

    # 基础校验
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")
    if df[cols].isna().any().any():
        where = {c: df.index[df[c].isna()].tolist() for c in cols if df[c].isna().any()}
        raise ValueError(f"这些列存在 NaN: {where}")

    X = df[cols].to_numpy(dtype=float)
    # 将“最大化”列取负，统一成最小化
    if maximize:
        X[:, len(minimize):] *= -1.0

    n = X.shape[0]
    is_front = np.ones(n, dtype=bool)

    # O(n^2) 非支配检测
    for i in range(n):
        if not is_front[i]:
            continue
        diff = X - X[i]               # [n, d]
        le = (diff <= 0).all(axis=1)  # 所有维度不大于
        lt = (diff <  0).any(axis=1)  # 至少一维严格小
        dominated_by_any = le & lt
        dominated_by_any[i] = False
        if dominated_by_any.any():
            is_front[i] = False

    return is_front

# -----------------------------
# 工具：坐标压缩与“好看”刻度
# -----------------------------
def tf_log1p10(x):
    x = np.asarray(x, dtype=float)
    return np.log10(1.0 + np.clip(x, 0, None))

def nice_log1p_ticks(data_raw, base_candidates=None, k=6, min_gap=0.10):
    """
    从原始单位中挑最多 k 个刻度（1-2-5 序列），在 log1p10 空间中保证最小间距 min_gap。
    返回原始刻度值数组（未变换）。
    """
    dmin, dmax = float(np.min(data_raw)), float(np.max(data_raw))
    if base_candidates is None:
        mags = np.arange(-4, 5)  # 10^-4 .. 10^4
        base = np.concatenate([1*np.power(10.0, mags),
                               2*np.power(10.0, mags),
                               5*np.power(10.0, mags)])
        base = np.unique(np.r_[0.0, base])
    else:
        base = np.asarray(base_candidates, dtype=float)
    cand = base[(base >= dmin) & (base <= dmax)]
    tvals = tf_log1p10(cand)
    keep, last = [], -1e9
    for v, tv in zip(cand, tvals):
        if tv - last >= min_gap:
            keep.append(v); last = tv
        if len(keep) >= k: break
    if len(keep) == 0:
        keep = [dmin, dmax]
    return np.array(keep, dtype=float)

# -----------------------------
# 工具：去重叠标注（黄金角偏移 + 白描边）
# -----------------------------
def annotate_points_3d(ax, X, Y, Z, labels, indices, color="#1f2937", fs=11, spread=0.015):
    for i in indices:
        ax.text(
            X[i] + spread, Y[i] + spread, Z[i],
            str(labels[i]),
            fontsize=fs, color=color, ha="left", va="bottom", zorder=5,
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )
# -----------------------------
# 主函数：漂亮的 3D 帕累托图
# -----------------------------
def plot_pareto_3d_pretty(
    res_df: pd.DataFrame,
    log_mass=True,
    top_front_k=10,
    top_dom_k=None,   # None=标全部被支配点；否则只标前 k 个（按能耗降序）
    view_elev=26, view_azim=-50,
    dpi=150
):
    """
    需要的列：sensors, Energy_mAh_per_day, Mass_g, Price_usd, LCB95
    可选：ParetoFront（若缺失会自动计算）
    """
    df = res_df.copy()
    need_cols = ["sensors", "Energy_mAh_per_day", "Mass_g", "Price_usd", "LCB95"]
    # need_cols = ["sensors", "Energy_mAh_per_day", "Price_usd", "LCB95"]  # extend data
    # 1) 必须包含这些列
    missing = [c for c in need_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing column(s): {missing}")

    # 2) 这些列不允许出现 NaN/None
    nan_mask = df[need_cols].isna()
    if nan_mask.any().any():
        cols_with_nan = [c for c in need_cols if nan_mask[c].any()]
        # 列出各列中含 NaN 的行索引（可根据需要裁剪前若干个）
        where = {c: df.index[nan_mask[c]].tolist() for c in cols_with_nan}
        raise ValueError(f"NaN found in required columns. Columns={cols_with_nan}, rows={where}")

    # 3) 计算 Pareto 前沿
    if "ParetoFront" not in df.columns:  # double check
        # df["ParetoFront"] = compute_pareto_front(
        #     df, cost_cols=("Energy_mAh_per_day", "Mass_g"), perf_col="LCB95"
        # )
        df["ParetoFront"] = pareto_front(
            df,
            minimize=("Energy_mAh_per_day", "Mass_g", "Price_usd"),
            maximize=("LCB95",)
        )


    # 数据列
    E  = df["Energy_mAh_per_day"].to_numpy()
    M  = df["Mass_g"].to_numpy()
    F  = df["LCB95"].to_numpy()
    P  = df["Price_usd"].to_numpy()
    L  = df["sensors"].astype(str).to_numpy()
    is_front = df["ParetoFront"].to_numpy()

    # 坐标变换（压缩跨度）
    X = tf_log1p10(E)
    Y = tf_log1p10(M) if log_mass else M

    # 点大小：sqrt 压缩
    Pmax = P.max() if P.max() > 0 else 1.0
    size_dom = 14 + (np.sqrt(P/Pmax))*95
    size_fr  = 22 + (np.sqrt(P/Pmax))*130

    # 颜色
    C_DOM = "#60a5fa"   # 冷蓝（被支配）
    C_FR  = "#f59e0b"   # 暖橙（前沿）

    dom_mask = ~is_front
    fr_mask  =  is_front

    # 画图
    fig = plt.figure(figsize=(9,7), dpi=dpi)
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')         # 正交投影
    ax.view_init(elev=view_elev, azim=view_azim)

    # 面板与网格（淡）
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0.95))
    ax.grid(True, linewidth=0.6, alpha=0.18)
    # 更淡的点划线网格
    for axinfo in (ax.xaxis._axinfo, ax.yaxis._axinfo, ax.zaxis._axinfo):
        axinfo["grid"]['linestyle'] = (0, (1, 3))
        axinfo["grid"]['linewidth'] = 0.6
        axinfo["grid"]['color']     = (0.5, 0.55, 0.7, 0.25)

    # 被支配点（X）
    sc_dom = ax.scatter(X[dom_mask], Y[dom_mask], F[dom_mask],
                        s=size_dom[dom_mask], c=C_DOM, marker='x',
                        linewidths=1.3, alpha=0.95, label="Dominated combos (X)")
    # 前沿点（▲）
    sc_fr  = ax.scatter(X[fr_mask], Y[fr_mask], F[fr_mask],
                        s=size_fr[fr_mask], c=C_FR, marker='^',
                        edgecolor='k', linewidths=0.4, alpha=0.98,
                        label="Pareto front (▲)")

    # 前沿连线（按能耗从小到大）
    idx_fr = np.where(fr_mask)[0]
    idx_sort = idx_fr[np.argsort(E[fr_mask])]
    ax.plot(X[idx_sort], Y[idx_sort], F[idx_sort], color='k', lw=1.0, alpha=0.65)

    # 标注：前沿 Top-K，被支配（全部或 TopK by Energy）
    top_front_idx = idx_fr[np.argsort(F[fr_mask])[::-1][:top_front_k]]
    # print(top_front_idx)
    # annotate_points_3d(ax, X, Y, F, L, top_front_idx, color="#1f2937", fs=11, spread=0.018)
    # 只想标注所有前沿点：
    idx_fr = np.where(fr_mask)[0]
    annotate_points_3d(ax, X, Y, F, L, idx_fr, color="#1f2937", fs=11, spread=0.018)

    if top_dom_k is None:
        idx_dom_to_annot = np.where(dom_mask)[0]
    else:
        # 只取能耗最高的前 k 个被支配点，减少遮挡
        idx_dom = np.where(dom_mask)[0]
        idx_dom_to_annot = idx_dom[np.argsort(E[dom_mask])[::-1][:top_dom_k]]
        # print('a')
        # print(idx_dom_to_annot)
    # annotate_points_3d(ax, X, Y, F, L, idx_dom_to_annot, color="#2c5a7b", fs=12, spread=0.015)

    # 轴标签
    ax.set_xlabel("Energy (mAh/day)")
    ax.set_ylabel("Mass (g)")
    ax.set_zlabel("LCB95 (Macro-F1)")

    # 刻度：原始单位显示，但位置用 log1p 后的坐标
    xticks_raw = nice_log1p_ticks(E, k=6, min_gap=0.10)
    ax.set_xticks(tf_log1p10(xticks_raw))
    ax.set_xticklabels([f"{v:g}" for v in xticks_raw])

    if log_mass:
        yticks_raw = nice_log1p_ticks(M, k=5, min_gap=0.10)
        ax.set_yticks(tf_log1p10(yticks_raw))
        ax.set_yticklabels([f"{v:g}" for v in yticks_raw])

    ax.tick_params(axis='x', labelsize=9, pad=2)
    ax.tick_params(axis='y', labelsize=9, pad=2)
    ax.tick_params(axis='z', labelsize=9, pad=2)

    # 标题 & 图例
    ax.set_title("LCB95 vs Energy & Mass (marker size = Price)")
    legend_handles = [
        Line2D([0],[0], marker="^", color="w", label="Pareto front (▲)",
               markerfacecolor=C_FR, markeredgecolor="k", markersize=12, linewidth=0.0),
        Line2D([0],[0], marker="x", color=C_DOM, label="Dominated combos (X)",
               markersize=9, linewidth=1.5),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    # fig.tight_layout()
    plt.show()
    return fig, ax


#-------------------------------pareto------------------------------------

sensor_result_dict = get_supervised_results(len_sw, batch_size,
                                            num_epochs,
                                            all_sensor_list,
                                            device)

raw_map = {}   # sen_tag -> list of micro-F1
summary, summary_full = [], []   # [(sen_tag, mean, var, std), ...]
all_tags = []
means, stds = [], []
for Sen in all_sensor_list:
    tag = sensor_list_to_tag(Sen, compute_gps=True)
    sen_result_dict = sensor_result_dict[tag]
    model_weight_list = sen_result_dict['weight_list']
    vals = []
    tag = sensor_list_to_tag(Sen, compute_gps=True)  # compute_gps 来自你的管线配置
    for i in range(len(model_weight_list)):
        m_weight = model_weight_list[i]
        macro_f1, micro_f1 = run_eval(m_weight, len_sw, device, Sen)
        vals.append(float(micro_f1))
    raw_map[tag] = vals

    m = float(np.mean(vals))
    v = float(np.var(vals, ddof=1))   # 样本方差
    s = float(np.std(vals, ddof=1))   # 样本标准差
    summary.append((tag, m, v, s))  # 作条形图
    summary_full.append((tag, vals))
    all_tags.append(tag)
    means.append(m)
    stds.append(s)



rows = []
count = 1
for s_info in summary_full:
    s = tag2sensor[s_info[0]]
    rows.append({
        "combo_id": str(count),
        "sensors": s,
        ** {f"F1_run{i + 1}": v for i, v in enumerate(s_info[1])}
    })
    count += 1

perf_df = pd.DataFrame(rows, columns=list(rows[0].keys()))

# -------------------------------------------------------
# 4) Compute costs by summing per-sensor specs per combo;
#    compute LCB95 from the (placeholder) 5 F1 runs.
# -------------------------------------------------------
# helper: sum costs
cost_map = sensor_specs.set_index("sensor")[["energy_mAh_per_day",
                                             "mass_g", "price_usd", "volume_mm3"]].to_dict(
    "index")

records = []
for _, r in perf_df.iterrows():
    senl = r["sensors"].split("+")
    E, M, P, V = sum_costs(senl)
    runs = np.array([value for key, value in r.items() if 'F1_run' in key], dtype=float)
    fbar = runs.mean()
    s = runs.std(ddof=1)
    lcb = float(fbar - z_095 * s / np.sqrt(len(runs)))
    records.append({
        "combo_id": r["combo_id"],
        "sensors": r["sensors"],
        "Energy_mAh_per_day": E,
        "Mass_g": M,
        "Price_usd": P,
        "Volume_mm3": V,
        "Fbar": fbar,
        "Std": s,
        "LCB95": lcb,
        # "n_rep": n_rep
    })
res_df = pd.DataFrame(records)


# -----------------------------
# 使用示例：
# 假设你已有一个 res_df（如你上面打印的 DataFrame）
# 直接调用：
fig, ax = plot_pareto_3d_pretty(res_df, log_mass=True, top_front_k=0, top_dom_k=0)
print('b')
# -----------------------------