
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepview.calculate_results.models.utils import (
labeldict_findstr,
majority_value,
    labeldict_findstr,
)
from sklearn.metrics import confusion_matrix


root_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko'
def load_data(file_name):
    with open(os.path.join(root_path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data
# fig2的几个active learning比较

# active learning采用random sampling方法
base_acc = []
base_pred_label_lists, base_true_label_lists, base_selected_labels_list = [], [], []
base_ari_list, base_nmi_list, base_fmi_list = [], [], []
base_silhouette_list, base_label_data_list, base_unlabel_data_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__random_Contrast1_warm20_seed%d_epoch30_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    base_acc.append(result_dict['test_micro_f1_list'])


entropy_acc = []
entropy_pred_label_lists, entropy_true_label_lists, entropy_selected_labels_list = [], [], []
entropy_ari_list, entropy_nmi_list, entropy_fmi_list = [], [], []
entropy_silhouette_list, entropy_label_data_list, entropy_unlabel_data_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__entropy_Contrast1_warm20_seed%d_epoch30_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    entropy_acc.append(result_dict['test_micro_f1_list'])


underRandom_acc = []
underRandom_pred_label_lists, underRandom_true_label_lists, underRandom_selected_labels_list = [], [], []
underRandom_ari_list, underRandom_nmi_list, underRandom_fmi_list = [], [], []
underRandom_silhouette_list, underRandom_label_data_list, underRandom_unlabel_data_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__underRandom_Contrast1_warm20_seed%d_epoch30_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    underRandom_acc.append(result_dict['test_micro_f1_list'])


overAugment_acc = []
overAugment_pred_label_lists, overAugment_true_label_lists, overAugment_selected_labels_list = [], [], []
overAugment_ari_list, overAugment_nmi_list, overAugment_fmi_list = [], [], []
overAugment_silhouette_list, overAugment_label_data_list, overAugment_unlabel_data_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__overAugment_Contrast1_warm20_seed%d_epoch30_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    overAugment_acc.append(result_dict['test_micro_f1_list'])


# representative sampling
repreSamp_acc = []
repreSamp_pred_label_lists, repreSamp_true_label_lists, repreSamp_selected_labels_list = [], [], []
repreSamp_ari_list, repreSamp_nmi_list, repreSamp_fmi_list = [], [], []
repreSamp_silhouette_list, repreSamp_label_data_list, repreSamp_unlabel_data_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__repreSamp_Contrast1_warm20_seed%d_epoch30_results.pkl' % sid
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    repreSamp_acc.append(result_dict['test_micro_f1_list'])


entropy1_acc = []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__entropy1_Contrast1_warm20_seed%d_epoch20_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    entropy1_acc.append(result_dict['test_micro_f1_list'])

entropy2_acc = []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__entropy2_Contrast1_warm20_seed%d_epoch20_results.pkl' % sid
    # result_dict = load_data(file_name)
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
    entropy2_acc.append(result_dict['test_micro_f1_list'])


num_iterations = 20
# num_samples = 100  # 每次迭代预测的样本数
classes = [0, 1, 2, 3, 4, 5, 6]  # 5 个类别
# classes = [0, 1, 2, 3, 4]  # 5 个类别

# 计算 F1 Score
f1_scores = {cls: [] for cls in classes}  # 每个类别的 F1-score
std_devs = {cls: [] for cls in classes}  # 每个类别的 F1-score

for i in range(num_iterations):
    for cls in classes:
        if cls == 0:
            f1scores = []
            for sid in range(5):
                f1scores.append(entropy_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        elif cls == 3:
            f1scores = []
            for sid in range(5):
                f1scores.append(base_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        elif cls == 4:
            f1scores = []
            for sid in range(5):
                f1scores.append(underRandom_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        elif cls == 5:
            f1scores = []
            for sid in range(5):
                f1scores.append(overAugment_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        elif cls == 6:
            f1scores = []
            for sid in range(5):
                f1scores.append(repreSamp_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        if cls == 1:
            f1scores = []
            for sid in range(5):
                f1scores.append(entropy1_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        if cls == 2:
            f1scores = []
            for sid in range(5):
                f1scores.append(entropy2_acc[sid][i])
            avg_f = np.average(f1scores)
            sem_f = np.std(f1scores) / np.sqrt(len(f1scores) + 1e-9)
            f1_scores[cls].append(avg_f * 100)
            std_devs[cls].append(sem_f * 100)
        else:
            continue
            print('error')


# 画图
# fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

# 颜色匹配图例
colors = {
    0: "#5470C6",  # 深蓝 - Attack
    1: "#91CC75",  # 绿色 - Investigation
    2: "#FAC858",  # 金黄 - Mount
    3: "#EE6666",  # 红色 - Category 3
    4: "#73C0DE",  # 天蓝 - Category 4
    5: "#FFB74D",  # 橙色 - Category 5
    6: "#9C27B0",  # 紫色 - Category 6
}

labels = {0: 'Entropy Sampling',
          3: 'Random Sampling',
          4: 'Under Sampling',
          5: 'Over Sampling',
          6: 'Representative Sampling',
          1: 'Least_confidence Sampling',
          2: 'Min_margin Sampling',
          7: 'Supervised Learning Baseline',
          }

# 绘制每个类别的曲线
for cls in f1_scores:
    ax.plot(range(1, num_iterations + 1), f1_scores[cls],
            color=colors[cls], linestyle="-", label=labels[cls], linewidth=2)

    # 添加标准误差的阴影区域
    ax.fill_between(range(1, num_iterations + 1),
                    np.array(f1_scores[cls]) - std_devs[cls],
                    np.array(f1_scores[cls]) + std_devs[cls],
                    color=colors[cls], alpha=0.1)

# 标注轴标签
ax.set_xlabel("Iteration no.", fontsize=14)
ax.set_ylabel("F1 Score (%)", fontsize=14)
# ax.set_xticklabels([])  # 隐藏 X 轴标签
# ax.set_xticks([1, 5, 10, 15, 20, 25, 30])  # 设置刻度位置
# ax.set_xticklabels(["1", "5", "10", "15", "20", "25", "30"])  # 设置刻度标签
ax.set_xticks([1, 5, 10, 15, 20])  # 设置刻度位置
ax.set_xticklabels(["1", "5", "10", "15", "20"])  # 设置刻度标签
#
# 添加每个类别的虚线
file_name = 'AccelTemp__supervisebase_seed2025_epoch49_results.pkl'  # supervised learning baseline
# with open(os.path.join(root_path, file_name), 'rb') as f:
#     pickle.load(f)
#     pickle.load(f)
#     pickle.load(f)
#     pickle.load(f)
#     pickle.load(f)
#     result_dict = pickle.load(f)
result_dict = load_data(file_name)
supervise_acc = result_dict['test_micro_f1_list']
ax.axhline(y=supervise_acc[-1]*100,
           color='grey', linestyle="--", label=labels[7], linewidth=1.5)

# 网格线优化
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="gray")
ax.set_xlim(1, num_iterations)
ax.set_ylim(75, 96)
# 添加图例
ax.legend(frameon=False)

# 保存高清图片
output_path_lines = "F1_activelearning_iter%s_class%s.png"%(str(num_iterations), str(len(classes)))
plt.savefig(output_path_lines, bbox_inches="tight", dpi=300)

# 显示图表
# plt.show()

