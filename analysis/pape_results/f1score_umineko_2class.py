
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
from sklearn.metrics import f1_score


root_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko'
def load_data(file_name):
    with open(os.path.join(root_path, file_name), 'rb') as f:
        data = pickle.load(f)
    return data


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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    base_acc.append(result_dict['test_micro_f1_list'])
    base_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    base_true_label_lists.append(result_dict2['test_truth_label_lists'])


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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    entropy_acc.append(result_dict['test_micro_f1_list'])
    entropy_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    entropy_true_label_lists.append(result_dict2['test_truth_label_lists'])


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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    underRandom_acc.append(result_dict['test_micro_f1_list'])
    underRandom_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    underRandom_true_label_lists.append(result_dict2['test_truth_label_lists'])


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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    overAugment_acc.append(result_dict['test_micro_f1_list'])
    overAugment_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    overAugment_true_label_lists.append(result_dict2['test_truth_label_lists'])


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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    repreSamp_acc.append(result_dict['test_micro_f1_list'])
    repreSamp_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    repreSamp_true_label_lists.append(result_dict2['test_truth_label_lists'])


entropy1_acc = []
entropy1_pred_label_lists, entropy1_true_label_lists, entropy1_selected_labels_list = [], [], []
for sid in [1, 2, 5, 10, 2025]:
    file_name = 'AccelTemp__entropy1_Contrast1_warm20_seed%d_epoch20_results.pkl' % sid
    with open(os.path.join(root_path, file_name), 'rb') as f:
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        pickle.load(f)
        result_dict = pickle.load(f)
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    entropy1_acc.append(result_dict['test_micro_f1_list'])
    entropy1_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    entropy1_true_label_lists.append(result_dict2['test_truth_label_lists'])

entropy2_acc = []
entropy2_pred_label_lists, entropy2_true_label_lists, entropy2_selected_labels_list = [], [], []
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
        result_dict1 = pickle.load(f)
        result_dict2 = pickle.load(f)
    entropy2_acc.append(result_dict['test_micro_f1_list'])
    entropy2_pred_label_lists.append(result_dict1['test_pred_label_lists'])
    entropy2_true_label_lists.append(result_dict2['test_truth_label_lists'])


num_iterations = 20
# num_samples = 100  # 每次迭代预测的样本数
# classes = [0, 1, 2, 3, 4, 5]  # 6 个类别
classes = [0, 1, 2, 3, 4]  # 5 个类别
# methods = [0, 1]  # 5 个类别


labels = labeldict_findstr


# 计算 F1 Score
f1_scores = {cls: [] for cls in classes}  # 每个类别的 F1-score
std_devs = {cls: [] for cls in classes}  # 每个类别的 F1-score
for i in range(num_iterations):
    for cls in classes: # number of classes
        tmpscore = []
        for sid in range(5):  # random seeds
            y_pred = entropy_pred_label_lists[sid][i]
            y_true = entropy_true_label_lists[sid][i]
            if cls in y_true:  # 确保类别存在
                f1 = f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
            else:
                f1 = 0.0  # 避免计算错误
            tmpscore.append(f1)
        avg_f = np.average(tmpscore)
        sem_f = np.std(tmpscore) / np.sqrt(len(tmpscore) + 1e-9)
        f1_scores[cls].append(avg_f * 100)
        std_devs[cls].append(sem_f * 100)


f1_scores2 = {cls: [] for cls in classes}  # 每个类别的 F1-score
std_devs2 = {cls: [] for cls in classes}  # 每个类别的 F1-score
for i in range(num_iterations):
    for cls in classes: # number of classes
        tmpscore = []
        for sid in range(5):  # random seeds
            y_pred = overAugment_pred_label_lists[sid][i]
            y_true = overAugment_true_label_lists[sid][i]
            if cls in y_true:  # 确保类别存在
                f1 = f1_score(y_true, y_pred, labels=[cls], average="macro", zero_division=0)
            else:
                f1 = 0.0  # 避免计算错误
            tmpscore.append(f1)
        avg_f = np.average(tmpscore)
        sem_f = np.std(tmpscore) / np.sqrt(len(tmpscore) + 1e-9)
        f1_scores2[cls].append(avg_f * 100)
        std_devs2[cls].append(sem_f * 100)

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


# 绘制每个类别的曲线
for cls in f1_scores:
    ax.plot(range(1, num_iterations + 1), f1_scores[cls],
            color=colors[cls], linestyle="-", label=labels[cls]+str(' (Entropy)'), linewidth=2)
    # 添加标准误差的阴影区域
    ax.fill_between(range(1, num_iterations + 1),
                    np.array(f1_scores[cls]) - std_devs[cls],
                    np.array(f1_scores[cls]) + std_devs[cls],
                    color=colors[cls], alpha=0.15)
for cls in f1_scores2:
    ax.plot(range(1, num_iterations + 1), f1_scores2[cls],
            color=colors[cls], linestyle="--", label=labels[cls]+str(' (Over)'), linewidth=1.5)
    # 添加标准误差的阴影区域
    ax.fill_between(range(1, num_iterations + 1),
                    np.array(f1_scores2[cls]) - std_devs2[cls],
                    np.array(f1_scores2[cls]) + std_devs2[cls],
                    color=colors[cls], alpha=0.08)

# # 标注轴标签
# ax.set_xlabel("Iteration no.", fontsize=14)
# ax.set_ylabel("F1 Score (%)", fontsize=14)
# ax.set_xticks([1, 5, 10, 15, 20])  # 设置刻度位置
# ax.set_xticklabels(["1", "5", "10", "15", "20"])  # 设置刻度标签
# # 网格线优化
# ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="gray")
# ax.set_xlim(1, num_iterations)
# ax.set_ylim(10, 100)
# # 添加图例
# ax.legend(frameon=False)

# 图表美化
ax.set_xlabel("Iteration no.", fontsize=14)
ax.set_ylabel("F1 Score (%)", fontsize=14)
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xticklabels(["1", "5", "10", "15", "20"])  # 设置刻度标签
ax.set_xlim(1, num_iterations)
ax.set_ylim(10, 100)
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7, color="gray")

# 图例美化
ax.legend(frameon=False, fontsize=10, ncol=2)
plt.tight_layout()


# 保存高清图片
output_path_lines = "F1_activelearning_iter%s_2class%s.png"%(str(num_iterations), str(len(classes)))
plt.savefig(output_path_lines, bbox_inches="tight", dpi=300)

# 显示图表
plt.show()

