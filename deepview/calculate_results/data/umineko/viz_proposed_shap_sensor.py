# 通过active_supContrast.py获得模型weight之后，计算每次iteration（1,10,20）的shap
## input multi modality的重要性
## 每个input对模型的重要性
### 参考model_shap.py


from scipy import stats
import pandas as pd
import seaborn as sns

import shap
import matplotlib.pyplot as plt

import os
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import scipy.stats as stats
import copy

from deepview.calculate_results.models.utils import (
    data_loader_umineko,
    AE_eval_time_series,
    majority_value,
    sliding_window,
    read_sensor_data,
    process_acc,
    process_acc_press,
    process_acc_temperature,
    process_acc_gyr,
    process_press_temperature,
    labeldict_findstr
)
from deepview.calculate_results.data.umineko.train_func import (
    train_model,
    evaluate_model,
    evaluate_supContrast_model,
    plot_confusion_matrix,
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_1s,
    SimpleNN_13s,
    SimpleNN_13s_SHAP,SimpleNN_13s_SHAP_clean,
    SimpleNN_11s,
    SimpleNN_33s,
    ContrastiveLoss,
    SupContrastiveLoss,
)
from deepview.calculate_results.data.umineko.active_utils import (
    uncertainty_sampling
)
from deepview.calculate_results.data.umineko.cluster_method import (
    vis_scatter_label_2d,
)


# sensor_type = 'Accel'
sensor_type = 'AccelPress'
# sensor_type = 'AccelTemp'
# sensor_type = 'AccelGyro'
# sensor_type = 'PressTemp'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
len_sw = 50
batch_size = 4000
name_label = '_entropy_Contrast%s_warm%s'%(str(1), str(20))


# ======= 1. 数据生成 ======= #
if os.path.exists(sensor_type+'_data.pkl'):
    with open(sensor_type+'_data.pkl', 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
        d_dict = pickle.load(file)
        data_b = d_dict['data']
        label_b = d_dict['label']
else:
    all_df = read_sensor_data()
    selected_df = all_df[all_df.label_id != -2]

    if (sensor_type == 'Accel'):
        columns = ['acc_x', 'acc_y', 'acc_z', 'label_id']
        selected_np = process_acc(selected_df, columns)
    elif (sensor_type == 'AccelPress'):
        columns = ['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']
        selected_np = process_acc_press(selected_df, columns)
    elif (sensor_type == 'AccelTemp'):
        columns = ['acc_x', 'acc_y', 'acc_z', 'temperature', 'label_id']
        selected_np = process_acc_temperature(selected_df, columns)
    elif (sensor_type == 'AccelGyro'):
        columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label_id']
        selected_np = process_acc_gyr(selected_df, columns)
    elif (sensor_type == 'PressTemp'):
        columns = ['pressure', 'temperature', 'label_id']
        selected_np = process_press_temperature(selected_df, columns)
    else:
        print('no model available')

    tmp_b = sliding_window(selected_np, len_sw, len_sw)
    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    # 将数据保存到文件
    with open(sensor_type+'_data.pkl', 'wb') as file:  # 'wb'表示以二进制写入模式打开文件
        pickle.dump({'data': data_b, 'label': label_b}, file)

# 将数据分为8:2，其中2为测试集
vote_label = majority_value(label_b)
X_train_full, X_test, y_train_full, y_test = train_test_split(data_b, label_b,
                                                                  test_size=0.2, stratify=vote_label,
                                                                  random_state=42)
# 从X_train_full中随机选择1%的数据作为X_labeled，其余为X_unlabeled
X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(X_train_full, y_train_full,
                                                                  test_size=0.99, random_state=42)

# 确保X_labeled和X_unlabeled的大小
print("X_labeled shape:", X_labeled.shape)
print("X_unlabeled shape:", X_unlabeled.shape)
print("X_test shape:", X_test.shape)


# ======= 2. 模型初始化 ======= #
# 初始化模型和优化器
if (sensor_type == 'Accel'):
    model = SimpleNN_1s()
elif (sensor_type == 'AccelPress') or (sensor_type == 'AccelTemp'):
    model = SimpleNN_13s_SHAP()
    model_clean = SimpleNN_13s_SHAP_clean()
elif (sensor_type == 'AccelGyro'):
    model = SimpleNN_33s()
elif (sensor_type == 'PressTemp'):
    model = SimpleNN_11s()
else:
    print('no model available')
model = model.to(device)
model_clean = model_clean.to(device)
classify_criterion = nn.CrossEntropyLoss()
Contrast_criterion = ContrastiveLoss()
supContrast_criterion = SupContrastiveLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

def unfreeze_encoders(model):
    # 处理模型的所有参数
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True

def freeze_encoders(model):
    # 处理模型的所有参数
    for param in model.parameters():
        param.requires_grad = True
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def copy_model(model_pre, model_new):
    # 1. 提取两个模型的 state_dict
    state_dict_a = model_pre.state_dict()
    state_dict_b = model_new.state_dict()

    # 2. 遍历 B 的所有参数，如果在 A 中找到同名且形状一致，就复制
    for name, param_b in state_dict_b.items():
        if name in state_dict_a:
            param_a = state_dict_a[name]
            # 检查一下形状是否一致
            if param_a.shape == param_b.shape:
                # 复制权重
                state_dict_b[name].copy_(param_a)
                # print(f"Copied weights for layer: {name}")
            else:
                print(f"Skip layer {name}: shape mismatch {param_a.shape} vs {param_b.shape}")
        else:
            print(f"Skip layer {name}: not found in model_a")

    # 3. 用更新过的 state_dict_b 回载到 model_b
    model_new.load_state_dict(state_dict_b)


# ======= 3. 计算SHAP ======= #
def plot_shap2(df):
    # 创建合并数据框
    merged_data = []
    sensors = df.sensor_label.unique()
    labels = df.main_label.unique()
    for label in labels:
        group_sensorA = df[(df['main_label'] == label) & (df['sensor_label'] == sensors[0])]
        group_sensorB = df[(df['main_label'] == label) & (df['sensor_label'] == sensors[1])]
        left_values = group_sensorA['value'].reset_index(drop=True)  # 取(1)的值
        right_values = group_sensorB['value'].reset_index(drop=True)  # 取(2)的值
        print('label %d is %s and %s'%(label, np.mean(left_values), np.mean(right_values)))

        # 添加左侧数据
        left_df = pd.DataFrame({
            'label': [label] * len(left_values),
            'value': left_values,
            'side': 'left'
        })

        # 添加右侧数据
        right_df = pd.DataFrame({
            'label': [label] * len(right_values),
            'value': right_values,
            'side': 'right'
        })

        # 使用 pd.concat() 合并
        merged_data.append(pd.concat([left_df, right_df], ignore_index=True))

    # 将所有合并的数据框合并为一个
    plot_data = pd.concat(merged_data, ignore_index=True)

    # 绘制小提琴图
    plt.figure(figsize=(14, 6))  # 增加画布大小

    # 绘制小提琴图，调整宽度
    sns.violinplot(x='label', y='value', hue='side', data=plot_data,
                   palette={'left': 'skyblue', 'right': 'lightcoral'},
                   split=True, inner=None, linewidth=1.2, width=0.9)

    # 计算均值并添加
    means = plot_data.groupby(['label', 'side'])['value'].mean().reset_index()
    for _, row in means.iterrows():
        # 计算置信区间
        ci = (
                stats.sem(
                    plot_data[
                        (plot_data['label'] == row['label']) & (plot_data['side'] == row['side'])
                        ]['value']
                )
                * stats.t.ppf(0.975, len(plot_data) - 1)
        )
        x_position = row['label'] + (-0.1 if row['side'] == 'left' else 0.1)
        y_position = row['value']

        # 使用 errorbar 绘制均值点和误差棒，并自定义美观样式
        plt.errorbar(
            x_position, y_position,
            yerr=ci,
            fmt='o',  # 使用圆圈标记
            ecolor='black',  # 误差棒颜色
            elinewidth=1.2,  # 误差棒线宽
            capsize=5,  # 误差棒两端横杠大小
            markersize=10,  # 圆圈大小
            markeredgecolor='black',  # 圆圈边框颜色
            markerfacecolor='white',  # 圆圈填充颜色
            markeredgewidth=1.5,  # 圆圈边缘线宽
            zorder=5  # 控制图层顺序，让它覆盖在小提琴图上
        )

    # 可选：给均值打上数值标签
    for _, row in means.iterrows():
        # 查找对应 x 坐标索引
        x_indices = np.where(labels == row['label'])[0]
        if len(x_indices) == 1:
            x_index = x_indices[0]
        x_offset = -0.1 if row['side'] == 'left' else 0.1
        plt.text(
            x_index + x_offset,
            row['value'] + (0.03 if row['side'] == 'left' else -0.03),
            f"{row['value']:.2f}",
            color='black',
            ha='center',
            fontsize=10
        )

    # 后续标题、坐标轴、图例等与原先一致
    plt.title('Violin Plot of Merged Values by Label with Mean and CI', fontsize=14)
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Side', loc='upper right', labels=['(1)', '(2)'], fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    plt.show()


def plot_shap(df, name_label, columns):
    # ============= 1. 数据整理 =============
    merged_data = []
    sensors = df.sensor_label.unique()      # 假设只有两种传感器，对应 (1) 与 (2)
    labels = df.main_label.unique()         # Activity 标签（如 0,1,2,3,4 等）

    for label in labels:
        group_sensorA = df[(df['main_label'] == label) & (df['sensor_label'] == sensors[0])]
        group_sensorB = df[(df['main_label'] == label) & (df['sensor_label'] == sensors[1])]
        left_values  = group_sensorA['value'].reset_index(drop=True)
        right_values = group_sensorB['value'].reset_index(drop=True)

        # 分别构建左右数据(这里仍然用 side='left' / 'right' 方便后面用调色板区分)
        left_df = pd.DataFrame({
            'label': [label]*len(left_values),
            'value': left_values,
            'side': 'left'
        })
        right_df = pd.DataFrame({
            'label': [label]*len(right_values),
            'value': right_values,
            'side': 'right'
        })
        merged_data.append(pd.concat([left_df, right_df], ignore_index=True))

    plot_data = pd.concat(merged_data, ignore_index=True)

    # ============= 2. 设置绘图风格 =============
    sns.set_style("ticks")                 # 去掉背景网格，仅保留刻度
    sns.set_context("paper", font_scale=1.3)  # “paper”风格下字体大小适中
    plt.figure(figsize=(10, 4), dpi=300)    # Nature 常见较小尺寸，高分辨率

    # ============= 3. 绘制小提琴图 =============
    # 自定义调色板：左传感器（Accel）为 skyblue，右传感器（Pressure）为 tomato
    palette = {'left': 'skyblue', 'right': 'tomato'}

    ax = sns.violinplot(
        x='label', y='value', hue='side', data=plot_data,
        palette=palette, split=True, inner=None, linewidth=1.0, width=0.8
    )

    # # ============= 4. 计算并绘制均值与置信区间 =============
    means = plot_data.groupby(['label', 'side'])['value'].mean().reset_index()
    for _, row in means.iterrows():
        # 计算 95% 置信区间
        ci = (
            stats.sem(
                plot_data[(plot_data['label'] == row['label'])
                          & (plot_data['side'] == row['side'])]['value']
            ) * stats.t.ppf(0.975, len(plot_data) - 1)
        )
        # 为左右传感器作 x 方向微调，以防止两点重叠
        x_offset = -0.2 if row['side'] == 'left' else 0.2
        x_pos = row['label'] + x_offset
        y_pos = row['value']

        # errorbar 可以一次性画圆圈(均值) + 误差棒
        plt.errorbar(
            x_pos, y_pos, yerr=ci,
            fmt='o',
            ecolor='black',       # 误差棒颜色
            elinewidth=1.2,
            capsize=3,            # 端点横杠长度
            markersize=6,         # 圆圈大小
            markeredgecolor='black',
            markerfacecolor='white',
            markeredgewidth=1.2,
            zorder=5
        )

    # ============= 5. 给均值加数值标签 (可选) =============
    for _, row in means.iterrows():
        x_offset = -0.2 if row['side'] == 'left' else 0.2
        x_indices = np.where(labels == row['label'])[0]
        if len(x_indices) == 1:
            x_index = x_indices[0]
        # 在圆圈上方或下方做微调
        y_text_offset = 0.01 if row['side'] == 'left' else -0.03
        plt.text(
            row['label'] + x_offset,
            row['value'] + y_text_offset,
            f"{row['value']:.2f}",
            ha='center', va='bottom' if row['side']=='left' else 'top',
            fontsize=9, color='black'
        )

    # ============= 6. 调整横坐标标签 (数字映射为字符串) =============
    # 若你的 labels 为散乱数字，需自行排序，这里假设从小到大排序：
    sorted_labels = sorted(labels)
    new_labels = [labeldict_findstr[l] for l in sorted_labels]  # 映射为字符串
    ax.set_xticks(range(len(sorted_labels)))
    ax.set_xticklabels(new_labels, rotation=0)

    # ============= 7. 设置坐标轴和图例等 =============
    # ax.set_xlabel("Activity", fontsize=12)
    ax.set_ylabel("SHAP value", fontsize=12)
    ax.set_title("Violin Plot of SHAP Values by Activity", fontsize=14, pad=10)

    # 调整图例：把 side => sensor, 并自定义标签顺序
    # sns.violinplot 的图例里，“left”/“right”顺序并不一定；这里手动加 legend
    # 如果想复用 Seaborn 自动生成的也可以，但要做点 hack 来改名字
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles, columns,     # 对应 'left', 'right' 的顺序
        title="Sensor", frameon=False, loc='upper right'
    )

    # 去除顶部和右侧脊线，更加简洁
    sns.despine(trim=True)

    plt.tight_layout()
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'figures/SHAP_sample_{name_label}.png', bbox_inches='tight')
    # plt.show()

def bee_swarm_plot(shap, shap_values, columns, name_label):
    for i in range(shap_values.shape[-1]):
        # plt.figure(figsize=(3, 6))  # 调整画布大小
        # # 取出第 i 个“通道”的 SHAP 值：形状 (batch, 50, 5)
        # shap_values_i = shap_values[:, :, :, i]
        # # 若这是多分类问题，可对最后一维求和，得到 (batch, 50)
        # shap_values_i_sum = shap_values_i.sum(axis=-1)
        # # 同理，取出第 i 个“通道”的特征数据：形状 (batch, 50)
        # test_data_i = test_data[:, :, i]
        # # 直接画 summary plot
        # shap.summary_plot(shap_values_i_sum,
        #                   test_data_i,
        #                   feature_names=columns,
        #                   plot_size=(3, 6),
        #                   show=False)

        # or
        plt.figure(figsize=(3, 3))  # 调整画布大小
        # 取出第 i 个“通道”的 SHAP 值：形状 (batch, 50, 5)
        shap_values_i = shap_values[:, :2, :, i]
        shap_values_i[:, 0, :] = np.sum(shap_values[:, :3, :, i], axis=1)
        shap_values_i[:, 1, :] = shap_values[:, 3:, :, i].sum(axis=1)
        # 若这是多分类问题，可对最后一维求和，得到 (batch, 50)
        shap_values_i_sum = shap_values_i.sum(axis=-1)
        # 同理，取出第 i 个“通道”的特征数据：形状 (batch, 50)
        test_data_i = test_data[:, :2, i]
        test_data_i[:, 0] = test_data[:, :3, i].sum(dim=1)
        test_data_i[:, 1] = test_data[:, 3:, i].sum(dim=1)
        # 直接画 summary plot
        shap.summary_plot(shap_values_i_sum,
                          test_data_i,
                          feature_names=columns,
                          plot_type='dot',
                          color_bar=False,
                          plot_size=(3, 3),
                          show=False)

        # 2) 调整布局，防止标签等被截掉
        plt.tight_layout()
        ax = plt.gca()
        # 若 y 轴上本来有 4 个特征，可以这么改：
        ax.set_yticks([0, 1])
        # ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(columns)
        # 3) 保存为 PNG 或 PDF
        # plt.savefig("shap_summary.png", dpi=300)  # 300dpi 通常分辨率就够
        plt.savefig(f'figures/beeSHAP2_{name_label}_act{i}.png', bbox_inches='tight')
    return

def calculate_shap(X_labeled, test_data, model, model_clean, sensor_type, name_label):
    model_clean.eval()
    copy_model(model, model_clean)
    # model_cpu = copy.deepcopy(model_clean).to("cpu")
    if (sensor_type == 'AccelPress'):
        sensor1 = 'Accelerometer'
        sensor2 = 'Pressure'
        columns = ['Accelerometer', 'Pressure']
        # columns = ['Accel_x', 'Accel_y', 'Accel_z', 'Pressure']
    elif (sensor_type == 'AccelTemp'):
        sensor1 = 'Accelerometer'
        sensor2 = 'Temperature'
        columns = ['Accelerometer', 'Temperature']
        # columns = ['Accel_x', 'Accel_y', 'Accel_z', 'Temperature']
    elif (sensor_type == 'AccelGyro'):
        sensor1 = 'Accelerometer'
        sensor2 = 'Gyroscope'
        columns = ['Accelerometer', 'Gyroscope']
        # columns = ['Accel_x', 'Accel_y', 'Accel_z', 'Gyro_x', 'Gyro_y', 'Gyro_z']
    elif (sensor_type == 'PressTemp'):
        sensor1 = 'Pressure'
        sensor2 = 'Temperature'
        columns = ['Pressure', 'Temperature']
    else:
        print('error at ploting shap values, no such sensor.')

    # ============ 4. 计算 SHAP 值 ============
    # SHAP 解释器 (使用 Kernel SHAP)
    explainer = shap.GradientExplainer(model_clean, X_labeled)
    shap_values = explainer.shap_values(test_data)  # 计算 SHAP 值
    bee_swarm_plot(shap, shap_values, columns, name_label)

    # # ============ 5. SHAP 重要性分析 ============
    # plot_violin_df = pd.DataFrame(columns=['main_label', 'sensor_label', 'value'])
    # for activity_id in range(shap_values.shape[-1]):  # every class
    #     # activity_str = labeldict_findstr[activity_id]
    #     shap_vals_for_class0 = shap_values[:,:,:,activity_id]  # shape: (batch_test, 4, 50)
    #
    #     if (sensor_type == 'AccelPress'):
    #         acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
    #         press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
    #     elif (sensor_type == 'AccelTemp'):
    #         acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
    #         press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
    #     elif (sensor_type == 'AccelGyro'):
    #         acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
    #         press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
    #     elif (sensor_type == 'PressTemp'):
    #         acc_shap_vals = shap_vals_for_class0[:, :1, :]  # (batch_test, 3, seq_len)
    #         press_shap_vals = shap_vals_for_class0[:, 1:, :]  # (batch_test, 1, seq_len)
    #     else:
    #         print('error at ploting shap values, no such sensor.')
    #
    #     # 计算所有数的均值
    #     acc_shap_mean = np.mean(acc_shap_vals, axis=(1,2))
    #     press_shap_mean = np.mean(press_shap_vals, axis=(1,2))
    #     # 组合标签，包括行为标签一列和sensor标签一列
    #
    #     test_label = [activity_id] * len(acc_shap_mean) * 2
    #     sensor_label_str = [sensor1]*len(acc_shap_mean) + [sensor2]*len(press_shap_mean)
    #     tmp_data = pd.DataFrame({
    #         'main_label': test_label,
    #         'sensor_label': sensor_label_str,
    #         'value': np.concatenate([acc_shap_mean, press_shap_mean])
    #     })
    #     # 使用 append 将新数据添加到原 DataFrame
    #     plot_violin_df = pd.concat([plot_violin_df, tmp_data], ignore_index=True)
    # plot_shap(plot_violin_df, name_label, columns)

    return



############################################
# 训练模型，并在epoch=1，10，20时计算sensor 的 SHAP值
# 主动学习迭代
update_size = max(int(0.01 * len(X_train_full)), 2)  # 每次选择 1% 的样本
batch_size = 4000
iteration = 1
warmup = 20
train_accuracy_list, train_macro_f1_list, train_micro_f1_list = [], [], []
test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
test_pred_label_lists, test_truth_label_lists = [], []
weight_list, loss_list, selected_labels_list = [], [], []
while len(X_unlabeled) > 0:
    print(f"==============Iteration {iteration}===============")
    print(f"Already labeled samples: {len(X_labeled)}")
    print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

    # 创建数据加载器
    labeled_dataset = data_loader_umineko(X_labeled.astype(float), y_labeled.astype(int))
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ## stage 1: train supCon model (所有数据都来训练)
    unfreeze_encoders(model)
    if iteration < warmup:
        epoch = 10
        model, _ = train_model(model, labeled_loader, supContrast_criterion, optimizer,
                                      epochs=epoch, device=device, if_contrast=True)

    ## stage 2: train supervised model, 获得分类器新参数
    freeze_encoders(model)  # only classifier is trainable
    model, avg_loss1 = train_model(model, labeled_loader, classify_criterion, optimizer,
                                  epochs=50, device=device, if_contrast=False)

    unfreeze_all(model)  # only projector is NOT trainable
    model, avg_loss2 = train_model(model, labeled_loader, classify_criterion, optimizer,
                            epochs=50, device=device, if_contrast=False)

    loss_list.append(np.average(avg_loss2))  # 记录50次epoch的平均loss

    # ======= 6. 模型测试 ======= #
    print('-------Training----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = (
        evaluate_model(model, labeled_loader, device))
    train_accuracy_list.append(accuracy)
    train_macro_f1_list.append(macro_f1)
    train_micro_f1_list.append(micro_f1)

    test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print('-------Test----------')
    (accuracy, macro_f1, micro_f1,
     rest_datan, pred_label_list, truth_label_list) = (
        evaluate_supContrast_model(model, test_loader, device, cal_threshold=0.5))
    test_accuracy_list.append(accuracy)
    test_micro_f1_list.append(micro_f1)
    test_macro_f1_list.append(macro_f1)
    test_pred_label_lists.append(pred_label_list)
    test_truth_label_lists.append(truth_label_list)

    # ======= 7. 计算SHAP ======= #
    if (iteration == 1)  or (iteration == 20)  or (iteration == 60):
        # 随机选择 10 个样本的索引
        random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
        # 获取随机选择的样本
        test_data = X_test[random_indices]
        test_data = torch.tensor(test_data).to(device=device, dtype=torch.float)
        train_data = torch.tensor(X_labeled).to(device=device, dtype=torch.float)
        label_vote = majority_value(y_test)
        test_label = label_vote[random_indices]
        calculate_shap(train_data, test_data,
                       model, model_clean,
                       sensor_type, sensor_type+name_label+'_iter'+str(iteration))

    # ======= 8. 保存变量 ======= #
    # 保存当前模型权重到列表
    weight_list.append(model.state_dict())

    # # stage 3: 更新数据集
    select_size = min(update_size, len(X_unlabeled))

    ### uncertainty_sampling
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels = (
        uncertainty_sampling(X_labeled, y_labeled,
                             X_unlabeled, y_unlabeled,
                             model, select_size, device))  # data变化了
    selected_labels_list.append(selected_labels)

    if iteration == 60:
        break
    iteration += 1

