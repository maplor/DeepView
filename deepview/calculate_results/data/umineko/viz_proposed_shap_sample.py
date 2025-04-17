# proposed method: entropy + Contrastive loss
# 当前文件用于计算 proposed method 的每个sample的 SHAP 值


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
def plot_shap(data, shap_value, test_label_flatten, name_label, sensor_type, activity_id):
    # 标准化重要程度为颜色值（0-1之间）
    # normed_importance 计算
    # 归一化 SHAP
    # normed_importance = (
    #         (shap_value - np.min(shap_value)) /
    #         (np.max(shap_value) - np.min(shap_value))
    # )
    cmap = plt.get_cmap('viridis')
    colors = cmap(shap_value)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    # 顶部子图
    for i in range(len(data) - 1):
        for j in range(data.shape[1]):
            ax1.plot([i, i + 1], [data[i, j], data[i + 1, j]],
                     color=colors[i], linewidth=1)

    ax1.set_ylabel('Sensor Value')
    ax1.set_title(f'Sensor {sensor_type} Colored by SHAP of {activity_id}')

    # 底部子图
    ax2.plot(test_label_flatten, color='blue', linewidth=1, label='Groundtruth Activity Label')
    # ax2.set_ylabel('Test Label Value')
    ax2.set_title('Activity Label Over Time')
    ax2.legend()

    # 将数值标签替换为字符串标签
    ax2.set_yticks(list(labeldict_findstr.keys()))
    ax2.set_yticklabels(list(labeldict_findstr.values()))

    # 让两张图共用同样的 x 轴范围
    ax1.set_xlim([0, len(test_label_flatten) - 1])

    # 独立的 colorbar 轴
    norm = plt.Normalize(np.min(shap_value), np.max(shap_value))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.92, 0.55, 0.02, 0.33])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('Importance Level')

    # 调整图像边距，避免左边文字被截断
    plt.subplots_adjust(left=0.2, right=0.9, hspace=0.3)
    # 或者使用 tight_layout
    # plt.tight_layout(rect=[0.15, 0, 0.9, 1])  # 留更多左边空间

    # plt.tight_layout()
    # plt.xticks(rotation=45, ha='right')
    plt.savefig(f'figures/SHAP_sample_{name_label}_{sensor_type}_{activity_id}.png', bbox_inches='tight')
    # # plt.show()

    return

def calculate_shap(X_labeled, test_data, test_label,
                   model, model_clean, sensor_type, name_label):
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

    # # ============ 5. SHAP 重要性分析 ============
    plot_violin_df = pd.DataFrame(columns=['main_label', 'sensor_label', 'value'])
    for activity_id in range(shap_values.shape[-1]):  # every class
        # activity_str = labeldict_findstr[activity_id]
        shap_vals_for_class0 = shap_values[:,:,:,activity_id]  # shape: (batch_test, 4, 50)

        if (sensor_type == 'AccelPress'):
            acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
            acc_test = test_data[:, :3, :].detach().cpu().numpy()  # (batch_test, 3, seq_len)
            press_test = test_data[:, 3:, :].detach().cpu().numpy()  # (batch_test, 3, seq_len)
            press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
        elif (sensor_type == 'AccelTemp'):
            acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
            acc_test = test_data[:, :3, :].detach().cpu().numpy()   # (batch_test, 3, seq_len)
            press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
            press_test = test_data[:, 3:, :].detach().cpu().numpy()
        elif (sensor_type == 'AccelGyro'):
            acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
            press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)
            acc_test = test_data[:, :3, :].detach().cpu().numpy()
            press_test = test_data[:, 3:, :].detach().cpu().numpy()
        elif (sensor_type == 'PressTemp'):
            acc_shap_vals = shap_vals_for_class0[:, :1, :]  # (batch_test, 3, seq_len)
            press_shap_vals = shap_vals_for_class0[:, 1:, :]  # (batch_test, 1, seq_len)
            acc_test = test_data[:, :1, :].detach().cpu().numpy()
            press_test = test_data[:, 1:, :].detach().cpu().numpy()
        else:
            print('error at ploting shap values, no such sensor.')

        # 计算所有数的均值
        acc_shap_mean = np.sum(acc_shap_vals, axis=(1))
        press_shap_mean = np.sum(press_shap_vals, axis=(1))

        # 确保sample没有shift，将batch flatten
        acc_shap_flatten = np.concatenate(acc_shap_mean)
        pres_shap_flatten = np.concatenate(press_shap_mean)
        test_label_flatten = np.concatenate(test_label)
        acc_data_flatten = np.concatenate(acc_test.transpose(0,2,1), axis=0)
        press_data_flatten = np.concatenate(press_test.transpose(0,2,1), axis=0)

        # 画出每个样本的 SHAP 值
        plot_shap(acc_data_flatten, acc_shap_flatten, test_label_flatten,
                  name_label, sensor1, labeldict_findstr[activity_id])
        plot_shap(press_data_flatten, pres_shap_flatten, test_label_flatten,
                  name_label, sensor2, labeldict_findstr[activity_id])

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
        # random_indices = np.random.choice(X_test.shape[0], size=100, replace=False)
        random_indices = np.arange(10)
        # 获取随机选择的样本
        test_data = X_test[random_indices]
        test_data = torch.tensor(test_data).to(device=device, dtype=torch.float)
        train_data = torch.tensor(X_labeled).to(device=device, dtype=torch.float)
        # label_vote = majority_value(y_test)
        test_label = y_test[random_indices]
        calculate_shap(train_data, test_data, test_label,
                       model, model_clean,
                       sensor_type,
                       sensor_type+name_label+'_iter'+str(iteration))

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

