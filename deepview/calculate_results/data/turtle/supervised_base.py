'''
使用2018，21年的数据，使用完全没有初始化的encoder，随机添加label，计算监督学习结果
两个结果：
latent space结果
active learning的F1 score
'''

import os
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

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
    process_press_temperature
)
from deepview.calculate_results.data.umineko.train_func import (
    train_model,
    evaluate_model,
    plot_confusion_matrix,
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_1s,
    SimpleNN_13s,
    SimpleNN_11s,
    SimpleNN_33s,
)
from deepview.calculate_results.data.umineko.cluster_method import (
    vis_scatter_label_2d,
)
import random

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

seed_value = 2025
# seed_value = 2
# seed_value = 5
# seed_value = 1
# seed_value = 10
# seed_value = 100
# seed_value = 11
set_random_seed(seed_value)

# sensor_type = 'Accel'
# sensor_type = 'AccelPress'
sensor_type = 'AccelTemp'
# sensor_type = 'AccelGyro'
# sensor_type = 'PressTemp'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
len_sw = 50
batch_size = 4000
name_label = '_supervisebase_seed%s'%str(seed_value)


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


# 初始化模型和优化器
if (sensor_type == 'Accel'):
    model = SimpleNN_1s()
elif (sensor_type == 'AccelPress') or (sensor_type == 'AccelTemp'):
    model = SimpleNN_13s()
elif (sensor_type == 'AccelGyro'):
    model = SimpleNN_33s()
elif (sensor_type == 'PressTemp'):
    model = SimpleNN_11s()
else:
    print('no model available')
model = model.to(device)
classify_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


# ======= 3. 模型训练 ======= #

def train_sup_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels = batch
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)

            data = data.to(device=device, dtype=torch.float)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            # outputs: supcontrast output; feature: feature extractor output
            outputs, features = model(data, if_contrast)
            loss = criterion(outputs, label_vote)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model, avg_loss



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



## 创建数据加载器
plot_dataset = data_loader_umineko(data_b.astype(float), label_b.astype(int))
plot_loader = DataLoader(plot_dataset, batch_size=batch_size,
                         shuffle=False,
                         drop_last=False)

## initial plot
# repres_list, sample_list, pred_list, label_list = \
#     AE_eval_time_series(plot_loader, model, device)
# ari, nmi, fmi, silhouette = (
#         vis_scatter_label_2d(repres_list,
#                              label_list, 0,
#                              sensor_type+name_label,
#                              plot_flag=True))

train_accuracy_list, train_macro_f1_list, train_micro_f1_list = [], [], []
test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
test_pred_label_lists, test_truth_label_lists = [], []
weight_list = []
loss_list = []
selected_labels_list = []
ari_list, nmi_list, fmi_list, silhouette_list = [], [], [], []
for iteration in range(50):
    print(f"==============Iteration {iteration}===============")

    # 创建数据加载器
    train_dataset = data_loader_umineko(X_train_full.astype(float), y_train_full.astype(int))
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=False, drop_last=False)

    ## stage 1: train supervised model, 获得分类器新参数
    model.train()
    model, avg_loss = train_sup_model(model, train_loader, classify_criterion, optimizer,
                                  epochs=10, device=device, if_contrast=False)
    loss_list.append(np.average(avg_loss))  # 记录50次epoch的平均loss


    # ======= 6. 模型测试 ======= #
    print('-------Training----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = evaluate_model(model,
                                                                                     train_loader,
                                                                                     device)
    train_accuracy_list.append(accuracy)
    train_macro_f1_list.append(macro_f1)
    train_micro_f1_list.append(micro_f1)

    test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print('-------Test----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = evaluate_model(model,
                                                                                     test_loader,
                                                                                     device)
    test_accuracy_list.append(accuracy)
    test_micro_f1_list.append(micro_f1)
    test_macro_f1_list.append(macro_f1)
    test_pred_label_lists.append(np.concatenate(pred_label_list, axis=0))
    test_truth_label_lists.append(np.concatenate(truth_label_list, axis=0))
    selected_labels_list.append(y_train_full)
    # ## plot
    # if (iteration == 1) or (iteration == 10) or (iteration == 20):
    #     plot_flag = True
    # else:
    #     plot_flag = False
    # repres_list, sample_list, pred_list, label_list = \
    #     AE_eval_time_series(plot_loader, model, device)
    # ari, nmi, fmi, silhouette = (
    #     vis_scatter_label_2d(repres_list,
    #                          label_list, iteration,
    #                          sensor_type + name_label,
    #                          plot_flag=plot_flag))

    # print('-------cluster results----------')
    # ari_list.append(ari)
    # nmi_list.append(nmi)
    # fmi_list.append(fmi)
    # silhouette_list.append(silhouette)

    # 保存当前模型权重到列表
    weight_list.append(model.state_dict())

    # 每十次迭代，也就是增加10%的数据后，保存一次模型和heatmap
    # if iteration % 10 == 0:
    # if (iteration == 1) or (iteration == 10) or (iteration == 20):
    #     pred_label_list_concat = np.concatenate(pred_label_list, axis=0)
    #     truth_label_list_concat = np.concatenate(truth_label_list, axis=0)
    #     plot_confusion_matrix(pred_label_list_concat, truth_label_list_concat,
    #                           name='%s_%s_pred_%s'%(sensor_type,name_label,str(iteration)))

    # if iteration == 20:
    #     break


# ======= 7. 保存结果 ======= #
with open(sensor_type+'_%s_epoch%s_results.pkl'%(name_label, str(iteration)), 'wb') as f:
    result_dict = {
        'train_accuracy_list': train_accuracy_list,
        'train_macro_f1_list': train_macro_f1_list,
        'train_micro_f1_list': train_micro_f1_list,
        'test_accuracy_list': test_accuracy_list,
        'test_macro_f1_list': test_macro_f1_list,
        'test_micro_f1_list': test_micro_f1_list,
        'test_pred_label_lists': test_pred_label_lists,
        'test_truth_label_lists': test_truth_label_lists,
        'selected_labels_list': selected_labels_list,
        'weight_list': weight_list,
        'ari_list': ari_list,
        'nmi_list': nmi_list,
        'fmi_list': fmi_list,
        'silhouette_list': silhouette_list
    }

    pickle.dump(result_dict, f)
print("All unlabeled samples have been labeled!")
