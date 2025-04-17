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
import random

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
    evaluate_supContrast_model,
    plot_confusion_matrix,
    unfreeze_encoders,
    freeze_encoders,
    unfreeze_all,
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_1s,
    SimpleNN_13s,
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
    plot_func
)

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
# sensor_type = 'AccelTemp'
# sensor_type = 'AccelGyro'
sensor_type = 'PressTemp'
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
len_sw = 50
name_label = '_activebase_seed%s'%str(seed_value)


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
                                                                  # random_state=42
                                                              )

    # def init_extractors(self):
    #
    #     rootPath = r'D:\code\DeepView\deepview\calculate_results\data\umineko'
    #     with open(r'%s\AE_reconstruct_epoch1499_datalen50_pressure.pkl'%rootPath, 'rb') as file:
    #         Pstate_dict = pickle.load(file)
    #     press_state_dict = Pstate_dict['weight_list'][5]
    #     with open(r'%s\AE_reconstruct_epoch1499_datalen50_accel.pkl'%rootPath, 'rb') as file:
    #         Astate_dict = pickle.load(file)
    #     accel_state_dict = Astate_dict['weight_list'][5]
    #
    #     if torch.cuda.is_available():
    #         # 加载模型权重
    #         # press_state_dict = torch.load(press_model_path, weights_only=False)
    #         # accel_state_dict = torch.load(accel_model_path, weights_only=False)
    #         # 创建一个新的状态字典，只包含以 'feature_extractor.' 开头的层
    #         new_press_state_dict = {k.replace('feature_extractor.', ''): v for k, v in press_state_dict.items() if
    #                                 k.startswith('feature_extractor.')}
    #         new_accel_state_dict = {k.replace('feature_extractor.', ''): v for k, v in accel_state_dict.items() if
    #                                 k.startswith('feature_extractor.')}
    #         # 加载到模型中
    #         self.pre_encoder.load_state_dict(new_press_state_dict)
    #         self.acc_encoder.load_state_dict(new_accel_state_dict)

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


# ======= 4. 主动学习策略 ======= #

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


# def evaluate_model(model, loader):
#     model.eval()
#     correct, total = 0, 0
#     ground_truth, prediction = [], []
#     data_list, predlabel_list, truelabel_list = [], [], []  # 保存新标签用于supCon训练
#     with (torch.no_grad()):
#         for data, labels in loader:
#             data = data.to(device=device, dtype=torch.float)
#             label_vote = majority_value(labels)
#             label_vote = torch.from_numpy(label_vote)
#             label_vote = label_vote.to(device=device, dtype=torch.long)
#             predicts, features = model(data, if_contrast=False)
#             # predictions = torch.argmax(predicts, dim=1)  # predicts(batch,numcls); predictions(batch)
#             max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
#             correct += (predictions == label_vote).sum().item()
#             ground_truth.append(label_vote.detach().cpu().numpy())
#             prediction.append(predictions.detach().cpu().numpy())
#             total += labels.size(0)
#
#             data_list.append(data.detach().cpu().numpy())
#
#             # new_label = predictions.detach().cpu().numpy()
#             # reshaped_array = new_label.reshape(labels.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
#             # reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))
#             # predlabel_list.append(reshaped_array)
#
#     # 计算准确率
#     accuracy = accuracy_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0))
#     # print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
#     # 计算 macro F1 分数
#     macro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='macro')
#     # print(f'Macro F1 Score: {macro_f1:.2f}')
#     # 计算 micro F1 分数
#     micro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='micro')
#     # print(f'Micro F1 Score: {micro_f1:.2f}')
#     print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
#     return accuracy, macro_f1, micro_f1, prediction, ground_truth


# ======= 5. 循环主动学习 ======= #
# 初始化模型和优化器
# model = SimpleNN()
# model = model.to(device)
# classify_criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
#

# def AE_eval_time_series(train_loader, model, device):
#     model.eval()
#
#     representation_list = []
#     sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
#     for i, (sample, label) in enumerate(train_loader):
#         sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
#         # input of autoencoder will be 3D, the backbone is 1d-cnn
#         output, x_encoded = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
#         tmp_representation = x_encoded.detach().cpu().numpy()
#         representation_list.append(tmp_representation)
#         sample_list.append(sample.detach().cpu().numpy())
#         label_list.append(label.detach().cpu().numpy())
#         pred_list.append(output.detach().cpu().numpy())
#
#     return representation_list, sample_list, pred_list, label_list




# 主动学习迭代
update_size = max(int(0.01 * len(X_train_full)), 2)  # 每次选择 1% 的样本
batch_size = 4000
iteration = 1


## 创建数据加载器
plot_dataset = data_loader_umineko(data_b.astype(float), label_b.astype(int))
plot_loader = DataLoader(plot_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
## initial plot
repres_list, sample_list, pred_list, label_list = \
    AE_eval_time_series(plot_loader, model, device)
plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
          str(0), 'AccelPress')

train_accuracy_list, train_macro_f1_list, train_micro_f1_list = [], [], []
test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
test_pred_label_lists, test_truth_label_lists = [], []
weight_list = []
loss_list = []
selected_labels_list = []
ari_list, nmi_list, fmi_list, silhouette_list = [], [], [], []
while len(X_unlabeled) > 0:
    print(f"==============Iteration {iteration}===============")
    print(f"Already labeled samples: {len(X_labeled)}")
    print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

    # 创建数据加载器
    labeled_dataset = data_loader_umineko(X_labeled.astype(float), y_labeled.astype(int))
    labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ## stage 1: train supervised model, 获得分类器新参数
    model.train()
    model, avg_loss = train_sup_model(model, labeled_loader, classify_criterion, optimizer,
                                  epochs=50, device=device, if_contrast=False)
    loss_list.append(np.average(avg_loss))  # 记录50次epoch的平均loss


    ## 随机选择 n 个样本
    select_size = min(update_size, len(X_unlabeled))
    selected_indices = np.random.choice(len(X_unlabeled), select_size, replace=False)

    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]

    ## 更新标注集和未标注数据池
    X_labeled_pre = X_labeled.copy()
    y_labeled_pre = y_labeled.copy()
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    # print(f"Labeled samples: {len(X_labeled)}")

    # if (iteration==1) or (iteration==10):
    #     # ### plot, 用于检查新增label位置
    #     xlabelpre_dataset = data_loader_umineko(X_labeled_pre.astype(float),
    #                                             y_labeled_pre.astype(int))
    #     xlabelpre_loader = DataLoader(xlabelpre_dataset,
    #                                   batch_size=batch_size,
    #                                   shuffle=False, drop_last=False)
    #     X_labeled_pre_list, _, _, y_labeled_pre_list = \
    #                 AE_eval_time_series(xlabelpre_loader, model, device)
    #     ##
    #     xlabel_dataset = data_loader_umineko(X_labeled.astype(float),
    #                                             y_labeled.astype(int))
    #     xlabel_loader = DataLoader(xlabel_dataset,
    #                                   batch_size=batch_size,
    #                                   shuffle=False, drop_last=False)
    #     X_labeled_list, _, _, y_labeled_list = \
    #         AE_eval_time_series(xlabel_loader, model, device)
    #     ##
    #     xunlabel_dataset = data_loader_umineko(X_unlabeled.astype(float),
    #                                             y_unlabeled.astype(int))
    #     xunlabel_loader = DataLoader(xunlabel_dataset,
    #                                   batch_size=batch_size,
    #                                   shuffle=False, drop_last=False)
    #     X_unlabeled_list, _, _, y_unlabeled_list = \
    #         AE_eval_time_series(xunlabel_loader, model, device)
    #
    #     plot_func_new_label_marker(X_labeled_list, X_unlabeled_list,
    #                                X_labeled_pre_list, y_labeled_list,
    #                                y_unlabeled_list, y_labeled_pre_list,
    #                                name='_SupBase_pred_%s'%str(iteration))

    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(plot_loader, model, device)
    ari, nmi, fmi, silhouette = (
        plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
                  iteration, sensor_type + name_label))

    # ======= 6. 模型测试 ======= #
    print('-------Training----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = evaluate_model(model, labeled_loader)
    train_accuracy_list.append(accuracy)
    train_macro_f1_list.append(macro_f1)
    train_micro_f1_list.append(micro_f1)

    test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print('-------Test----------')
    accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = evaluate_model(model, test_loader)
    test_accuracy_list.append(accuracy)
    test_micro_f1_list.append(micro_f1)
    test_macro_f1_list.append(macro_f1)
    test_pred_label_lists.append(np.concatenate(pred_label_list, axis=0))
    test_truth_label_lists.append(np.concatenate(truth_label_list, axis=0))

    # print('-------cluster results----------')
    ari_list.append(ari)
    nmi_list.append(nmi)
    fmi_list.append(fmi)
    silhouette_list.append(silhouette)

    # 保存当前模型权重到列表
    weight_list.append(model.state_dict())
    # 如果你在训练模型时使用了GPU，并希望在CPU上加载权重，可以使用以下代码
    # model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))


    selected_labels_list.append(selected_labels)
    # 每十次迭代，也就是增加10%的数据后，保存一次模型和heatmap
    # if iteration % 10 == 0:
    if (iteration == 1) or (iteration == 10) or (iteration == 20):
        plot_confusion_matrix(pred_label_list, truth_label_list, name='_SupBase_pred_%s'%str(iteration))

    iteration += 1
plot_confusion_matrix(pred_label_list, truth_label_list,
                      name='%s_%s_pred_%s'%(sensor_type,name_label,str(iteration)))


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
        'weight_list': weight_list,
        'ari_list': ari_list,
        'nmi_list': nmi_list,
        'fmi_list': fmi_list,
        'silhouette_list': silhouette_list
    }

    pickle.dump(result_dict, f)
print("All unlabeled samples have been labeled!")

# # plot loss
# plt.figure()
# plt.plot(train_accuracy_list, label='train_accuracy')
# plt.plot(train_macro_f1_list, label='train_macro_f1')
# plt.plot(train_micro_f1_list, label='train_micro_f1')
# plt.legend()
# plt.grid()
# plt.xlabel('Iteration')
# plt.xlabel('Accuracy/F1')
# plt.title('supervised learning with active learning baseline - train set')
# plt.savefig('SupBase_train_accuracy.png')
# # plt.show()
# plt.close()
#
# plt.figure()
# plt.plot(test_accuracy_list, label='test_accuracy')
# plt.plot(test_macro_f1_list, label='test_macro_f1')
# plt.plot(test_micro_f1_list, label='test_micro_f1')
# plt.legend()
# plt.grid()
# plt.xlabel('Iteration')
# plt.xlabel('Accuracy/F1')
# plt.title('supervised learning with active learning baseline - test set')
# plt.savefig('SupBase_test_accuracy.png')
# # plt.show()
# plt.close()
#
#
# plt.figure()
# plt.plot(loss_list, label='train_loss')
# plt.legend()
# plt.grid()
# plt.xlabel('Iteration')
# plt.xlabel('Loss value')
# plt.title('supervised learning with active learning baseline - train loss')
# plt.savefig('SupBase_loss.png')
# # plt.show()
# plt.close()