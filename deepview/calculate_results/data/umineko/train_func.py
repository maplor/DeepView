import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from deepview.calculate_results.models.utils import (
    majority_value,
    labeldict_findstr,
labeldict_findstr_omizu,
labeldict_findstr_turtle,
)


def train_model(model, loader, criterion, optimizer, epochs=500, device='cpu', if_contrast=True):
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
            outputs, fea = model(data, if_contrast)
            loss = criterion(outputs, label_vote)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss.append(np.mean(losses))
        # print(f"Epoch {epoch + 1}, Loss: {avg_loss[-1]}")
    return model, avg_loss


# 数据增强函数
def augment(data):
    """
    对时序数据进行增强。

    Args:
        data: 输入数据，形状为 (batch_size, 3, 50)。

    Returns:
        增强后的数据，形状为 (batch_size, 3, 50)。
    """
    datalen = data.shape[-1]
    datadim = data.shape[-2]
    augmented_data = []
    for seq in data:
        # 1. 时间抖动 (Time Jittering)
        if random.random() < 0.5:
            jitter = np.random.normal(0, 0.01, seq.shape) # 添加少量噪声
            seq = seq + torch.from_numpy(jitter).float().to(seq.device)

        # 2. 时间缩放 (Time Warping)
        if random.random() < 0.5:
            alpha = np.random.uniform(0.8, 1.2) # 随机缩放比例
            new_length = int(datalen * alpha)
            # 使用插值进行缩放
            if new_length > 0: # 避免new_length为0的情况
                seq_np = seq.cpu().numpy()
                seq_resized = np.resize(seq_np, (datadim, new_length))
                seq = torch.from_numpy(seq_resized).float().to(seq.device)
                if new_length < datalen:
                    padding = torch.zeros(datadim, datalen - new_length).to(seq.device)
                    seq = torch.cat([seq, padding], dim=1)
                elif new_length > datalen:
                    seq = seq[:, :datalen]


        # 3. 随机裁剪 (Random Cropping)
        if random.random() < 0.5:
            start = np.random.randint(0, 10)
            seq = seq[:, start:start+datalen]
            if seq.shape[1] < datalen:
                padding = torch.zeros(datadim, datalen - seq.shape[1]).to(seq.device)
                seq = torch.cat([seq, padding], dim=1)

        augmented_data.append(seq)
    return torch.stack(augmented_data)

def train_model_aug(model, loader, criterion, optimizer, epochs=500, device='cpu', contrast_weight=0.5):
    model.train()
    avg_loss = []
    for epoch in range(epochs):
        losses = []
        for batch in loader:
            data, labels = batch

            data = data.to(device=device, dtype=torch.float)

            # 数据增强
            data_aug = augment(data)

            _, features = model(data, if_contrast=True)
            _, aug_features = model(data_aug, if_contrast=True)
            contrast_loss = criterion(torch.concat([features, aug_features], dim=0))

            # # 加权求和
            # loss = (1 - contrast_weight) * class_loss + contrast_weight * contrast_loss

            losses.append(contrast_loss.item())

            optimizer.zero_grad()
            contrast_loss.backward()
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


def evaluate_model(model, loader, device):
    # supervised learning
    model.eval()
    # device = model.device
    ground_truth, prediction = [], []
    data_list, predlabel_list, truelabel_list = [], [], []  # 保存新标签用于supCon训练
    with (torch.no_grad()):
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            predicts, fea = model(data, if_contrast=False)
            max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
            ground_truth.append(label_vote.detach().cpu().numpy())
            prediction.append(predictions.detach().cpu().numpy())
            data_list.append(data.detach().cpu().numpy())

    # 计算准确率
    accuracy = accuracy_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0))
    # print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    # 计算 macro F1 分数
    macro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='macro')
    # print(f'Macro F1 Score: {macro_f1:.2f}')
    # 计算 micro F1 分数
    micro_f1 = f1_score(np.concatenate(ground_truth, axis=0), np.concatenate(prediction, axis=0), average='micro')
    # print(f'Micro F1 Score: {micro_f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    return accuracy, macro_f1, micro_f1, prediction, ground_truth


def evaluate_supContrast_model(model, loader, device, cal_threshold=0.0):
    model.eval()
    # device = model.device
    correct, total = 0, 0
    threshold_acc, threshold_total = 0, 0
    data_list, predlabel_list, truelabel_list = [], [], []  # 保存新标签用于supCon训练
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device=device, dtype=torch.float)
            label_vote = majority_value(labels)
            label_vote = torch.from_numpy(label_vote)
            label_vote = label_vote.to(device=device, dtype=torch.long)
            predicts, fea = model(data, if_contrast=False)
            max_values, predictions = torch.max(predicts, dim=1)  # max_values用来挑选threshold
            correct += (predictions == label_vote).sum().item()

            ## 计算预测值大于cal_threshold前提下的准确率
            # 将 max_values 小于 cal_threshold 的位置上的 max_indices 赋值为 -1
            predictions[max_values < cal_threshold] = -1
            threshold_acc += (predictions == label_vote).sum().item()
            # # 计算 predictions 中 -1 的个数
            # count_neg_ones = (predictions == -1).sum().item()

            ## 保存新标签和数据
            # 获取 predictions 中 -1 的位置
            pos_positions = torch.nonzero(predictions != -1).squeeze()  # squeeze() 去掉多余的维度
            if len(pos_positions) > 0:
                new_data = data[pos_positions].detach().cpu().numpy()
                data_list.append(new_data)

                new_label = predictions[pos_positions].detach().cpu().numpy()
                # reshaped_array = new_label.reshape(new_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))  # 复制到 (batch, 50)

                t_label = label_vote[pos_positions].detach().cpu().numpy()
                # reshaped_t = t_label.reshape(t_label.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_t = np.tile(reshaped_t, (1, labels.shape[1]))  # 复制到 (batch, 50)

                predlabel_list.append(new_label)
                truelabel_list.append(t_label)
            else:
                data_list.append(data.detach().cpu().numpy())

                new_label = predictions.detach().cpu().numpy()
                # reshaped_array = new_label.reshape(labels.shape[0], 1)  # 将形状从 (batch,) 转换为 (batch, 1)
                # reshaped_array = np.tile(reshaped_array, (1, labels.shape[1]))
                predlabel_list.append(new_label)

                truelabel_list.append(label_vote.detach().cpu().numpy())

    # 重新组合 data和label
    concat_data = np.concatenate(data_list, axis=0)
    concat_label = np.concatenate(predlabel_list, axis=0)
    true_label = np.concatenate(truelabel_list, axis=0)
    # 计算准确率
    accuracy = accuracy_score(true_label, concat_label)
    # print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    # 计算 macro F1 分数
    macro_f1 = f1_score(true_label, concat_label, average='macro')
    # print(f'Macro F1 Score: {macro_f1:.2f}')
    # 计算 micro F1 分数
    micro_f1 = f1_score(true_label, concat_label, average='micro')
    # print(f'Micro F1 Score: {micro_f1:.2f}')
    print(f'Accuracy: {accuracy:.2f}, Macro F1 Score: {macro_f1:.2f}, Micro F1 Score: {micro_f1:.2f}')
    return (accuracy, macro_f1, micro_f1,  # accuracy
            concat_data, concat_label, true_label)  # label



#----------------eval--------------------------

def AE_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input of autoencoder will be 3D, the backbone is 1d-cnn
        output, x_encoded = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list


#-------------parameters---------------------
def freeze_encoders(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = False
    for param in model.linear.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

def freeze_projector(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

# 解冻特征提取器函数：stage1，训练对比损失
def unfreeze_encoders(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

def unfreeze_all(model):
    for name, module in model.named_modules():
        if "encoder" in name:
            for param in module.parameters():
                param.requires_grad = True
    for param in model.linear.parameters():
        param.requires_grad = True
    for param in model.projector.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True


#------------------viz result-----------------------
def plot_confusion_matrix(pred_label, truth_label, name, is_omizu=False):
    # # 将列表展开为一维数组
    # truth_label = np.concatenate(truth_label_list, axis=0)
    # pred_label = np.concatenate(pred_label_list, axis=0)

    # 计算混淆矩阵
    cm = confusion_matrix(truth_label, pred_label)

    # 获取完整的类别标签列表，确保所有类别都能显示
    if 'omizu' in is_omizu:
        class_names = sorted(set(labeldict_findstr_omizu.values()))  # 确保按顺序排列
    elif is_omizu == 'turtle':
        class_names = sorted(set(labeldict_findstr_turtle.values()))  # 确保按顺序排列
    else:
        class_names = sorted(set(labeldict_findstr.values()))  # 确保按顺序排列

    # 设置热图尺寸
    plt.figure(figsize=(10, 7))

    # 绘制热图，确保所有标签完整显示
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    # 添加标题和标签
    plt.title('Confusion Matrix Heatmap')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # 调整刻度标签显示，防止标签重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45, va='center')

    # 保存热图到文件
    plt.savefig(f'figures/heatmap_{name}.png', bbox_inches='tight')

    # 显示热图
    # plt.show()
    plt.close()
    return
