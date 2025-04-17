import numpy as np
import torch
import random
from scipy.stats import entropy
from deepview.calculate_results.models.utils import (
    majority_value,
)
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
OMP_NUM_THREADS=23


def compute_loss_contribution(model,
                              unlabeled_data,
                              labeled_data,
                              labeled_labels,
                              device):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(unlabeled_data, device=device, dtype=torch.float32),
                                        if_contrast=True)
        _, labeled_embeddings = model(torch.tensor(labeled_data, device=device, dtype=torch.float32), if_contrast=True)

        # 计算相似性矩阵 (unlabeled x labeled)
        similarities = torch.matmul(unlabeled_embeddings, labeled_embeddings.T)
        label_vote = majority_value(labeled_labels)
        label_vote = torch.from_numpy(label_vote)
        labeled_labels = label_vote.to(device=device, dtype=torch.long)

        loss_contributions = []
        for i, sim in enumerate(similarities):  # 遍历未标注样本
            # 获取与已标注数据的正负样本掩码
            pos_mask = (labeled_labels == labeled_labels[i % len(labeled_labels)]).float()
            neg_mask = 1 - pos_mask

            # 避免掩码全为 0
            if pos_mask.sum() == 0:
                pos_mask[torch.argmax(sim)] = 1
            if neg_mask.sum() == 0:
                neg_mask[torch.argmin(sim)] = 1

            # 归一化相似性以避免指数计算溢出
            sim = sim - sim.max()  # 数值稳定性优化

            # 正样本损失
            pos_sim = sim * pos_mask
            pos_loss = -torch.logsumexp(pos_sim, dim=-1) + torch.logsumexp(sim, dim=-1)  # 使用 logsumexp 避免显式指数计算

            # 负样本损失
            neg_sim = sim * neg_mask
            neg_loss = -torch.logsumexp(neg_sim, dim=-1) + torch.logsumexp(sim, dim=-1)

            # 合并正负样本损失
            total_loss = pos_loss + neg_loss
            loss_contributions.append(total_loss.item())

    return np.array(loss_contributions)


def supContrast_contribution(X_labeled, y_labeled,
                             X_unlabeled, y_unlabeled,
                             model, update_size):
    # 计算每个样本的贡献度
    ## 计算未标注数据（all-labeled-test）的对比损失贡献
    loss_contributions = compute_loss_contribution(model, X_unlabeled, X_labeled, y_labeled)

    # 选择贡献度最大的样本
    selected_indices = np.argsort(loss_contributions)[-update_size:]
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    ## 更新标注集和未标注数据池
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled


def data_sampling(sampling_method, X_labeled, y_labeled,
                  X_unlabeled, y_unlabeled,
                  model, update_size,
                  majority_label=0, minority_label=0, device='cpu'):
    '''
    selected_data = (newX_labeled, newy_labeled),表示新添加的标注数据
    '''
    if sampling_method == "random":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            random_sampling(X_labeled, y_labeled,
                            X_unlabeled, y_unlabeled, update_size))
    elif sampling_method == "entropy":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device))
    elif sampling_method == "entropy1":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device, choice=1))
    elif sampling_method == "entropy2":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            uncertainty_sampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 model, update_size, device,choice=2))
    elif sampling_method == "underRandom":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data = (
            random_undersampling(X_labeled, y_labeled,
                                 X_unlabeled, y_unlabeled,
                                 majority_label, update_size, undersample_ratio=1.0))
    elif sampling_method == "overAugment":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data\
            = data_augmentation_oversampling(
                                            X_labeled, y_labeled,
                                            X_unlabeled, y_unlabeled,
                                            minority_label, update_size,
                                            augmentation_factor=2)
    elif sampling_method == "repreSamp":
        X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data\
            = representative_sampling(
                                    X_labeled, y_labeled,
                                    X_unlabeled, y_unlabeled,
                                    model, update_size, device)
    # elif sampling_method == "diversity":
    #     X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data\
    #         = diversity_sampling(
    #                             X_labeled, y_labeled,
    #                             X_unlabeled, y_unlabeled,
    #                             model, update_size, device)
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    return X_labeled_new, y_labeled_new, X_unlabeled_new, y_unlabeled_new, selected_data


def least_confidence(probabilities, n_samples):
    '''最小置信度方法选择模型对其预测最不自信的样本，通常通过计算每个样本的最大预测概率来实现'''
    # 选择最大置信度的样本
    max_confidence = np.max(probabilities, axis=1)
    # 选择最小置信度的样本
    selected_indices = np.argsort(max_confidence)[:n_samples]
    return selected_indices

def min_margin(probabilities, n_samples):
    '''最小边际方法选择模型对其预测最不自信的样本，通常通过计算每个样本的最大预测概率来实现'''
    # 计算最大边际
    margins = np.partition(probabilities, -2, axis=1)[:, -1] - np.partition(probabilities, -1, axis=1)[:, -2]
    # 选择最小边际的样本
    selected_indices = np.argsort(margins)[:n_samples]
    return selected_indices

# 计算样本不确定性（基于熵, 还有least_confidence and min_margin可以选择）
'''
对于需要快速响应的在线学习任务，可以选择最小置信度；而在需要更全面评估不确定性的情况下，最大熵可能更合适。
'''
def uncertainty_sampling(X_labeled, y_labeled,
                         X_unlabeled, y_unlabeled,
                         model, update_size, device, choice=0):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
        probs = model.classifier(unlabeled_embeddings)

        if choice==0:
            uncertainty = entropy(probs.detach().cpu().numpy().T)  # 计算每个样本的熵
            # 选择贡献度最大的样本
            selected_indices = np.argsort(uncertainty)[-update_size:]
        elif choice ==1 :
            selected_indices = least_confidence(probs.detach().cpu().numpy(), update_size)
        else:
            selected_indices = min_margin(probs.detach().cpu().numpy(), update_size)

        selected_samples = X_unlabeled[selected_indices]
        selected_labels = y_unlabeled[selected_indices]
        ## 更新标注集和未标注数据池
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Labeled samples: {len(X_labeled)}")

        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)


def random_sampling(X_labeled, y_labeled,
                    X_unlabeled, y_unlabeled, update_size):
    """
    在未标注数据集中随机选取一定数量的样本更新到标注集中。

    参数:
    --------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, D)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled, )
    X_unlabeled : np.ndarray
        未标注数据特征 (N_unlabeled, D)
    y_unlabeled : np.ndarray
        未标注数据标签 (N_unlabeled, )
    model : torch.nn.Module
        模型实例，随机采样不依赖模型，但保留接口以保持格式一致
    update_size : int
        本轮迭代需要从未标注集中选取的样本数量
    device : torch.device
        设备信息 (cpu or cuda)，随机采样中也不会用到，但保留以统一函数签名

    返回:
    --------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代所选数据的标签 (便于后续统计等)
    """

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)

    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)


def random_undersampling(X_labeled, y_labeled,
                         X_unlabeled, y_unlabeled,
                         majority_label, update_size, undersample_ratio=1.0):
    """
    在已标注数据集中，对多数类样本进行随机欠采样，以达到类别平衡。

    参数:
    --------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, D)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled, )
    majority_label : int or str
        多数类的类别标签
    undersample_ratio : float
        欠采样比例，表示欠采样后，多数类样本相对于少数类的比例。

    返回:
    --------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代被移除的数据的标签 (便于后续统计等)
    """

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 统计类别样本数量
    selected_labels_batch = majority_value(selected_labels)
    unique, counts = np.unique(selected_labels_batch, return_counts=True)
    class_counts = dict(zip(unique, counts))

    if majority_label not in class_counts:
        # 更新 X_labeled, y_labeled
        X_labeled = np.vstack([X_labeled, selected_samples])
        y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
        # 从未标注集中剔除已选样本
        X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
        y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
        print(f"Error: 指定的多数类 {majority_label} 在数据集中不存在！")
        return X_labeled, y_labeled, X_unlabeled, y_unlabeled, np.array([])

    # 计算少数类样本数量
    minority_count = min(class_counts.values())

    # 计算多数类需要保留的样本数量
    majority_count_target = int(minority_count * undersample_ratio)

    # 获取多数类样本的索引
    majority_indices = np.where(selected_labels_batch == majority_label)[0]

    # 随机选择需要保留的多数类样本
    selected_majority_indices = np.random.choice(majority_indices, majority_count_target, replace=False)

    # 计算被移除的样本
    removed_indices = np.setdiff1d(majority_indices, selected_majority_indices)

    # 获取所有少数类样本的索引
    minority_indices = np.where(selected_labels_batch != majority_label)[0]

    # 组合新数据索引
    selected_indices = np.concatenate([selected_majority_indices, minority_indices])

    # 生成新的标注数据
    X_selected_new = selected_samples[selected_indices]
    y_selected_new = selected_labels[selected_indices]

    # 更新 X_labeled, y_labeled
    X_labeled_new = np.vstack([X_labeled, X_selected_new])
    y_labeled_new = np.concatenate([y_labeled, y_selected_new], axis=0)

    print(
        f"Under-sampling completed: Majority class reduced from {class_counts[majority_label]} → {majority_count_target}"
    )
    print(f"Labeled samples: {len(X_labeled_new)}")

    return X_labeled_new, y_labeled_new, X_unlabeled, y_unlabeled, (X_selected_new, y_selected_new)


def data_augmentation_oversampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   minority_label, update_size,
                                   augmentation_factor=2):
    """
    通过数据增强（Data Augmentation）对少数类样本进行过采样。

    参数:
    ----------
    X_labeled : np.ndarray
        已标注数据特征 (N_labeled, T) (时间序列或其他数据)
    y_labeled : np.ndarray
        已标注数据标签 (N_labeled,)
    minority_label : int or str
        需要增强的少数类标签
    augmentation_factor : int
        每个少数类样本生成多少个新的样本（倍数）

    返回:
    ----------
    X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels
      - 更新后的已标注数据与未标注数据
      - selected_labels: 本轮迭代生成的新样本的标签 (便于后续统计等)
    """

    # augmentation_factor = random.choice([1, 2, 3, 4])

    # 随机选取 update_size 个索引
    selected_indices = np.random.choice(len(X_unlabeled), update_size, replace=False)

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 获取少数类样本
    y_labeled_batch = majority_value(selected_labels)
    minority_indices = np.where(y_labeled_batch == minority_label)[0]

    X_minority = selected_samples[minority_indices]

    # 存储合成的新样本
    X_synthetic = []
    y_synthetic = []
    if len(X_minority) == 0:
        X_new_labeled = selected_samples
        y_new_labeled = selected_labels
    else:
        for x, y in zip(X_minority, minority_indices):
            for _ in range(augmentation_factor):
                # 对单个少数类样本进行数据增强
                x_augmented = augment_sample(x)
                X_synthetic.append(x_augmented)
                y_synthetic.append(selected_labels[y])

        # 转换为 NumPy 数组
        X_synthetic = np.array(X_synthetic)
        y_synthetic = np.array(y_synthetic)
        # y_synthetic = selected_labels[minority_indices]

        X_new_labeled = np.vstack([X_synthetic, selected_samples])
        y_new_labeled = np.concatenate([y_synthetic, selected_labels])

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, X_new_labeled])
    y_labeled = np.concatenate([y_labeled, y_new_labeled], axis=0)

    # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
    # selected_labels = y_synthetic  # 记录新增的样本标签

    print(f"Data Augmentation Oversampling completed: Added {len(X_synthetic)} new samples for label {minority_label}")
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (X_new_labeled, y_new_labeled)


def augment_sample(x):
    """
    针对单个样本进行数据增强（适用于时序数据）。

    增强策略：
    1. 加噪声（随机高斯噪声）
    2. 时移（随机向前或向后平移）
    3. 放缩（信号幅度缩放）
    4. 翻转（适用于对称信号）

    参数:
    ----------
    x : np.ndarray
        输入的单个样本 (T,)

    返回:
    ----------
    x_augmented : np.ndarray
        增强后的样本
    """
    x_aug = x.copy()

    # 1. **加高斯噪声** (Noise Injection)
    if np.random.rand() < 0.5:  # 50% 概率应用
        noise = np.random.normal(loc=0, scale=0.02 * np.std(x), size=x.shape)
        x_aug += noise

    # 2. **时移 (Time Shifting)**
    if np.random.rand() < 0.5:
        shift = np.random.randint(-3, 3)  # 在 [-3, 3] 之间随机平移
        x_aug = np.roll(x_aug, shift)

    # 3. **放缩 (Scaling)**
    if np.random.rand() < 0.5:
        scale_factor = np.random.uniform(0.9, 1.1)  # 在 [0.9, 1.1] 之间缩放
        x_aug *= scale_factor

    # 4. **翻转 (Flipping)**
    if np.random.rand() < 0.3:  # 30% 概率应用
        x_aug = -x_aug

    return x_aug



def representative_sampling(X_labeled, y_labeled,
                                   X_unlabeled, y_unlabeled,
                                   model, update_size, device):
    '''代表性采样选择那些能代表整个数据分布的样本，通常使用聚类算法（如K-means）来确定样本的代表性。'''
    # 使用K-means聚类
    # 先获得latent representation
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
                                        if_contrast=True)
    data = unlabeled_embeddings.detach().cpu().numpy()
    # 再选择new data
    kmeans = KMeans(n_clusters=update_size, n_init=10).fit(data)
    labels = kmeans.labels_
    # 初始化一个数组来存储每个聚类的最近点索引
    selected_indices = np.zeros(kmeans.n_clusters, dtype=int)
    # 遍历每个聚类
    for cluster_index in range(kmeans.n_clusters):
        # 获取当前聚类中所有点的索引
        cluster_points_indices = np.where(labels == cluster_index)[0]
        # 获取这些点的坐标
        cluster_points = data[cluster_points_indices]
        # 计算这些点到当前聚类中心的距离
        distances = np.linalg.norm(cluster_points -
                                   kmeans.cluster_centers_[cluster_index], axis=1)
        # 找到距离最近的点的索引
        closest_point_index_within_cluster = np.argmin(distances)
        # 转换为原始数据集中的索引
        selected_indices[cluster_index] = cluster_points_indices[closest_point_index_within_cluster]

    # 根据索引取对应样本及标签
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]
    # 从未标注集中剔除已选样本
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    # 更新 X_labeled, y_labeled
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)

    # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
    # selected_labels = y_synthetic  # 记录新增的样本标签
    print(f"Labeled samples: {len(X_labeled)}")

    return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)

# def diversity_sampling(X_labeled, y_labeled,
#                        X_unlabeled, y_unlabeled,
#                        model, update_size, device):
#     '''
#     这个方法不适用于减少标签数据量
#     多样性采样选择在特征空间中相互独立的样本，以确保训练数据的多样性，从而防止模型过拟合
#     '''
#
#     model.eval()
#     with torch.no_grad():
#         # 计算未标注数据和已标注数据的嵌入
#         _, unlabeled_embeddings = model(torch.tensor(X_unlabeled, device=device, dtype=torch.float32),
#                                         if_contrast=True)
#     unlabeled_embeddings = unlabeled_embeddings.detach().cpu().numpy()
#     _, labeled_embeddings = model(torch.tensor(X_labeled, device=device, dtype=torch.float32),
#                                     if_contrast=True)
#     labeled_embeddings = labeled_embeddings.detach().cpu().numpy()
#
#     selected_indices = []
#     # 迭代选择样本
#     for _ in range(1, update_size):
#         # 计算当前已选择样本与所有样本的距离
#         distances = pairwise_distances(unlabeled_embeddings, labeled_embeddings)
#         # 选择与已选样本距离最远的样本
#         farthest_index = np.argmax(distances.sum(axis=1))
#         selected_indices.append(farthest_index)
#
#     # 根据索引取对应样本及标签
#     selected_samples = X_unlabeled[selected_indices]
#     selected_labels = y_unlabeled[selected_indices]
#     # 从未标注集中剔除已选样本
#     X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
#     y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)
#
#     # 更新 X_labeled, y_labeled
#     X_labeled = np.vstack([X_labeled, selected_samples])
#     y_labeled = np.concatenate([y_labeled, selected_labels], axis=0)
#
#     # # 由于数据增强的样本不属于原始未标注数据，X_unlabeled 和 y_unlabeled 为空
#     # selected_labels = y_synthetic  # 记录新增的样本标签
#     print(f"Labeled samples: {len(X_labeled)}")
#
#     return X_labeled, y_labeled, X_unlabeled, y_unlabeled, (selected_samples, selected_labels)
