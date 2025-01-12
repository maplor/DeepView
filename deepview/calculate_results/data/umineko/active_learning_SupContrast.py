import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ======= 1. 数据生成 ======= #
class CustomDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.data[idx], self.labels[idx]
        else:
            return self.data[idx]

# 生成二分类数据
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
X = StandardScaler().fit_transform(X)

# 将数据分为8:2，其中2为测试集
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化标注数据集（1%）和未标注数据池（99%）
initial_size = int(0.01 * len(X_train_full))
X_labeled = X_train_full[:initial_size]
y_labeled = y_train_full[:initial_size]
X_unlabeled = X_train_full[initial_size:]
y_unlabeled = y_train_full[initial_size:]

# ======= 2. 模型与 Supervised Contrastive Learning ======= #
class SimpleNN(nn.Module):
    def __init__(self, input_dim, embedding_dim=64):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.embedding(x)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = nn.functional.normalize(embeddings, dim=1)
        logits = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.unsqueeze(1)
        mask = (labels == labels.T).float()  # Positive pairs mask
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(mask.device)  # Remove self-pairs

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        loss = -(mask * log_prob).sum(1) / mask.sum(1)
        return loss.mean()

# ======= 3. 模型训练 ======= #
def train_scl_model(model, loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            data, labels = batch
            embeddings = model(data)
            loss = criterion(embeddings, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# ======= 4. 主动学习策略 ======= #
def compute_loss_contribution(model, unlabeled_data, labeled_data, labeled_labels):
    model.eval()
    with torch.no_grad():
        # 计算未标注数据和已标注数据的嵌入
        unlabeled_embeddings = model(torch.tensor(unlabeled_data, dtype=torch.float32))
        labeled_embeddings = model(torch.tensor(labeled_data, dtype=torch.float32))

        # 计算相似性矩阵 (unlabeled x labeled)
        similarities = torch.matmul(unlabeled_embeddings, labeled_embeddings.T)
        labeled_labels = torch.tensor(labeled_labels, dtype=torch.long)

        loss_contributions = []
        for i, sim in enumerate(similarities):  # 遍历未标注样本
            # 获取与已标注数据的正负样本掩码
            pos_mask = (labeled_labels == labeled_labels[i % len(labeled_labels)]).float()
            neg_mask = 1 - pos_mask

            # 计算正样本损失
            pos_sim = sim * pos_mask  # 仅保留正样本相似度
            pos_loss = -torch.log(torch.exp(pos_sim).sum() / (torch.exp(sim).sum() + 1e-10))  # 避免数值问题

            # 计算负样本损失
            neg_sim = sim * neg_mask  # 仅保留负样本相似度
            neg_loss = -torch.log(1 - torch.exp(neg_sim).sum() / (torch.exp(sim).sum() + 1e-10))

            # 合并正负样本损失
            total_loss = pos_loss + neg_loss
            loss_contributions.append(total_loss.item())

    return np.array(loss_contributions)



# ======= 5. 循环主动学习 ======= #
# 初始化模型和优化器
model = SimpleNN(input_dim=X.shape[1])
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 主动学习迭代
batch_size = int(0.01 * len(X_train_full))  # 每次选择 1% 的样本
while len(X_unlabeled) > 0:
    print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

    # 创建数据加载器
    labeled_dataset = CustomDataset(X_labeled, y_labeled)
    labeled_loader = DataLoader(labeled_dataset, batch_size=32, shuffle=True)

    # 训练模型
    model = train_scl_model(model, labeled_loader, criterion, optimizer)

    # 计算未标注数据的对比损失贡献
    loss_contributions = compute_loss_contribution(model, X_unlabeled, X_labeled, y_labeled)

    # 选择对比损失贡献最大的样本
    select_size = min(batch_size, len(X_unlabeled))
    selected_indices = np.argsort(loss_contributions)[-select_size:]
    selected_samples = X_unlabeled[selected_indices]
    selected_labels = y_unlabeled[selected_indices]

    # 更新标注集和未标注数据池
    X_labeled = np.vstack([X_labeled, selected_samples])
    y_labeled = np.hstack([y_labeled, selected_labels])
    X_unlabeled = np.delete(X_unlabeled, selected_indices, axis=0)
    y_unlabeled = np.delete(y_unlabeled, selected_indices, axis=0)

    print(f"Labeled samples: {len(X_labeled)}")

print("All unlabeled samples have been labeled!")

# ======= 6. 模型测试 ======= #
test_dataset = CustomDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, labels in loader:
            embeddings = model(data)
            predictions = torch.argmax(embeddings, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
