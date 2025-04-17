import torch
import torch.nn as nn
import shap
import numpy as np
import matplotlib.pyplot as plt
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_13s_SHAP,
)

import torch
import torch.nn as nn
import shap


# 假设你已经定义好了 SimpleNN_13s_SHAP 并且去掉了最后的 Softmax


# -----------------------------------------------------------
# 下面是使用 SHAP 的示例流程
# -----------------------------------------------------------

# 1. 初始化模型并加载训练好的参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleNN_13s_SHAP(input_dim=128 * 6, number_classes=5)
# model.load_state_dict(torch.load("your_trained_model.ckpt", map_location=device))
model.to(device)
model.eval()

# 2. 准备数据: background_data 用于 SHAP 估计基准分布, test_data 用于解释
#    这里假设你已经有一个 DataLoader 或者可以直接拿到张量
#    data 维度应为 (batch_size, 4, seq_len)，4 个通道分别是 3 个加速度和 1 个压力
#    这里仅作示例，你需要根据自己项目来获取真实的数据

# 例如从你的 DataLoader 中取一部分做 background_data
# background_data, _ = next(iter(your_dataloader))  # (batch_bg, 4, 50) for example
# background_data = background_data.to(device)

# 同理，取另一个 batch 做 test_data
# test_data, _ = next(iter(your_dataloader))  # (batch_test, 4, 50)
# test_data = test_data.to(device)

# 假设我们演示时，直接用同一个 batch 当 background 和 test，这里仅作演示
background_data = torch.randn((8, 4, 50)).to(device)
test_data = torch.randn((8, 4, 50)).to(device)

# 3. 构建 DeepExplainer
explainer = shap.GradientExplainer(model, background_data)

# 4. 计算 test_data 对应的 SHAP 值
#    如果是多分类， shap_values 通常返回一个 list，每个元素是 (batch_test, 4, seq_len)，
#    分别对应每个类别的 SHAP 值
shap_values = explainer.shap_values(test_data)

# 5. 可视化 / 进一步分析
#    如果是多分类，你可以指定自己关心的某一个 class 的 shap_values
#    比如取 shap_values[0] (对应第0类) 或者取 shap_values[1] 等
#    这里为了演示，假设我们只取第 0 类
shap_vals_for_class0 = shap_values[0]  # shape: (batch_test, 4, 50)

# 6. 分别查看加速度和压力的贡献
#    对于加速度(3 通道)： shap_vals_for_class0[:, :3, :]
#    对于压力(1 通道)：   shap_vals_for_class0[:, 3:, :]

acc_shap_vals = shap_vals_for_class0[:, :3, :]  # (batch_test, 3, seq_len)
press_shap_vals = shap_vals_for_class0[:, 3:, :]  # (batch_test, 1, seq_len)

print("加速度传感器 SHAP 值形状:", acc_shap_vals.shape)
print("压力传感器 SHAP 值形状:", press_shap_vals.shape)

# 7. 你可以使用 shap 的内置可视化函数做一些简要的可视化：
# shap.image_plot 或者 shap.force_plot 等
# 但注意，这些函数大多是针对图像或表格数据做的，你需要先把数据reshape为可解释的形式。
# 或者你可以用 matplotlib 之类自己定制可视化。
