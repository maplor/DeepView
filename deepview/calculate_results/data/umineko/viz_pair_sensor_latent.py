import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

from deepview.calculate_results.models.utils import (
    data_loader_umineko,
    sliding_window,
    AE_eval_time_series,
    majority_value,
)
from scipy.stats import entropy
from deepview.calculate_results.data.umineko.train_func import (
    train_model,
    evaluate_model,
    evaluate_supContrast_model,
    plot_confusion_matrix,
    unfreeze_encoders,
    freeze_encoders,
    freeze_projector,
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN,  # 31
    SimpleNN_33s,
    SimpleNN_11s,
    ContrastiveLoss,
    SupContrastiveLoss,
)
from deepview.calculate_results.data.umineko.cluster_method import (
    vis_scatter_label_2d,
    plot_func
)
from deepview.calculate_results.models.utils import (
    gaussian_std,
    set_random_seed,
    read_sensor_data,
    process_sensor_np,
)

all_df = read_sensor_data()
# 删除无标签的数据
selected_df = all_df[all_df.label_id != -2]
# columns = ['acc_x', 'acc_y', 'acc_z', 'temperature', 'label_id']
# columns = ['acc_x', 'acc_y', 'acc_z',
#            'gyro_x', 'gyro_y', 'gyro_z',
#            'label_id']
columns = ['acc_x', 'acc_y', 'acc_z',
           'acc_x', 'acc_y', 'acc_z',
           'label_id']
# columns = ['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']
# columns = ['temperature', 'pressure', 'label_id']
selected_np = process_sensor_np(selected_df, columns)

len_sw = 50
device = 'cuda'
sensor_type = 'AccelAccel'
# sensor_type = 'TempPress'
# sensor_type = 'AccelPress'
# sensor_type = 'AccelGyro'

tmp_b = sliding_window(selected_np, len_sw, len_sw)
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]

batch_size = 1024
train_set_r = data_loader_umineko(data_b.astype(float),
                                  label_b.astype(int), device=device)
train_loader = DataLoader(train_set_r, batch_size=batch_size,
                          shuffle=False, drop_last=False)


# model = SimpleNN()
# model = SimpleNN_11s()
model = SimpleNN_33s()
model = model.to(device)
classify_criterion = nn.CrossEntropyLoss()
Contrast_criterion = ContrastiveLoss()
supContrast_criterion = SupContrastiveLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

for epoch in range(20):
    model, _ = train_model(model, train_loader,
                           supContrast_criterion, optimizer,
                          epochs=20, device=device, if_contrast=True)

    repres_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(train_loader, model, device)
    # ari, nmi, fmi, silhouette = (
    #     vis_scatter_label_2d(repres_list,
    #                          label_list, iteration,
    #                          sensor_type+name_label,
    #                          plot_flag=plot_flag))
    ari, nmi, fmi, silhouette = (
        plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
                  epoch, sensor_type+'_sup8'))
    print('')