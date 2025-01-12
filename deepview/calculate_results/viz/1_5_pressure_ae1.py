import os

import torch.optim as optim
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from deepview.calculate_results.data.umineko.umineko_data import (
    read_umineko_path,
    extract_data_from_year_back,
    label_dict,
)

from deepview.calculate_results.models.utils import (
    sliding_window,
    data_loader_umineko,
    MSEloss_weighted,
    # torch,
    AE_eval_time_series,
    AE_train_time_series_resnet,
    # np,
    # tqdm
    Autoencoder1d,
    plot_reconstruction_result,
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

# Set a fixed random seed
seed_value = 2025
set_random_seed(seed_value)

def forward_fill(arr):
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]
    return arr

def gaussian_std(X):
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized

back_label_path = r'D:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\umineko\v.1.0.0'
file_paths = read_umineko_path(umi_root_path)

year = '2018'
data_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy' % year
if os.path.exists(data_path):
    tmp = np.load(data_path, allow_pickle='TRUE').item()
    df_list = tmp['raw_data']
    df_clean_list = tmp['labeled_data']

selected_df, selected_clean_df = (
    extract_data_from_year_back(df_list, df_clean_list, 1))

selected_df['label_id'] = selected_df['label'].map(label_dict)
selected_clean_df['label_id'] = selected_clean_df['label'].map(label_dict)

# 将df_list没有标签的位置赋值-2，标记灰色，其他label正常标记。观察是否有label的标记能在不同灰色cluster中
# Replace NaN with -2 in column A
selected_df['label_id'] = selected_df['label_id'].fillna(-2)

device = 'cuda'
len_sw = 300
sensor_type = 'pressure'
activity_label = False
if activity_label:
    selected_columns = ['pressure', 'label_id', 'animal_tag']  # without timestamps，activity labels
else:
    selected_df['animal_tag_id'] = pd.factorize(selected_df['animal_tag'])[0]
    selected_columns = ['pressure', 'animal_tag_id', 'label_id']  # 将animal tag作为label，判断鸟的embedding是否分开

selected_np = selected_df[selected_columns].values

############################################

data_np = selected_np[:, :-2]
data_np[:, 0] = forward_fill(data_np[:, 0].astype(float))
percentile_99 = np.percentile(data_np[:, 0], 99.9)
data_np[data_np[:, 0] > percentile_99, 0] = percentile_99
percentile_1 = np.percentile(data_np[:, 0], 0.1)
data_np[data_np[:, 0] < percentile_1, 0] = percentile_1
# len(np.where(np.isnan(data_np[:, 0].astype(float))==True)[0])
# plt.plot(data_np[:,0])
# plt.show()
# plt.cla()
tmp_b_stand = gaussian_std(data_np)

##########################################

selected_np[:, :-2] = tmp_b_stand

tmp_b = sliding_window(selected_np[:, :-1], len_sw, len_sw)


##############################################################
# concatenate list
data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
label_b = tmp_b[:, :, -1]  # [B, Len]

batch_size = 512
train_set_r = data_loader_umineko(data_b.astype(float), label_b.astype(int), device=device)
train_loader = DataLoader(train_set_r, batch_size=batch_size,
                          shuffle=False, drop_last=False)

# test loaders
selected_df[selected_columns] = selected_np  # update sensor columns
# prepare a list to save data of each csv file for generating testloader (animala_tag)
grouped = selected_df.groupby('animal_tag')  # 根据某一列进行分组
df_list = [group[selected_columns] for _, group in grouped]  # 将每个分组的 DataFrame 保存到一个列表中

test_loaders = []
# countl = 0
for testdf in df_list:
    selected_np = testdf[selected_columns].values

    tmp_b = sliding_window(selected_np[:, :-1], len_sw, len_sw)
    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    batch_size = 512
    test_set_r = data_loader_umineko(data_b.astype(float), label_b.astype(int), device=device)
    tmp_loader = DataLoader(test_set_r, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    test_loaders.append(tmp_loader)
    print('testloader generator')

model = Autoencoder1d()
model = model.to(device)
full_model_path = r'D:\code\DeepView\deepview\calculate_results\viz\AE_reconstruct_epoch799_datalen300_pressure.pth'
if torch.cuda.is_available():
    model.load_state_dict(torch.load(full_model_path, weights_only=False))
else:
    model.load_state_dict(torch.load(full_model_path, weights_only=False, map_location=torch.device('cpu')))

criterion = MSEloss_weighted()

criterion = criterion.to(device)

learning_rate = 0.001
# optimizer = optim.Adam(
#     model.parameters(), lr=learning_rate, amsgrad=True
# )
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# training
start_epoch = 0
num_epochs = 1
# lr = 0.0001
training_loss = []
for epoch in tqdm(range(start_epoch, num_epochs)):

    # learning_rate = adjust_learning_rate(
    #     learning_rate, optimizer, epoch, p_scheduler='cosine', p_epochs=num_epochs)

    losses = AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device)
    training_loss.append(np.average(losses))
    if (epoch == num_epochs - 1):
    # if (epoch % 100 == 0) or (epoch == num_epochs - 1):
        #     print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
        # reconstruction result
        representation_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(train_loader, model, device)
        plot_reconstruction_result(representation_list, sample_list, pred_list, label_list,
                                   '%s_train_epoch_%s' % (sensor_type, str(epoch)))

        with (torch.no_grad()):
            model.eval()
            for idx, test_loader in enumerate(test_loaders):
                representation_list, sample_list, pred_list, label_list \
                    = AE_eval_time_series(test_loader, model, device)
                plot_reconstruction_result(representation_list, sample_list, pred_list, label_list,
                                           '%s_test_%s_epoch_%s' % (sensor_type, str(idx), str(epoch)))

print('Saving model at: ' + 'AE_reconstruct_epoch%s' % str(epoch) \
      + '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
torch.save(model.state_dict(), 'AE_reconstruct_epoch%s' % str(epoch) + \
           '_datalen%s_' % str(len_sw) + sensor_type + '.pth')
