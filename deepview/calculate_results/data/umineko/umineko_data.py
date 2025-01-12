import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd


from scipy.interpolate import interp1d
import math
from datetime import datetime
from tqdm import tqdm
from sklearn.manifold import TSNE
import copy
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from deepview.generate_training_dataset.trainingsetmanipulation import (
    format_timestamp,
    process_gps)
from deepview.generate_training_dataset.utils import resampling
from deepview.clustering_pytorch.datasets.factory import sliding_window

from deepview.calculate_results.models.utils import (
    Resnet,
     load_weights,
adjust_learning_rate,
MSEloss,
    AE_train_time_series_resnet,
AE_eval_time_series,
Autoencoder1d,
Autoencoder2d,
Autoencoder3d,
EarlyStopper
)


# get label id from label string
label_dict = {
    'ground_stationary': 0,  #  if a bird is resting on the ground, including standing
    'ground_active': 0,
    'stationary': 1,  #  if a bird is resting on the sea surface
    'preening': 1,
    'bathing': 2,
    'bathing_poss': 2,
    'body_shaking': 2,
    # 'flight_take_off': 4,
    # 'flight_cruising': 3,
    'flying_active': 3,  # flying active -> flapping
    'flying_passive': 4,  # flying passive -> flying without flapping, including gliding
    'foraging': 5,
    'poss_foraging': 5,
    'foraging_fish_poss': 5,
    'foraging_insect_poss': 5,
    'forgaing_insect': 5,
    'foraging_non-fish': 5,
    'foraging_steal': 5,
    'foraging_poss': 5,
    'foraging_dive': 5,
    # 'surface_seizing': 11,
    'unknown': -2,  #-1
}

label_dict_origin = {
    'ground_stationary': 0,  #  if a bird is resting on the ground, including standing
    'stationary': 0,  #  if a bird is resting on the sea surface
    'preening': 0,
    'bathing': 1,
    'bathing_poss': 1,
    'flight_take_off': 4,
    'flight_cruising': 3,
    'flying_active': 2,  # flying active -> flapping
    'flying_passive': 2,  # flying passive -> flying without flapping, including gliding
    'foraging': 5,
    'poss_foraging': 5,
    'foraging_fish_poss': 6,
    'foraging_insect_poss': 7,
    'forgaing_insect': 7,
    'foraging_non-fish': 8,
    'foraging_steal': 9,
    'foraging_poss': 9,
    'foraging_dive': 10,
    'surface_seizing': 11,
    'body_shaking': 12,
    'ground_active': 13,
    'unknown': -2,  #-1
}

class data_loader_umineko(Dataset):
    def __init__(self, samples, labels, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels = torch.tensor(labels)  # check data type

    def __getitem__(self, index):
        target = self.labels[index]
        sample = self.samples[index]
        return sample, target

    def __len__(self):
        return len(self.labels)

def get_info_from_csv(p):
    # 使用 os.path.basename() 获取路径中的最后一个文件名
    file_name = os.path.basename(p)
    # file_name
    # 去掉扩展名
    file_name_without_extension = file_name.rsplit('.', 1)[0]

    # 按下划线分割
    parts = file_name_without_extension.split('_')

    # 提取所需的信息
    # 假设你需要的固定模式是：<name><year>_<other>_<code>_<id>
    name_year = parts[0]  # 如 'Umineko2022'
    code = parts[-2]  # 如 'LB09'

    # 将 'Umineko2022' 分成 'Umineko' 和 '2022'
    name = ''.join(filter(str.isalpha, name_year))  # 提取字母部分
    year = ''.join(filter(str.isdigit, name_year))  # 提取数字部分

    # 将提取的信息放入数组
    result = [name.lower(), int(year), code]
    return result

def get_backid_samplerate(back_label_path, result):

    # 因为有些传感器的位置不同，需要另一个csv文件标记
    # back_label_path = r'D:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
    back_label_pd = pd.read_csv(back_label_path)

    [species, year, tag] = result
    # filtering
    filtered_rows = back_label_pd[(back_label_pd['species'] == species) &
                                  (back_label_pd['animal_tag'] == tag)  &
                                  (back_label_pd['year'] == year)]

    # Get the values from column C for the filtered rows
    result_values = [0, 0]
    result_values[0] = filtered_rows['back'].values[0]
    result_values[1] = filtered_rows['acc_sampling_rate'].values[0]
    return result_values


def read_umineko_path(umi_root_path):

    # umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\umineko\v.1.0.0'
    # because it contains data of three years, try to extract data separately
    year_list = [2018, 2019, 2022]

    # get all filenames
    file_paths = {}
    # Iterate over all files in the directory
    for filename in os.listdir(umi_root_path):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            file_path = os.path.join(umi_root_path, filename)
            # get file paths and save them by year
            for y in year_list:
                if str(y) in filename:
                    if file_paths.get(y) is not None:
                        file_paths[y].append(file_path)
                    else:
                        file_paths.setdefault(y, []).append(file_path)
            # # Read the CSV file into a DataFrame
            # df = pd.read_csv(file_path)
            # # Append the DataFrame to the list
            # dataframes.append(df)
    return file_paths

def fill_gps(df):
    gps_df = df[['timestamp', 'latitude', 'longitude']]
    gps_df = gps_df.dropna()
    gps_ready, _ = process_gps(gps_df)
    gps_ready = gps_ready.bfill().ffill()
    df_ready = pd.merge(df, gps_ready.drop(columns=['latitude', 'longitude']), on='timestamp', how='outer')
    df_ready[['GPS_velocity', 'GPS_bearing']] = df_ready[['GPS_velocity', 'GPS_bearing']].fillna(method='bfill')
    df_ready[['GPS_velocity', 'GPS_bearing']] = df_ready[['GPS_velocity', 'GPS_bearing']].fillna(method='ffill')
    # selected_columns = ['GPS_velocity', 'GPS_bearing',
    #                     'label_id']  # without timestamps
    # df_selected = df_ready[selected_columns]
    # df_ffill = df_selected.fillna(method='bfill')
    # df_ffill = df_ffill.fillna(method='ffill')
    return df_ready

def fill_pressure(df):
    gps_df = df[['timestamp', 'pressure']]
    gps_ready = gps_df.bfill().ffill()
    df_ready = pd.merge(df, gps_ready.drop(columns=['pressure']), on='timestamp', how='outer')
    df_ready['pressure'] = df_ready['pressure'].fillna(method='bfill')
    return df_ready

def fill_temperature(df):
    gps_df = df[['timestamp', 'temperature']]
    gps_ready = gps_df.bfill().ffill()
    df_ready = pd.merge(df, gps_ready.drop(columns=['temperature']), on='timestamp', how='outer')
    df_ready['temperature'] = df_ready['temperature'].fillna(method='bfill')
    return df_ready

def read_umineko_data(back_label_path, file_paths, standardHz=25):

    # output dflist 和 dfcleanlist 分别是原始数据(label+unlabel data)和标签数据(label data)

    # read csv files as dataframe
    ## remove nan rows

    # Initialize an empty DataFrame
    df_list, df_clean_list = [], []
    # standardHz = 25

    for year, path_list in file_paths.items():
        df = pd.DataFrame()
        df_cleaned = pd.DataFrame()
        # for p in path_list[:2]:
        for p in path_list:
            # if select_year != str(year):
            #     continue
            tp = pd.read_csv(p, dtype={"logger_id": "string",
                                       "animal_tag": "string",
                                       "timestamp": "string",
                                       "acc_x": "float",
                                       "acc_y": "float",
                                       "acc_z": "float",
                                       # "latitude": "float",
                                       # "longitude": "float",
                                       # "gps_status": "string",
                                       "gyro_x": "float",
                                       "gyro_y": "float",
                                       "gyro_z": "float",
                                       # "mag_x": "float",
                                       # "mag_y": "float",
                                       # "mag_z": "float",
                                       # "illumination": "float",
                                       # "pressure": "float",
                                       # "temperature": "float",
                                       # "activity_class": "int",
                                       "label": "string",
                                       })

            tp = format_timestamp(tp)  # 在这里timestamp字符串生成datetime和unixtime

            tp['year'] = int(year)  # add a new column

            csv_info = get_info_from_csv(p)
            [backID, sample_rate] = get_backid_samplerate(back_label_path, csv_info)
            tp['back'] = int(backID)  # add a backID column, identify position of sensor
            tp['species'] = csv_info[0]  # add a species(umineko) column

            # same sampling rate of each csv file, now assume 25Hz
            if sample_rate != standardHz:
                print('resampling to 25Hz...')
                # 先把timestamp保存下来
                save_t = tp.timestamp.values
                # delete timestamp
                tp = tp.drop('timestamp', axis=1)

                # If the data frame contains more than one minute of recordings,
                # use the data frame, otherwise, discard it.
                tp.reset_index(inplace=True)
                # if check_df == True:
                #     display(df_list[i].head(5))
                print('length of previous data is %d' % len(tp))
                tp = resampling(
                    df=tp,
                    intermediate_sampling_rate=sample_rate,
                    output_sampling_rate=standardHz
                )
                print('length of new data is %d' % len(tp))

                tp['timestamp'] = tp['datetime'].apply(
                    lambda x: datetime.utcfromtimestamp(float(x / 1000000000.0)).strftime('%Y-%m-%dT%H:%M:%S.%f')[
                              :-3] + 'Z')

            tp['label_id'] = tp['label'].map(label_dict)
            # 将df_list没有标签的位置赋值-2，标记灰色，其他label正常标记。观察是否有label的标记能在不同灰色cluster中
            # Replace NaN with -2 in column A
            tp['label_id'] = tp['label_id'].fillna(-2)
            # 需要给其他传感器数据做填充，否则损失太多数据
            tp = fill_gps(tp)
            tp = fill_temperature(tp)
            tp = fill_pressure(tp)
            tp_cleaned = tp.dropna(subset=['label'])

            # If the DataFrame is empty, initialize it with the first data
            if df.empty:
                df = tp
                df_cleaned = tp_cleaned
            else:
                # Otherwise, concatenate the new data to the existing DataFrame
                df = pd.concat([df, tp], ignore_index=True)
                df_cleaned = pd.concat([df_cleaned, tp_cleaned], ignore_index=True)

            # break
        df_list.append(df)
        df_clean_list.append(df_cleaned)
        # break

    return df_list, df_clean_list

# def extract_data_from_year_back(df_list, df_clean_list, backid):
def extract_data_from_year_back(selected_df, selected_clean_df, backid):
    selected_df = selected_df[
        # (selected_df['year'] == 2018) &  # unique
                              (selected_df['back'] == backid) &
                              (selected_df['species'] == 'umineko')]  #unique


    selected_clean_df = selected_clean_df[
        # (selected_clean_df['year'] == 2018) &
                              (selected_clean_df['back'] == backid) &
                              (selected_clean_df['species'] == 'umineko')]

    return selected_df, selected_clean_df

def get_accel_batch_data(df_ready, len_sw, batch_size, device):
    selected_columns = ['acc_x', 'acc_y', 'acc_z',
                        'label_id']  # without timestamps
    tmp_b = sliding_window(df_ready[selected_columns], len_sw)

    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    train_set_r = data_loader_umineko(data_b, label_b, device=device)
    train_loader = DataLoader(train_set_r, batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader


# todo 这段function有修改
def get_gps_batch_data(df_ready, len_sw, batch_size, device):
    # # 先保留共同的unixtime column，再删除gps没有的row计算速度和角度，最后拼回去
    # gps_df = df[['timestamp','latitude','longitude']]
    # gps_df = gps_df.dropna()
    # gps_ready, _ = process_gps(gps_df)
    # gps_ready = gps_ready.bfill().ffill()
    # df_ready = pd.merge(df, gps_ready.drop(columns=['latitude','longitude']), on='timestamp', how='outer')

    selected_columns = ['GPS_velocity', 'GPS_bearing',
                        'label_id']  # without timestamps
    df_selected = df_ready[selected_columns]
    df_ffill = df_selected.fillna(method='bfill')
    df_ffill = df_ffill.fillna(method='ffill')
    tmp_b = sliding_window(df_ffill, len_sw)

    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    train_set_r = data_loader_umineko(data_b, label_b, device=device)
    train_loader = DataLoader(train_set_r, batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader

def get_gps_raw_batch_data(df_ready, len_sw, batch_size, device):
    selected_columns = ['latitude', 'longitude',
                        'label_id']  # without timestamps
    df_selected = df_ready[selected_columns]
    df_ffill = df_selected.fillna(method='bfill')
    df_ffill = df_ffill.fillna(method='ffill')
    tmp_b = sliding_window(df_ffill, len_sw)

    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    train_set_r = data_loader_umineko(data_b, label_b, device=device)
    train_loader = DataLoader(train_set_r, batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader

def get_temperature_batch_data(df_ready, len_sw, batch_size, device):
    selected_columns = ['temperature',
                        'label_id']  # without timestamps
    df_selected = df_ready[selected_columns]
    df_ffill = df_selected.fillna(method='bfill')
    df_ffill = df_ffill.fillna(method='ffill')
    tmp_b = sliding_window(df_ffill, len_sw)

    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    train_set_r = data_loader_umineko(data_b, label_b, device=device)
    train_loader = DataLoader(train_set_r, batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader

def get_pressure_batch_data(df_ready, len_sw, batch_size, device):
    selected_columns = ['pressure',
                        'label_id']  # without timestamps
    df_selected = df_ready[selected_columns]
    df_ffill = df_selected.fillna(method='bfill')
    df_ffill = df_ffill.fillna(method='ffill')
    tmp_b = sliding_window(df_ffill, len_sw)

    # concatenate list
    data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
    label_b = tmp_b[:, :, -1]  # [B, Len]

    train_set_r = data_loader_umineko(data_b, label_b, device=device)
    train_loader = DataLoader(train_set_r, batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader

if __name__ == "__main__":
    # print("Hello, World!")

    back_label_path = r'D:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
    umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\umineko\v.1.0.0'
    file_paths = read_umineko_path(umi_root_path)

    # # data_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko.pkl'
    # year = '2018'
    # data_path = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'%year
    # if os.path.exists(data_path):
    #     # with open(data_path, 'rb') as f:
    #     #     tmp = pickle.load(f)
    #     #     df_list = tmp['raw_data']
    #     #     df_clean_list = tmp['labeled_data']
    #     tmp = np.load(data_path, allow_pickle='TRUE').item()
    #     df_list = tmp['raw_data']
    #     df_clean_list = tmp['labeled_data']
    # else:
    #     df_list, df_clean_list = read_umineko_data(back_label_path, file_paths, select_year=year)
    #     # save all data
    #     with open(data_path, 'wb') as f:
    #         pickle.dump({'raw_data': df_list,
    #                      'labeled_data': df_clean_list}, f)
    #     dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
    #     # np.save(dp % "_2018", dict(raw_data=df_list[0], labeled_data=df_clean_list[0]))
    #     # np.save(dp % "_2019", dict(raw_data=df_list[1], labeled_data=df_clean_list[1]))
    #     # np.save(dp % "_2022", dict(raw_data=df_list[2], labeled_data=df_clean_list[2]))
    #     np.save(data_path, dict(raw_data=df_list[0], labeled_data=df_clean_list[0]))  # list only len=0
    #
    # for loop for all umineko data

    df_list, df_clean_list = read_umineko_data(back_label_path, file_paths)
    for idx, year in enumerate(['2018', '2019', '2022']):
        dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
        np.save(dp % year, {'raw_data': df_list[idx], 'labeled_data': df_clean_list[idx]})
    print('')

    # selected_df, selected_clean_df = (
    #     extract_data_from_year_back(df_list, df_clean_list, 1))
    #
    # # selected_df['label_id'] = selected_df['label'].map(label_dict)
    # # selected_clean_df['label_id'] = selected_clean_df['label'].map(label_dict)
    #
    # # 将df_list没有标签的位置赋值-2，标记灰色，其他label正常标记。观察是否有label的标记能在不同灰色cluster中
    # # Replace NaN with -2 in column A
    # selected_df['label_id'] = selected_df['label_id'].fillna(-2)
    #
    # device = 'cuda'
    # len_sw = 300
    # sensor_type = 'pressure'
    # # train_loader = get_accel_batch_data(selected_clean_df,
    # #                                     len_sw,
    # #                                     batch_size=512,
    # #                                     device=device)
    #
    # # train_loader = get_gps_batch_data(selected_clean_df,
    # #                                     len_sw,
    # #                                     batch_size=512,
    # #                                     device=device)
    # # train_loader = get_gps_raw_batch_data(selected_clean_df,
    # #                                   len_sw,
    # #                                   batch_size=512,
    # #                                   device=device)
    # # train_loader = get_temperature_batch_data(selected_clean_df,
    # #                                       len_sw,
    # #                                       batch_size=512,
    # #                                       device=device)
    # train_loader = get_pressure_batch_data(selected_clean_df,
    #                                           len_sw,
    #                                           batch_size=512,
    #                                           device=device)
    #
    # print('')
    #
    # model = Autoencoder1d()
    #
    # # class_num = len(set(list(label_dict.values())))
    # # # Example usage
    # # model = Resnet(
    # #         output_size=class_num,
    # #         is_reconst=True,
    # #         len_sw=len_sw,
    # #         # is_simclr=True,
    # #         # is_eva=True,
    # #         resnet_version=1,
    # #                )
    # model = model.to(device)
    # # dirname = os.path.dirname(__file__)
    # # checkpoint = os.path.join(
    # # os.getcwd(), "model_check_point", "mtl_best.mdl"
    # # )
    # checkpoint = r'D:\code\DeepView\deepview\calculate_results\model_check_point\mtl_best.mdl'
    # print(checkpoint)
    # #
    # # load_weights(
    # # checkpoint, model, my_device=device, is_dist=True, name_start_idx=1
    # # )
    # #
    # #
    # criterion = MSEloss()
    # criterion = criterion.to(device)
    #
    # learning_rate = 0.0001
    # optimizer = optim.Adam(
    #     model.parameters(), lr=learning_rate, amsgrad=True
    # )
    # optimizer = optim.Adam(
    #             model.parameters(), lr=learning_rate, amsgrad=True
    #         )
    # lambda1 = lambda epoch: 1.0**epoch
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    #
    # # training
    # start_epoch = 0
    # num_epochs = 1800
    # # lr = 0.0001
    # early_stopper = EarlyStopper(patience=3, min_delta=10)
    # # for epoch in np.arange(n_epochs):
    # #     train_loss = train_one_epoch(model, train_loader)
    # #     validation_loss = validate_one_epoch(model, validation_loader)
    # #     if early_stopper.early_stop(validation_loss):
    # #         break
    # for epoch in tqdm(range(start_epoch, num_epochs)):
    #
    #     learning_rate = adjust_learning_rate(
    #         learning_rate, optimizer, epoch, p_scheduler='cosine', p_epochs=num_epochs)
    #
    #     losses = AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device)
    #
    #     if early_stopper.early_stop(losses):
    #         break
    #     if (epoch % 10 == 0) or (epoch == num_epochs - 1):
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    #
    # # print('Saving model at: ' + 'Resnet_ssl_pretrain_epoch%s' % str(epoch) \
    # #       + '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
    # # torch.save(model.state_dict(), 'Resnet_ssl_pretrain_epoch%s' % str(epoch) + \
    # #            '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
    #
    # print('Saving model at: ' + 'AE_ssl_pretrain_epoch%s_raw' % str(epoch) \
    #       + '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
    # torch.save(model.state_dict(), 'AE_ssl_pretrain_epoch%s_raw' % str(epoch) + \
    #            '_datalen%s_' % str(len_sw) +sensor_type+ '.pth')
    #
    # # full_model_path = r'D:\code\DeepView\deepview\calculate_results\data\Resnet_ssl_conv_pretrain_epoch79_datalen300_accel_labeldata.pth'
    # # out_channels = 32
    # # model = Resnet(output_size=class_num,
    # #     is_reconst=True,
    # #     len_sw= len_sw,
    # #     # is_simclr=True,
    # #     # is_eva=True,
    # #     resnet_version=1,)
    # # device = 'cuda'
    # # model = model.to(device)
    # # if torch.cuda.is_available():
    # #     model.load_state_dict(torch.load(full_model_path, weights_only=False))
    # # else:
    # #     model.load_state_dict(torch.load(full_model_path, weights_only=False, map_location=torch.device('cpu')))
    #
    #
    # # reconstruction result
    # representation_list, sample_list, pred_list, label_list = \
    #             AE_eval_time_series(train_loader, model, device)
    #
    # # tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    # repre_concat = np.concatenate(representation_list)
    # repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
    #
    # sample_concat = np.concatenate(sample_list)
    # sample_concat = sample_concat.transpose(0,2,1)
    # sample_reshape = sample_concat.reshape(-1, sample_concat.shape[-1])
    #
    # pred_concat = np.concatenate(pred_list)
    # pred_concat = pred_concat.transpose(0, 2, 1)
    # pred_reshape = pred_concat.reshape(-1, pred_concat.shape[-1])
    # fig, axes = plt.subplots(3, 1, figsize=(8, 6))
    # axes[0].plot(sample_reshape[5000:10000, 0], 'r', label='groundtruthY')
    # axes[0].plot(pred_reshape[5000:10000, 0], 'b-.', label='predictY')
    # axes[0].set_title('AE_Umineko2018_back1_%s'%sensor_type)
    # # axes[0].set_title('ResNet_SSL_pretrained_Reconstruct_Umineko2018_back1')
    # axes[0].set_xlabel('timestamp')
    # axes[0].set_ylabel(sensor_type)
    # axes[0].legend()
    #
    # axes[1].plot(sample_reshape[5000:10000, 0], 'r', label='groundtruthY')
    # # axes[1].plot(pred_reshape[10000:length, 1], 'b-.', label='predictY')
    # # axes[1].set_title('ResNet_SSL_pretrained_Reconstruct_Umineko2018_back1')
    # axes[1].set_xlabel('timestamp')
    # axes[1].set_ylabel(sensor_type)
    # axes[1].legend()
    #
    # # axes[2].plot(sample_reshape[10000:length, 1], 'r', label='groundtruthY')
    # axes[2].plot(pred_reshape[5000:10000, 0], 'b-.', label='predictY')
    # # axes[2].set_title('ResNet_SSL_pretrained_Reconstruct_Umineko2018_back1')
    # axes[2].set_xlabel('timestamp')
    # axes[2].set_ylabel(sensor_type)
    # axes[2].legend()
    #
    # # Adjust layout
    # plt.tight_layout()
    # # Show the figure
    # # plt.show()
    # # plt.savefig('AEreconst_only_labeled.png')
    # plt.savefig('AE_Umineko2018_back1_labeled_%s.png'%sensor_type)
    # plt.close()
