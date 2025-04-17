import numpy as np
import os
import pandas as pd
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader

from deepview.generate_training_dataset.trainingsetmanipulation import (
    format_timestamp,
    process_gps)
from deepview.generate_training_dataset.utils import resampling
from deepview.clustering_pytorch.datasets.factory import sliding_window


# label_dict = {
#     'ground_stationary': 0,  #  if a bird is resting on the ground, including standing
#     'ground_active': 0,
#     'stationary': 1,  #  if a bird is resting on the sea surface
#     'preening': 1,
#     'bathing': 2,
#     'bathing_poss': 2,
#     'body_shaking': 2,
#     # 'flight_take_off': 4,
#     # 'flight_cruising': 3,
#     'flying_active': 3,  # flying active -> flapping
#     'flying_passive': 4,  # flying passive -> flying without flapping, including gliding
#     'foraging': -2,# 5,
#     'poss_foraging': -2,# 5,
#     'foraging_fish_poss': -2,# 5,
#     'foraging_insect_poss': -2,# 5,
#     'forgaing_insect': -2,# 5,
#     'foraging_non-fish': -2,# 5,
#     'foraging_steal': -2,# 5,
#     'foraging_poss': -2,# 5,
#     'foraging_dive': -2,# 5,
#     # 'surface_seizing': 11,
#     'unknown': -2,  #-1
# }

label_dict = {
'stationary': 0,
'preening': 1,
'bathing': 2,
'flight_take_off': 3,
'flight_cruising': 4, #3,
'foraging_dive': 5, #4,
'surface_seizing': 6, #4,
'body_shaking': 7, #-2,
'unknown': -2
}

year_list = [2018, 2020, 2021, 2022]

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
    code = parts[-2]  # animal_tag, 如 'LB09'

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
                                  (back_label_pd['animal_tag'].str.contains(tag))  &
                                  (back_label_pd['year'] == year)]

    # Get the values from column C for the filtered rows
    result_values = [0, 0]
    result_values[0] = filtered_rows['back'].values[0]
    result_values[1] = filtered_rows['acc_sampling_rate'].values[0]
    return result_values


def read_umineko_path(umi_root_path):

    # umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\umineko\v.1.0.0'
    # because it contains data of three years, try to extract data separately


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
    umi_root_path = r'D:\logbot-data\BioTaggerData\export\raw\omizunagidori\v1.0.0'
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
    # for idx, year in enumerate(['2017', '2018', '2020', '2021', '2022']):
    for idx, year in enumerate(year_list):
        dp = r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori_%s.npy'
        np.save(dp % str(year), {'raw_data': df_list[idx], 'labeled_data': df_clean_list[idx]})
    print('')

