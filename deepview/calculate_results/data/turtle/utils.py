
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import pickle


#######data processing#######
# original_freq = 20
# target_freq = 25


def resample_data(df_h5, original_freq=20, target_freq=25):
    # === Step 2: 重采样至 25Hz ===
    # original_freq = 20
    # target_freq = 25
    duration = len(df_h5) / original_freq
    new_len = int(duration * target_freq)

    df_resampled = pd.DataFrame()
    for col in df_h5.columns:
        df_resampled[col] = np.interp(
            np.linspace(0, len(df_h5) - 1, new_len),
            np.arange(len(df_h5)),
            df_h5[col]
        )

    df_resampled['Time'] = np.linspace(0, duration, new_len)

    return df_resampled

def read_h5file(h5_file_path, original_freq=20, target_freq=25):
    # === Step 1: 加载 H5 数据 ===
    # h5_file_path = os.path.join(root_path, "CC-07-48_08-04-2019_3.h5")
    with h5py.File(h5_file_path, 'r') as h5_file:
        sensor_data = np.array(h5_file['data'])

    columns = ['AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ', 'Depth']
    df_h5 = pd.DataFrame(sensor_data, columns=columns)

    if original_freq != target_freq:
        # Resample the data if original frequency is not equal to target frequency
        df_h5 = resample_data(df_h5)
    return df_h5

def read_csvfile(csv_file_path, target_freq=25):
    # === Step 3: 加载 CSV 行为标签 ===
    # csv_file_path = os.path.join(root_path, "Behaviors_CC-07-48_08-04-2019_3.csv")
    df_behaviors = pd.read_csv(csv_file_path, sep=';')

    df_behaviors.rename(columns={
        'Start (s)': 'StartSync',
        'Stop (s)': 'EndSync'
    }, inplace=True)
    # === Step 4: 构造行为标签时间轴 ===
    df_behavior_timeline = pd.DataFrame(columns=['Time', 'Behavior'])
    for _, row in df_behaviors.iterrows():
        start = row['StartSync']
        end = row['EndSync']
        label = row['Behavior']
        if pd.notnull(start) and pd.notnull(end) and end > start:
            timeline = pd.DataFrame({
                'Time': np.linspace(start, end, int((end - start) * target_freq)),
                'Behavior': label
            })
            if not timeline.empty:
                df_behavior_timeline = pd.concat([df_behavior_timeline, timeline], ignore_index=True)

    return df_behavior_timeline

def merge_label2data(df_behavior_timeline, df_resampled, target_freq=25):
    # === Step 5: 行为标签与时间对齐 ===
    df_behavior_full = pd.merge_asof(
        df_resampled[['Time']],
        df_behavior_timeline.sort_values('Time'),
        on='Time',
        direction='backward'
    )

    # 合并为最终 DataFrame
    df_final = pd.concat([df_resampled, df_behavior_full['Behavior']], axis=1)
    df_final.rename(columns={'Behavior': 'Label'}, inplace=True)
    return df_final

##################################test code#########################################
# # 替换为你自己的行为标签CSV路径列表，或使用通配符匹配
# root_path = r'D:\logbot-data\turtle'
# csv_files = glob.glob(os.path.join(root_path, "*.csv"))
# # h5_files = glob.glob(os.path.join(root_path, "*.h5"))
#
# # 初始化行为时间统计表
# data_list = []
#
# for file in csv_files:
#     print(file)
#     # 去掉 "Behaviors_" 并替换 ".csv" 为 ".h5"
#     h5_file = file.replace('Behaviors_', '').replace('.csv', '.h5')
#     df_resampled = read_h5file(h5_file, original_freq=20, target_freq=25)
#     df_behavior_timeline = read_csvfile(file, target_freq=25)
#     df_final = merge_label2data(df_behavior_timeline, df_resampled, target_freq=25)
#     data_list.append(df_final)
#
#
# with open('turtle.pkl', 'wb') as f:
#     for df in data_list:
#         pickle.dump(df, f)
