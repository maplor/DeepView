import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from scipy.interpolate import interp1d,CubicSpline
import math
import copy
import random
from deepview.calculate_results.data.umineko.umineko_data import label_dict
import re
# import numpy as np
# import pandas as pd
from typing import List, Tuple, Dict, Union, Callable, Any


labeldict_findstr = {
                     0: 'ground_stationary',
                     1: 'stationary',
                     2: 'bathing',
                     3: 'flying_active',
                     4: 'flying_passive'}

labeldict_findstr_large = {
    0: 'ground_stationary',
    1: 'stationary',
    2: 'bathing_flypass',
    3: 'flying_active',
    # 4: 'bathing_flypass'
}

labeldict_findstr_omizu = {
                     0: 'stationary',
                     1: 'bathing',
                     2: 'take_off',
                    3: 'cruising',  # 3,
                    4: 'foraging_dive',  # 4,
                    5: 'dipping',  #4, # dipping
                    }

labeldict_findstr_omizu_org = {
0: 'stationary',
1: 'preening', #
2: 'bathing',
3: 'flight_take_off',
4: 'flight_cruising', #3,
5: 'foraging_dive', #4,
6: 'surface_seizing', #4, # dipping
}
# omizu
# stationary,bathing,take-off,cruising flight, foraging dive, dipping

# labeldict_findstr_turtle = {
# 0:'Resting in flow',
# 1:'Resting',
# 2:'Swimming horizontally',
# 3:'Stay in surface',
# 4:'Swimming descent',
# 5:'Swimming ascent',
# 6:'other',
# }

labeldict_findstr_turtle = {  # refer to viz_latent_subact.py
                     0: 'Resting',
                     1: 'Swimming',
                     2: 'Stay in surface',
                     3: 'Gliding',
                     4: 'Feeding',
                     5: 'Scratching',
                     6: 'Breathing',
}


labeldict_findstr_bear = {
                0:'dig',
                1:'eat',
                2:'groom',
                3:'head_shake',
                4:'pounce',
                5:'rest',
                6:'roll',
                7:'run',
                8:'swim',
                9:'walk',
}

label_colors = {
    0: "#5470C6",  # 深蓝 - Attack
    1: "#91CC75",  # 绿色 - Investigation
    2: "#FAC858",  # 金黄 - Mount
    3: "#EE6666",  # 红色 - Category 3
    4: "#73C0DE",  # 天蓝 - Category 4
    5: 'brown',     # 棕色
    6: 'pink',      # 粉色
    7: 'cyan',      # 青色
    8: 'magenta',   # 品红
    9: 'lime',      # 青柠色
    10: 'teal',     # 蓝绿色
    11: 'violet',   # 紫罗兰色
    12: 'gold',     # 金色
    13: 'coral',    # 珊瑚色
    14: 'salmon'    # 三文鱼色
}

back_label_path = r'E:\logbot-data\BioTaggerData\masterLabelsByOtsuka\animal_id.csv'
back_label_pd = pd.read_csv(back_label_path)

# def set_random_seed(seed):
#     # Set seed for Python's random module
#     random.seed(seed)
#
#     # Set seed for NumPy
#     np.random.seed(seed)
#
#     # Set seed for PyTorch
#     torch.manual_seed(seed)
#
#     # If using CUDA, set seed for GPU as well
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)  # For multi-GPU setups



def get_backid_samplerate(result):
    [species, year, tag] = result
    # filtering
    filtered_rows = back_label_pd[(back_label_pd['species'] == species) &
                                  (back_label_pd['animal_tag'] == tag) &
                                  (back_label_pd['year'] == year)]

    # Get the values from column C for the filtered rows
    result_values = [0, 0]
    result_values[0] = filtered_rows['back'].values[0]
    result_values[1] = filtered_rows['acc_sampling_rate'].values[0]
    return result_values


##########################raw data processing###################################
def read_sensor_data():
    raw_data, labeled_data = [], []
    for year in ['2018', '2022']:
        dp = r'D:\code\DeepView\deepview\calculate_results\data\umineko_%s.npy'
        data = np.load(dp % year, allow_pickle=True).item()
        # Access the individual components
        raw_ = data['raw_data']
        labeled_ = data['labeled_data']
        if year == '2018':
            df_raw_2018 = raw_
            df_2018 = labeled_
        # elif year=='2019':
        #     df_raw_2019 = raw_
        #     df_2019 = labeled_
        elif year == '2022':
            df_raw_2022 = raw_
            df_2022 = labeled_
        else:
            print('Error: year not found')
            break

    selected_df = pd.concat([df_raw_2018, df_raw_2022], ignore_index=True)
    selected_df['filename'] = selected_df['year'].astype(int).astype(str) + '_' + selected_df['animal_tag']
    animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10',
                       '2018_LB11', '2018_LB12', '2018_LB13',
                       '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
    selected_df = selected_df[selected_df['filename'].isin(animal_tag_list)]

    selected_df['label_id'] = selected_df['label'].map(label_dict)
    selected_df['label_id'] = selected_df['label_id'].fillna(-2)

    # # 删除无标签的数据
    # selected_df = selected_df[selected_df.label_id != -2]
    return selected_df

def pd2np(selected_df, columns):
    # 选择需要的列
    selected_df = selected_df[columns]
    # selected_df = selected_df[['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']]
    fill_selected_df = selected_df.ffill().bfill()  # pressure has nan values
    selected_np = fill_selected_df.values
    return selected_np

def gaussian_std(X):
    mean_val = np.mean(X.astype(float), axis=0)
    std_val = np.std(X.astype(float), axis=0)
    X_standardized = (X - mean_val) / np.maximum(std_val, 10 ** -5)
    return X_standardized, mean_val, std_val

def process_accel(acc_np):
    tmp_acc_stand, mean_val, std_val = gaussian_std(acc_np)
    return tmp_acc_stand, mean_val, std_val

def process_press(pre_np):
    data = pre_np - 1013.25  # standard pressure

    # 计算均值和标准差
    mean_val = np.mean(data)
    std_val = np.std(data)
    # 识别超过3个标准差的异常值
    outlier_mask = np.abs(data - mean_val) > 5 * std_val
    indices = np.where(data > outlier_mask)
    # Convert the result to a list (optional)
    indices_list = indices[0].tolist()
    # 创建数据副本
    data_replaced = np.copy(data)

    # 用周围最近的正常值替换异常值
    for i in indices_list:
        if outlier_mask[i]:
            # 查找前一个正常值
            j = i - 1
            # 查找后一个正常值
            k = i + 1

            # 找到前一个正常值
            while j >= 0 and outlier_mask[j]:
                j -= 1
            # 找到后一个正常值
            while k < len(data) and outlier_mask[k]:
                k += 1

            # 用最近的正常值替换异常值
            if j >= 0 and (k >= len(data) or i - j <= k - i):
                data_replaced[i] = data[j]
            elif k < len(data):
                data_replaced[i] = data[k]
    press_stand, mean_pre, std_pre = gaussian_std(data_replaced)
    return press_stand, mean_pre, std_pre

def process_acc(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    selected_np[:, :3] = acc_np
    return selected_np

def process_acc_press(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    pres_np, mean_prs, std_prs = process_press(selected_np[:, 3:4])
    selected_np[:, :3] = acc_np
    selected_np[:, 3:4] = pres_np
    return selected_np

def process_press_temperature(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_press(selected_np[:, 0:1])
    pres_np, mean_prs, std_prs = gaussian_std(selected_np[:, 1:2])
    selected_np[:, 0:1] = acc_np
    selected_np[:, 1:2] = pres_np
    return selected_np

def process_acc_temperature(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    temp_stand, mean_tem, std_tem = gaussian_std(selected_np[:, 3:4])
    selected_np[:, :3] = acc_np
    selected_np[:, 3:4] = temp_stand

    # plt.figure()
    # plt.plot(temp_stand, 'b')
    # plt.plot(selected_np[:, 3:4], 'r')
    # plt.savefig('acc_temp.png')
    return selected_np

def process_acc_gyr(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    gyr_np, _, _ = process_accel(selected_np[:, 3:6])
    selected_np[:, :3] = acc_np
    selected_np[:, 3:6] = gyr_np
    return selected_np

def process_acc_gps(selected_df, columns):
    selected_df, gps_len = process_gps(selected_df)
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    gps_np, _, _ = process_accel(selected_np[:, 3:5])
    selected_np[:, :3] = acc_np
    selected_np[:, 3:5] = gps_np
    return selected_np

def process_acc_gyr_dpt(selected_df, columns):
    selected_np = pd2np(selected_df, columns)
    acc_np, mean_acc, std_acc = process_accel(selected_np[:, 0:3])
    gyr_np, _, _ = process_accel(selected_np[:, 3:6])
    dpt_np, _, _ = gaussian_std(selected_np[:, 6:7])  # 这里的dpt是深度值，没有大气压
    selected_np[:, :3] = acc_np
    selected_np[:, 3:6] = gyr_np
    selected_np[:, 6:7] = dpt_np
    return selected_np

# 重构上面代码，根据sensor查找对应的process函数

# -----------------------------
# 你已有的工具函数（这里直接使用）
# def gaussian_std(x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: ...
# def pd2np(df: pd.DataFrame, columns: List[str]) -> np.ndarray: ...
# def process_gps(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]: ...
# -----------------------------

# ------------------- 6 种原子传感器列 -------------------
# todo 需要根据不同data自己定义
# SENSOR_COLUMNS: Dict[str, List[str]] = {
#     "Accel": ['acc_x', 'acc_y', 'acc_z'],
#     "Gyro":  ['gyro_x', 'gyro_y', 'gyro_z'],
#     "Mag":  ['mag_x', 'mag_y', 'mag_z'],
#     "Dpt":   ['depth'],
#     "Press": ['pressure'],
#     "Illum": ['illumination'],
#     "Temp":  ['temperature'],
#     "GPS":   ['latitude', 'longitude'],  # 注意：GPS分支根据 compute_gps 走差异化路径
# }  # 两种鸟

SENSOR_COLUMNS: Dict[str, List[str]] = {
    "Accel": ['AccX', 'AccY', 'AccZ'],
    "Gyro":  ['GyrX', 'GyrY', 'GyrZ'],
    # "Mag":  ['mag_x', 'mag_y', 'mag_z'],
    "Dpt":   ['Depth'],
    # "Press": ['pressure'],
    # "Illum": ['illumination'],
    # "Temp":  ['temperature'],
    # "GPS":   ['latitude', 'longitude'],  # 注意：GPS分支根据 compute_gps 走差异化路径
}

def sensor_list_to_tag(sensors: Union[str, List[str]], *, compute_gps: bool = True) -> str:
    """
    将原子传感器列表（或单个字符串）转为文件名友好的短标签。
    例：
      ['Accel','Gyro','Dpt'] → 'acc_gyr_dpt'
      ['GPS'] (compute_gps=True)  → 'gpsbv'
      ['GPS'] (compute_gps=False) → 'gpsll'
    """
    if isinstance(sensors, str):
        sensors = [sensors]

    if not sensors:
        raise ValueError("sensors 不能为空")

    # 别名到规范名
    alias2canon = {
        'acc': 'Accel', 'accel': 'Accel',
        'gyr': 'Gyro', 'gyro': 'Gyro',
        'Mag': 'mag', 'Mag': 'magnitude', 'mag': 'mag', 'magnitude': 'mag',
        'prs': 'Press', 'press': 'Press', 'pressure': 'Press',
        'tmp': 'Temp', 'temp': 'Temp', 'temperature': 'Temp',
        'gps': 'GPS', 'latlon': 'GPS',
        'dpt': 'Dpt', 'depth': 'Dpt',
        'Illum': 'illum', 'illum': 'illumination',
        # 已是规范名时也要支持：
        'accel': 'Accel', 'gyro': 'Gyro', 'Mag': 'mag',
        'press': 'Press', 'temp': 'Temp', 'gps': 'GPS',
        'dpt': 'Dpt', 'illum': 'illumination',
    }

    canon_list = []
    for s in sensors:
        key = str(s).strip()
        if key in SENSOR_COLUMNS:              # 已是规范名
            canon = key
        else:
            k = key.lower()
            if k in alias2canon:
                canon = alias2canon[k]
            else:
                raise ValueError(f"未知的传感器名称: {s}")
        canon_list.append(canon)

    # 去重并按固定优先级排序，稳定文件名
    precedence = ['Accel', 'Gyro', 'Mag', 'Press', 'Temp', 'GPS', 'Dpt', 'Illum']
    uniq = sorted(set(canon_list), key=lambda x: precedence.index(x))

    # 规范名到短标签
    abbr = {
        'Accel': 'acc',
        'Gyro':  'gyr',
        'Mag':  'mag',
        'Press': 'prs',
        'Temp':  'tmp',
        'GPS':   'gpsbv' if compute_gps else 'gpsll',
        'Dpt':   'dpt',
        'Illum':   'illum',
    }

    parts = [abbr[c] for c in uniq]
    return "_".join(parts)

# ------------------- 通用校验 -------------------
def _ensure_columns(df: pd.DataFrame, cols: List[str], ctx: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{ctx}] Missing columns: {missing}")

# ------------------- 三轴/单通道处理 -------------------
def _proc_tri_axis(x3: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x_norm, mean_, std_ = gaussian_std(x3)
    return x_norm, {"mean": mean_, "std": std_}

def _proc_single_chan(x1: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    x_norm, mean_, std_ = gaussian_std(x1)
    return x_norm, {"mean": mean_, "std": std_}

# ------------------- 压力自适应（是否已减 1013.25） -------------------
def _proc_press_adaptive(x1: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    p = x1.reshape(-1).astype(float)
    med = np.nanmedian(p)

    # 1) 可能是 Pa → 转 hPa
    if med > 20000:  # e.g. 101325
        p = p / 100.0
        med = np.nanmedian(p)

    # 2) 若像原始大气压（~1000 hPa 量级）→ 减 1013.25
    if med > 200:
        p = p - 1013.25

    # 3) 异常点置 NaN + 最近邻插值（向量化实现）
    mean_val = np.nanmean(p)
    std_val  = np.nanstd(p)
    if std_val and not np.isnan(std_val):
        mask = np.abs(p - mean_val) > 5 * std_val
        p = p.astype(float)
        p[mask] = np.nan
    # 最近邻插值，两端用前后向填充兜底
    s = pd.Series(p).interpolate(method='nearest', limit_direction='both')
    p_filled = s.to_numpy().reshape(-1, 1)

    # 4) 标准化
    p_norm, mean_, std_ = gaussian_std(p_filled)
    return p_norm, {"mean": mean_, "std": std_}

# ------------------- 小工具：按“最近邻”把列插满 -------------------
def _interpolate_series_nearest(arr: np.ndarray) -> np.ndarray:
    s = pd.Series(arr)
    s = s.interpolate(method='nearest', limit_direction='both')
    return s.to_numpy()

def _interpolate_2cols_nearest(arr2: np.ndarray) -> np.ndarray:
    # arr2: [N,2]
    a = _interpolate_series_nearest(arr2[:, 0])
    b = _interpolate_series_nearest(arr2[:, 1])
    return np.column_stack([a, b])

# ------------------- GPS：两种模式 -------------------
def _gps_bearing_velocity_from_latlon(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    ts_col: str  = 'timestamp'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用有值的 lat/lon 计算 bearing 和 velocity，并把结果插值到全长度。
    返回：bearing_full[N], velocity_full[N]  （均已做最近邻插值填满）
    """
    _ensure_columns(df, [lat_col, lon_col, ts_col], "GPS(derived)")

    # 仅在 lat/lon 非 NaN 的行上计算
    idx_valid = df[lat_col].notna() & df[lon_col].notna()
    if idx_valid.sum() <= 1:
        # GPS 信息不足，直接返回全 NaN，再由上层决定如何处理
        N = len(df)
        nan_vec = np.full(N, np.nan, dtype=float)
        return nan_vec, nan_vec

    lat = np.radians(df.loc[idx_valid, lat_col].to_numpy())
    lon = np.radians(df.loc[idx_valid, lon_col].to_numpy())
    ts  = pd.to_datetime(df.loc[idx_valid, ts_col])

    # 差分
    lat1 = np.roll(lat, 1); lat1[0] = np.nan
    lat2 = lat
    lon_diff = lon - np.roll(lon, 1); lon_diff[0] = np.nan
    lat_diff = lat - np.roll(lat, 1); lat_diff[0] = np.nan

    # 时间差（秒）
    dt = ts.diff().dt.total_seconds().to_numpy()

    # Haversine 距离
    a = (np.sin(lat_diff / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * (np.sin(lon_diff / 2) ** 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371000.0
    distance = R * c

    # 速度（m/s）
    vel = distance / dt
    # 航向角（度）
    x = np.sin(lon_diff) * np.cos(lat2)
    y = (np.cos(lat1) * np.sin(lat2) -
         np.sin(lat1) * np.cos(lat2) * np.cos(lon_diff))
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    # 第一个样本无前项，置 NaN
    vel[0] = np.nan
    bearing[0] = np.nan

    # 回填到原长度（对无 GPS 的行置 NaN，然后整体插值为最近邻）
    N = len(df)
    bearing_full = np.full(N, np.nan, dtype=float)
    velocity_full = np.full(N, np.nan, dtype=float)
    bearing_full[df.index.get_indexer(df.index[idx_valid])] = bearing
    velocity_full[df.index.get_indexer(df.index[idx_valid])] = vel

    # 最近邻插值到全长度
    bearing_full = _interpolate_series_nearest(bearing_full)
    velocity_full = _interpolate_series_nearest(velocity_full)

    return bearing_full, velocity_full


def _median_step_seconds(t_index: pd.DatetimeIndex) -> float:
    """估计全时间轴的步长（秒），用于把“秒阈值”换算成插值的 limit（连续NaN长度）"""
    if len(t_index) < 2:
        return 1.0
    dt = np.diff(t_index.view('i8')) / 1e9  # ns → s
    step = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    return max(step, 1e-6)

def _bridge_small_gaps_time(s: pd.Series, limit_sec: float) -> pd.Series:
    """仅跨越≤limit_sec的小缺口做 time-interp；更长的缺口保持 NaN"""
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    step = _median_step_seconds(s.index)
    limit = max(1, int(round(limit_sec / step)))
    return s.interpolate(method='time', limit=limit)

def _bridge_small_gaps_nearest(s: pd.Series, limit_sec: float) -> pd.Series:
    """最近邻“短桥接”：先 ffill/bfill 限制长度，再保留长缺口为 NaN"""
    if not isinstance(s.index, pd.DatetimeIndex):
        return s
    step = _median_step_seconds(s.index)
    limit = max(1, int(round(limit_sec / step)))
    s1 = s.copy()
    s1 = s1.ffill(limit=limit).bfill(limit=limit)
    return s1

def _gps_bearing_velocity_from_latlon_gapaware(
    df: pd.DataFrame,
    *,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    ts_col: str  = 'timestamp',
    max_bridge_gap_sec: float = 5.0,      # “小缺口”判定阈值
    edge_policy: str = 'stationary',      # 'stationary' 或 'nearest'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1) 仅用有效 lat/lon 计算 bearing & velocity（前向差分）
    2) 以全时间轴为索引：短缺口内插，长缺口按策略填充
    3) 首尾按策略填充；保证无 NaN
    返回: bearing_full[N], velocity_full[N]
    """
    for c in (lat_col, lon_col, ts_col):
        if c not in df.columns:
            raise KeyError(f"[GPS] Missing column: {c}")
    t_full = pd.to_datetime(df[ts_col])
    N = len(df)

    # 有效点
    mask_valid = df[lat_col].notna() & df[lon_col].notna()
    if mask_valid.sum() <= 1:
        # 几乎没有 GPS，可直接全静止
        bearing_full = np.zeros(N, dtype=float)
        velocity_full = np.zeros(N, dtype=float)
        return bearing_full, velocity_full

    lat = np.radians(df.loc[mask_valid, lat_col].to_numpy(float))
    lon = np.radians(df.loc[mask_valid, lon_col].to_numpy(float))
    ts  = pd.to_datetime(df.loc[mask_valid, ts_col])

    # 前向差分（避免第一个样本无“前项”导致大段 NaN）
    lat_prev = np.roll(lat, 1); lat_prev[0] = lat[0]
    lon_prev = np.roll(lon, 1); lon_prev[0] = lon[0]
    dlat = lat - lat_prev
    dlon = lon - lon_prev

    dt = ts.diff().dt.total_seconds().to_numpy()
    if len(dt) > 1:
        dt[0] = max(1e-6, (ts.iloc[1] - ts.iloc[0]).total_seconds())
    else:
        dt[0] = 1.0

    # Haversine 距离 → 速度
    a = (np.sin(dlat/2)**2 + np.cos(lat_prev)*np.cos(lat)*np.sin(dlon/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    R = 6371000.0
    distance = R * c
    vel = distance / dt

    # bearing（使用向量法）
    x = np.sin(dlon) * np.cos(lat)
    y = np.cos(lat_prev) * np.sin(lat) - np.sin(lat_prev) * np.cos(lat) * np.cos(dlon)
    bearing = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0

    # 以完整时间轴为索引构造 Series，并对齐
    s_vel = pd.Series(vel, index=ts).reindex(t_full)
    s_brg = pd.Series(bearing, index=ts).reindex(t_full)

    # 只“桥接”小缺口：velocity 用 time-interp；bearing 用最近邻（避免跨 0° 线性插值失真）
    s_vel = _bridge_small_gaps_time(s_vel, max_bridge_gap_sec)
    s_brg = _bridge_small_gaps_nearest(s_brg, max_bridge_gap_sec)

    # 首尾与长缺口填充策略
    if edge_policy == 'stationary':
        # velocity：剩余 NaN（首尾或长缺口）全部置 0
        s_vel = s_vel.fillna(0.0)
        # bearing：用最近邻补全（首尾/长缺口），若仍有 NaN（全程无值），置 0
        s_brg = s_brg.ffill().bfill().fillna(0.0)
    elif edge_policy == 'nearest':
        # 两者都用最近邻补全
        s_vel = s_vel.ffill().bfill()
        s_brg = s_brg.ffill().bfill()
        # 极端情况兜底
        s_vel = s_vel.fillna(0.0)
        s_brg = s_brg.fillna(0.0)
    else:
        raise ValueError("edge_policy must be 'stationary' or 'nearest'")

    return s_brg.to_numpy(dtype=float), s_vel.to_numpy(dtype=float)

def _gps_latlon_interpolated_gapaware(
    df: pd.DataFrame,
    *,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    ts_col: str  = 'timestamp',
    max_bridge_gap_sec: float = 5.0,
) -> np.ndarray:
    """
    时间驱动的线性插值只跨越短缺口；长缺口再 ffill/bfill。
    最终不留 NaN，返回 [N,2] 的 lat/lon。
    """
    for c in (lat_col, lon_col):
        if c not in df.columns:
            raise KeyError(f"[GPS] Missing column: {c}")
    t_full = pd.to_datetime(df[ts_col]) if ts_col in df.columns else pd.RangeIndex(len(df))
    lat = pd.Series(df[lat_col].astype(float).values, index=t_full)
    lon = pd.Series(df[lon_col].astype(float).values, index=t_full)

    if isinstance(t_full, pd.DatetimeIndex):
        lat = _bridge_small_gaps_time(lat, max_bridge_gap_sec)
        lon = _bridge_small_gaps_time(lon, max_bridge_gap_sec)
    else:
        # 没有 timestamp 就用最近邻的小桥接
        lat = _bridge_small_gaps_nearest(lat, max_bridge_gap_sec)
        lon = _bridge_small_gaps_nearest(lon, max_bridge_gap_sec)

    # 首尾/长缺口兜底
    lat = lat.ffill().bfill().fillna(0.0)
    lon = lon.ffill().bfill().fillna(0.0)

    return np.column_stack([lat.to_numpy(float), lon.to_numpy(float)])

def _gps_latlon_interpolated(
    df: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude'
) -> np.ndarray:
    """
    直接对 [latitude, longitude] 做最近邻插值（到全长度），返回 [N,2]
    """
    _ensure_columns(df, [lat_col, lon_col], "GPS(raw)")
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)
    lat_i = _interpolate_series_nearest(lat)
    lon_i = _interpolate_series_nearest(lon)
    return np.column_stack([lat_i, lon_i])

# ------------------- 主入口 -------------------
def process_sensors(
    selected_df: pd.DataFrame,
    sensors: List[str],
    *,
    append_label: bool = True,
    label_col: str = "label_id",
    compute_gps: bool = True,   # 默认：先转 bearing/velocity，再插值，再返回二者
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    处理任意顺序的 ['Accel','Gyro','Dpt','Press','Temp','GPS'] 组合，按给定顺序拼接输出。
    - Accel/Gyro：三轴标准化（高频；向量化）
    - Dpt/Temp：单通道标准化
    - Press：自适应判断是否已减 1013.25，异常点最近邻插值 + 标准化
    - GPS：
        * compute_gps=True：由 lat/lon/timestamp 算 bearing/velocity → 最近邻插值 → 标准化 → 返回这两列
        * compute_gps=False：对 lat/lon 最近邻插值 → 标准化 → 返回这两列
    返回：
      combined_np: [N, sum(dim per sensor) (+1 if label)]
      stats:       每个原子块的统计参数（均值/标准差）
    """
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("`sensors` 必须是非空列表，如 ['Accel','Gyro','GPS'].")

    allowed = set(SENSOR_COLUMNS.keys())
    bad = [s for s in sensors if s not in allowed]
    if bad:
        raise ValueError(f"不支持的传感器: {bad}. 仅支持 {sorted(allowed)}")

    df = selected_df
    blocks: List[np.ndarray] = []
    stats: Dict[str, Dict[str, np.ndarray]] = {}

    for sensor in sensors:
        if sensor in ("Accel", "Gyro", "Mag"):
            cols = SENSOR_COLUMNS[sensor]
            _ensure_columns(df, cols, sensor)
            X = pd2np(df, cols)
            if X.ndim == 1: X = X.reshape(-1, 1)
            Xp, S = _proc_tri_axis(X)

        elif sensor in ["Dpt", "Temp", "Illum"]:
            cols = SENSOR_COLUMNS[sensor]
            _ensure_columns(df, cols, sensor)
            X = pd2np(df, cols)
            if X.ndim == 1: X = X.reshape(-1, 1)
            Xp, S = _proc_single_chan(X)

        # elif sensor == "Temp":
        #     cols = SENSOR_COLUMNS[sensor]
        #     _ensure_columns(df, cols, sensor)
        #     X = pd2np(df, cols)
        #     if X.ndim == 1: X = X.reshape(-1, 1)
        #     Xp, S = _proc_single_chan(X)

        elif sensor == "Press":
            cols = SENSOR_COLUMNS[sensor]
            _ensure_columns(df, cols, sensor)
            X = pd2np(df, cols)
            if X.ndim == 1: X = X.reshape(-1, 1)
            Xp, S = _proc_press_adaptive(X)

        elif sensor == "GPS":
            if compute_gps:
                bearing_full, velocity_full = _gps_bearing_velocity_from_latlon_gapaware(
                    selected_df,
                    lat_col='latitude', lon_col='longitude', ts_col='timestamp',
                    max_bridge_gap_sec=30,
                    edge_policy='stationary',
                )
                X_raw = np.column_stack([bearing_full, velocity_full])
                Xp, mean_, std_ = gaussian_std(X_raw)
                S = {"mean": mean_, "std": std_}
            else:
                # 直接对 lat/lon 插值，然后标准化
                X_raw = _gps_latlon_interpolated(df)  # [N,2]
                Xp, mean_, std_ = gaussian_std(X_raw)
                S = {"mean": mean_, "std": std_}
        else:
            raise ValueError(f"未实现的处理器: {sensor}")

        if Xp.shape[0] != len(df):
            raise RuntimeError(f"[{sensor}] 长度不匹配: {Xp.shape[0]} vs {len(df)}")

        blocks.append(Xp)
        stats[sensor] = S

    combined_np = np.hstack(blocks) if len(blocks) > 1 else blocks[0]

    if append_label and (label_col in df.columns):
        y = pd2np(df, [label_col]).reshape(-1, 1)
        combined_np = np.hstack([combined_np, y])

    return combined_np, stats


###########################raw data processing###################################


def majority_value(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    majority = []
    for row in arr:
        values, counts = np.unique(row, return_counts=True)
        majority.append(values[np.argmax(counts)])
    return np.array(majority).astype(int)


def find_majority_minority(label_b):
    unique_labels, counts = np.unique(label_b, return_counts=True)

    if len(unique_labels) == 1:
        print(f"Only one label present: {unique_labels[0]}")
        return unique_labels[0], unique_labels[0]  # 只有一个类别，返回相同的值

    # 计算多数类和少数类
    majority_label = unique_labels[np.argmax(counts)]
    minority_label = unique_labels[np.argmin(counts)]

    return majority_label, minority_label

def AE_eval_time_series_old(train_loader, model, device, memotimes=30):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, _, label2d) in enumerate(train_loader):

        # if i > memotimes:  # cpu memory not enough
        #     continue
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # print(type(output))
        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label2d.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list


@torch.inference_mode()  # PyTorch>=1.9，等价于 no_grad + 更少的状态开销
def AE_eval_time_series(train_loader, model, device, memotimes=None,
                        return_major_label=False,
                        pool='mean',  # None/'mean'/'max'
                        concat=False):
    """
    评估/特征导出：前向计算并收集 embedding、输出、标签、输入。
    - train_loader: 迭代返回 (sample, major_label, label2d) 的 DataLoader
    - pool: 若 x_encoded 为 [B, C, T]，可在此做时域聚合得到 [B, C]
    - concat: True 则在函数内拼接为单个 np.ndarray；False 返回“按批列表”
    - return_major_label: 同时返回主标签以便下游直接使用
    """
    model.eval()

    representation_list, sample_list, out_list, label2d_list = [], [], [], []
    major_label_list = []  # 可选

    for i, batch in enumerate(train_loader):
        # 兼容两种返回形态
        if len(batch) == 3:
            sample, major_label, label2d = batch
        else:
            sample, label2d = batch
            major_label = None

        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        x_encoded, output = model(sample)  # [B,C,T?], [B,num_classes] 或重构

        # 可选的时域聚合，统一成 [B, D]
        if pool is not None and x_encoded.ndim == 3:
            if pool == 'mean':
                x_encoded = x_encoded.mean(dim=-1)
            elif pool == 'max':
                x_encoded = x_encoded.max(dim=-1).values
            else:
                raise ValueError(f"Unknown pool='{pool}'")

        # 收集到 CPU / numpy
        representation_list.append(x_encoded.cpu().numpy())
        out_list.append(output.cpu().numpy())
        label2d_list.append(label2d.cpu().numpy())
        sample_list.append(sample.cpu().numpy())
        if return_major_label and major_label is not None:
            major_label_list.append(major_label.cpu().numpy())

        if memotimes is not None and i >= memotimes:
            break

    # 是否在函数内拼接
    if concat:
        import numpy as np
        representation_list = np.concatenate(representation_list, axis=0)
        out_list = np.concatenate(out_list, axis=0)
        label2d_list = np.concatenate(label2d_list, axis=0)
        sample_list = np.concatenate(sample_list, axis=0)
        if return_major_label and len(major_label_list) > 0:
            major_label_list = np.concatenate(major_label_list, axis=0)
        return (representation_list, sample_list, out_list, label2d_list, major_label_list) if return_major_label \
               else (representation_list, sample_list, out_list, label2d_list)

    # 默认：按批列表
    return (representation_list, sample_list, out_list, label2d_list, major_label_list) if return_major_label \
           else (representation_list, sample_list, out_list, label2d_list)


def Classify_eval_time_series(train_loader, model, device):
    model.eval()

    representation_list = []
    sample_list, timestamp_list, label_list, pred_list, timestr_list, flag_list = [], [], [], [], [], []
    for i, (sample, label) in enumerate(train_loader):
        sample = sample.to(device=device, non_blocking=True, dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # x_encoded = model.feature_extractor(sample)
        # print(type(output))
        # x_encoded, output = model(input_).view(b, 2, -1)  # output.shape=b,2,128, split the first dim into 2 parts
        tmp_representation = x_encoded.detach().cpu().numpy()
        representation_list.append(tmp_representation)
        sample_list.append(sample.detach().cpu().numpy())
        label_list.append(label.detach().cpu().numpy())
        pred_list.append(output.detach().cpu().numpy())

    return representation_list, sample_list, pred_list, label_list

def plot_func(sensordim, sample_reshape, pred_reshape, sensortype, axis_dict, name, start, end):
    fig, axes = plt.subplots(sensordim * 3, 1, figsize=(8, 6))
    for acc_axis in range(sensordim):
        col = acc_axis
        axes[0 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
        axes[0 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
        axes[0 + sensordim * col].set_title('Autoencoder_Reconstruct_Umineko_%s_axis%s' % (sensortype, axis_dict[col]))
        axes[0 + sensordim * col].set_xlabel('timestamp')
        axes[0 + sensordim * col].set_ylabel('value')
        axes[0 + sensordim * col].legend(loc="right")

        axes[1 + sensordim * col].plot(sample_reshape[start:end, col], 'r', label='groundtruth%s' % axis_dict[col])
        axes[1 + sensordim * col].set_xlabel('timestamp')
        axes[1 + sensordim * col].set_ylabel('value')
        axes[1 + sensordim * col].legend(loc="right")

        axes[2 + sensordim * col].plot(pred_reshape[start:end, col], 'b-', label='predict%s' % axis_dict[col])
        axes[2 + sensordim * col].set_xlabel('timestamp')
        axes[2 + sensordim * col].set_ylabel('value')
        axes[2 + sensordim * col].legend(loc="right")

    # Adjust layout
    plt.tight_layout()
    plt.title(name)
    # Show the figure
    plt.savefig(name + '_%s.png' % sensortype)
    plt.close('all')
    return

def plot_reconstruction_result(sensortype, representation_list, sample_list, pred_list, label_list, name='accel'):
    # tsne latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1).astype(float)

    sample_concat = np.concatenate(sample_list)
    # sample_reshape = sample_concat.reshape(-1, 3)
    sample_concat = sample_concat.transpose(0, 2, 1)
    sample_reshape = sample_concat.reshape(-1, sample_concat.shape[-1])

    pred_concat = np.concatenate(pred_list)
    # pred_reshape = pred_concat.reshape(-1, 3)
    pred_concat = pred_concat.transpose(0, 2, 1)
    pred_reshape = pred_concat.reshape(-1, pred_concat.shape[-1])

    # sensortype = 'pressure'
    if sensortype == 'accel':
        axis_dict = {0:'x', 1:'y', 2:'z'}
        sensordim = len(axis_dict)
    elif sensortype == 'pressure':
        axis_dict = {0:'pressure'}
        sensordim = len(axis_dict)
    else:
        print('no such sensor type')

    plot_func(sensordim, sample_reshape, pred_reshape, sensortype, axis_dict, 'all_'+name, start=0, end=-1)
    plot_func(sensordim, sample_reshape, pred_reshape, sensortype, axis_dict, name, start=0, end=10000)

    # umap_3d = UMAP(n_components=3)
    #
    # proj_3d_gyro = umap_3d.fit_transform(repre_reshape[start:end])
    #
    # # set point size
    # # point_size = np.ones(proj_3d_gyro.shape[0]) * 1
    # label_concat = np.concatenate(label_list)
    # label_concat_vote = majority_value(label_concat)
    # # grey_idx = np.where(label_concat_vote == -2)[0]
    # # point_size[grey_idx] = 0.5
    # # create a dict from actID to act:
    # labeldict_findstr = {-2: 'unknown',
    #                      0: 'ground_stationary',
    #                      1: 'stationary',
    #                      2: 'bathing',
    #                      3: 'flying_active',
    #                      4: 'flying_passive',
    #                      5: 'foraging'}
    #
    # label_concat_vote_str = [labeldict_findstr[i] for i in label_concat_vote]
    # # label_concat_vote_str = np.char.mod('%d', label_concat_vote)
    #
    # fig_3d = px.scatter_3d(
    #     proj_3d_gyro, x=0, y=1, z=2,
    #     color=label_concat_vote_str[start:end],
    #     labels={'color': 'activity'},
    #     # color_discrete_map={ '-2.0': ('rgba(239, 239, 240, 1)')},
    #     color_discrete_map={'unknown': 'lightgrey'},
    #     # size=point_size
    # )
    # # Reduce marker size for all points
    # for trace in fig_3d.data:
    #     trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
    #
    # # Update transparency for traces where activity is '-2.0'
    # fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
    # fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
    # fig_3d.write_html(name+'_%s.html'%sensortype)
    # print('')
    # fig_3d.write_html("test.html")
# ################################################################################
#     umap_3d = UMAP(n_components=2)
#     proj_3d_gyro = umap_3d.fit_transform(repre_reshape)
#     # set point size
#     df = pd.DataFrame(proj_3d_gyro, columns=['UMAP Dim.1', 'UMAP Dim.2'])
#     fig_3d = px.scatter(
#         df, x='UMAP Dim.1', y='UMAP Dim.2',
#         color=label_concat_vote_str,
#         labels={'color': 'activity'},
#         color_discrete_map={'unknown': 'lightgrey'},
#     )
#     # Reduce marker size for all points
#     for trace in fig_3d.data:
#         trace.marker.size = 6  # Adjust the size to your preference (e.g., 6)
#     # Update transparency for traces where activity is '-2.0'
#     fig_3d.for_each_trace(lambda trace: trace.update(marker=dict(opacity=0.5)) if trace.name == '-2.0' else ())
#     fig_3d.update_traces(marker=dict(line=dict(width=0)))  # remove boundary of point
#     fig_3d.write_image(name+'_actlabel.pdf')
# #

    # from sklearn.cluster import AgglomerativeClustering
    # # Apply Agglomerative Clustering
    # agglo = AgglomerativeClustering(n_clusters=5)  # Adjust n_clusters
    # cluster_labels = agglo.fit_predict(repre_reshape)
    #
    # # Replace activity labels with the new HDBSCAN cluster labels
    # # Convert to a DataFrame for visualization
    # df = pd.DataFrame(proj_3d_gyro, columns=['UMAP Dim.1', 'UMAP Dim.2'])
    # df['cluster'] = cluster_labels
    #
    # # Visualize the new clusters
    # fig = px.scatter(
    #     df, x='UMAP Dim.1', y='UMAP Dim.2', color='cluster',
    #     title="Umineko2018, back1, Clusters Generated by Agglomerative Clustering",
    #     color_discrete_sequence=px.colors.qualitative.Set1
    # )
    # fig.write_image(name+'_Agglomerative.pdf')
    #
    #
    # from sklearn.mixture import GaussianMixture
    # # Apply Gaussian Mixture Model
    # gmm = GaussianMixture(n_components=5, random_state=42)  # Adjust n_components
    # cluster_labels = gmm.fit_predict(repre_reshape)
    # # Replace activity labels with the new GMM cluster labels
    # df = pd.DataFrame(proj_3d_gyro, columns=['UMAP Dim.1', 'UMAP Dim.2'])
    # df['cluster'] = cluster_labels
    # # Visualize the clusters
    # fig = px.scatter(
    #     df, x='UMAP Dim.1', y='UMAP Dim.2', color='cluster',
    #     title="Umineko2018, back1, Clusters Generated by Gaussian Mixture Model",
    #     color_discrete_sequence=px.colors.qualitative.Set1
    # )
    # fig.write_image(name+'_GMM.pdf')
    #
    # from sklearn.cluster import KMeans
    # # Apply K-Means clustering
    # kmeans = KMeans(n_clusters=5, random_state=42)  # Adjust n_clusters as needed
    # cluster_labels = kmeans.fit_predict(repre_reshape)
    #
    # # Replace activity labels with the new K-Means cluster labels
    # df = pd.DataFrame(proj_3d_gyro, columns=['UMAP Dim.1', 'UMAP Dim.2'])
    # df['cluster'] = cluster_labels
    # # Visualize the clusters
    # fig = px.scatter(
    #     df, x='UMAP Dim.1', y='UMAP Dim.2', color='cluster',
    #     title="Umineko2018, back1, Clusters Generated by K-Means",
    #     color_discrete_sequence=px.colors.qualitative.Set1
    # )
    # fig.write_image(name+'_Kmeans.pdf')
    #
    # from sklearn.cluster import DBSCAN
    # dbscan = DBSCAN(eps=0.25, min_samples=10)  # Adjust 'eps' and 'min_samples' as needed
    # cluster_labels = dbscan.fit_predict(proj_3d_gyro)
    # df = pd.DataFrame(proj_3d_gyro, columns=['UMAP Dim.1', 'UMAP Dim.2'])
    # df['cluster'] = cluster_labels
    # # Visualize the new clusters
    # fig = px.scatter(
    #     df, x='UMAP Dim.1', y='UMAP Dim.2', color='cluster',
    #     title="Umineko2018, back1, Clusters Generated by DBSCAN",
    #     color_discrete_sequence=px.colors.qualitative.Set1
    # )
    # fig.write_image(name+'_DBSCAN.pdf')
    return

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

def format_timestamp(df):
    # if 'datetime' not in df.columns:
    s = df['timestamp'].str.replace('T', ' ').str.replace('Z', '')
    # df = df.drop('timestamp', axis=1)
    s_datetime = pd.to_datetime(s)  # to datetime64[ns]
    df.insert(loc=0, column='datetime', value=s_datetime)
    # round at 1 millisecond
    df['datetime'] = df['datetime'].dt.round('1L')
    # unixtime
    unixtime = df['datetime'].apply(lambda t: t.timestamp())
    df.insert(loc=1, column='unixtime', value=unixtime)
    return df

def check_if_has_str(listdata):
    # check if a list contains string values
    for l in listdata:
        if type(l) == str:
            return True
    return False

def resampling(df, intermediate_sampling_rate=100, output_sampling_rate=25):
    if np.sum(df['unixtime'].duplicated()) > 1:
        # print(len(df[df['unixtime'].duplicated()]))
        df.drop_duplicates(subset='datetime', keep=False, inplace=True)
        # print(len(df[df['unixtime'].duplicated()]))
        print("duplicated index detected -> duplicates removed")
    else:
        print("No duplicates")

    # Generate original time indices
    original_time = np.arange(len(df)) / intermediate_sampling_rate
    # Generate new time indices
    new_length = int(len(df) * output_sampling_rate / intermediate_sampling_rate)
    new_time = np.arange(new_length) / output_sampling_rate
    # Create a new DataFrame to store the resampled values
    resampled_df = pd.DataFrame(index=new_time)
    for column in df.columns:
        # Step 1: Extract unique characters from the column
        unique_chars = df[column].unique()

        if check_if_has_str(unique_chars):

            # Step 2: Create a dictionary that maps each character to a unique integer
            char_to_int = {char: i for i, char in enumerate(unique_chars, start=1)}
            # Step 3: Use the dictionary to replace the characters with their corresponding integers
            df[column] = df[column].map(char_to_int)

            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

            # Step 4: Create an inverse mapping dictionary
            int_to_char = {v: k for k, v in char_to_int.items()}
            # Step 5: Use the inverse mapping dictionary to convert the integers back to the original characters
            resampled_df[column] = resampled_df[column].map(int_to_char)
        else:
            # Create interpolation function for each column
            interp_func = interp1d(original_time, df[column].values, kind='linear', fill_value='extrapolate')
            # Generate new values at the new sampling rate
            resampled_df[column] = interp_func(new_time)

    resampled_df.drop(columns=['index'], inplace=True)
    return resampled_df

def sliding_window(data, len_sw, step=300):
    # input is a segment of data
    # output is data, timestamp, domain(filename), label
    # sampling rate = 25Hz
    # window size = 900, step = window size/2
    # for umineko data, 目前只处理有标签的data，用slidewin取segment时，保证最后一块segment一定取到。
    # 同时，以每个单独的标签（起止时间）为单位，不要把其他时间的不同label的segment混在一起。

    # batch_size = 512
    # len_sw = 90
    # step = int(len_sw / 2)

    if isinstance(data, pd.DataFrame):
        data1 = data.copy()
        datanp = data1.values
    else:
        datanp = data.copy()

    # generate batch of data by overlapping the training set
    data_batch = []
    for idx in range(0, datanp.shape[0] - len_sw - step, step):  # step10
        data_batch.append(datanp[idx: idx + len_sw, :])
    data_batch.append(datanp[-1 - len_sw: -1, :])  # last batch
    xlist = np.stack(data_batch, axis=0)  # [B, Len90, dim6]
    # [samples, timestamps, labels] = xlist
    # x_win_train = xlist.reshape((batch_size, xlist.shape[1], xlist.shape[-1]))  # [B, Len, dim]
    # print(" ..after sliding window: train inputs {0}".format(xlist.shape))
    return xlist

def process_pressure_sensor(df, columns):
    if 'pressure' in columns:
        df['pressure'] = df['pressure'] - 1013.25
    return df

def process_gps(df):
    # identify if gps exists
    # if exists, calculate velocity and angle
    gps_len = len(df)
    df_columns = df.columns
    if ('latitude' in df_columns) or\
        ('longitude' in df_columns):
        # Extract rows where both latitude and longitude are not NaN
        df_non_nan = df.dropna(subset=['latitude', 'longitude'])

        # get sampling rate of GPS signal,newlen*oldHz/oldlen得到GPShz，所以这里传newlen
        gps_len = len(df_non_nan)

        # Calculate differences, handling NaN by filling with zeros
        df_non_nan['lat_diff'] = np.radians(df_non_nan['latitude'].diff())
        df_non_nan['lon_diff'] = np.radians(df_non_nan['longitude'].diff())

        # Convert latitude to radians, handling NaN by filling with zeros
        df_non_nan['lat1'] = np.radians(df_non_nan['latitude'].shift())
        df_non_nan['lat2'] = np.radians(df_non_nan['latitude'])

        # Calculate time difference in seconds
        df_non_nan['timestamp'] = pd.to_datetime(df_non_nan['timestamp'])
        df_non_nan['time_diff'] = df_non_nan['timestamp'].diff().dt.total_seconds()

        # Haversine formula
        a = (np.sin(df_non_nan['lat_diff'] / 2) ** 2 +
             np.cos(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.sin(df_non_nan['lon_diff'] / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        R = 6371000  # Earth radius in meters
        df_non_nan['distance'] = R * c
        # Calculate velocity (m/s)
        df_non_nan['GPS_velocity'] = df_non_nan['distance'] / df_non_nan['time_diff']

        # Calculate bearing
        x = np.sin(df_non_nan['lon_diff']) * np.cos(df_non_nan['lat2'])
        y = (np.cos(df_non_nan['lat1']) * np.sin(df_non_nan['lat2']) -
             np.sin(df_non_nan['lat1']) * np.cos(df_non_nan['lat2']) * np.cos(df_non_nan['lon_diff']))
        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        df_non_nan['GPS_bearing'] = (initial_bearing + 360) % 360

        # Merge velocity and bearing back to the original dataframe
        df = df.merge(df_non_nan[['GPS_velocity', 'GPS_bearing']], left_index=True, right_index=True, how='left')

    return df, gps_len




class data_loader_umineko(Dataset):
    def __init__(self, samples, labels1d, labels2d, device='cpu'):
        self.samples = torch.tensor(samples).to(device)  # check data type
        self.labels1d = torch.LongTensor(labels1d).to(device)  # check data type

        self.labels2d = torch.tensor(labels2d)  # check data type

    def __getitem__(self, index):
        target2d = self.labels2d[index]
        target1d = self.labels1d[index]
        sample = self.samples[index]
        return sample, target1d, target2d

    def __len__(self):
        return len(self.labels1d)

def adjust_learning_rate(lr, optimizer, epoch, p_scheduler, p_epochs):
    # lr = 0.0001

    if p_scheduler == 'cosine':
        lr_decay_rate = 0.1
        eta_min = lr * (lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p_epochs)) / 2

    # elif p['scheduler'] == 'step':
    #     steps = np.sum(epoch > np.array(p['scheduler_kwargs']['lr_decay_epochs']))
    #     if steps > 0:
    #         lr = lr * (p['scheduler_kwargs']['lr_decay_rate'] ** steps)

    elif p_scheduler == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p_scheduler))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class MSEloss(nn.Module):
    def __init__(self):
        super(MSEloss, self).__init__()

    def forward(self, input, target):
        '''
        input: raw sensor data
        target: reconstructed sensor data
        the mse loss makes the target data to be similar to the input data
        '''
        loss = nn.MSELoss()
        output = loss(input, target)
        return output


def get_optimizer(p_opti, model):
    params = model.parameters()
    if p_opti == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    nesterov=False,
                                    weight_decay=0.0001,
                                    momentum=0.9,
                                    lr=0.0001)

    elif p_opti == 'adam':
        optimizer = torch.optim.Adam(params,
                                     weight_decay=0.0001,
                                     lr=0.01)
    else:
        raise ValueError('Invalid optimizer {}'.format(p_opti))

    return optimizer


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def AE_train_time_series(train_loader, model, criterion, optimizer):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(dtype=torch.float)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(sample, output)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses

def AE_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device=None):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    if device is None:
        from deepview.utils.device import get_device
        device = get_device()  # cuda > mps (Apple Silicon) > cpu
    # losses = AverageMeter('Loss', ':.4e')
    losses = []

    model.train()
    warm_up_step = 1
    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(device=device, dtype=torch.float)

        # # input of autoencoder will be 3D, the backbone is 1d-cnn
        # noisy_data = sample + torch.randn_like(sample) * 0.1

        x_encoded, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        loss = criterion(output, sample)
        # loss = criterion(sample, output)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (epoch >= warm_up_step):
        # scheduler.step()
        scheduler.step(np.average(losses))
        # print(
        #     f'Epoch {epoch + 1}, Learning Rate: {scheduler.optimizer.param_groups[0]["lr"]}')
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses

def Classify_train_time_series_resnet(train_loader, model, criterion, optimizer, epoch, scheduler, device=None):
    """
    Train according to the scheme from SimCLR
    https://arxiv.org/abs/2002.05709
    """
    if device is None:
        from deepview.utils.device import get_device
        device = get_device()  # cuda > mps (Apple Silicon) > cpu
    losses = AverageMeter('Loss', ':.4e')

    model.train()
    warm_up_step = 5
    outputs, labels = [], []
    for i, (sample, label) in enumerate(train_loader):
        # aug_sample1 = gen_aug(sample, 't_warp')  # t_warp, out.shape=batch64,width3,height900
        # reshape data by adding channel to 1, and transpose height and width
        # sample = sample.to(device=device, non_blocking=True, dtype=torch.float)
        sample = sample.to(device=device, dtype=torch.float)
        # label = label.to(device=device, dtype=torch.long)

        # input of autoencoder will be 3D, the backbone is 1d-cnn
        # sample = sample.transpose(0, 2, 1)
        feature, output = model(sample)  # x_encoded.shape=batch512,outchannel128,len13
        # print(output.shape)
        # print(type(label[:,0]))
        label_concat_vote = majority_value(label)
        label_concat_vote = torch.asarray(label_concat_vote).to(device=device, dtype=torch.long)
        loss = criterion(output, label_concat_vote)
        losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        outputs.append(output.detach().cpu().numpy())
        labels.append(label[:, 0].detach().cpu().numpy())

    if (epoch >= warm_up_step):
        scheduler.step()
    #     if i % 10 == 0:
    #         print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    # print('loss of the ' + str(epoch) + '-th training epoch is :' + losses.__str__())
    return losses,outputs,labels


def load_weights(
    weight_path, model, my_device="cpu", name_start_idx=2, is_dist=False
):
    # only need to change weights name when the
    # model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    if is_dist:
        for key in pretrained_dict:
            para_names = key.split(".")
            new_key = ".".join(para_names[name_start_idx:])
            pretrained_dict_v2[new_key] = pretrained_dict_v2.pop(key)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)
    print("%d Weights loaded" % len(pretrained_dict))

def freeze_feature_extractor(model):
    # Check if parameters are frozen
    for name, param in model.feature_extractor.named_parameters():
        print(f"{name} requires_grad: {param.requires_grad}")
    return model


def Permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[1])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[-1]))
    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            np.random.shuffle(splits)
            warp = np.concatenate(splits).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)

def get_cubic_spline_interpolation(x_eval, x_data, y_data):
    """
    Get values for the cubic spline interpolation
    """
    cubic_spline = CubicSpline(x_data, y_data)
    return cubic_spline(x_eval)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(
        X, sigma
    )  # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)  # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [
        (X.shape[0] - 1) / tt_cum[-1, 0],
        (X.shape[0] - 1) / tt_cum[-1, 1],
        (X.shape[0] - 1) / tt_cum[-1, 2],
    ]
    tt_cum[:, 0] = tt_cum[:, 0] * t_scale[0]
    tt_cum[:, 1] = tt_cum[:, 1] * t_scale[1]
    tt_cum[:, 2] = tt_cum[:, 2] * t_scale[2]
    return tt_cum


def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (
        np.ones((X.shape[1], 1))
        * (np.arange(0, X.shape[0], (X.shape[0] - 1) / (knot + 1)))
    ).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:, 0], yy[:, 0])
    cs_y = CubicSpline(xx[:, 1], yy[:, 1])
    cs_z = CubicSpline(xx[:, 2], yy[:, 2])
    return np.array([cs_x(x_range), cs_y(x_range), cs_z(x_range)]).transpose()


def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:, 0] = np.interp(x_range, tt_new[:, 0], X[:, 0])
    X_new[:, 1] = np.interp(x_range, tt_new[:, 1], X[:, 1])
    X_new[:, 2] = np.interp(x_range, tt_new[:, 2], X[:, 2])
    return X_new


def time_warp(sample, sigma=0.2):
    sample = np.swapaxes(sample, 0, 1)
    sample = DA_TimeWarp(sample, sigma=sigma)
    sample = np.swapaxes(sample, 0, 1)
    return torch.from_numpy(sample)


