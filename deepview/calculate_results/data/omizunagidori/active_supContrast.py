'''
使用2018，21年的数据，使用完全没有初始化的encoder，随机添加label，计算监督学习结果
两个结果：
latent space结果
active learning的F1 score
'''
import copy
import os
import random
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch
import pickle
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from deepview.calculate_results.models.utils import (
    find_majority_minority,
    data_loader_umineko,
    AE_eval_time_series,
    majority_value,
    sliding_window,
    # read_sensor_data,
    process_acc,
    process_acc_press,
    process_acc_temperature,
    process_acc_gyr,
    process_press_temperature,
process_acc_gps
)
from deepview.calculate_results.data.umineko.train_func import (
    train_model,
    evaluate_model,
    evaluate_supContrast_model,
    plot_confusion_matrix,
    unfreeze_encoders,
    freeze_encoders,
    unfreeze_all,
)
from deepview.calculate_results.data.umineko.model_func import (
    SimpleNN_1s,
    SimpleNN_13s,
    SimpleNN_11s,
    SimpleNN_33s,
    SimpleNN_32s,
    ContrastiveLoss,
    SupContrastiveLoss,
)
from deepview.calculate_results.data.umineko.active_utils import (
    data_sampling
)
from deepview.calculate_results.data.umineko.cluster_method import (
    vis_scatter_label_2d,
    # plot_func
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

label_dict = {
'stationary': 0,
'preening': 0,
'bathing': 1,
'flight_take_off': 2,
'flight_cruising': 2, #3,
'foraging_dive': 3, #4,
'surface_seizing': 3, #4,
'body_shaking': -2, #-2,
'unknown': -2
}

def read_sensor_data():
    # raw_data, labeled_data = [], []
    for year in ['2018', '2020', '2021', '2022']:
        dp = r'D:\code\DeepView\deepview\calculate_results\data\omizunagidori_%s.npy'
        data = np.load(dp % year, allow_pickle=True).item()
        # Access the individual components
        raw_ = data['raw_data']
        labeled_ = data['labeled_data']
        if year == '2018':
            df_raw_2018 = raw_
            df_2018 = labeled_
        elif year=='2021':
            df_raw_2021 = raw_
            df_2021 = labeled_
        elif year == '2020':
            df_raw_2020 = raw_
            df_2020 = labeled_
        elif year == '2022':
            df_raw_2022 = raw_
            df_2022 = labeled_
        else:
            print('Error: year not found')
            break

    selected_df = pd.concat([df_raw_2018, df_raw_2020, df_raw_2021, df_raw_2022], ignore_index=True)
    selected_df['filename'] = selected_df['year'].astype(int).astype(str) + '_' + selected_df['animal_tag']
    # animal_tag_list = ['2018_LB07', '2018_LB08', '2018_LB09', '2018_LB10',
    #                    '2018_LB11', '2018_LB12', '2018_LB13',
    #                    '2022_LB02', '2022_LB03', '2022_LB08', '2022_LB09']
    # selected_df = selected_df[selected_df['filename'].isin(animal_tag_list)]

    selected_df['label_id'] = selected_df['label'].map(label_dict)
    selected_df['label_id'] = selected_df['label_id'].fillna(-2)

    # # 删除无标签的数据
    # selected_df = selected_df[selected_df.label_id != -2]
    return selected_df

# all_df = read_sensor_data()

def run(seed_value, sensor_type='AccelTemp', sampling_method = 'entropy'):
    # Set a fixed random seed
    # seed_value = 2025
    # seed_value = 2
    # seed_value = 5
    # seed_value = 1
    # seed_value = 10
    # seed_value = 100
    # seed_value = 11
    set_random_seed(seed_value)

    # sensor_type = 'Accel'
    # sensor_type = 'AccelPress'
    # sensor_type = 'AccelTemp'
    # sensor_type = 'AccelGyro'
    # sensor_type = 'PressTemp'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # sampling_method = 'entropy'
    # sampling_method = 'overAugment'
    # sampling_method = 'underRandom'
    # sampling_method = 'random'
    SupCount = 1  # supcontrastive learning
    # warmup = 0  # warmup epoch of supcontrastive learning
    warmup = 20  # warmup epoch of supcontrastive learning
    name_label = '_%s_Contrast%s_warm%s_seed%s'%(
        sampling_method, str(SupCount), str(warmup), str(seed_value))
    # name_label = '_entropy_Contrast%s_freeze%s'%(str(SupCount), str(warmup))

    print(name_label)
    len_sw = 50



    # ======= 1. 数据生成 ======= #
    if os.path.exists(sensor_type+'_data.pkl'):
        with open(sensor_type+'_data.pkl', 'rb') as file:  # 'rb'表示以二进制读取模式打开文件
            d_dict = pickle.load(file)
            data_b = d_dict['data']
            label_b = d_dict['label']
    else:
        all_df = read_sensor_data()
        selected_df = all_df[all_df.label_id != -2]

        if (sensor_type == 'Accel'):
            columns = ['acc_x', 'acc_y', 'acc_z', 'label_id']
            selected_np = process_acc(selected_df, columns)
        elif (sensor_type == 'AccelPress'):
            columns = ['acc_x', 'acc_y', 'acc_z', 'pressure', 'label_id']
            selected_np = process_acc_press(selected_df, columns)
        elif (sensor_type == 'AccelTemp'):
            columns = ['acc_x', 'acc_y', 'acc_z', 'temperature', 'label_id']
            selected_np = process_acc_temperature(selected_df, columns)
        elif (sensor_type == 'AccelGyro'):
            columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label_id']
            selected_np = process_acc_gyr(selected_df, columns)
        elif (sensor_type == 'AccelGPS'):  # todo
            columns = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing', 'label_id']
            selected_np = process_acc_gps(selected_df, columns)
        elif (sensor_type == 'PressTemp'):
            columns = ['pressure', 'temperature', 'label_id']
            selected_np = process_press_temperature(selected_df, columns)
        else:
            print('no model available')

        tmp_b = sliding_window(selected_np, len_sw, len_sw)
        # concatenate list
        data_b = np.transpose(tmp_b[:, :, :-1], (0, 2, 1))  # [B, Len, dim-1] -> [B, dim-1, Len]
        label_b = tmp_b[:, :, -1]  # [B, Len]

        # 将数据保存到文件
        with open(sensor_type+'_data.pkl', 'wb') as file:  # 'wb'表示以二进制写入模式打开文件
            pickle.dump({'data': data_b, 'label': label_b}, file)

    majority_label, minority_label = find_majority_minority(label_b)
    # 将数据分为8:2，其中2为测试集
    vote_label = majority_value(label_b)
    X_train_full, X_test, y_train_full, y_test = (
                              train_test_split(data_b, label_b,
                              test_size=0.2, stratify=vote_label,
                              # random_state=42
                                               ))

    # 从X_train_full中随机选择1%的数据作为X_labeled，其余为X_unlabeled
    X_labeled, X_unlabeled, y_labeled, y_unlabeled = (
                                train_test_split(X_train_full,
                                                 y_train_full,
                                                 test_size=0.99,
                                                 # random_state=42
                                                 ))

    # 确保X_labeled和X_unlabeled的大小
    print("X_labeled shape:", X_labeled.shape)
    print("X_unlabeled shape:", X_unlabeled.shape)
    print("X_test shape:", X_test.shape)


    # ======= 5. 循环主动学习 ======= #
    # 初始化模型和优化器
    if (sensor_type == 'Accel'):
        model = SimpleNN_1s()
    elif (sensor_type == 'AccelPress') or (sensor_type == 'AccelTemp'):
        model = SimpleNN_13s()
    elif (sensor_type == 'AccelGyro'):
        model = SimpleNN_33s()
    elif (sensor_type == 'AccelGPS'):
        model = SimpleNN_32s()
    elif (sensor_type == 'PressTemp'):
        model = SimpleNN_11s()
    else:
        print('no model available')
    model = model.to(device)
    classify_criterion = nn.CrossEntropyLoss()
    Contrast_criterion = ContrastiveLoss()
    supContrast_criterion = SupContrastiveLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)


    # 主动学习迭代
    update_size = max(int(0.01 * len(X_train_full)), 2)  # 每次选择 1% 的样本
    batch_size = 4000
    iteration = 1

    ## 创建数据加载器
    plot_dataset = data_loader_umineko(data_b.astype(float), label_b.astype(int))
    plot_loader = DataLoader(plot_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    ## initial plot
    repres_list, sample_list, pred_list, label_list = \
        AE_eval_time_series(plot_loader, model, device)
    ari, nmi, fmi, silhouette = (
            vis_scatter_label_2d(repres_list,
                                 label_list, 0,
                                 sensor_type+name_label,
                                 plot_flag=True,
                                 is_omizu=True))

    train_accuracy_list, train_macro_f1_list, train_micro_f1_list = [], [], []
    test_accuracy_list, test_macro_f1_list, test_micro_f1_list = [], [], []
    test_pred_label_lists, test_truth_label_lists = [], []
    weight_list, loss_list, selected_labels_list = [], [], []
    ari_list, nmi_list, fmi_list, silhouette_list = [], [], [], []
    label_data_list, unlabel_data_list = [], []
    while len(X_unlabeled) > 0:
        print(f"==============Iteration {iteration}===============")
        print(f"Already labeled samples: {len(X_labeled)}")
        print(f"Remaining unlabeled samples: {len(X_unlabeled)}")

        # 创建数据加载器
        labeled_dataset = data_loader_umineko(X_labeled.astype(float), y_labeled.astype(int))
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        ## stage 1: train supCon model (所有数据都来训练)
        unfreeze_encoders(model)
        if iteration < warmup:
            epoch = 10
            # epoch = 500
            model, _ = train_model(model, labeled_loader, supContrast_criterion, optimizer,
                                          epochs=epoch, device=device, if_contrast=True)


        ## stage 2: train supervised model, 获得分类器新参数
        freeze_encoders(model)  # only classifier is trainable
        model, avg_loss1 = train_model(model, labeled_loader, classify_criterion, optimizer,
                                      epochs=50, device=device, if_contrast=False)
                                      # epochs=10, device=device, if_contrast=False)


        unfreeze_all(model)  # only projector is NOT trainable
        # freeze_projector(model)  # only projector is NOT trainable
        model, avg_loss2 = train_model(model, labeled_loader, classify_criterion, optimizer,
                                epochs=50, device=device, if_contrast=False)


        loss_list.append(np.average(avg_loss2))  # 记录50次epoch的平均loss

        ## plot
        if (iteration == 1) or (iteration == 10) or (iteration == 20):
            plot_flag = True
        else:
            plot_flag = False
        repres_list, sample_list, pred_list, label_list = \
            AE_eval_time_series(plot_loader, model, device)
        ari, nmi, fmi, silhouette = (
            vis_scatter_label_2d(repres_list,
                                 label_list, iteration,
                                 sensor_type+name_label,
                                 plot_flag=plot_flag,
                                 is_omizu=True))


        # ======= 6. 模型测试 ======= #
        print('-------Training----------')
        accuracy, macro_f1, micro_f1, pred_label_list, truth_label_list = (
            evaluate_model(model, labeled_loader, device))
        train_accuracy_list.append(accuracy)
        train_macro_f1_list.append(macro_f1)
        train_micro_f1_list.append(micro_f1)

        test_dataset = data_loader_umineko(X_test.astype(float), y_test.astype(int))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        print('-------Test----------')
        (accuracy, macro_f1, micro_f1,
         rest_datan, pred_label_list, truth_label_list) = (
            evaluate_supContrast_model(model, test_loader, device, cal_threshold=0.5))
        test_accuracy_list.append(accuracy)
        test_micro_f1_list.append(micro_f1)
        test_macro_f1_list.append(macro_f1)
        test_pred_label_lists.append(pred_label_list)
        test_truth_label_lists.append(truth_label_list)

        print('-------cluster results----------')
        ari_list.append(ari)
        nmi_list.append(nmi)
        fmi_list.append(fmi)
        silhouette_list.append(silhouette)

        # 保存当前模型权重到列表
        weight_list.append(copy.deepcopy(model.state_dict()))
        # 保存当前label和unlabel数据
        label_data_list.append((X_labeled, y_labeled))
        unlabel_data_list.append((X_unlabeled, y_unlabeled))

        # # stage 3: 更新数据集
        select_size = min(update_size, len(X_unlabeled))

        ### uncertainty_sampling
        X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_data = (
            data_sampling(sampling_method, X_labeled, y_labeled,
                          X_unlabeled, y_unlabeled,
                          model, select_size,
                          majority_label=majority_label, minority_label=minority_label,
                          device=device))

        # X_labeled, y_labeled, X_unlabeled, y_unlabeled, selected_labels = (
        #     uncertainty_sampling(X_labeled, y_labeled,
        #                          X_unlabeled, y_unlabeled,
        #                          model, select_size, device))  # data变化了
        selected_labels_list.append(selected_data)  # selected_data = (newX_labeled, newy_labeled)

        # 每十次迭代，也就是增加10%的数据后，保存一次模型和heatmap
        # if iteration % 10 == 0:
        if (iteration == 1) or (iteration == 10) or (iteration == 20):
            plot_confusion_matrix(pred_label_list, truth_label_list,
                                  name='%s_%s_pred_%s'%(sensor_type,name_label,str(iteration)), if_omizu=True)
        # if iteration == 20:
        #     break
        iteration += 1
    # # last heatmap
    # plot_confusion_matrix(test_pred_label_lists[-1], test_truth_label_lists[-1],
    #                       name='%s_%s_pred_%s'%(sensor_type,name_label,str(iteration)))
    # repres_list, sample_list, pred_list, label_list = \
    #         AE_eval_time_series(plot_loader, model, device)
    # # ari, nmi, fmi, silhouette = (
    # #     plot_func(repres_list, sample_list, label_list, sample_list,  # not necessary to plot pred_list
    # #           -1, sensor_type+name_label))
    # ari, nmi, fmi, silhouette = (
    #         vis_scatter_label_2d(repres_list,
    #                              label_list, -1,
    #                              sensor_type+name_label))
    # # 保存结果
    with open(sensor_type+'_%s_epoch%s_results.pkl'%(name_label, str(iteration)), 'wb') as f:
        result_dict = {
            'train_accuracy_list': train_accuracy_list,
            'train_macro_f1_list': train_macro_f1_list,
            'train_micro_f1_list': train_micro_f1_list,
            'test_accuracy_list': test_accuracy_list,
            'test_macro_f1_list': test_macro_f1_list,
            'test_micro_f1_list': test_micro_f1_list,
            'test_pred_label_lists': test_pred_label_lists,
            'test_truth_label_lists': test_truth_label_lists,
            'selected_labels_list': selected_labels_list,
            'label_data_list': label_data_list,
            'unlabel_data_list': unlabel_data_list,
            'weight_list': weight_list,
            'ari_list': ari_list,
            'nmi_list': nmi_list,
            'fmi_list': fmi_list,
            'silhouette_list': silhouette_list
        }

        pickle.dump(result_dict, f)


    print("All unlabeled samples have been labeled!")

# run(seed_value=2, sensor_type='AccelGPS', sampling_method = 'entropy')

# run(seed_value=1, sensor_type='AccelTemp', sampling_method = 'random')
# run(seed_value=10, sensor_type='AccelTemp', sampling_method = 'entropy')
# run(seed_value=5, sensor_type='AccelTemp', sampling_method = 'random')

run(seed_value=1, sensor_type='AccelTemp', sampling_method = 'underRandom')
run(seed_value=10, sensor_type='AccelTemp', sampling_method = 'underRandom')
run(seed_value=2, sensor_type='AccelTemp', sampling_method = 'underRandom')
run(seed_value=5, sensor_type='AccelTemp', sampling_method = 'underRandom')
run(seed_value=2025, sensor_type='AccelTemp', sampling_method = 'underRandom')
