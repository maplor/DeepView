
'''
# assume the unsupervised learning model is AE_CNN, used to get latent space of the labeling tab
# now, output the scatter map of the latent space using one_day data
# refer to runscl.py to compare the scatter map of after the encoder of the AE_CNN is finetuned via scl and simclr
## scl: supervised contrastive learning
## simclr: contrastive learning
'''


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from deepview.clustering_pytorch.datasets import (
    prepare_all_data,
)
from deepview.clustering_pytorch.nnet.common_config import get_model
from deepview.clustering_pytorch.datasets.factory import prepare_unsup_dataset
from deepview.clustering_pytorch.nnet.train_utils import AE_eval_time_series, AE_eval_time_series_labelflag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

#-----------------prepare data-------------------------------
def set_loader(labeled_flag):
    # 相同代码在clustering pytorch/core的train.py出现
    data_path = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsupervised-datasets\allDataSet'
    select_filenames = ['Omizunagidori2018_raw_data_9B36578_lb0009_25Hz.pkl']
    data_len = 180
    data_column = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    batch_size = 1028
    augment = False

    train_dataloader, num_channel = prepare_all_data(data_path,
                                                     select_filenames,
                                                     data_len,
                                                     data_column,
                                                     batch_size,
                                                     augment,
                                                     device,
                                                     labeled_flag=labeled_flag)
    return train_dataloader, num_channel  # trainloader修改label，从batch,len,1变成batch，去掉无标签数据

#-----------------prepare model-------------------------------


#-----------------evaluate model-------------------------------



#-----------------plot-------------------------------


def main():
    model_name = 'AE_CNN'
    p_setup = 'autoencoder'
    # num_channel = 5  # 根据模型文件pkl的名字，找到使用多少传感器，计算总列数
    # data_len = 180  # 根据模型文件pkl的名字
    full_model_path = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration.pth'
    new_column_names = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    labeled_flag = False
    # build data loader
    train_loader, _ = set_loader(labeled_flag=labeled_flag)

    # build model and criterion
    # model, criterion = set_model()
    model = get_model(p_backbone=model_name, p_setup=p_setup, num_channel=len(new_column_names))

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(full_model_path))
    else:
        model.load_state_dict(torch.load(full_model_path, map_location=torch.device('cpu')))

    model = model.to(device)

    # p_setup = 'autoencoder'
    # if not labeled_flag:
    representation_list, _, _, label_list, flag_lists = \
        AE_eval_time_series_labelflag(train_loader, model, device)
    # else:
    #     representation_list, _, _, _ = \
    #         AE_eval_time_series(train_loader, model, device)


    # PCA latent representation to shape=(2, len) PCA降维到形状为 (2, len)
    repre_concat = np.concatenate(representation_list)
    repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
    repre_tsne = reduce_dimension_with_tsne(repre_reshape)
    flag_concat = np.concatenate(flag_lists)
    label_concat = np.concatenate(label_list)

    # Create a scatter plot
    plt.figure(figsize=(8, 6))

    real_label_idxs = np.where(flag_concat[:,0,0] == 0)[0]
    x = repre_tsne[real_label_idxs, 0]
    y = repre_tsne[real_label_idxs, 1]
    plt.scatter(x, y, color='grey', alpha=0.5, label='unlabeled')

    # real_label_idxs = np.where(flag_concat[:,0,0] == 1)[0]
    # x = repre_tsne[real_label_idxs, 0]
    # y = repre_tsne[real_label_idxs, 1]
    # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
    # or
    real_label_idxs = np.where(flag_concat[:,0,0] == 1)[0]
    x = repre_tsne[real_label_idxs, 0]
    y = repre_tsne[real_label_idxs, 1]
    real_label_concat = label_concat[real_label_idxs,0,0]
    color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
    for i in range(len(real_label_concat)):
        plt.scatter(x[int(i)], y[int(i)],
                    color=color_dict[real_label_concat[int(i)]],
                    alpha=0.5)

    # Add labels and title
    plt.xlabel('X-axis Label')
    plt.ylabel('Y-axis Label')
    plt.title('autoencoder,encoder latent representation')
    plt.legend()

    # Show the plot
    # plt.show()
    plt.savefig('autoencoder.png')

    return


def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array

if __name__ == '__main__':
    main()
