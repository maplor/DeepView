
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
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from deepview.clustering_pytorch.datasets import (
    prepare_all_data,
)
from deepview.clustering_pytorch.nnet.common_config import get_model
from deepview.clustering_pytorch.datasets.factory import prepare_unsup_dataset
from deepview.clustering_pytorch.nnet.train_utils import AE_eval_time_series, AE_eval_time_series_labelflag

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# import matplotlib
# matplotlib.use('TkAgg')  # or 'Agg', 'Qt5Agg', etc.
# import matplotlib.pyplot as plt

import pyqtgraph as pg
from PySide6 import QtWidgets
#-----------------prepare data-------------------------------
def set_loader(labeled_flag, data_path, select_filenames, data_column):
    # 相同代码在clustering pytorch/core的train.py出现
    data_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsupervised-datasets\allDataSet'
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



#-----------------plot dot-------------------------------


# def main(model_name, full_model_path, new_column_names, labeled_flag, data_path, select_filenames):
def handle_old_scatter_data(model_name, full_model_path, new_column_names, labeled_flag, data_path, select_filenames):
    # model_name = 'AE_CNN'
    p_setup = 'autoencoder'
    # num_channel = 5  # 根据模型文件pkl的名字，找到使用多少传感器，计算总列数
    # data_len = 180  # 根据模型文件pkl的名字
    # full_model_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug4\AE_CNN_epoch0_datalen180_gps-acceleration.pth'
    # full_model_path = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration.pth'
    # new_column_names = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    # labeled_flag = False
    # build data loader
    train_loader, _ = set_loader(labeled_flag, data_path, select_filenames, new_column_names)

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
    # plt.figure(figsize=(8, 6))

    # Add labels and title
    # plt.xlabel('X-axis Label')
    # plt.ylabel('Y-axis Label')
    # plt.title('autoencoder,encoder latent representation')
    # plt.legend()
    #
    # # Show the plot
    # # plt.show()
    # plt.savefig('autoencoder.png')
    

    return repre_tsne, flag_concat, label_concat


def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array

if __name__ == '__main__':
    model_name = 'AE_CNN'
    p_setup = 'autoencoder'
    # num_channel = 5  # 根据模型文件pkl的名字，找到使用多少传感器，计算总列数
    # data_len = 180  # 根据模型文件pkl的名字
    full_model_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug4\AE_CNN_epoch0_datalen180_gps-acceleration.pth'
    # full_model_path = r'C:\Users\dell\Desktop\ss-cc-2024-08-05\unsup-models\iteration-0\ssAug5\AE_CNN_epoch29_datalen180_gps-acceleration.pth'
    new_column_names = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
    labeled_flag = False
    data_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsupervised-datasets\allDataSet'
    select_filenames = ['Omizunagidori2018_raw_data_9B36578_lb0009_25Hz.pkl']

    repre_tsne, flag_concat, label_concat = handle_old_scatter_data(model_name, full_model_path, new_column_names, labeled_flag, data_path, select_filenames)

    real_label_idxs_unlabeled = np.where(flag_concat[:, 0, 0] == 0)[0]
    x_unlabeled = repre_tsne[real_label_idxs_unlabeled, 0]
    y_unlabeled = repre_tsne[real_label_idxs_unlabeled, 1]


    # real_label_idxs = np.where(flag_concat[:,0,0] == 1)[0]
    # x = repre_tsne[real_label_idxs, 0]
    # y = repre_tsne[real_label_idxs, 1]
    # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
    # or
    real_label_idxs_labeled = np.where(flag_concat[:, 0, 0] == 1)[0]
    x_labeled = repre_tsne[real_label_idxs_labeled, 0]
    y_labeled = repre_tsne[real_label_idxs_labeled, 1]
    real_label_concat = label_concat[real_label_idxs_labeled, 0, 0]
    app = QtWidgets.QApplication([])
    win = pg.GraphicsLayoutWidget(show=True, title="Autoencoder Latent Representation")
    plot = win.addPlot(title="Autoencoder, Encoder Latent Representation")

    scatter1 = pg.ScatterPlotItem(x=x_unlabeled, y=y_unlabeled, pen=pg.mkPen(None),
                                  brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                  data=real_label_idxs_unlabeled)
    # print(real_label_idxs_unlabeled)
    plot.addItem(scatter1)

    color_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
    brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
    scatter2 = pg.ScatterPlotItem(x=x_labeled, y=y_labeled, pen=pg.mkPen(None),
                                  brush=brushes, size=5, data=real_label_idxs_labeled)
    # print("2ge")
    # print(real_label_idxs_labeled)
    plot.addItem(scatter2)

    last_modified_points = []

    def change_points_properties(indices, new_size, new_color):
        global last_modified_points

        # Restore original properties of last modified points
        for p, original_size, original_brush in last_modified_points:
            p.setSize(original_size)
            p.setBrush(original_brush)

        last_modified_points = []  # Clear the list

        # Change properties of new points
        for scatter in [scatter1, scatter2]:
            points = scatter.points()
            for p in points:
                if p.data() in indices:
                    # Save current properties
                    original_size = p.size()
                    original_brush = p.brush()
                    last_modified_points.append((p, original_size, original_brush))

                    # Change to new properties
                    p.setSize(new_size)
                    p.setBrush(pg.mkBrush(new_color))

    scatter1.sigClicked.connect(lambda _, points: on_click(points))
    scatter2.sigClicked.connect(lambda _, points: on_click(points))

    def on_click(points):
        idxs = []  # 初始化 idxs 列表
        for p in points:
            idx = p.data()
            # print(f"Clicked point: Index={idx}, X={p.pos().x()}, Y={p.pos().y()}")
            idxs.append(idx)  # 将 idx 添加到 idxs 列表中
        change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))

    app.exec()
