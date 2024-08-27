import os
from pathlib import Path
import pickle
import pandas as pd
import pyqtgraph as pg
import numpy as np
import argparse

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox
)

from PySide6.QtCore import (
    QObject, Signal, Slot, QTimer, Qt
)

# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep
)

# 
from deepview.gui.supervised_contrastive_learning.runscl_cl import parse_options, set_loader, set_model, train, evaluate
from deepview.gui.supervised_contrastive_learning.util import adjust_learning_rate, warmup_learning_rate, set_optimizer, save_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
p_setup = 'simclr'
if p_setup == 'simclr':
    AUGMENT = True  # 也存到yaml文件或者opt里
else:
    AUGMENT = False  # 是否用data augment，作为参数存储

class NewScatterMapWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # Create a vertical layout
        self.layout = QHBoxLayout(self)

        self.map_1 = pg.PlotWidget()
        self.map_2 = pg.PlotWidget()

        # self.map_1 = pg.GraphicsLayoutWidget(show=False, title="Autoencoder Latent Representation")
        # self.map_2 = pg.GraphicsLayoutWidget(show=False, title="Autoencoder Latent Representation")

        self.layout.addWidget(self.map_1)
        self.layout.addWidget(self.map_2)

        # TODO Data setup，改为从主界面调用

        # self.add_data_to_plot()

    # TUDO 需要将参数拉出来
    def add_data_to_plot(self):
        opt_dict = parse_options()
        opt = argparse.Namespace(**opt_dict)
        # build data loader
        train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=True)

        # build model and criterion
        model, criterion = set_model(opt)
        # build optimizer
        optimizer = set_optimizer(opt, model)

        # training routine
        '''
        opt.epochs参数从Parameter选框中读取
        当点击 Apply SCL按钮后运行下面程序
        '''
        epoch = 0
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        # evaluate and plot
        train_loader, _ = set_loader(augment=AUGMENT, labeled_flag=False)
        '''
        第一张New scatter map生成方式
        '''
        # self.map_1 = self.map_1.addPlot(title=f"supContrast,{epoch}")

        repre_tsne, flag_concat, label_concat = evaluate(model, epoch, train_loader, fig_name='supContrast')

        real_label_idxs_unlabeled_1 = np.where(flag_concat == 0)[0]
        x_unlabeled_1 = repre_tsne[real_label_idxs_unlabeled_1, 0]
        y_unlabeled_1 = repre_tsne[real_label_idxs_unlabeled_1, 1]
        self.scatter1 = pg.ScatterPlotItem(x=x_unlabeled_1, y=y_unlabeled_1, pen=pg.mkPen(None),
                                    brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                    data=real_label_idxs_unlabeled_1)

        self.map_1.addItem(self.scatter1)

        real_label_idxs_labeled_1 = np.where(flag_concat == 1)[0]
        x_labeled_1 = repre_tsne[real_label_idxs_labeled_1, 0]
        y_labeled_1 = repre_tsne[real_label_idxs_labeled_1, 1]
        # plt.scatter(x, y, color='blue', alpha=0.5, label='labeled')
        real_label_concat = label_concat[real_label_idxs_labeled_1]
        color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
        brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter2 = pg.ScatterPlotItem(x=x_labeled_1, y=y_labeled_1, pen=pg.mkPen(None),
                                    brush=brushes, size=5, data=real_label_idxs_labeled_1)
        self.map_1.addItem(self.scatter2)

        # standard contrastive learning
        opt.method = 'SimCLR'
        for epoch in range(1, opt.epochs + 1):
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)
        # evaluate and plot
        '''
        第2张New scatter map生成方式
        '''
        # self.map_2 = self.map_2.addPlot(title=f"Contrast,{epoch}")

        repre_tsne, flag_concat, label_concat = evaluate(model, epoch, train_loader, fig_name='Contrast')
        real_label_idxs_unlabeled_2 = np.where(flag_concat == 0)[0]
        x_unlabeled_2 = repre_tsne[real_label_idxs_unlabeled_2, 0]
        y_unlabeled_2 = repre_tsne[real_label_idxs_unlabeled_2, 1]
        self.scatter3 = pg.ScatterPlotItem(x=x_unlabeled_2, y=y_unlabeled_2, pen=pg.mkPen(None),
                                    brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                    data=real_label_idxs_unlabeled_2)

        self.map_2.addItem(self.scatter3)

        real_label_idxs_labeled_2 = np.where(flag_concat == 1)[0]
        x_labeled_2 = repre_tsne[real_label_idxs_labeled_2, 0]
        y_labeled_2 = repre_tsne[real_label_idxs_labeled_2, 1]
    
        real_label_concat = label_concat[real_label_idxs_labeled_2]
        color_dict = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow'}
        brushes2 = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter4 = pg.ScatterPlotItem(x=x_labeled_2, y=y_labeled_2, pen=pg.mkPen(None),
                                    brush=brushes2, size=5, data=real_label_idxs_labeled_2)
        self.map_2.addItem(self.scatter4)

        # Connect signals
        self.scatter1.sigClicked.connect(self.on_click)
        self.scatter2.sigClicked.connect(self.on_click)
        self.scatter3.sigClicked.connect(self.on_click)
        self.scatter4.sigClicked.connect(self.on_click)




    def change_points_properties(self, indices, new_size, new_color):
        # Restore original properties of last modified points
        for p, original_size, original_brush in self.last_modified_points:
            p.setSize(original_size)
            p.setBrush(original_brush)

        self.last_modified_points = []  # Clear the list

        # Change properties of new points
        for scatter in [self.scatter1, self.scatter2, self.scatter3, self.scatter4]:
            points = scatter.points()
            for p in points:
                if p.data() in indices:
                    # Save current properties
                    original_size = p.size()
                    original_brush = p.brush()
                    self.last_modified_points.append((p, original_size, original_brush))

                    # Change to new properties
                    p.setSize(new_size)
                    p.setBrush(pg.mkBrush(new_color))

    def on_click(self, points):
        idxs = []  # Initialize idxs list
        for p in points:
            idx = p.data()
            idxs.append(idx)  # Add idx to idxs list
        self.change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))