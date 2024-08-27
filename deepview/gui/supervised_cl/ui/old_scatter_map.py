import os
from pathlib import Path
import pickle
import pandas as pd
import pyqtgraph as pg
import numpy as np

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
from deepview.gui.supervised_contrastive_learning.load_unsup_model_feature import handle_old_scatter_data


class OldScatterMapWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        self.repre_tsne, self.flag_concat, self.label_concat = None, None, None

        # Create a GraphicsLayoutWidget
        # self.win = pg.GraphicsLayoutWidget(show=True, title="Autoencoder Latent Representation")
        self.win = pg.PlotWidget()
        self.layout.addWidget(self.win)

        # Add a plot to the widget
        # self.plot = self.win.addPlot(title="Autoencoder, Encoder Latent Representation")

        # TODO Data setup，改为从主界面调用
        self.model_name = 'AE_CNN'
        self.p_setup = 'autoencoder'
        self.full_model_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsup-models\iteration-0\fastAug4\AE_CNN_epoch0_datalen180_gps-acceleration.pth'
        self.new_column_names = ['acc_x', 'acc_y', 'acc_z', 'GPS_velocity', 'GPS_bearing']
        self.labeled_flag = False
        self.data_path = r'C:\Users\user\Desktop\fast-test-2024-08-04\unsupervised-datasets\allDataSet'
        self.select_filenames = ['Omizunagidori2018_raw_data_9B36578_lb0009_25Hz.pkl']

        # Initialize scatter plot items
        self.scatter1 = None
        self.scatter2 = None
        self.last_modified_points = []

        # TODO Add data to plot,绑定到display data按钮
        # self.add_data_to_plot()

    def add_data_to_plot(self):
        # Load data
        self.repre_tsne, self.flag_concat, self.label_concat = handle_old_scatter_data(
            self.model_name, self.full_model_path, self.new_column_names,
            self.labeled_flag, self.data_path, self.select_filenames
        )
        real_label_idxs_unlabeled = np.where(self.flag_concat[:, 0, 0] == 0)[0]
        x_unlabeled = self.repre_tsne[real_label_idxs_unlabeled, 0]
        y_unlabeled = self.repre_tsne[real_label_idxs_unlabeled, 1]

        real_label_idxs_labeled = np.where(self.flag_concat[:, 0, 0] == 1)[0]
        x_labeled = self.repre_tsne[real_label_idxs_labeled, 0]
        y_labeled = self.repre_tsne[real_label_idxs_labeled, 1]
        real_label_concat = self.label_concat[real_label_idxs_labeled, 0, 0]

        # Create scatter plot items
        self.scatter1 = pg.ScatterPlotItem(x=x_unlabeled, y=y_unlabeled, pen=pg.mkPen(None),
                                           brush=pg.mkBrush(128, 128, 128, 128), size=5,
                                           data=real_label_idxs_unlabeled)
        self.win.addItem(self.scatter1)

        color_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
        brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter2 = pg.ScatterPlotItem(x=x_labeled, y=y_labeled, pen=pg.mkPen(None),
                                           brush=brushes, size=5, data=real_label_idxs_labeled)
        self.win.addItem(self.scatter2)

        # Connect signals
        # self.scatter1.sigClicked.connect(self.on_click)
        # self.scatter2.sigClicked.connect(self.on_click)
        self.scatter1.sigClicked.connect(lambda plot, points: self.on_click(self.scatter1, points))
        self.scatter2.sigClicked.connect(lambda plot, points: self.on_click(self.scatter2, points))

    def change_points_properties(self, indices, new_size, new_color):
        # Restore original properties of last modified points
        for p, original_size, original_brush in self.last_modified_points:
            p.setSize(original_size)
            p.setBrush(original_brush)

        self.last_modified_points = []  # Clear the list

        # Change properties of new points
        for scatter in [self.scatter1, self.scatter2]:
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

    # def on_click(self, points):
    def on_click(self, scatter, points):
        idxs = []  # Initialize idxs list
        for p in points:
            idx = p.data()
            idxs.append(idx)  # Add idx to idxs list
        self.change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))
