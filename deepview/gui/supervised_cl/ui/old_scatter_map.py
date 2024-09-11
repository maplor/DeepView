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


# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep
)



from deepview.gui.label_with_interactive_plot.utils import (
featureExtraction,
get_data_from_pkl,
)


class OldScatterMapWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = self.main_window.root_cfg['sensor_dict']
        self.repre_tsne, self.flag_concat, self.label_concat = None, None, None

        # Create a vertical layout
        self.layout = QVBoxLayout(self)
        
        self.old_map = pg.PlotWidget()
        self.layout.addWidget(self.old_map)

        # Initialize scatter plot items
        self.scatter1 = None
        self.scatter2 = None

        self.existing_labels_status = True
        self.manual_labels_status = True
        
        # TODO Add data to plot,绑定到display data按钮


    def display_data(self, data, model_path, data_length, column_names, model_name):
        self.generate_AE_data(data, model_path, data_length, column_names, model_name)
        # self.generate_AE_data(data, model_name, data_length, column_names)


    def update_existing_labels_status(self, status):
        self.existing_labels_status = status

    def update_manual_labels_status(self, status):
        self.manual_labels_status = status

    def generate_test_data(self):
        num_points = 100
        repre_tsne = np.random.rand(num_points, 2)
        
        # 创建符合三维索引的flag_concat和label_concat
        # xiqxin: flag_concat 表示该tsne点是否有标签？？？
        flag_concat = np.random.randint(0, 2, (num_points, 1, 1))
        label_concat = np.random.randint(0, 4, (num_points, 1, 1))
        #
        # 检查到底传哪些参数 （flag_concat, label_concat）
        self.add_data_to_plot(repre_tsne, flag_concat, label_concat)

    def generate_AE_data(self, data, model_path, data_length, column_names, model_name):
        '''
        之后替换generate_test_data函数
        生成左侧autoencoder的latent representation
        代码复用label_with_interactive_plot/init.py的featureExtraction function
        '''

        # # 特征提取：找到数据帧中的列名
        # model_filename = self.main_window.select_model_widget.modelComboBox.currentText()

        # # preprocessing: find column names in dataframe
        # model_name, data_length, column_names = \
        #     get_param_from_path(model_filename)  # 从路径获取模型参数

        # data, _ = get_data_from_pkl(self.main_window.select_model_widget.RawDatacomboBox.currentText(),
        #                             self.main_window.cfg)
        # get representations
        start_indice, end_indice, pos = featureExtraction(self.main_window.root,
                                                          data,
                                                          data_length,
                                                          column_names,
                                                          model_path,
                                                          model_name)

        # xiqxin: flag_concat 表示该tsne点是否有标签？？？
        num_points = pos.shape[0]
        flag_concat = np.random.randint(0, 2, (num_points, 1, 1))
        label_concat = np.random.randint(0, 4, (num_points, 1, 1))

        # todo 检查参数和generate_test_data函数最后的是否一致
        self.add_data_to_plot(pos, flag_concat, label_concat)
        return

    def add_data_to_plot(self, repre_tsne, flag_concat, label_concat):
        # Load data
        # self.repre_tsne, self.flag_concat, self.label_concat = handle_old_scatter_data(
        #     self.model_name, self.full_model_path, self.new_column_names,
        #     self.labeled_flag, self.data_path, self.select_filenames
        # )

        self.repre_tsne, self.flag_concat, self.label_concat = repre_tsne, flag_concat, label_concat

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
        self.old_map.addItem(self.scatter1)

        color_dict = {0: 'b', 1: 'r', 2: 'g', 3: 'y'}
        brushes = [pg.mkBrush(color_dict[label]) for label in real_label_concat]
        self.scatter2 = pg.ScatterPlotItem(x=x_labeled, y=y_labeled, pen=pg.mkPen(None),
                                           brush=brushes, size=5, data=real_label_idxs_labeled)
        self.old_map.addItem(self.scatter2)

        # Connect signals
        # self.scatter1.sigClicked.connect(self.on_click)
        # self.scatter2.sigClicked.connect(self.on_click)
        self.scatter1.sigClicked.connect(lambda plot, points: self.on_click(self.scatter1, points))
        self.scatter2.sigClicked.connect(lambda plot, points: self.on_click(self.scatter2, points))

    def change_points_properties(self, indices, new_size, new_color):
        # Restore original properties of last modified points
        for p, original_size, original_brush in self.main_window.last_modified_points:
            p.setSize(original_size)
            p.setBrush(original_brush)

        self.main_window.last_modified_points = []  # Clear the list
        
        # 检查 new_scatter_map_widget 是否已初始化
        if hasattr(self.main_window, 'new_scatter_map_widget') and \
           hasattr(self.main_window.new_scatter_map_widget, 'scatter1') and \
           self.main_window.new_scatter_map_widget.scatter1 is not None:
            # Change properties of new points
            for scatter in [self.scatter1, self.scatter2, self.main_window.new_scatter_map_widget.scatter1, self.main_window.new_scatter_map_widget.scatter2, self.main_window.new_scatter_map_widget.scatter3, self.main_window.new_scatter_map_widget.scatter4]:
                points = scatter.points()
                for p in points:
                    if p.data() in indices:
                        # Save current properties
                        original_size = p.size()
                        original_brush = p.brush()
                        self.main_window.last_modified_points.append((p, original_size, original_brush))

                        # Change to new properties
                        p.setSize(new_size)
                        p.setBrush(pg.mkBrush(new_color))
        else:
            # 如果控件未初始化，显示消息框
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("All scatter plots have not yet been initialized.")
            msg_box.setInformativeText("Please initialize the scatter plots before continuing.")
            msg_box.setWindowTitle("Plotting error.")
            msg_box.exec_()


    # def on_click(self, points):
    def on_click(self, scatter, points):
        idxs = []  # Initialize idxs list
        for p in points:
            idx = p.data()
            idxs.append(idx)  # Add idx to idxs list
        self.change_points_properties(idxs, new_size=14, new_color=(255, 0, 0, 255))
