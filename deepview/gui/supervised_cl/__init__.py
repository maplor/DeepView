import datetime
import json
import logging
import os
import pickle
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
from PySide6.QtCore import (
    QObject, Signal, Slot, QTimer, Qt
)
# 从PySide6.QtCore导入QTimer, QRectF, Qt
from PySide6.QtCore import QRectF
# 从PySide6.QtWidgets导入多个类
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox
)
from PySide6.QtGui import QPalette, QColor

# 从sklearn.manifold导入TSNE
from sklearn.manifold import TSNE
# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep
)

from deepview.gui.supervised_cl.ui.select_model_widget import SelectModelWidget

from deepview.gui.supervised_cl.ui.new_scatter_map import NewScatterMapWidget
from deepview.gui.supervised_cl.ui.old_scatter_map import OldScatterMapWidget



class SupervisedClWidget(QWidget):
    dataChanged = Signal(pd.DataFrame)

    def __init__(self, root, cfg) -> None:
        super().__init__()

        # 保存根对象
        self.root = root
        # 读取根对象的配置
        root_cfg = read_config(root.config)
        # 保存标签字典
        self.label_dict = root_cfg['label_dict']
        # 保存传感器字典
        self.sensor_dict = root_cfg['sensor_dict']
        # 创建一个空的DataFrame
        self.data = pd.DataFrame()
        # 配置
        self.cfg = cfg
        # 模型参数
        # 初始化模型路径列表
        self.model_path = []
        # 初始化模型名称
        self.model_name = ''
        # 初始化数据长度
        self.data_length = 90
        # 初始化列名列表
        self.column_names = []

        # 主
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        # 顶部
        self.top_layout = QHBoxLayout()
        # 选择显示标签部分，分左右两部分
        self.select_lable_layout = QHBoxLayout()
        # 散点图头上标题
        self.scatter_title = QHBoxLayout()
        # 散点图区域
        self.all_scatter_area = QHBoxLayout()


        # 左上模型选择区域
        self.select_model_widget = SelectModelWidget(self)
        
        # self.select_model_widget.display_button.connect()

        # self.top_layout.addLayout(self.select_model_widget)
        self.top_layout.addWidget(self.select_model_widget)

        # 显示标签选择框部分
        
        # 散点图头上标题部分
        # 创建一个标签
        label = QLabel("Label names and colors display")
        label.setAlignment(Qt.AlignCenter)

        # 设置标签的样式
        palette = label.palette()
        palette.setColor(QPalette.Window, QColor("#ADD8E6"))  # 浅蓝色背景
        palette.setColor(QPalette.WindowText, QColor("#FFFFFF"))  # 白色字体
        label.setAutoFillBackground(True)
        label.setPalette(palette)
        self.scatter_title.addWidget(label)

        # 散点图部分
        self.old_scatter_map_widget = OldScatterMapWidget(self)
        self.new_scatter_map_widget = NewScatterMapWidget(self)
        # 添加部件到布局
        self.all_scatter_area.addWidget(self.old_scatter_map_widget, stretch=1)
        self.all_scatter_area.addWidget(self.new_scatter_map_widget, stretch=1)
        # self.all_scatter_area.addWidget(self.old_scatter_map_widget)
        # self.all_scatter_area.addWidget(self.new_scatter_map_widget)

        # 按顺序摆放各个区域
        self.main_layout.addLayout(self.top_layout)

        self.main_layout.addLayout(self.scatter_title)
        self.main_layout.addLayout(self.all_scatter_area)


    
    def handleCompute(self):

        self.get_data_from_pkl(self.select_model_widget.RawDatacomboBox.currentText())


    # 从.pkl文件获取数据的方法
    def get_data_from_pkl(self, filename):
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 构建文件路径
        datapath = os.path.join(self.cfg["project_path"], unsup_data_path, filename)
        # 打开.pkl文件
        with open(datapath, 'rb') as f:
            # 加载数据
            self.data = pickle.load(f)
            # 将UNIX时间戳转换为ISO 8601格式
            self.data['timestamp'] = pd.to_datetime(self.data['unixtime'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').str[:-4] + 'Z'
            self.data['index'] = self.data.index  # Add an index column
            self.dataChanged.emit(self.data)
        return
    
    def get_model_param_from_path(self, model_path):
        # 从路径获取模型参数的方法
        if model_path:
            model_name, data_length, column_names = get_param_from_path(model_path)
            # 保存到主窗口的属性
            self.model_path = model_path
            self.model_name = model_name
            self.data_length = data_length
            self.column_names = column_names
