import datetime
import json
import logging
import pickle
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
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

from deepview.gui.supervised_cl.data_display import SupervisedClDataMixin

# 从sklearn.manifold导入TSNE
from sklearn.manifold import TSNE
# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_raw_data_folder,
    grab_files_in_folder_deep
)

from deepview.gui.supervised_cl.ui.select_model_widget import SelectModelWidget

from deepview.gui.supervised_cl.ui.new_scatter_map import NewScatterMapWidget
from deepview.gui.supervised_cl.ui.old_scatter_map import OldScatterMapWidget
from deepview.gui.supervised_cl.ui.select_parameters_widget import SelectParametersWidget
from deepview.gui.supervised_cl.ui.data_attributs import DataAttributesWidget


class SupervisedClWidget(SupervisedClDataMixin, QWidget):
    dataChanged = Signal(pd.DataFrame)

    def __init__(self, root, cfg) -> None:
        super().__init__()

        # 保存根对象
        self.root = root
        # 读取根对象的配置
        self.root_cfg = read_config(root.config)
        # 保存标签字典
        self.label_dict = self.root_cfg['label_dict']
        # 保存传感器字典
        self.sensor_dict = self.root_cfg['sensor_dict']
        # 创建一个空的DataFrame
        self.data = pd.DataFrame()
        # 配置
        self.cfg = cfg
        # 模型参数
        self.data_path = ''
        self.data_name = ''
        # 初始化模型路径列表
        self.model_path = []
        # 初始化模型名称
        self.model_name = ''
        # 初始化数据长度
        self.data_length = 180  # todo 默认，但是需要从文件名中读取
        # 初始化列名列表
        self.column_names = []


        # 用于三个散点图的交互
        self.last_modified_points = []

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
        

        # 右上参数选择区域
        self.select_parameters_widget = SelectParametersWidget(self)
        self.top_layout.addWidget(self.select_model_widget, stretch=1)
        self.top_layout.addWidget(self.select_parameters_widget, stretch=1)
        

        # 显示标签选择框部分
        self.old_checkbox_layout = QHBoxLayout()
        self.old_checkbox_layout.setAlignment(Qt.AlignLeft)
        self.old_existing_checkbox = QCheckBox("Show existing\n    labels")
        self.old_existing_checkbox.setChecked(True)
        self.old_checkbox_layout.addWidget(self.old_existing_checkbox)
        self.old_manual_checkbox = QCheckBox("Show manual\n    labels")
        self.old_manual_checkbox.setChecked(True)
        self.old_checkbox_layout.addWidget(self.old_manual_checkbox)

        self.new_checkbox_layout = QHBoxLayout()
        self.new_checkbox_layout.setAlignment(Qt.AlignLeft)
        self.new_existing_checkbox = QCheckBox("Show existing\n    labels")
        self.new_existing_checkbox.setChecked(True)
        # 获取checkbox的值

        self.new_checkbox_layout.addWidget(self.new_existing_checkbox)
        self.new_manual_checkbox = QCheckBox("Show manual\n    labels")
        self.new_manual_checkbox.setChecked(True)
        self.new_checkbox_layout.addWidget(self.new_manual_checkbox)
        # 设置勾选框的背景颜色为紫色
        # self.setStyleSheet("QCheckBox { background-color: #c76dff; color: white; padding-right: 17px;}")

        # 设置对象名称
        self.old_existing_checkbox.setObjectName("oldExistingCheckbox")
        self.old_manual_checkbox.setObjectName("oldManualCheckbox")
        self.new_existing_checkbox.setObjectName("newExistingCheckbox")
        self.new_manual_checkbox.setObjectName("newManualCheckbox")

        # 设置特定复选框的样式
        self.setStyleSheet("""
            QCheckBox#oldExistingCheckbox, QCheckBox#oldManualCheckbox,
            QCheckBox#newExistingCheckbox, QCheckBox#newManualCheckbox {
                background-color: #c76dff;
                color: white;
                padding-right: 17px;
            }
        """)


        # 添加到布局
        self.select_lable_layout.addLayout(self.old_checkbox_layout)
        self.select_lable_layout.addLayout(self.new_checkbox_layout)

        # 散点图头上标题部分
        # 创建一个标签
        label = QLabel("Label names and colors display")
        label.setAlignment(Qt.AlignHCenter)

        # 设置标签的样式
        label.setStyleSheet("""
            QLabel {
                background-color: #1692f7;  /* 浅蓝色背景 */
                color: #FFFFFF;  /* 白色字体 */
                font-size: 16px;  /* 字体大小 */
            }
        """)


        self.scatter_title.addWidget(label)

        # # 散点图部分
        # self.old_scatter_map_widget = OldScatterMapWidget(self,
        #                                                   data,
        #                                                   model_filename,
        #                                                   data_length,
        #                                                   data_columns)
        # self.new_scatter_map_widget = NewScatterMapWidget(self,
        #                                                   data,
        #                                                   model_filename,
        #                                                   data_length,
        #                                                   data_columns)
        self.old_scatter_map_widget = OldScatterMapWidget(self)
        self.new_scatter_map_widget = NewScatterMapWidget(self)



        # 添加部件到布局
        self.all_scatter_area.addWidget(self.old_scatter_map_widget, stretch=1)
        self.all_scatter_area.addWidget(self.new_scatter_map_widget, stretch=1)


        # 底部组件

        # self.data_attribute_widget = DataAttributesWidget(self)

        # 按顺序摆放各个区域
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.select_lable_layout)
        self.main_layout.addLayout(self.scatter_title)
        self.main_layout.addLayout(self.all_scatter_area, stretch=1)
        # self.main_layout.addWidget(self.data_attribute_widget)

        # data display按钮连接到display_old_scatter_data方法
        self.select_model_widget.display_button.clicked.connect(self.display_old_scatter_data)
        # applySCL_button按钮连接到display_new_scatter_data方法
        self.select_parameters_widget.applySCL_button.clicked.connect(self.display_new_scatter_data)
        # 保存按钮连接到save_model方法
        self.select_parameters_widget.save_model_button.clicked.connect(self.save_model)
        # 勾选框改变触发
        self.old_existing_checkbox.stateChanged.connect(self.old_scatter_map_widget.update_existing_labels_status)
        self.old_manual_checkbox.stateChanged.connect(self.old_scatter_map_widget.update_manual_labels_status)
        self.new_existing_checkbox.stateChanged.connect(self.new_scatter_map_widget.update_existing_labels_status)
        self.new_manual_checkbox.stateChanged.connect(self.new_scatter_map_widget.update_manual_labels_status)

        

    # def process_and_display_data(self, widget, display_method):
    #     # read sensor data
    #     model_filename = self.select_model_widget.modelComboBox.currentText()
    #     # preprocessing: find column names in dataframe
    #     model_name, data_length, column_names = get_param_from_path(model_filename)
    #     # data, _ = get_data_from_pkl(self.select_model_widget.RawDatacomboBox.currentText(), self.cfg)
    #     # transfer sensor name to columns
    #     data_columns = transfer_sensor2columns(column_names, self.sensor_dict)
    #     display_method(data, model_filename, data_length, data_columns)

    # def display_old_scatter_data(self):
    #     self.process_and_display_data(self.old_scatter_map_widget, self.old_scatter_map_widget.display_data)

    # def display_new_scatter_data(self):
    #     self.process_and_display_data(self.new_scatter_map_widget, self.new_scatter_map_widget.display_data)
