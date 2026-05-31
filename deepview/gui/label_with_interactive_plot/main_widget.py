# 导入数学模块
# 从typing模块导入List类型
# 导入torch模块
import datetime
import json
import logging
import os
import pickle
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from PySide6 import QtGui
import cv2
import time

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import torch
from PySide6.QtCore import (
    QObject, Signal, Slot, QTime, QTimer, Qt
)
# 从PySide6.QtCore导入QTimer, QRectF, Qt
from PySide6.QtCore import QRectF
from PySide6.QtCore import QRunnable, QThreadPool, Slot, QThread, QObject, Signal, QFileSystemWatcher
# 从PySide6.QtWidgets导入多个类
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox, QDoubleSpinBox, QFileDialog, QCalendarWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QStandardItemModel, QStandardItem, QColor, QPainter
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QPushButton, QMessageBox, QInputDialog

from PySide6.QtGui import QImage, QPixmap
from datetime import datetime, timedelta
import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QCalendarWidget, QLabel
from PySide6.QtCore import QTime, Qt, QRectF
from PySide6.QtGui import QMouseEvent, QPainter, QColor
import sqlite3
from PySide6.QtCore import QDate
from PySide6.QtGui import QTextCharFormat
from PySide6.QtWidgets import QTextEdit, QTimeEdit, QPushButton


# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep,
    get_db_folder
)

from deepview.gui.label_with_interactive_plot.utils import (
    get_data_from_pkl,
    featureExtraction,
    find_data_columns,
    generate_filename
)

from deepview.gui.label_with_interactive_plot.styles import combobox_style_light, combobox_style_dark
from deepview.gui.label_with_interactive_plot.backend import Backend, BackendMap
from deepview.gui.label_with_interactive_plot._video import VideoControlsMixin
from deepview.gui.label_with_interactive_plot._labeling import LabelControlsMixin
from deepview.gui.label_with_interactive_plot._layout import LayoutMixin
from deepview.gui.label_with_interactive_plot.widgets.combo import (
    LabelOption,
    ReComboBox,
)
from deepview.gui.label_with_interactive_plot.widgets.time_selector import (
    ClickableLabel,
    DateTimeSelector,
)
from deepview.gui.label_with_interactive_plot.video import VideoEditor, VideoProcessor
from deepview.gui.label_with_interactive_plot.workers import (
    HandleComputeWorker,
    SaveCsvTask,
    TaskSignals,
    find_nearest_index,
)


# 创建一个蓝色的pg.mkPen对象，宽度为2
clickedPen = pg.mkPen('b', width=2)

# 定义LabelWithInteractivePlot类，继承自QWidget
class LabelWithInteractivePlot(LabelControlsMixin, VideoControlsMixin, LayoutMixin, QWidget):
    dataChanged = Signal(pd.DataFrame)

    def __init__(self, root, cfg) -> None:
        super().__init__()
        # 创建一个日志记录器
        self.end_indice = []
        self.start_indice = []
        self.logger = logging.getLogger("GUI")
        self.backend = Backend()
        self.backend_map = BackendMap()
        self.yaml = YAML()

        self.select_video_widget = None
        self.plot_window = None
        self.time_series = None
        self.modelComboBox = None
        self.RawDatacomboBox = None

        self.save_csv_thread_pool = QThreadPool()

        # self.data改变时同步数据到backend
        self.dataChanged.connect(self.backend.handle_data_changed)

        # 将折线图backend的点击折线图事件连接到backend_map的高亮地图散点方法
        self.backend.highlightDotByindex.connect(self.backend_map.triggeLineMapHighlightDotByIndex)
        # 点击折线散点高亮散点图散点
        self.backend.highlightScatterDotByindexSign.connect(self.handle_highlight_scatter_dot_by_index)
        # 点击地图散点高亮折线图散点
        self.backend_map.highlightLineChartDotByindex.connect(self.backend.triggeLineChartHighlightDotByIndex)
        # 点击地图散点高亮散点图散点
        self.backend_map.highlightLineChartDotByindex.connect(self.handle_highlight_scatter_dot_by_index)
        self.backend.getSelectedAreaByHtml.connect(self.handleReflectToLatent)
        self.backend.setStartEndTime.connect(self.setStartEndTime)
        self.backend.setStartAndEndDataSign.connect(self.backend_map.highlightLineChartTwoDots)
        self.backend.getSelectedAreaToSaveSign.connect(self.getSelectedAreaToSave)
        self.backend.getSelectedAreaToSaveTimerSign.connect(self.getSelectedAreaToSaveTimer)


        self.button_style = """QPushButton {
            background-color: #1ea123; 
            border: none;
            color: white;
            padding: 5px 10px;
            text-align: center;
            text-decoration: none;
            font-size: 12px;
            margin: 4px 2px;
            border-radius: 10px; 
        }

        QPushButton:pressed {
            background-color: #148f1d; 
        }"""

        self.remove_button_style = """QPushButton { 
            border: none;
            color:#6D6D6D; 
            font-size: 15px; 
            }
        """

        # 初始化特征提取按钮为None
        self.featureExtractBtn = None
        # 初始化当前高亮散点为None
        self.current_highlight_map_scatter = None

        # 保存根对象
        self.root = root
        # 读取根对象的配置
        root_cfg = read_config(root.config)
        # 保存标签字典
        self.label_dict = root_cfg['label_dict']


        # 初始化最后修改的点
        self.last_modified_points = []
        # 预定义颜色数组
        self.colorPalette = ['#91cc75', '#5470c6', '#fac858', '#ee6666',
                             '#73c0de', '#3ba272', '#fc8452', '#9a60b4',
                             '#ea7ccc', '#fff018', '#6800ff', '#4bb0ff',
                             '#1bff00', '#09ffdb']
        # 保存标签颜色字典 
        self.label_colors = {}
        self.init_label_colors()

        # 保存传感器字典
        self.sensor_dict = root_cfg['sensor_dict']
        self.all_sensor = list(self.sensor_dict.keys())

        # 初始化主布局、顶部布局和底部布局
        self.initLayout()

        # 创建一个QTimer对象
        self.computeTimer = QTimer()

        # 创建一个空的DataFrame
        self.data = pd.DataFrame()
        # 配置
        self.cfg = cfg

        self.db_path = os.path.join(self.cfg["project_path"], get_db_folder(), "database.db")
        self.video_path = os.path.join(self.cfg["project_path"], "videos")

        # 模型参数
        self.model_path_list = None
        # 初始化模型路径列表
        self.model_path = []
        # 初始化模型名称
        self.model_name = ''
        # 初始化数据长度
        self.data_length = 180
        # 初始化列名列表
        self.column_names = []

        # 手动校准视频时间
        self.offset = 0.0

        # 初始化最小时间
        self.min_time = 0

        # 初始化视频路径
        self.cap = None

        # 状态
        # 初始化训练状态为False
        self.isTraining = False
        # 初始化模式为空字符串
        self.mode = ''

        # 创建右上模型数据选择区域
        self.createModelSelectLabelArea()

        # 创建右中按钮
        self.createSettingArea()

        # 创建左中按钮区域
        self.createLeftBotton()

        # 创建左上视频区域
        self.createVideoArea()

        # 初始化定时器，保存到csv
        self.init_timer()

        # 更新按钮状态
        self.updateBtn()

        self.model_watcher = QFileSystemWatcher()
        self.data_watcher = QFileSystemWatcher()

        self.model_watcher.directoryChanged.connect(self.update_model_combobox)
        self.data_watcher.directoryChanged.connect(self.update_data_combobox)

        self.update_model_combobox()
        self.update_data_combobox()

    def init_label_colors(self):
        self.label_colors = {}
        for i, label in enumerate(self.label_dict.keys()):
            # 使用取余运算符来循环使用颜色
            color_index = i % len(self.colorPalette)
            self.label_colors[label] = self.colorPalette[color_index]

        # self.logger.debug(
        #     "Attempting..."
        # )

        self.timer.start(600000)


    '''
    ==================================================
    左区域折线图
    ==================================================
    '''
    # 在外层定义




    '''
    ==================================================
    右上区域复选框: 列表
    - self.checkboxList 列表(QCheckBox)
    ==================================================
    '''

    # 创建右上角的选择模型和数据复选框
    def createModelSelectLabelArea(self):

        # 日历按钮
        self.calendar_btn = QPushButton('Calendar')
        self.calendar_btn.setStyleSheet(self.button_style)
        self.calendar_btn.clicked.connect(self.open_calendar)
        self.first_row_layout.addWidget(self.calendar_btn, alignment=Qt.AlignLeft)
        
        # 第一行布局,包含Select model标签和选择框，, alignment=Qt.AlignLeft
        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        # self.first_row1_layout = QHBoxLayout()
        # self.first_row1_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setStyleSheet(self.button_style)
        self.refresh_btn.clicked.connect(self.handleRefresh)

        # self.first_row1_layout.addWidget(self.refresh_btn, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # 第二行布局
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # self.first_row_layout.addLayout(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)

        featureExtractBtn = self.createFeatureExtractButton()
        # self.first_row_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.second_row1_layout = QHBoxLayout()
        # self.first_row_layout.addLayout(self.second_row1_layout)
        # self.second_row1_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # check_box布局
        self.checkbox_layout = QHBoxLayout()

        # 颜色展示
        self.color_layout = QHBoxLayout()

        self.second_row_layout.addLayout(self.checkbox_layout)
        self.second_row_layout.addLayout(self.color_layout)


        # TODO: 将一部分功能改到日历中
        # 第三行布局 Display data 按钮
        # self.third_row1_layout = QHBoxLayout()
        # featureExtractBtn = self.createFeatureExtractButton()
        # self.labelColorBtn = self.createToggleLabelColor()  # 单击可以让右下散点图显示已有标签

        # self.third_row1_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.third_row1_layout.addWidget(labelColorBtn, alignment=Qt.AlignRight)

        # self.nestend_layout.addLayout(self.first_row1_layout)
        # self.nestend_layout.addLayout(self.second_row1_layout)
        # self.nestend_layout.addLayout(self.checkbox_layout)
        # self.nestend_layout.addLayout(self.color_layout)
        # self.nestend_layout.addLayout(self.third_row1_layout)

        self.renderColumnList()
        # self.clear_color_layout()
        self.display_colors(self.label_colors)
    
    def handleRefresh(self):
        self.update_model_combobox()
        self.update_data_combobox()

    def open_calendar(self):
        self.calendar = DateTimeSelector(self)
        self.calendar.show()


    # TODO 全部的closeEvent都没生效，需要找这个项目的closeEvent方法
    def closeEvent(self, event):
        if self.calendar:
            self.calendar.close()
        super().closeEvent(event)


    def display_colors(self, colors):
        # 创建水平布局并添加标签和颜色框
        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()

    def clear_color_layout(self):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
    
    def clear_color_layout_and_display(self, colors):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
                # 创建水平布局并添加标签和颜色框

        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()



    '''
    ==================================================
    右中区域复选框: 列表
    ==================================================
    '''

    def createSettingArea(self):
        self.charts_show_button_layout.addStretch()
        self.labelColorBtn = self.createToggleLabelColor()
        self.charts_show_button_layout.addWidget(self.labelColorBtn)
        # 第一行生成选框按钮
        # self.first_row2_layout = QHBoxLayout()
        addRegionBtn = QPushButton('Generate area')
        # 设置按钮样式
        addRegionBtn.setStyleSheet(self.button_style)
        # 设置按钮最小宽度
        # addRegionBtn.setFixedWidth(160)
        addRegionBtn.clicked.connect(self.handleAddRegion)
        self.charts_show_button_layout.addWidget(addRegionBtn)
        # self.first_row2_layout.addWidget(addRegionBtn, alignment=Qt.AlignLeft)
        # self.first_row2_layout.addStretch()

        # 第二行Threshold输入框
        # self.second_row2_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Input threshold")
        self.charts_show_button_layout.addWidget(self.input_box)

        # self.second_row2_layout.addWidget(QLabel("Threshold:"))
        # self.second_row2_layout.addWidget(self.input_box)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # 第三行两个按钮
        # self.third_row2_layout = QHBoxLayout()
        toLabelBtn = QPushButton('Find data')  # Save to label
        # 设置按钮样式
        toLabelBtn.setStyleSheet(self.button_style)
        toLabelBtn.clicked.connect(self.handleToLabel)

        clearEmptyRegionBtn = QPushButton('Clear data')
        # 设置按钮样式
        clearEmptyRegionBtn.setStyleSheet(self.button_style)
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)

        self.charts_show_button_layout.addWidget(toLabelBtn)
        self.charts_show_button_layout.addWidget(clearEmptyRegionBtn)
        self.charts_show_button_layout.addStretch()


        # # 添加一个伸缩因子来创建间距
        # # self.third_row2_layout.addStretch(1)
        # self.third_row2_layout.addWidget(toLabelBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(1)  # Increase the stretch factor to create more space
        # self.third_row2_layout.addWidget(clearEmptyRegionBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(10)

        # self.row2_layout.addLayout(self.first_row2_layout)
        # self.row2_layout.addLayout(self.second_row2_layout)
        # self.row2_layout.addLayout(self.third_row2_layout)

    '''
    ==================================================
    顶部区域复选框: 列表
    - self.checkboxList 列表(QCheckBox)
    ==================================================
    '''

    # 创建顶部区域的方法
    def createTopArea(self):
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # 将标签添加到顶部布局
        self.top_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # 将组合框添加到顶部布局
        self.top_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)

        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        # 将标签添加到顶部布局
        self.top_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        # 将组合框添加到顶部布局
        self.top_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)

        # 创建特征提取按钮
        featureExtractBtn = self.createFeatureExtractButton()
        # 将按钮添加到顶部布局
        self.top_layout.addWidget(featureExtractBtn, alignment=Qt.AlignLeft)

        featureColor = self.createToggleLabelColor()
        # 将按钮添加到顶部布局
        self.top_layout.addWidget(featureColor, alignment=Qt.AlignLeft)

        # createToggleLabelColor

        # 添加一个伸缩项以填充其余空间并保持左对齐
        self.top_layout.addStretch()

    # # 从row_data的csv创建原始数据组合框的方法
    # def createRawDataComboBox(self):
    #     # 创建标签
    #     RawDataComboBoxLabel = QLabel('Select data:')

    #     # 创建组合框
    #     RawDatacomboBox = QComboBox()
    #     # 获取原始数据文件夹路径
    #     raw_data_path = get_raw_data_folder()
    #     # 获取所有.csv文件路径
    #     rawdata_file_path_list = list(
    #         Path(os.path.join(self.cfg["project_path"], raw_data_path)).glob('*.csv'),
    #     )
    #     # 遍历路径列表
    #     for path in rawdata_file_path_list:
    #         # 将文件名添加到组合框
    #         RawDatacomboBox.addItem(str(path.name))
    #     # 保存组合框
    #     self.RawDatacomboBox = RawDatacomboBox

    #     # combbox change组合框改变时的处理
    #     # 打开第一个.csv文件
    #     self.get_data_from_csv(rawdata_file_path_list[0].name)
    #     self.RawDatacomboBox.currentTextChanged.connect(
    #         # 连接组合框文本改变事件到get_data_from_csv方法
    #         self.get_data_from_csv
    #     )
    #     # 返回标签和组合框
    #     return RawDataComboBoxLabel, RawDatacomboBox

    # 创建原始数据组合框的方法
    def createRawDataComboBox(self):
        # find data at here:C:\Users\dell\Desktop\aa-bbb-2024-04-28\unsupervised-datasets\allDataSet
        # 创建标签
        RawDataComboBoxLabel = QLabel('Select data:')

        # 创建组合框
        RawDatacomboBox = QComboBox()
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 获取所有.pkl文件路径
        rawdata_file_path_list = list(
            Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        )
        # 遍历路径列表
        for path in rawdata_file_path_list:
            # 将文件名添加到组合框
            RawDatacomboBox.addItem(str(path.name))
        # 保存组合框
        self.RawDatacomboBox = RawDatacomboBox

        # combbox change组合框改变时的处理
        # 改变不再打开pkl，在点击dataplay再加载数据
        # 打开第一个.pkl文件
        # self.get_data_from_pkl(rawdata_file_path_list[0].name)
        # self.RawDatacomboBox.currentTextChanged.connect(
        #     # 连接组合框文本改变事件到get_data_from_pkl方法
        #     self.get_data_from_pkl
        # )
        # 返回标签和组合框
        return RawDataComboBoxLabel, RawDatacomboBox

    def get_data_from_csv(self, filename):
        raw_data_path = get_unsupervised_set_folder()
        datapath = os.path.join(self.cfg["project_path"], raw_data_path, filename)
        with open(datapath, 'rb') as f:
            self.data = pickle.load(f)
        # self.data = pd.read_csv(datapath, low_memory=False)
        #
        # # 将 timestamp 列转换为 datetime 对象
        # self.data['datetime'] = pd.to_datetime(self.data['timestamp'])
        #
        # # 生成 unixtime 列（秒级时间戳）
        # self.data['unixtime'] = self.data['datetime'].astype('int64') // 10 ** 9

        # 添加时间戳列
        self.data['_timestamp'] = pd.to_datetime(self.data['datetime']).apply(lambda x: x.timestamp())

        # 复制经纬度并进行线性插值
        self.data['_latitude'] = self.data['latitude'].interpolate()
        self.data['_longitude'] = self.data['longitude'].interpolate()

        # 保留经纬度非空值
        # self.data = self.data.dropna(subset=['acc_x', 'acc_y', 'acc_z'])
        self.data['index'] = self.data.index  # Add an index column
        self.dataChanged.emit(self.data)
        return

    def update_model_combobox(self):
        if self.modelComboBox is None:
            return
        # 清空原有的选项
        self.modelComboBox.clear()

        # 获取根对象配置
        config = self.root.config
        # 读取配置
        cfg = read_config(config)
        # 获取无监督模型文件夹路径
        unsup_model_path = get_unsup_model_folder(cfg)

        full_path = os.path.join(self.cfg["project_path"], unsup_model_path)
        model_path_list = grab_files_in_folder_deep(full_path, ext='*.pth')

        # # 获取所有.pth文件路径
        # model_path_list = grab_files_in_folder_deep(
        #     os.path.join(self.cfg["project_path"], unsup_model_path),
        #     ext='*.pth')
        # 保存模型路径列表
        self.model_path_list = model_path_list
        if model_path_list:
            # 遍历路径列表
            for path in model_path_list:
                self.modelComboBox.addItem(str(Path(path).name))

        # 更新监控的目录
        self.model_watcher.removePaths(self.model_watcher.directories())
        self.model_watcher.addPath(full_path)
    

    def update_data_combobox(self):
        if self.RawDatacomboBox is None:
            return
        # 清空原有的选项
        self.RawDatacomboBox.clear()

        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()

        full_path = os.path.join(self.cfg["project_path"], unsup_data_path)
        rawdata_file_path_list = list(Path(full_path).glob('*.pkl'))

        # 获取所有.pkl文件路径
        # rawdata_file_path_list = list(
        #     Path(os.path.join(self.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        # )
        # 遍历路径列表
        for path in rawdata_file_path_list:
            self.RawDatacomboBox.addItem(str(Path(path).name))

         # 更新监控的目录
        self.data_watcher.removePaths(self.data_watcher.directories())
        self.data_watcher.addPath(full_path)

    # 创建模型组合框的方法
    def createModelComboBox(self):
        # 创建标签
        modelComboBoxLabel = QLabel('Select model:')

        # 创建组合框
        modelComboBox = QComboBox()
        # 从deepview.utils导入辅助函数
        # from deepview.utils import auxiliaryfunctions
        # Read file path for pose_config file. >> pass it on
        # 获取根对象配置
        config = self.root.config
        # 读取配置
        cfg = read_config(config)
        # 获取无监督模型文件夹路径
        unsup_model_path = get_unsup_model_folder(cfg)

        # 获取所有.pth文件路径
        model_path_list = grab_files_in_folder_deep(
            os.path.join(self.cfg["project_path"], unsup_model_path),
            ext='*.pth')
        # 保存模型路径列表
        self.model_path_list = model_path_list
        if model_path_list:
            # 遍历路径列表
            for path in model_path_list:
                # 将文件名添加到组合框
                modelComboBox.addItem(str(Path(path).name))
            # modelComboBox.currentIndexChanged.connect(self.handleModelComboBoxChange)

            self.modelComboBox = modelComboBox

            # if selection changed, run this code
            # 如果选择改变，运行这段代码
            model_name, data_length, column_names = \
                get_param_from_path(modelComboBox.currentText())  # 从路径获取模型参数
            # 保存模型路径
            self.model_path = modelComboBox.currentText()
            # 保存模型名称
            self.model_name = model_name
            # 保存数据长度
            self.data_length = data_length
            # 保存列名列表
            self.column_names = column_names
        modelComboBox.currentTextChanged.connect(
            # 连接组合框文本改变事件到get_model_param_from_path方法
            self.get_model_param_from_path
        )
        # 返回标签和组合框
        return modelComboBoxLabel, modelComboBox

    # 从路径获取模型参数的方法
    def get_model_param_from_path(self, model_path):
        # set model information according to model name
        # 根据模型名称设置模型信息
        if model_path:
            model_name, data_length, column_names = \
                get_param_from_path(model_path)
            # 保存模型路径
            self.model_path = model_path
            # 保存模型名称
            self.model_name = model_name
            # 保存数据长度
            self.data_length = data_length
            # 保存列名列表
            self.column_names = column_names
        return

    # 创建特征提取按钮的方法
    def createFeatureExtractButton(self):
        # 创建按钮
        featureExtractBtn = QPushButton('Data display')
        # 设置按钮样式
        featureExtractBtn.setStyleSheet(self.button_style)
        # 保存按钮
        self.featureExtractBtn = featureExtractBtn
        # 设置按钮宽度
        featureExtractBtn.setFixedWidth(160)
        # 设置按钮不可用
        featureExtractBtn.setEnabled(False)
        # 连接按钮点击事件到handleCompute方法
        # featureExtractBtn.clicked.connect(self.handleCompute)
        featureExtractBtn.clicked.connect(self.start_handle_compute)
        # 返回按钮
        return featureExtractBtn

    def createToggleLabelColor(self):
        # 创建按钮
        featureExtractBtn = QPushButton('Data Coloring')
        # 设置按钮样式
        featureExtractBtn.setStyleSheet(self.button_style)
        self.is_toggled = True
        # 设置按钮宽度
        featureExtractBtn.setFixedWidth(160)
        # 设置按钮不可用
        # featureExtractBtn.setEnabled(False)
        # 连接按钮点击事件到handleCompute方法
        featureExtractBtn.clicked.connect(self.toggleLabelColor)
        return featureExtractBtn

    # 处理计算的方法
    def handleCompute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        # 获取combobox的内容
        self.data, self.dataChanged = get_data_from_pkl(self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged)

        # 延时100毫秒调用handleComputeAsyn方法
        self.computeTimer.singleShot(100, self.handleComputeAsyn)

    # 异步处理计算的方法
    def handleComputeAsyn(self):
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

        # 初始化图表之后不用添加spacer了
        # self.right_layout.removeItem(self.spacer)

        # 渲染右侧图表（特征提取功能）
        self.renderRightPlot()  # feature extraction function here

        # 设置训练状态为False
        self.isTraining = False

        self.updateBtn()


    def handel_calendar_data(self, data):
        self.data = data
        self.dataChanged.emit(self.data)
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)
        self.start_handle_compute()

    def start_handle_compute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        self.handle_compute_thread = QThread()
        self.handle_compute_worker = HandleComputeWorker(self.root, self.data, self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged, self.sensor_dict, self.column_names, self.data_length, self.model_path, self.model_name)
        self.handle_compute_worker.moveToThread(self.handle_compute_thread)
        self.handle_compute_thread.started.connect(self.handle_compute_worker.run)
        self.handle_compute_worker.finished.connect(self.handle_compute_finished)
        self.handle_compute_worker.stopped.connect(self.on_training_stopped)
        self.handle_compute_worker.dataChangedSignal.connect(self.compute_data_changed)
        self.handle_compute_worker.finished.connect(self.handle_compute_thread.quit)
        self.handle_compute_worker.finished.connect(self.handle_compute_worker.deleteLater)
        self.handle_compute_thread.finished.connect(self.handle_compute_thread.deleteLater)
        self.handle_compute_thread.start()

    def stop_training(self):
        self.handle_compute_worker.stop()


    def on_training_stopped(self):
        # TODO 绑定停止训练的方法
        # self.stop_button.setEnabled(False)
        self.isTraining = False
        self.updateBtn()
        print("Training was stopped.")

    def compute_data_changed(self, data):
        self.data = data
        self.min_time = self.data['unixtime'].min()
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

    def handle_compute_finished(self, data):
        (spots, start_indice, end_indice) = data
        # 设置训练状态为False
        self.isTraining = False
        self.updateBtn()
        # 清除中央绘图区域
        self.viewC.clear()
        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem
        scatterItem.sigClicked.connect(self.handleScatterItemClick)
        self.viewC.addItem(scatterItem)
        return

    def handleScatterItemClick(self, scatterItem, points):
        if len(points) >= 1:
            index, start, end = points[0].data()

            # start为latent space的切片索引，index为原始索引（对应切片的开始索引）
            lat, lon = self.data.loc[start, 'latitude'], self.data.loc[start, 'longitude']

            if pd.isna(lat) or pd.isna(lon):
                print("Latitude or longitude is missing.")
                return

            # 点击散点图高亮地图散点
            self.backend_map.triggeLineMapHighlightDotByIndex(start, lat, lon)
            # 点击散点图高亮折线图散点
            self.backend.triggeLineChartHighlightDotByIndex(start)
            # 点击散点图高亮自己
            self.handle_highlight_scatter_dot_by_index(index, True)

        return

    # 更新按钮状态的方法
    def updateBtn(self):
        # enabled 启用按钮
        if self.isTraining:
            # 如果在训练，设置按钮不可用
            self.featureExtractBtn.setEnabled(False)
        else:
            # 如果不在训练，设置按钮可用
            self.featureExtractBtn.setEnabled(True)

    # 渲染列列表的方法
    def renderColumnList(self):
        # 清空 layout
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 初始化复选框列表
        self.checkboxList = []
        # 遍历列名列表
        for column in self.all_sensor:
            # 创建复选框
            cb = QCheckBox(column)
            # 设置复选框为选中状态，如果在列名列表中
            cb.setChecked(column in self.column_names)
            # 将复选框添加到布局中
            self.checkbox_layout.addWidget(cb)
            # 将复选框添加到列表中
            self.checkboxList.append(cb)
            # 连接复选框状态改变事件到handleCheckBoxStateChange方法
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # 添加一个伸缩项以填充剩余区域并保持复选框左对齐
        self.checkbox_layout.addStretch()

    # 处理复选框状态改变的方法
    def handleCheckBoxStateChange(self):
        # 创建新选择列列表
        newSelectColumn = []
        # 遍历列名列表
        for i, column in enumerate(self.all_sensor):
            # 如果复选框被选中
            if self.checkboxList[i].isChecked():
                # 添加列到新选择列列表
                newSelectColumn.append(column)
        # 打印选择列
        # self.selectColumn = newSelectColumn
        print('selectColumn: %s' % (newSelectColumn))
        # self.current_select_sensor_column = newSelectColumn

        metadata = find_charts_data_columns(self.sensor_dict, newSelectColumn)

        # 更新左下图表
        self.backend.handleComboxSelection(metadata)

    '''
    ==================================================
    bottom left area: plot左下区域: 图表
    - self.viewL GraphicsLayoutWidget
    - self.leftPlotList list(PlotItem)
    ==================================================
    '''

    # 创建左侧图表的方法
    def createLeftPlot(self):  # 创建左侧图表的方法
        viewL = QVBoxLayout()  # 创建一个垂直布局
        self.viewL = viewL  # 保存布局
        self.splitter = QSplitter(Qt.Vertical)  # 创建一个垂直分割器
        self.plot_widgets = [None] * len(self.column_names)  # 初始化图表小部件列表
        self.click_begin = True  # 初始化点击开始状态为True
        self.start_line = [None] * len(self.column_names)  # 初始化开始线列表
        self.end_line = [None] * len(self.column_names)  # 初始化结束线列表
        self.regions = []  # 初始化区域列表
        for _ in range(len(self.column_names)):  # 遍历列名列表
            self.regions.append([])  # 为每个列创建一个空的区域列表
        self.bottom_layout.addLayout(viewL, 2)  # 将布局添加到底部布局中

        self.resetLeftPlot()  # 重置左侧图表
        self.splitter.setSizes([100] * len(self.column_names))  # 设置分割器大小
        self.viewL.addWidget(self.splitter)  # 将分割器添加到布局中

    def resetLeftPlot(self):  # 重置左侧图表的方法
        # 重置小部件
        for i in range(len(self.column_names)):  # 遍历列名列表
            if self.plot_widgets[i] is not None:  # 如果图表小部件不为None
                self.splitter.removeWidget(self.plot_widgets[i])  # 从分割器中移除小部件
                self.plot_widgets[i].close()  # 关闭小部件
                self.plot_widgets[i] = None  # 设置小部件为None
        # 添加小部件
        for i, columns in enumerate(self.column_names):  # 遍历列名列表
            real_columns = self.sensor_dict[columns]  # 获取真实列名列表
            plot = pg.PlotWidget(title=columns, name=columns, axisItems={'bottom': pg.DateAxisItem()})  # 创建图表小部件
            for j, c in enumerate(real_columns):  # 遍历真实列名列表
                plot.plot(self.data['_timestamp'], self.data[c], pen=pg.mkPen(j))  # 绘制数据
            # plot.plot(self.data['datetime'], self.data[columns[0]], pen=pg.mkPen(i))
            plot.scene().sigMouseClicked.connect(self.mouse_clicked)  # 连接鼠标点击事件到mouse_clicked方法
            plot.scene().sigMouseMoved.connect(self.mouse_moved)  # 连接鼠标移动事件到mouse_moved方法
            self.plot_widgets[i] = plot  # 保存图表小部件
            self.splitter.addWidget(plot)  # 将图表小部件添加到分割器中

    def updateLeftPlotList(self):
        # 遍历每一列的名称
        for i, column in enumerate(self.column_names):
            # 显示每个绘图窗口
            self.plot_widgets[i].show()

    # def _to_idx(self, start_ts, end_ts):
    #     # 根据给定的时间戳范围筛选数据，并获取对应的索引
    #     selected_indices = self.data[(self.data['_timestamp'] >= start_ts)
    #                                  & (self.data['_timestamp'] <= end_ts)].index
    #     # 返回起始和结束索引
    #     return selected_indices.values[0], selected_indices.values[-1]

    # def _to_time(self, start_idx, end_idx):
    #     # 根据给定的索引范围获取起始和结束时间戳
    #     start_ts = self.data.loc[start_idx, '_timestamp']
    #     end_ts = self.data.loc[end_idx, '_timestamp']
    #     # 返回起始和结束时间戳
    #     return start_ts, end_ts

    # _timestamp于unixtime相同，改用unixtime
    def _to_idx(self, start_ts, end_ts):
        # 根据给定的时间戳范围筛选数据，并获取对应的索引
        selected_indices = self.data[(self.data['unixtime'] >= start_ts)
                                     & (self.data['unixtime'] <= end_ts)].index
        # 返回起始和结束索引
        return selected_indices.values[0], selected_indices.values[-1]

    def _to_time(self, start_idx, end_idx):
        # 根据给定的索引范围获取起始和结束时间戳
        start_ts = self.data.loc[start_idx, 'unixtime']
        end_ts = self.data.loc[end_idx, 'unixtime']
        # 返回起始和结束时间戳
        return start_ts, end_ts

    def _add_region(self, pos):
        if self.click_begin:
            # 如果是第一次点击，记录开始位置
            self.click_begin = False
            for i, plot in enumerate(self.plot_widgets):
                # 创建并添加起始和结束的垂直线
                self.start_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                self.end_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                plot.addItem(self.start_line[i])
                plot.addItem(self.end_line[i])
        else:
            # 如果是第二次点击，记录结束位置并创建区域
            self.click_begin = True
            for i, plot in enumerate(self.plot_widgets):
                # 移除起始和结束的垂直线
                plot.removeItem(self.start_line[i])
                plot.removeItem(self.end_line[i])

                # 创建一个线性区域并添加到绘图窗口
                region = pg.LinearRegionItem([self.start_line[i].value(), self.end_line[i].value()],
                                             brush=(0, 0, 255, 100))
                region.sigRegionChanged.connect(self._region_changed)

                self.start_line[i] = None
                self.end_line[i] = None

                plot.addItem(region)
                self.regions[i].append(region)
                # 获取选中的索引范围
                start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
                print(f'Selected range: from index {start_idx} to index {end_idx}')

    def _region_changed(self, region):
        idx = 0
        # 找到当前改变的区域索引
        for reg_lst in self.regions:
            for i, reg in enumerate(reg_lst):
                if reg == region:
                    idx = i
                    break
        # 同步更新所有绘图窗口中的相应区域
        for reg_lst in self.regions:
            reg_lst[idx].setRegion(region.getRegion())

    def _del_region(self, pos):
        # 删除点击位置对应的区域
        for i, pwidget in enumerate(self.plot_widgets):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    pwidget.removeItem(reg)
                    self.regions[i].remove(reg)

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    print(f'Delete region({start_idx}, {int(end_idx)})')
                    break

    def _edit_region(self, pos):
        set_val = None
        # 编辑点击位置对应的区域
        for i, _ in enumerate(self.regions):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    if set_val is None:
                        # 弹出对话框选择标签
                        dialog = LabelOption(self.label_dict)
                        if dialog.exec() == QDialog.Accepted:
                            set_val = dialog.confirm_selection()
                        else:
                            set_val = None

                    # 设置区域的颜色和标签
                    reg.setBrush(self.checkColor(set_val))
                    reg.label = set_val

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    print(f'Edit region({start_idx}, {end_idx}) label: {set_val}')

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'scatterItem'):
            pos = self.plot_widgets[0].plotItem.vb.mapToView(event.pos())
            # print(f'Clicked at {event.pos()} mapSceneToView {pos.x()},{pos.y()} mapToView {pos2.x()},{pos2.y()}')

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            # 动态更新结束线的位置
            for line in self.end_line:
                line.setPos(pos.x())

    '''
    ==================================================
    bottom center area: result plot底部中心区域:结果图
    - self.viewC PlotWidget
    - self.selectRect QRect
    - self.lastChangePoint list(SpotItem)
    - self.lastMarkList list(LinearRegionItem)
    ==================================================
    '''

    def createCenterPlot(self):
        # 创建一个用于显示中央绘图区域的PlotWidget
        viewC = pg.PlotWidget()
        self.viewC = viewC
        # 禁用右键菜单
        self.viewC.setMenuEnabled(False)
        self.viewC.setBackground('w')
        # 将该PlotWidget添加到底部布局中
        self.row3_layout.addWidget(viewC, 2)

    def checkColor(self, label, first=False):
        if first:
            # 如果是第一次调用，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        if label not in list(self.label_dict.keys()):
            # # 如果标签不在标签字典中，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        # 定义一组颜色
        list_color = [pg.mkBrush(0, 0, 255, 120),
                      pg.mkBrush(255, 0, 0, 120),
                      pg.mkBrush(0, 255, 0, 120),
                      pg.mkBrush(255, 255, 255, 120),
                      pg.mkBrush(255, 0, 255, 120),
                      pg.mkBrush(0, 255, 255, 120),
                      pg.mkBrush(255, 255, 0, 120),
                      pg.mkBrush(5, 5, 5, 120)]
        count = 0
        for lstr, _ in self.label_dict.items():
            if label == lstr:
                # 根据标签返回相应的颜色
                return list_color[count % len(list_color)]
            count += 1

    # 更新右侧散点图的颜色
    def updateRightPlotColor(self):
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # 如果first=False，使用已有的标签
            # 如果first=True，使用手动标签
            color = self.checkColor(self.data.loc[start, 'label'], first=True)
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)

        # 更新散点数据
        for reg in self.regions[0]:
            # 获取区域的起始和结束索引
            idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

    def renderRightPlot(self):
        # 清除中央绘图区域
        self.viewC.clear()

        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))

        new_column_names = find_data_columns(self.sensor_dict, self.column_names)

        # 将数据切割成片段以获取潜在特征和索引
        start_indice, end_indice, pos = featureExtraction(self.root,
                                                          self.data,
                                                          self.data_length,
                                                          new_column_names,
                                                          self.model_path,
                                                          self.model_name)

        # 保存数据到scatterItem的属性中
        n = len(start_indice)
        spots = [{'pos': pos[i, :],
                  'data': (i, start_indice[i], end_indice[i]),
                  'brush': self.checkColor(self.data.loc[i * self.data_length, 'label'], first=True)}
                 for i in range(n)]

        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)
        return

    def find_i_by_indice(self, indice, start_indice, end_indice):
        '''
        map和sensor data都用原始索引，latent space用切片索引，这里从原始索引查找切片
        indice为原始索引
        start indice为切片索引
        '''
        for i in range(len(start_indice)):
            if start_indice[i] <= indice < end_indice[i]:
                return i
        return None  # 如果没有找到合适的范围

    # TODO 点击过快可能报错    self._plot.updateSpots(self._data.reshape(1)) AttributeError: 'NoneType' object has no attribute 'updateSpots'
    def handle_highlight_scatter_dot_by_index(self, index, useRawIndex = False):
        self.jump_to_timestamp(index)
        indice = index
        if not useRawIndex:
            indice = self.find_i_by_indice(index, self.start_indice, self.end_indice)
        if self.last_modified_points:
            for p, original_size, original_brush in self.last_modified_points:
                p.setSize(original_size)
                p.setBrush(original_brush)
            
        self.last_modified_points = []  # Clear the list
        if indice is not None:
            for spot in self.scatterItem.points():
                i, start, end = spot.data()
                if i == indice:
                    # Save current properties
                    original_size = spot.size()
                    original_brush = spot.brush()
                    self.last_modified_points.append((spot, original_size, original_brush))
                    spot.setSize(15)
                    spot.setBrush(pg.mkBrush(255, 0, 0, 255))

        

    # 显示原始标签在右下散点图上
    def toggleLabelColor(self):

        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # spots.append({
            # 'pos': (pos.x(), pos.y()),
            # 'data': (i, start, end),
            # 'brush': pg.mkBrush(color)
            # })
            if self.data.loc[start, 'label_flag'] == 0:
                original_brush = spot.brush()  # 读取原有颜色
                color = original_brush.color()  # 默认使用原有颜色
                # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            else:
                if self.is_toggled:
                    color = self.checkColor(self.data.loc[start, 'label'], first=False)
                else:
                    original_brush = spot.brush()  # 读取原有颜色
                    color = original_brush.color()  # 默认使用原有颜色
                    # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)
        # Toggle the flag
        self.is_toggled = not self.is_toggled

        # # 更新散点数据
        # for reg in self.regions[0]:
        #     # 获取区域的起始和结束索引
        #     idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]),
        #                                       int(reg.getRegion()[1]))
        #     for spot in spots:
        #         if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
        #             spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

        return

    def select_random_continuous_seconds(self, num_samples=100, points_per_second=90):
        # 随机选择连续的秒数数据段
        selected_dfs = []
        start_indice = []
        end_indice = []

        while len(selected_dfs) < num_samples:
            start_idx = np.random.randint(0, len(self.data) - points_per_second)
            end_idx = start_idx + points_per_second - 1
            selected_range = self.data.iloc[start_idx:end_idx + 1]

            if not selected_range[['acc_x', 'acc_y', 'acc_z']].isna().any().any():
                selected_dfs.append(selected_range)  # 从start_idx到end_idx的数据段
                start_indice.append(start_idx)
                end_indice.append(end_idx)

        return start_indice, end_indice, selected_dfs

    '''
    ==================================================
    bottom right area: setting panel
    - self.settingPannel QVBoxLayout
    - self.currentLabel str
    - self.maxColumn int
    - self.maxRow int
    ==================================================
    '''

    # 创建右侧设置面板
    def createRightSettingPannel(self):
        settingPannel = QVBoxLayout()
        self.settingPannel = settingPannel
        self.bottom_layout.addLayout(self.settingPannel)

        self.settingPannel.setAlignment(Qt.AlignTop)

        self.createLabelButton()
        self.createRegionBtn()
        self.createSaveButton()

    # 创建保存按钮
    def createSaveButton(self):
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.handleSaveButton)
        self.settingPannel.addWidget(saveButton)



    # def getSelectedAreaToSave(self, area_data):
    #     # print(areaData)
    #     try:
    #         area_data = json.loads(area_data)  # 解析 JSON 字符串
    #         # print("Parsed data:", areaData)
    #     except json.JSONDecodeError as e:
    #         print("Failed to decode JSON:", e)
    #         return
    #     for reg in area_data:
    #         name = reg[0].get("name")
    #         first_timestamp = reg[0].get("timestamp", {}).get("start")
    #         second_timestamp = reg[0].get("timestamp", {}).get("end")
    #         self.data.loc[(self.data['unixtime'] >= int(first_timestamp)) & (
    #                 self.data['unixtime'] <= int(second_timestamp)), 'label'] = name
    #     self.handleSaveButton()

    def getSelectedAreaToSave(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 0)
        save_task.signals.save_csv_finished.connect(self.on_save_finished)
        self.save_csv_thread_pool.start(save_task)

    def on_save_finished(self, new_path):
        QMessageBox.information(None, "保存CSV", f"文件已保存于 {new_path}", QMessageBox.Ok)
    
    def getSelectedAreaToSaveTimer(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 1)
        self.save_csv_thread_pool.start(save_task)

    # 处理保存按钮点击事件
    def handleSaveButton(self):
        # for reg in self.regions[0]:
        #     if hasattr(reg, 'label') and reg.label:
        #         regionRange = reg.getRegion()
        #         self.data.loc[(self.data['_timestamp'] >= int(regionRange[0])) & (self.data['_timestamp'] <= int(regionRange[1])), 'label'] = reg.label

        os.makedirs(os.path.join(self.cfg["project_path"], "edit-data", ), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText())
        # edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText().replace(".pkl", ".csv"))
        try:  # 如果文件存在就新建
            if os.path.exists(edit_data_path):
                for num in range(1, 100, 1):
                    firstname = edit_data_path.split('Hz')[0]
                    new_path = firstname + '_' + str(num) + '.pkl'
                    if not os.path.exists(new_path):
                        self.data.to_csv(new_path)
                        break
            else:
                new_path = edit_data_path
                self.data.to_csv(edit_data_path)
        except:
            print('save data error!')
        else:
            print(f'文件已经保存在{new_path}')

    # 创建标签按钮
    def createLabelButton(self):
        self.add_mode = QPushButton("Label Add Mode", self)
        self.add_mode.clicked.connect(partial(self._change_mode, "add"))
        self.edit_mode = QPushButton("Label Edit Mode", self)
        self.edit_mode.clicked.connect(partial(self._change_mode, "edit"))
        self.del_mode = QPushButton("Label Delete Mode", self)
        self.del_mode.clicked.connect(partial(self._change_mode, "del"))
        self.refresh = QPushButton("Refresh Spots", self)
        self.refresh.clicked.connect(self.updateRightPlotColor)
        self.settingPannel.addWidget(self.add_mode)
        self.settingPannel.addWidget(self.edit_mode)
        self.settingPannel.addWidget(self.del_mode)
        self.settingPannel.addWidget(self.refresh)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

    # 改变模式
    def _change_mode(self, mode: str):
        print(f'Change mode to "{mode}"')
        self.mode = mode

    # 创建区域按钮
    def createRegionBtn(self):
        addRegionBtn = QPushButton('Add region')
        addRegionBtn.clicked.connect(self.handleAddRegion)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter threshold")

        toLabelBtn = QPushButton('Reflect to Data')  # Save to label
        toLabelBtn.clicked.connect(self.handleToLabel)
        self.settingPannel.addWidget(addRegionBtn)
        self.settingPannel.addWidget(self.input_box)
        self.settingPannel.addWidget(toLabelBtn)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

        # Clear empty region 清除空区域
        clearEmptyRegionBtn = QPushButton('Clear Empty Region')
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)
        self.settingPannel.addWidget(clearEmptyRegionBtn)

    def handleAddRegion(self):
        if hasattr(self, 'rightRegionRoi'):
            return

        rect = self.viewC.viewRect()
        w = rect.width()
        h = rect.height()
        x = rect.x()
        y = rect.y()

        # create ROI
        roi = pg.ROI([x + w * 0.45, y + h * 0.45], [w * 0.1, h * 0.1])
        # 上
        roi.addScaleHandle([0.5, 1], [0.5, 0])
        # 右
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        # 下
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        # 左
        roi.addScaleHandle([0, 0.5], [1, 0.5])
        # 右下
        roi.addScaleHandle([1, 0], [0, 1])

        self.viewC.addItem(roi)

        self.rightRegionRoi = roi

        # roi.sigRegionChanged.connect(self.handleROIChange)
        # roi.sigRegionChangeFinished.connect(self.handleROIChangeFinished)
        # self.handleROIChange(roi)
        # self.handleROIChangeFinished(roi)

    # 处理反射到标签的方法
    def handleToLabel(self):
        if not hasattr(self, 'rightRegionRoi'):  # 如果没有右侧区域ROI，提示用户先添加区域
            print('Add region first.')
            return

        pos: pg.Point = self.rightRegionRoi.pos()
        size: pg.Point = self.rightRegionRoi.size()

        self.rightRegionRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.rightRegionRect)

        # 是否需要合并区间
        rectangles = []
        for p in points:
            index, start, end = p.data()
            startT, endT = self._to_time(start, end)
            rectangles.append((startT, endT))
        # combine rectangles 合并矩形，数据为开始结束时间
        if self.input_box.text() == "":
            combined_rectangles = combine_rectangles(rectangles, float(30))  # set default value
        else:
            combined_rectangles = combine_rectangles(rectangles, float(self.input_box.text()))

        # 传递combined_rectangles到backend
        markData = []
        for startT, endT in combined_rectangles:
            # print(startT, endT)
            start_id, end_id = self._to_idx(startT, endT)
            start_timestamp = self.data.loc[start_id, 'timestamp']
            end_timestamp = self.data.loc[end_id, 'timestamp']

            # 创建markData
            start_Area = {
                'name': 'data',
                'xAxis': start_timestamp,
                'itemStyle': {
                    'color': 'rgba(0, 0, 255, 0.39)'
                }
            }
            end_Area = {
                'xAxis': end_timestamp,
            }
            newArray = [start_Area, end_Area]

            markData.append(newArray)
        # print(markData)
        # 将 markData 转换为 JSON 字符串
        mark_data = json.dumps(markData)
        # 传递markData到backend
        self.backend.setMarkData(mark_data)

    def handleClearEmptyRegion(self):
        # 绑定html的Clear
        self.backend.clearMarkData()


def combine_rectangles(rectangles, threshold_seconds=100):
    if not rectangles:
        return []

    # 将矩形按开始时间排序
    rectangles.sort(key=lambda x: x[0])

    combined_rectangles = []
    current_start, current_end = rectangles[0]

    for start, end in rectangles[1:]:
        # 如果当前时间段与下一个时间段间隔小于阈值
        if (start - current_end) <= threshold_seconds:
            # 合并时间段
            current_end = max(current_end, end)
        else:
            # 否则，将当前时间段加入结果列表，并更新当前时间段
            combined_rectangles.append((current_start, current_end))
            current_start, current_end = start, end

    # 添加最后一个时间段
    combined_rectangles.append((current_start, current_end))

    return combined_rectangles


def find_charts_data_columns(sensor_dict, column_names):
    # new_column_names = []
    metadatas = []
    for column_name in column_names:
        # real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # new_column_names.extend(real_names) # 将实际列名添加到新的列名列表中
        if column_name.upper() == "GPS":
            real_names = ['GPS_velocity', 'GPS_bearing']
        else:
            real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        # 创建元数据信息
        metadata = {
            "name": column_name,
            "xAxisName": "timestamp",
            "yAxisName": "Y Axis 1",
            "series": real_names
        }
        metadatas.append(metadata)
    return metadatas
