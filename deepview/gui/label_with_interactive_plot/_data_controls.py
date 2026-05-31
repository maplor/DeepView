import os
import pickle
from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QComboBox, QPushButton

from deepview.utils.auxiliaryfunctions import (
    get_param_from_path,
    get_unsup_model_folder,
    get_unsupervised_set_folder,
    grab_files_in_folder_deep,
    read_config,
)


class DataControlsMixin:
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