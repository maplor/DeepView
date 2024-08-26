

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


# 创建一个蓝色的pg.mkPen对象，宽度为2
clickedPen = pg.mkPen('b', width=2)


# 定义LabelOption类，继承自QDialog
class LabelOption(QDialog):
    def __init__(self, label_dict):
        super().__init__()

        # 创建一个垂直布局
        layout = QVBoxLayout()
        # 保存标签字典
        self.label_dict = label_dict
        # 创建一个空字典来保存单选按钮
        self.radio_buttons = {}
        # 遍历标签字典
        for label, lid in label_dict.items():
            # 为每个标签创建一个单选按钮
            self.radio_buttons[label] = QRadioButton(label)
            # 将单选按钮添加到布局中
            layout.addWidget(self.radio_buttons[label])

        # 创建确认按钮
        self.confirm_button = QPushButton("Confirm")
        # 连接确认按钮的点击事件到confirm_selection方法
        self.confirm_button.clicked.connect(self.confirm_selection)
        # 将确认按钮添加到布局中
        layout.addWidget(self.confirm_button)
        # 设置布局
        self.setLayout(layout)

    # 确认选择的方法
    def confirm_selection(self):
        # 遍历标签字典
        for label, lid in self.label_dict.items():
            # 如果单选按钮被选中
            if self.radio_buttons[label].isChecked():
                # 设置选中的选项为当前标签
                selected_option = label
                # 接受对话框，关闭对话框
                self.accept()
                # 返回选中的选项
                return selected_option
            else:
                # 如果没有选中任何选项，设置为None
                selected_option = None
        # 接受对话框，关闭对话框
        self.accept()
        # 返回选中的选项
        return selected_option


# 创建一个函数来找到最近的有效索引
def find_nearest_index(target_index, valid_indices):
    if len(valid_indices) == 0:
        return None
    nearest_index = valid_indices[np.abs(valid_indices - target_index).argmin()]
    return nearest_index


class Backend(QObject):
    highlightDotByindex = Signal(int, float, float)
    getSelectedAreaByHtml = Signal(str)
    setStartEndTime = Signal(str, str)
    getSelectedAreaToSaveSign = Signal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.select_option = None

    # 创建一个函数来找到最近的有效索引

    @Slot(pd.DataFrame)
    def handle_data_changed(self, data):
        self.data = data  # Update the data attribute
        # print("Backend's DataFrame has been updated:")
        # print(self.data)
    
    @Slot()
    def handle_label_change(self, option):
        self.select_option = option

    @Slot(result='QString')
    def get_label_option(self):
        result = self.select_option if self.select_option is not None else ""
        # print(f"Returning: {result}")
        return result
    # 通过索引高亮散点，点击地图散点高亮折线图散点
    @Slot()
    def triggeLineChartHighlightDotByIndex(self, index):
        print(f"Triggering highlight dot({index})...")
        self.view.page().runJavaScript(f"highlightLineChartDotByIndex('{index}')")

    # 设置开始结束时间到标签
    @Slot(str, str)
    def setStartEndTimeToLabel(self, start_time, end_time):
        print(f"Setting start time: {start_time}, end time: {end_time}")
        self.setStartEndTime.emit(start_time, end_time)

    # 通过索引高亮散点，点击折线图散点高亮地图散点
    @Slot(int)
    def handleHighlightDotByIndex(self, index):
        lat, lon = self.data.loc[index, 'latitude'], self.data.loc[index, 'longitude']

        # # 获取所有非空纬度和经度的索引
        # valid_lat_indices = self.data.index[~self.data['latitude'].isna()].to_numpy()
        # valid_lon_indices = self.data.index[~self.data['longitude'].isna()].to_numpy()

        # # 如果纬度或经度为空，找到最近的有效值
        # if pd.isna(lat) or pd.isna(lon):

        #     if pd.isna(lat):
        #         nearest_lat_index = find_nearest_index(index, valid_lat_indices)
        #         if nearest_lat_index is not None:
        #             lat = self.data.loc[nearest_lat_index, 'latitude']
        #     if pd.isna(lon):
        #         nearest_lon_index = find_nearest_index(index, valid_lon_indices)
        #         if nearest_lon_index is not None:
        #             lon = self.data.loc[nearest_lon_index, 'longitude']

        if pd.isna(lat) or pd.isna(lon):
            print("Latitude or longitude is missing.")
            return

        print("handleing highlight dot...")
        self.highlightDotByindex.emit(index, lat, lon)

    # add label按钮点击事件
    @Slot()
    def handleAddLabel(self, selected_option):
        print("Adding label...")
        self.view.page().runJavaScript(f"addLabel('{selected_option}')")

    # delete label按钮点击事件
    @Slot(int)
    def handleDeleteLabel(self, status):
        print("Deleting label...")
        self.view.page().runJavaScript(f"deleteLabel('{status}')")

    # 从框选的散点图设置折线图markData
    @Slot(str)
    def setMarkData(self, data):
        print("Setting mark data...")
        self.view.page().runJavaScript(f"setMarkData('{data}')")

    # 清空折线图markData
    @Slot()
    def clearMarkData(self):
        print("Clearing mark data...")
        self.view.page().runJavaScript("clearMarkData()")

    # 从html获取折线图框选区域
    # @Slot(result='QVariant')
    @Slot()
    def getSelectedArea(self):
        print("etSelectedArea..")
        return self.view.page().runJavaScript("getSelectedArea()", 0, self.test_callback)

    def test_callback(self, result):
        self.getSelectedAreaByHtml.emit(result)

    @Slot()
    def getSelectedAreaToSave(self):
        print("getSelectedAreaToSave..")
        return self.view.page().runJavaScript("getSelectedArea()", 0, self.save_callback)

    def save_callback(self, result):
        print(result)
        self.getSelectedAreaToSaveSign.emit(result)

    # 添加label弹出确认提示框
    @Slot(result='QVariant')
    def confirmOverlap(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("存在重复区间，是否替换")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    # 删除标签弹出确认提示框
    @Slot(result='QVariant')
    def confirmDelete(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("是否删除选中的标记区域？")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    @Slot()
    def desplayData(self, data, metadata=None):
        if isinstance(data, pd.DataFrame):
            # 将列名转换为列表
            columns_list = data.columns.tolist()
            logging.debug(f"columns_list: {columns_list}")

            # TODO 根据传入信号选择需要的列
            # 指定要排除的列,Python 的 json 模块不能直接序列化某些自定义对象或非基本数据类型（如 datetime、Timestamp 等
            # columns_to_drop = ['datetime', 'logger_id', 'animal_tag', 'gps_status',
            #                    'activity_class', 'label']
            #
            # # 删除指定列
            # data = data.drop(columns=columns_to_drop, errors='ignore')

            series_combined = ["timestamp", "unixtime", "index"] + [item for data in metadata for item in
                                                                    data["series"]]
            data = data[series_combined]

            # 将空字符串替换为 None
            data = data.replace('', None)

            # 将 NaN 值替换为 None,避免转换为json出错
            data = data.where(pd.notnull(data), None)

            # 将 DataFrame 转换为字典列表
            data_records = data.to_dict(orient='records')

            if metadata is None:
                # 创建元数据信息
                metadata = [
                    {
                        "name": "acceleration",
                        "xAxisName": "timestamp",
                        "yAxisName": "Y Axis 1",
                        "series": ["acc_x", "acc_y", "acc_z"]
                    }
                ]

            # 将元数据和数据打包到一个字典中
            result = {
                "metadata": metadata,
                "data": data_records
            }
        else:
            # 如果数据不是 DataFrame，则直接使用传入的数据
            result = data
        # 将结果转换为 JSON 格式
        json_data = json.dumps(result)
        # print(json_data)
        self.view.page().runJavaScript(f"displayData('{json_data}')")

    # combox选择事件
    @Slot()
    def handleComboxSelection(self, charts_data):
        charts_data = json.dumps(charts_data)
        print("Combox selection...")
        self.view.page().runJavaScript(f"handleComboxChange('{charts_data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")

    @Slot(str)
    def receiveData(self, data):
        print("Received data from frontend:", data)

    @Slot()
    def triggerUpdate(self):
        self.view.page().runJavaScript("getInputValue()")  # 调用前端的getInputValue函数


class BackendMap(QObject):
    highlightLineChartDotByindex = Signal(int)

    def __init__(self):
        super().__init__()

    # 点击折线图高亮地图散点，没有则添加新点
    @Slot(str, float, float)
    def triggeLineMapHighlightDotByIndex(self, index, lat, lon):
        print(f"Triggering highlight dot({index})...")
        # self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}, {lat}, {lon}')")
        self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}', {lat}, {lon})")

    # 点击地图高亮折线图散点
    @Slot(int)
    def handleHighlightLineDotByIndex(self, index):
        print("handleing highlight dot...")
        self.highlightLineChartDotByindex.emit(index)

    # 在data display之后读取gps边界，然后作为初始地图
    @Slot()
    def desplayMapData(self, data):
        if isinstance(data, pd.DataFrame):
            # 将列名转换为列表
            columns_list = data.columns.tolist()
            logging.debug(f"columns_list: {columns_list}")
            # 选择需要的列 index、latitude 和 longitude，并去除 latitude 和 longitude 中的缺失值。
            data = data[['index', 'latitude', 'longitude']].dropna(subset=['latitude', 'longitude'])
            # 使用 iloc 按索引进行降采样, 一万个条目取一个。
            interval = 10*60*25  # 25 is sampling rate, 10 is minutes
            data = data.iloc[::interval]

            data = data.to_dict(orient='records')
        data = json.dumps(data)
        self.view.page().runJavaScript(f"displayMapData('{data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")


# 定义LabelWithInteractivePlot类，继承自QWidget
class LabelWithInteractivePlot(QWidget):
    dataChanged = Signal(pd.DataFrame)

    def __init__(self, root, cfg) -> None:
        super().__init__()
        # 创建一个日志记录器
        self.logger = logging.getLogger("GUI")
        self.backend = Backend()
        self.backend_map = BackendMap()

        # self.data改变时同步数据到backend
        self.dataChanged.connect(self.backend.handle_data_changed)

        # 将折线图backend的点击折线图事件连接到backend_map的高亮地图散点方法
        self.backend.highlightDotByindex.connect(self.backend_map.triggeLineMapHighlightDotByIndex)
        # 点击地图散点高亮折线图散点
        self.backend_map.highlightLineChartDotByindex.connect(self.backend.triggeLineChartHighlightDotByIndex)
        self.backend.getSelectedAreaByHtml.connect(self.handleReflectToLatent)
        self.backend.setStartEndTime.connect(self.setStartEndTime)
        self.backend.getSelectedAreaToSaveSign.connect(self.getSelectedAreaToSave)

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
        # 保存传感器字典
        self.sensor_dict = root_cfg['sensor_dict']

        # 初始化主布局、顶部布局和底部布局
        self.initLayout()

        # 创建一个QTimer对象
        self.computeTimer = QTimer()

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
        self.data_length = 180
        # 初始化列名列表
        self.column_names = []

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

        # 初始化定时器，保存到csv
        self.init_timer()

        # 更新按钮状态
        self.updateBtn()

        # self.logger.debug(
        #     "Attempting..."
        # )

    # 初始化布局的方法
    def initLayout(self):

        # 创建主水平布局
        self.main_layout = QHBoxLayout()

        # 设置主布局
        self.setLayout(self.main_layout)

        # 创建左侧布局
        self.left_layout = QVBoxLayout()

        # 创建左侧row1布局
        self.left_row1_layout = QHBoxLayout()
        self.left_layout.addLayout(self.left_row1_layout)

        # 创建左侧row2布局
        self.left_row2_layout = QVBoxLayout()
        self.left_layout.addLayout(self.left_row2_layout)

        # 创建左侧row3布局
        self.left_row3_layout = QHBoxLayout()
        self.left_layout.addLayout(self.left_row3_layout)

        # 创建右侧布局
        self.right_layout = QVBoxLayout()

        # 创建一个 QVBoxLayout 用于 row1_layout 中的多行布局
        self.nestend_layout = QVBoxLayout()

        # 创建 row1_layout 并将嵌套布局添加到其中
        self.row1_layout = QHBoxLayout()
        self.row1_layout.addLayout(self.nestend_layout)

        self.right_layout.addLayout(self.row1_layout)

        self.row2_layout = QVBoxLayout()
        self.right_layout.addLayout(self.row2_layout)

        self.row3_layout = QHBoxLayout()
        self.right_layout.addLayout(self.row3_layout)

        # # 创建一个弹簧 (QSpacerItem)
        # self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.right_layout.addItem(self.spacer)

        # 将左侧和右侧布局添加到主布局中，并设置相同的 stretch 因子，使它们宽度相同
        left_column = QWidget()
        left_column.setLayout(self.left_layout)
        self.main_layout.addWidget(left_column, 1)

        right_column = QWidget()
        right_column.setLayout(self.right_layout)
        self.main_layout.addWidget(right_column, 1)

        # 创建中心图表
        self.createCenterPlot()


    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.backend.getSelectedAreaToSave)
        # 设置定时器每隔五分钟（300000 毫秒）触发一次
        self.timer.start(600000)


    '''
    ==================================================
    左中按键区域
    ==================================================
    '''

    def createLeftBotton(self):
        # 第一行start time显示框
        self.left_start_time_layout = QHBoxLayout()
        self.start_input_box = QLineEdit(self)
        self.start_input_box.setPlaceholderText("start time")
        self.left_start_time_layout.addWidget(QLabel("Start time:"))
        self.left_start_time_layout.addWidget(self.start_input_box)
        self.left_start_time_layout.addStretch()
        self.left_row2_layout.addLayout(self.left_start_time_layout)

        # 第二行end time显示框
        self.left_end_time_layout = QHBoxLayout()
        self.end_input_box = QLineEdit(self)
        self.end_input_box.setPlaceholderText("end time")
        self.left_end_time_layout.addWidget(QLabel("End time: "))
        self.left_end_time_layout.addWidget(self.end_input_box)
        self.left_end_time_layout.addStretch()
        self.left_row2_layout.addLayout(self.left_end_time_layout)

        # 第三行label选项框
        self.left_label_layout = QHBoxLayout()
        self.label_combobox = QComboBox()
        

        for label in self.label_dict.keys():
            self.label_combobox.addItem(label)
        self.backend.handle_label_change(self.label_combobox.currentText())
        self.label_combobox.currentTextChanged.connect(
            self.backend.handle_label_change
            )
        self.left_label_layout.addWidget(QLabel("Label:     "))
        self.left_label_layout.addWidget(self.label_combobox, alignment=Qt.AlignLeft)

        # 保存csv按钮
        self.save_csv_btn = QPushButton('Save csv')
        self.save_csv_btn.clicked.connect(self.backend.getSelectedAreaToSave)
        self.save_csv_btn.setStyleSheet(self.button_style)
        self.left_label_layout.addWidget(self.save_csv_btn)
        self.left_label_layout.addStretch()
        self.left_row2_layout.addLayout(self.left_label_layout)

        # 第四行三个按钮
        self.left_button_layout = QHBoxLayout()
        # add label按钮
        add_label_btn = QPushButton('Add label')
        add_label_btn.clicked.connect(lambda: self.backend.handleAddLabel(self.label_combobox.currentText()))
        # 设置按钮样式
        add_label_btn.setStyleSheet(self.button_style)

        # delete label按钮
        delete_label_btn = QPushButton('Delete label')
        delete_label_btn.setCheckable(True)  # Make the button checkable
        delete_label_btn.clicked.connect(lambda: self.backend.handleDeleteLabel(int(delete_label_btn.isChecked())))
        # Set the button style based on the checked state
        delete_label_btn.setStyleSheet(
            self.button_style + "background-color: red;" if delete_label_btn.isChecked() else self.button_style + "background-color: green;")

        # Reflect to latent space按钮
        reflect_to_latent_btn = QPushButton('View on latent space')
        reflect_to_latent_btn.clicked.connect(lambda: self.backend.getSelectedArea())
        # 设置按钮样式
        reflect_to_latent_btn.setStyleSheet(self.button_style)

        # 将按钮添加到布局中
        self.left_button_layout.addWidget(add_label_btn)
        self.left_button_layout.addStretch(1)
        self.left_button_layout.addWidget(delete_label_btn)
        self.left_button_layout.addStretch(1)
        self.left_button_layout.addWidget(reflect_to_latent_btn)
        self.left_button_layout.addStretch(10)

        self.left_row2_layout.addLayout(self.left_button_layout)

    def setStartEndTime(self, start_time, end_time):
        self.start_input_box.setText(start_time)
        self.end_input_box.setText(end_time)

    # 定义一个函数将ISO格式的时间字符串转换为Unix时间戳
    def iso_to_timestamp(self, iso_str):
        dt = datetime.datetime.fromisoformat(iso_str.rstrip('Z'))
        timestamp = dt.timestamp()

        # return dt.timestamp()
        return round(timestamp, 5)

    # 更新右侧散点图的颜色
    def handleReflectToLatent(self, areaData):
        # print(areaData)
        try:
            areaData = json.loads(areaData)  # 解析 JSON 字符串
            # print("Parsed data:", areaData)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
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

        for reg in areaData:
            # 获取区域的起始和结束索引
            name = reg[0].get("name")
            first_timestamp = reg[0].get("timestamp", {}).get("start")
            second_timestamp = reg[0].get("timestamp", {}).get("end")
            color = reg[0].get("itemStyle", {}).get("color")

            idx_begin, idx_end = self._to_idx(int(first_timestamp), int(second_timestamp))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = pg.mkBrush(color)

        self.scatterItem.setData(spots=spots)

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
        # 第一行布局,包含Select model标签和选择框，, alignment=Qt.AlignLeft
        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        self.first_row1_layout = QHBoxLayout()
        self.first_row1_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        self.first_row1_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.first_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # 第二行布局
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        self.second_row1_layout = QHBoxLayout()
        self.second_row1_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        self.second_row1_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)
        self.second_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # check_box布局
        self.checkbox_layout = QHBoxLayout()

        # 第三行布局 Display data 按钮
        self.third_row1_layout = QHBoxLayout()
        featureExtractBtn = self.createFeatureExtractButton()
        labelColorBtn = self.createToggleLabelColor()  # 单击可以让右下散点图显示已有标签

        self.third_row1_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        self.third_row1_layout.addWidget(labelColorBtn, alignment=Qt.AlignRight)

        self.nestend_layout.addLayout(self.first_row1_layout)
        self.nestend_layout.addLayout(self.second_row1_layout)
        self.nestend_layout.addLayout(self.checkbox_layout)
        self.nestend_layout.addLayout(self.third_row1_layout)

        self.renderColumnList()

    '''
    ==================================================
    右中区域复选框: 列表
    ==================================================
    '''

    def createSettingArea(self):
        # 第一行生成选框按钮
        self.first_row2_layout = QHBoxLayout()
        addRegionBtn = QPushButton('Generate area')
        # 设置按钮样式
        addRegionBtn.setStyleSheet(self.button_style)
        # 设置按钮最小宽度
        # addRegionBtn.setFixedWidth(160)
        addRegionBtn.clicked.connect(self.handleAddRegion)
        self.first_row2_layout.addWidget(addRegionBtn, alignment=Qt.AlignLeft)
        self.first_row2_layout.addStretch()

        # 第二行Threshold输入框
        self.second_row2_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter threshold")

        self.second_row2_layout.addWidget(QLabel("Threshold:"))
        self.second_row2_layout.addWidget(self.input_box)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # 第三行两个按钮
        self.third_row2_layout = QHBoxLayout()
        toLabelBtn = QPushButton('Find data')  # Save to label
        # 设置按钮样式
        toLabelBtn.setStyleSheet(self.button_style)
        toLabelBtn.clicked.connect(self.handleToLabel)

        clearEmptyRegionBtn = QPushButton('Clear data')
        # 设置按钮样式
        clearEmptyRegionBtn.setStyleSheet(self.button_style)
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)

        # 添加一个伸缩因子来创建间距
        # self.third_row2_layout.addStretch(1)
        self.third_row2_layout.addWidget(toLabelBtn, alignment=Qt.AlignLeft)
        self.third_row2_layout.addStretch(1)  # Increase the stretch factor to create more space
        self.third_row2_layout.addWidget(clearEmptyRegionBtn, alignment=Qt.AlignLeft)
        self.third_row2_layout.addStretch(10)

        self.row2_layout.addLayout(self.first_row2_layout)
        self.row2_layout.addLayout(self.second_row2_layout)
        self.row2_layout.addLayout(self.third_row2_layout)

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
            self.data['timestamp'] = pd.to_datetime(self.data['unixtime'],
                                                    unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ').str[:-4] + 'Z'
            self.data['index'] = self.data.index  # Add an index column
            self.dataChanged.emit(self.data)
        return

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
        featureExtractBtn.clicked.connect(self.handleCompute)
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
        # 获取combobox的内容
        self.get_data_from_pkl(self.RawDatacomboBox.currentText())

        # 延时100毫秒调用handleComputeAsyn方法
        self.computeTimer.singleShot(100, self.handleComputeAsyn)

    # 异步处理计算的方法
    def handleComputeAsyn(self):
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.desplayData(self.data, metadatas)
        self.backend_map.desplayMapData(self.data)

        # 初始化图表之后不用添加spacer了
        # self.right_layout.removeItem(self.spacer)

        # 渲染右侧图表（特征提取功能）
        self.renderRightPlot()  # feature extraction function here

        # 设置训练状态为False
        self.isTraining = False

        self.updateBtn()

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

        # for i in reversed(range(self.checkbox_layout.count())):
        #     # 删除布局中的所有小部件
        #     self.checkbox_layout.itemAt(i).widget().deleteLater()

        # 初始化复选框列表
        self.checkboxList = []
        # 遍历列名列表
        for column in self.column_names:
            # 创建复选框
            cb = QCheckBox(column)
            # 设置复选框为选中状态
            cb.setChecked(True)
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
        for i, column in enumerate(self.column_names):
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

        # 将数据切割成片段以获取潜在特征和索引
        start_indice, end_indice, pos = self.featureExtraction()

        # 保存数据到scatterItem的属性中
        n = len(start_indice)
        spots = [{'pos': pos[i, :],
                  'data': (i, start_indice[i], end_indice[i]),
                  'brush': self.checkColor(self.data.loc[i * self.data_length, 'label'], first=True)}
                 for i in range(n)]

        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)
        return

        # 显示原始标签在右下散点图上

    def toggleLabelColor(self):

        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            if self.data.loc[start, 'label_flag'] == 0:
                color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            else:
                if self.is_toggled:
                    color = self.checkColor(self.data.loc[start, 'label'], first=False)
                else:
                    color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
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

    def featureExtraction(self):
        # 特征提取：找到数据帧中的列名
        # preprocessing: find column names in dataframe
        new_column_names = find_data_columns(self.sensor_dict, self.column_names)

        segments, start_indices = split_dataframe(self.data, self.data_length)
        end_indices = [i + self.data_length for i in start_indices]

        # 获取潜在特征
        # get latent feature
        from deepview.clustering_pytorch.nnet.common_config import get_model
        from deepview.clustering_pytorch.datasets.factory import prepare_unsup_dataset
        from deepview.clustering_pytorch.nnet.train_utils import AE_eval_time_series, simclr_eval_time_series

        train_loader = prepare_unsup_dataset(segments, new_column_names)

        config = self.root.config
        cfg = read_config(config)
        unsup_model_path = get_unsup_model_folder(cfg)
        full_model_path = os.path.join(self.cfg["project_path"], unsup_model_path, self.model_path)

        if 'AE_CNN' in self.model_path.upper():
            p_setup = 'autoencoder'
        elif 'simclr' in self.model_path.upper():
            p_setup = 'simclr'
        else:
            raise ValueError("Invalid model type")

        model = get_model(p_backbone=self.model_name, p_setup=p_setup, num_channel=len(new_column_names))
        # model.load_state_dict(torch.load(full_model_path))

        if torch.cuda.is_available():
            model.load_state_dict(torch.load(full_model_path))
        else:
            model.load_state_dict(torch.load(full_model_path, map_location=torch.device('cpu')))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        if p_setup == 'autoencoder':
            representation_list, _, _, _, _ = \
                AE_eval_time_series(train_loader, model, device)
        elif p_setup == 'simclr':
            representation_list, _ = simclr_eval_time_series(train_loader, model, device)

        # PCA latent representation to shape=(2, len) PCA降维到形状为 (2, len)
        repre_concat = np.concatenate(representation_list)
        repre_reshape = repre_concat.reshape(repre_concat.shape[0], -1)
        repre_tsne = reduce_dimension_with_tsne(repre_reshape)

        return start_indices, end_indices, repre_tsne


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
    #         self.data.loc[(self.data['_timestamp'] >= int(first_timestamp)) & (
    #                 self.data['_timestamp'] <= int(second_timestamp)), 'label'] = name
    #     self.handleSaveButton()

    def getSelectedAreaToSave(self, area_data):
        # print(areaData)
        try:
            area_data = json.loads(area_data)  # 解析 JSON 字符串
            # print("Parsed data:", areaData)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            return
        for reg in area_data:
            name = reg[0].get("name")
            first_timestamp = reg[0].get("timestamp", {}).get("start")
            second_timestamp = reg[0].get("timestamp", {}).get("end")
            self.data.loc[(self.data['unixtime'] >= int(first_timestamp)) & (
                    self.data['unixtime'] <= int(second_timestamp)), 'label'] = name
        self.handleSaveButton()

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
                self.data.to_csv(edit_data_path)
        except:
            print('save data error!')
        else:
            print('save success')

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


def split_dataframe(df, segment_size):
    '''
    将DataFrame按segment_size分割成多个片段
    split the dataframe into segments based on segment_size
    '''
    segments = []
    start_indices = []
    num_segments = len(df) // segment_size

    for i in range(num_segments):
        start_index = i * segment_size
        end_index = start_index + segment_size
        segment = df.iloc[start_index:end_index]
        segments.append(segment)
        start_indices.append(start_index)

    # 处理最后一个片段，如果其大小不足一个segment_size
    # Handle the last segment which may not have the full segment size
    start_index = num_segments * segment_size
    last_segment = df.iloc[start_index:]
    if len(last_segment) == segment_size:
        segments.append(last_segment)
        start_indices.append(start_index)

    return segments, start_indices


def find_data_columns(sensor_dict, column_names):
    new_column_names = []
    for column_name in column_names:
        real_names = sensor_dict[column_name]  # 获取每个列名对应的实际列名
        new_column_names.extend(real_names)  # 将实际列名添加到新的列名列表中
    return new_column_names


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


def reduce_dimension_with_tsne(array, method='tsne'):
    # tsne or pca
    tsne = TSNE(n_components=2)  # 创建TSNE对象，降维到2维
    reduced_array = tsne.fit_transform(array)  # 对数组进行降维
    return reduced_array
