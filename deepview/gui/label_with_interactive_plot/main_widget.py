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
from deepview.gui.label_with_interactive_plot._selection_area import SelectionAreaMixin
from deepview.gui.label_with_interactive_plot._data_controls import DataControlsMixin
from deepview.gui.label_with_interactive_plot._settings_area import SettingsAreaMixin
from deepview.gui.label_with_interactive_plot._compute import ComputeMixin
from deepview.gui.label_with_interactive_plot._interaction_controls import InteractionControlsMixin
from deepview.gui.label_with_interactive_plot._left_plot import LeftPlotMixin
from deepview.gui.label_with_interactive_plot._center_plot import CenterPlotMixin
from deepview.gui.label_with_interactive_plot._chart_utils import (
    combine_rectangles,
    find_charts_data_columns,
)
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
class LabelWithInteractivePlot(LabelControlsMixin, VideoControlsMixin, LayoutMixin, SelectionAreaMixin, DataControlsMixin, SettingsAreaMixin, ComputeMixin, InteractionControlsMixin, LeftPlotMixin, CenterPlotMixin, QWidget):
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

    '''
    ==================================================
    bottom center area: result plot底部中心区域:结果图
    - self.viewC PlotWidget
    - self.selectRect QRect
    - self.lastChangePoint list(SpotItem)
    - self.lastMarkList list(LinearRegionItem)
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
