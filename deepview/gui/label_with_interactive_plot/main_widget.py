import logging
import os

from ruamel.yaml import YAML
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QFileSystemWatcher, QThreadPool, QTimer, Signal
from PySide6.QtWidgets import QWidget

from deepview.utils.auxiliaryfunctions import (
    get_db_folder,
    read_config,
)
from deepview.gui.label_with_interactive_plot.backend import Backend, BackendMap
from deepview.gui.label_with_interactive_plot._center_plot import CenterPlotMixin
from deepview.gui.label_with_interactive_plot._compute import ComputeMixin
from deepview.gui.label_with_interactive_plot._data_controls import DataControlsMixin
from deepview.gui.label_with_interactive_plot._interaction_controls import InteractionControlsMixin
from deepview.gui.label_with_interactive_plot._labeling import LabelControlsMixin
from deepview.gui.label_with_interactive_plot._layout import LayoutMixin
from deepview.gui.label_with_interactive_plot._left_plot import LeftPlotMixin
from deepview.gui.label_with_interactive_plot._right_settings import RightSettingsMixin
from deepview.gui.label_with_interactive_plot._selection_area import SelectionAreaMixin
from deepview.gui.label_with_interactive_plot._settings_area import SettingsAreaMixin
from deepview.gui.label_with_interactive_plot._video import VideoControlsMixin


# 创建一个蓝色的pg.mkPen对象，宽度为2
clickedPen = pg.mkPen('b', width=2)

# 定义LabelWithInteractivePlot类，继承自QWidget
class LabelWithInteractivePlot(LabelControlsMixin, VideoControlsMixin, LayoutMixin, SelectionAreaMixin, DataControlsMixin, SettingsAreaMixin, ComputeMixin, InteractionControlsMixin, LeftPlotMixin, CenterPlotMixin, RightSettingsMixin, QWidget):
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
