import os
import pandas as pd
from pathlib import Path

from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QUrl, Signal, Slot
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit
from PySide6.QtWidgets import QLabel
from deepview.utils import auxiliaryfunctions
from deepview.gui.components import (
    DefaultTab,
    DefaultWebTab,
    # TestfileSpinBox,
    # _create_horizontal_layout,
    # _create_label_widget,
    # _create_vertical_layout,
)

from deepview.gui.label_with_interactive_plot import LabelWithInteractivePlot, Backend
import pyqtgraph as pg

from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWidgets import QSizePolicy


class LabelWithInteractivePlotTab(DefaultWebTab):
    update_theme_signal = Signal(str)

    def __init__(self, root, parent, h1_description):
        super(LabelWithInteractivePlotTab, self).__init__(root, parent, h1_description)
        self.root = root
        self.label_with_interactive_plot = None

    @Slot(str)
    def update_theme(self, theme):
        if theme == 'dark':
            self.web_view.page().runJavaScript("updateTheme('dark');")
            self.web_view_map.page().runJavaScript("updateMapTheme('dark');")
            # 设置前景颜色为黑色
            self.label_with_interactive_plot.viewC.setBackground('k')

        else:
            self.web_view.page().runJavaScript("updateTheme('light');")
            self.web_view_map.page().runJavaScript("updateMapTheme('light');")

            # 设置前景颜色为白色
            self.label_with_interactive_plot.viewC.setBackground('w')

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()

    def _set_page(self):
        config = self.root.config  # project/config.yaml

        # Read file path for pose_config file. >> pass it on
        cfg = auxiliaryfunctions.read_config(config)

        self.label_with_interactive_plot = LabelWithInteractivePlot(self.root, cfg)
        # 设置折线图的配置
        self.channel = QWebChannel()
        self.label_with_interactive_plot.backend.view = self.web_view  # 让backend有一个view的引用
        self.channel.registerObject('backend', self.label_with_interactive_plot.backend)
        self.web_view.page().setWebChannel(self.channel)

        self.web_view.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        self.web_view.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        project_root = os.path.dirname(os.path.abspath(__file__))

        # 构建 HTML 文件的路径
        html_file_path = os.path.join(project_root, '..', 'html', 'line_chart_index.html')

        # 将路径转换为本地文件 URL
        local_file_url = QUrl.fromLocalFile(html_file_path)

        # 加载 HTML 文件到 QWebEngineView
        self.web_view.setUrl(local_file_url)

        self.label_with_interactive_plot.left_row3_layout.addWidget(self.web_view)

        # 设置地图的配置
        self.channel_map = QWebChannel()
        self.label_with_interactive_plot.backend_map.view = self.web_view_map
        self.channel_map.registerObject('backendmap', self.label_with_interactive_plot.backend_map)
        self.web_view_map.page().setWebChannel(self.channel_map)

        self.web_view_map.settings().setAttribute(QWebEngineSettings.JavascriptEnabled, True)
        self.web_view_map.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, True)

        # 构建 HTML 文件的路径
        html_file_path_map = os.path.join(project_root, '..', 'html', 'map_chart_index.html')

        # 将路径转换为本地文件 URL
        local_file_url_map = QUrl.fromLocalFile(html_file_path_map)

        # 加载 HTML 文件到 QWebEngineView
        self.web_view_map.setUrl(local_file_url_map)

        self.label_with_interactive_plot.left_row1_layout.addWidget(self.web_view_map)

        # 在这里调用init文件
        self.main_layout.addWidget(self.label_with_interactive_plot)


def get_plot_data(config):
    """
    Extracts the scoremap, locref, partaffinityfields (if available).

    Returns a dictionary
    read data and model structure in this function
    ----------
    config : string
        Full path of the config.yaml file as a string.
    """

    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)

    rawdata_file = list(
        Path(os.path.join(cfg["project_path"], "raw-data")).glob('*.csv'),
    )[0]

    if not rawdata_file:
        print('can not find raw data')
        return

    edit_data_path = os.path.join(cfg["project_path"], "edit-data", rawdata_file.name)
    if Path(os.path.join(cfg["project_path"], "edit-data", rawdata_file.name)).exists():
        df = pd.read_csv(edit_data_path)
    else:
        df = pd.read_csv(rawdata_file, low_memory=False)
        df['label'] = ""

    df['datetime'] = pd.to_datetime(df['timestamp']).apply(lambda x: x.timestamp())

    os.chdir(str(start_path))  # Change the current working directory to the specified path

    return df
