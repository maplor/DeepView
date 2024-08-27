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

from deepview.gui.supervised_cl import SupervisedClWidget


from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWidgets import QSizePolicy


class SupervisedCLTab(DefaultTab):
    update_theme_signal = Signal(str)

    def __init__(self, root, parent, h1_description):
        super(SupervisedCLTab, self).__init__(root, parent, h1_description)
        self.root = root
        self.supervised_cl_widget = None

    @Slot(str)
    def update_theme(self, theme):
        if theme == 'dark':
            # self.web_view.page().runJavaScript("updateTheme('dark');")
            # self.web_view_map.page().runJavaScript("updateMapTheme('dark');")
            # 设置前景颜色为黑色
            # self.supervised_cl_widget.viewC.setBackground('k')
            pass

        else:
            # self.web_view.page().runJavaScript("updateTheme('light');")
            # self.web_view_map.page().runJavaScript("updateMapTheme('light');")

            # 设置前景颜色为白色
            # self.supervised_cl_widget.viewC.setBackground('w')
            pass

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()

    def _set_page(self):
        config = self.root.config  # project/config.yaml

        # Read file path for pose_config file. >> pass it on
        cfg = auxiliaryfunctions.read_config(config)

        self.supervised_cl_widget = SupervisedClWidget(self.root, cfg)

        # 在这里调用init文件
        self.main_layout.addWidget(self.supervised_cl_widget)


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
