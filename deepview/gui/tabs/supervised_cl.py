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
        self.is_initialized = False
        self.supervised_cl_widget = None

    @Slot(str)
    def update_theme(self, theme):
        if not self.is_initialized:
            return
        if theme == 'dark':
            self.supervised_cl_widget.old_scatter_map_widget.old_map.setBackground('k')
            self.supervised_cl_widget.new_scatter_map_widget.map_1.setBackground('k')
            self.supervised_cl_widget.new_scatter_map_widget.map_2.setBackground('k')

        else:
            self.supervised_cl_widget.old_scatter_map_widget.old_map.setBackground('w')
            self.supervised_cl_widget.new_scatter_map_widget.map_1.setBackground('w')
            self.supervised_cl_widget.new_scatter_map_widget.map_2.setBackground('w')

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self.is_initialized = True
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
