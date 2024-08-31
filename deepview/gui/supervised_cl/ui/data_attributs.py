import os
from pathlib import Path
import pickle
import pandas as pd
import pyqtgraph as pg
import numpy as np

from PySide6 import QtWidgets

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox,
    QCheckBox,
    QDialog,
    QRadioButton,
    QSplitter,
    QFrame,
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QComboBox, QPushButton, QSpacerItem, QSizePolicy, QLineEdit,
    QMessageBox
)
from PySide6.QtCore import Qt

# 从deepview.utils.auxiliaryfunctions导入多个函数
from deepview.utils.auxiliaryfunctions import (
    read_config,
    get_param_from_path,
    get_unsupervised_set_folder,
    get_raw_data_folder,
    get_unsup_model_folder,
    grab_files_in_folder_deep
)

from deepview.gui.components import (
    DefaultTab,
    # ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)

from deepview.gui.label_with_interactive_plot.utils import (
featureExtraction,
get_data_from_pkl,
)
from deepview.utils import auxiliaryfunctions


class dataAttributesWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = self.main_window.root_cfg['sensor_dict']
        self.repre_tsne, self.flag_concat, self.label_concat = None, None, None

        # Create a vertical layout
        self.layout = QVBoxLayout(self)

        # ---------
        self.layout.addWidget(_create_label_widget("Data Attributes", "font:bold"))
        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.layout.addLayout(self.dataset_attributes_dataset)
        # ---------
        


        

    def _generate_layout_attributes_dataset(self, layout):
        trainingsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()

        select_label = QtWidgets.QLabel("Select dataset file")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scrollContent = QtWidgets.QWidget(scroll)
        grid = QtWidgets.QGridLayout(scrollContent)
        grid.setAlignment(Qt.AlignTop)
        scrollContent.setLayout(grid)
        scroll.setWidget(scrollContent)

        # 创建“全选”按钮
        cb_label = QtWidgets.QLabel("Select all files")
        self.select_all_checkbox = QtWidgets.QCheckBox("All")
        selected = QtWidgets.QVBoxLayout()
        selected.addWidget(self.select_all_checkbox)
        # self.layout.addWidget(self.select_all_checkbox)
        # 连接“全选”按钮的状态改变信号到槽函数
        self.select_all_checkbox.stateChanged.connect(self.select_all)

        self.display_dataset_cb_list = []
        column_list = []
        rowNum = 3  # default one row 3 columns
        self.checkboxes = QtWidgets.QCheckBox('')
        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                    os.path.join(self.root.project_folder, trainingsetfolder),
                    relative=False,
            ):
                if len(column_list) == 0:
                    df = pd.read_pickle(filename)
                    column_list = list(df.columns)
                self.checkboxes = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                grid.addWidget(self.checkboxes, len(self.display_dataset_cb_list) // rowNum,
                               len(self.display_dataset_cb_list) % rowNum)
                self.display_dataset_cb_list.append(self.checkboxes)  # display filenames

        # 标志位，用于控制槽函数逻辑
        self.updating = False

        # 连接各个选项的状态改变信号到槽函数
        for checkbox in self.display_dataset_cb_list:
            checkbox.stateChanged.connect(self.update_select_all_checkbox)



        layout.addWidget(cb_label, 0, 0)
        layout.addLayout(selected, 0, 1)
        layout.addWidget(select_label, 1, 0)
        layout.addWidget(scroll, 1, 1)


    def select_all(self, state):
        if not self.updating:
            self.updating = True
            # 根据“全选”按钮的状态设置各个选项的状态
            for checkbox in self.display_dataset_cb_list:
                checkbox.setChecked(state == Qt.Checked)
            self.update_selected_items()
            self.updating = False

    def update_select_all_checkbox(self):
        if not self.updating:
            self.updating = True
            # 检查所有选项的状态以更新“全选”按钮的状态
            all_checked = all(checkbox.isChecked() for checkbox in self.display_dataset_cb_list)
            any_unchecked = any(not checkbox.isChecked() for checkbox in self.display_dataset_cb_list)
            if all_checked:
                self.select_all_checkbox.setCheckState(Qt.Checked)
            elif any_unchecked:
                self.select_all_checkbox.setCheckState(Qt.Unchecked)
            else:
                self.select_all_checkbox.setTristate(False)
                self.select_all_checkbox.setCheckState(Qt.PartiallyChecked)
            self.update_selected_items()
            self.updating = False

    def update_selected_items(self):
        # 更新当前选中的选项
        selected_items = [checkbox.text() for checkbox in self.display_dataset_cb_list if checkbox.isChecked()]
        print(f"当前选中的选项: {selected_items}")


    def train_network(self):
        newSelectFilename = []
        for cb in self.display_dataset_cb_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        select_filenames = newSelectFilename