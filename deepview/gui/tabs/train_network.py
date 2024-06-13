#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
from pathlib import Path

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QShowEvent
from PySide6.QtWidgets import QProgressBar

import pandas as pd

from deepview.gui.components import (
    DefaultTab,
    # ShuffleSpinBox,
    _create_grid_layout,
    _create_label_widget,
)
from deepview.gui.widgets import ConfigEditor

import deepview
from deepview.utils import auxiliaryfunctions

from PySide6.QtCore import Signal


class TrainNetwork(DefaultTab):
    # 定义进度信号
    progress_update = Signal(int)

    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)

        # get sensor/columns dictionary from config.yaml
        root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = root_cfg['sensor_dict']

        self.models = ['AE_CNN', 'SimCLR_LSTM']
        self.select_column = []
        self.max_iter = 30
        self.learning_rate = 0.0005
        self.batch_size = 32
        self.net_type = self.models[0]
        self.data_length = 180

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Model Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        # ---------
        self.main_layout.addWidget(_create_label_widget("Data Attributes", "font:bold"))
        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.main_layout.addLayout(self.dataset_attributes_dataset)
        # ---------

        # set processing window
        self.setWindowTitle("Progress Demo")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.train_network)

        # self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        # 连接信号和槽
        self.progress_update.connect(self.updateProgress)

    def updateProgress(self, value):
        self.progress_bar.setValue(value)

    def _generate_layout_attributes(self, layout):
        available_width = self.screen().availableGeometry().width()
        net_label = QtWidgets.QLabel("Network type")
        net_label.setFixedWidth(available_width/10)
        self.display_net_type = QtWidgets.QComboBox()
        self.display_net_type.addItems(self.models)
        self.display_net_type.setFixedWidth(available_width/10)
        self.display_net_type.currentIndexChanged.connect(self.log_net_choice)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Maximum iterations")
        dispiters_label.setFixedWidth(available_width/10)
        self.display_iters_spin = QtWidgets.QSpinBox()
        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(10000)
        self.display_iters_spin.setValue(30)
        self.display_iters_spin.setFixedWidth(available_width/10)
        self.display_iters_spin.valueChanged.connect(self.log_display_iters)

        # Save iterations
        saveiters_label = QtWidgets.QLabel("Learning rate")
        saveiters_label.setFixedWidth(available_width/10)
        self.save_iters_spin = QtWidgets.QLineEdit()
        self.save_iters_spin.setFixedWidth(2)
        # self.save_iters_spin.setMinimum(1)
        # self.save_iters_spin.setMaximum(1)
        self.save_iters_spin.setText("0.0005")
        self.save_iters_spin.setFixedWidth(available_width/10)
        self.save_iters_spin.textChanged.connect(self.log_init_lr)

        # Max iterations
        maxiters_label = QtWidgets.QLabel("Batch size")
        maxiters_label.setFixedWidth(available_width/10)
        self.batchsize_spin = QtWidgets.QSpinBox()
        self.batchsize_spin.setMinimum(1)
        self.batchsize_spin.setMaximum(10000)
        self.batchsize_spin.setValue(32)
        self.batchsize_spin.setFixedWidth(available_width/10)
        self.batchsize_spin.valueChanged.connect(self.log_batch_size)

        layout.addWidget(net_label, 0, 0)
        layout.addWidget(self.display_net_type, 0, 1)
        layout.addWidget(dispiters_label, 0, 2)
        layout.addWidget(self.display_iters_spin, 0, 3)
        layout.addWidget(saveiters_label, 0, 4)
        layout.addWidget(self.save_iters_spin, 0, 5)
        layout.addWidget(maxiters_label, 0, 6)
        layout.addWidget(self.batchsize_spin, 0, 7)

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

        # self.display_dataset_container = QtWidgets.QHBoxLayout()
        self.display_dataset_cb_list = []

        column_list = []

        rowNum = 3  # default one row 3 columns

        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                    os.path.join(self.root.project_folder, trainingsetfolder),
                    relative=False,
            ):
                if len(column_list) == 0:
                    df = pd.read_pickle(filename)
                    column_list = list(df.columns)
                cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                grid.addWidget(cb, len(self.display_dataset_cb_list) // rowNum,
                               len(self.display_dataset_cb_list) % rowNum)
                self.display_dataset_cb_list.append(cb)  # display filenames

        net_label = QtWidgets.QLabel("Input data columns")
        self.display_column_container = QtWidgets.QHBoxLayout()
        self.display_column_cb_list = []

        # create checkbox according to data columns
        combined_columns = list(self.sensor_dict.keys())
        # self.data_column = combined_columns
        for column in combined_columns:
            cb = QtWidgets.QCheckBox(column)
            self.display_column_container.addWidget(cb)
            cb.stateChanged.connect(self.log_data_columns)
            self.display_column_cb_list.append(cb)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Input data length")
        self.display_datalen_spin = QtWidgets.QSpinBox()
        self.display_datalen_spin.setMinimum(1)
        self.display_datalen_spin.setMaximum(10000)
        self.display_datalen_spin.setValue(int(self.data_length))
        self.display_datalen_spin.valueChanged.connect(self.log_display_datalen)

        layout.addWidget(select_label, 0, 0)
        layout.addWidget(scroll, 0, 1)
        layout.addWidget(net_label, 1, 0)
        layout.addLayout(self.display_column_container, 1, 1)
        layout.addWidget(dispiters_label, 2, 0)
        layout.addWidget(self.display_datalen_spin, 2, 1)

    def log_data_columns(self, value):
        self.root.logger.info(f"Select input data columns to {self.select_column}")
        sender = self.sender()
        if sender.isChecked():
            if sender.text() not in self.select_column:
                self.select_column.append(sender.text())
        else:
            if sender.text() in self.select_column:
                self.select_column.remove(sender.text())
        print(self.select_column)

    def log_display_datalen(self, value):
        self.root.logger.info(f"Display input data length set to {value}")
        print(int(value))
        self.data_length = int(value)

    def log_net_choice(self, net):
        self.root.logger.info(f"Network type set to {self.display_net_type.currentText()}")
        print(self.display_net_type.currentText())
        self.net_type = self.display_net_type.currentText()

    def log_display_iters(self, value):
        self.root.logger.info(f"Run iters (epochs) set to {value}")
        print(int(value))
        self.max_iter = int(value)

    def log_init_lr(self, value):
        self.root.logger.info(f"Learning rate set to {value}")
        print(float(value))
        self.learning_rate = float(value)

    def log_batch_size(self, value):
        self.root.logger.info(f"Batch size set to {value}")
        print(int(value))
        self.batch_size = int(value)

    def log_save_iters(self, value):
        self.root.logger.info(f"Save iters set to {value}")

    def log_max_iters(self, value):
        self.root.logger.info(f"Max iters set to {value}")

    def log_snapshots(self, value):
        self.root.logger.info(f"Max snapshots to keep set to {value}")

    def open_posecfg_editor(self):
        editor = ConfigEditor(self.root.model_cfg_path)  # pose_cfg_path
        editor.show()

    def train_network(self):
        self.progress_bar.setValue(0)

        config = self.root.config

        net_type = str(self.net_type.upper())
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())
        max_iter = int(self.display_iters_spin.text())

        data_length = int(self.display_datalen_spin.text())

        newSelectFilename = []
        for cb in self.display_dataset_cb_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        select_filenames = newSelectFilename

        newSelectColumn = []
        for i, cb in enumerate(self.display_column_cb_list):
            if cb.isChecked():
                newSelectColumn.append(cb.text())
        data_columns = []
        for sensor in newSelectColumn:
            data_columns.extend(self.sensor_dict[sensor])

        deepview.train_network(
            self.sensor_dict,
            self.progress_update,
            config,
            select_filenames,
            net_type=net_type,
            lr=learning_rate,
            batch_size=batch_size,
            num_epochs=max_iter,
            data_len=data_length,
            data_column=data_columns
        )
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("The network is now trained and ready to evaluate.")
        msg.setInformativeText(
            "Use the function 'evaluate_network' to evaluate the network."
        )

        msg.setWindowTitle("Info")
        msg.setMinimumWidth(900)
        self.logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        self.logo = self.logo_dir + "/assets/logo.png"
        msg.setWindowIcon(QIcon(self.logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()


def get_sensor_columns(strings):
    '''
    其实放在config文件里更好，直接定义sensors，然后在代码中处理
    define a list of sensors, find sensors used in target data
    '''
    combined_strings = []
    current_combined = strings[0]

    for i in range(1, len(strings)):
        prefix_length = min(len(current_combined), len(strings[i]))
        prefix_length = next((k for k in range(prefix_length, 0, -1) if current_combined[:k] == strings[i][:k]), 0)

        if prefix_length > 0:
            current_combined = current_combined[:prefix_length]
        else:
            combined_strings.append(current_combined)
            current_combined = strings[i]

    combined_strings.append(current_combined)
    return combined_strings
