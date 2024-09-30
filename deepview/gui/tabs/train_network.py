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


class TrainWorker(QtCore.QObject):
    progress = QtCore.Signal(int)
    finished = QtCore.Signal()
    stopped = QtCore.Signal()


    def __init__(self,config, net_type, sensor_dict, select_filenames, learning_rate, batch_size, max_iter, data_length, data_columns):
        super().__init__()
        self.config = config
        self.net_type = net_type
        self.sensor_dict = sensor_dict
        self.select_filenames = select_filenames
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.data_length = data_length
        self.data_columns = data_columns
        self._is_running = True

 
    def train_network(self):
        self.progress.emit(0)


        # Call the training function
        deepview.train_network(
            self.sensor_dict,
            self.progress,
            self.config,
            self.select_filenames,
            net_type=self.net_type,
            lr=self.learning_rate,
            batch_size=self.batch_size,
            num_epochs=self.max_iter,
            data_len=self.data_length,
            data_column=self.data_columns,
            stop_callback=self.check_running
        )
        # if not self._is_running:
        #     self.stopped.emit()
        #     return

        # 根据运行状态发出信号
        if self._is_running:
            self.finished.emit()
        else:
            self.stopped.emit()
        # self.finished.emit()

    def stop(self):
        self._is_running = False
    
    def check_running(self):
        return self._is_running



class TrainNetwork(DefaultTab):
    # 定义进度信号
    progress_update = Signal(int)

    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)
        self.root = root
        # get sensor/columns dictionary from config.yaml
        root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = root_cfg['sensor_dict']

        self.models = ['AE_CNN', 'SimCLR_LSTM', 'shortAE']
        self.select_column = []
        self.max_iter = 30
        self.learning_rate = 0.0001
        self.batch_size = 512
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

        self.button_layout = QtWidgets.QHBoxLayout()
        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.start_training)
        # self.ok_button.clicked.connect(self.train_network)

        self.stop_button = QtWidgets.QPushButton("Stop")
        self.stop_button.setMinimumWidth(150)
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)

        self.button_layout.addStretch()
        self.button_layout.addWidget(self.ok_button)
        self.button_layout.addWidget(self.stop_button)

        # self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.progress_bar)
        # self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)
        self.main_layout.addLayout(self.button_layout)


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
        self.save_iters_spin.setText("0.0001")
        self.save_iters_spin.setFixedWidth(available_width/10)
        self.save_iters_spin.textChanged.connect(self.log_init_lr)

        # Max iterations
        maxiters_label = QtWidgets.QLabel("Batch size")
        maxiters_label.setFixedWidth(available_width/10)
        self.batchsize_spin = QtWidgets.QSpinBox()
        self.batchsize_spin.setMinimum(1)
        self.batchsize_spin.setMaximum(10000)
        self.batchsize_spin.setValue(1028)
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

        layout.addWidget(cb_label, 0, 0)
        layout.addLayout(selected, 0, 1)
        layout.addWidget(select_label, 1, 0)
        layout.addWidget(scroll, 1, 1)
        layout.addWidget(net_label, 2, 0)
        layout.addLayout(self.display_column_container, 2, 1)
        layout.addWidget(dispiters_label, 3, 0)
        layout.addWidget(self.display_datalen_spin, 3, 1)

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


    def start_training(self):
        
        # TODO: check if training is already in progress
        if hasattr(self, 'training_in_progress') and self.training_in_progress:
            print("Training is already in progress.")
            return

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
        data_columns = transfer_sensor2columns(newSelectColumn, self.sensor_dict)


        self.train_thread = QtCore.QThread()
        self.train_worker = TrainWorker(config, net_type, self.sensor_dict, select_filenames, learning_rate, batch_size, max_iter, data_length, data_columns)
        self.train_worker.moveToThread(self.train_thread)

        self.train_worker.progress.connect(self.progress_bar.setValue)
        self.train_worker.finished.connect(self.on_training_finished)
        self.train_worker.stopped.connect(self.on_training_stopped)
        self.train_thread.started.connect(self.train_worker.train_network)
        self.train_worker.finished.connect(self.clean_up)


        self.ok_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.training_in_progress = True  # 设置标志
        self.train_thread.start()

    def clean_up(self):
        self.training_in_progress = False  # 重置标志
        self.train_thread.quit()
        self.train_thread.wait()
        self.train_worker.deleteLater()
        self.train_thread.deleteLater()
        self.ok_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def stop_training(self):
        self.train_worker.stop()

    def on_training_finished(self):
        self.stop_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.show_message("The network is now trained and ready to use.")

    def on_training_stopped(self):
        self.stop_button.setEnabled(False)
        self.ok_button.setEnabled(True)
        self.clean_up()
        self.show_message("Training was stopped.")

    def show_message(self, text):
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(text)
        msg.setWindowTitle("Info")
        msg.setMinimumWidth(900)
        logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
        logo = logo_dir + "assets/logo.png"
        msg.setWindowIcon(QIcon(logo))
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
        msg.exec_()


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
        # data_columns = []
        # for sensor in newSelectColumn:
        #     # replace columns of GPS sensor
        #     if sensor.upper() == "GPS":
        #         data_columns.extend(['GPS_velocity', 'GPS_bearing'])
        #     else:
        #         data_columns.extend(self.sensor_dict[sensor])
        data_columns = transfer_sensor2columns(newSelectColumn, self.sensor_dict)

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
        msg.setText("The network is now trained and ready to use.")
        msg.setInformativeText(
            "Use the function 'Label with Interaction Plot' to visualize the data."
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

def transfer_sensor2columns(newSelectColumn, sensor_dict):
    data_columns = []
    for sensor in newSelectColumn:
        # replace columns of GPS sensor
        if sensor.upper() == "GPS":
            data_columns.extend(['GPS_velocity', 'GPS_bearing'])
        else:
            data_columns.extend(sensor_dict[sensor])
    return data_columns