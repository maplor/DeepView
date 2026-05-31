import os

import pandas as pd
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QProgressBar

from deepview.gui.components import _create_grid_layout, _create_label_widget
from deepview.gui.widgets import ConfigEditor
from deepview.utils import auxiliaryfunctions


class TrainNetworkUiMixin:
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