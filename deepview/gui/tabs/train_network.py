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

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QShowEvent
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


class TrainNetwork(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(TrainNetwork, self).__init__(root, parent, h1_description)

        # use the default model_cfg file for default values
        default_pose_cfg_path = os.path.join(
            Path(deepview.__file__).parent, "model_cfg.yaml"
        )  # 检查路径是否是unsup-models下面的model_cfg.yaml

        pose_cfg = auxiliaryfunctions.read_plainconfig(default_pose_cfg_path)
        # set default values of windows
        # self.display_iters = str(pose_cfg["display_iters"])
        # self.save_iters = str(pose_cfg["save_iters"])
        # self.max_iters = str(pose_cfg["multi_step"][-1][-1])

        self.net_type = str(pose_cfg['net_type'])
        self.learning_rate = str(pose_cfg['lr_init'])
        self.batch_size = str(pose_cfg['batch_size'])
        self.max_iter = str(pose_cfg['max_epochs'])

        self.data_length = str(pose_cfg['data_length'])
        self.data_column = str(pose_cfg['data_columns'])
        # self._set_page()

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
        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 5, 5, 5))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.main_layout.addLayout(self.dataset_attributes_dataset)
        # ---------

        # self.edit_posecfg_btn = QtWidgets.QPushButton("Edit model_cfg.yaml")
        # self.edit_posecfg_btn.setMinimumWidth(150)
        # self.edit_posecfg_btn.clicked.connect(self.open_posecfg_editor)

        self.ok_button = QtWidgets.QPushButton("Train Network")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.train_network)

        # self.main_layout.addWidget(self.edit_posecfg_btn, alignment=Qt.AlignRight)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)


    def _generate_layout_attributes(self, layout):
        # Shuffle
        # shuffle_label = QtWidgets.QLabel("Shuffle")
        # self.shuffle = ShuffleSpinBox(root=self.root, parent=self)
        net_label = QtWidgets.QLabel("Network type")
        self.display_net_type = QtWidgets.QComboBox()
        self.display_net_type.addItems(['CNN_AE', 'DeepConvLSTM'])
        self.display_net_type.currentIndexChanged.connect(self.log_net_choice)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Maximum iterations")
        self.display_iters_spin = QtWidgets.QSpinBox()
        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(10000)
        self.display_iters_spin.setValue(int(self.max_iter))
        self.display_iters_spin.valueChanged.connect(self.log_display_iters)

        # Save iterations
        saveiters_label = QtWidgets.QLabel("Learning rate")
        self.save_iters_spin = QtWidgets.QLineEdit()
        # self.save_iters_spin.setMinimum(1)
        # self.save_iters_spin.setMaximum(1)
        self.save_iters_spin.setText(self.learning_rate)
        self.save_iters_spin.textChanged.connect(self.log_init_lr)

        # Max iterations
        maxiters_label = QtWidgets.QLabel("Batch size")
        self.batchsize_spin = QtWidgets.QSpinBox()
        self.batchsize_spin.setMinimum(1)
        self.batchsize_spin.setMaximum(10000)
        self.batchsize_spin.setValue(int(self.batch_size))
        self.batchsize_spin.valueChanged.connect(self.log_batch_size)

        layout.addWidget(net_label, 0, 0)
        layout.addWidget(self.display_net_type, 0, 1)
        layout.addWidget(dispiters_label, 0, 2)
        layout.addWidget(self.display_iters_spin, 0, 3)
        layout.addWidget(saveiters_label, 0, 4)
        layout.addWidget(self.save_iters_spin, 0, 5)
        layout.addWidget(maxiters_label, 0, 6)
        layout.addWidget(self.batchsize_spin, 0, 7)
        # layout.addWidget(snapkeep_label, 0, 8)
        # layout.addWidget(self.snapshots, 0, 9)
        # layout.addWidget()

    def _generate_layout_attributes_dataset(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        trainingsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
        # todo todo: 需要做成复选框，选多个csv文件
        select_label = QtWidgets.QLabel("Select dataset file")
        # self.display_dataset_cb = QtWidgets.QComboBox()
        self.display_dataset_container = QtWidgets.QHBoxLayout()
        self.display_dataset_cb_list = []

        column_list = []

        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, trainingsetfolder),
                relative=False,
            ):
                if len(column_list) == 0:
                    df = pd.read_pickle(filename)
                    column_list = list(df.columns)
                cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                self.display_dataset_container.addWidget(cb)
                self.display_dataset_cb_list.append(cb)

        net_label = QtWidgets.QLabel("Input data columns")
        self.display_column_container = QtWidgets.QHBoxLayout()
        self.display_column_cb_list = []
        # read from first data file
        for column in column_list:
            cb = QtWidgets.QCheckBox(column)
            self.display_column_container.addWidget(cb)
            self.display_column_cb_list.append(cb)

        # self.display_column_cb = QtWidgets.QComboBox()
        # self.display_column_cb.addItems(['acc_x', 'acc_y', 'acc_z'])
        # self.display_column_cb.currentIndexChanged.connect(self.log_data_columns)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Input data length")
        self.display_datalen_spin = QtWidgets.QSpinBox()
        self.display_datalen_spin.setMinimum(1)
        self.display_datalen_spin.setMaximum(10000)
        self.display_datalen_spin.setValue(int(self.data_length))
        self.display_datalen_spin.valueChanged.connect(self.log_display_datalen)


        layout.addWidget(select_label, 0, 0)
        layout.addLayout(self.display_dataset_container, 0, 1)
        layout.addWidget(net_label, 1, 0)
        layout.addLayout(self.display_column_container, 1, 1)
        layout.addWidget(dispiters_label, 2, 0)
        layout.addWidget(self.display_datalen_spin, 2, 1)



    def log_data_columns(self, value):
        self.root.logger.info(f"Select input data columns to {value}")

    def log_display_datalen(self, value):
        self.root.logger.info(f"Display input data length set to {value}")

    def log_net_choice(self, net):
        # todo, bug, when selecting DeepConvLSTM, bug
        self.root.logger.info(f"Network type set to {net.upper()}")
        # self.root.logger.info(f"TODO: test dataset set to {net.upper()}")

    def log_display_iters(self, value):
        self.root.logger.info(f"Run iters (epochs) set to {value}")

    def log_init_lr(self, value):
        self.root.logger.info(f"Learning rate set to {value}")

    def log_batch_size(self, value):
        self.root.logger.info(f"Batch size set to {value}")
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
        config = self.root.config

        net_type = str(self.net_type.upper())
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())
        # self.batch_size = str(pose_cfg['batch_size'])
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
        data_column = newSelectColumn


        deepview.train_network(
            config,
            select_filenames,
            net_type=net_type,
            lr=learning_rate,
            batch_size=batch_size,
            num_epochs=max_iter,
            data_len=data_length,
            data_column=data_column
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
