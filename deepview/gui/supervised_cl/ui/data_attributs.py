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
    QMessageBox, QSpinBox
)

from PySide6.QtCore import (
    QObject, Signal, Slot, QTimer, Qt
)


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
from deepview.gui.supervised_cl.ui import styles

class DataAttributesWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # root_cfg = auxiliaryfunctions.read_config(self.root.config)
        self.sensor_dict = self.main_window.root_cfg['sensor_dict']


        self.augmentation_list = self.main_window.select_parameters_widget.augmentation_list
        self.augmentation_CLR_choice = self.augmentation_list[0]
        self.augmentation_SimCLR_choice = self.augmentation_list[0]

        self.methods = self.main_window.select_parameters_widget.methods
        self.method = self.methods[0]

        self.max_iter = 10
        self.learning_rate = 0.0001
        self.batch_size = 512



        # Create a vertical layout
        layout = QVBoxLayout(self)
        
        # 第一行tation组合框
        self.first_layout = QHBoxLayout()
        first_nested_layout = QHBoxLayout()
        first_nested_layout.setAlignment(Qt.AlignLeft)

        augmentationComboBoxLabel = QLabel('Select augmentation:')
        # 创建CLR组合框
        self.augmentationComboBox_CLR = QComboBox()
        self.augmentationComboBox_CLR.addItems(self.augmentation_list)
        # 默认选择第11个
        self.augmentationComboBox_CLR.setCurrentIndex(10)

        # 连接组合框文本改变事件到主窗口的方法
        self.augmentationComboBox_CLR.currentTextChanged.connect(
            self.log_augmentation_CLR_choice
        )

        # 创建SimCLR组合框
        self.augmentationComboBox_SimCLR = QComboBox()
        self.augmentationComboBox_SimCLR.addItems(self.augmentation_list)
        # 默认选择第3个
        self.augmentationComboBox_SimCLR.setCurrentIndex(2)

        self.augmentationComboBox_SimCLR.currentTextChanged.connect(
            self.log_augmentation_SimCLR_choice
        )

        first_nested_layout.addWidget(augmentationComboBoxLabel)
        first_nested_layout.addWidget(self.augmentationComboBox_CLR)
        first_nested_layout.addWidget(self.augmentationComboBox_SimCLR)

        self.first_layout.addLayout(first_nested_layout)


        # 第二行Parameters
        self.second_layout = QHBoxLayout()

        nested_layout = QHBoxLayout()
        nested_layout.setAlignment(Qt.AlignLeft)

        # ParametersLabel
        # parametersLabel = QLabel("Parameters:")

        method_label = QLabel("method:")
        self.display_method_type = QComboBox()
        self.display_method_type.addItems(self.methods)
        self.display_method_type.currentIndexChanged.connect(self.log_method_choice)

        # Display iterations
        dispiters_label = QLabel("Maximum iterations")
        self.display_iters_spin = QSpinBox()
        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(10000)
        # self.display_iters_spin.setMaximumWidth(10)
        self.display_iters_spin.setValue(30)
        self.display_iters_spin.valueChanged.connect(self.log_display_iters)

        # Save iterations
        saveiters_label = QLabel("Learning rate")
        self.save_iters_spin = QLineEdit()
        # self.save_iters_spin.setFixedWidth(2)
        self.save_iters_spin.setFixedWidth(100)
        # self.save_iters_spin.setMaximumWidth(10)
        self.save_iters_spin.setText("0.0001")
        self.save_iters_spin.textChanged.connect(self.log_init_lr)

        # Max iterations
        maxiters_label = QLabel("Batch size")
        self.batchsize_spin = QSpinBox()
        self.batchsize_spin.setMinimum(1)
        self.batchsize_spin.setMaximum(10000)
        # self.batchsize_spin.setMaximumWidth(10)
        self.batchsize_spin.setValue(1028)
        self.batchsize_spin.valueChanged.connect(self.log_batch_size)


        # nested_layout.addWidget(parametersLabel)
        nested_layout.addWidget(method_label)
        nested_layout.addWidget(self.display_method_type, alignment=Qt.AlignLeft)
        nested_layout.addWidget(dispiters_label)
        nested_layout.addWidget(self.display_iters_spin, alignment=Qt.AlignLeft)
        nested_layout.addWidget(saveiters_label)
        nested_layout.addWidget(self.save_iters_spin, alignment=Qt.AlignLeft)
        nested_layout.addWidget(maxiters_label)
        nested_layout.addWidget(self.batchsize_spin, alignment=Qt.AlignLeft)

        self.second_layout.addLayout(nested_layout)

        # 添加布局到主布局
        layout.addLayout(self.first_layout)
        layout.addLayout(self.second_layout)

        # ---------
        layout.addWidget(_create_label_widget("Data Attributes", "font:bold"))
        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        layout.addLayout(self.dataset_attributes_dataset)
        # ---------

        
        # 第四行 ApplySCL 按钮
        self.third_layout = QHBoxLayout()

        # self.applySCL_button = QPushButton('Apply SCL')
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignLeft)
        # 设置按钮宽度
        # self.applySCL_button.setFixedWidth(160)
        # self.applySCL_button.setStyleSheet(styles.button_style)

        # 保存模型按钮
        self.save_model_button = QPushButton('Save model')

        # 设置按钮宽度
        self.save_model_button.setFixedWidth(160)
        self.save_model_button.setStyleSheet(styles.button_style)

        # TODO 连接按钮点击事件到保存模型的方法
        # self.save_model_button.clicked.connect(self.save_model)

        # button_layout.addWidget(self.applySCL_button)
        button_layout.addWidget(self.save_model_button)

        self.third_layout.addLayout(button_layout)
        # self.third_layout.addWidget(self.applySCL_button)


        layout.addLayout(self.third_layout)

        
        self.setLayout(layout)




        

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
        if os.path.exists(os.path.join(self.main_window.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                    os.path.join(self.main_window.root.project_folder, trainingsetfolder),
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


    def log_augmentation_CLR_choice(self, augmentation):
        self.main_window.root.logger.info(f"CLR augmentation set to {self.augmentationComboBox_CLR.currentText()}")
        print(self.augmentationComboBox_CLR.currentText())
        self.augmentation_CLR_choice = self.augmentationComboBox_CLR.currentText()

    def log_augmentation_SimCLR_choice(self, augmentation):
        self.main_window.root.logger.info(f"SimCLR augmentation set to {self.augmentationComboBox_SimCLR.currentText()}")
        print(self.augmentationComboBox_SimCLR.currentText())
        self.augmentation_SimCLR_choice = self.augmentationComboBox_SimCLR.currentText()

    def log_method_choice(self, method):
        self.main_window.root.logger.info(f"Method set to {self.display_method_type.currentText()}")
        print(self.display_method_type.currentText())
        self.method = self.display_method_type.currentText()


    def log_display_iters(self, value):
        self.main_window.root.logger.info(f"Run iters (epochs) set to {value}")
        print(int(value))
        self.max_iter = int(value)

    def log_init_lr(self, value):
        self.main_window.root.logger.info(f"Learning rate set to {value}")
        print(float(value))
        self.learning_rate = float(value)

    def log_batch_size(self, value):
        self.main_window.root.logger.info(f"Batch size set to {value}")
        print(int(value))
        self.batch_size = int(value)



    def train_network(self):
        newSelectFilename = []
        for cb in self.display_dataset_cb_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        select_filenames = newSelectFilename