import os
from pathlib import Path
import pickle
import pandas as pd

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

# from deepview.gui.supervised_contrastive_learning import styles

from deepview.gui.supervised_cl.ui import styles
# from deepview.utils import auxiliaryfunctions
import numpy as np

class SelectParametersWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window  # 保存对主界面的引用

        # get sensor/columns dictionary from config.yaml
        # root_cfg = auxiliaryfunctions.read_config(self.main_window.config)

        self.sensor_dict = self.main_window.root_cfg['sensor_dict']

        self.repre_tsne, self.flag_concat, self.label_concat = None, None, None

        # clustering_pytorch/nnet/util.py
        self.augmentation_list = ['na', 'shuffle', 'jit_scale', 'perm_jit', 'resample', 'noise', 'scale', 'negate', 't_flip', 'perm', 't_warp']
        
        self.augmentation_CLR_choice = self.augmentation_list[0]
        self.augmentation_SimCLR_choice = self.augmentation_list[0]

        self.methods = ['SupCon', 'SimCLR']
        self.method = self.methods[0]
        self.max_iter = 10
        self.learning_rate = 0.0001
        self.batch_size = 512


        # 访问主窗口的数据
        # current_data = self.main_window.data

        # 创建垂直布局
        layout = QVBoxLayout()


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

        # self.first_layout.addWidget(augmentationComboBoxLabel)
        # self.first_layout.addWidget(self.augmentationComboBox_CLR)
        # self.first_layout.addWidget(self.augmentationComboBox_SimCLR)



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

        # self.second_layout.addWidget(parametersLabel, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(method_label, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(self.display_method_type, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(dispiters_label, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(self.display_iters_spin, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(saveiters_label, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(self.save_iters_spin, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(maxiters_label, alignment=Qt.AlignLeft)
        # self.second_layout.addWidget(self.batchsize_spin, alignment=Qt.AlignLeft)

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
        

        # 第四行 ApplySCL 按钮
        self.third_layout = QHBoxLayout()

        self.applySCL_button = QPushButton('Apply SCL')
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignLeft)
        # 设置按钮宽度
        self.applySCL_button.setFixedWidth(160)
        self.applySCL_button.setStyleSheet(styles.button_style)

        # 保存模型按钮
        self.save_model_button = QPushButton('Save model')

        # 设置按钮宽度
        self.save_model_button.setFixedWidth(160)
        self.save_model_button.setStyleSheet(styles.button_style)

        # TODO 连接按钮点击事件到保存模型的方法
        # self.save_model_button.clicked.connect(self.save_model)

        button_layout.addWidget(self.applySCL_button)
        button_layout.addWidget(self.save_model_button)

        self.third_layout.addLayout(button_layout)
        # self.third_layout.addWidget(self.applySCL_button)

        # 添加布局到主布局
        layout.addLayout(self.first_layout)
        layout.addLayout(self.second_layout)
        layout.addLayout(self.third_layout)

        self.setLayout(layout)


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

        config = self.main_window.root.config
        
        augmentation_CLR_choice = str(self.augmentation_CLR_choice.upper())
        augmentation_SimCLR_choice = str(self.augmentation_SimCLR_choice.upper())
        method = str(self.method.upper())
        max_iter = int(self.display_iters_spin.text())
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())


        # 从主界面获取选中的filename
        select_filenames = self.main_window.data_name
        file_path = self.main_window.data_path

        data_len, sensors = self.parse_model_filename()

        num_channel = self.calculate_num_channels(sensors, self.sensor_dict)
        


    # Parse the model filename to extract sensor types and data length，解析pkl文件名以提取传感器类型和数据长度
    def parse_model_filename(filename):
        basename = os.path.basename(filename)
        parts = basename.split('_')
        
        # Extract data_len from the filename
        data_len = int(parts[2].replace('datalen', ''))
        
        # Extract sensor types from the filename
        sensor_part = parts[3].replace('.pth', '')
        sensors = sensor_part.split('-')
        
        return data_len, sensors

    # Calculate the total number of columns based on sensors
    def calculate_num_channels(sensors, sensor_dict):
        num_columns = 0
        for sensor in sensors:
            if sensor in sensor_dict:
                num_columns += len(sensor_dict[sensor])
        return num_columns

