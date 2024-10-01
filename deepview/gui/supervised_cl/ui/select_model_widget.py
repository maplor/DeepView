import os
from pathlib import Path
import pickle
import pandas as pd

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

# from deepview.gui.supervised_contrastive_learning import styles

from deepview.utils import auxiliaryfunctions
from deepview.gui.supervised_cl.ui import styles


class SelectModelWidget(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window  # 保存对主界面的引用

        # 访问主窗口的数据
        # current_data = self.main_window.data

        # 创建垂直布局
        layout = QVBoxLayout()

        # 第一行布局,包含Select model标签和选择框，, alignment=Qt.AlignLeft
        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        self.first_row1_layout = QHBoxLayout()
        self.first_row1_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        self.first_row1_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.first_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # 第二行布局
        # 创建原始数据组合框和标签
        # RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # self.second_row1_layout = QHBoxLayout()
        # self.second_row1_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        self.second_row1_layout = QVBoxLayout()
        # ---------
        # self.second_row1_layout.addWidget(_create_label_widget("Data Attributes", "font:bold"))
        self.dataset_attributes_dataset = _create_grid_layout(margins=(0, 0, 0, 0))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.second_row1_layout.addLayout(self.dataset_attributes_dataset)
        # ---------
        # self.second_row1_layout.addStretch()



        self.third_row1_layout = QHBoxLayout()
        self.display_button = QPushButton('Data display')
        # 设置按钮宽度
        self.display_button.setFixedWidth(160)
        self.display_button.setStyleSheet(styles.button_style)
        self.third_row1_layout.addWidget(self.display_button, alignment=Qt.AlignLeft)

        layout.addLayout(self.first_row1_layout)
        layout.addLayout(self.second_row1_layout)
        layout.addLayout(self.third_row1_layout)

        self.setLayout(layout)



    def train_network(self):
        file_name = self.RawDatacomboBox.currentText()
        model_name = self.modelComboBox.currentText()

        model_path = self.model_path
        model_name = self.model_name
        data_length = self.data_length
        column_names = self.column_names
        

        



    # 创建原始数据组合框的方法
    def createRawDataComboBox(self):
        # find data at here:C:\Users\dell\Desktop\aa-bbb-2024-04-28\unsupervised-datasets\allDataSet
        # 创建标签
        RawDataComboBoxLabel = QLabel('Select data:')

        # 创建组合框
        RawDatacomboBox = QComboBox()
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 获取所有.pkl文件路径
        rawdata_file_path_list = list(
            Path(os.path.join(self.main_window.cfg["project_path"], unsup_data_path)).glob('*.pkl'),
        )
        # 遍历路径列表
        for path in rawdata_file_path_list:
            # 将文件名添加到组合框
            RawDatacomboBox.addItem(str(path.name))

        # 初始选择的数据参数设置
        initial_data = RawDatacomboBox.currentText()
        self.get_data_from_pkl(initial_data)

        # 连接组合框文本改变事件到主窗口的方法
        RawDatacomboBox.currentTextChanged.connect(
            self.get_data_from_pkl
        )
        # 保存组合框
        self.RawDatacomboBox = RawDatacomboBox

        # 返回标签和组合框
        return RawDataComboBoxLabel, RawDatacomboBox




    # 创建模型组合框的方法
    def createModelComboBox(self):
        # 创建标签
        modelComboBoxLabel = QLabel('Select model:')

        # 创建组合框
        modelComboBox = QComboBox()
        # 从deepview.utils导入辅助函数
        # from deepview.utils import auxiliaryfunctions
        # Read file path for pose_config file. >> pass it on
        # 获取根对象配置
        config = self.main_window.root.config
        # 读取配置
        cfg = read_config(config)
        # 获取无监督模型文件夹路径
        unsup_model_path = get_unsup_model_folder(cfg)

        # 获取所有.pth文件路径
        model_path_list = grab_files_in_folder_deep(
            os.path.join(self.main_window.cfg["project_path"], unsup_model_path),
            ext='*.pth')
        # 保存模型路径列表到主窗口
        self.main_window.model_path_list = model_path_list

        if model_path_list:
            for path in model_path_list:
                modelComboBox.addItem(str(Path(path).name))

            # 初始选择的模型参数设置
            initial_model = modelComboBox.currentText()
            # self.main_window.get_model_param_from_path(initial_model)
            self.get_model_param_from_path(initial_model)

        # 连接组合框文本改变事件到主窗口的方法
        modelComboBox.currentTextChanged.connect(
            # self.main_window.get_model_param_from_path
            self.get_model_param_from_path
        )

        # 保存组合框
        self.modelComboBox = modelComboBox

        return modelComboBoxLabel, modelComboBox
    

    # 从.pkl文件获取数据的方法
    def get_data_from_pkl(self, filename):
        self.data_name = filename
        # 获取无监督数据集文件夹路径
        unsup_data_path = get_unsupervised_set_folder()
        # 构建文件路径
        self.data_path = os.path.join(self.main_window.cfg["project_path"], unsup_data_path, filename)

        self.main_window.data_name = self.data_name
        self.main_window.data_path = self.data_path

        return



    def get_model_param_from_path(self, model_path):
        # 从路径获取模型参数的方法
        if model_path:
            model_name, data_length, column_names = get_param_from_path(model_path)
            # 保存到主窗口的属性
            self.model_path = model_path
            self.model_name = model_name
            self.data_length = data_length
            self.column_names = column_names

            self.main_window.model_path = model_path
            self.main_window.model_name = model_name
            self.main_window.data_length = data_length
            self.main_window.column_names = column_names



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


