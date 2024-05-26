

# after new labeled are created by biologists, use this code to evaluate supervised learning performance

#
import os
import matplotlib.image as mpimg
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit
from deepview.utils import auxiliaryfunctions

from deepview.gui.components import (
    DefaultTab,
    TestfileSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_grid_layout,
    _create_vertical_layout,
)

# from deepview.gui.widgets import ConfigEditor
import deepview
# from deepview.utils import auxiliaryfunctions
# from deepview.utils.auxiliaryfunctions import get_evaluation_folder
from deepview.utils.auxiliaryfunctions import (
    get_unsupervised_set_folder,
)


'''
read all pkl files about processed sensor data
list+select training, evaluation, test datasets
start training
'''

class SupervisedLearningNewLabels(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(SupervisedLearningNewLabels, self).__init__(root, parent, h1_description)

        # self.bodyparts_to_use = self.root.all_bodyparts
        self.root = root

        # self._set_page()

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))

        self.opt_button = QtWidgets.QPushButton("Calulate result")
        self.opt_button.setMinimumWidth(150)
        self.opt_button.clicked.connect(self.supervise_calculate)  # 实现bottun

        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 5, 5, 5))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.main_layout.addLayout(self.dataset_attributes_dataset)

        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)

    def supervise_calculate(self):
        # calculation functions are in supervised_learning folder
        # todo: add parameters later

        # 当前从“unsupervised-datasets”中读取pkl原始数据
        # todo，从“training-datasets”中读取数据，这个数据在打标签时从“”copy并更新
        datapath = get_unsupervised_set_folder()
        files = auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, datapath),
                relative=False,
            )

        print("calulate supervised learning result")
        newSelectFilename = []
        for cb in self.display_dataset_train_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        train_filenames = newSelectFilename

        newSelectFilename = []
        for cb in self.display_dataset_test_list:
            if cb.isChecked():
                newSelectFilename.append(cb.text())
        test_filenames = newSelectFilename

        deepview.train_sup_network(self.root,
                                   files,
                                   train_filenames,
                                   test_filenames)
        return


    def _generate_layout_attributes_dataset(self, layout):
        # layout.setColumnMinimumWidth(3, 300)

        trainingsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
        select_train_label = QtWidgets.QLabel("Select training dataset file")

        scroll_1 = QtWidgets.QScrollArea()
        scroll_1.setWidgetResizable(True)
        scrollContent_1 = QtWidgets.QWidget(scroll_1)
        grid_1 = QtWidgets.QGridLayout(scrollContent_1)
        grid_1.setAlignment(Qt.AlignTop)
        scrollContent_1.setLayout(grid_1)
        scroll_1.setWidget(scrollContent_1)

        rowNum = 3

        self.display_dataset_train_list = []
        self.display_dataset_test_list = []

        # column_list = []

        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, trainingsetfolder),
                relative=False,
            ):
                cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                grid_1.addWidget(cb, len(self.display_dataset_train_list) // rowNum, len(self.display_dataset_train_list) % rowNum)
                self.display_dataset_train_list.append(cb)

        valsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
        select_val_label = QtWidgets.QLabel("Select test dataset file")
        
        scroll_2 = QtWidgets.QScrollArea()
        scroll_2.setWidgetResizable(True)
        scrollContent_2 = QtWidgets.QWidget(scroll_2)
        grid_2 = QtWidgets.QGridLayout(scrollContent_2)
        grid_2.setAlignment(Qt.AlignTop)
        scrollContent_2.setLayout(grid_2)
        scroll_2.setWidget(scrollContent_2)
        
        for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, valsetfolder),
                relative=False,
        ):
            cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
            grid_2.addWidget(cb, len(self.display_dataset_test_list) // rowNum, len(self.display_dataset_test_list) % rowNum)
            self.display_dataset_test_list.append(cb)

        testsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
        select_test_label = QtWidgets.QLabel("Select test dataset file")
        self.display_test_dataset_container = QtWidgets.QHBoxLayout()
        for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, testsetfolder),
                relative=False,
        ):
            cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
            self.display_test_dataset_container.addWidget(cb)

        layout.addWidget(select_train_label, 0, 0)
        layout.addWidget(scroll_1, 0, 1)
        layout.addWidget(select_val_label, 1, 0)
        layout.addWidget(scroll_2, 1, 1)
        # layout.addWidget(select_test_label, 2, 0)
        # layout.addWidget(self.display_test_dataset_container, 2, 1)

