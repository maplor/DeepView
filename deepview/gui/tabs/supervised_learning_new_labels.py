# This Python file uses the following encoding: utf-8

# after new labeled are created by biologists, use this code to evaluate supervised learning performance

#
import os
# import matplotlib.image as mpimg
# from matplotlib.backends.backend_qt5agg import (
#     FigureCanvasQTAgg as FigureCanvas,
# )
# from matplotlib.figure import Figure
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
# from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit
from deepview.utils import auxiliaryfunctions

from deepview.gui.components import (
    DefaultTab,
    # TestfileSpinBox,
    # _create_horizontal_layout,
    _create_label_widget,
    _create_grid_layout,
    # _create_vertical_layout,
)

from deepview.gui.widgets import ConfigEditor
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

    # 在第一次渲染 tab 时才构造内容
    def firstShowEvent(self, event: QShowEvent) -> None:
        self._set_page()


    def _set_page(self):

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.main_layout.addWidget(_create_label_widget(""))  # dummy label

        # ---------
        self.main_layout.addWidget(_create_label_widget("Select data", "font:bold"))
        # self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 0, 0, 0))
        # # self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        # self.main_layout.addLayout(self.dataset_attributes_dataset)



        self.opt_button = QtWidgets.QPushButton("Calculate result")
        self.opt_button.setMinimumWidth(150)
        self.opt_button.clicked.connect(self.supervise_calculate)  # 实现bottun

        self.dataset_attributes_dataset = _create_grid_layout(margins=(20, 5, 5, 5))
        self._generate_layout_attributes_dataset(self.dataset_attributes_dataset)
        self.main_layout.addLayout(self.dataset_attributes_dataset)

        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)

    def supervise_calculate(self):
        # calculation functions are in supervised_learning folder

        # load parameters
        # net_type = str(self.net_type.upper())
        net_type = self.display_net_type.currentText()
        learning_rate = float(self.save_iters_spin.text())
        batch_size = int(self.batchsize_spin.text())
        windowlen = int(self.save_datalen.text())
        max_iter = int(self.display_iters_spin.text())

        # 当前从“unsupervised-datasets”中读取pkl原始数据
        datapath = get_unsupervised_set_folder()
        files = auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, datapath),
                relative=False,
            )

        print("Calculate supervised learning result")
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
                                   net_type,
                                   learning_rate,
                                   batch_size,
                                   windowlen,
                                   max_iter,
                                   files,
                                   train_filenames,
                                   test_filenames)
        return

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        net_label = QtWidgets.QLabel("Network type")
        self.display_net_type = QtWidgets.QComboBox()
        self.display_net_type.addItems(['DeepConvLSTM'])
        self.display_net_type.currentIndexChanged.connect(self.log_net_choice)

        # Display iterations
        dispiters_label = QtWidgets.QLabel("Maximum iterations")
        self.display_iters_spin = QtWidgets.QSpinBox()
        self.display_iters_spin.setMinimum(1)
        self.display_iters_spin.setMaximum(10000)
        self.display_iters_spin.setValue(10)
        self.display_iters_spin.valueChanged.connect(self.log_display_iters)

        # Save iterations
        saveiters_label = QtWidgets.QLabel("Learning rate")
        self.save_iters_spin = QtWidgets.QLineEdit()
        self.save_iters_spin.setText('0.0005')
        self.save_iters_spin.textChanged.connect(self.log_init_lr)

        # input data length
        savedatalen_label = QtWidgets.QLabel("Input length")
        self.save_datalen = QtWidgets.QSpinBox()
        self.save_datalen.setValue(90)
        self.save_datalen.valueChanged.connect(self.log_datalen)

        # Max iterations
        maxiters_label = QtWidgets.QLabel("Batch size")
        self.batchsize_spin = QtWidgets.QSpinBox()
        self.batchsize_spin.setMinimum(1)
        self.batchsize_spin.setMaximum(10000)
        self.batchsize_spin.setValue(64)
        self.batchsize_spin.valueChanged.connect(self.log_batch_size)

        layout.addWidget(net_label, 0, 0)
        layout.addWidget(self.display_net_type, 0, 1)
        layout.addWidget(dispiters_label, 0, 2)
        layout.addWidget(self.display_iters_spin, 0, 3)
        layout.addWidget(saveiters_label, 0, 4)
        layout.addWidget(self.save_iters_spin, 0, 5)
        layout.addWidget(savedatalen_label, 0, 6)
        layout.addWidget(self.save_datalen, 0, 7)
        layout.addWidget(maxiters_label, 0, 8)
        layout.addWidget(self.batchsize_spin, 0, 9)
        return

    def _generate_layout_attributes_dataset(self, layout):

        trainingsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()

        select_train_label = QtWidgets.QLabel("Select training dataset file")

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scrollContent = QtWidgets.QWidget(scroll)
        grid = QtWidgets.QGridLayout(scrollContent)
        grid.setAlignment(Qt.AlignTop)
        scrollContent.setLayout(grid)
        scroll.setWidget(scrollContent)

        self.display_dataset_cb_list = []

        rowNum = 3

        self.display_train_dataset_container = QtWidgets.QHBoxLayout()
        self.display_dataset_train_list = []
        self.display_dataset_test_list = []

        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, trainingsetfolder),
                relative=False,
            ):
                cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                grid.addWidget(cb, len(self.display_dataset_train_list) // rowNum,
                               len(self.display_dataset_train_list) % rowNum)
                self.display_dataset_train_list.append(cb)

        editsetfolder = auxiliaryfunctions.get_edit_data_folder()
        if os.path.exists(os.path.join(self.root.project_folder, editsetfolder)):
            for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, editsetfolder),
                relative=False,
            ):
                cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
                grid.addWidget(cb, len(self.display_dataset_train_list) // rowNum,
                               len(self.display_dataset_train_list) % rowNum)
                # self.display_train_dataset_container.addWidget(cb)
                self.display_dataset_train_list.append(cb)

        # -------------------test set container---------------------
        scroll_test = QtWidgets.QScrollArea()
        scroll_test.setWidgetResizable(True)
        scrollContent = QtWidgets.QWidget(scroll_test)
        grid = QtWidgets.QGridLayout(scrollContent)
        grid.setAlignment(Qt.AlignTop)
        scrollContent.setLayout(grid)
        scroll_test.setWidget(scrollContent)

        valsetfolder = auxiliaryfunctions.get_unsupervised_set_folder()
        select_val_label = QtWidgets.QLabel("Select test dataset file")
        # select_val_label = QtWidgets.QLabel("Select validation dataset file")
        # self.display_dataset_cb = QtWidgets.QComboBox()
        self.display_val_dataset_container = QtWidgets.QHBoxLayout()
        for filename in auxiliaryfunctions.grab_files_in_folder(
                os.path.join(self.root.project_folder, valsetfolder),
                relative=False,
        ):

            cb = QtWidgets.QCheckBox(os.path.split(filename)[-1])
            grid.addWidget(cb, len(self.display_dataset_test_list) // rowNum,
                           len(self.display_dataset_test_list) % rowNum)
            self.display_dataset_test_list.append(cb)


        layout.addWidget(select_train_label, 0, 8)
        layout.addLayout(self.display_train_dataset_container, 0, 9)
        layout.addWidget(scroll, 0, 9)
        layout.addWidget(select_val_label, 1, 8)
        layout.addLayout(self.display_val_dataset_container, 1, 9)
        layout.addWidget(scroll_test, 1, 9)
        # layout.addWidget(select_test_label, 2, 0)
        # layout.addWidget(self.display_test_dataset_container, 2, 1)

    def log_net_choice(self, net):
        self.root.logger.info(f"Supervised Network type set to {net.upper()}")
        # self.root.logger.info(f"TODO: test dataset set to {net.upper()}")

    def log_display_iters(self, value):
        self.root.logger.info(f"Run Supervised iters (epochs) set to {value}")

    def log_init_lr(self, value):
        self.root.logger.info(f"Supervised Learning rate set to {value}")

    def log_batch_size(self, value):
        self.root.logger.info(f"Supervised Batch size set to {value}")

    def log_save_iters(self, value):
        self.root.logger.info(f"Save Supervised iters set to {value}")

    def log_max_iters(self, value):
        self.root.logger.info(f"Max Supervised iters set to {value}")

    def log_datalen(self, value):
        self.root.logger.info(f"Input data length of Supervised model set to {value}")