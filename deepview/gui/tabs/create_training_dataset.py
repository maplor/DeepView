

import os

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QProgressBar

from deepview.gui.components import (
    DefaultTab,
    _create_grid_layout,
    _create_label_widget,
)

from deepview.gui.dv_params import DVParams

import deepview

from deepview.utils.auxiliaryfunctions import (
    get_data_and_metadata_filenames,
    get_unsupervised_set_folder,
)


class CreateTrainingDataset(DefaultTab):
    # 定义进度信号
    progress_update = Signal(int)

    def __init__(self, root, parent, h1_description):
        super(CreateTrainingDataset, self).__init__(root, parent, h1_description)

        self.model_comparison = False

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        # set processing window
        self.setWindowTitle("Progress Demo")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.ok_button = QtWidgets.QPushButton("Create Training Dataset")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.create_training_dataset)

        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        # 连接信号和槽
        self.progress_update.connect(self.updateProgress)

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        # Neural Network
        nnet_label = QtWidgets.QLabel("Sampling rate (Hz)")
        self.samplerate_choice = QtWidgets.QLineEdit()
        self.samplerate_choice.setText("25")
        self.samplerate_choice.textChanged.connect(self.log_samplerate_choice(self.samplerate_choice.text()))

        validator = QtGui.QIntValidator()
        self.samplerate_choice.setValidator(validator)
        layout.addWidget(nnet_label, 0, 2)
        layout.addWidget(self.samplerate_choice, 0, 3)
        # layout.addWidget(augmentation_label, 0, 4)
        # layout.addWidget(self.aug_choice, 0, 5)

    # def log_net_choice(self, net):
    #     self.root.logger.info(f"Network architecture set to {net.upper()}")

    def log_samplerate_choice(self, boxtext):
        self.root.logger.info(f"Preprocess sensor data sampling rate to {boxtext}")

    def log_augmentation_choice(self, augmentation):
        self.root.logger.info(f"Image augmentation set to {augmentation.upper()}")

    def create_training_dataset(self):
        self.progress_bar.setValue(0)

        # 这是处理raw data的函数，包括生成folder，preprocess data和生成config
        # 三个入参为DeepView界面上的三个选项
        deepview.create_training_dataset(
            self.root,
            self.progress_update,
            self.root.config,
            sample_rate=self.samplerate_choice.text(),  # sampling rate (int)
            # augmenter_type=self.aug_choice.currentText(),  # augmentation (string)
        )
        # Check that training data files were indeed created.
        trainingsetfolder = get_unsupervised_set_folder()  # training-datasets/../..

        if os.path.exists(os.path.join(self.root.project_folder, trainingsetfolder)):
            # generate a pop-up window
            msg = _create_message_box(
                "The unsupervised dataset is successfully created.",
                "Use the function 'train_network' to start training. Happy training!",
            )
            msg.exec_()
            self.root.writer.write("Training dataset successfully created.")
        else:
            msg = _create_message_box(
                "The training dataset could not be created.",
                "Make sure there are annotated data under labeled-data.",
            )
            msg.exec_()
            self.root.writer.write("Training dataset creation failed.")


    def updateProgress(self, value):
        self.progress_bar.setValue(value)

#--------------message box setting-------------------------
def _create_message_box(text, info_text):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(text)
    msg.setInformativeText(info_text)

    msg.setWindowTitle("Info")
    msg.setMinimumWidth(900)
    logo_dir = os.path.dirname(os.path.realpath("logo.png")) + os.path.sep
    logo = logo_dir + "/assets/logo.png"
    msg.setWindowIcon(QIcon(logo))
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    return msg

