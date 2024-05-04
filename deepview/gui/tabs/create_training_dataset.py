

import os

from PySide6 import QtWidgets, QtGui
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

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
    def __init__(self, root, parent, h1_description):
        super(CreateTrainingDataset, self).__init__(root, parent, h1_description)

        self.model_comparison = False

        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))
        self.layout_attributes = _create_grid_layout(margins=(20, 0, 0, 0))
        self._generate_layout_attributes(self.layout_attributes)
        self.main_layout.addLayout(self.layout_attributes)

        self.ok_button = QtWidgets.QPushButton("Create Training Dataset")
        self.ok_button.setMinimumWidth(150)
        self.ok_button.clicked.connect(self.create_training_dataset)

        self.main_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

    def _generate_layout_attributes(self, layout):
        layout.setColumnMinimumWidth(3, 300)

        # Augmentation method
        augmentation_label = QtWidgets.QLabel("Normalization method(TODO)")
        self.aug_choice = QtWidgets.QComboBox()
        self.aug_choice.addItems(DVParams.IMAGE_AUGMENTERS)
        self.aug_choice.setCurrentText("imgaug")
        self.aug_choice.currentTextChanged.connect(self.log_augmentation_choice)

        # Neural Network
        nnet_label = QtWidgets.QLabel("Sampling rate (Hz)")
        self.samplerate_choice = QtWidgets.QLineEdit()
        # self.net_choice = QtWidgets.QComboBox()
        # nets = DVParams.NNETS.copy()  # a string list, string is model names
        # if not self.root.is_multianimal:  # not false = true
        #     nets.remove("dlcrnet_ms5")
        # self.net_choice.addItems(nets)
        self.samplerate_choice.setText("25")
        self.samplerate_choice.textChanged.connect(self.log_samplerate_choice(self.samplerate_choice.text()))

        # 添加验证器以确保只能输入整数
        validator = QtGui.QIntValidator()
        self.samplerate_choice.setValidator(validator)
        # layout.addWidget(shuffle_label, 0, 0)
        # layout.addWidget(self.shuffle, 0, 1)
        layout.addWidget(nnet_label, 0, 2)
        layout.addWidget(self.samplerate_choice, 0, 3)
        layout.addWidget(augmentation_label, 0, 4)
        layout.addWidget(self.aug_choice, 0, 5)

    # def log_net_choice(self, net):
    #     self.root.logger.info(f"Network architecture set to {net.upper()}")

    def log_samplerate_choice(self, boxtext):
        self.root.logger.info(f"Preprocess sensor data sampling rate to {boxtext}")

    def log_augmentation_choice(self, augmentation):
        self.root.logger.info(f"Image augmentation set to {augmentation.upper()}")

    def create_training_dataset(self):
        # 这是处理raw data的函数，包括生成folder，preprocess data和生成config
        # 三个入参为DeepView界面上的三个选项
        deepview.create_training_dataset(
            self.root.config,
            # shuffle,
            # Shuffles=[self.shuffle.value()],
            sample_rate=self.samplerate_choice.text(),  # sampling rate (int)
            augmenter_type=self.aug_choice.currentText(),  # augmentation (string)
        )
        # Check that training data files were indeed created.
        trainingsetfolder = get_unsupervised_set_folder()  # training-datasets/../..
        # filenames = list(
        #     get_data_and_metadata_filenames(
        #         trainingsetfolder,
        #         # self.root.cfg["TrainingFraction"][0],
        #         # self.shuffle.value(),
        #         self.root.cfg,
        #     )
        # )

        # if all(
        #         os.path.exists(os.path.join(self.root.project_folder, file))
        #         for file in filenames
        # ):
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

