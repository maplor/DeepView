

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
from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit


from deepview.gui.components import (
    DefaultTab,
    TestfileSpinBox,
    _create_horizontal_layout,
    _create_label_widget,
    _create_vertical_layout,
)
# from deepview.gui.widgets import ConfigEditor
import deepview
# from deepview.utils import auxiliaryfunctions
from deepview.utils.auxiliaryfunctions import get_evaluation_folder

'''
复选框选择训练数据
根据训练数据更新测试数据的复选框
下拉框选择模型
button:训练
'''

class SupervisedLearningNewLabels(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(SupervisedLearningNewLabels, self).__init__(root, parent, h1_description)

        # self.bodyparts_to_use = self.root.all_bodyparts

        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Attributes", "font:bold"))

        self.opt_button = QtWidgets.QPushButton("Calulate result")
        self.opt_button.setMinimumWidth(150)
        self.opt_button.clicked.connect(self.supervise_calculate)  # 实现bottun

        self.main_layout.addWidget(self.opt_button, alignment=Qt.AlignRight)


    def supervise_calculate(self):
        # calculation functions are in supervised_learning folder
        print("calulate supervised learning result")