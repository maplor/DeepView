
from PySide6 import QtWidgets
from PySide6.QtCore import Qt
# from PySide6.QtWidgets import QPushButton, QFileDialog, QLineEdit


from deepview.gui.components import (
    DefaultTab,
    # TestfileSpinBox,
    # _create_horizontal_layout,
    _create_label_widget,
    # _create_vertical_layout,
)

# from mad_gui_main.mad_gui import start_gui
# from mad_gui import start_gui
# from mad_gui.plugins import ExampleImporter
# from myalgorithm import MyAlgorithm # you need to create this file and class, see below
# from deepview.mad_labeling.myimporter import CustomImporter
# from deepview.mad_labeling.myexporter import CustomExporter


class LabelData(DefaultTab):
    def __init__(self, root, parent, h1_description):
        super(LabelData, self).__init__(root, parent, h1_description)

        # self.bodyparts_to_use = self.root.all_bodyparts

        self._set_page()

    def _set_page(self):
        self.main_layout.addWidget(_create_label_widget("Mad-GUI", "font:bold"))

        self.ev_nw_button = QtWidgets.QPushButton("Evaluate Network")
        self.ev_nw_button.setMinimumWidth(150)
        self.ev_nw_button.clicked.connect(self.evaluate_network)  # 实现bottun

        self.main_layout.addWidget(self.ev_nw_button, alignment=Qt.AlignRight)


    # def evaluate_network(self):
    #     start_gui(plugins=[CustomImporter, CustomExporter])
    #     return

