# This Python file uses the following encoding: utf-8
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

from PySide6 import QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QShowEvent
# from deeplabcut.gui.dlc_params import DLCParams
# from deeplabcut.gui.widgets import ConfigEditor

from PySide6.QtWebEngineWidgets import QWebEngineView


class DefaultTab(QtWidgets.QWidget):
    def __init__(
        self,
        root: QtWidgets.QMainWindow,
        parent: QtWidgets.QWidget = None,
        h1_description: str = "",
    ):
        super(DefaultTab, self).__init__(parent)

        self.parent = parent
        self.root = root

        self.h1_description = h1_description

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self.main_layout)

        self._init_default_layout()

        self.firstShow = False

    def showEvent(self, event: QShowEvent) -> None:
        if not self.firstShow:
            self.firstShowEvent(event)
            self.firstShow = True

        return super().showEvent(event)
    
    def firstShowEvent(self, event: QShowEvent) -> None:
        return

    def _init_default_layout(self):
        # Add tab header
        self.main_layout.addWidget(
            _create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10))
        )

        # Add separating line
        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Raised)
        self.separator.setLineWidth(0)
        self.separator.setMidLineWidth(1)
        policy = QtWidgets.QSizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.separator.setSizePolicy(policy)
        self.main_layout.addWidget(self.separator)

# 添加QWebEngineView以便调用
class DefaultWebTab(QtWidgets.QWidget):
    def __init__(
            self,
            root: QtWidgets.QMainWindow,
            parent: QtWidgets.QWidget = None,
            h1_description: str = "",
    ):
        super(DefaultWebTab, self).__init__(parent)
        # os.environ["QTWEBENGINE_REMOTE_DEBUGGING"] = "9024"
        self.web_view = QWebEngineView()
        self.web_view_map = QWebEngineView()


        self.parent = parent
        self.root = root

        self.h1_description = h1_description

        self.main_layout = QtWidgets.QVBoxLayout()
        self.main_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setLayout(self.main_layout)

        self._init_default_layout()

        self.firstShow = False

    def showEvent(self, event: QShowEvent) -> None:
        if not self.firstShow:
            self.firstShowEvent(event)
            self.firstShow = True

        return super().showEvent(event)

    def firstShowEvent(self, event: QShowEvent) -> None:
        return

    def _init_default_layout(self):
        # Add tab header
        self.main_layout.addWidget(
            _create_label_widget(self.h1_description, "font:bold;", (10, 10, 0, 10))
        )

        # Add separating line
        self.separator = QtWidgets.QFrame()
        self.separator.setFrameShape(QtWidgets.QFrame.HLine)
        self.separator.setFrameShadow(QtWidgets.QFrame.Raised)
        self.separator.setLineWidth(0)
        self.separator.setMidLineWidth(1)
        policy = QtWidgets.QSizePolicy()
        policy.setVerticalPolicy(QtWidgets.QSizePolicy.Policy.Fixed)
        policy.setHorizontalPolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        self.separator.setSizePolicy(policy)
        self.main_layout.addWidget(self.separator)



class TestfileSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(TestfileSpinBox, self).__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.testfile)
        self.valueChanged.connect(self.root.update_testfile)  # set logger


class TrainingSetSpinBox(QtWidgets.QSpinBox):
    def __init__(self, root, parent):
        super(TrainingSetSpinBox, self).__init__(parent)

        self.root = root
        self.parent = parent

        self.setMaximum(100)
        self.setValue(self.root.trainingset_index)
        self.valueChanged.connect(self.root.update_trainingset)


def _create_grid_layout(
    alignment=None,
    spacing: int = 20,
    margins: tuple = None,
) -> QtWidgets.QGridLayout():
    layout = QtWidgets.QGridLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    if margins:
        layout.setContentsMargins(*margins)

    return layout

def _create_label_widget(
    text: str,
    style: str = "",
    margins: tuple = (20, 10, 0, 10),
) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setContentsMargins(*margins)
    label.setStyleSheet(style)

    return label

def _create_horizontal_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QHBoxLayout():
    layout = QtWidgets.QHBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout


def _create_vertical_layout(
    alignment=None, spacing: int = 20, margins: tuple = (20, 0, 0, 0)
) -> QtWidgets.QVBoxLayout():
    layout = QtWidgets.QVBoxLayout()
    layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    layout.setSpacing(spacing)
    layout.setContentsMargins(*margins)

    return layout