# This Python file uses the following encoding: utf-8
import ast
import os
# import warnings

# import matplotlib.colors as mcolors
# import napari  # image viewer for Python
# import numpy as np
# import pandas as pd
# from matplotlib.collections import LineCollection
# from matplotlib.path import Path
# from matplotlib.backends.backend_qt5agg import (
#     NavigationToolbar2QT,
#     FigureCanvasQTAgg as FigureCanvas,
# )
# from matplotlib.figure import Figure
# from matplotlib.widgets import RectangleSelector, Button, LassoSelector
from queue import Queue
from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QStandardItemModel, QStandardItem, QCursor, QAction
# from scipy.spatial import cKDTree as KDTree
# from skimage import io  # scikit-image is a collection of algorithms for image processing

from deepview.utils import auxiliaryfunctions

#--------------own packages------------------
# from deeplabcut.utils import auxiliaryfunctions
# from deeplabcut.utils.auxfun_videos import VideoWriter



class StreamReceiver(QtCore.QThread):
    new_text = QtCore.Signal(str)

    def __init__(self, queue):
        super(StreamReceiver, self).__init__()
        self.queue = queue

    def run(self):
        while True:
            text = self.queue.get()
            self.new_text.emit(text)


class StreamWriter:
    def __init__(self):
        self.queue = Queue()

    def write(self, text):
        if text != "\n":
            self.queue.put(text)

    def flush(self):
        pass


class ClickableLabel(QtWidgets.QLabel):
    signal = QtCore.Signal()

    def __init__(self, text="", color="turquoise", parent=None):
        super(ClickableLabel, self).__init__(text, parent)
        self._default_style = self.styleSheet()
        self.color = color
        self.setStyleSheet(f"color: {self.color}")

    def mouseReleaseEvent(self, event):
        self.signal.emit()

    def enterEvent(self, event):
        self.setCursor(QCursor(QtCore.Qt.PointingHandCursor))
        self.setStyleSheet(f"color: {self.color}")

    def leaveEvent(self, event):
        self.unsetCursor()
        self.setStyleSheet(self._default_style)


#  to be checked...
class DragDropListView(QtWidgets.QListView):
    def __init__(self, parent=None):
        super(DragDropListView, self).__init__(parent)
        self.parent = parent
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.model = QStandardItemModel(self)
        self.setModel(self.model)
        self._default_style = self.styleSheet()

    @property
    def items(self):
        for i in range(self.model.rowCount()):
            yield self.model.item(i)

    @property
    def state(self):
        tests = [item.checkState() == QtCore.Qt.Checked for item in self.items]
        n_checked = sum(tests)
        if all(tests):
            state = QtCore.Qt.Checked
        elif any(tests):
            state = QtCore.Qt.PartiallyChecked
        else:
            state = QtCore.Qt.Unchecked
        return state, n_checked

    def add_item(self, path):
        item = QStandardItem(path)
        item.setCheckable(True)
        item.setCheckState(QtCore.Qt.Checked)
        self.model.appendRow(item)

    def clear(self):
        self.model.removeRows(0, self.model.rowCount())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                self.add_item(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if not file.startswith("."):
                            self.add_item(os.path.join(root, file))


class ItemSelectionFrame(QtWidgets.QFrame):
    def __init__(self, items, parent=None):
        super(ItemSelectionFrame, self).__init__(parent)
        self.setFrameShape(self.Shape.StyledPanel)
        self.setLineWidth(0)

        self.select_box = QtWidgets.QCheckBox("Files")
        self.select_box.setChecked(True)
        self.select_box.stateChanged.connect(self.toggle_select)

        self.fancy_list = DragDropListView(self)
        self._model = self.fancy_list.model
        self._model.rowsInserted.connect(self.check_select_box)
        self._model.rowsRemoved.connect(self.check_select_box)
        self._model.itemChanged.connect(self.check_select_box)
        for item in items:
            self.fancy_list.add_item(item)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.select_box)
        self.layout.addWidget(self.fancy_list)

    @property
    def selected_items(self):
        for item in self.fancy_list.items:
            if item.checkState() == QtCore.Qt.Checked:
                yield item.text()

    def check_select_box(self):
        state, n_checked = self.fancy_list.state
        if self.select_box.checkState() != state:
            self.select_box.blockSignals(True)
            self.select_box.setCheckState(state)
            self.select_box.blockSignals(False)
        string = "file"
        if n_checked > 1:
            string += "s"
        self.select_box.setText(f"{n_checked} {string} selected")

    def toggle_select(self, state):
        state = QtCore.Qt.CheckState(state)
        if state == QtCore.Qt.PartiallyChecked:
            return
        for item in self.fancy_list.items:
            if item.checkState() != state:
                item.setCheckState(state)


class CustomDelegate(QtWidgets.QItemDelegate):
    # Hack to make the first column read-only, as we do not want users to touch it.
    # The cleaner solution would be to use a QTreeView and QAbstractItemModel,
    # but that is a lot of rework for little benefits.
    def createEditor(self, parent, option, index):
        if index.column() != 0:
            return super(CustomDelegate, self).createEditor(parent, option, index)
        return None


# TODO Insert new video
# TODO Insert skeleton link
class ItemCreator(QtWidgets.QDialog):
    created = QtCore.Signal(QtWidgets.QTreeWidgetItem)

    def __init__(self, parent=None):
        super(ItemCreator, self).__init__(parent)
        self.parent = parent
        vbox = QtWidgets.QVBoxLayout(self)
        self.field1 = QtWidgets.QLineEdit(self)
        self.field1.setPlaceholderText("Parameter")
        self.field2 = QtWidgets.QLineEdit(self)
        self.field2.setPlaceholderText("Value")
        create_button = QtWidgets.QPushButton(self)
        create_button.setText("Create")
        create_button.clicked.connect(self.form_item)
        vbox.addWidget(self.field1)
        vbox.addWidget(self.field2)
        vbox.addWidget(create_button)
        self.show()

    def form_item(self):
        key = self.field1.text()
        value = self.field2.text()
        item = QtWidgets.QTreeWidgetItem([key, value])
        item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        self.created.emit(item)
        self.accept()

class ContextMenu(QtWidgets.QMenu):
    def __init__(self, parent):
        super(ContextMenu, self).__init__(parent)
        self.parent = parent
        self.current_item = parent.tree.currentItem()
        insert = QAction("Insert", self)
        insert.triggered.connect(self.create_item)
        delete = QAction("Delete", self)
        delete.triggered.connect(parent.remove_items)
        self.addAction(insert)
        self.addAction(delete)
        if self.current_item.text(0) == "project_path":
            fix_path = QAction("Fix Path", self)
            fix_path.triggered.connect(self.fix_path)
            self.addAction(fix_path)

    def create_item(self):
        creator = ItemCreator(self)
        creator.created.connect(self.parent.insert)

    def fix_path(self):
        self.current_item.setText(1, os.path.split(self.parent.filename)[0])


class DictViewer(QtWidgets.QWidget):
    def __init__(self, cfg, filename="", parent=None):
        super(DictViewer, self).__init__(parent)
        self.cfg = cfg
        self.filename = filename
        self.parent = parent
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setItemDelegate(CustomDelegate())
        self.tree.setHeaderLabels(["Parameter", "Value"])
        self.tree.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSortingEnabled(False)
        self.tree.setHeaderHidden(False)
        self.tree.itemChanged.connect(self.edit_value)
        self.tree.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.pop_context_menu)

        self.root = self.tree.invisibleRootItem()
        self.tree.addTopLevelItem(self.root)
        self.populate_tree(cfg, self.root)

        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.tree)
        layout2 = QtWidgets.QVBoxLayout()
        layout2.addWidget(QtWidgets.QLabel(filename))
        layout2.addWidget(self.tree)
        self.setLayout(layout2)

    def pop_context_menu(self, point):
        index = self.tree.indexAt(point)
        if not index.isValid():
            return
        menu = ContextMenu(self)
        menu.exec_(self.tree.mapToGlobal(point))

    def get_position_in_parent(self, item):
        parent = item.parent() or self.root
        index = parent.indexOfChild(item)
        return index, parent

    def insert(self, item):
        current = self.tree.selectedItems()[0]
        ind, parent = self.get_position_in_parent(current)
        parent.insertChild(ind + 1, item)

        value = self.cast_to_right_type(item.text(1))
        if parent is self.root:
            self.set_value(self.cfg, [item.text(0)], value)
        else:
            keys, _ = self.walk_recursively_to_root(current)
            self.set_value(self.cfg, keys, value, ind + 1)

    def remove(self, item):
        ind, parent = self.get_position_in_parent(item)
        keys, value = self.walk_recursively_to_root(item)
        if item.parent() and item.childCount():  # Handle nested dict or list
            keys = [keys[0], value]
        success = self.remove_key(self.cfg, keys, ind)
        if success:
            parent.removeChild(item)

    def remove_items(self):
        for item in self.tree.selectedItems():
            self.remove(item)

    @staticmethod
    def cast_to_right_type(val):
        try:
            val = ast.literal_eval(val)
        except ValueError:
            # Leave untouched when it is already a string
            pass
        except SyntaxError:
            # Slashes also raise the error, but no need to print anything since it is then likely to be a path
            if os.path.sep not in val:
                print("Consider removing leading zeros or spaces in the string.")
        return val

    @staticmethod
    def walk_recursively_to_root(item):
        vals = []
        # Walk backwards across parents to get all keys
        while item is not None:
            for i in range(item.columnCount() - 1, -1, -1):
                vals.append(item.text(i))
            item = item.parent()
        *keys, value = vals[::-1]
        return keys, value

    @staticmethod
    def get_nested_key(cfg, keys):
        temp = cfg
        for key in keys[:-1]:
            try:
                temp = temp.setdefault(key, {})
            except AttributeError:  # Handle nested lists
                temp = temp[int(key)]
        return temp

    def edit_value(self, item):
        keys, value = self.walk_recursively_to_root(item)
        if (
            "crop" not in keys
        ):  # 'crop' should not be cast, otherwise it is understood as a list
            value = self.cast_to_right_type(value)
        self.set_value(self.cfg, keys, value)

    def set_value(self, cfg, keys, value, ind=None):
        temp = self.get_nested_key(cfg, keys)
        try:  # Work for a dict
            temp[keys[-1]] = value
        except TypeError:  # Needed to index a list
            if ind is None:  # Edit the list in place
                temp[self.tree.currentIndex().row()] = value
            else:
                temp.insert(ind, value)

    def remove_key(self, cfg, keys, ind=None):
        if not len(keys):  # Avoid deleting a parent list or dict
            return
        temp = self.get_nested_key(cfg, keys)
        try:
            temp.pop(keys[-1])
        except TypeError:
            if ind is None:
                ind = self.tree.currentIndex().row()
            temp.pop(ind)
        return True

    def populate_tree(self, data, tree_widget):
        if isinstance(data, dict):
            for key, val in data.items():
                self.add_row(key, val, tree_widget)
        elif isinstance(data, list):
            for i, val in enumerate(data):
                self.add_row(str(i), val, tree_widget)
        else:
            print("This should never be reached!")

    def add_row(self, key, val, tree_widget):
        if isinstance(val, dict) or isinstance(val, list):
            item = QtWidgets.QTreeWidgetItem([key])
            self.populate_tree(val, item)
        else:
            item = QtWidgets.QTreeWidgetItem([key, str(val)])
            item.setFlags(item.flags() | QtCore.Qt.ItemIsEditable)
        tree_widget.addChild(item)


class ConfigEditor(QtWidgets.QDialog):
    def __init__(self, config, parent=None):
        super(ConfigEditor, self).__init__(parent)
        self.config = config
        if config.endswith("config.yaml"):
            self.read_func = auxiliaryfunctions.read_config
            self.write_func = auxiliaryfunctions.write_config
        else:
            self.read_func = auxiliaryfunctions.read_plainconfig
            self.write_func = auxiliaryfunctions.write_plainconfig
        self.cfg = self.read_func(config)
        self.parent = parent
        self.setWindowTitle("Configuration Editor")
        if parent is not None:
            self.setMinimumWidth(parent.screen_width // 2)
            self.setMinimumHeight(parent.screen_height // 2)
        self.viewer = DictViewer(self.cfg, config, self)

        self.save_button = QtWidgets.QPushButton("Save", self)
        self.save_button.setDefault(True)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.close)

        vbox = QtWidgets.QVBoxLayout(self)
        vbox.addWidget(self.viewer)
        hbox = QtWidgets.QHBoxLayout()
        hbox.addWidget(self.save_button)
        hbox.addWidget(self.cancel_button)
        vbox.addLayout(hbox)

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def accept(self):
        self.write_func(self.config, self.cfg)
        super(ConfigEditor, self).accept()
