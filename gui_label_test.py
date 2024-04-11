import sys
import pandas as pd
import pyqtgraph as pg
from functools import partial
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QDialog,
    QVBoxLayout,
    QRadioButton
)

class LabelOption(QDialog):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout()

        # 创建单选框
        self.radio_button1 = QRadioButton("move")
        self.radio_button2 = QRadioButton("stand")

        layout.addWidget(self.radio_button1)
        layout.addWidget(self.radio_button2)

        # 创建确认按钮
        self.confirm_button = QPushButton("Confirm")
        self.confirm_button.clicked.connect(self.confirm_selection)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def confirm_selection(self):
        if self.radio_button1.isChecked():
            selected_option = "move"
        elif self.radio_button2.isChecked():
            selected_option = "stand"
        else:
            selected_option = None

        # 关闭窗口并返回选中的值
        self.accept()

        # 可以通过返回值或者信号来传递选中的值
        return selected_option

class InteractivePlot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Plot")
        self.resize(800, 600)

        # 创建图形窗口
        self.plot_widget = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
        self.setCentralWidget(self.plot_widget)

        # 读取CSV文件中的数据
        self.data = pd.read_csv("Umineko2018_small_data_LB07_lb0001.csv")
        self.data['datetime'] = pd.to_datetime(self.data['timestamp']).apply(lambda x: x.timestamp())
        self.plot_data()

        self.add_buttons()
        self.mode = 'add'

        self.click_begin = True
        self.start_line = None
        self.end_line = None
        self.regions = []

    def add_buttons(self):
        self.add_mode = QPushButton("Add", self)
        self.add_mode.setGeometry(40, 10, 100, 30)
        self.add_mode.clicked.connect(partial(self._change_mode, "add"))

        self.edit_mode = QPushButton("Edit", self)
        self.edit_mode.setGeometry(160, 10, 100, 30)
        self.edit_mode.clicked.connect(partial(self._change_mode, "edit"))

        self.del_mode = QPushButton("Delete", self)
        self.del_mode.setGeometry(280, 10, 100, 30)
        self.del_mode.clicked.connect(partial(self._change_mode, "del"))

    def _change_mode(self, mode: str):
        print(f'Change mode to "{mode}"')
        self.mode = mode

    def plot_data(self):
        # 将数据拆分为时间戳和值
        timestamps = self.data['datetime']
        values = self.data['acc_x']

        # 绘制数据
        self.plot_widget.plot(timestamps, values, pen='b')

        # 添加交互功能
        self.plot_widget.scene().sigMouseClicked.connect(self.mouse_clicked)
        self.plot_widget.scene().sigMouseMoved.connect(self.mouse_moved)

    def _to_idx(self, start_ts, end_ts):
        selected_indices = self.data[(self.data['datetime'] >= start_ts)
                                     & (self.data['datetime'] <= end_ts)].index
        return selected_indices.values[0], selected_indices.values[-1]

    def _add_region(self, pos):
        if self.click_begin:    # 记录开始位置
            self.click_begin = False
            self.start_line = pg.InfiniteLine(pos.x(), angle=90, movable=False)
            self.end_line = pg.InfiniteLine(pos.x(), angle=90, movable=False)
            self.plot_widget.addItem(self.start_line)
            self.plot_widget.addItem(self.end_line)
        else:                   # 记录结束位置
            self.click_begin = True
            self.plot_widget.removeItem(self.start_line)
            self.plot_widget.removeItem(self.end_line)

            region = pg.LinearRegionItem([self.start_line.value(), self.end_line.value()], brush=(0, 0, 255, 100))

            self.start_line = None
            self.end_line = None

            self.plot_widget.addItem(region)
            self.regions.append(region)

            start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
            print(f'Selected range: from index {start_idx} to index {end_idx}')

    def _del_region(self, pos):
        for item in self.regions:
            if item.getRegion()[0] < pos.x() and item.getRegion()[1] > pos.x():
                self.plot_widget.removeItem(item)
                self.regions.remove(item)

                start_idx, end_idx = self._to_idx(int(item.getRegion()[0]), int(item.getRegion()[1]))
                print(f'Delete region({start_idx}, {int(end_idx)})')
                break

    def _edit_region(self, pos):
        for item in self.regions:
            if item.getRegion()[0] < pos.x() and item.getRegion()[1] > pos.x():
                dialog = LabelOption()
                if dialog.exec() == QDialog.Accepted:
                    selected_option = dialog.confirm_selection()
                    if selected_option == 'move':
                        item.setBrush(QColor(255, 0, 0, 100))
                    elif selected_option == 'stand':
                        item.setBrush(QColor(0, 255, 0, 100))

                    start_idx, end_idx = self._to_idx(int(item.getRegion()[0]), int(item.getRegion()[1]))
                    print(f'Edit region({start_idx}, {end_idx}) label: {selected_option}')
                    break

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.plot_widget.plotItem.vb.mapSceneToView(event.pos()) # 将场景坐标映射到图形坐标系中

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widget.plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            self.end_line.setPos(pos.x())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractivePlot()
    window.show()
    sys.exit(app.exec())
