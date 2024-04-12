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
    QRadioButton,
    QWidget,
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

        # 创建主窗口的布局
        self.layout = QVBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.setCentralWidget(central_widget)

        # 读取CSV文件中的数据
        self.data = pd.read_csv("Umineko2018_small_data_LB07_lb0001.csv")
        self.data['datetime'] = pd.to_datetime(self.data['timestamp']).apply(lambda x: x.timestamp())

        self.columnList = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
        self.selectColumn = ['acc_x', 'acc_y', 'acc_z']
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']

        # 绘制数据
        self.plot_widgets = []
        self.plot_data()

        self.add_buttons()
        self.mode = 'add'

        self.click_begin = True
        self.start_line = [None] * len(self.selectColumn)
        self.end_line = [None] * len(self.selectColumn)
        self.regions = []
        for _ in range(len(self.selectColumn)):
            self.regions.append([])

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
        timestamps = self.data['datetime']
        for i, col in enumerate(self.selectColumn):
            values = self.data[col]
            pwidget = pg.PlotWidget(axisItems={'bottom': pg.DateAxisItem()})
            pwidget.plot(timestamps, values, pen=self.colors[i])

            # 添加交互功能
            pwidget.scene().sigMouseClicked.connect(self.mouse_clicked)
            pwidget.scene().sigMouseMoved.connect(self.mouse_moved)
            self.layout.addWidget(pwidget)
            self.plot_widgets.append(pwidget)

    def _to_idx(self, start_ts, end_ts):
        selected_indices = self.data[(self.data['datetime'] >= start_ts)
                                     & (self.data['datetime'] <= end_ts)].index
        return selected_indices.values[0], selected_indices.values[-1]

    def _add_region(self, pos):
        if self.click_begin:    # 记录开始位置
            self.click_begin = False
            for i, pwidget in enumerate(self.plot_widgets):
                self.start_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                self.end_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                pwidget.addItem(self.start_line[i])
                pwidget.addItem(self.end_line[i])
        else:                   # 记录结束位置
            self.click_begin = True
            for i, pwidget in enumerate(self.plot_widgets):
                pwidget.removeItem(self.start_line[i])
                pwidget.removeItem(self.end_line[i])

                region = pg.LinearRegionItem([self.start_line[i].value(), self.end_line[i].value()], brush=(0, 0, 255, 100))
                region.sigRegionChanged.connect(self._region_changed)

                self.start_line[i] = None
                self.end_line[i] = None

                pwidget.addItem(region)
                self.regions[i].append(region)

            start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
            print(f'Selected range: from index {start_idx} to index {end_idx}')

    def _region_changed(self, region):
        idx = 0
        for reg_lst in self.regions:
            for i, reg in enumerate(reg_lst):
                if reg == region:
                    idx = i
                    break
        for reg_lst in self.regions:
            reg_lst[idx].setRegion(region.getRegion())

    def _del_region(self, pos):
        for i, pwidget in enumerate(self.plot_widgets):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    pwidget.removeItem(reg)
                    self.regions[i].remove(reg)
                    break
        start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
        print(f'Delete region({start_idx}, {int(end_idx)})')

    def _edit_region(self, pos):
        set_val = None
        for i, _ in enumerate(self.regions):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    if set_val is None:
                        dialog = LabelOption()
                        if dialog.exec() == QDialog.Accepted:
                            set_val = dialog.confirm_selection()
                        else:
                            set_val = 'None'
                    if set_val == 'move':
                        reg.setBrush(QColor(255, 0, 0, 100))
                    elif set_val == 'stand':
                        reg.setBrush(QColor(0, 255, 0, 100))
        start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
        print(f'Edit region({start_idx}, {end_idx}) label: {set_val}')

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event.pos()) # 将场景坐标映射到图形坐标系中

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            for line in self.end_line:
                line.setPos(pos.x())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InteractivePlot()
    window.show()
    sys.exit(app.exec())
