import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QSplitter, QVBoxLayout

from deepview.gui.label_with_interactive_plot.widgets.combo import LabelOption


class LeftPlotMixin:
    # 创建左侧图表的方法
    def createLeftPlot(self):  # 创建左侧图表的方法
        viewL = QVBoxLayout()  # 创建一个垂直布局
        self.viewL = viewL  # 保存布局
        self.splitter = QSplitter(Qt.Vertical)  # 创建一个垂直分割器
        self.plot_widgets = [None] * len(self.column_names)  # 初始化图表小部件列表
        self.click_begin = True  # 初始化点击开始状态为True
        self.start_line = [None] * len(self.column_names)  # 初始化开始线列表
        self.end_line = [None] * len(self.column_names)  # 初始化结束线列表
        self.regions = []  # 初始化区域列表
        for _ in range(len(self.column_names)):  # 遍历列名列表
            self.regions.append([])  # 为每个列创建一个空的区域列表
        self.bottom_layout.addLayout(viewL, 2)  # 将布局添加到底部布局中

        self.resetLeftPlot()  # 重置左侧图表
        self.splitter.setSizes([100] * len(self.column_names))  # 设置分割器大小
        self.viewL.addWidget(self.splitter)  # 将分割器添加到布局中

    def resetLeftPlot(self):  # 重置左侧图表的方法
        # 重置小部件
        for i in range(len(self.column_names)):  # 遍历列名列表
            if self.plot_widgets[i] is not None:  # 如果图表小部件不为None
                self.splitter.removeWidget(self.plot_widgets[i])  # 从分割器中移除小部件
                self.plot_widgets[i].close()  # 关闭小部件
                self.plot_widgets[i] = None  # 设置小部件为None
        # 添加小部件
        for i, columns in enumerate(self.column_names):  # 遍历列名列表
            real_columns = self.sensor_dict[columns]  # 获取真实列名列表
            plot = pg.PlotWidget(title=columns, name=columns, axisItems={'bottom': pg.DateAxisItem()})  # 创建图表小部件
            for j, c in enumerate(real_columns):  # 遍历真实列名列表
                plot.plot(self.data['_timestamp'], self.data[c], pen=pg.mkPen(j))  # 绘制数据
            # plot.plot(self.data['datetime'], self.data[columns[0]], pen=pg.mkPen(i))
            plot.scene().sigMouseClicked.connect(self.mouse_clicked)  # 连接鼠标点击事件到mouse_clicked方法
            plot.scene().sigMouseMoved.connect(self.mouse_moved)  # 连接鼠标移动事件到mouse_moved方法
            self.plot_widgets[i] = plot  # 保存图表小部件
            self.splitter.addWidget(plot)  # 将图表小部件添加到分割器中

    def updateLeftPlotList(self):
        # 遍历每一列的名称
        for i, column in enumerate(self.column_names):
            # 显示每个绘图窗口
            self.plot_widgets[i].show()

    # def _to_idx(self, start_ts, end_ts):
    #     # 根据给定的时间戳范围筛选数据，并获取对应的索引
    #     selected_indices = self.data[(self.data['_timestamp'] >= start_ts)
    #                                  & (self.data['_timestamp'] <= end_ts)].index
    #     # 返回起始和结束索引
    #     return selected_indices.values[0], selected_indices.values[-1]

    # def _to_time(self, start_idx, end_idx):
    #     # 根据给定的索引范围获取起始和结束时间戳
    #     start_ts = self.data.loc[start_idx, '_timestamp']
    #     end_ts = self.data.loc[end_idx, '_timestamp']
    #     # 返回起始和结束时间戳
    #     return start_ts, end_ts

    # _timestamp于unixtime相同，改用unixtime
    def _to_idx(self, start_ts, end_ts):
        # 根据给定的时间戳范围筛选数据，并获取对应的索引
        selected_indices = self.data[(self.data['unixtime'] >= start_ts)
                                     & (self.data['unixtime'] <= end_ts)].index
        # 返回起始和结束索引
        return selected_indices.values[0], selected_indices.values[-1]

    def _to_time(self, start_idx, end_idx):
        # 根据给定的索引范围获取起始和结束时间戳
        start_ts = self.data.loc[start_idx, 'unixtime']
        end_ts = self.data.loc[end_idx, 'unixtime']
        # 返回起始和结束时间戳
        return start_ts, end_ts

    def _add_region(self, pos):
        if self.click_begin:
            # 如果是第一次点击，记录开始位置
            self.click_begin = False
            for i, plot in enumerate(self.plot_widgets):
                # 创建并添加起始和结束的垂直线
                self.start_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                self.end_line[i] = pg.InfiniteLine(pos.x(), angle=90, movable=False)
                plot.addItem(self.start_line[i])
                plot.addItem(self.end_line[i])
        else:
            # 如果是第二次点击，记录结束位置并创建区域
            self.click_begin = True
            for i, plot in enumerate(self.plot_widgets):
                # 移除起始和结束的垂直线
                plot.removeItem(self.start_line[i])
                plot.removeItem(self.end_line[i])

                # 创建一个线性区域并添加到绘图窗口
                region = pg.LinearRegionItem([self.start_line[i].value(), self.end_line[i].value()],
                                             brush=(0, 0, 255, 100))
                region.sigRegionChanged.connect(self._region_changed)

                self.start_line[i] = None
                self.end_line[i] = None

                plot.addItem(region)
                self.regions[i].append(region)
                # 获取选中的索引范围
                start_idx, end_idx = self._to_idx(int(region.getRegion()[0]), int(region.getRegion()[1]))
                self.logger.debug("Selected range: from index %s to index %s", start_idx, end_idx)

    def _region_changed(self, region):
        idx = 0
        # 找到当前改变的区域索引
        for reg_lst in self.regions:
            for i, reg in enumerate(reg_lst):
                if reg == region:
                    idx = i
                    break
        # 同步更新所有绘图窗口中的相应区域
        for reg_lst in self.regions:
            reg_lst[idx].setRegion(region.getRegion())

    def _del_region(self, pos):
        # 删除点击位置对应的区域
        for i, pwidget in enumerate(self.plot_widgets):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    pwidget.removeItem(reg)
                    self.regions[i].remove(reg)

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    self.logger.debug("Delete region(%s, %s)", start_idx, int(end_idx))
                    break

    def _edit_region(self, pos):
        set_val = None
        # 编辑点击位置对应的区域
        for i, _ in enumerate(self.regions):
            for reg in self.regions[i]:
                if reg.getRegion()[0] < pos.x() and reg.getRegion()[1] > pos.x():
                    if set_val is None:
                        # 弹出对话框选择标签
                        dialog = LabelOption(self.label_dict)
                        if dialog.exec() == QDialog.Accepted:
                            set_val = dialog.confirm_selection()
                        else:
                            set_val = None

                    # 设置区域的颜色和标签
                    reg.setBrush(self.checkColor(set_val))
                    reg.label = set_val

                    start_idx, end_idx = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
                    self.logger.debug("Edit region(%s, %s) label: %s", start_idx, end_idx, set_val)

    def mouse_clicked(self, event):
        if event.button() == Qt.LeftButton and hasattr(self, 'scatterItem'):
            pos = self.plot_widgets[0].plotItem.vb.mapToView(event.pos())

            if self.mode == 'add':
                self._add_region(pos)
            elif self.mode == 'edit':
                self._edit_region(pos)
            elif self.mode == 'del':
                self._del_region(pos)

    def mouse_moved(self, event):
        pos = self.plot_widgets[0].plotItem.vb.mapSceneToView(event)
        if not self.click_begin:
            # 动态更新结束线的位置
            for line in self.end_line:
                line.setPos(pos.x())