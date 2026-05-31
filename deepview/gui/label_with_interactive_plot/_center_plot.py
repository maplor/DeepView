import numpy as np
import pyqtgraph as pg

from deepview.gui.label_with_interactive_plot.utils import (
    featureExtraction,
    find_data_columns,
)


class CenterPlotMixin:
    def createCenterPlot(self):
        # 创建一个用于显示中央绘图区域的PlotWidget
        viewC = pg.PlotWidget()
        self.viewC = viewC
        # 禁用右键菜单
        self.viewC.setMenuEnabled(False)
        self.viewC.setBackground('w')
        # 将该PlotWidget添加到底部布局中
        self.row3_layout.addWidget(viewC, 2)

    def checkColor(self, label, first=False):
        if first:
            # 如果是第一次调用，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        if label not in list(self.label_dict.keys()):
            # # 如果标签不在标签字典中，返回默认的白色笔刷
            # return pg.mkBrush(255, 255, 255, 120)
            # 改成灰色
            return pg.mkBrush(72, 72, 96, 120)

        # 定义一组颜色
        list_color = [pg.mkBrush(0, 0, 255, 120),
                      pg.mkBrush(255, 0, 0, 120),
                      pg.mkBrush(0, 255, 0, 120),
                      pg.mkBrush(255, 255, 255, 120),
                      pg.mkBrush(255, 0, 255, 120),
                      pg.mkBrush(0, 255, 255, 120),
                      pg.mkBrush(255, 255, 0, 120),
                      pg.mkBrush(5, 5, 5, 120)]
        count = 0
        for lstr, _ in self.label_dict.items():
            if label == lstr:
                # 根据标签返回相应的颜色
                return list_color[count % len(list_color)]
            count += 1

    # 更新右侧散点图的颜色
    def updateRightPlotColor(self):
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # 如果first=False，使用已有的标签
            # 如果first=True，使用手动标签
            color = self.checkColor(self.data.loc[start, 'label'], first=True)
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)

        # 更新散点数据
        for reg in self.regions[0]:
            # 获取区域的起始和结束索引
            idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]), int(reg.getRegion()[1]))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

    def renderRightPlot(self):
        # 清除中央绘图区域
        self.viewC.clear()

        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))

        new_column_names = find_data_columns(self.sensor_dict, self.column_names)

        # 将数据切割成片段以获取潜在特征和索引
        start_indice, end_indice, pos = featureExtraction(self.root,
                                                          self.data,
                                                          self.data_length,
                                                          new_column_names,
                                                          self.model_path,
                                                          self.model_name)

        # 保存数据到scatterItem的属性中
        n = len(start_indice)
        spots = [{'pos': pos[i, :],
                  'data': (i, start_indice[i], end_indice[i]),
                  'brush': self.checkColor(self.data.loc[i * self.data_length, 'label'], first=True)}
                 for i in range(n)]

        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        self.viewC.addItem(scatterItem)
        return

    def find_i_by_indice(self, indice, start_indice, end_indice):
        '''
        map和sensor data都用原始索引，latent space用切片索引，这里从原始索引查找切片
        indice为原始索引
        start indice为切片索引
        '''
        for i in range(len(start_indice)):
            if start_indice[i] <= indice < end_indice[i]:
                return i
        return None  # 如果没有找到合适的范围

    # TODO 点击过快可能报错    self._plot.updateSpots(self._data.reshape(1)) AttributeError: 'NoneType' object has no attribute 'updateSpots'
    def handle_highlight_scatter_dot_by_index(self, index, useRawIndex = False):
        self.jump_to_timestamp(index)
        indice = index
        if not useRawIndex:
            indice = self.find_i_by_indice(index, self.start_indice, self.end_indice)
        if self.last_modified_points:
            for p, original_size, original_brush in self.last_modified_points:
                p.setSize(original_size)
                p.setBrush(original_brush)
            
        self.last_modified_points = []  # Clear the list
        if indice is not None:
            for spot in self.scatterItem.points():
                i, start, end = spot.data()
                if i == indice:
                    # Save current properties
                    original_size = spot.size()
                    original_brush = spot.brush()
                    self.last_modified_points.append((spot, original_size, original_brush))
                    spot.setSize(15)
                    spot.setBrush(pg.mkBrush(255, 0, 0, 255))

        

    # 显示原始标签在右下散点图上
    def toggleLabelColor(self):

        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # spots.append({
            # 'pos': (pos.x(), pos.y()),
            # 'data': (i, start, end),
            # 'brush': pg.mkBrush(color)
            # })
            if self.data.loc[start, 'label_flag'] == 0:
                original_brush = spot.brush()  # 读取原有颜色
                color = original_brush.color()  # 默认使用原有颜色
                # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            else:
                if self.is_toggled:
                    color = self.checkColor(self.data.loc[start, 'label'], first=False)
                else:
                    original_brush = spot.brush()  # 读取原有颜色
                    color = original_brush.color()  # 默认使用原有颜色
                    # color = self.checkColor(self.data.loc[start, 'label'], first=True)  # 相同背景色
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)
        # Toggle the flag
        self.is_toggled = not self.is_toggled

        # # 更新散点数据
        # for reg in self.regions[0]:
        #     # 获取区域的起始和结束索引
        #     idx_begin, idx_end = self._to_idx(int(reg.getRegion()[0]),
        #                                       int(reg.getRegion()[1]))
        #     for spot in spots:
        #         if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
        #             spot['brush'] = reg.brush

        self.scatterItem.setData(spots=spots)

        return

    def select_random_continuous_seconds(self, num_samples=100, points_per_second=90):
        # 随机选择连续的秒数数据段
        selected_dfs = []
        start_indice = []
        end_indice = []

        while len(selected_dfs) < num_samples:
            start_idx = np.random.randint(0, len(self.data) - points_per_second)
            end_idx = start_idx + points_per_second - 1
            selected_range = self.data.iloc[start_idx:end_idx + 1]

            if not selected_range[['acc_x', 'acc_y', 'acc_z']].isna().any().any():
                selected_dfs.append(selected_range)  # 从start_idx到end_idx的数据段
                start_indice.append(start_idx)
                end_indice.append(end_idx)

        return start_indice, end_indice, selected_dfs