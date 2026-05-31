import pandas as pd
from PySide6.QtWidgets import QCheckBox

from deepview.gui.label_with_interactive_plot._chart_utils import find_charts_data_columns


class InteractionControlsMixin:
    def handleScatterItemClick(self, scatterItem, points):
        if len(points) >= 1:
            index, start, end = points[0].data()

            # start为latent space的切片索引，index为原始索引（对应切片的开始索引）
            lat, lon = self.data.loc[start, 'latitude'], self.data.loc[start, 'longitude']

            if pd.isna(lat) or pd.isna(lon):
                print("Latitude or longitude is missing.")
                return

            # 点击散点图高亮地图散点
            self.backend_map.triggeLineMapHighlightDotByIndex(start, lat, lon)
            # 点击散点图高亮折线图散点
            self.backend.triggeLineChartHighlightDotByIndex(start)
            # 点击散点图高亮自己
            self.handle_highlight_scatter_dot_by_index(index, True)

        return

    # 更新按钮状态的方法
    def updateBtn(self):
        # enabled 启用按钮
        if self.isTraining:
            # 如果在训练，设置按钮不可用
            self.featureExtractBtn.setEnabled(False)
        else:
            # 如果不在训练，设置按钮可用
            self.featureExtractBtn.setEnabled(True)

    # 渲染列列表的方法
    def renderColumnList(self):
        # 清空 layout
        while self.checkbox_layout.count():
            item = self.checkbox_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # 初始化复选框列表
        self.checkboxList = []
        # 遍历列名列表
        for column in self.all_sensor:
            # 创建复选框
            cb = QCheckBox(column)
            # 设置复选框为选中状态，如果在列名列表中
            cb.setChecked(column in self.column_names)
            # 将复选框添加到布局中
            self.checkbox_layout.addWidget(cb)
            # 将复选框添加到列表中
            self.checkboxList.append(cb)
            # 连接复选框状态改变事件到handleCheckBoxStateChange方法
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # 添加一个伸缩项以填充剩余区域并保持复选框左对齐
        self.checkbox_layout.addStretch()

    # 处理复选框状态改变的方法
    def handleCheckBoxStateChange(self):
        # 创建新选择列列表
        newSelectColumn = []
        # 遍历列名列表
        for i, column in enumerate(self.all_sensor):
            # 如果复选框被选中
            if self.checkboxList[i].isChecked():
                # 添加列到新选择列列表
                newSelectColumn.append(column)
        # 打印选择列
        # self.selectColumn = newSelectColumn
        print('selectColumn: %s' % (newSelectColumn))
        # self.current_select_sensor_column = newSelectColumn

        metadata = find_charts_data_columns(self.sensor_dict, newSelectColumn)

        # 更新左下图表
        self.backend.handleComboxSelection(metadata)