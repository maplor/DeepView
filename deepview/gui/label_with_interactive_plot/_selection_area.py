from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from deepview.gui.label_with_interactive_plot.widgets.time_selector import DateTimeSelector


class SelectionAreaMixin:
    # 创建右上角的选择模型和数据复选框
    def createModelSelectLabelArea(self):

        # 日历按钮
        self.calendar_btn = QPushButton('Calendar')
        self.calendar_btn.setStyleSheet(self.button_style)
        self.calendar_btn.clicked.connect(self.open_calendar)
        self.first_row_layout.addWidget(self.calendar_btn, alignment=Qt.AlignLeft)
        
        # 第一行布局,包含Select model标签和选择框，, alignment=Qt.AlignLeft
        # 创建模型组合框和标签
        modelComboBoxLabel, modelComboBox = self.createModelComboBox()
        # self.first_row1_layout = QHBoxLayout()
        # self.first_row1_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBoxLabel, alignment=Qt.AlignLeft)
        self.first_row_layout.addWidget(modelComboBox, alignment=Qt.AlignLeft)
        self.refresh_btn = QPushButton('Refresh')
        self.refresh_btn.setStyleSheet(self.button_style)
        self.refresh_btn.clicked.connect(self.handleRefresh)

        # self.first_row1_layout.addWidget(self.refresh_btn, alignment=Qt.AlignLeft)
        # self.first_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # 第二行布局
        # 创建原始数据组合框和标签
        RawDataComboBoxLabel, RawDatacomboBox = self.createRawDataComboBox()
        # self.first_row_layout.addLayout(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.first_row_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)

        featureExtractBtn = self.createFeatureExtractButton()
        # self.first_row_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.second_row1_layout = QHBoxLayout()
        # self.first_row_layout.addLayout(self.second_row1_layout)
        # self.second_row1_layout.addWidget(RawDataComboBoxLabel, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addWidget(RawDatacomboBox, alignment=Qt.AlignLeft)
        # self.second_row1_layout.addStretch()  # 添加一个伸缩因子来填充剩余空间

        # check_box布局
        self.checkbox_layout = QHBoxLayout()

        # 颜色展示
        self.color_layout = QHBoxLayout()

        self.second_row_layout.addLayout(self.checkbox_layout)
        self.second_row_layout.addLayout(self.color_layout)


        # TODO: 将一部分功能改到日历中
        # 第三行布局 Display data 按钮
        # self.third_row1_layout = QHBoxLayout()
        # featureExtractBtn = self.createFeatureExtractButton()
        # self.labelColorBtn = self.createToggleLabelColor()  # 单击可以让右下散点图显示已有标签

        # self.third_row1_layout.addWidget(featureExtractBtn, alignment=Qt.AlignRight)
        # self.third_row1_layout.addWidget(labelColorBtn, alignment=Qt.AlignRight)

        # self.nestend_layout.addLayout(self.first_row1_layout)
        # self.nestend_layout.addLayout(self.second_row1_layout)
        # self.nestend_layout.addLayout(self.checkbox_layout)
        # self.nestend_layout.addLayout(self.color_layout)
        # self.nestend_layout.addLayout(self.third_row1_layout)

        self.renderColumnList()
        # self.clear_color_layout()
        self.display_colors(self.label_colors)
    
    def handleRefresh(self):
        self.update_model_combobox()
        self.update_data_combobox()

    def open_calendar(self):
        self.calendar = DateTimeSelector(self)
        self.calendar.show()


    # TODO 全部的closeEvent都没生效，需要找这个项目的closeEvent方法
    def closeEvent(self, event):
        if self.calendar:
            self.calendar.close()
        super().closeEvent(event)


    def display_colors(self, colors):
        # 创建水平布局并添加标签和颜色框
        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()

    def clear_color_layout(self):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
    
    def clear_color_layout_and_display(self, colors):
        # 移除并删除所有布局项
        while self.color_layout.count() > 0:  # 改为0以清除所有
            item = self.color_layout.takeAt(0)
            if item.layout():
                while item.layout().count():
                    widget = item.layout().takeAt(0).widget()
                    if widget:
                        widget.deleteLater()
                item.layout().deleteLater()
                # 创建水平布局并添加标签和颜色框

        for color_name, color_value in colors.items():
            layout = QHBoxLayout()

            label = QLabel(color_name + ":")
            layout.addWidget(label)

            color_frame = QFrame()
            color_frame.setFixedSize(20, 20)
            color_frame.setStyleSheet(f"background-color: {color_value};")
            layout.addWidget(color_frame)

            self.color_layout.addLayout(layout)
        self.color_layout.addStretch()