from PySide6.QtCore import QRectF
from PySide6.QtWidgets import QLineEdit, QPushButton


class SettingsAreaMixin:
    def createSettingArea(self):
        self.charts_show_button_layout.addStretch()
        self.labelColorBtn = self.createToggleLabelColor()
        self.charts_show_button_layout.addWidget(self.labelColorBtn)
        # 第一行生成选框按钮
        # self.first_row2_layout = QHBoxLayout()
        addRegionBtn = QPushButton('Generate area')
        # 设置按钮样式
        addRegionBtn.setStyleSheet(self.button_style)
        # 设置按钮最小宽度
        # addRegionBtn.setFixedWidth(160)
        addRegionBtn.clicked.connect(self.handleAddRegion)
        self.charts_show_button_layout.addWidget(addRegionBtn)
        # self.first_row2_layout.addWidget(addRegionBtn, alignment=Qt.AlignLeft)
        # self.first_row2_layout.addStretch()

        # 第二行Threshold输入框
        # self.second_row2_layout = QHBoxLayout()

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Input threshold")
        self.charts_show_button_layout.addWidget(self.input_box)

        # self.second_row2_layout.addWidget(QLabel("Threshold:"))
        # self.second_row2_layout.addWidget(self.input_box)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # 第三行两个按钮
        # self.third_row2_layout = QHBoxLayout()
        toLabelBtn = QPushButton('Find data')  # Save to label
        # 设置按钮样式
        toLabelBtn.setStyleSheet(self.button_style)
        toLabelBtn.clicked.connect(self.handleToLabel)

        clearEmptyRegionBtn = QPushButton('Clear data')
        # 设置按钮样式
        clearEmptyRegionBtn.setStyleSheet(self.button_style)
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)

        self.charts_show_button_layout.addWidget(toLabelBtn)
        self.charts_show_button_layout.addWidget(clearEmptyRegionBtn)
        self.charts_show_button_layout.addStretch()


        # # 添加一个伸缩因子来创建间距
        # # self.third_row2_layout.addStretch(1)
        # self.third_row2_layout.addWidget(toLabelBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(1)  # Increase the stretch factor to create more space
        # self.third_row2_layout.addWidget(clearEmptyRegionBtn, alignment=Qt.AlignLeft)
        # self.third_row2_layout.addStretch(10)

        # self.row2_layout.addLayout(self.first_row2_layout)
        # self.row2_layout.addLayout(self.second_row2_layout)
        # self.row2_layout.addLayout(self.third_row2_layout)