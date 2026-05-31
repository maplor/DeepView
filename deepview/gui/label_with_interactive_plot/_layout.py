from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget


class LayoutMixin:
    def initLayout(self):
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)


        # 创建第一行三个按钮布局
        self.first_row_layout = QHBoxLayout()


        # 创建第二行选择框和颜色布局
        self.second_row_layout = QVBoxLayout()


        # 创建第三行（包含三个图和按钮）布局
        self.third_row_layout = QHBoxLayout()
        # 地图布局
        self.left_row1_layout = QHBoxLayout()
        # 视频布局
        self.left_row1_video_layout = QVBoxLayout()
        # 散点图布局
        self.row3_layout = QHBoxLayout()
        # 按钮布局,放入垂直的多个按钮
        self.charts_show_button_layout = QVBoxLayout()

        self.third_row_layout.addLayout(self.left_row1_layout, 1)
        self.third_row_layout.addLayout(self.left_row1_video_layout, 1)
        self.third_row_layout.addLayout(self.row3_layout, 1)
        self.third_row_layout.addLayout(self.charts_show_button_layout, 0)


        # 创建第四行布局，一个折线图，一个聚合按钮输入框图表占最大，按钮占最小
        self.fourth_row_layout = QHBoxLayout()

        # 折线图布局
        self.left_row3_layout = QHBoxLayout()

        # 按键聚合布局
        self.nestend_button_layout = QVBoxLayout()
        self.label_edit_button_layout = QVBoxLayout()
        # 放两个时间输入框
        self.st_end_time_layout = QVBoxLayout()
        self.first_edit_button_layout = QHBoxLayout()
        self.second_edit_button_layout = QHBoxLayout()
        self.third_edit_button_layout = QHBoxLayout()

        self.label_edit_button_layout.addLayout(self.first_edit_button_layout)
        self.label_edit_button_layout.addLayout(self.second_edit_button_layout)
        self.label_edit_button_layout.addLayout(self.third_edit_button_layout)

        self.nestend_button_layout.addStretch()
        self.nestend_button_layout.addLayout(self.st_end_time_layout)
        self.nestend_button_layout.addLayout(self.label_edit_button_layout)
        self.nestend_button_layout.addStretch()

        self.fourth_row_layout.addLayout(self.left_row3_layout,1)
        self.fourth_row_layout.addLayout(self.nestend_button_layout,0)


        self.main_layout.addLayout(self.first_row_layout,0)
        self.main_layout.addLayout(self.second_row_layout,0)
        self.main_layout.addLayout(self.third_row_layout,1)
        self.main_layout.addLayout(self.fourth_row_layout,1)
        # 创建中心图表
        self.createCenterPlot()








    # 初始化布局的方法
    def initLayout_old(self):

        # 创建主水平布局
        self.main_layout = QHBoxLayout()

        # 设置主布局
        self.setLayout(self.main_layout)

        # 创建左侧布局
        self.left_layout = QVBoxLayout()



        # 创建左侧row1布局
        self.left_row1_layout_all = QHBoxLayout()
        self.left_row1_layout = QHBoxLayout()
        self.left_row1_layout = QHBoxLayout()
        self.left_row1_layout_all.addLayout(self.left_row1_layout)
        self.left_row1_layout_all.addLayout(self.left_row1_video_layout)
        self.left_layout.addLayout(self.left_row1_layout_all)
        # self.left_layout.addLayout(self.left_row1_layout)

        # 创建左侧row2布局
        self.left_row2_layout = QVBoxLayout()
        self.left_layout.addLayout(self.left_row2_layout)

        # 创建左侧row3布局
        self.left_row3_layout = QHBoxLayout()
        self.left_layout.addLayout(self.left_row3_layout)

        # 创建右侧布局
        self.right_layout = QVBoxLayout()

        # 创建一个 QVBoxLayout 用于 row1_layout 中的多行布局
        self.nestend_layout = QVBoxLayout()

        # 创建 row1_layout 并将嵌套布局添加到其中
        self.row1_layout = QHBoxLayout()
        self.row1_layout.addLayout(self.nestend_layout)

        self.right_layout.addLayout(self.row1_layout)

        self.row2_layout = QVBoxLayout()
        self.right_layout.addLayout(self.row2_layout)

        self.row3_layout = QHBoxLayout()
        self.right_layout.addLayout(self.row3_layout)

        # # 创建一个弹簧 (QSpacerItem)
        # self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # self.right_layout.addItem(self.spacer)

        # 将左侧和右侧布局添加到主布局中，并设置相同的 stretch 因子，使它们宽度相同
        left_column = QWidget()
        left_column.setLayout(self.left_layout)
        self.main_layout.addWidget(left_column, 1)

        right_column = QWidget()
        right_column.setLayout(self.right_layout)
        self.main_layout.addWidget(right_column, 1)

        # 创建中心图表
        self.createCenterPlot()

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(lambda: self.backend.getSelectedAreaToSave(1))
        # 设置定时器每隔五分钟（300000 毫秒）触发一次
        self.timer.start(600000)