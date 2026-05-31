import datetime
import json
import os

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from deepview.gui.label_with_interactive_plot.styles import combobox_style_light


class LabelControlsMixin:
    def createLeftBotton(self):
        # 第一行start time显示框
        self.left_start_time_layout = QHBoxLayout()
        self.start_input_box = QLineEdit(self)
        self.start_input_box.setPlaceholderText("start time")
        self.left_start_time_layout.addWidget(QLabel("Start time:"))
        self.left_start_time_layout.addWidget(self.start_input_box)
        self.left_start_time_layout.addStretch()
        self.st_end_time_layout.addLayout(self.left_start_time_layout)

        # self.left_row2_layout.addLayout(self.left_start_time_layout)

        # 第二行end time显示框
        self.left_end_time_layout = QHBoxLayout()
        self.end_input_box = QLineEdit(self)
        self.end_input_box.setPlaceholderText("end time")
        self.left_end_time_layout.addWidget(QLabel("End time: "))
        self.left_end_time_layout.addWidget(self.end_input_box)
        self.left_end_time_layout.addStretch()
        self.st_end_time_layout.addLayout(self.left_end_time_layout)
        # self.left_row2_layout.addLayout(self.left_end_time_layout)

        # 第三行label选项框
        # self.left_label_layout = QHBoxLayout()

        self.label_combobox = QComboBox()
        self.label_combobox.setStyleSheet(combobox_style_light)
        self.label_combobox.setModel(QStandardItemModel(self.label_combobox))

        # self.comboBoxHandler = ReComboBox(self.label_combobox, self.label_dict)
        # self.label_dict
        # self.label_combobox = ReComboBox()


        # for label in self.label_dict.keys():
        #     self.comboBoxHandler.addItem(label)

        for item in self.label_dict.keys():
            self.addItem(item)

        self.label_combobox.currentTextChanged.connect(
            self.backend.handle_label_change
        )

        # for label in self.label_dict.keys():
        #     self.label_combobox.addItem(label)
        # self.backend.handle_label_change(self.label_combobox.currentText())
        # self.label_combobox.currentTextChanged.connect(
        #     self.backend.handle_label_change
        # )

        # self.left_label_layout.addWidget(QLabel("Label:     "))
        # self.left_label_layout.addWidget(self.label_combobox, alignment=Qt.AlignLeft)
        self.first_edit_button_layout.addWidget(QLabel("Label:     "))
        self.first_edit_button_layout.addWidget(self.label_combobox, alignment=Qt.AlignLeft)

        # 创建label按钮
        add_label_btn = QPushButton('Create label')
        add_label_btn.clicked.connect(self.add_item)
        add_label_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(add_label_btn)
        self.first_edit_button_layout.addWidget(add_label_btn)

        # 暂时不用
        self.save_label_btn = QPushButton('Save label')
        self.save_label_btn.clicked.connect(self.save_label)
        self.save_label_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(self.save_label_btn)


        # 保存csv按钮 TODO 问一下这个还要不
        self.save_csv_btn = QPushButton('Save csv')
        self.save_csv_btn.clicked.connect(lambda: self.backend.getSelectedAreaToSave(0))
        self.save_csv_btn.setStyleSheet(self.button_style)
        # self.left_label_layout.addWidget(self.save_csv_btn)
        # self.left_label_layout.addStretch()
        # self.left_row2_layout.addLayout(self.left_label_layout)




        # 第四行三个按钮
        # self.left_button_layout = QHBoxLayout()
        # add label按钮
        add_label_btn = QPushButton('Add label')
        add_label_btn.clicked.connect(lambda: self.backend.handleAddLabel(self.label_combobox.currentText()))
        # 设置按钮样式
        add_label_btn.setStyleSheet(self.button_style)

        # delete label按钮
        delete_label_btn = QPushButton('Delete label')
        delete_label_btn.setCheckable(True)  # Make the button checkable
        delete_label_btn.clicked.connect(lambda: self.backend.handleDeleteLabel(int(delete_label_btn.isChecked())))
        # Set the button style based on the checked state
        delete_label_btn.setStyleSheet(
            self.button_style + "background-color: red;" if delete_label_btn.isChecked() else self.button_style + "background-color: green;")

        # Reflect to latent space按钮
        reflect_to_latent_btn = QPushButton('View on latent space')
        reflect_to_latent_btn.clicked.connect(lambda: self.backend.getSelectedArea())
        # 设置按钮样式
        reflect_to_latent_btn.setStyleSheet(self.button_style)


        self.second_edit_button_layout.addWidget(add_label_btn)
        self.second_edit_button_layout.addWidget(delete_label_btn)
        self.second_edit_button_layout.addWidget(self.save_csv_btn)

        self.third_edit_button_layout.addWidget(reflect_to_latent_btn)


        # # 将按钮添加到布局中
        # self.left_button_layout.addWidget(add_label_btn)
        # self.left_button_layout.addStretch(1)
        # self.left_button_layout.addWidget(delete_label_btn)
        # self.left_button_layout.addStretch(1)
        # self.left_button_layout.addWidget(reflect_to_latent_btn)
        # self.left_button_layout.addStretch(10)

        # self.left_row2_layout.addLayout(self.left_button_layout)

    def addItem(self, itemTxt):
        QS_item = QStandardItem(itemTxt)
        # QS_item.setBackground(QColor('#19232d'))
        # QS_item.setForeground(QColor('#ffffff'))
        QS_item.setBackground(QColor('#ffffff'))
        QS_item.setForeground(QColor('#19232d'))
        QS_item.setText(itemTxt)
        self.label_combobox.model().appendRow(QS_item)
        index = self.label_combobox.count() - 1
        self.add_btn(index, itemTxt)

    def add_btn(self, _index, _itemTxt):
        layout = QHBoxLayout()
        layout.setContentsMargins(75, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)
        button = QPushButton('x')
        button.setFixedSize(20, 20)
        button.setStyleSheet(self.remove_button_style)
        layout.addWidget(button)
        widget = QWidget()
        widget.setLayout(layout)
        item = self.label_combobox.model().item(_index)
        item.setSizeHint(widget.sizeHint())
        self.label_combobox.view().setIndexWidget(item.index(), widget)
        button.clicked.connect(lambda: self.remove_Row(_itemTxt))

    def remove_Row(self, i):
        reply = QMessageBox.question(
            None,
            "Confirm Delete",
            f"Are you sure you want to remove '{i}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            index = self.label_combobox.findText(i)
            if index != -1:
                self.label_combobox.model().removeRow(index)
                # Remove the item from the dictionary
                if i in self.label_dict:
                    del self.label_dict[i]
                # Update the label colors
                self.init_label_colors()
                self.backend.updateLabelColors(self.label_colors)
                # self.clear_color_layout()
                # self.display_colors(self.label_colors)
                self.clear_color_layout_and_display(self.label_colors)


    def add_item(self):
        key, ok = QInputDialog.getText(self, 'Add Label', 'Enter the Label:')
        if ok and key:
            if key not in self.label_dict:
                # value = f"{self.label_combobox.count() + 1}"
                value = self.label_combobox.count() + 1
                self.label_dict[key] = value
                self.addItem(key)
                # self.comboBoxHandler.addItem(key) updateLabelColors
                print(self.label_dict)
                self.init_label_colors()
                self.backend.updateLabelColors(self.label_colors)
                # self.clear_color_layout()
                # self.display_colors(self.label_colors)
                self.clear_color_layout_and_display(self.label_colors)

            else:
                QMessageBox.warning(self, 'Error', 'Label already exists.')




    def save_label(self):
        # 保存标签字典
        # save_label_dict(self.root, self.label_dict)
        config_path = os.path.join(self.cfg["project_path"], "config.yaml")
        with open(config_path, 'r') as f:
            config = self.yaml.load(f)
        config['label_dict'] = self.label_dict
        with open(config_path, 'w') as f:
            self.yaml.dump(config, f)
        print("Saving label Successfully")




    def setStartEndTime(self, start_time, end_time):
        self.start_input_box.setText(start_time)
        self.end_input_box.setText(end_time)

    # 定义一个函数将ISO格式的时间字符串转换为Unix时间戳
    def iso_to_timestamp(self, iso_str):
        dt = datetime.datetime.fromisoformat(iso_str.rstrip('Z'))
        timestamp = dt.timestamp()

        # return dt.timestamp()
        return round(timestamp, 5)

    # 更新右侧散点图的颜色
    def handleReflectToLatent(self, areaData):
        # print(areaData)
        try:
            areaData = json.loads(areaData)  # 解析 JSON 字符串
            # print("Parsed data:", areaData)
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
        spots = []
        for spot in self.scatterItem.points():
            pos = spot.pos()
            i, start, end = spot.data()
            # 如果first=False，使用已有的标签
            # 如果first=True，使用手动标签
            color = self.checkColor(self.data.loc[start, 'label'], first=True)
            # original_brush = spot.brush()  # 读取原有颜色
            # color = original_brush.color()  # 默认使用原有颜色
            # spots.append({
            # 'pos': (pos.x(), pos.y()),
            # 'data': (i, start, end),
            # 'brush': pg.mkBrush(color)
            # })
            spot = {'pos': (pos.x(), pos.y()), 'data': (i, start, end),
                    'brush': pg.mkBrush(color)}
            spots.append(spot)

        for reg in areaData:
            # 获取区域的起始和结束索引
            name = reg[0].get("name")
            first_timestamp = reg[0].get("timestamp", {}).get("start")
            second_timestamp = reg[0].get("timestamp", {}).get("end")
            new_color = reg[0].get("itemStyle", {}).get("color")

            idx_begin, idx_end = self._to_idx(int(first_timestamp), int(second_timestamp))
            for spot in spots:
                if idx_begin < spot['data'][1] and idx_end > spot['data'][2]:
                    spot['brush'] = pg.mkBrush(new_color)

        self.scatterItem.setData(spots=spots)