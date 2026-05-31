from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)


# 带删除的选择框
class ReComboBox:
    def __init__(self, _comboBox, label_dict):
        self.comboBox = _comboBox
        self.comboBox.setModel(QStandardItemModel(self.comboBox))
        self.label_dict = label_dict

    def addItem(self, itemTxt):
        QS_item = QStandardItem(itemTxt)
        # 设置文字颜色
        QS_item.setBackground(QColor('#19232d'))
        QS_item.setForeground(QColor('#ffffff'))
        QS_item.setText(itemTxt)
        self.comboBox.model().appendRow(QS_item)
        index = self.comboBox.count()-1
        self.comboBox.view().repaint()
        self.add_btn(index, itemTxt)

    def add_btn(self, _index, _itemTxt):
        # 创建一个水平布局，并将标签和删除按钮添加到其中
        layout = QHBoxLayout()
        layout.setContentsMargins(75, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)  # Align the button to the right
        button = QPushButton('x')
        button.setStyleSheet("QPushButton { border: none; color:#6D6D6D ; font-size: 15px}")
        button.setFixedSize(20, 20)
        layout.addWidget(button)
        # 将水平布局添加到下拉菜单项的QWidget中
        widget = QWidget()
        widget.setLayout(layout)
        item = self.comboBox.model().item(_index)
        item.setSizeHint(widget.sizeHint())
        self.comboBox.view().setIndexWidget(item.index(), widget)
        # 将按钮连接到槽函数，用于从下拉列表中删除相应的项目
        button.clicked.connect(lambda: self.remove_Row(_itemTxt))

    def remove_Row(self, i):
        reply = QMessageBox.question(
            None,
            "Confirm Delete",
            f"Are you sure you want to remove '{i}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            index = self.comboBox.findText(i)
            if index != -1:
                self.comboBox.model().removeRow(index)
                # Remove the item from the dictionary
                if i in self.label_dict:
                    del self.label_dict[i]
                    # print(self.label_dict)
                    # print(f"Removed {i} from dictionary")


# 定义LabelOption类，继承自QDialog
class LabelOption(QDialog):
    def __init__(self, label_dict):
        super().__init__()

        # 创建一个垂直布局
        layout = QVBoxLayout()
        # 保存标签字典
        self.label_dict = label_dict
        # 创建一个空字典来保存单选按钮
        self.radio_buttons = {}
        # 遍历标签字典
        for label, lid in label_dict.items():
            # 为每个标签创建一个单选按钮
            self.radio_buttons[label] = QRadioButton(label)
            # 将单选按钮添加到布局中
            layout.addWidget(self.radio_buttons[label])

        # 创建确认按钮
        self.confirm_button = QPushButton("Confirm")
        # 连接确认按钮的点击事件到confirm_selection方法
        self.confirm_button.clicked.connect(self.confirm_selection)
        # 将确认按钮添加到布局中
        layout.addWidget(self.confirm_button)
        # 设置布局
        self.setLayout(layout)

    # 确认选择的方法
    def confirm_selection(self):
        # 遍历标签字典
        for label, lid in self.label_dict.items():
            # 如果单选按钮被选中
            if self.radio_buttons[label].isChecked():
                # 设置选中的选项为当前标签
                selected_option = label
                # 接受对话框，关闭对话框
                self.accept()
                # 返回选中的选项
                return selected_option
            else:
                # 如果没有选中任何选项，设置为None
                selected_option = None
        # 接受对话框，关闭对话框
        self.accept()
        # 返回选中的选项
        return selected_option