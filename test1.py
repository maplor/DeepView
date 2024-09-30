import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox, QHBoxLayout, QPushButton, QMessageBox, QInputDialog


class myComboBox:
    def __init__(self, _comboBox):
        self.comboBox = _comboBox
        self.comboBox.setModel(QStandardItemModel(self.comboBox))

    def addItem(self,itemTxt):
        QS_item = QStandardItem(itemTxt)
        self.comboBox.model().appendRow(QS_item)
        index = self.comboBox.count()-1
        self.add_btn(index, itemTxt)

    def add_btn(self, _index, _itemTxt):
        # 创建一个水平布局，并将标签和删除按钮添加到其中
        layout = QHBoxLayout()
        layout.setContentsMargins(75, 0, 0, 0)
        layout.setAlignment(Qt.AlignRight)  # Align the button to the right
        button = QPushButton('x')
        button.setFixedSize(20, 20)
        button.setStyleSheet("QPushButton { border: none; color:#6D6D6D ; font-size: 15px}")
        layout.addWidget(button)
        # 将水平布局添加到下拉菜单项的QWidget中
        widget = QWidget()
        widget.setLayout(layout)
        item = self.comboBox.model().item(_index)
        item.setSizeHint(widget.sizeHint())
        self.comboBox.view().setIndexWidget(item.index(), widget)
        # 将按钮连接到槽函数，用于从下拉列表中删除相应的项目
        button.clicked.connect(lambda: self.remove_Row(_itemTxt))

    def remove_Row(self,i):
        reply = QMessageBox.question(
            None,
            "Confirm Delete",
            f"Are you sure you want to remove '{i}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            index = self.comboBox.findText(i)
            self.comboBox.model().removeRow(index)


#------------------ myComboBox 类测试 ---------------------------------#


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.comboBox = QComboBox()
        self.comboBoxHandler = myComboBox(self.comboBox)

        for fruit in ['item1', 'item2', 'item3', 'item4']:
            self.comboBoxHandler.addItem(fruit)
        
        self.addButton = QPushButton("Add Item")
        self.addButton.clicked.connect(self.add_item)

        layout = QVBoxLayout()
        layout.addWidget(self.comboBox)
        layout.addWidget(self.addButton)
        self.setLayout(layout)

    # 弹出窗口添加新项目
    def add_item(self):
        new_item, ok = QInputDialog.getText(self, "Add Item", "Enter new item:")
        if ok:
            self.comboBoxHandler.addItem(new_item)

        # new_item = f"item{self.comboBox.count() + 1}"
        # self.comboBoxHandler.addItem(new_item)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = Widget()
    widget.show()
    app.exec()