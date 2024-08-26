
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QCheckBox, QLabel, QPushButton, QMessageBox

class CheckBoxWindow(QWidget):
    def __init__(self):
        super().__init__()

        # 创建主布局
        self.layout = QVBoxLayout()

        # 创建“全选”按钮
        self.select_all_checkbox = QCheckBox("全选")
        self.layout.addWidget(self.select_all_checkbox)

        # 创建多个选项
        self.checkboxes = [QCheckBox(f"选项 {i+1}") for i in range(5)]
        for checkbox in self.checkboxes:
            self.layout.addWidget(checkbox)

        # 创建显示当前选中选项的标签
        self.selected_items_label = QLabel("当前选中的选项: []")
        self.layout.addWidget(self.selected_items_label)

        # 创建显示已选选项的按钮
        self.show_selected_button = QPushButton("显示已选选项")
        self.layout.addWidget(self.show_selected_button)

        # 连接“全选”按钮的状态改变信号到槽函数
        self.select_all_checkbox.stateChanged.connect(self.select_all)

        # 连接各个选项的状态改变信号到槽函数
        for checkbox in self.checkboxes:
            checkbox.stateChanged.connect(self.update_select_all_checkbox)

        # 连接显示已选选项按钮的点击信号到槽函数
        self.show_selected_button.clicked.connect(self.show_selected_items)

        self.setLayout(self.layout)

        # 标志位，用于控制槽函数逻辑
        self.updating = False

    def select_all(self, state):
        if not self.updating:
            self.updating = True
            # 根据“全选”按钮的状态设置各个选项的状态
            for checkbox in self.checkboxes:
                checkbox.setChecked(state == Qt.Checked)
            self.update_selected_items()
            self.updating = False

    def update_select_all_checkbox(self):
        if not self.updating:
            self.updating = True
            # 检查所有选项的状态以更新“全选”按钮的状态
            all_checked = all(checkbox.isChecked() for checkbox in self.checkboxes)
            any_unchecked = any(not checkbox.isChecked() for checkbox in self.checkboxes)
            if all_checked:
                self.select_all_checkbox.setCheckState(Qt.Checked)
            elif any_unchecked:
                self.select_all_checkbox.setCheckState(Qt.Unchecked)
            else:
                self.select_all_checkbox.setTristate(False)
                self.select_all_checkbox.setCheckState(Qt.PartiallyChecked)
            self.update_selected_items()
            self.updating = False

    def update_selected_items(self):
        # 更新当前选中的选项
        selected_items = [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]
        self.selected_items_label.setText(f"当前选中的选项: {selected_items}")

    def show_selected_items(self):
        # 显示当前选中的选项
        selected_items = [checkbox.text() for checkbox in self.checkboxes if checkbox.isChecked()]
        QMessageBox.information(self, "已选选项", f"当前选中的选项: {selected_items}")

if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import Qt

    app = QApplication(sys.argv)
    window = CheckBoxWindow()
    window.show()
    sys.exit(app.exec_())
