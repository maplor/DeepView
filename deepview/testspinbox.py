
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Main Window")

        # Create a button
        self.button = QPushButton("Open New Window", self)
        self.button.clicked.connect(self.open_new_window)

    def open_new_window(self):
        new_window = QMainWindow()
        new_window.setWindowTitle("New Window")
        new_window.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
