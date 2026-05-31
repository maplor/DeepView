import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the main window
        self.setWindowTitle("Scatter Plot Toggle Colors")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # Create a button
        self.button = QPushButton("Toggle Colors")
        layout.addWidget(self.button)

        # Create a Matplotlib figure and canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create scatter plot data
        self.x = np.random.rand(100)
        self.y = np.random.rand(100)
        self.colors = np.full(100, 'blue')  # Initially all points are blue

        # Plot the scatter plot
        self.scatter = self.ax.scatter(self.x, self.y, c=self.colors)
        self.canvas.draw()

        # Connect the button click to the function
        self.button.clicked.connect(self.toggle_colors)

        # A flag to check the toggle state
        self.is_toggled = False

    def toggle_colors(self):
        # Toggle colors between blue and red for half of the points
        if self.is_toggled:
            self.colors[:50] = 'blue'
        else:
            self.colors[:50] = 'red'

        # Update scatter plot
        self.scatter.set_color(self.colors)
        self.canvas.draw()

        # Toggle the flag
        self.is_toggled = not self.is_toggled


# Create the Qt Application
app = QApplication(sys.argv)

# Create and show the main window
window = MainWindow()
window.show()

# Run the main Qt loop
sys.exit(app.exec())
