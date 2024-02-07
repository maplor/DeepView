
import sys
from PySide6 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PointClickHandler:
    def __init__(self, ax, points):
        self.ax = ax
        self.points = points
        self.text = ax.text(0, 0, "", visible=False, ha='center', va='center', fontsize=10)
        self.cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        if event.inaxes == self.ax:
            # Check if the click occurred within the axes
            x, y = event.xdata, event.ydata
            index = self.find_closest_point(x, y)

            # Display index on the figure
            self.text.set_text(f"Index: {index}")
            self.text.set_position((x, y))
            self.text.set_visible(True)
            self.ax.figure.canvas.draw()
            # read and save index
            print('clicked index is ' + str(index))

    def find_closest_point(self, x, y):
        # Find the index of the closest point to the clicked coordinates
        distances = [(i, (x - px) ** 2 + (y - py) ** 2) for i, (px, py) in enumerate(self.points)]
        index, _ = min(distances, key=lambda x: x[1])
        return index


class GridCanvas(QtWidgets.QDialog):
    def __init__(self, points, parent=None):
        super().__init__(parent)
        self.points = points
        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.scatter(*zip(*points))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create an instance of the PointClickHandler
        self.click_handler = PointClickHandler(self.ax, points)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Example data: list of points (x, y)
    points = [(1, 2), (3, 4), (5, 6), (7, 8)]

    # Create and show the dialog
    window = GridCanvas(points)
    window.show()

    sys.exit(app.exec_())
