import sys
import random
import pandas as pd
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from PySide6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Simulate data
        self.data = self.simulate_data()

        self.initUI()

    def simulate_data(self):
        timestamps = pd.date_range(start='2023-01-01', periods=100, freq='T')
        latitude = [random.uniform(-90, 90) for _ in range(100)]
        longitude = [random.uniform(-180, 180) for _ in range(100)]
        accelerometer_x = [random.uniform(-10, 10) for _ in range(100)]
        accelerometer_y = [random.uniform(-10, 10) for _ in range(100)]
        accelerometer_z = [random.uniform(-10, 10) for _ in range(100)]

        data = pd.DataFrame({
            'timestamp': timestamps,
            'latitude': latitude,
            'longitude': longitude,
            'accelerometer_x': accelerometer_x,
            'accelerometer_y': accelerometer_y,
            'accelerometer_z': accelerometer_z
        })
        data['index'] = data.index  # Add an index column
        return data

    def initUI(self):
        self.setWindowTitle("GPS and IMU Data Plotter")

        # Create main widget
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)  # Change to QHBoxLayout for horizontal layout

        # Create figures
        self.gps_fig = Figure()
        self.imu_fig = Figure()

        # Create canvas for figures
        self.gps_canvas = FigureCanvas(self.gps_fig)
        self.imu_canvas = FigureCanvas(self.imu_fig)

        # Add canvases to layout
        main_layout.addWidget(self.imu_canvas)  # Add IMU canvas first for left position
        main_layout.addWidget(self.gps_canvas)  # Add GPS canvas second for right position

        self.setCentralWidget(main_widget)

        # Plot data
        self.plot_data()

    def plot_data(self):
        # Plot GPS data with geopandas and contextily
        self.gps_ax = self.gps_fig.add_subplot(111)

        # Convert to GeoDataFrame
        self.data['geometry'] = [Point(xy) for xy in zip(self.data['longitude'], self.data['latitude'])]
        gdf = gpd.GeoDataFrame(self.data, geometry='geometry', crs="EPSG:4326")

        # Plot GeoDataFrame
        gdf.plot(ax=self.gps_ax, marker='o', color='blue', markersize=5, alpha=0.5)
        ctx.add_basemap(self.gps_ax, crs=gdf.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

        self.gps_ax.set_title("GPS Data")
        self.gps_ax.set_xlabel("Longitude")
        self.gps_ax.set_ylabel("Latitude")

        # Plot IMU data using the index as x-axis
        self.imu_ax = self.imu_fig.add_subplot(111)
        self.imu_ax.plot(self.data['index'], self.data['accelerometer_x'], label='Accelerometer X')
        self.imu_ax.plot(self.data['index'], self.data['accelerometer_y'], label='Accelerometer Y')
        self.imu_ax.plot(self.data['index'], self.data['accelerometer_z'], label='Accelerometer Z')
        self.imu_ax.set_title("IMU Data")
        self.imu_ax.set_xlabel("Timestamp")
        self.imu_ax.set_ylabel("IMU Values")
        self.imu_ax.legend()

        # Set x-ticks to display timestamps
        self.imu_ax.set_xticks(self.data['index'][::10])
        self.imu_ax.set_xticklabels(self.data['timestamp'][::10].dt.strftime('%H:%M:%S'))

        # Connect events
        self.imu_fig.canvas.mpl_connect('button_press_event', self.on_imu_click)
        self.gps_fig.canvas.mpl_connect('button_press_event', self.on_gps_click)

        self.gps_canvas.draw()
        self.imu_canvas.draw()

    def on_imu_click(self, event):
        # Get the index of the clicked IMU point
        ind = self.get_closest_point(event,
                                     self.data[['index', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z']],
                                     is_imu=True)
        if ind is not None:
            gps_point = self.data.iloc[ind][['longitude', 'latitude']]

            # Clear previous highlights
            self.clear_highlights(self.gps_ax, None)

            # Highlight the corresponding GPS data point
            self.gps_ax.scatter([gps_point['longitude']], [gps_point['latitude']], color='red')

            self.gps_canvas.draw()

    def on_gps_click(self, event):
        # Get the index of the clicked GPS point
        ind = self.get_closest_point(event, self.data[['longitude', 'latitude']], is_imu=False)
        if ind is not None:
            imu_point = self.data.iloc[ind][['index', 'accelerometer_x', 'accelerometer_y', 'accelerometer_z']]

            # Clear previous highlights
            self.clear_highlights(self.imu_ax, None)

            # Highlight the corresponding IMU data point
            self.imu_ax.plot(self.data['index'], self.data['accelerometer_x'], label='Accelerometer X', color='blue')
            self.imu_ax.plot(self.data['index'], self.data['accelerometer_y'], label='Accelerometer Y', color='orange')
            self.imu_ax.plot(self.data['index'], self.data['accelerometer_z'], label='Accelerometer Z', color='green')
            self.imu_ax.scatter([imu_point['index']], [imu_point['accelerometer_x']], color='red')
            self.imu_ax.scatter([imu_point['index']], [imu_point['accelerometer_y']], color='red')
            self.imu_ax.scatter([imu_point['index']], [imu_point['accelerometer_z']], color='red')

            self.imu_canvas.draw()

    def get_closest_point(self, event, data, is_imu):
        # Ensure the click is within the plot area
        if event.xdata is None or event.ydata is None:
            return None

        if is_imu:
            # Calculate the distance of the click from each index
            distances = [abs(event.xdata - idx) for idx in data['index']]
        else:
            # Calculate the distance of the click from each GPS point
            distances = [(event.xdata - x) ** 2 + (event.ydata - y) ** 2 for x, y in data.values]

        min_dist = min(distances)
        min_index = distances.index(min_dist)

        # Return the index of the closest point if the distance is small enough
        if min_dist < 1:  # Adjust this threshold as needed
            return min_index
        return None

    def clear_highlights(self, ax, scatter):
        # Clear previous highlights by removing all collections except the original scatter plot
        while len(ax.collections) > 1:
            ax.collections.pop()

        if scatter:
            # Re-draw the original scatter plot
            ax.add_collection(scatter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlotWindow()
    window.show()
    sys.exit(app.exec())
