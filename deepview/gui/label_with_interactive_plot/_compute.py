import pyqtgraph as pg
from PySide6.QtCore import QThread

from deepview.gui.label_with_interactive_plot._chart_utils import find_charts_data_columns
from deepview.gui.label_with_interactive_plot.utils import get_data_from_pkl
from deepview.gui.label_with_interactive_plot.workers import HandleComputeWorker


class ComputeMixin:
    # 处理计算的方法
    def handleCompute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        # 获取combobox的内容
        self.data, self.dataChanged = get_data_from_pkl(self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged)

        # 延时100毫秒调用handleComputeAsyn方法
        self.computeTimer.singleShot(100, self.handleComputeAsyn)

    # 异步处理计算的方法
    def handleComputeAsyn(self):
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

        # 初始化图表之后不用添加spacer了
        # self.right_layout.removeItem(self.spacer)

        # 渲染右侧图表（特征提取功能）
        self.renderRightPlot()  # feature extraction function here

        # 设置训练状态为False
        self.isTraining = False

        self.updateBtn()


    def handel_calendar_data(self, data):
        self.data = data
        self.dataChanged.emit(self.data)
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)
        self.start_handle_compute()

    def start_handle_compute(self):
        # 打印开始训练
        print('start training...')
        # 设置训练状态为True
        self.isTraining = True
        # 更新按钮状态
        self.updateBtn()

        # 重新设置选框
        self.renderColumnList()

        self.handle_compute_thread = QThread()
        self.handle_compute_worker = HandleComputeWorker(self.root, self.data, self.RawDatacomboBox.currentText(), self.cfg, self.dataChanged, self.sensor_dict, self.column_names, self.data_length, self.model_path, self.model_name)
        self.handle_compute_worker.moveToThread(self.handle_compute_thread)
        self.handle_compute_thread.started.connect(self.handle_compute_worker.run)
        self.handle_compute_worker.finished.connect(self.handle_compute_finished)
        self.handle_compute_worker.stopped.connect(self.on_training_stopped)
        self.handle_compute_worker.dataChangedSignal.connect(self.compute_data_changed)
        self.handle_compute_worker.finished.connect(self.handle_compute_thread.quit)
        self.handle_compute_worker.finished.connect(self.handle_compute_worker.deleteLater)
        self.handle_compute_thread.finished.connect(self.handle_compute_thread.deleteLater)
        self.handle_compute_thread.start()

    def stop_training(self):
        self.handle_compute_worker.stop()


    def on_training_stopped(self):
        # TODO 绑定停止训练的方法
        # self.stop_button.setEnabled(False)
        self.isTraining = False
        self.updateBtn()
        print("Training was stopped.")

    def compute_data_changed(self, data):
        self.data = data
        self.min_time = self.data['unixtime'].min()
        metadatas = find_charts_data_columns(self.sensor_dict, self.column_names)
        self.backend.displayData(self.data, metadatas, self.label_colors)
        self.backend_map.displayMapData(self.data)

    def handle_compute_finished(self, data):
        (spots, start_indice, end_indice) = data
        # 设置训练状态为False
        self.isTraining = False
        self.updateBtn()
        # 清除中央绘图区域
        self.viewC.clear()
        # 创建一个散点图项
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None))
        self.start_indice = start_indice
        self.end_indice = end_indice
        # 在散点图中绘制点
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem
        scatterItem.sigClicked.connect(self.handleScatterItemClick)
        self.viewC.addItem(scatterItem)
        return