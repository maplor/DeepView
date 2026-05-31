import json
import logging

import pandas as pd
from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtWidgets import QMessageBox


class Backend(QObject):
    highlightDotByindex = Signal(int, float, float)
    # TODO 参数待定
    highlightScatterDotByindexSign = Signal(int)
    getSelectedAreaByHtml = Signal(str)
    setStartEndTime = Signal(str, str)
    setStartAndEndDataSign = Signal(str, str, str, str, str, str)
    getSelectedAreaToSaveSign = Signal(str)
    getSelectedAreaToSaveTimerSign = Signal(str)

    def __init__(self):
        super().__init__()
        self.data = None
        self.select_option = None

    # 创建一个函数来找到最近的有效索引
    @Slot(pd.DataFrame)
    def handle_data_changed(self, data):
        self.data = data  # Update the data attribute
        # print("Backend's DataFrame has been updated:")
        # print(self.data)

    @Slot()
    def handle_label_change(self, option):
        self.select_option = option

    @Slot(result='QString')
    def get_label_option(self):
        result = self.select_option if self.select_option is not None else ""
        # print(f"Returning: {result}")
        return result

    # 通过索引高亮散点，点击地图散点高亮折线图散点
    @Slot()
    def triggeLineChartHighlightDotByIndex(self, index):
        print(f"Triggering highlight dot({index})...")
        self.view.page().runJavaScript(f"highlightLineChartDotByIndex('{index}')")

    # 设置开始结束时间到标签
    @Slot(str, str)
    def setStartEndTimeToLabel(self, start_time, end_time):
        print(f"Setting start time: {start_time}, end time: {end_time}")
        self.setStartEndTime.emit(start_time, end_time)

    @Slot(str, str, str, str, str, str)
    def setStartAndEndData(self, id1, lon1, lat1, id2, lon2, lat2):
        # print("Setting start and end data...")
        self.setStartAndEndDataSign.emit(id1, lon1, lat1, id2, lon2, lat2)

    # 通过索引高亮散点，点击折线图散点高亮散点图散点
    @Slot(int)
    def handleHighlightScatterDotByIndex(self, index):
        print(f"Triggering highlight dot({index})...")
        self.highlightScatterDotByindexSign.emit(index)

    # 通过索引高亮散点，点击折线图散点高亮地图散点
    @Slot(int)
    def handleHighlightDotByIndex(self, index):
        lat, lon = self.data.loc[index, 'latitude'], self.data.loc[index, 'longitude']

        if pd.isna(lat) or pd.isna(lon):
            print("Latitude or longitude is missing.")
            return

        print("handleing highlight dot...")
        self.highlightDotByindex.emit(index, lat, lon)

    # add label按钮点击事件
    @Slot()
    def handleAddLabel(self, selected_option):
        print("Adding label...")
        self.view.page().runJavaScript(f"addLabel('{selected_option}')")

    # delete label按钮点击事件
    @Slot(int)
    def handleDeleteLabel(self, status):
        print("Deleting label...")
        self.view.page().runJavaScript(f"deleteLabel('{status}')")

    # 从框选的散点图设置折线图markData
    @Slot(str)
    def setMarkData(self, data):
        print("Setting mark data...")
        self.view.page().runJavaScript(f"setMarkData('{data}')")

    # 清空折线图markData
    @Slot()
    def clearMarkData(self):
        print("Clearing mark data...")
        self.view.page().runJavaScript("clearMarkData()")

    # 从html获取折线图框选区域
    # @Slot(result='QVariant')
    @Slot()
    def getSelectedArea(self):
        print("etSelectedArea..")
        return self.view.page().runJavaScript("getSelectedArea()", 0, self.test_callback)

    def test_callback(self, result):
        self.getSelectedAreaByHtml.emit(result)

    @Slot()
    def getSelectedAreaToSave(self, is_timer):
        print("getSelectedAreaToSave..")
        # 使用lambda传递参数给save_callback
        self.view.page().runJavaScript("getSelectedArea()", 0, lambda result: self.save_callback(result, is_timer))

    def save_callback(self, result, is_timer):
        # 根据传入的参数选择要发射的信号
        if is_timer == 1:
            self.getSelectedAreaToSaveTimerSign.emit(result)
        elif is_timer == 0:
            self.getSelectedAreaToSaveSign.emit(result)

    # 添加label弹出确认提示框
    @Slot(result='QVariant')
    def confirmOverlap(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Labels overlapped, Over write?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    # 删除标签弹出确认提示框
    @Slot(result='QVariant')
    def confirmDelete(self):
        # 显示确认对话框
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Question)
        msg_box.setText("Do you want to delete the label?")
        # msg_box.setText("是否删除选中的标记区域？")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        result = msg_box.exec() == QMessageBox.Yes
        # print(result)
        # 返回布尔值
        return result

    @Slot()
    def displayData(self, data, metadata=None, label_colors=None):
        if isinstance(data, pd.DataFrame):
            series_combined = ["timestamp", "unixtime", "index", "latitude", "longitude"] + [item for data in metadata
                                                                                             for item in
                                                                                             data["series"]]
            data = data[series_combined]

            # 将空字符串替换为 None
            data = data.replace('', None)

            # 将 NaN 值替换为 None,避免转换为json出错
            data = data.where(pd.notnull(data), None)

            # 将 DataFrame 转换为字典列表
            data_records = data.to_dict(orient='records')

            if metadata is None:
                # 创建元数据信息
                metadata = [
                    {
                        "name": "acceleration",
                        "xAxisName": "timestamp",
                        "yAxisName": "Y Axis 1",
                        "series": ["acc_x", "acc_y", "acc_z"]
                    }
                ]

            # 将元数据和数据打包到一个字典中
            result = {
                "metadata": metadata,
                "data": data_records,
                "labelColors": label_colors
            }
        else:
            # 如果数据不是 DataFrame，则直接使用传入的数据
            result = data
        # 将结果转换为 JSON 格式
        json_data = json.dumps(result)
        # print(json_data)
        self.view.page().runJavaScript(f"displayData('{json_data}')")

    # 更新labelColors
    @Slot()
    def updateLabelColors(self, label_colors):
        label_colors = json.dumps(label_colors)
        self.view.page().runJavaScript(f"handleLabelColorChange('{label_colors}')")

    # combox选择事件
    @Slot()
    def handleComboxSelection(self, charts_data):
        charts_data = json.dumps(charts_data)
        print("Combox selection...")
        self.view.page().runJavaScript(f"handleComboxChange('{charts_data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")

    @Slot(str)
    def receiveData(self, data):
        print("Received data from frontend:", data)

    @Slot()
    def triggerUpdate(self):
        self.view.page().runJavaScript("getInputValue()")  # 调用前端的getInputValue函数


class BackendMap(QObject):
    highlightLineChartDotByindex = Signal(int)

    def __init__(self):
        super().__init__()

    # 点击折线图高亮地图散点，没有则添加新点
    @Slot(str, float, float)
    def triggeLineMapHighlightDotByIndex(self, index, lat, lon):
        print(f"Triggering highlight dot({index})...")
        # self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}, {lat}, {lon}')")
        self.view.page().runJavaScript(f"highlightByIndexAndLatLng('{index}', {lat}, {lon})")

    # 点击地图高亮折线图散点
    @Slot(int)
    def handleHighlightLineDotByIndex(self, index):
        print("handleing highlight dot...")
        self.highlightLineChartDotByindex.emit(index)

    @Slot()
    def highlightLineChartTwoDots(self, id1, lon1, lat1, id2, lon2, lat2):
        print("highlightLineChartTwoDots...")
        self.view.page().runJavaScript(f"highlightTwoMarkers('{id1}', {lat1}, {lon1}, '{id2}', {lat2}, {lon2})")

    # 在data display之后读取gps边界，然后作为初始地图
    @Slot()
    def displayMapData(self, data):
        if isinstance(data, pd.DataFrame):
            # 将列名转换为列表
            columns_list = data.columns.tolist()
            logging.debug(f"columns_list: {columns_list}")
            # 选择需要的列 index、latitude 和 longitude，并去除 latitude 和 longitude 中的缺失值。
            data = data[['index', 'latitude', 'longitude']].dropna(subset=['latitude', 'longitude'])
            # 使用 iloc 按索引进行降采样, 一万个条目取一个。
            interval = 10 * 60 * 25  # 25 is sampling rate, 10 is minutes
            data = data.iloc[::interval]

            data = data.to_dict(orient='records')
        data = json.dumps(data)
        self.view.page().runJavaScript(f"displayMapData('{data}')")

    def handleJavaScriptLog(self, result):
        print(f"JavaScript log: {result}")