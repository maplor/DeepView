import json
import os
from functools import partial

import pyqtgraph as pg
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QFrame, QLineEdit, QMessageBox, QPushButton, QVBoxLayout

from deepview.gui.label_with_interactive_plot._chart_utils import combine_rectangles
from deepview.gui.label_with_interactive_plot.workers import SaveCsvTask


class RightSettingsMixin:
    # 创建右侧设置面板
    def createRightSettingPannel(self):
        settingPannel = QVBoxLayout()
        self.settingPannel = settingPannel
        self.bottom_layout.addLayout(self.settingPannel)

        self.settingPannel.setAlignment(Qt.AlignTop)

        self.createLabelButton()
        self.createRegionBtn()
        self.createSaveButton()

    # 创建保存按钮
    def createSaveButton(self):
        saveButton = QPushButton('Save')
        saveButton.clicked.connect(self.handleSaveButton)
        self.settingPannel.addWidget(saveButton)



    # def getSelectedAreaToSave(self, area_data):
    #     # print(areaData)
    #     try:
    #         area_data = json.loads(area_data)  # 解析 JSON 字符串
    #         # print("Parsed data:", areaData)
    #     except json.JSONDecodeError as e:
    #         print("Failed to decode JSON:", e)
    #         return
    #     for reg in area_data:
    #         name = reg[0].get("name")
    #         first_timestamp = reg[0].get("timestamp", {}).get("start")
    #         second_timestamp = reg[0].get("timestamp", {}).get("end")
    #         self.data.loc[(self.data['unixtime'] >= int(first_timestamp)) & (
    #                 self.data['unixtime'] <= int(second_timestamp)), 'label'] = name
    #     self.handleSaveButton()

    def getSelectedAreaToSave(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 0)
        save_task.signals.save_csv_finished.connect(self.on_save_finished)
        self.save_csv_thread_pool.start(save_task)

    def on_save_finished(self, new_path):
        QMessageBox.information(None, "保存CSV", f"文件已保存于 {new_path}", QMessageBox.Ok)
    
    def getSelectedAreaToSaveTimer(self, area_data):
        print("Saving CSV in the background.")
        combo_box_text = self.RawDatacomboBox.currentText()
        save_task = SaveCsvTask(area_data, self.data, self.cfg, combo_box_text, 1)
        self.save_csv_thread_pool.start(save_task)

    # 处理保存按钮点击事件
    def handleSaveButton(self):
        # for reg in self.regions[0]:
        #     if hasattr(reg, 'label') and reg.label:
        #         regionRange = reg.getRegion()
        #         self.data.loc[(self.data['_timestamp'] >= int(regionRange[0])) & (self.data['_timestamp'] <= int(regionRange[1])), 'label'] = reg.label

        os.makedirs(os.path.join(self.cfg["project_path"], "edit-data", ), exist_ok=True)
        edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText())
        # edit_data_path = os.path.join(self.cfg["project_path"], "edit-data", self.RawDatacomboBox.currentText().replace(".pkl", ".csv"))
        try:  # 如果文件存在就新建
            if os.path.exists(edit_data_path):
                for num in range(1, 100, 1):
                    firstname = edit_data_path.split('Hz')[0]
                    new_path = firstname + '_' + str(num) + '.pkl'
                    if not os.path.exists(new_path):
                        self.data.to_csv(new_path)
                        break
            else:
                new_path = edit_data_path
                self.data.to_csv(edit_data_path)
        except:
            print('save data error!')
        else:
            print(f'文件已经保存在{new_path}')

    # 创建标签按钮
    def createLabelButton(self):
        self.add_mode = QPushButton("Label Add Mode", self)
        self.add_mode.clicked.connect(partial(self._change_mode, "add"))
        self.edit_mode = QPushButton("Label Edit Mode", self)
        self.edit_mode.clicked.connect(partial(self._change_mode, "edit"))
        self.del_mode = QPushButton("Label Delete Mode", self)
        self.del_mode.clicked.connect(partial(self._change_mode, "del"))
        self.refresh = QPushButton("Refresh Spots", self)
        self.refresh.clicked.connect(self.updateRightPlotColor)
        self.settingPannel.addWidget(self.add_mode)
        self.settingPannel.addWidget(self.edit_mode)
        self.settingPannel.addWidget(self.del_mode)
        self.settingPannel.addWidget(self.refresh)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

    # 改变模式
    def _change_mode(self, mode: str):
        print(f'Change mode to "{mode}"')
        self.mode = mode

    # 创建区域按钮
    def createRegionBtn(self):
        addRegionBtn = QPushButton('Add region')
        addRegionBtn.clicked.connect(self.handleAddRegion)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText("Enter threshold")

        toLabelBtn = QPushButton('Reflect to Data')  # Save to label
        toLabelBtn.clicked.connect(self.handleToLabel)
        self.settingPannel.addWidget(addRegionBtn)
        self.settingPannel.addWidget(self.input_box)
        self.settingPannel.addWidget(toLabelBtn)

        # cache select region 缓存选定区域
        self.rightRegionRect = QRectF(0, 0, 1, 1)

        # Add horizontal line 添加水平线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.settingPannel.addWidget(line)

        # Clear empty region 清除空区域
        clearEmptyRegionBtn = QPushButton('Clear Empty Region')
        clearEmptyRegionBtn.clicked.connect(self.handleClearEmptyRegion)
        self.settingPannel.addWidget(clearEmptyRegionBtn)

    def handleAddRegion(self):
        if hasattr(self, 'rightRegionRoi'):
            return

        rect = self.viewC.viewRect()
        w = rect.width()
        h = rect.height()
        x = rect.x()
        y = rect.y()

        # create ROI
        roi = pg.ROI([x + w * 0.45, y + h * 0.45], [w * 0.1, h * 0.1])
        # 上
        roi.addScaleHandle([0.5, 1], [0.5, 0])
        # 右
        roi.addScaleHandle([1, 0.5], [0, 0.5])
        # 下
        roi.addScaleHandle([0.5, 0], [0.5, 1])
        # 左
        roi.addScaleHandle([0, 0.5], [1, 0.5])
        # 右下
        roi.addScaleHandle([1, 0], [0, 1])

        self.viewC.addItem(roi)

        self.rightRegionRoi = roi

        # roi.sigRegionChanged.connect(self.handleROIChange)
        # roi.sigRegionChangeFinished.connect(self.handleROIChangeFinished)
        # self.handleROIChange(roi)
        # self.handleROIChangeFinished(roi)

    # 处理反射到标签的方法
    def handleToLabel(self):
        if not hasattr(self, 'rightRegionRoi'):  # 如果没有右侧区域ROI，提示用户先添加区域
            print('Add region first.')
            return

        pos: pg.Point = self.rightRegionRoi.pos()
        size: pg.Point = self.rightRegionRoi.size()

        self.rightRegionRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.rightRegionRect)

        # 是否需要合并区间
        rectangles = []
        for p in points:
            index, start, end = p.data()
            startT, endT = self._to_time(start, end)
            rectangles.append((startT, endT))
        # combine rectangles 合并矩形，数据为开始结束时间
        if self.input_box.text() == "":
            combined_rectangles = combine_rectangles(rectangles, float(30))  # set default value
        else:
            combined_rectangles = combine_rectangles(rectangles, float(self.input_box.text()))

        # 传递combined_rectangles到backend
        markData = []
        for startT, endT in combined_rectangles:
            # print(startT, endT)
            start_id, end_id = self._to_idx(startT, endT)
            start_timestamp = self.data.loc[start_id, 'timestamp']
            end_timestamp = self.data.loc[end_id, 'timestamp']

            # 创建markData
            start_Area = {
                'name': 'data',
                'xAxis': start_timestamp,
                'itemStyle': {
                    'color': 'rgba(0, 0, 255, 0.39)'
                }
            }
            end_Area = {
                'xAxis': end_timestamp,
            }
            newArray = [start_Area, end_Area]

            markData.append(newArray)
        # print(markData)
        # 将 markData 转换为 JSON 字符串
        mark_data = json.dumps(markData)
        # 传递markData到backend
        self.backend.setMarkData(mark_data)

    def handleClearEmptyRegion(self):
        # 绑定html的Clear
        self.backend.clearMarkData()