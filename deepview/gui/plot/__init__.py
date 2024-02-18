from PySide6.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QCheckBox
from PySide6.QtCore import QTimer, QRectF, Qt
import numpy as np
import pandas as pd
import pyqtgraph as pg

np.random.seed(0)
df = pd.DataFrame({'Col ' + str(i + 1): np.random.rand(30) for i in range(6)})  # 随机生成2组30以内的数

clickedPen = pg.mkPen('b', width=2)

class PlotWithInteraction(QWidget):
    def __init__(self) -> None:
        super().__init__()

        # 初始化布局
        self.initLayout()

        # TODO 模拟计算时间，完成正常逻辑后删除
        self.computeTimer = QTimer()

        self.columnList = ['AAA', 'BBB', 'CCC']
        # self.columnCheckStateList = list(map(lambda _x: False, self.columnList))
        self.selectColumn = []

        # 缓存框选区域
        self.selectRect = QRectF(0, 0, 1, 1)

        # 缓存改变的点
        self.lastChange = []

        # status
        self.isTarining = False

        # 创建顶部多选框
        self.createCheckBoxList()

        # 创建下方界面
        self.createLeftPlot()
        self.createCenterBtn()
        self.createRightPlot()

        self.renderLeftPlot()
        self.updateBtn()

    def initLayout(self):
        self.main_layout = QVBoxLayout()

        self.setLayout(self.main_layout)

        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addLayout(self.bottom_layout)

    '''
    ==================================================
    顶部多选框
    self.checkboxList list(QCheckBox)
    ==================================================
    '''
    def createCheckBoxList(self):
        self.checkboxList = []
        for column in self.columnList:
            cb = QCheckBox(column)
            self.top_layout.addWidget(cb)
            self.checkboxList.append(cb)
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # 添加一个伸缩体，填满剩余区域，让选项居左
        self.top_layout.addStretch()

    def handleCheckBoxStateChange(self):
        newSelectColumn = []
        for i, column in enumerate(self.columnList):
            if self.checkboxList[i].isChecked():
                newSelectColumn.append(column)
        self.selectColumn = newSelectColumn
        print('selectColumn: %s'%(newSelectColumn))

        self.renderLeftPlot()
        self.updateBtn()
    
    '''
    ==================================================
    左侧图表区域
    self.viewL GraphicsLayoutWidget
    self.leftPlotList list(PlotItem)
    ==================================================
    '''
    def createLeftPlot(self):
        viewL = pg.GraphicsLayoutWidget()
        self.viewL = viewL

        self.leftPlotList = []
        for column in self.columnList:
            plot = pg.PlotItem(title=column, name=column)
            index = len(self.leftPlotList)
            
            plot.plot(df['Col ' + str(index * 2 + 1)], df['Col ' + str(index * 2 + 2)], symbol='o', pen=pg.mkPen(index))
            # 可选 - 除了第一个plot，都 link 第一个 plot，让可视区域同步
            # if index > 0:
            #     plot.setXLink(self.leftPlotList[0])
            #     plot.setYLink(self.leftPlotList[0])
            self.leftPlotList.append(plot)

        self.bottom_layout.addWidget(viewL)

    def renderLeftPlot(self):
        # 清空已有 item
        self.viewL.clear()

        # 添加选中项对应图形
        for i in range(len(self.columnList)):
            if self.checkboxList[i].isChecked():
                self.viewL.addItem(self.leftPlotList[i])
                self.viewL.nextRow()
                self.leftPlotList[i].autoRange()

        if len(self.selectColumn) == 0:
            self.viewL.addLabel('Please select the column of interest above')

    '''
    ==================================================
    中间按钮
    self.computeBtn QPushButton
    ==================================================
    '''
    def createCenterBtn(self):
        btn = QPushButton('Train')
        btn.setFixedWidth(100)
        btn.setEnabled(False)
        btn.clicked.connect(self.handleCompute)
        self.bottom_layout.addWidget(btn)
        self.computeBtn = btn

    def handleCompute(self):
        print('start train...')
        self.isTarining = True
        self.updateBtn()

        # 模拟计算时间
        self.computeTimer.singleShot(1500, self.handleComputeFinish)

    def handleComputeFinish(self):
        print('finish train')
        self.isTarining = False
        self.updateBtn()

    def updateBtn(self):
        # enabled
        if len(self.selectColumn) == 0 or self.isTarining:
            self.computeBtn.setEnabled(False)
        else:
            self.computeBtn.setEnabled(True)
        
        # text
        if self.isTarining:
            self.computeBtn.setText('Training...')
        else:
            self.computeBtn.setText('Train')

    '''
    ==================================================
    右侧结果
    self.viewR PlotWidget
    ==================================================
    '''
    def createRightPlot(self):
        viewR = pg.PlotWidget()
        self.viewR = viewR

        n = 100
        scatterItem = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        pos = np.random.normal(size=(2,n), scale=100)
        spots = [{'pos': pos[:,i], 'data': i} for i in range(n)]
        scatterItem.addPoints(spots)
        self.scatterItem = scatterItem

        viewR.addItem(scatterItem)

        rect = viewR.viewRect()
        w = rect.width()
        h = rect.height()
        x = rect.x()
        y = rect.y()

        roi = pg.ROI([x + w * 0.4, y + h * 0.4], [w * 0.2, h * 0.2])
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
        # 左上 - 旋转
        # roi.addRotateHandle([0, 1], [0.5, 0.5])

        viewR.addItem(roi)

        roi.sigRegionChanged.connect(self.handleROIChange)

        self.handleROIChange(roi)

        self.bottom_layout.addWidget(viewR)

    def handleROIChange(self, roi: pg.ROI):
        pos: pg.Point = roi.pos()
        size: pg.Point = roi.size()

        self.selectRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.selectRect)
        print('points: %s'%([i.data() for i in points]))

        # 重置上次点的状态
        for p in self.lastChange:
            p.resetPen()

        # 修改点状态，并记录点
        for p in points:
            p.setPen(clickedPen)
        self.lastChange = points

if __name__ == '__main__':
    app = QApplication([])
    window = PlotWithInteraction()
    window.show()
    app.exec()        
