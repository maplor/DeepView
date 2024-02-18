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

        # init main_layout & top_layout & bottom_layout
        self.initLayout()

        # TODO simulation training time, delete it after finish
        self.computeTimer = QTimer()

        self.columnList = ['AAA', 'BBB', 'CCC']
        self.selectColumn = []

        # status
        self.isTarining = False

        self.createCheckBoxList()

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
    top area checkbox: list
    - self.checkboxList list(QCheckBox)
    ==================================================
    '''
    def createCheckBoxList(self):
        self.checkboxList = []
        for column in self.columnList:
            cb = QCheckBox(column)
            self.top_layout.addWidget(cb)
            self.checkboxList.append(cb)
            cb.stateChanged.connect(self.handleCheckBoxStateChange)

        # add a stretch to fill the remaining area and keep the checkbox on the left
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
    bottom left area: plot
    - self.viewL GraphicsLayoutWidget
    - self.leftPlotList list(PlotItem)
    ==================================================
    '''
    def createLeftPlot(self):
        viewL = pg.GraphicsLayoutWidget()
        self.viewL = viewL
        self.bottom_layout.addWidget(viewL)

        self.leftPlotList = []
        for column in self.columnList:
            plot = pg.PlotItem(title=column, name=column)
            index = len(self.leftPlotList)
            
            plot.plot(df['Col ' + str(index * 2 + 1)], df['Col ' + str(index * 2 + 2)], symbol='o', pen=pg.mkPen(index))
            # optional - link plot viewbox range
            # if index > 0:
            #     plot.setXLink(self.leftPlotList[0])
            #     plot.setYLink(self.leftPlotList[0])
            self.leftPlotList.append(plot)

    def renderLeftPlot(self):
        # clear all items
        self.viewL.clear()

        # add plot item according to select column
        for i in range(len(self.columnList)):
            if self.checkboxList[i].isChecked():
                self.viewL.addItem(self.leftPlotList[i])
                self.viewL.nextRow()
                self.leftPlotList[i].autoRange()

        # placeholder
        if len(self.selectColumn) == 0:
            self.viewL.addLabel('Please select the column of interest above')

    '''
    ==================================================
    bottom center area: button
    - self.computeBtn QPushButton
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

        # simulation training time
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
    bottom right area: result plot
    - self.viewR PlotWidget
    - self.selectRect QRect
    - self.lastChange list(SpotItem)
    ==================================================
    '''
    def createRightPlot(self):
        viewR = pg.PlotWidget()
        self.viewR = viewR
        self.bottom_layout.addWidget(viewR)

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

        # create ROI
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

        # cache select region
        self.selectRect = QRectF(0, 0, 1, 1)

        # cache highlight point
        self.lastChange = []

        roi.sigRegionChanged.connect(self.handleROIChange)
        self.handleROIChange(roi)

    def handleROIChange(self, roi: pg.ROI):
        pos: pg.Point = roi.pos()
        size: pg.Point = roi.size()

        self.selectRect.setRect(pos.x(), pos.y(), size.x(), size.y())
        points = self.scatterItem.pointsAt(self.selectRect)
        print('points: %s'%([i.data() for i in points]))

        # reset last change points
        for p in self.lastChange:
            p.resetPen()

        # change point state and cache
        for p in points:
            p.setPen(clickedPen)
        self.lastChange = points

if __name__ == '__main__':
    app = QApplication([])
    window = PlotWithInteraction()
    window.show()
    app.exec()        
