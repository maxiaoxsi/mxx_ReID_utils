import os
import sys
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileSystemModel, QTreeView, QListView, QLabel, QPushButton, 
                             QScrollArea, QSplitter, QToolBar, QSizePolicy, QLineEdit,
                             QComboBox, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                             QStackedWidget, QMessageBox, QStyledItemDelegate)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QTransform, QMouseEvent, QPainter, QColor, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QDir, QSize, QRect, QTimer, QThread, pyqtSignal


class ToolBar(QToolBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent=parent
        self.setIconSize(QSize(24, 24))
        
        # 添加工具栏按钮
        self._btn_prev = QPushButton("上一张")
        self._btn_next = QPushButton("下一张")
        self._btn_zoom_in = QPushButton("放大")
        self._btn_zoom_out = QPushButton("缩小")
        self._btn_fit = QPushButton("适应窗口")
        
        self.addWidget(self._btn_prev)
        self.addWidget(self._btn_next)
        self.addSeparator()
        self.addWidget(self._btn_zoom_in)
        self.addWidget(self._btn_zoom_out)
        self.addSeparator()
        self.addWidget(self._btn_fit)
        
        # 连接按钮信号
        self._btn_prev.clicked.connect(self._parent.show_previous_image)
        self._btn_next.clicked.connect(self._parent.show_next_image)
        self._btn_zoom_in.clicked.connect(self._parent.zoom_in)
        self._btn_zoom_out.clicked.connect(self._parent.zoom_out)
        self._btn_fit.clicked.connect(self._parent.fit_to_window)
