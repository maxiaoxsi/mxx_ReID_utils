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


class BackBar(QToolBar):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent=parent
        self.setIconSize(QSize(24, 24))
        self._btn_back = QPushButton("返回")
        self._btn_back.setVisible(True)  # 初始隐藏返回按钮
        self.setVisible(False)
        self._btn_back.clicked.connect(self._parent.on_back_clicked)
        self.addWidget(self._btn_back)