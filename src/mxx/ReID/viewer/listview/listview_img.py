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

from .listview_base import ListViewBase

class ImgListView(ListViewBase):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.clicked.connect(self._parent.on_img_selected)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_B:
            self._parent.on_back_clicked()
            return
        super().keyPressEvent(event=event)
            
    