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

from ..scrollarea.scrollarea_tgt import TgtScrollArea
from ..scrollarea.scrollarea_ref import RefScrollArea

class DisplayWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._splitter = QSplitter(Qt.Vertical)

        self._area_display_tgt = TgtScrollArea(self._parent)
        self._area_display_ref = RefScrollArea(self._parent)
        

        self._splitter.addWidget(self._area_display_tgt)
        self._splitter.addWidget(self._area_display_ref)
        self._splitter.setSizes([400, 400])
        self._layout.addWidget(self._splitter)

    def refresh_img_selected(self, img_tgt, img_ref_list):
        self._area_display_tgt.refresh_img_selected(
            img_tgt=img_tgt,
        )
        self._area_display_ref.refresh_img_selected(
            img_ref_list=img_ref_list,
        )