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

from ..label.label_reid import ReIDLabel
from .scrollarea_base import ScrollAreaBase

class TgtScrollArea(ScrollAreaBase):
    def __init__(self, parent):
        super().__init__(parent)
        self._widget_display_tgt = self._init_widget_display_tgt()
        # 创建网格布局容器
        self.setWidget(self._widget_display_tgt)

    def _init_widget_display_tgt(self):
        widget_display_tgt = QWidget()
        layout_display_tgt = QGridLayout(widget_display_tgt)
        layout_display_tgt.setSpacing(10)  # 设置网格间距
        
        ''' display ref, manikin, skeleton, back, fore '''
        self._label_img_tgt = ReIDLabel() 
        layout_display_tgt.addWidget(self._label_img_tgt, 0, 0)  # row, col, rowspan, colspan
        self._label_img_manikin = ReIDLabel()
        layout_display_tgt.addWidget(self._label_img_manikin, 0, 1)
        self._label_img_skeleton = ReIDLabel()
        layout_display_tgt.addWidget(self._label_img_skeleton, 0, 2)
        self._label_img_background = ReIDLabel()
        layout_display_tgt.addWidget(self._label_img_background, 0, 3)
        self._label_img_foreground = ReIDLabel()
        layout_display_tgt.addWidget(self._label_img_foreground, 0, 4)
        return widget_display_tgt   
    
    def refresh_img_selected(self, img_tgt):
        self._label_img_tgt.set_img(img=img_tgt, type_img="reid")
        self._label_img_manikin.set_img(img=img_tgt, type_img="smplx_manikin")
        self._label_img_skeleton.set_img(img=img_tgt, type_img="smplx_skeleton")
        self._label_img_background.set_img(img=img_tgt, type_img="background")
        self._label_img_foreground.set_img(img=img_tgt, type_img="foreground")
            
    
