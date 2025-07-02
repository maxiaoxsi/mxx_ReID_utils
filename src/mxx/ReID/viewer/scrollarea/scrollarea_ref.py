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

class RefScrollArea(ScrollAreaBase):
    def __init__(self, parent):
        super().__init__(parent)
        self._widget_display_ref = self._init_widget_display_ref()
        # 创建网格布局容器
        self.setWidget(self._widget_display_ref)

    def _init_widget_display_ref(self):
        widget_display_ref = QWidget()
        layout_display_ref = QGridLayout(widget_display_ref)
        layout_display_ref.setSpacing(10)  # 设置网格间距
        
        ''' display ref, smplx, skeleton, back, fore '''
        self._label_img_ref_list = []
        # 4 ref imgs
        for i in range(4):
            image_label = ReIDLabel() 
            self._label_img_ref_list.append(image_label)
            layout_display_ref.addWidget(image_label, 0, i)
        return widget_display_ref
    
    def refresh_img_selected(self, img_ref_list):
        for (img, label) in zip(img_ref_list, self._label_img_ref_list):
                label.set_img(img=img, type_img="reid")

            