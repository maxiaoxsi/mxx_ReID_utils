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

from .widget_dataset import DatasetWidget
from .widget_display import DisplayWidget
from .widget_drn import DrnWidget 

class MainWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._layout = QHBoxLayout(self)
        
        # 设置可以接收键盘事件
        self.setFocusPolicy(Qt.StrongFocus)

        # 使用QSplitter实现可调整大小的左右分割
        splitter = QSplitter(Qt.Horizontal)

        # widget dataset on left, include img_list and annot_table
        self._widget_dataset = DatasetWidget(self._parent)
        # widget for img display
        self._widget_display = DisplayWidget(self._parent)
        # widget for direction img list of selected img
        self._widget_drn = DrnWidget(self._parent)
        
        # 添加所有部件到分割器
        splitter.addWidget(self._widget_dataset)
        splitter.addWidget(self._widget_display)
        splitter.addWidget(self._widget_drn)

        # 设置分割比例
        splitter.setSizes([300, 600, 300])
        # 主布局
        self._layout.addWidget(splitter)

    def get_id_person_clicked(self, index):
        return self._widget_dataset.get_id_person_clicked(index)
    
    def get_id_img_drn_selected(self, index):
        return self._widget_drn.get_id_img_drn_selected(index)

    def refresh_person_selected(self, id_person, keys_img):
        self._widget_dataset.refresh_person_selected(id_person, keys_img)

    def refresh_drn_selected(self, id_person, keys_img):
        self._widget_drn.refresh_drn_selected(id_person, keys_img)

    def set_stackwidget_person(self):
        self._widget_dataset.set_stackwidget_person(0)

    def get_name_img_clicked(self, index):
        return self._widget_dataset.get_name_img_clicked(index)
    
    def refresh_img_selected(self, img_tgt, img_ref_list, imgList_matched_dict):
        self._widget_dataset.refresh_img_selected(
            img_tgt=img_tgt,
        )
        self._widget_display.refresh_img_selected(
            img_tgt=img_tgt,
            img_ref_list=img_ref_list,
        )

    def set_img_drn_selected(self, img_drn):
        self._widget_drn.set_img_selected(img=img_drn)

    def on_dataset_selected(self):
        self._widget_dataset.on_dataset_selected()

    def set_focus(self, id_widget):
        if id_widget == "drn":
            self._widget_drn.set_focus("drn")
        elif id_widget == "person_and_img":
            self._widget_dataset.set_focus("person_and_img")


    
    