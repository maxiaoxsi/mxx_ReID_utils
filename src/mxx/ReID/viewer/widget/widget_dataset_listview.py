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

from ..bar.bar_back import BackBar
from ..listview.listview_img import ImgListView
from ..listview.listview_person import PersonListView

class DatasetListViewWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._idx = 0

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0,0,0,0)

        self._bar_back_person = BackBar(self._parent)

        self._layout.addWidget(self._bar_back_person)
        
        self._widget_stacked = QStackedWidget()
        # 第一级：行人ID列表
        self._listview_person = PersonListView(self._parent)
        self._listview_img = ImgListView(self._parent)
        self._widget_stacked.addWidget(self._listview_person)
        self._widget_stacked.addWidget(self._listview_img)
        self._listview_focus = self._listview_person
        self._layout.addWidget(self._widget_stacked)
        
    def on_dataset_selected(self):
        self._listview_person.on_dataset_selected()

    def get_id_person_clicked(self, index):
        return self._listview_person.get_id(index)
    
    def refresh_person_selected(self, id_person, keys_img):
        self._listview_img.refresh_items(id_person, keys_img)
        self._widget_stacked.setCurrentIndex(1)
        self._idx = 1
        self._listview_focus = self._listview_img
        self._bar_back_person.setVisible(True)

    def set_stackwidget_person(self, idx):
        self._idx = idx
        self._widget_stacked.setCurrentIndex(idx)
        if idx == 0:
            self._bar_back_person.setVisible(False)
            self._listview_focus = self._listview_person
            self._listview_focus.setFocus()
        else:
            self._listview_focus = self._listview_img
            self._listview_focus.setFocus()



    def get_name_img_clicked(self, index):
        return self._listview_img.get_id(index)
    
    def set_focus(self):
        self._listview_focus.setFocus()

    # def keyPressEvent(self, event):
    #     """捕获键盘按键事件"""
    #     key = event.key()
    #     if key == Qt.Key_J:
    #         self._listview_focus.trigger_item_click_by_index()
    #     if key == Qt.Key_D:
    #         self._widget_main.set_focus("drn")
    #         self._focus = "drn"
    #         self.refresh_statusbar()
    #         return


        