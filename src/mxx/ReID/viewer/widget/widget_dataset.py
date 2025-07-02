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

from ..table.table_annot import AnnotTable
from .widget_dataset_listview import DatasetListViewWidget

class DatasetWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._splitter = QSplitter(Qt.Vertical)
        # show all the person and img in dataset
        self._imgList_datset_listview = DatasetListViewWidget(self._parent)
        # show annot 
        self._table_annot = AnnotTable(self._parent)
        

        self._splitter.addWidget(self._imgList_datset_listview)        
        self._splitter.addWidget(self._table_annot)
        self._splitter.setSizes([400, 400])

        self._layout.addWidget(self._splitter)

    def on_dataset_selected(self):
        self._imgList_datset_listview.on_dataset_selected()

    def get_id_person_clicked(self, index):
        return self._imgList_datset_listview.get_id_person_clicked(index)

    def refresh_person_selected(self, id_person, keys_img):
        self._imgList_datset_listview.refresh_person_selected(id_person, keys_img)

    def set_stackwidget_person(self, idx):
        self._imgList_datset_listview.set_stackwidget_person(idx)

    def get_name_img_clicked(self, index):
        return self._imgList_datset_listview.get_name_img_clicked(index)
    
    def refresh_img_selected(self, img_tgt):
        self._table_annot.set_img(img_tgt)

    def set_focus(self, id_widget):
        if id_widget == "person_and_img":
            self._imgList_datset_listview.set_focus()