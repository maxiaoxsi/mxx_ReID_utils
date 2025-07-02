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
from ..listview.listview_drn import DrnListView
from ..table.table_annot import AnnotTable

class DrnWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._parent = parent

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        
        self._splitter = QSplitter(Qt.Vertical)

        # Top widget with combo box and tree view
        self._listView_drn = DrnListView(self._parent)
        
        self._label_img_drn_selected = ReIDLabel()
        self._label_img_drn_selected.setText("点击上方列表中的图片进行预览")

        self._table_annot = AnnotTable(self._parent)
        
        self._splitter.addWidget(self._listView_drn)        
        self._splitter.addWidget(self._label_img_drn_selected)
        self._splitter.addWidget(self._table_annot)
        self._splitter.setSizes([250, 150, 400]) # Initial sizes
        self._layout.addWidget(self._splitter)

    
    def refresh_drn_selected(self, id_person, keys_img):
        self._label_img_drn_selected.clear()
        self._table_annot.clear()
        self._listView_drn.refresh_items(id_person=id_person, keys_items=keys_img)

    def get_id_img_drn_selected(self, index):
        return self._listView_drn.get_id(index)
    
    def set_img_selected(self, img):
        self._label_img_drn_selected.set_img(img=img, type_img="reid")
        self._table_annot.set_img(img)

    def set_focus(self, id_widget):
        if id_widget == "drn":
            self._listView_drn.setFocus()
        