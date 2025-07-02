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


class SearchWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setFixedHeight(30)  # 设置固定高度为30像素
        self._init_layout()
        # person id
        self._init_id()
        # img drn
        self._init_drn()          
        self._layout.addWidget(self._label_id)
        self._layout.addWidget(self._input_id)
        self._layout.addWidget(self._label_drn)
        self._layout.addWidget(self._combo_drn)
        self._layout.addStretch()
        

    def _init_layout(self):
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(5, 5, 5, 5)
        self._layout.setAlignment(Qt.AlignVCenter)  # 垂直居中对齐

    def _init_drn(self):
        self._label_drn = QLabel("direction:")
        self._label_drn.setAlignment(Qt.AlignVCenter)
        self._label_drn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._label_drn.setFixedHeight(20)
        self._combo_drn = QComboBox()
        self._combo_drn.addItems(["all", "front", "back", "left", "right"])
        self._combo_drn.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._combo_drn.setFixedHeight(20)
        # 添加选择触发搜索功能
        self._combo_drn.currentIndexChanged.connect(self._parent.on_search_drn)
        
    def _init_id(self):
        self._label_id = QLabel("行人ID:")
        self._label_id.setAlignment(Qt.AlignVCenter)
        self._label_id.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._label_id.setFixedHeight(20)
        self._input_id = QLineEdit()
        self._input_id.setPlaceholderText("输入行人ID")
        self._input_id.setAlignment(Qt.AlignVCenter)
        self._input_id.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._input_id.setFixedHeight(20)
        self._input_id.setFocusPolicy(Qt.ClickFocus)
        # 添加回车键触发搜索功能
        self._input_id.returnPressed.connect(self._parent.on_search)
        
    def set_focus(self, id_widget):
        if id_widget == "search":
            self._input_id.setFocus()
        return
        
    @property
    def person_id_search(self):
        return self._input_id.text()
    
    @property
    def person_drn_search(self):
        return self._combo_drn.currentText()