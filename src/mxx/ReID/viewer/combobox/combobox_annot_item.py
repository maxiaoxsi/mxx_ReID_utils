import os
import sys
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileSystemModel, QTreeView, QLabel, QPushButton, 
                             QScrollArea, QSplitter, QToolBar, QSizePolicy, QLineEdit,
                             QComboBox, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                             QStackedWidget, QMessageBox, QStyledItemDelegate)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QTransform, QMouseEvent, QPainter, QColor, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QDir, QSize, QRect, QTimer, QThread, pyqtSignal

class AnnotItemComboBox(QComboBox):
    def __init__(self, parent, key):
        super().__init__(parent)
        self._parent = parent
        self.addItems(["True", "False"])
        if key in [True, False]:
            if key == True:
                self.setCurrentText("True")
            else:
                self.setCurrentText("False")
            return
        elif key in ["True", "False"]:
            self.setCurrentText(key)
            return
        else:
            raise Exception(f"AnnotItemComboBox got wrong key:{key}")