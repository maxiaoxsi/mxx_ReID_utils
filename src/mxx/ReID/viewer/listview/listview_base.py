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

class ListViewBase(QListView):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self._parent = parent
        self._model = QStandardItemModel()
        self.setModel(self._model)
        self._idx = 0

    def get_id(self, index):
        return self._model.itemFromIndex(index).text()
    
    def refresh_items(self, id_person, keys_items, is_trigger = True):
        self._model.clear()
        self._model.setHorizontalHeaderLabels([f"选中: {id_person}"])
        # 这里添加具体ID下的图片列表（示例数据）
        for key in keys_items:
            item = QStandardItem(key)
            self._model.appendRow(item)

        # 自动触发第一个元素的点击事件
        if is_trigger:
            self.trigger_first_item_click()

    # 之后处理：通过快捷键快速操作
    def keyPressEvent(self, event):
        """捕获键盘按键事件"""
        key = event.key()
        
        if key == Qt.Key_J:
            self.trigger_item_click_next()
        elif key == Qt.Key_K:
            self.trigger_item_click_prev()
        elif key == Qt.Key_Escape:
            self._parent.set_focus("main")
        else:
            super().keyPressEvent(event)
    
    def trigger_first_item_click(self):
        """触发第一个元素的点击事件"""
        if self._model.rowCount() > 0:
            # 创建第一个元素的索引
            first_index = self._model.index(0, 0)
            self._idx = 0
            # 设置当前选中项
            self.setCurrentIndex(first_index)
            # 触发点击事件
            self.clicked.emit(first_index)
            return True
        return False
    
    def trigger_item_click_next(self):
        if self.trigger_item_click_by_index(self._idx + 1):
            self._idx = self._idx + 1
    
    def trigger_item_click_prev(self):
        if self.trigger_item_click_by_index(self._idx - 1):
            self._idx = self._idx - 1

    def trigger_item_click_by_index(self, row_index):
        """根据索引触发指定元素的点击事件"""
        if 0 <= row_index < self._model.rowCount():
            # 创建指定元素的索引
            item_index = self._model.index(row_index, 0)
            # 设置当前选中项
            self.setCurrentIndex(item_index)
            # 触发点击事件
            self.clicked.emit(item_index)
            return True
        return False

    
    def trigger_first_item_click_delayed(self, delay_ms=100):
        """延迟触发第一个元素的点击事件"""
        if self._model.rowCount() > 0:
            QTimer.singleShot(delay_ms, self.trigger_first_item_click)
            return True
        return False