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

from .listview_base import ListViewBase

class PersonListView(ListViewBase):
    def __init__(self, parent):
        super().__init__(parent)

        self.clicked.connect(self._parent.on_person_dataset_selected)
        
        # 安全地获取行人ID列表
        dataset = self._parent.dataset
        try:
            if dataset:
                keys_id_person = dataset.get_person_keys()
                # 将ID转换为整数进行排序
                keys_sorted = sorted(keys_id_person, key=lambda x: int(x))
                self.refresh_items("加载数据集", keys_sorted, is_trigger=False)
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载行人ID列表失败: {str(e)}")
            print(f"加载行人ID列表错误: {traceback.format_exc()}")


    def on_dataset_selected(self):
        dataset = self._parent.dataset
        try:
            if dataset:
                keys_id_person = dataset.get_person_keys()
                # 将ID转换为整数进行排序
                keys_sorted = sorted(keys_id_person, key=lambda x: int(x))
                self.refresh_items("加载数据集", keys_sorted)
                self.click_item()
        except Exception as e:
            QMessageBox.warning(self, "警告", f"加载行人ID列表失败: {str(e)}")
            print(f"加载行人ID列表错误: {traceback.format_exc()}")

    def trigger_item_click_by_index(self, row_index):
        """根据索引触发指定元素的点击事件"""
        if 0 <= row_index < self._model.rowCount():
            # 创建指定元素的索引
            item_index = self._model.index(row_index, 0)
            # 设置当前选中项
            self.setCurrentIndex(item_index)
            return True
        return False
    
    def click_item(self):
        item_index = self._model.index(self._idx, 0)
        self.setCurrentIndex(item_index)
        self.clicked.emit(item_index)

    def keyPressEvent(self, event):
        """捕获键盘按键事件"""
        key = event.key()
        if key == Qt.Key_Enter or key == Qt.Key_Return:
            self.click_item()
            return
        super().keyPressEvent(event=event)
        