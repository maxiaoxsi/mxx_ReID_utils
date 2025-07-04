import os
import sys
import traceback
import yaml
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QFileSystemModel, QTreeView, QListView, QLabel, QPushButton, 
                             QScrollArea, QSplitter, QToolBar, QSizePolicy, QLineEdit,
                             QComboBox, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                             QStackedWidget, QMessageBox, QStyledItemDelegate)
from PyQt5.QtGui import QPixmap, QImage, QPalette, QTransform, QMouseEvent, QPainter, QColor, QFont, QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt, QDir, QSize, QRect, QTimer, QThread, pyqtSignal

from ..label.label_reid import ReIDLabel
from ...dataset import ReIDDataset

from ..combobox.combobox_annot_item import AnnotItemComboBox

class AnnotTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self._img = None
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["属性", "值"])
        self.horizontalHeader().setStretchLastSection(True)
        self.setAlternatingRowColors(True)  # 交替行颜色，更易读
        
        # 设置行高相关属性
        self.verticalHeader().setDefaultSectionSize(25)  # 设置默认行高为25像素
        self.verticalHeader().setMinimumSectionSize(20)  # 设置最小行高为20像素
        self.verticalHeader().setMaximumSectionSize(50)  # 设置最大行高为50像素
        
        # 设置列宽
        self.setColumnWidth(0, 120)  # 第一列（属性列）宽度150像素
        self.setColumnWidth(1, 130)  # 第二列（值列）宽度200像素
    
    def set_img(self, img):
        self._img = img
        key_bool_list = []
        key_str_list = []

        if hasattr(img.annot, 'get_key_str_list'):
            key_bool_list = img.annot.get_key_bool_list()
        if hasattr(img.annot, 'get_key_str_list'):
            key_str_list = img.annot.get_key_str_list()
        self.setRowCount(len(key_bool_list) + len(key_str_list))

        # print(key_bool_list)
        for i, key in enumerate(key_bool_list):
            key_item = QTableWidgetItem(str(key))
            # key_item.setEditable(False)
            self.setItem(i, 0, key_item)
            combo_box = AnnotItemComboBox(self, img[key])
            self.setCellWidget(i, 1, combo_box)
            self.setRowHeight(i, 25) 
        

        for i, key in enumerate(key_str_list):
            key_item = QTableWidgetItem(str(key))
            # key_item.setEditable(False)
            self.setItem(len(key_bool_list) + i, 0, key_item)
            value_item = QTableWidgetItem(str(img[key]))
            # value_item.setEditable(True)
            self.setItem(len(key_bool_list) + i, 1, value_item)
            self.setRowHeight(len(key_bool_list) + i, 25)  # 背包行设置为35像素


    def set_row_height(self, row, height):
        """设置指定行的高度"""
        if 0 <= row < self.rowCount():
            self.setRowHeight(row, height)
    
    def set_all_rows_height(self, height):
        """设置所有行的高度"""
        for row in range(self.rowCount()):
            self.setRowHeight(row, height)
    
    def auto_adjust_row_heights(self):
        """根据内容自动调整行高"""
        for row in range(self.rowCount()):
            # 获取该行所有单元格的内容
            max_height = 25  # 最小高度
            
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item:
                    # 根据文本长度估算所需高度
                    text = item.text()
                    if len(text) > 20:
                        max_height = max(max_height, 35)
                    elif len(text) > 10:
                        max_height = max(max_height, 30)
            
            self.setRowHeight(row, max_height)
    
    def on_cell_clicked(self, row, column):
        """处理单元格点击事件"""
        print(f"点击了第 {row + 1} 行，第 {column + 1} 列")
        
        # 获取点击的单元格内容
        item = self.item(row, column)
        if item:
            print(f"单元格内容: {item.text()}")
        
        # 如果是第一列（属性列），可以在这里添加特殊处理
        if column == 0:
            print(f"点击了属性: {item.text() if item else 'None'}")
        
        # 如果是第二列（值列），可以在这里添加特殊处理
        elif column == 1:
            print(f"点击了值: {item.text() if item else 'None'}")
            
            # 如果该单元格有自定义控件（如ComboBox），可以获取其值
            cell_widget = self.cellWidget(row, column)
            if cell_widget and hasattr(cell_widget, 'currentText'):
                print(f"控件当前值: {cell_widget.currentText()}")