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


class ReIDLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)  # 启用鼠标追踪
        self.original_pixmap = None
        self.scale_factor = 1.0
        self.image_name = ""  # 添加图片名属性
        
    def setImageName(self, name):
        """设置图片名"""
        self.image_name = name
        self.update()  # 触发重绘
        
    def paintEvent(self, event):
        """重写绘制事件，添加图片名标签"""
        super().paintEvent(event)
        if self.image_name:
            painter = QPainter(self)
            painter.setPen(Qt.white)  # 设置文字颜色为白色
            painter.setFont(QFont("Arial", 10))  # 设置字体
            
            # 创建半透明背景
            text_rect = QRect(0, 0, self.width(), 30)
            painter.fillRect(text_rect, QColor(0, 0, 0, 128))  # 半透明黑色背景
            
            # 绘制文字
            painter.drawText(text_rect, Qt.AlignCenter, self.image_name)
        
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self.original_pixmap:
            # 获取点击位置相对于图片的坐标
            pos = event.pos()
            
            # 获取图片在标签中的实际显示区域
            pixmap_rect = self.rect()
            pixmap_size = self.pixmap().size()
            
            # 计算图片在标签中的实际位置（居中显示）
            x_offset = (pixmap_rect.width() - pixmap_size.width()) // 2
            y_offset = (pixmap_rect.height() - pixmap_size.height()) // 2
            
            # 计算相对于图片的坐标
            x = pos.x() - x_offset
            y = pos.y() - y_offset
            
            # 检查点击是否在图片范围内
            if 0 <= x < pixmap_size.width() and 0 <= y < pixmap_size.height():
                # 计算原始图片中的坐标
                original_x = int(x / self.scale_factor)
                original_y = int(y / self.scale_factor)
                
                # 确保坐标在原始图片范围内
                if 0 <= original_x < self.original_pixmap.width() and 0 <= original_y < self.original_pixmap.height():
                    # 获取父窗口的状态栏
                    main_window = self.window()
                    if main_window:
                        # 获取点击位置的颜色
                        color = self.original_pixmap.toImage().pixelColor(original_x, original_y)
                        main_window.statusBar().showMessage(
                            f"点击位置: ({original_x}, {original_y}), 颜色: RGB({color.red()}, {color.green()}, {color.blue()})")




class AttributeDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_obj = None

    def createEditor(self, parent, option, index):
        if not self.img_obj:
            return None
            
        model = index.model()
        key_item = model.item(index.row(), 0)
        if not key_item:
            return None
        key = key_item.text()
        
        key_bool_list = []
        if hasattr(self.img_obj, 'get_key_bool_list'):
            key_bool_list = self.img_obj.get_key_bool_list()
        
        key_str_list = []
        if hasattr(self.img_obj, 'get_key_str_list'):
            key_str_list = self.img_obj.get_key_str_list()

        if key in key_bool_list:
            editor = QComboBox(parent)
            editor.addItems(["yes", "no"])
            return editor
        elif key in key_str_list:
            return QLineEdit(parent)
        else:
            return None

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.EditRole)
        if isinstance(editor, QComboBox):
            editor.setCurrentText(str(value))
        elif isinstance(editor, QLineEdit):
            editor.setText(str(value))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QComboBox):
            new_value = editor.currentText()
        elif isinstance(editor, QLineEdit):
            new_value = editor.text()
        else:
            super().setModelData(editor, model, index)
            return

        key_item = model.item(index.row(), 0)
        key = key_item.text()
        old_value = model.data(index, Qt.EditRole)

        if str(old_value) == new_value:
            return

        reply = QMessageBox.question(self.parent(), '确认修改', 
                                     f"确定要将属性 '{key}' 的值从 '{old_value}' 修改为 '{new_value}' 吗?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            try:
                self.img_obj[key] = new_value
                if hasattr(self.img_obj, 'save'):
                    self.img_obj.save()
                
                model.setData(index, new_value, Qt.EditRole)
                
                window = self.parent()
                if isinstance(window, QMainWindow):
                    window.statusBar().showMessage(f"属性 '{key}' 已更新为 '{new_value}'", 3000)
            except Exception as e:
                QMessageBox.critical(self.parent(), "错误", f"修改属性失败: {str(e)}")

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)
