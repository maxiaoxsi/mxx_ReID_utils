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

        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setScaledContents(False)
        
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

    def set_img(self, img, type_img, label_img=None):
        try:
            if img is None or (type_img is not 'reid' and not img['is_smplx']):
                self.setText("without img")
                return
            # path_img = img.get_path(type_img)
            img_pil = img.get_img_pil(type_img)
            img_bytes = img_pil.tobytes('raw', 'RGB')
            w, h = img_pil.size
            qimage = QImage(img_bytes, w, h, QImage.Format_RGB888)
            pixmap_img = QPixmap(qimage)
            if pixmap_img.isNull():
                self.setText("can't load img")
                return
            self.setPixmap(pixmap_img)
            self._scale_factor = 1#self.scale_factor
            if label_img is not None:
                if label_img == 'name_img':
                    label_img = os.path.basename(path_img)
                self.setImageName(label_img)  # 设置图片名
            self.adjustSize()
        except Exception as e:
            self.setText(f"load err: {str(e)}")
            print(f"设置图片错误: {traceback.format_exc()}")

