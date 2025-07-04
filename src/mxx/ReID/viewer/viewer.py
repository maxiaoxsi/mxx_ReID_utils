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

from .bar.bar_tool import ToolBar
from .widget.widget_main import MainWidget
from .widget.widget_search import SearchWidget
from ..dataset import ReIDDataset

class ReIDViewer(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.setWindowTitle('ReID Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化所有变量
        self.current_image_path = None
        self.image_labels = []  # 存储所有图像标签
        self.image_files = []
        self.current_image_index = -1
        self.scale_factor = 1.0
        self._dataset_dict = {}
        self._person_selected = None
        self._img_tgt = None
        self._img_ref_list = []
        self._img_previewed_right = None
        self._id_person_selected = -1
        self._drn = 'all'
        self._focus = 'main'

        

        # 初始化数据集
        for i, name_dataset in enumerate(cfg['dataset']['path_cfg_list']):
            path_cfg = cfg['dataset']['path_cfg_list'][name_dataset]
            print(path_cfg)
            try:
                dataset= ReIDDataset(
                    path_cfg=path_cfg,
                    img_size=(512, 512),
                    stage=1,           
                )
                self._dataset_dict[name_dataset] = dataset
                if i == 0:
                    self._dataset = dataset
            except Exception as e:
                QMessageBox.critical(self, "错误", f"dataset:{name_dataset}:\ncfg file {path_cfg} not found!\nException: {str(e)}")
                print(f"数据集初始化错误: {traceback.format_exc()}")
        
        # 初始化UI
        self.initUI()
        self.on_dataset_selected()
        
        # 设置定时器用于处理事件循环
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_events)
        self.timer.start(100)  # 每100ms处理一次事件

    def process_events(self):
        """处理Qt事件循环，防止界面卡死"""
        QApplication.processEvents()

    def initUI(self):
        # 主窗口部件和布局
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self._bar_tool = ToolBar(self)
        self._widget_search = SearchWidget(self)
        self._widget_main = MainWidget(self)
        layout.addWidget(self._bar_tool)
        layout.addWidget(self._widget_search)
        layout.addWidget(self._widget_main)
        self.setCentralWidget(widget)
        self.setFocusPolicy(Qt.StrongFocus)
        # self._widget_main.setFocus()
        self.setFocus()
 
    def on_person_dataset_selected(self, index):
        """点击左侧person id节点"""
        try:
            self._id_person_selected = index.data()
            if not self._dataset:
                QMessageBox.warning(self, "警告", "数据集未初始化")
                return
            
            self._person_selected = self._dataset.get_person(self._id_person_selected)
            self.statusBar().showMessage(f"选中行人ID: {self._id_person_selected}")
            keys_img = self._person_selected.keys
            self._widget_main.refresh_person_selected(self._id_person_selected, keys_img)
            # self._widget_search._combo_drn.setCurrentText("all")
            self.on_search_drn()
            self._widget_main.set_focus("person_and_img")
            self._focus="person_and_img"
            self.refresh_statusbar()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载行人信息失败: {str(e)}")
            print(f"加载行人信息错误: {traceback.format_exc()}")

    def on_img_drn_selected(self, index):
        """Click on an image in the top right list to preview."""
        try:
            # print(index.row())
            img = self._img_list_drn_selected[index.row()]
            self._widget_main.set_img_drn_selected(img)
        except Exception as e:
            QMessageBox.warning(self, "预览失败", f"无法加载图片预览: {str(e)}")
            print(f"预览图片错误: {traceback.format_exc()}")

    def on_dataset_selected(self):
        self._widget_main.on_dataset_selected()

    def on_img_selected(self, index):
        """点击详情视图中的图片项"""
        try:
            name_img = self._widget_main.get_name_img_clicked(index)
            
            if not self._person_selected:
                QMessageBox.warning(self, "警告", "未选择行人")
                return
                
            self._img_tgt = self._person_selected[name_img]
            # self.left_attr_delegate.img = self._img_tgt
            (
                img_ref_list,
                _,
                imgList_matched_dict
            ) = self._person_selected._get_imgList_from_img_set(
                stage=1,
                idx_img_tgt=name_img,
                is_select_bernl=False,
            )
            self._img_ref_list = img_ref_list
            self._imgList_drn_dict = imgList_matched_dict
            self._widget_main.refresh_img_selected(
                img_tgt=self._img_tgt,
                img_ref_list=self._img_ref_list,
                imgList_matched_dict=self._imgList_drn_dict,
            )
            # self.on_orientation_changed(self.combo_orientation_right.currentIndex())

            self.refresh_statusbar()
            # 展示选中的图片为5张图片
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败: {str(e)}")
            print(f"加载图片错误: {traceback.format_exc()}")

    def on_back_clicked(self):
        """点击返回按钮"""
        self._widget_main.set_stackwidget_person()

    

    def on_search(self):
        """处理搜索按钮点击事件"""
        person_id = self._widget_search.person_id_search
        drn = self._widget_search.person_drn_search

        self.refresh_statusbar()
        # 这里可以添加搜索逻辑

    def on_search_drn(self):
        """Handle orientation selection change for the right panel."""
        self._drn = self._widget_search.person_drn_search
        if not hasattr(self, '_imgList_drn_dict') or not self._imgList_drn_dict:
            return
        if self._drn == 'all' and 'all' not in self._imgList_drn_dict:
            self._imgList_drn_dict['all']  = sum(self._imgList_drn_dict.values(), [])
            self._img_list_drn_selected = self._imgList_drn_dict[self._drn]
        else:
            self._img_list_drn_selected = self._imgList_drn_dict[self._drn]
        # print(self._img_list_drn_selected)
        keys_img = [item.get_name() for item in self._img_list_drn_selected]
        self._widget_main.refresh_drn_selected(self._id_person_selected, keys_img=keys_img)
        self.refresh_statusbar()
    
    
        
    def refresh_statusbar(self):
        self.statusBar().showMessage(f"focus: {self._focus}, selected_person: {self._id_person_selected}, drn: {self._drn}")


    def keyPressEvent(self, event):
        """捕获键盘按键事件"""
        key = event.key()
        if key == Qt.Key_D:
            self._widget_main.set_focus("drn")
            self._focus = "drn"
            self.refresh_statusbar()
            return
        elif key == Qt.Key_P:
            print("1111")
            self._widget_main.set_focus("person_and_img")
            self._focus = "person&img"
            self.refresh_statusbar()
            return
        elif key == Qt.Key_M:
            self._widget_main.setFocus()
        elif key == Qt.Key_S:
            self._widget_search.set_focus("search")
        elif key == Qt.Key_J:
            # 按下J键的处理逻辑
            print("main")
            # self.on_key_j_pressed()
        elif key == Qt.Key_K:
            # 按下K键的处理逻辑
            print("main")
            # self.on_key_k_pressed()
        else:
            # 其他按键交给父类处理
            super().keyPressEvent(event)


















    
    
    
    def show_previous_image(self):
        """显示上一张图片"""
        if not self.image_files or self.current_image_index <= 0:
            return
        
        self.current_image_index -= 1
        self.current_image_path = self.image_files[self.current_image_index]
        self.load_image(self.current_image_path)
    
    def show_next_image(self):
        """显示下一张图片"""
        if not self.image_files or self.current_image_index >= len(self.image_files) - 1:
            return
        
        self.current_image_index += 1
        self.current_image_path = self.image_files[self.current_image_index]
        self.load_image(self.current_image_path)
    
    def zoom_in(self):
        """放大图片"""
        self.scale_image(1.25)
    
    def zoom_out(self):
        """缩小图片"""
        self.scale_image(0.8)
    
    def scale_image(self, factor):
        """缩放图片"""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
            
        self.scale_factor *= factor
        self.update_display_image()
    
    def update_display_image(self):
        """更新显示的图片（应用缩放和旋转）"""
        transform = QTransform()
        transform.scale(self.scale_factor, self.scale_factor)
        
        self.display_pixmap = self.original_pixmap.transformed(
            transform, Qt.SmoothTransformation)
        
        # 更新所有标签的显示
        for label in self.image_labels:
            label.setPixmap(self.display_pixmap)
            label.scale_factor = self.scale_factor
            label.adjustSize()
        
        # 调整滚动条位置
        self.adjust_scroll_bar(self.scroll_area.horizontalScrollBar(), 1.0)
        self.adjust_scroll_bar(self.scroll_area.verticalScrollBar(), 1.0)
    
    def adjust_scroll_bar(self, scroll_bar, factor):
        """调整滚动条位置"""
        scroll_bar.setValue(int(factor * scroll_bar.value() + ((factor - 1) * scroll_bar.pageStep() / 2)))
    
    def fit_to_window(self):
        """适应窗口大小"""
        if not hasattr(self, 'original_pixmap') or self.original_pixmap.isNull():
            return
        
        # 计算缩放比例
        viewport_size = self.scroll_area.viewport().size()
        pixmap_size = self.original_pixmap.size()
        
        scale_factor = min(viewport_size.width() / pixmap_size.width(),
                          viewport_size.height() / pixmap_size.height())
        
        self.scale_factor = scale_factor
        self.update_display_image()
    
    def closeEvent(self, event):
        """重写关闭事件，确保程序正常退出"""
        try:
            # 停止定时器
            if hasattr(self, 'timer'):
                self.timer.stop()
            
            # 清理资源
            if hasattr(self, '_dataset'):
                del self._dataset
            
            event.accept()
        except Exception as e:
            print(f"关闭事件错误: {traceback.format_exc()}")
            event.accept()

    @property
    def dataset(self):
        return self._dataset

    def set_focus(self, id_widget):
        if id_widget == "main":
            self.setFocus()
            self._focus = "main"
        self.refresh_statusbar()
