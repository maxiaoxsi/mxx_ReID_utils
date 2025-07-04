import sys
from numpy import fft
import yaml
import traceback
from PyQt5.QtWidgets import QApplication

from mxx.ReID.viewer import ReIDViewer

if __name__ == '__main__':
    path_cfg = './configs/cfg_viewer.yaml'
    with open(path_cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    app = QApplication(sys.argv)
    # 设置应用程序属性，确保正常退出
    app.setQuitOnLastWindowClosed(True)
    
    try:
        viewer = ReIDViewer(cfg=cfg)
        viewer.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序启动错误: {traceback.format_exc()}")
        sys.exit(1)