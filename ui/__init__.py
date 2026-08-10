# ui/__init__.py
"""
UI 包入口模块。

提供图形界面应用程序的启动入口，包含高 DPI 适配、主题加载和主窗口初始化。
"""
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from .main_window import MainWindow
from .theme_manager import ThemeManager

def run_gui():
    """
    启动 PINN PDE 求解器的图形界面应用程序。

    执行流程：
        1. 启用高 DPI 缩放支持
        2. 创建 QApplication 实例
        3. 从配置文件加载主题偏好
        4. 应用主题样式到主窗口
        5. 创建并显示主窗口
        6. 进入应用程序事件循环

    :return: 无返回值（退出时返回系统退出码）
    """
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    ThemeManager.instance().load_from_config()
    theme = ThemeManager.instance().current
    app.setStyleSheet(f"""QMainWindow {{ background-color: {theme.bg_start.name()}; }}""")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
