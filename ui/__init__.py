import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from .main_window import MainWindow
from .theme_manager import ThemeManager

def run_gui():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    ThemeManager.instance().load_from_config()
    theme = ThemeManager.instance().current
    app.setStyleSheet(f"""QMainWindow {{ background-color: {theme.bg_start.name()}; }}""")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
