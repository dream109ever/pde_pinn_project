# ui/main_window.py
"""
应用主窗口模块。

提供 PDE 科学计算平台的主窗口，管理页面切换和模式选择。
包含模式选择页、精确解析解求解页、PINN 输入页和 PINN 训练绘图页。
"""
import os
from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from PyQt5.QtGui import QIcon
from .pages.mode_selection_page import ModeSelectionPage
from .pages.solver_page import SolverPage
from .pages.pinn_input_page import PinnInputPage
from .pages.pinn_plot_page import PinnPlotPage

class MainWindow(QMainWindow):
    """应用主窗口，管理页面切换和模式选择。"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PDE 科学计算平台")
        self.resize(840, 560)
        icon_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "icons/window_icon.png"))
        if os.path.exists(icon_path): self.setWindowIcon(QIcon(icon_path))
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.mode_page = ModeSelectionPage()
        self.mode_page.mode_selected.connect(self.switch_mode)
        self.stack.addWidget(self.mode_page)
        self.solver_page = SolverPage(back_to_menu_cb=self.show_home)
        self.eq_input_page = PinnInputPage(back_to_menu_cb=self.show_home)
        self.eq_input_page.equation_configured.connect(self.go_to_solver_config)
        self.solver_plot_page = PinnPlotPage(back_to_input_cb=lambda: self.stack.setCurrentWidget(self.eq_input_page))
        self.stack.addWidget(self.solver_page)
        self.stack.addWidget(self.eq_input_page)
        self.stack.addWidget(self.solver_plot_page)
    def switch_mode(self, mode_type: str):
        """
        根据选择的模式切换到对应页面。

        :param mode_type: 模式类型，"exact" 表示精确解析解模式，"pinn" 表示 PINN 神经网络模式
        :type mode_type: str
        """
        if mode_type == "exact":
            self.stack.setCurrentWidget(self.solver_page)
        elif mode_type == "pinn":
            self.stack.setCurrentWidget(self.eq_input_page)
    def go_to_solver_config(self, result_data: dict):
        """
        当用户在 PINN 输入页面点击"下一页"时触发。

        :param result_data: 方程配置数据字典
        :type result_data: dict
        """
        self.solver_plot_page.set_problem_config(result_data)
        self.stack.setCurrentWidget(self.solver_plot_page)
    def show_home(self):
        """返回主页（模式选择页）。"""
        self.stack.setCurrentWidget(self.mode_page)
