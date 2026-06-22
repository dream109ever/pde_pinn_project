# src/gui.py
import os
import sys
import torch
import numpy as np
import sympy as sp
import qdarkstyle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import uic
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QBrush, QPainter, QColor
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QSpinBox, QDoubleSpinBox, QLabel, QComboBox, QGroupBox,
                             QFileDialog, QMessageBox, QLineEdit, QTabWidget, QCheckBox, QStackedWidget, QFrame, QScrollArea, QSizePolicy, QTextEdit, QApplication)
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from .function_factory import PDEConfig, BoundaryCondition, ConditionConfig, parse_expression, generate_analytical_solution
from .data_utils import DomainSampler
from .network_factory import build_model, build_network, ComplexityAnalyzer, NetworkConfigGenerator
from .trainer import PINNTrainer
from .data_utils import DomainSampler
from .visualization import plot_loss_history, plot_solution

def resource_path(relative_path):
    try:
        # 打包后：从临时目录 _MEIPASS 读取
        base_path = sys._MEIPASS
    except AttributeError:
        # 开发模式：项目根目录（src 的父目录）
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def data_path(relative_path):
    """
    获取可写数据文件的路径（如 results, config）
    在打包后，放在可执行文件同级目录；开发时放在项目根目录
    """
    if getattr(sys, 'frozen', False):
        # 打包后：可执行文件所在目录
        base_path = os.path.dirname(sys.executable)
    else:
        # 开发模式：项目根目录（src 的父目录）
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class WelcomePage(QWidget):
    """欢迎页（主界面）"""
    def __init__(self):
        super().__init__()

        bg_path = resource_path(os.path.join('ui', 'welcome_page.png'))
        if os.path.exists(bg_path):
            self.background_pixmap = QPixmap(bg_path)
        else:
            self.background_pixmap = None
        
        self.init_ui()

    def paintEvent(self, event):
        """重写绘图事件，绘制背景图片"""
        painter = QPainter(self)
        if self.background_pixmap is not None and not self.background_pixmap.isNull():
            # 缩放图片以完全填充窗口（保持宽高比，裁剪边缘）
            scaled = self.background_pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
            # 计算绘制位置：居中显示
            rect = scaled.rect()
            rect.moveCenter(self.rect().center())
            painter.drawPixmap(rect, scaled)
        else:
            # 无图片时使用纯色背景
            painter.fillRect(self.rect(), QColor(240, 240, 240))
        # 调用父类 paintEvent 确保子控件正常绘制
        super().paintEvent(event)

    def init_ui(self): 
        layout = QVBoxLayout(self)
        layout.addStretch()
        title = QLabel("PINN 偏微分方程求解器")
        title.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background: transparent; color: #2c3e50; margin-bottom: 40px;")
        layout.addWidget(title)
        subtitle = QLabel("基于物理信息神经网络")
        subtitle.setFont(QFont("Microsoft YaHei", 14))
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("background: transparent; color: #7f8c8d; margin-bottom: 60px;")
        layout.addWidget(subtitle)
        self.start_btn = QPushButton("开始使用")
        self.start_btn.setFixedSize(200, 60)
        self.start_btn.setFont(QFont("Microsoft YaHei", 16))
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 30px;
                padding: 10px 20px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #1f618d;
            }
        """)
        layout.addWidget(self.start_btn, alignment=Qt.AlignCenter)
        layout.addStretch()
        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(20, 0, 20, 0)
        left_info_layout = QVBoxLayout()
        left_info_layout.setSpacing(2)
        version = QLabel("version: 1.0.0")
        version.setFont(QFont("Microsoft YaHei", 10))
        version.setStyleSheet("background: transparent; color: #5a6a7a;")
        left_info_layout.addWidget(version)
        author = QLabel("by: dream109ever")
        author.setFont(QFont("Microsoft YaHei", 10))
        author.setStyleSheet("background: transparent; color: #5a6a7a;")
        left_info_layout.addWidget(author)
        bottom_layout.addLayout(left_info_layout)
        bottom_layout.addStretch()
        self.exit_btn = QPushButton("退出")
        self.exit_btn.setFixedSize(100, 40)
        self.exit_btn.setFont(QFont("Microsoft YaHei", 12))
        self.exit_btn.clicked.connect(self.close_application)
        bottom_layout.addWidget(self.exit_btn)
        layout.addLayout(bottom_layout)
    def close_application(self):
        QApplication.quit()

class EquationPage(QWidget):
    def __init__(self):
        super().__init__()
        ui_path = resource_path(os.path.join('ui', 'equationpage.ui'))
        uic.loadUi(ui_path, self)
        self.preset_combo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        for group in [self.eq_group, self.field_group, self.bc_group]:
            group.setStyleSheet("QGroupBox { margin-top: 20px; }")
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        self.has_t_check.stateChanged.connect(self.on_time_toggled)
        self.back_button.setFixedSize(100, 40)
        self.next_page_button.setFixedSize(100, 40)
        self.bottom_layout.setContentsMargins(20, 0, 20, 12)
        self.load_preset(self.preset_combo.currentText())
        self.connect_signals()
        self._cond_edit_map = {
            self.bc_x0_alpha: 'x0_alpha',
            self.bc_x0_beta: 'x0_beta',
            self.bc_x0_gamma: 'x0_gamma',
            self.bc_x1_alpha: 'x1_alpha',
            self.bc_x1_beta: 'x1_beta',
            self.bc_x1_gamma: 'x1_gamma',
            self.bc_y0_alpha: 'y0_alpha',
            self.bc_y0_beta: 'y0_beta',
            self.bc_y0_gamma: 'y0_gamma',
            self.bc_y1_alpha: 'y1_alpha',
            self.bc_y1_beta: 'y1_beta',
            self.bc_y1_gamma: 'y1_gamma',
            self.ic_displacement_alpha: 'ic_alpha',
            self.ic_displacement_beta: 'ic_beta',
            self.ic_displacement_gamma: 'ic_gamma',
        }
    @staticmethod
    def _parse_expr(text: str):
        """解析表达式：尝试转为浮点数，否则返回字符串"""
        text = text.strip()
        if text == '':
            return 0
        try:
            return float(text)
        except ValueError:
            return text
    def _get_coeff_name(self, edit):
        for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            if getattr(self, f"{key}_edit", None) == edit:
                return key
        return None
    def _get_coeff_edits(self):
        """返回所有系数输入框控件列表"""
        return [self.A_edit, self.B_edit, self.C_edit, self.D_edit, self.E_edit, self.F_edit, self.G_edit]
    def _get_range_edits(self):
        """返回所有定义域范围输入框列表"""
        return [
            self.x_min_edit, self.x_max_edit, self.y_min_edit, self.y_max_edit, self.t_min_edit, self.t_max_edit
        ]
    def _get_condition_edits(self):
        """返回所有条件系数输入框列表"""
        return [
            self.bc_x0_alpha, self.bc_x0_beta, self.bc_x0_gamma, self.bc_x1_alpha, self.bc_x1_beta, self.bc_x1_gamma, 
            self.bc_y0_alpha, self.bc_y0_beta, self.bc_y0_gamma, self.bc_y1_alpha, self.bc_y1_beta, self.bc_y1_gamma, 
            self.ic_displacement_alpha, self.ic_displacement_beta, self.ic_displacement_gamma
        ]
    def load_preset(self, preset_name: str):
        """根据预设类型填充系数，并自动设置时间复选框状态"""
        presets = {
            "自定义": ({}, False),
            "一维波动 (u_tt - a^2 u_xx = 0)": ({'A': '-1', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'F': '0', 'G': '0'}, True),
            "一维有源波动 (u_tt - a^2 u_xx = G)": ({'A': '-1', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'F': '0', 'G': 'sin(pi*x)*sin(pi*t)'}, True),
            "一维阻尼波动 (u_tt + 2b u_t - a^2 u_xx = 0)": ({'A': '-1', 'B': '0', 'C': '1', 'D': '0', 'E': '2', 'F': '0', 'G': '0'}, True),
            "电报方程 (u_tt + 2b u_t - a^2 u_xx + c u = 0)": ({'A': '-1', 'B': '0', 'C': '1', 'D': '0', 'E': '2', 'F': '1', 'G': '0'}, True),
            "一维热传导 (u_t - a^2 u_xx = 0)": ({'A': '-1', 'B': '0', 'C': '0', 'D': '0', 'E': '1', 'F': '0', 'G': '0'}, True),
            "一维有源热传导 (u_t - a^2 u_xx = G)": ({'A': '-1', 'B': '0', 'C': '0', 'D': '0', 'E': '1', 'F': '0', 'G': 'sin(pi*x)*sin(pi*t)'}, True),
            "二维拉普拉斯 (u_xx+u_yy = 0)": ({'A': '1', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'F': '0', 'G': '0'}, False),
            "二维泊松 (u_xx+u_yy = G)": ({'A': '1', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'F': '0', 'G': 'sin(pi*x)*sin(pi*y)'}, False),
            "亥姆霍兹 (u_xx + u_yy + a^2 u = 0)": ({'A': '1', 'B': '0', 'C': '1', 'D': '0', 'E': '0', 'F': '1', 'G': '0'}, False),
        }
        if preset_name not in presets:
            return
        coeff_dict, has_t = presets[preset_name]
        for key, value in coeff_dict.items():
            edit = getattr(self, f"{key}_edit", None)
            if edit is not None:
                edit.setText(value)
        self.has_t_check.blockSignals(True)
        self.has_t_check.setChecked(has_t)
        self.has_t_check.blockSignals(False)
        self.update_editable_fields(preset_name, coeff_dict)
        self.update_time_dependent_ui(has_t)
    def on_time_toggled(self, state):
        """用户手动勾选/取消时间复选框时的响应"""
        has_t = (state == Qt.Checked)
        self.update_time_dependent_ui(has_t)
    def update_editable_fields(self, preset_name: str, coeff_dict: dict):
        """
        根据预设类型控制控件可编辑性：
        - 内置预设：禁用时间复选框，并禁止修改系数为0的输入框
        - 自定义：启用时间复选框，恢复所有系数输入框可编辑
        """
        is_custom = (preset_name == "自定义")
        self.has_t_check.setEnabled(is_custom)
        for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            edit = getattr(self, f"{key}_edit", None)
            if edit is None:
                continue
            if is_custom:
                edit.setReadOnly(False)
                edit.setEnabled(True)
                edit.setToolTip("")
                edit.setStyleSheet("")
            else:
                val = coeff_dict.get(key, '0')
                if val == '0' and not is_custom:
                    edit.setEnabled(False)
                    edit.setToolTip("内置函数类型不支持修改无关系数")
                else:
                    edit.setEnabled(True)
                    edit.setToolTip("")
    def update_time_dependent_ui(self, has_t: bool):
        """
        根据是否包含时间，更新界面控件的可用性和提示。
        - 含时间时：禁用 y 范围输入框和 y 边界标签页，并添加悬停提示。
        - 不含时间：恢复 y 范围输入框和 y 边界标签页。
        """
        for edit in [self.y_min_edit, self.y_max_edit]:
            edit.setEnabled(not has_t)
            edit.setToolTip("包含时间时，暂不支持 y 方向范围" if has_t else "")
        for edit in [self.t_min_edit, self.t_max_edit]:
            edit.setEnabled(has_t)
            edit.setToolTip("不含时间时，无时间范围" if has_t else "")
        tab_widget = self.tabWidget
        if has_t:
            self.B_label.setText("B (u_xt)")
            self.C_label.setText("C (u_tt)")
            self.E_label.setText("E (u_t)")
        else:
            self.B_label.setText("B (u_xy)")
            self.C_label.setText("C (u_yy)")
            self.E_label.setText("E (u_y)")
        for i in range(tab_widget.count()):
            if tab_widget.tabText(i) == "空间边界y":
                tab_widget.setTabEnabled(i, not has_t)
                tab_widget.setTabToolTip(i, "包含时间时，暂不支持 y 方向边界" if has_t else "")
                break
        for i in range(tab_widget.count()):
            if tab_widget.tabText(i) == "初始条件":
                tab_widget.setTabEnabled(i, has_t)
                tab_widget.setTabToolTip(i, "仅当包含时间维度时可用" if not has_t else "")
                break
        self.has_t_check.setToolTip("当前方程包含时间维度" if has_t else "不含时间维度")
    def connect_signals(self):
        """为输入框连接 textChanged 和 editingFinished 信号"""
        for edit in self._get_range_edits():
            edit.textChanged.connect(self.on_range_changed)
            edit.returnPressed.connect(lambda e=edit: self._on_range_return_pressed(e))
        for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            edit = getattr(self, f"{key}_edit", None)
            if edit is not None:
                edit.returnPressed.connect(self._on_coeff_return_pressed)
        for edit in self._get_condition_edits():
            if edit is not None:
                edit.editingFinished.connect(lambda e=edit: self.validate_conditions(None, False))
                edit.returnPressed.connect(lambda e=edit: self._on_condition_return_pressed(e, None))
    def on_range_changed(self):
        """当任意范围输入框内容变化时，更新边界分组框标题"""
        try:
            x_min = float(self.x_min_edit.text()) if self.x_min_edit.text() else 0.0
            x_max = float(self.x_max_edit.text()) if self.x_max_edit.text() else 1.0
            y_min = float(self.y_min_edit.text()) if self.y_min_edit.text() else 0.0
            y_max = float(self.y_max_edit.text()) if self.y_max_edit.text() else 1.0
        except ValueError:
            return
        self.groupBox_x0.setTitle(f"x = {x_min:.2g}")
        self.groupBox_x1.setTitle(f"x = {x_max:.2g}")
        if self.y_min_edit.isEnabled():
            self.groupBox_y0.setTitle(f"y = {y_min:.2g}")
            self.groupBox_y1.setTitle(f"y = {y_max:.2g}")
    def validate_range(self, edit=None):
        focused = self.focusWidget()
        pairs = [ (self.x_min_edit, self.x_max_edit, 'x'), (self.y_min_edit, self.y_max_edit, 'y'), (self.t_min_edit, self.t_max_edit, 't'), ]
        for min_edit, max_edit, axis in pairs:
            if not min_edit.isEnabled():
                continue
            try:
                float(min_edit.text().strip())
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"{axis}_min 必须是合法数字")
                min_edit.setFocus()
                min_edit.selectAll()
                return False
            try:
                float(max_edit.text().strip())
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"{axis}_max 必须是合法数字")
                max_edit.setFocus()
                max_edit.selectAll()
                return False
        for min_edit, max_edit, axis in pairs:
            if not min_edit.isEnabled():
                continue
            min_val = float(min_edit.text().strip())
            max_val = float(max_edit.text().strip())
            if min_val >= max_val:
                QMessageBox.warning(self, "范围错误", f"{axis}_min 必须小于 {axis}_max")
                if focused not in [min_edit, max_edit]:
                    min_edit.setFocus()
                self.focusWidget().selectAll()
                return False
        return True
    def _on_range_return_pressed(self, edit):
        if self.validate_range(edit):
            edit.deselect()
            self.setFocus()
    def validate_coeffs(self, name: str, next: bool):
        """
        检查系数输入框的内容：
        - 空字符串：若 next=True 则报错，否则跳过
        - 非空：用 sympy 校验是否为合法表达式，且只包含 x, y 及已知函数
        """
        focused = self.focusWidget()
        preset_name = self.preset_combo.currentText()
        is_custom = (preset_name == "自定义")
        for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            edit = getattr(self, f"{key}_edit", None)
            text = edit.text().strip()
            if text == "":
                if next:
                    QMessageBox.warning(self, "输入错误", f"系数 {key} 的表达式不能为空")
                    if focused != edit:
                        edit.setFocus()
                    self.focusWidget().selectAll()
                    return False
                continue
            if not is_custom and key != 'G':
                try:
                    float(text)
                except ValueError:
                    QMessageBox.warning(self, "输入错误", f"系数 {key} 必须输入数值，不允许表达式")
                    if focused != edit:
                        edit.setFocus()
                    self.focusWidget().selectAll()
                    return False
                continue
            try:
                if self.has_t_check.isChecked():
                    x_sym, t_sym = sp.symbols('x t')
                    allowed = {x_sym, t_sym}
                else:
                    x_sym, y_sym = sp.symbols('x y')
                    allowed = {x_sym, y_sym}
                sp_expr = sp.sympify(text)
                free_syms = sp_expr.free_symbols
                extra = free_syms - allowed
                if extra:
                    QMessageBox.warning(
                        self,
                        "表达式错误",
                        f"系数 {key} 的表达式 '{text}' 包含未定义的变量：{', '.join(str(s) for s in extra)}"
                    )
                    if focused != edit:
                        edit.setFocus()
                    self.focusWidget().selectAll()
                    return False
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "表达式错误",
                    f"系数 {key} 的表达式 '{text}' 无效：{str(e)}"
                )
                if focused != edit:
                    edit.setFocus()
                self.focusWidget().selectAll()
                return False
            var_names = ('x', 't') if self.has_t_check.isChecked() else ('x', 'y')
            try:
                parse_expression(text, var_names=var_names)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "表达式错误",
                    f"系数 {key} 的表达式 '{text}' 无法转换为可调用函数：{str(e)}"
                )
                if focused != edit:
                    edit.setFocus()
                self.focusWidget().selectAll()
                return False
        return True
    def _on_coeff_return_pressed(self):
        sender = self.sender()
        coeff_map = { self.A_edit: 'A', self.B_edit: 'B', self.C_edit: 'C', self.D_edit: 'D', 
                      self.E_edit: 'E', self.F_edit: 'F', self.G_edit: 'G', }
        name = coeff_map.get(sender)
        if name is not None and self.validate_coeffs(name, next=False):
            sender.clearFocus()
    def _switch_to_condition_tab(self, group: str):
        """
        根据条件分组切换到对应的标签页。
        group: 'x0', 'x1', 'y0', 'y1', 'ic'
        """
        if group in ['x0', 'x1']:
            self.tabWidget.setCurrentIndex(0)
        elif group in ['y0', 'y1']:
            self.tabWidget.setCurrentIndex(1)
        elif group == 'ic':
            self.tabWidget.setCurrentIndex(2)
    def validate_conditions(self, name: str = None, next: bool = False):
        """
        检查边界/初始条件输入框的内容。
        - 如果 name 为 None，则检查所有条件输入框。
        - 如果指定了 name（如 'x0_alpha'），则仅检查该输入框。
        - 对于非自定义预设，alpha 和 beta 只允许数值，gamma 允许表达式。
        - 对于自定义预设，所有系数均允许表达式。
        - next=True 时，空字符串报错；否则允许为空。
        """
        if name is not None:
            return self._validate_single_condition(name, next)
        for edit in self._get_condition_edits():
            if edit is None:
                continue
            if not edit.isEnabled():
                continue
            if not self._validate_single_condition(edit, next):
                return False
        return True
    def _validate_single_condition(self, edit, next: bool) -> bool:
        """
        检查单个条件输入框（内部辅助方法）。
        返回 True 表示校验通过，False 表示校验失败。
        """
        focused = self.focusWidget()
        cond_name = self._cond_edit_map.get(edit)
        if cond_name is None:
            return True
        parts = cond_name.split('_')
        if len(parts) != 2:
            return True
        group, sym = parts[0], parts[1]
        if edit is None or not edit.isEnabled():
            return True
        text = edit.text().strip()
        if text == "":
            if next:
                QMessageBox.warning(self, "输入错误", f"{group} 的 {sym} 不能为空")
                self._switch_to_condition_tab(group)
                if focused != edit:
                    edit.setFocus()
                self.focusWidget().selectAll()
                return False
            return True
        preset_name = self.preset_combo.currentText()
        is_custom = (preset_name == "自定义")
        if sym in ['alpha', 'beta'] and not is_custom:
            try:
                float(text)
                return True
            except ValueError:
                QMessageBox.warning(self, "输入错误", f"{group} 的 {sym} 必须输入数值，不允许表达式")
                self._switch_to_condition_tab(group)
                if focused != edit:
                    edit.setFocus()
                self.focusWidget().selectAll()
                return False
        try:
            if self.has_t_check.isChecked():
                x_sym, t_sym = sp.symbols('x t')
                allowed = {x_sym, t_sym}
            else:
                x_sym, y_sym = sp.symbols('x y')
                allowed = {x_sym, y_sym}
            sp_expr = sp.sympify(text)
            free_syms = sp_expr.free_symbols
            extra = free_syms - allowed
            if extra:
                QMessageBox.warning(
                    self,
                    "表达式错误",
                    f"{group} 的 {sym} 表达式 '{text}' 包含未定义的变量：{', '.join(str(s) for s in extra)}"
                )
                self._switch_to_condition_tab(group)
                if focused != edit:
                    edit.setFocus()
                self.focusWidget().selectAll()
                return False
        except Exception as e:
            QMessageBox.warning(
                self,
                "表达式错误",
                f"{group} 的 {sym} 表达式 '{text}' 无效：{str(e)}"
            )
            self._switch_to_condition_tab(group)
            if focused != edit:
                edit.setFocus()
            self.focusWidget().selectAll()
            return False
        var_names = ('x', 't') if self.has_t_check.isChecked() else ('x', 'y')
        try:
            parse_expression(text, var_names=var_names)
        except Exception as e:
            QMessageBox.warning(
                self,
                "表达式错误",
                f"{group} 的 {sym} 表达式 '{text}' 无法转换为可调用函数：{str(e)}"
            )
            self._switch_to_condition_tab(group)
            if focused != edit:
                edit.setFocus()
            self.focusWidget().selectAll()
            return False
        return True
    def _on_condition_return_pressed(self, edit, name):
        if self.validate_conditions(None, True):
            edit.deselect()
            self.setFocus()
    def mousePressEvent(self, event):
        child = self.childAt(event.pos())
        interactive_types = (
            QLineEdit, QComboBox, QPushButton, QCheckBox, QTextEdit, QTabWidget
        )
        if child is None or not isinstance(child, interactive_types):
            focused = self.focusWidget()
            if isinstance(focused, QComboBox):
                focused.clearFocus()
                focused.hidePopup()
            elif isinstance(focused, QLineEdit):
                if focused in self._get_range_edits():
                    if self.validate_range(focused) and self.validate_coeffs(None, False) and self.validate_conditions(None, False):
                        focused.deselect()
                        self.setFocus()
                elif focused in self._get_coeff_edits():
                    if self.validate_coeffs(self._get_coeff_name(focused), False) and self.validate_range() and self.validate_conditions(None, False):
                        focused.deselect()
                        self.setFocus()
                elif focused in self._get_condition_edits():
                    if self.validate_conditions(focused, False) and self.validate_range() and self.validate_coeffs(None, False):
                        focused.deselect()
                        self.setFocus()
                else:
                    self.setFocus()
        super().mousePressEvent(event)
    def validate_all(self) -> bool:
        """执行所有输入校验：范围、系数、条件。全部通过返回 True，否则返回 False（已弹窗并定位）"""
        if not self.validate_range():
            return False
        if not self.validate_coeffs(None, True):
            return False
        if not self.validate_conditions(None, True):
            return False
        return True
    def get_equation_config(self):
        """从界面读取系数，返回 (coeff_dict, has_t)"""
        coeffs = {}
        for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            edit = getattr(self, f"{key}_edit", None)
            if edit is None:
                continue
            text = edit.text().strip()
            if text == '':
                text = '0'
            try:
                val = float(text)
            except ValueError:
                val = text
            coeffs[key] = val
        has_t = self.has_t_check.isChecked()
        return coeffs, has_t
    def get_boundary_conditions(self) -> list[BoundaryCondition]:
        """从界面读取所有边界条件和初始条件，返回 BoundaryCondition 列表"""
        conditions = []
        has_t = self.has_t_check.isChecked()
        def make_condition(type_combo, alpha_edit, beta_edit, gamma_edit):
            cond_type = type_combo.currentText()
            alpha = self._parse_expr(alpha_edit.text())
            beta = self._parse_expr(beta_edit.text())
            gamma = self._parse_expr(gamma_edit.text())
            return ConditionConfig({'alpha': alpha, 'beta': beta, 'gamma': gamma})
        try:
            x_min = float(self.x_min_edit.text())
            x_max = float(self.x_max_edit.text())
        except ValueError:
            x_min, x_max = 0.0, 1.0
        try:
            y_min = float(self.y_min_edit.text()) if self.y_min_edit.isEnabled() else 0.0
            y_max = float(self.y_max_edit.text()) if self.y_max_edit.isEnabled() else 0.0
        except ValueError:
            y_min, y_max = 0.0, 1.0
        try:
            t_min = float(self.t_min_edit.text()) if has_t else 0.0
        except ValueError:
            t_min = 0.0
        cond = make_condition(self.bc_x0_type, self.bc_x0_alpha, self.bc_x0_beta, self.bc_x0_gamma)
        conditions.append(BoundaryCondition(f'x={x_min}', cond, is_initial=False))
        cond = make_condition(self.bc_x1_type, self.bc_x1_alpha, self.bc_x1_beta, self.bc_x1_gamma)
        conditions.append(BoundaryCondition(f'x={x_max}', cond, is_initial=False))
        if self.y_min_edit.isEnabled():
            cond = make_condition(self.bc_y0_type, self.bc_y0_alpha, self.bc_y0_beta, self.bc_y0_gamma)
            conditions.append(BoundaryCondition(f'y={y_min}', cond, is_initial=False))
        if self.y_max_edit.isEnabled():
            cond = make_condition(self.bc_y1_type, self.bc_y1_alpha, self.bc_y1_beta, self.bc_y1_gamma)
            conditions.append(BoundaryCondition(f'y={y_max}', cond, is_initial=False))
        if has_t:
            cond = make_condition(self.ic_displacement_type, self.ic_displacement_alpha, self.ic_displacement_beta, self.ic_displacement_gamma)
            conditions.append(BoundaryCondition(f't={t_min}', cond, is_initial=True))
            # 初始速度（目前未支持，可留作扩展）
        return conditions
    def get_domain(self):
        """获取定义域范围（仅用于采样器）"""
        x_min = float(self.x_min_edit.text())
        x_max = float(self.x_max_edit.text())
        y_min = float(self.y_min_edit.text()) if self.y_min_edit.isEnabled() else 0.0
        y_max = float(self.y_max_edit.text()) if self.y_max_edit.isEnabled() else 0.0
        t_min = float(self.t_min_edit.text()) if self.has_t_check.isChecked() else 0.0
        t_max = float(self.t_max_edit.text()) if self.has_t_check.isChecked() else 0.0
        return (x_min, x_max), (y_min, y_max), (t_min, t_max)
    def get_pde_and_conditions(self):
        """
        从界面读取方程配置和边界/初始条件，返回 (PDEConfig, list[BoundaryCondition])
        """
        coeffs, has_t = self.get_equation_config()
        conditions = self.get_boundary_conditions()
        pde = PDEConfig(coeffs, has_t=has_t)
        x_range, y_range, t_range = self.get_domain()
        return pde, conditions, (x_range, y_range, t_range)


class TrainWorker(QThread):
    progress = pyqtSignal(int, float, float, float)  # epoch, total_loss, pde_loss, bc_loss
    finished = pyqtSignal()
    error = pyqtSignal(str)
    def __init__(self, trainer, sampler, conditions, n_epochs, batch_size, n_boundary):
        super().__init__()
        self.trainer = trainer
        self.sampler = sampler
        self.conditions = conditions
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_boundary = n_boundary
        self._is_running = True
    def stop(self):
        self._is_running = False
    def run(self):
        try:
            for epoch in range(self.n_epochs):
                if not self._is_running:
                    break
                # 采样内部点
                interior_pts = self.sampler.sample_interior(self.batch_size)
                # 构建边界点列表
                boundary_pts_list = []
                for bc in self.conditions:
                    if bc.is_initial:
                        pts = self.sampler.sample_initial(self.batch_size // 2)
                    else:
                        info = bc.get_location_info()
                        if info['type'] == 'string':
                            pts = self.sampler.sample_boundary_by_axis(info['axis'], info['value'], self.n_boundary)
                        else:
                            all_pts, _ = self.sampler.sample_boundary(self.n_boundary * 4)
                            mask = info['func'](all_pts)
                            pts = all_pts[mask]
                            if len(pts) == 0:
                                pts, _ = self.sampler.sample_boundary(self.n_boundary)
                    boundary_pts_list.append(pts)
                # 训练一步
                total_loss, pde_loss, bc_loss = self.trainer.train_step(interior_pts, boundary_pts_list)
                self.progress.emit(epoch, total_loss, pde_loss, bc_loss)
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

class TrainPage(QWidget):
    training_finished = pyqtSignal()
    def __init__(self):
        super().__init__()
        ui_path = resource_path(os.path.join('ui', 'trainvisualizepage.ui'))
        uic.loadUi(ui_path, self)
        for group in [self.pde_group, self.net_group, self.train_group, self.loss_group]:
            group.setStyleSheet("QGroupBox { margin-top: 20px; }")
        # ---- 替换占位控件为 Matplotlib 画布 ----
        # 损失曲线
        self.fig_loss = Figure(figsize=(6, 3), dpi=100)
        self.canvas_loss = FigureCanvas(self.fig_loss)
        self.canvas_loss.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.loss_canvas_placeholder.parent().layout()
        if layout:
            layout.replaceWidget(self.loss_canvas_placeholder, self.canvas_loss)
        self.loss_canvas_placeholder.deleteLater()
        # 2D 等高线
        self.fig_2d = Figure(figsize=(6, 4), dpi=100)
        self.canvas_2d = FigureCanvas(self.fig_2d)
        self.canvas_2d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.canvas_2d_placeholder.parent().layout()
        if layout:
            layout.replaceWidget(self.canvas_2d_placeholder, self.canvas_2d)
        self.canvas_2d_placeholder.deleteLater()
        # 3D 曲面
        self.fig_3d = Figure(figsize=(6, 4), dpi=100)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        self.canvas_3d.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout = self.canvas_3d_placeholder.parent().layout()
        if layout:
            layout.replaceWidget(self.canvas_3d_placeholder, self.canvas_3d)
        self.canvas_3d_placeholder.deleteLater()
        # ---- 初始化绘图 ----
        self.init_plots()
        # ---- 按钮信号 ----
        self.start_btn.clicked.connect(self.start_training)
        self.stop_btn.clicked.connect(self.stop_training)
        self.stop_btn.setEnabled(False)
        self.pushButton.clicked.connect(self.auto_build)
        self.save_btn.clicked.connect(self.save_results)
        self.is_restarted = True
        self.is_training = False
        self.restart_btn.clicked.connect(self.restart_training)
        self.restart_btn.setEnabled(True)
        self.start_btn.setText("开始训练")
        # ---- 状态变量 ----
        self.pde = None
        self.conditions = None
        self.domain = None
        self.trainer = None
        self.worker = None
        self.cbar_2d = None
        self.cbar_3d = None
        self.analytical_func = None
        self.loss_history = {'total': [], 'pde': [], 'bc': []}
        self.editable_widgets = [
            self.hidden_dims_edit,
            self.activation_combo,
            self.epoch_spin,
            self.batch_spin,
            self.boundary_spin,
            self.pushButton,
        ]
    def init_plots(self):
        # 损失曲线
        if len(self.fig_loss.axes) == 0:
            self.fig_loss.add_subplot(111)
        ax = self.fig_loss.axes[0]
        ax.clear()
        ax.set_title("训练损失")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_yscale('log')
        ax.grid(True)
        self.fig_loss.tight_layout()
        self.canvas_loss.draw()
        # 2D 等高线
        self.fig_2d.clear()
        ax2 = self.fig_2d.add_subplot(111)
        ax2.set_title("解等高线")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()
        # 3D 曲面
        self.fig_3d.clear()
        ax3 = self.fig_3d.add_subplot(111, projection='3d')
        ax3.set_title("解曲面")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("u")
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
    def set_analytical_func(self, func):
        self.analytical_func = func
    def setup_training(self, pde: PDEConfig, conditions: list[BoundaryCondition], domain: tuple):
        self.pde = pde
        self.conditions = conditions
        self.domain = domain
        # 更新 pde_group 中的信息标签
        # 方程表达式
        eq_str = " + ".join([f"{c}*{term}" for term, c in pde.coeffs.items() if c != 0])
        self.label.setText(eq_str if eq_str else "0 = 0")
        self.label_2.setText(pde.equation_type)
        x_range, y_range, t_range = domain
        domain_str = f"x∈[{x_range[0]},{x_range[1]}]"
        if y_range[0] != 0 or y_range[1] != 0:
            domain_str += f", y∈[{y_range[0]},{y_range[1]}]"
        if pde.has_t:
            domain_str += f", t∈[{t_range[0]},{t_range[1]}]"
        self.label_6.setText(domain_str)
    def auto_build(self):
        """
        自动构建网络配置：根据当前方程和边界条件，使用 ComplexityAnalyzer
        生成推荐网络结构，并填入界面控件。
        """
        # 检查是否已从 EquationPage 获取数据
        if self.pde is None or self.conditions is None:
            QMessageBox.warning(self, "警告", "请先从方程页面配置并进入训练页")
            return
        # 确定输入维度
        has_t = self.pde.has_t
        input_dim = 2 #if has_t else 2  # 若含时间，输入为 (x, y, t)，否则 (x, y)
        output_dim = 1
        # 使用 ComplexityAnalyzer 计算复杂度
        analyzer = ComplexityAnalyzer()
        score = analyzer.compute_complexity(self.pde, self.conditions)
        # 生成配置
        generator = NetworkConfigGenerator(base_config={
            'input_dim': input_dim,
            'output_dim': output_dim
        })
        config = generator.generate_config(score)
        # 将推荐参数填入界面控件
        hidden_dims_str = ",".join(str(d) for d in config['hidden_dims'])
        self.hidden_dims_edit.setText(hidden_dims_str)
        # 设置激活函数（如果 combo 中有该选项）
        index = self.activation_combo.findText(config['activation'])
        if index >= 0:
            self.activation_combo.setCurrentIndex(index)
        else:
            # 若未找到，默认选 tanh
            self.activation_combo.setCurrentText('tanh')
        # 可选：更新状态标签
        self.status_label.setText(f"自动构建完成")
    def start_training(self):
        if self.pde is None:
            QMessageBox.warning(self, "错误", "请先从方程页面配置")
            return
        # 读取训练参数
        try:
            hidden_dims = [int(x.strip()) for x in self.hidden_dims_edit.text().split(',') if x.strip()]
            activation = self.activation_combo.currentText()
            epochs = self.epoch_spin.value()
            batch_size = self.batch_spin.value()
            n_boundary = self.boundary_spin.value()
            lr = self.lr_spin.value()
        except Exception as e:
            QMessageBox.critical(self, "参数错误", f"读取参数失败：{e}")
            return
        # ----- 判断是否从头训练 -----
        if self.trainer is None or self.is_restarted:
            # 从头训练：构建网络、采样器、训练器
            has_t = self.pde.has_t
            input_dim = 2 # if not has_t else 3
            output_dim = 1
            model = build_network(input_dim, output_dim, hidden_dims, activation=activation)
            model.train()
            x_range, y_range, t_range = self.domain
            if not has_t:
                sampler = DomainSampler(x_range=x_range, y_range=y_range)
            else:
                sampler = DomainSampler(x_range=x_range, y_range=t_range)
            self.trainer = PINNTrainer(model, self.pde, self.conditions, lr=lr)
            self.sampler = sampler
            # 清空损失历史（从头训练必须清空）
            self.loss_history = {'total': [], 'pde': [], 'bc': []}
            self.init_plots()    # 重置图表
            self.is_restarted = False
            self.start_btn.setText("继续训练")
        else:
            for param_group in self.trainer.optimizer.param_groups:
                param_group['lr'] = lr
        self.is_training = True
        self.update_controls_enabled()
        # ----- 启动训练线程 -----
        self.worker = TrainWorker(
            trainer=self.trainer,
            sampler=self.sampler,
            conditions=self.conditions,
            n_epochs=epochs,
            batch_size=batch_size,
            n_boundary=n_boundary
        )
        self.worker.progress.connect(self.update_loss)
        self.worker.finished.connect(self.on_training_finished)
        self.worker.error.connect(self.on_training_error)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.restart_btn.setEnabled(False)
        self.is_training = True
        self.status_label.setText("训练中...")
        self.worker.start()
    def stop_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.restart_btn.setEnabled(True)
            self.is_training = False
            self.update_controls_enabled()
            self.status_label.setText("训练已停止")
    def restart_training(self):
        """重新开始：清空训练器、历史，重置按钮状态"""
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "警告", "请先停止当前训练")
            return
        # 清空训练器和采样器
        self.trainer = None
        self.sampler = None
        # 清空损失历史
        self.loss_history = {'total': [], 'pde': [], 'bc': []}
        # 重置图表（清空数据，保留轴）
        self.init_plots()   # 确保 init_plots 只清空不新建轴
        # 重置状态
        self.cbar_2d = None
        self.cbar_3d = None
        self.is_restarted = True
        self.start_btn.setText("开始训练")
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.restart_btn.setEnabled(True)
        self.update_controls_enabled()
        self.status_label.setText("已重置，准备开始")
    def update_controls_enabled(self):
        """
        根据 self.is_restarted 和 self.is_training 控制各控件的可编辑状态。
        - 若 is_restarted=True（未开始或重置后）：所有控件可编辑。
        - 若 is_restarted=False 且 is_training=False（训练已停止）：仅学习率可编辑，其他控件禁用。
        - 若 is_training=True（训练中）：所有控件禁用（包括学习率）。
        """
        if self.is_training:
            # 训练中：禁用所有控件（包括学习率）
            for w in self.editable_widgets:
                w.setEnabled(False)
            self.lr_spin.setEnabled(False)
        elif self.is_restarted:
            # 重置状态：所有控件可编辑
            for w in self.editable_widgets:
                w.setEnabled(True)
            self.lr_spin.setEnabled(True)
        else:
            # 训练已停止（未重置）：仅学习率可编辑
            for w in self.editable_widgets:
                w.setEnabled(False)
            self.lr_spin.setEnabled(True)
    def update_loss(self, epoch, total, pde, bc):
        self.loss_history['total'].append(total)
        self.loss_history['pde'].append(pde)
        self.loss_history['bc'].append(bc)
        ax = self.fig_loss.axes[0]  # 直接使用第一个轴
        ax.clear()
        epochs = range(1, len(self.loss_history['total']) + 1)
        ax.plot(epochs, self.loss_history['total'], label='Total')
        ax.plot(epochs, self.loss_history['pde'], label='PDE')
        ax.plot(epochs, self.loss_history['bc'], label='BC')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True)
        ax.set_title("训练损失")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        self.fig_loss.tight_layout()
        self.canvas_loss.draw()
        self.status_label.setText(f"Epoch {epoch}: Total {total:.3e}")
    def on_training_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.restart_btn.setEnabled(True)    # 允许重新开始
        self.is_training = False
        self.update_controls_enabled()
        self.status_label.setText("训练完成")
        self.plot_solution()
        self.training_finished.emit()
    def on_training_error(self, msg):
        QMessageBox.critical(self, "训练错误", msg)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText("训练出错")
    def plot_solution(self, n_points=150):
        if self.trainer is None:
            return
        model = self.trainer.model
        has_t = self.pde.has_t
        x_range, y_range, t_range = self.domain
        # 生成网格和预测数据
        if has_t:
            x = torch.linspace(x_range[0], x_range[1], n_points)
            t = torch.linspace(t_range[0], t_range[1], n_points)
            X, T = torch.meshgrid(x, t, indexing='ij')
            pts = torch.stack([X.flatten(), T.flatten()], dim=1)
            x_label, y_label = 'x', 't'
            Y = T
        else:
            x = torch.linspace(x_range[0], x_range[1], n_points)
            y = torch.linspace(y_range[0], y_range[1], n_points)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
            x_label, y_label = 'x', 'y'
        model.eval()
        with torch.no_grad():
            Z_pred = model(pts).numpy().reshape(n_points, n_points)
        # ---- 2D 绘图 ----
        self.fig_2d.clear()
        if self.analytical_func is not None:
            # 解析解
            Z_true = self.analytical_func(pts).numpy().reshape(n_points, n_points)
            error = np.abs(Z_pred - Z_true)
            # 左：预测解
            ax_left = self.fig_2d.add_subplot(1, 2, 1)
            cf1 = ax_left.contourf(X.numpy(), Y.numpy(), Z_pred, levels=50, cmap='viridis')
            ax_left.set_title("预测解")
            ax_left.set_xlabel(x_label)
            ax_left.set_ylabel(y_label)
            self.fig_2d.colorbar(cf1, ax=ax_left)
            # 右：误差图
            ax_right = self.fig_2d.add_subplot(1, 2, 2)
            cf2 = ax_right.contourf(X.numpy(), Y.numpy(), error, levels=50, cmap='hot')
            ax_right.set_title("绝对误差")
            ax_right.set_xlabel(x_label)
            ax_right.set_ylabel(y_label)
            self.fig_2d.colorbar(cf2, ax=ax_right)
        else:
            # 无解析解，只画预测解
            ax = self.fig_2d.add_subplot(111)
            cf = ax.contourf(X.numpy(), Y.numpy(), Z_pred, levels=50, cmap='viridis')
            ax.set_title("预测解")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            self.fig_2d.colorbar(cf, ax=ax)
        self.fig_2d.tight_layout()
        self.canvas_2d.draw()
        # ---- 3D 绘图 ----
        self.fig_3d.clear()
        if self.analytical_func is not None:
            # 左：预测解
            ax_left = self.fig_3d.add_subplot(1, 2, 1, projection='3d')
            surf1 = ax_left.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', edgecolor='none')
            ax_left.set_title("预测解曲面")
            ax_left.set_xlabel(x_label)
            ax_left.set_ylabel(y_label)
            ax_left.set_zlabel('u')
            # 右：解析解
            ax_right = self.fig_3d.add_subplot(1, 2, 2, projection='3d')
            surf2 = ax_right.plot_surface(X.numpy(), Y.numpy(), Z_true, cmap='plasma', edgecolor='none')
            ax_right.set_title("解析解曲面")
            ax_right.set_xlabel(x_label)
            ax_right.set_ylabel(y_label)
            ax_right.set_zlabel('u')
            # 添加colorbar（可选，但需分别添加）
            self.fig_3d.colorbar(surf1, ax=ax_left)
            self.fig_3d.colorbar(surf2, ax=ax_right)
        else:
            ax = self.fig_3d.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', edgecolor='none')
            ax.set_title("预测解曲面")
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel('u')
            self.fig_3d.colorbar(surf, ax=ax)
        self.fig_3d.tight_layout()
        self.canvas_3d.draw()
    # def plot_solution(self, n_points=150):
    #     if self.trainer is None:
    #         return
    #     model = self.trainer.model
    #     has_t = self.pde.has_t
    #     x_range, y_range, t_range = self.domain
    #     # 生成网格和预测数据
    #     if has_t:
    #         # x = torch.linspace(x_range[0], x_range[1], n_points)
    #         # y = torch.linspace(y_range[0], y_range[1], n_points)
    #         # X, Y = torch.meshgrid(x, y, indexing='ij')
    #         # pts = torch.stack([X.flatten(), Y.flatten(), torch.full((n_points*n_points,), t_range[0])], dim=1)
    #         x = torch.linspace(x_range[0], x_range[1], n_points)
    #         t = torch.linspace(t_range[0], t_range[1], n_points)
    #         X, Y = torch.meshgrid(x, t, indexing='ij')
    #         pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    #         x_label = 'x'
    #         y_label = 't'
    #     else:
    #         x = torch.linspace(x_range[0], x_range[1], n_points)
    #         y = torch.linspace(y_range[0], y_range[1], n_points)
    #         X, Y = torch.meshgrid(x, y, indexing='ij')
    #         pts = torch.stack([X.flatten(), Y.flatten()], dim=1)
    #         x_label = 'x'
    #         y_label = 'y'
    #     model.eval()
    #     with torch.no_grad():
    #         Z = model(pts).numpy().reshape(n_points, n_points)
    #     # ---- 重建 2D 图 ----
    #     self.fig_2d.clear()
    #     ax2 = self.fig_2d.add_subplot(111)
    #     cf = ax2.contourf(X.numpy(), Y.numpy(), Z, levels=50, cmap='viridis')
    #     ax2.set_title("解等高线")
    #     ax2.set_xlabel(x_label)
    #     ax2.set_ylabel(y_label)
    #     self.fig_2d.colorbar(cf, ax=ax2)
    #     self.fig_2d.tight_layout()
    #     self.canvas_2d.draw()
    #     # ---- 重建 3D 图 ----
    #     self.fig_3d.clear()
    #     ax3 = self.fig_3d.add_subplot(111, projection='3d')
    #     surf = ax3.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', edgecolor='none')
    #     ax3.set_title("解曲面")
    #     ax3.set_xlabel(x_label)
    #     ax3.set_ylabel(y_label)
    #     # ax3.set_xlabel("x")
    #     # ax3.set_ylabel("y")
    #     # ax3.set_zlabel("u")
    #     self.fig_3d.colorbar(surf, ax=ax3)
    #     self.fig_3d.tight_layout()
    #     self.canvas_3d.draw()
    def mousePressEvent(self, event):
        """点击不可交互控件时清除焦点"""
        child = self.childAt(event.pos())
        interactive_types = (QLineEdit, QComboBox, QPushButton, QCheckBox, QTextEdit, QTabWidget)
        if child is None or not isinstance(child, interactive_types):
            focused = self.focusWidget()
            if isinstance(focused, QComboBox):
                focused.clearFocus()
                focused.hidePopup()
            else:
                self.setFocus()
        super().mousePressEvent(event)
    def save_results(self):
        """保存当前训练结果到 results/ 目录，按计数编号"""
        import os
        import json
        import shutil
        from datetime import datetime
        # 获取项目根目录（src 的父目录）
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(project_root, 'results')
        config_dir = os.path.join(project_root, 'config')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        # 计数文件
        counter_file = os.path.join(config_dir, 'save_counter.json')
        if os.path.exists(counter_file):
            try:
                with open(counter_file, 'r') as f:
                    data = json.load(f)
                    count = data.get('count', 0) + 1
            except (json.JSONDecodeError, ValueError):
                count = 1
        else:
            count = 1
        # 格式化编号，如 001
        idx_str = f"{count:03d}"
        # 更新计数
        with open(counter_file, 'w') as f:
            json.dump({'count': count}, f, indent=4)
        # 创建子文件夹
        models_dir = os.path.join(results_dir, 'models')
        logs_dir = os.path.join(results_dir, 'logs')
        figures_dir = os.path.join(results_dir, 'figures')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        # 构建基础文件名
        base_name = f"result_{idx_str}"
        # ---- 1. 保存模型参数 ----
        if self.trainer is not None:
            model_path = os.path.join(models_dir, f"{base_name}_model.pth")
            torch.save(self.trainer.model.state_dict(), model_path)
            print(f"模型已保存: {model_path}")
        else:
            QMessageBox.warning(self, "警告", "没有训练好的模型可保存")
        # ---- 2. 保存方程和条件信息 ----
        log_path = os.path.join(logs_dir, f"{base_name}_info.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结果编号: {idx_str}\n")
            f.write("\n=== 方程配置 ===\n")
            if self.pde is not None:
                f.write(f"方程类型: {self.pde.equation_type}\n")
                f.write(f"系数: {self.pde.coeffs}\n")
                f.write(f"包含时间: {self.pde.has_t}\n")
            else:
                f.write("无方程配置\n")
            f.write("\n=== 边界/初始条件 ===\n")
            if self.conditions is not None:
                for i, bc in enumerate(self.conditions):
                    f.write(f"条件 {i+1}: {bc.location}, is_initial={bc.is_initial}\n")
                    cond = bc.get_condition()
                    f.write(f"  类型: {cond.condition_type}, α={cond.get_alpha()}, β={cond.get_beta()}, γ={cond.get_gamma()}\n")
            else:
                f.write("无条件配置\n")
            f.write("\n=== 训练参数 ===\n")
            f.write(f"隐藏层: {self.hidden_dims_edit.text()}\n")
            f.write(f"激活函数: {self.activation_combo.currentText()}\n")
            f.write(f"迭代轮数: {self.epoch_spin.value()}\n")
            f.write(f"内部点批量: {self.batch_spin.value()}\n")
            f.write(f"边界点数: {self.boundary_spin.value()}\n")
            f.write(f"学习率: {self.lr_spin.value()}\n")
            f.write("\n=== 损失信息 ===\n")
            if self.loss_history and len(self.loss_history['total']) > 0:
                f.write(f"最终总损失: {self.loss_history['total'][-1]:.6e}\n")
                f.write(f"最终PDE损失: {self.loss_history['pde'][-1]:.6e}\n")
                f.write(f"最终边界损失: {self.loss_history['bc'][-1]:.6e}\n")
                f.write(f"总训练轮数: {len(self.loss_history['total'])}\n")
            else:
                f.write("无损失记录\n")
        print(f"日志已保存: {log_path}")
        # ---- 3. 保存图表 ----
        try:
            fig_loss_path = os.path.join(figures_dir, f"{base_name}_loss.png")
            self.fig_loss.savefig(fig_loss_path, dpi=150, bbox_inches='tight')
            print(f"损失图已保存: {fig_loss_path}")
            fig_2d_path = os.path.join(figures_dir, f"{base_name}_solution_2d.png")
            self.fig_2d.savefig(fig_2d_path, dpi=150, bbox_inches='tight')
            print(f"2D解图已保存: {fig_2d_path}")
            fig_3d_path = os.path.join(figures_dir, f"{base_name}_solution_3d.png")
            self.fig_3d.savefig(fig_3d_path, dpi=150, bbox_inches='tight')
            print(f"3D解图已保存: {fig_3d_path}")
        except Exception as e:
            QMessageBox.warning(self, "保存图表失败", str(e))
            return
        QMessageBox.information(self, "保存成功", f"结果已保存为编号 {idx_str}\n路径: {results_dir}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN 求解器")
        self.setGeometry(100, 100, 1680, 1120)
        self.center()
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.welcome_page = WelcomePage()
        self.equation_page = EquationPage()
        self.train_page = TrainPage()
        self.stack.addWidget(self.welcome_page)     # index 0
        self.stack.addWidget(self.equation_page)    # index 1
        self.stack.addWidget(self.train_page)       # index 2
        self.welcome_page.start_btn.clicked.connect(lambda: self.go_to_content(1))
        self.equation_page.back_button.clicked.connect(lambda: self.go_to_content(0))
        self.equation_page.next_page_button.clicked.connect(lambda: self.go_to_content(2))
        self.train_page.back_button.clicked.connect(lambda: self.go_to_content(1))
        self.train_page.training_finished.connect(self.on_training_finished)
        # self.train_page.training_error.connect(self.on_training_error)
    def center(self):
        """将窗口居中显示"""
        screen = QApplication.primaryScreen().geometry()
        size = self.geometry()
        self.move(int((screen.width() - size.width()) / 2), int((screen.height() - size.height()) / 2))
    def go_to_content(self, index: int):
        if index == 1 and self.stack.currentWidget() == self.train_page:
            if self.train_page.is_training:
                QMessageBox.warning(self, "提示", "训练正在进行中，请先停止训练再返回")
                return
            else:
                self.train_page.restart_training()
        elif index == 2:
            if not self.equation_page.validate_all():
                return
            pde, conditions, domain = self.equation_page.get_pde_and_conditions()
            self.train_page.setup_training(pde, conditions, domain)
            self.train_page.auto_build()
            analytical_func = generate_analytical_solution(pde, conditions)
            self.train_page.set_analytical_func(analytical_func)
        self.stack.setCurrentIndex(index)
    def on_training_finished(self):
        QMessageBox.information(self, "训练完成", "训练已成功结束！")
    def on_training_error(self, msg):
        QMessageBox.critical(self, "训练错误", msg)

def run_gui():
    # app = QApplication(sys.argv)
    # window = PINNGUI()
    # window.show()
    # sys.exit(app.exec_())
    app = QApplication(sys.argv)
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', light=True))
    icon_path = resource_path(os.path.join('ui', 'window_icon.png'))
    app.setWindowIcon(QIcon(icon_path))
    # app.setStyleSheet("""
    #     /* 顶层窗口（QMainWindow、QWidget 等）背景渐变 */
    #     QMainWindow {
    #         background: qlineargradient(
    #             x1:0, y1:0, x2:1, y2:1,
    #             stop:0 #ff9a9e,   /* 粉红 */
    #             stop:0.25 #fecfef, /* 浅粉紫 */
    #             stop:0.5 #a1c4fd, /* 淡蓝 */
    #             stop:0.75 #c2e9fb, /* 天蓝 */
    #             stop:1 #fbc2eb    /* 粉紫 */
    #         );
    #     }
    #     EquationPage, TrainPage, WelcomePage {
    #         background: transparent;
    #     }
    #     QGroupBox, QFrame, QLabel {
    #         background: transparent;
    #         border: none;
    #     }
    #     /* 分组框标题样式 */
    #     QGroupBox::title {
    #         background: transparent;
    #     }

    #     /* 按钮悬停效果 */
    #     QPushButton:hover {
    #         background: rgba(230, 240, 255, 220);
    #     }
    # """)
    window = MainWindow()
    # window.welcome_page.setStyleSheet("""
    #     background-image: url("D:/vscode/pde_pinn_project/ui/welcome_page.png");
    #     background-repeat: no-repeat;
    #     background-position: center;
    #     background-color: transparent;
    #     background-size: cover;
        
    # """)
    # window.welcome_page.setContentsMargins(0, 0, 0, 0)
    # window.welcome_page.setAttribute(Qt.WA_StyledBackground, True)
    # bg_path = resource_path(os.path.join('ui', 'welcome_page.png'))
    # bg_url = "file:///" + bg_path.replace("\\", "/")
    # window.welcome_page.setStyleSheet(f"""
    #     WelcomePage {{
    #         border-image: url("{bg_url}") 0 0 0 0 stretch stretch;
    #         border: 0px;
    #         margin: 0px;
    #         padding: 0px;
    #     }}
    #     WelcomePage QLabel, WelcomePage QFrame, WelcomePage QGroupBox {{
    #         background: transparent;
    #         border: none;
    #     }}
    # """)
    # window.welcome_page.setStyleSheet("""
    #     WelcomePage QLabel, WelcomePage QFrame, WelcomePage QGroupBox {
    #         background: transparent;
    #         border: none;
    #     }
    # """)
    # palette = window.welcome_page.palette()
    # pixmap = QPixmap(bg_path)
    # palette.setBrush(QPalette.Window, QBrush(pixmap))
    # window.welcome_page.setPalette(palette)
    # window.welcome_page.setAutoFillBackground(True)
    window.show()
    sys.exit(app.exec_())


































# class TrainWorker(QThread):
#     progress = pyqtSignal(int, float, float, float)  # epoch, total_loss, pde_loss, bc_loss
#     finished = pyqtSignal()
#     error = pyqtSignal(str)
#     def __init__(self, trainer, sampler, conditions, n_epochs, batch_size, n_boundary):
#         super().__init__()
#         self.trainer = trainer
#         self.sampler = sampler
#         self.conditions = conditions
#         self.n_epochs = n_epochs
#         self.batch_size = batch_size
#         self.n_boundary = n_boundary
#         self._is_running = True
#     def stop(self):
#         self._is_running = False
#     def run(self):
#         try:
#             for epoch in range(self.n_epochs):
#                 if not self._is_running:
#                     break
#                 interior_pts = self.sampler.sample_interior(self.batch_size)
#                 boundary_pts_list = []
#                 for bc in self.conditions:
#                     if bc.is_initial:
#                         pts = self.sampler.sample_initial(self.batch_size // 2)
#                     else:
#                         info = bc.get_location_info()
#                         if info['type'] == 'string':
#                             pts = self.sampler.sample_boundary_by_axis(info['axis'], info['value'], self.n_boundary)
#                         else:
#                             all_pts, _ = self.sampler.sample_boundary(self.n_boundary * 4)
#                             mask = info['func'](all_pts)
#                             pts = all_pts[mask]
#                             if len(pts) == 0:
#                                 pts, _ = self.sampler.sample_boundary(self.n_boundary)
#                     boundary_pts_list.append(pts)
#                 total_loss, pde_loss, bc_loss = self.trainer.train_step(interior_pts, boundary_pts_list)
#                 self.progress.emit(epoch, total_loss, pde_loss, bc_loss)
#             self.finished.emit()
#         except Exception as e:
#             self.error.emit(str(e))




# # ============ 页面2：网络与训练控制 ============
# class TrainPage(QWidget):
#     """网络结构、训练参数与启动控制页面"""
#     # 信号：训练请求
#     trainRequested = pyqtSignal(dict)  # 传递训练参数字典

#     def __init__(self):
#         super().__init__()
#         self.init_ui()

#     def init_ui(self):
#         layout = QVBoxLayout(self)

#         # 网络结构
#         net_group = QGroupBox("网络结构")
#         net_layout = QGridLayout()
#         net_layout.addWidget(QLabel("隐藏层 (逗号分隔):"), 0, 0)
#         self.hidden_dims_edit = QLineEdit("64,64,32")
#         net_layout.addWidget(self.hidden_dims_edit, 0, 1)
#         net_layout.addWidget(QLabel("激活函数:"), 1, 0)
#         self.activation_combo = QComboBox()
#         self.activation_combo.addItems(['tanh', 'relu', 'sigmoid', 'sin'])
#         net_layout.addWidget(self.activation_combo, 1, 1)
#         net_group.setLayout(net_layout)
#         layout.addWidget(net_group)

#         # 训练参数
#         train_group = QGroupBox("训练参数")
#         train_layout = QGridLayout()
#         train_layout.addWidget(QLabel("迭代轮数:"), 0, 0)
#         self.epoch_spin = QSpinBox()
#         self.epoch_spin.setRange(100, 100000)
#         self.epoch_spin.setValue(5000)
#         train_layout.addWidget(self.epoch_spin, 0, 1)
#         train_layout.addWidget(QLabel("内部点批量:"), 1, 0)
#         self.batch_spin = QSpinBox()
#         self.batch_spin.setRange(100, 10000)
#         self.batch_spin.setValue(2000)
#         train_layout.addWidget(self.batch_spin, 1, 1)
#         train_layout.addWidget(QLabel("每条边界点数:"), 2, 0)
#         self.boundary_spin = QSpinBox()
#         self.boundary_spin.setRange(20, 500)
#         self.boundary_spin.setValue(100)
#         train_layout.addWidget(self.boundary_spin, 2, 1)
#         train_layout.addWidget(QLabel("学习率:"), 3, 0)
#         self.lr_spin = QDoubleSpinBox()
#         self.lr_spin.setRange(1e-6, 1e-1)
#         self.lr_spin.setValue(1e-3)
#         self.lr_spin.setDecimals(6)
#         train_layout.addWidget(self.lr_spin, 3, 1)
#         train_group.setLayout(train_layout)
#         layout.addWidget(train_group)

#         # 按钮区
#         btn_layout = QHBoxLayout()
#         self.start_btn = QPushButton("开始训练")
#         self.start_btn.clicked.connect(self.emit_train_request)
#         btn_layout.addWidget(self.start_btn)
#         self.stop_btn = QPushButton("停止训练")
#         self.stop_btn.setEnabled(False)
#         btn_layout.addWidget(self.stop_btn)
#         layout.addLayout(btn_layout)

#         # 状态
#         self.status_label = QLabel("就绪")
#         layout.addWidget(self.status_label)

#         layout.addStretch()

#     def emit_train_request(self):
#         """收集训练参数并发出信号"""
#         params = {
#             'hidden_dims': [int(x.strip()) for x in self.hidden_dims_edit.text().split(',') if x.strip()],
#             'activation': self.activation_combo.currentText(),
#             'epochs': self.epoch_spin.value(),
#             'batch_size': self.batch_spin.value(),
#             'n_boundary': self.boundary_spin.value(),
#             'lr': self.lr_spin.value()
#         }
#         self.trainRequested.emit(params)

#     def set_running(self, running):
#         """更新按钮状态"""
#         self.start_btn.setEnabled(not running)
#         self.stop_btn.setEnabled(running)
#         if running:
#             self.status_label.setText("训练中...")
#         else:
#             self.status_label.setText("训练结束")


# # ============ 页面3：可视化 ============
# class VisualizePage(QWidget):
#     """损失曲线和解的可视化页面"""
#     def __init__(self):
#         super().__init__()
#         self.loss_history = {'total': [], 'pde': [], 'bc': []}
#         self.model = None
#         self.has_t = False
#         self.init_ui()

#     def init_ui(self):
#         layout = QVBoxLayout(self)

#         # 损失曲线
#         self.fig_loss = Figure(figsize=(6, 2.5), dpi=100)
#         self.canvas_loss = FigureCanvas(self.fig_loss)
#         layout.addWidget(self.canvas_loss)

#         # 解可视化（标签页）
#         self.sol_tab = QTabWidget()
#         self.fig_2d = Figure(figsize=(6, 4), dpi=100)
#         self.canvas_2d = FigureCanvas(self.fig_2d)
#         self.sol_tab.addTab(self.canvas_2d, "2D 等高线")
#         self.fig_3d = Figure(figsize=(6, 4), dpi=100)
#         self.canvas_3d = FigureCanvas(self.fig_3d)
#         self.sol_tab.addTab(self.canvas_3d, "3D 曲面")
#         layout.addWidget(self.sol_tab)

#         self.init_plots()

#     def init_plots(self):
#         ax = self.fig_loss.add_subplot(111)
#         ax.set_title("训练损失")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss")
#         ax.set_yscale('log')
#         ax.grid(True)
#         self.fig_loss.tight_layout()
#         self.canvas_loss.draw()

#         ax2 = self.fig_2d.add_subplot(111)
#         ax2.set_title("解等高线")
#         ax2.set_xlabel("x")
#         ax2.set_ylabel("y")
#         self.fig_2d.tight_layout()
#         self.canvas_2d.draw()

#         ax3 = self.fig_3d.add_subplot(111, projection='3d')
#         ax3.set_title("解曲面")
#         ax3.set_xlabel("x")
#         ax3.set_ylabel("y")
#         ax3.set_zlabel("u")
#         self.fig_3d.tight_layout()
#         self.canvas_3d.draw()

#     def update_loss(self, epoch, total, pde, bc):
#         self.loss_history['total'].append(total)
#         self.loss_history['pde'].append(pde)
#         self.loss_history['bc'].append(bc)

#         ax = self.fig_loss.axes[0]
#         ax.clear()
#         epochs = range(1, len(self.loss_history['total']) + 1)
#         ax.plot(epochs, self.loss_history['total'], label='Total')
#         ax.plot(epochs, self.loss_history['pde'], label='PDE')
#         ax.plot(epochs, self.loss_history['bc'], label='BC')
#         ax.set_yscale('log')
#         ax.legend()
#         ax.grid(True)
#         ax.set_title("训练损失")
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel("Loss")
#         self.fig_loss.tight_layout()
#         self.canvas_loss.draw()

#     def set_model(self, model, has_t):
#         self.model = model
#         self.has_t = has_t

#     def plot_solution(self, n_points=150):
#         if self.model is None:
#             return
#         if self.has_t:
#             # t=0 截面
#             x = torch.linspace(0,1,n_points)
#             y = torch.linspace(0,1,n_points)
#             X, Y = torch.meshgrid(x, y, indexing='ij')
#             pts = torch.stack([X.flatten(), Y.flatten(), torch.zeros_like(X.flatten())], dim=1)
#         else:
#             x = torch.linspace(0,1,n_points)
#             y = torch.linspace(0,1,n_points)
#             X, Y = torch.meshgrid(x, y, indexing='ij')
#             pts = torch.stack([X.flatten(), Y.flatten()], dim=1)

#         self.model.eval()
#         with torch.no_grad():
#             Z = self.model(pts).numpy().reshape(n_points, n_points)

#         # 2D
#         ax2 = self.fig_2d.axes[0]
#         ax2.clear()
#         cf = ax2.contourf(X.numpy(), Y.numpy(), Z, levels=50, cmap='viridis')
#         ax2.set_title("解等高线")
#         ax2.set_xlabel("x")
#         ax2.set_ylabel("y")
#         self.fig_2d.colorbar(cf, ax=ax2)
#         self.fig_2d.tight_layout()
#         self.canvas_2d.draw()

#         # 3D
#         ax3 = self.fig_3d.axes[0]
#         ax3.clear()
#         surf = ax3.plot_surface(X.numpy(), Y.numpy(), Z, cmap='viridis', edgecolor='none')
#         ax3.set_title("解曲面")
#         ax3.set_xlabel("x")
#         ax3.set_ylabel("y")
#         ax3.set_zlabel("u")
#         self.fig_3d.colorbar(surf, ax=ax3)
#         self.fig_3d.tight_layout()
#         self.canvas_3d.draw()

#     def clear_history(self):
#         self.loss_history = {'total': [], 'pde': [], 'bc': []}
#         self.init_plots()

#     def save_figures(self, dir_path):
#         self.fig_loss.savefig(f"{dir_path}/loss.png", dpi=150)
#         self.fig_2d.savefig(f"{dir_path}/solution_2d.png", dpi=150)
#         self.fig_3d.savefig(f"{dir_path}/solution_3d.png", dpi=150)


# # ============ 主窗口（堆叠布局） ============
# class PINNGUI(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("PINN 偏微分方程求解器")
#         self.setGeometry(50, 50, 1300, 850)

#         central = QWidget()
#         self.setCentralWidget(central)
#         main_layout = QVBoxLayout(central)

#         # ---- 导航栏 ----
#         nav_layout = QHBoxLayout()
#         self.btn_eq = QPushButton("1. 方程与边界")
#         self.btn_eq.clicked.connect(lambda: self.stack.setCurrentIndex(0))
#         self.btn_train = QPushButton("2. 训练控制")
#         self.btn_train.clicked.connect(lambda: self.stack.setCurrentIndex(1))
#         self.btn_vis = QPushButton("3. 结果可视化")
#         self.btn_vis.clicked.connect(lambda: self.stack.setCurrentIndex(2))
#         nav_layout.addWidget(self.btn_eq)
#         nav_layout.addWidget(self.btn_train)
#         nav_layout.addWidget(self.btn_vis)
#         nav_layout.addStretch()
#         main_layout.addLayout(nav_layout)

#         # ---- 堆叠区域 ----
#         self.stack = QStackedWidget()
#         self.eq_page = EquationPage()
#         self.train_page = TrainPage()
#         self.vis_page = VisualizePage()
#         self.stack.addWidget(self.eq_page)
#         self.stack.addWidget(self.train_page)
#         self.stack.addWidget(self.vis_page)
#         main_layout.addWidget(self.stack)

#         # ---- 连接信号 ----
#         self.train_page.trainRequested.connect(self.start_training)

#         # 保存按钮（在底部）
#         btn_save = QPushButton("保存结果图片")
#         btn_save.clicked.connect(self.save_figures)
#         main_layout.addWidget(btn_save)

#         # 初始状态
#         self.trainer = None
#         self.worker = None

#     def start_training(self, params):
#         """由 TrainPage 触发的训练流程"""
#         try:
#             # 1. 从 EquationPage 读取配置
#             coeffs, has_t = self.eq_page.get_equation_config()
#             pde_config = PDEConfig(coeffs, has_t=has_t)
#             conditions = self.eq_page.get_boundary_conditions()

#             # 2. 构建网络
#             input_dim = 2 if not has_t else 3
#             output_dim = 1
#             model = build_model(
#                 input_dim, output_dim,
#                 params['hidden_dims'],
#                 activation=params['activation']
#             )
#             model.train()

#             # 3. 采样器
#             if not has_t:
#                 sampler = DomainSampler(x_range=(0,1), y_range=(0,1))
#             else:
#                 sampler = DomainSampler(x_range=(0,1), y_range=(0,1), t_range=(0,1))

#             # 4. 训练器
#             trainer = PINNTrainer(model, pde_config, conditions, lr=params['lr'])
#             self.trainer = trainer

#             # 5. 保存模型和配置到可视化页面
#             self.vis_page.set_model(model, has_t)
#             self.vis_page.clear_history()

#             # 6. 启动训练线程
#             self.worker = TrainWorker(
#                 trainer, sampler, conditions,
#                 params['epochs'],
#                 params['batch_size'],
#                 params['n_boundary']
#             )
#             self.worker.progress.connect(self.vis_page.update_loss)
#             self.worker.finished.connect(self.training_finished)
#             self.worker.error.connect(self.show_error)
#             self.worker.start()

#             # 7. 更新UI状态
#             self.train_page.set_running(True)

#         except Exception as e:
#             QMessageBox.critical(self, "配置错误", str(e))

#     def training_finished(self):
#         self.train_page.set_running(False)
#         if self.trainer is not None:
#             self.vis_page.plot_solution()
#         # 切换到可视化页面
#         self.stack.setCurrentIndex(2)

#     def show_error(self, msg):
#         QMessageBox.critical(self, "训练错误", msg)
#         self.training_finished()

#     def save_figures(self):
#         dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录")
#         if dir_path:
#             self.vis_page.save_figures(dir_path)
#             QMessageBox.information(self, "提示", f"图片已保存至 {dir_path}")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     # # 全局样式
#     # app.setStyleSheet("""
#     #     QWidget {
#     #         background-color: #ecf0f1;
#     #     }
#     # """)
#     app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5', light=True))
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())