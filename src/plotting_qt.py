# src/plotting_qt.py
"""
PINN 求解器 Qt 可视化控件模块
提供集成于 PyQt5/PySide 界面中的 Matplotlib 绘图画布控件，支持：
1. Training Loss 曲线实时/离线绘制
2. 1D 稳态/含时问题的双子图对比 (曲线 + 绝对误差)
3. 2D 稳态/含时问题的双子图对比 (3D 曲面 + 2D 绝对误差云图)
4. 图角全局相对 L2 误差等量化指标 Text Box 标注
5. Qt 原生 QSlider 时间滑块交互联动
"""

import torch
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton, QTabWidget
from PyQt5.QtCore import Qt
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
from typing import Optional, Callable, Dict, Union, List
from .plotting_core import (
    prepare_1d_data,
    prepare_2d_data,
    prepare_transient_1d_data,
    prepare_transient_2d_data,
)
matplotlib.use('Qt5Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 全局样式常量（统一控制字体、线条、图例等大小）
# ============================================================================
# 图表尺寸
PLOT_FIGSIZE_LOSS = (6, 4)      # Loss 曲线尺寸
PLOT_FIGSIZE_1D = (10, 4.5)     # 1D 稳态/含时尺寸
PLOT_FIGSIZE_2D = (11, 5)       # 2D 稳态/含时尺寸
# 字体大小
PLOT_TITLE_FONTSIZE = 7        # 标题字体
PLOT_AXIS_LABELSIZE = 7        # 坐标轴标签字体
PLOT_TICK_LABELSIZE = 7         # 刻度标签字体
PLOT_LEGEND_FONTSIZE = 7        # 图例字体
PLOT_TEXTBOX_FONTSIZE = 8       # Text Box 内部字体
# 线条粗细
PLOT_LINEWIDTH = 0.6            # 主线条宽度
PLOT_LINEWIDTH_THICK = 0.8      # 较粗线条（如误差填充线）
PLOT_LINEWIDTH_THIN = 0.4       # 较细线条（如辅助线）
# 其他
PLOT_CONTOUR_LEVELS = 50        # 等高线层数
PLOT_MARGIN_LEFT = 0.10
PLOT_MARGIN_RIGHT = 0.95
PLOT_MARGIN_TOP = 0.90
PLOT_MARGIN_BOTTOM = 0.12
# ============================================================================
# 0. 控件基类（统一处理 Qt 控件样式，避免重复代码）
# ============================================================================
class BasePlotWidget(QWidget):
    """
    所有绘图控件的基类，提供统一的 apply_theme 方法。
    子类只需在 __init__ 中创建 slider / label / tab_widget 等属性，
    基类会自动应用样式。
    """
    def apply_theme(self, theme):
        """外部调用，传入 theme 对象来更新 Qt 控件样式"""
        if theme is None:
            return
        # ---- QSlider 样式 ----
        slider_style = f"""
            QSlider::groove:horizontal {{
                height: 4px;
                background: {theme.btn_border};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {theme.btn_hover_bg};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {theme.btn_hover_bg};
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
            QSlider::handle:horizontal:hover {{
                background: {theme.btn_hover_border};
            }}
        """
        if hasattr(self, 'slider') and self.slider is not None:
            self.slider.setStyleSheet(slider_style)
        # ---- QLabel 样式 ----
        label_style = f"""
            color: {theme.text_primary};
            font-weight: bold;
            font-size: 12px;
            background: transparent;
        """
        if hasattr(self, 'label') and self.label is not None:
            self.label.setStyleSheet(label_style)
        # ---- QTabWidget 样式 ----
        tab_style = f"""
            QTabWidget::pane {{
                background-color: {theme.card_bg};
                border: 1px solid {theme.btn_border};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {theme.btn_bg};
                color: {theme.text_primary};
                padding: 4px 12px;
                border: 1px solid {theme.btn_border};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
            }}
            QTabBar::tab:hover {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
            }}
        """
        if hasattr(self, 'tab_widget') and self.tab_widget is not None:
            self.tab_widget.setStyleSheet(tab_style)
    def _clear_colorbars(self, fig):
        if fig is None:
            fig = self.figure
        for ax in list(fig.axes):
            if isinstance(ax, matplotlib.colorbar.Colorbar) or \
               'colorbar' in ax.__class__.__name__.lower() or \
               (hasattr(ax, 'get_label') and ax.get_label() and 'colorbar' in ax.get_label().lower()):
                try:
                    ax.remove()
                except Exception:
                    pass
                if ax in fig.axes:
                    fig.axes.remove(ax)
        fig.set_layout_engine('none')
        fig.set_layout_engine('constrained')
        fig.canvas.draw_idle()
# ============================================================================
# 1. Training Loss 曲线控件
# ============================================================================
def _apply_theme_to_figure(fig):
    """统一为 Figure 和所有 Axes 应用主题背景色"""
    try:
        from ui.theme_manager import ThemeManager
        import re
        theme = ThemeManager.instance().current
        bg_color = theme.card_bg
        text_color = theme.text_primary
        def parse_color(c):
            if not c: return '#FFFFFF'
            rgba_match = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)', c)
            if rgba_match:
                return (int(rgba_match.group(1))/255.0, int(rgba_match.group(2))/255.0, 
                        int(rgba_match.group(3))/255.0, float(rgba_match.group(4)) if rgba_match.group(4) else 1.0)
            return c
        bg = parse_color(bg_color)
        fg = parse_color(text_color)
        fig.patch.set_facecolor(bg)
        for ax in fig.axes:
            ax.set_facecolor(bg)
            ax.title.set_color(fg)
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            if hasattr(ax, 'zaxis'):
                ax.zaxis.label.set_color(fg)
            ax.tick_params(colors=fg, labelcolor=fg)
    except:
        pass
class LossPlotWidget(QWidget):
    """训练 Loss 曲线可视化控件"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=PLOT_FIGSIZE_LOSS, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_yscale('log')
        self.ax.tick_params(labelsize=PLOT_TICK_LABELSIZE)
        self.ax.yaxis.set_major_locator(ticker.LogLocator(numticks=6))
        self.ax.yaxis.set_minor_locator(ticker.NullLocator()) 
        self.ax.set_title('Training Loss', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        self.ax.set_xlabel('Epoch / Iteration', fontsize=PLOT_AXIS_LABELSIZE)
        self.ax.set_ylabel('Loss', fontsize=PLOT_AXIS_LABELSIZE)
        self.ax.grid(True, which='both', linestyle='--', alpha=0.3)
        # self.figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        self.placeholder_text = self.ax.text(
            0.5, 0.5, "等待训练数据...", ha='center', va='center',
            fontsize=PLOT_AXIS_LABELSIZE, color='gray', transform=self.ax.transAxes
        )
        self.lines = {}
        _apply_theme_to_figure(self.figure)
        self.canvas.draw()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
    def update_plot(self, history: Union[Dict[str, List[float]], List[float]]):
        """更新损失曲线（只更新数据，不重建整图）"""
        if self.placeholder_text is not None:
            self.placeholder_text.remove()
            self.placeholder_text = None
        if isinstance(history, dict):
            for name, values in history.items():
                if len(values) > 0:
                    if name not in self.lines:
                        line, = self.ax.plot(values, label=name, linewidth=PLOT_LINEWIDTH)
                        self.lines[name] = line
                    else:
                        self.lines[name].set_data(range(len(values)), values)
            if self.lines:
                self.ax.legend(loc='upper right', fontsize=PLOT_LEGEND_FONTSIZE)
        elif isinstance(history, (list, np.ndarray)):
            name = 'Loss'
            values = history
            if len(values) > 0:
                if name not in self.lines:
                    line, = self.ax.plot(values, label=name, color='blue', linewidth=PLOT_LINEWIDTH)
                    self.lines[name] = line
                    self.ax.legend(loc='upper right', fontsize=PLOT_LEGEND_FONTSIZE)
                else:
                    self.lines[name].set_data(range(len(values)), values)
        if self.lines:
            max_len = 0
            all_vals = []
            for line in self.lines.values():
                ydata = line.get_ydata()
                if len(ydata) > 0:
                    max_len = max(max_len, len(ydata))
                    all_vals.extend(ydata)
            if max_len > 0 and all_vals:
                self.ax.set_xlim(-max_len * 0.02, max_len * 1.02)
                y_min = min(all_vals)
                y_max = max(all_vals)
                if y_min > 0:
                    self.ax.set_ylim(y_min * 0.5, y_max * 2)
                else:
                    self.ax.set_ylim(1e-7, y_max * 2)
        _apply_theme_to_figure(self.figure)
        self.canvas.draw()
# ============================================================================
# 2. 一维稳态问题控件 (1D Steady)
# ============================================================================
class Steady1DPlotWidget(BasePlotWidget):
    """
    一维稳态问题可视化控件
    - 支持三种布局模式:
      * 'horizontal': 左右并排 (预测+误差)
      * 'vertical': 上下并排 (预测+误差)
      * 'overlay': 叠加切换 (通过左上角按钮切换显示预测或误差)
    """
    def __init__(self, parent=None, mode: str = 'horizontal'):
        super().__init__(parent)
        self.mode = mode
        self._overlay_state = 0  # 0: 显示预测, 1: 显示误差
        self._has_true = False
        self._current_data = None
        self.figure = Figure(figsize=PLOT_FIGSIZE_1D, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # 布局：画布 + 控制栏
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        # overlay 模式使用 QTabWidget
        if self.mode == 'overlay':
            self.tab_widget = QTabWidget()
            self.tab_widget.setVisible(True)
            # 标签页1: 预测解
            self.pred_figure = Figure(figsize=PLOT_FIGSIZE_1D)
            self.pred_canvas = FigureCanvasQTAgg(self.pred_figure)
            self.pred_ax = self.pred_figure.add_subplot(111)
            self.tab_widget.addTab(self.pred_canvas, "预测解")
            # 标签页2: 误差图
            self.err_figure = Figure(figsize=PLOT_FIGSIZE_1D)
            self.err_canvas = FigureCanvasQTAgg(self.err_figure)
            self.err_ax = self.err_figure.add_subplot(111)
            self.tab_widget.addTab(self.err_canvas, "误差图")
            main_layout.addWidget(self.tab_widget)
        else:
            main_layout.addWidget(self.canvas)
    def _update_overlay_display_tab(self):
        """overlay 模式：分别更新两个标签页"""
        if self._current_data is None:
            return
        data = self._current_data
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        # --- 标签页1: 预测解 ---
        self.pred_figure.clear()
        self.pred_ax = self.pred_figure.add_subplot(111)
        if u_pred is not None:
            self.pred_ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=PLOT_LINEWIDTH_THICK)
        if u_true is not None:
            self.pred_ax.plot(x, u_true, 'r--', label='Exact Solution', linewidth=PLOT_LINEWIDTH_THICK)
        self.pred_ax.set_title('1D Prediction vs Exact Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        self.pred_ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_ylabel('u(x)', fontsize=PLOT_AXIS_LABELSIZE)
        if u_pred is not None or u_true is not None:
            self.pred_ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        self.pred_ax.grid(True, alpha=0.3)
        # self.pred_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.pred_figure)
        self.pred_canvas.draw_idle()
        # --- 标签页2: 误差图 ---
        self.err_figure.clear()
        self.err_ax = self.err_figure.add_subplot(111)
        if u_true is not None and u_pred is not None:
            abs_err = np.abs(u_pred - u_true)
            self.err_ax.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=PLOT_LINEWIDTH_THICK)
            self.err_ax.fill_between(x, abs_err, color='magenta', alpha=0.15)
            l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            text_str = (
                "$\mathbf{Global\ Metrics}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            self.err_ax.text(0.04, 0.96, text_str, transform=self.err_ax.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            self.err_ax.set_title('1D Absolute Error Distribution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            self.err_ax.set_ylabel('$|u_{pred} - u_{true}|$', fontsize=PLOT_AXIS_LABELSIZE)
            self.err_ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            self.err_ax.text(0.5, 0.5, "无误差数据", ha='center', va='center')
            self.err_ax.set_title('1D Error (No data)')
        self.err_ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.grid(True, alpha=0.3)
        # self.err_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.err_figure)
        self.err_canvas.draw_idle()
    def set_data(
        self,
        model: Optional[torch.nn.Module] = None,
        x_range: tuple = (0, 1),
        true_func: Optional[Callable] = None,
        exact_func: Optional[Callable] = None,
        n_points: int = 200,
    ):
        """更新并绘制 1D 稳态预测结果"""
        true_func = true_func if true_func is not None else exact_func
        self.figure.clear()
        # 准备数据
        data = prepare_1d_data(model, x_range, n_points, true_func)
        self._current_data = data
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        self._has_true = (u_true is not None)
        # 判断显示模式
        if self.mode == 'overlay':
            self._update_overlay_display_tab()
            return
        if u_true is not None and u_pred is not None:
            if self.mode == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1)
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:  # vertical
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2)
            # 左/上：解对比
            ax1.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=PLOT_LINEWIDTH_THICK) if u_pred is not None else None
            ax1.plot(x, u_true, 'r--', label='Exact Solution', linewidth=PLOT_LINEWIDTH_THICK)
            ax1.set_title('1D Prediction vs Exact Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax1.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_ylabel('u(x)', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            ax1.grid(True, alpha=0.3)
            # 右/下：绝对误差
            if u_pred is not None:
                abs_err = np.abs(u_pred - u_true)
                ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=PLOT_LINEWIDTH_THICK)
                ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
                l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
                text_str = (
                    "$\mathbf{Global\ Metrics}$\n"
                    f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                    f"• Max Abs Error: {abs_err.max():.3e}\n"
                    f"• Mean Abs Error: {abs_err.mean():.3e}"
                )
                ax2.text(0.04, 0.96, text_str, transform=ax2.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
                ax2.set_title('1D Absolute Error Distribution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
                ax2.set_ylabel('$|u_{pred} - u_{true}|$', fontsize=PLOT_AXIS_LABELSIZE)
            else:
                ax2.text(0.5, 0.5, "无误差数据（无预测解）", ha='center', va='center')
                ax2.set_title('1D Error (No prediction)')
            ax2.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.grid(True, alpha=0.3)
        elif u_true is not None:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.plot(x, u_true, 'g-', linewidth=PLOT_LINEWIDTH_THICK, label='Analytical Solution')
            ax.set_title('Analytical Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('u(x)', fontsize=PLOT_AXIS_LABELSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            ax.grid(True, alpha=0.3)
        else:
            # 无真实解，单图显示预测
            ax = self.figure.add_subplot(1, 1, 1)
            if u_pred is not None:
                ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=PLOT_LINEWIDTH_THICK)
                ax.set_title('Predicted Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
                ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            else:
                ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=PLOT_AXIS_LABELSIZE)
                ax.set_title('No Data', fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('u(x)', fontsize=PLOT_AXIS_LABELSIZE)
            ax.grid(True, alpha=0.3)
        _apply_theme_to_figure(self.figure)
        # self.figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        self.canvas.draw()
# ============================================================================
# 3. 一维含时问题控件 (1D Transient)
# ============================================================================
class Transient1DPlotWidget(BasePlotWidget):
    """
    一维含时问题交互式可视化控件
    支持 horizontal / vertical / overlay 三种布局模式
    """
    def __init__(self, parent=None, mode: str = 'horizontal'):
        super().__init__(parent)
        self.mode = mode
        self._overlay_state = 0
        self._has_true = False
        self._current_data = None
        self.model = None
        self.exact_func = None
        self.x_range = (0, 1)
        self.t_range = (0, 1)
        self.n_points = 200
        self.figure = Figure(figsize=PLOT_FIGSIZE_1D, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        # 控制栏
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        if self.mode == 'overlay':
            # overlay 模式：使用 QTabWidget
            self.tab_widget = QTabWidget()
            self.tab_widget.setVisible(True)
            # 标签页1：预测解
            self.pred_figure = Figure(figsize=PLOT_FIGSIZE_1D)
            self.pred_canvas = FigureCanvasQTAgg(self.pred_figure)
            self.pred_ax = self.pred_figure.add_subplot(111)
            self.tab_widget.addTab(self.pred_canvas, "预测解")
            # 标签页2：误差图
            self.err_figure = Figure(figsize=PLOT_FIGSIZE_1D)
            self.err_canvas = FigureCanvasQTAgg(self.err_figure)
            self.err_ax = self.err_figure.add_subplot(111)
            self.tab_widget.addTab(self.err_canvas, "误差图")
            main_layout.addWidget(self.tab_widget)
            # 时间滑块（overlay 模式下也要有）
            self.label = QLabel("Time (t): 0.000")
            self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self._on_slider_changed)
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.label)
            slider_layout.addWidget(self.slider)
            main_layout.addLayout(slider_layout)
        else:
            self.label = QLabel("Time (t): 0.000")
            self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self._on_slider_changed)
            main_layout.addWidget(self.canvas)
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.label)
            slider_layout.addWidget(self.slider)
            main_layout.addLayout(slider_layout)
    def _update_overlay_display_tab(self):
        """overlay 模式：分别更新两个标签页"""
        if self._current_data is None:
            return
        data = self._current_data
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        t_val = data.get('t_val', 0.0)
        # --- 标签页1: 预测解 ---
        self.pred_figure.clear()
        self.pred_ax = self.pred_figure.add_subplot(111)
        if u_pred is not None:
            self.pred_ax.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=PLOT_LINEWIDTH_THICK)
        if u_true is not None:
            self.pred_ax.plot(x, u_true, 'r--', label='Exact', linewidth=PLOT_LINEWIDTH_THICK)
        self.pred_ax.set_title(f'1D Profile (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        self.pred_ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_ylabel('u(x, t)', fontsize=PLOT_AXIS_LABELSIZE)
        if u_pred is not None or u_true is not None:
            self.pred_ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        self.pred_ax.grid(True, alpha=0.3)
        # self.pred_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.pred_figure)
        self.pred_canvas.draw_idle()
        # --- 标签页2: 误差图 ---
        self.err_figure.clear()
        self.err_ax = self.err_figure.add_subplot(111)
        if u_true is not None and u_pred is not None:
            abs_err = np.abs(u_pred - u_true)
            self.err_ax.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=PLOT_LINEWIDTH_THICK)
            self.err_ax.fill_between(x, abs_err, color='magenta', alpha=0.15)
            l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            self.err_ax.text(0.04, 0.96, text_str, transform=self.err_ax.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            self.err_ax.set_title(f'1D Absolute Error (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            self.err_ax.set_ylabel('$|u_{pred} - u_{true}|$', fontsize=PLOT_AXIS_LABELSIZE)
            self.err_ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
        else:
            self.err_ax.text(0.5, 0.5, "无误差数据", ha='center', va='center')
            self.err_ax.set_title('1D Error (No data)')
        self.err_ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.grid(True, alpha=0.3)
        # self.err_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.err_figure)
        self.err_canvas.draw_idle()
    def set_data(self, model=None, x_range=(0, 1), t_range=(0, 1),
                 exact_func=None, true_func=None, n_points=200):
        self.model = model
        self.x_range = x_range
        self.t_range = t_range
        self.exact_func = exact_func if exact_func is not None else true_func
        self.n_points = n_points
        self.slider.setValue(0)
        self.update_plot()
    def _on_slider_changed(self, value: int):
        t_min, t_max = self.t_range
        t_val = t_min + (t_max - t_min) * (value / 100.0)
        self.label.setText(f"Time (t): {t_val:.3f}")
        self.update_plot(t_val)
    def update_plot(self, t_val: Optional[float] = None):
        t_min, t_max = self.t_range
        if t_val is None:
            t_val = t_min + (t_max - t_min) * (self.slider.value() / 100.0)
        self.figure.clear()
        # 准备数据
        data = prepare_transient_1d_data(self.model, self.x_range, t_val, self.n_points, self.exact_func)
        self._current_data = data
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        self._has_true = (u_true is not None)
        if self.mode == 'overlay':
            self._update_overlay_display_tab()
            return
        if u_true is not None and u_pred is not None:
            if self.mode == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1)
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2)
            ax1.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=PLOT_LINEWIDTH_THICK) if u_pred is not None else None
            ax1.plot(x, u_true, 'r--', label='Exact', linewidth=PLOT_LINEWIDTH_THICK)
            ax1.set_title(f'1D Profile (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax1.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_ylabel('u(x, t)', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            ax1.grid(True, alpha=0.3)
            if u_pred is not None:
                abs_err = np.abs(u_pred - u_true)
                ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=PLOT_LINEWIDTH_THICK)
                ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
                l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
                text_str = (
                    f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                    f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                    f"• Max Abs Error: {abs_err.max():.3e}\n"
                    f"• Mean Abs Error: {abs_err.mean():.3e}"
                )
                ax2.text(0.04, 0.96, text_str, transform=ax2.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                         verticalalignment='top', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
                ax2.set_title(f'1D Absolute Error (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
                ax2.set_ylabel('$|u_{pred} - u_{true}|$', fontsize=PLOT_AXIS_LABELSIZE)
            else:
                ax2.text(0.5, 0.5, "无误差数据", ha='center', va='center')
                ax2.set_title('1D Error (No prediction)')
            ax2.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.grid(True, alpha=0.3)
        elif u_true is not None:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.plot(x, u_true, 'g-', linewidth=PLOT_LINEWIDTH_THICK, label=f'Analytical Solution (t={t_val:.3f})')
            ax.set_title(f'Analytical Solution (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('u(x, t)', fontsize=PLOT_AXIS_LABELSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            ax.grid(True, alpha=0.3)
        elif u_pred is not None:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=PLOT_LINEWIDTH_THICK)
            ax.set_title(f'Predicted Solution (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('u(x, t)', fontsize=PLOT_AXIS_LABELSIZE)
            ax.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            ax.grid(True, alpha=0.3)
        else:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_title('No Data', fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel('x', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('u(x, t)', fontsize=PLOT_AXIS_LABELSIZE)
            ax.grid(True, alpha=0.3)
        _apply_theme_to_figure(self.figure)
        # self.figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        self.canvas.draw()
# ============================================================================
# 4. 二维稳态问题控件 (2D Steady)
# ============================================================================
class Steady2DPlotWidget(BasePlotWidget):
    """二维稳态问题可视化控件，支持 horizontal / vertical / overlay 三种模式"""
    def __init__(self, parent=None, mode: str = 'horizontal'):
        super().__init__(parent)
        self.mode = mode
        self._overlay_state = 0
        self._has_true = False
        self._current_data = None
        self.figure = Figure(figsize=PLOT_FIGSIZE_2D, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        if self.mode == 'overlay':
            # overlay 模式：使用 QTabWidget
            self.tab_widget = QTabWidget()
            self.tab_widget.setVisible(True)
            # 标签页1：预测解（3D 曲面）
            self.pred_figure = Figure(figsize=PLOT_FIGSIZE_2D)
            self.pred_canvas = FigureCanvasQTAgg(self.pred_figure)
            self.pred_ax = self.pred_figure.add_subplot(111, projection='3d')
            self.tab_widget.addTab(self.pred_canvas, "预测解")
            # 标签页2：误差图（2D 云图）
            self.err_figure = Figure(figsize=PLOT_FIGSIZE_2D)
            self.err_canvas = FigureCanvasQTAgg(self.err_figure)
            self.err_ax = self.err_figure.add_subplot(111)
            self.tab_widget.addTab(self.err_canvas, "误差图")
            main_layout.addWidget(self.tab_widget)
        else:
            main_layout.addWidget(self.canvas)
    def _update_overlay_display_tab(self):
        """overlay 模式：分别更新两个标签页"""
        if self._current_data is None:
            return
        data = self._current_data
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        Z_true = data['Z_true']
        # --- 标签页1: 预测解 (3D 曲面) ---
        self.pred_figure.clear()
        self.pred_ax = self.pred_figure.add_subplot(111, projection='3d')
        if Z_pred is not None:
            surf = self.pred_ax.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            self._clear_colorbars(self.pred_figure)
            self.pred_figure.colorbar(surf, ax=self.pred_ax, shrink=0.5, aspect=10, pad=0.08)
        self.pred_ax.set_title('3D Predicted Solution $u_{pred}(x, y)$', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        self.pred_ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.view_init(elev=30, azim=-60)
        # self.pred_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.pred_figure)
        self.pred_canvas.draw_idle()
        # --- 标签页2: 误差图 (2D 云图) ---
        self.err_figure.clear()
        self.err_ax = self.err_figure.add_subplot(111)
        if Z_pred is not None and Z_true is not None:
            abs_err = np.abs(Z_pred - Z_true)
            cf = self.err_ax.contourf(X, Y, abs_err, levels=PLOT_CONTOUR_LEVELS, cmap='inferno')
            self._clear_colorbars(self.err_figure)
            self.err_figure.colorbar(cf, ax=self.err_ax, label='Absolute Error $|u_{pred} - u_{true}|$')
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                "$\mathbf{Global\ Metrics}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            self.err_ax.text(0.04, 0.96, text_str, transform=self.err_ax.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            self.err_ax.set_title('2D Absolute Error Distribution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        else:
            self.err_ax.text(0.5, 0.5, "无误差数据", ha='center', va='center')
            self.err_ax.set_title('2D Error (No data)')
        self.err_ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.set_aspect('equal')
        # self.err_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.err_figure)
        self.err_canvas.draw_idle()
    def set_data(self, model=None, x_range=(0, 1), y_range=(0, 1), true_func=None, exact_func=None, n_points=100):
        true_func = true_func if true_func is not None else exact_func
        self.figure.clear()
        data = prepare_2d_data(model, x_range, y_range, n_points, true_func)
        self._current_data = data
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        Z_true = data['Z_true']
        self._has_true = (Z_true is not None)
        if self.mode == 'overlay':
            self._update_overlay_display_tab()
            return
        if Z_true is not None and Z_pred is not None:
            if self.mode == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1, projection='3d')
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:
                ax1 = self.figure.add_subplot(2, 1, 1, projection='3d')
                ax2 = self.figure.add_subplot(2, 1, 2)
            surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            ax1.set_title('3D Predicted Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax1.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.view_init(elev=30, azim=-60)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
            abs_err = np.abs(Z_pred - Z_true)
            cf = ax2.contourf(X, Y, abs_err, levels=PLOT_CONTOUR_LEVELS, cmap='inferno')
            self._clear_colorbars(self.figure)
            self.figure.colorbar(cf, ax=ax2, label='Absolute Error')
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                "$\mathbf{Global\ Metrics}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(0.04, 0.96, text_str, transform=ax2.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            ax2.set_title('2D Absolute Error', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax2.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.set_aspect('equal')
        elif Z_true is not None:
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_true, cmap='plasma', edgecolor='none', alpha=0.9)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            ax.set_title('Analytical Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax.view_init(elev=30, azim=-60)
        elif Z_pred is not None:
            # 只有预测
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
            ax.set_title('Predicted Solution', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax.view_init(elev=30, azim=-60)
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_title('No Data', fontsize=PLOT_TITLE_FONTSIZE)
        _apply_theme_to_figure(self.figure)
        # self.figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        self.canvas.draw()
# ============================================================================
# 5. 二维含时问题控件 (2D Transient)
# ============================================================================
class Transient2DPlotWidget(BasePlotWidget):
    """二维含时问题交互式可视化控件，支持 horizontal / vertical / overlay 三种模式"""
    def __init__(self, parent=None, mode: str = 'horizontal'):
        super().__init__(parent)
        self.mode = mode
        self._overlay_state = 0
        self._has_true = False
        self._current_data = None
        self.model = None
        self.exact_func = None
        self.x_range = (0, 1)
        self.y_range = (0, 1)
        self.t_range = (0, 1)
        self.n_points = 60
        self.figure = Figure(figsize=PLOT_FIGSIZE_2D, constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        if self.mode == 'overlay':
            # overlay 模式：使用 QTabWidget
            self.tab_widget = QTabWidget()
            self.tab_widget.setVisible(True)
            # 标签页1：预测解（3D 曲面）
            self.pred_figure = Figure(figsize=PLOT_FIGSIZE_2D)
            self.pred_canvas = FigureCanvasQTAgg(self.pred_figure)
            self.pred_ax = self.pred_figure.add_subplot(111, projection='3d')
            self.tab_widget.addTab(self.pred_canvas, "预测解")
            # 标签页2：误差图（2D 云图）
            self.err_figure = Figure(figsize=PLOT_FIGSIZE_2D)
            self.err_canvas = FigureCanvasQTAgg(self.err_figure)
            self.err_ax = self.err_figure.add_subplot(111)
            self.tab_widget.addTab(self.err_canvas, "误差图")
            main_layout.addWidget(self.tab_widget)
            # 时间滑块（overlay 模式下也要有）
            self.label = QLabel("Time (t): 0.000")
            self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self._on_slider_changed)
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.label)
            slider_layout.addWidget(self.slider)
            main_layout.addLayout(slider_layout)
        else:
            self.label = QLabel("Time (t): 0.000")
            self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setRange(0, 100)
            self.slider.setValue(0)
            self.slider.valueChanged.connect(self._on_slider_changed)
            main_layout.addWidget(self.canvas)
            slider_layout = QHBoxLayout()
            slider_layout.addWidget(self.label)
            slider_layout.addWidget(self.slider)
            main_layout.addLayout(slider_layout)
    def _update_overlay_display_tab(self):
        """overlay 模式：分别更新两个标签页"""
        if self._current_data is None:
            return
        data = self._current_data
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        Z_true = data['Z_true']
        t_val = data.get('t_val', 0.0)
        # --- 标签页1: 预测解 (3D 曲面) ---
        self.pred_figure.clear()
        self.pred_ax = self.pred_figure.add_subplot(111, projection='3d')
        if Z_pred is not None:
            surf = self.pred_ax.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            self._clear_colorbars(self.pred_figure)
            self.pred_figure.colorbar(surf, ax=self.pred_ax, shrink=0.5, aspect=10, pad=0.08)
        self.pred_ax.set_title(f'3D Predicted $u_{{pred}}$ (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        self.pred_ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
        self.pred_ax.view_init(elev=30, azim=-60)
        # self.pred_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.pred_figure)
        self.pred_canvas.draw_idle()
        # --- 标签页2: 误差图 (2D 云图) ---
        self.err_figure.clear()
        self.err_ax = self.err_figure.add_subplot(111)
        if Z_pred is not None and Z_true is not None:
            abs_err = np.abs(Z_pred - Z_true)
            cf = self.err_ax.contourf(X, Y, abs_err, levels=PLOT_CONTOUR_LEVELS, cmap='inferno')
            self._clear_colorbars(self.err_figure)
            self.err_figure.colorbar(cf, ax=self.err_ax, label='Absolute Error $|u_{pred} - u_{true}|$')
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            self.err_ax.text(0.04, 0.96, text_str, transform=self.err_ax.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            self.err_ax.set_title(f'2D Absolute Error (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
        else:
            self.err_ax.text(0.5, 0.5, "无误差数据", ha='center', va='center')
            self.err_ax.set_title('2D Error (No data)')
        self.err_ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
        self.err_ax.set_aspect('equal')
        # self.err_figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        _apply_theme_to_figure(self.err_figure)
        self.err_canvas.draw_idle()
    def set_data(self, model=None, x_range=(0, 1), y_range=(0, 1), t_range=(0, 1),
                 exact_func=None, true_func=None, n_points=80):
        self.model = model
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.exact_func = exact_func if exact_func is not None else true_func
        self.n_points = n_points
        self.slider.setValue(0)
        self.update_plot()
    def _on_slider_changed(self, value: int):
        t_min, t_max = self.t_range
        t_val = t_min + (t_max - t_min) * (value / 100.0)
        self.label.setText(f"Time (t): {t_val:.3f}")
        self.update_plot(t_val)
    def update_plot(self, t_val: Optional[float] = None):
        t_min, t_max = self.t_range
        if t_val is None:
            t_val = t_min + (t_max - t_min) * (self.slider.value() / 100.0)
        self.figure.clear()
        data = prepare_transient_2d_data(self.model, self.x_range, self.y_range, t_val, self.n_points, self.exact_func)
        self._current_data = data
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        Z_true = data['Z_true']
        self._has_true = (Z_true is not None)
        if self.mode == 'overlay':
            self._update_overlay_display_tab()
            return
        if Z_true is not None and Z_pred is not None:
            if self.mode == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1, projection='3d')
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:
                ax1 = self.figure.add_subplot(2, 1, 1, projection='3d')
                ax2 = self.figure.add_subplot(2, 1, 2)
            surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            ax1.set_title(f'3D Predicted (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax1.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax1.view_init(elev=30, azim=-60)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
            abs_err = np.abs(Z_pred - Z_true)
            cf = ax2.contourf(X, Y, abs_err, levels=PLOT_CONTOUR_LEVELS, cmap='inferno')
            self._clear_colorbars(self.figure)
            self.figure.colorbar(cf, ax=ax2, label='Absolute Error')
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(0.04, 0.96, text_str, transform=ax2.transAxes, fontsize=PLOT_TEXTBOX_FONTSIZE,
                     verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc'))
            ax2.set_title(f'2D Error (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax2.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax2.set_aspect('equal')
        elif Z_true is not None:
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_true, cmap='plasma', edgecolor='none', alpha=0.9)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
            ax.set_title(f'Analytical Solution (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax.view_init(elev=30, azim=-60)
        elif Z_pred is not None:
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            self._clear_colorbars(self.figure)
            self.figure.colorbar(surf, ax=ax, shrink=0.6, aspect=10)
            ax.set_title(f'Predicted Solution (t = {t_val:.3f})', fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_zlabel('u', fontsize=PLOT_AXIS_LABELSIZE)
            ax.view_init(elev=30, azim=-60)
        else:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "无数据", ha='center', va='center', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_title('No Data', fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlabel('X', fontsize=PLOT_AXIS_LABELSIZE)
            ax.set_ylabel('Y', fontsize=PLOT_AXIS_LABELSIZE)
        _apply_theme_to_figure(self.figure)
        # self.figure.subplots_adjust(left=PLOT_MARGIN_LEFT, right=PLOT_MARGIN_RIGHT, top=PLOT_MARGIN_TOP, bottom=PLOT_MARGIN_BOTTOM)
        self.canvas.draw()
