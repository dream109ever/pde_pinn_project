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

from typing import Optional, Callable, Dict, Union, List
import numpy as np
import torch

# Qt 依赖 (基于 PyQt5，若采用 PySide2/PySide6，只需微调此处 import)
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel
from PyQt5.QtCore import Qt

# Matplotlib Qt 后端
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# 导入数据准备与核心绘图逻辑
from .visualization import (
    prepare_1d_data,
    prepare_2d_data,
    prepare_transient_1d_data,
    prepare_transient_2d_data,
)

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 1. Training Loss 曲线控件
# ============================================================================
class LossPlotWidget(QWidget):
    """
    训练 Loss 曲线可视化控件
    支持展示 total loss 及各分项 loss (pde_loss, bc_loss, ic_loss 等)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
    def update_plot(self, history: Union[Dict[str, List[float]], List[float]]):
        """
        更新 Loss 曲线
        :param history: Loss 字典 (如 {'loss': [...], 'pde_loss': [...]}) 或单一 Loss 列表
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if isinstance(history, dict):
            for name, values in history.items():
                if len(values) > 0:
                    ax.plot(values, label=name, linewidth=1.5)
            ax.legend(loc='upper right')
        elif isinstance(history, (list, np.ndarray)):
            if len(history) > 0:
                ax.plot(history, label='Loss', color='blue', linewidth=1.5)
                ax.legend(loc='upper right')
        ax.set_yscale('log')
        ax.set_title('Training Loss Convergence', fontsize=11, fontweight='bold')
        ax.set_xlabel('Epoch / Iteration')
        ax.set_ylabel('Loss (Log Scale)')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
# ============================================================================
# 2. 一维稳态问题控件 (1D Steady)
# ============================================================================
class Steady1DPlotWidget(QWidget):
    """
    一维稳态问题可视化控件
    - 左图：PINN 预测解 vs 真实解对比曲线
    - 右图：空间绝对误差分布曲线
    - 右图角：标注全局相对 L2 误差及绝对误差统计
    """
    def __init__(self, parent=None, orientation: str = 'horizontal'):
        super().__init__(parent)
        self.orientation = orientation
        self.figure = Figure(figsize=(10, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
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
        if model is None:
            if true_func is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "无神经网络模型且无解析解", ha='center', va='center')
                self.canvas.draw()
                return
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            u_true = true_func(x_vals)
            ax = self.figure.add_subplot(111)
            ax.plot(x_vals, u_true, 'g-', linewidth=2, label='解析解 u(x)')
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title("1D 稳态解析解曲线")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            self.canvas.draw()
            return
        data = prepare_1d_data(model, x_range, n_points, true_func)
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        if u_true is not None:
            if self.orientation == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1)
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:  # vertical
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2)
            # 左图：解对比
            ax1.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
            ax1.plot(x, u_true, 'r--', label='Exact Solution', linewidth=2)
            ax1.set_title('1D Prediction vs Exact Solution', fontsize=11, fontweight='bold')
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # 右图：绝对误差
            abs_err = np.abs(u_pred - u_true)
            ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=2)
            ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
            ax2.set_title('1D Absolute Error Distribution', fontsize=11, fontweight='bold')
            ax2.set_xlabel('x')
            ax2.set_ylabel('$|u_{pred} - u_{true}|$')
            ax2.grid(True, alpha=0.3)
            # 量化指标 Text Box
            l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            text_str = (
                "$\mathbf{Global\ Metrics}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(
                0.04, 0.96, text_str,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
            )
        else:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
            ax.set_title('1D Predicted Solution', fontsize=11, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
# ============================================================================
# 3. 一维含时问题控件 (1D Transient)
# ============================================================================
class Transient1DPlotWidget(QWidget):
    """
    一维含时问题交互式可视化控件
    - 上方：画布 (当前时刻 t 的 1D 波形对比 + 绝对误差曲线)
    - 下方：Qt 时间滑块 (QSlider) 与时间展示标签
    """
    def __init__(self, parent=None, orientation: str = 'horizontal'):
        super().__init__(parent)
        self.orientation = orientation
        self.model = None
        self.exact_func = None
        self.x_range = (0, 1)
        self.t_range = (0, 1)
        self.n_points = 200
        self.figure = Figure(figsize=(10, 4.5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.label = QLabel("Time (t): 0.000")
        self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.label)
        control_layout.addWidget(self.slider)
        layout.addLayout(control_layout)
    def set_data(
        self,
        model: Optional[torch.nn.Module] = None,
        x_range: tuple = (0, 1),
        t_range: tuple = (0, 1),
        exact_func: Optional[Callable] = None,
        true_func: Optional[Callable] = None,
        n_points: int = 200,
    ):
        """配置模型与物理参数，初始化视图"""
        self.model = model
        self.x_range = x_range
        self.t_range = t_range
        self.exact_func = exact_func if exact_func is not None else true_func
        self.n_points = n_points
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
        if self.model is None:
            if self.exact_func is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "无模型且无解析解", ha='center', va='center')
                self.canvas.draw()
                return
            x_vals = np.linspace(self.x_range[0], self.x_range[1], self.n_points)
            u_true = self.exact_func(x_vals, t_val)
            ax = self.figure.add_subplot(111)
            ax.plot(x_vals, u_true, 'g-', linewidth=2, label=f'解析解 u(x, t={t_val:.2f})')
            ax.set_xlabel('x')
            ax.set_ylabel('u')
            ax.set_title(f"1D 含时解析解波形 (t = {t_val:.2f})")
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend()
            self.canvas.draw()
            return
        data = prepare_transient_1d_data( self.model, self.x_range, t_val, self.n_points, self.exact_func )
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        if u_true is not None:
            if self.orientation == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1)
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:  # vertical
                ax1 = self.figure.add_subplot(2, 1, 1)
                ax2 = self.figure.add_subplot(2, 1, 2)
            # 左图：预测 vs 真实
            ax1.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=2)
            ax1.plot(x, u_true, 'r--', label='Exact', linewidth=2)
            ax1.set_title(f'1D Profile (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x, t)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            # 右图：绝对误差
            abs_err = np.abs(u_pred - u_true)
            ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=2)
            ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
            ax2.set_title(f'1D Absolute Error (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax2.set_xlabel('x')
            ax2.set_ylabel('$|u_{pred} - u_{true}|$')
            ax2.grid(True, alpha=0.3)
            # 量化指标 Text Box
            l2_rel_err = np.linalg.norm(u_pred - u_true) / (np.linalg.norm(u_true) + 1e-8)
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(
                0.04, 0.96, text_str,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
            )
        else:
            ax = self.figure.add_subplot(1, 1, 1)
            ax.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=2)
            ax.set_title(f'1D Predicted Solution (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, t)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        self.figure.tight_layout()
        self.canvas.draw()
# ============================================================================
# 4. 二维稳态问题控件 (2D Steady)
# ============================================================================
class Steady2DPlotWidget(QWidget):
    """
    二维稳态问题可视化控件
    - 左图：PINN 预测解 3D 曲面图 (支持鼠标旋转视角)
    - 右图：2D 绝对误差等高线云图
    - 右图角：标注全局相对 L2 误差及绝对误差统计
    """
    def __init__(self, parent=None, orientation: str = 'horizontal'):
        super().__init__(parent)
        self.orientation = orientation
        self.figure = Figure(figsize=(11, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
    def set_data(
        self,
        model: Optional[torch.nn.Module] = None,
        x_range: tuple = (0, 1),
        y_range: tuple = (0, 1),
        true_func: Optional[Callable] = None,
        exact_func: Optional[Callable] = None,
        n_points: int = 100,
    ):
        """更新并绘制 2D 稳态预测结果"""
        true_func = true_func if true_func is not None else exact_func
        self.figure.clear()
        if model is None:
            if true_func is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "无模型且无解析解", ha='center', va='center')
                self.canvas.draw()
                return
            x_vals = np.linspace(x_range[0], x_range[1], n_points)
            y_vals = np.linspace(y_range[0], y_range[1], n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            Z_true = true_func(X, Y)
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_true, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title("2D 稳态解析解 3D 曲面")
            self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            self.canvas.draw()
            return
        data = prepare_2d_data(model, x_range, y_range, n_points, true_func)
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        if true_func is not None:
            Z_true = data['Z_true']
            if self.orientation == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1, projection='3d')
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:  # vertical
                ax1 = self.figure.add_subplot(2, 1, 1, projection='3d')
                ax2 = self.figure.add_subplot(2, 1, 2)
            # 左图：3D 预测曲面
            surf = ax1.plot_surface(
                X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True
            )
            ax1.set_title('3D Predicted Solution $u_{pred}(x, y)$', fontsize=11, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            self.figure.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
            # 右图：2D 绝对误差云图
            abs_err = np.abs(Z_pred - Z_true)
            cf = ax2.contourf(X, Y, abs_err, levels=50, cmap='inferno')
            self.figure.colorbar(cf, ax=ax2, label='Absolute Error $|u_{pred} - u_{true}|$')
            ax2.set_title('2D Absolute Error Distribution', fontsize=11, fontweight='bold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
            # 量化指标 Text Box
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                "$\mathbf{Global\ Metrics}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(
                0.04, 0.96, text_str,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
            )
        else:
            ax1 = self.figure.add_subplot(1, 1, 1, projection='3d')
            surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            ax1.set_title('3D Predicted Solution', fontsize=11, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            self.figure.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
        self.figure.tight_layout()
        self.canvas.draw()
# ============================================================================
# 5. 二维含时问题控件 (2D Transient)
# ============================================================================
class Transient2DPlotWidget(QWidget):
    """
    二维含时问题交互式可视化控件
    - 上方：画布 (当前时刻 t 的 3D 预测曲面图 + 2D 绝对误差云图)
    - 下方：Qt 时间滑块 (QSlider) 与时间展示标签
    """
    def __init__(self, parent=None, orientation: str = 'horizontal'):
        super().__init__(parent)
        self.orientation = orientation
        self.model = None
        self.exact_func = None
        self.x_range = (0, 1)
        self.y_range = (0, 1)
        self.t_range = (0, 1)
        self.n_points = 80
        self.figure = Figure(figsize=(11, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.label = QLabel("Time (t): 0.000")
        self.label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.label)
        control_layout.addWidget(self.slider)
        layout.addLayout(control_layout)
    def set_data(
        self,
        model: Optional[torch.nn.Module] = None,
        x_range: tuple = (0, 1),
        y_range: tuple = (0, 1),
        t_range: tuple = (0, 1),
        exact_func: Optional[Callable] = None,
        true_func: Optional[Callable] = None,
        n_points: int = 80,
    ):
        """配置模型与物理参数，初始化视图"""
        self.model = model
        self.x_range = x_range
        self.y_range = y_range
        self.t_range = t_range
        self.exact_func = exact_func if exact_func is not None else true_func
        self.n_points = n_points
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
        if self.model is None:
            if self.exact_func is None:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, "无神经网络模型且无解析解", ha='center', va='center')
                self.canvas.draw()
                return
            x_vals = np.linspace(self.x_range[0], self.x_range[1], self.n_points)
            y_vals = np.linspace(self.y_range[0], self.y_range[1], self.n_points)
            X, Y = np.meshgrid(x_vals, y_vals)
            try:
                Z_true = self.exact_func(X, Y, t_val)
                # 处理常数标量输出的情况
                if np.isscalar(Z_true):
                    Z_true = np.full_like(X, Z_true)
            except Exception as e:
                ax = self.figure.add_subplot(111)
                ax.text(0.5, 0.5, f"解析解计算失败: {e}", ha='center', va='center')
                self.canvas.draw()
                return
            # 绘制 3D 动态曲面
            ax = self.figure.add_subplot(111, projection='3d')
            surf = ax.plot_surface(X, Y, Z_true, cmap='viridis', edgecolor='none', alpha=0.9)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('u')
            ax.set_title(f"2D 含时解析解 3D 曲面 (t = {t_val:.2f})")
            self.figure.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            self.canvas.draw()
            return
        data = prepare_transient_2d_data( self.model, self.x_range, self.y_range, t_val, self.n_points, self.exact_func )
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        if self.exact_func is not None:
            Z_true = data['Z_true']
            if self.orientation == 'horizontal':
                ax1 = self.figure.add_subplot(1, 2, 1, projection='3d')
                ax2 = self.figure.add_subplot(1, 2, 2)
            else:  # vertical
                ax1 = self.figure.add_subplot(2, 1, 1, projection='3d')
                ax2 = self.figure.add_subplot(2, 1, 2)
            # 左图：3D 预测曲面 (按住鼠标可旋转)
            surf = ax1.plot_surface( X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95, antialiased=True )
            ax1.set_title(f'3D Predicted $u_{{pred}}$ (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            self.figure.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
            # 右图：2D 绝对误差云图
            abs_err = np.abs(Z_pred - Z_true)
            cf = ax2.contourf(X, Y, abs_err, levels=50, cmap='inferno')
            self.figure.colorbar(cf, ax=ax2, label='Absolute Error $|u_{pred} - u_{true}|$')
            ax2.set_title(f'2D Absolute Error (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
            # 动态量化指标 Text Box
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {abs_err.max():.3e}\n"
                f"• Mean Abs Error: {abs_err.mean():.3e}"
            )
            ax2.text(
                0.04, 0.96, text_str,
                transform=ax2.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
            )
        else:
            ax1 = self.figure.add_subplot(1, 1, 1, projection='3d')
            surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            ax1.set_title(f'3D Predicted Solution (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            self.figure.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
        self.figure.tight_layout()
        self.canvas.draw()
