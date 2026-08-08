# src/visualization.py
"""
Notebook 环境下的可视化包装层。

本模块依赖 plotting_core 提供绘图逻辑，仅负责 Notebook 环境下的展示。
所有绘图函数都调用 plotting_core 中的 draw_* 函数 + 数据准备函数，
然后使用 plt.show() 显示。
"""

import torch
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display
from typing import Optional, Callable
from .plotting_core import (
    _safe_call_exact_func, 
    prepare_1d_data,
    prepare_2d_data,
    prepare_transient_1d_data,
    prepare_transient_2d_data,
    prepare_1d_time_slice_data,
    draw_loss_curve,
    draw_1d_time_slice,
    draw_1d_multiple_models,
    draw_3d_surface,
    draw_3d_compare,
)

# 绘图函数
def plot_loss_history(history, figsize=(10, 6), log_scale=True, save_path=None):
    """
    绘制训练损失曲线（Notebook 版本）。

    :param history: 包含 'total_loss', 'pde_loss', 'bc_loss' 的训练历史字典
    :type history: dict
    :param figsize: 图形尺寸 (width, height)
    :type figsize: tuple
    :param log_scale: 是否使用对数 Y 轴
    :type log_scale: bool
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw_loss_curve(ax, history, log_scale)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_1d_solution(
    model,
    x_range=(0, 1),
    n_points=200,
    true_func=None,
    title='1D Steady Solution',
    save_path=None,
):
    """
    一维稳态问题：双子图对比（曲线对比 + 绝对误差曲线 + 全局 L2 指标）。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param n_points: 采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    data = prepare_1d_data(model, x_range, n_points, true_func)
    x = data['x']
    u_pred = data['u_pred']
    u_true = data['u_true']
    if u_true is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
        ax1.plot(x, u_true, 'r--', label='Exact Solution', linewidth=2)
        ax1.set_title('1D Prediction vs Exact Solution', fontsize=11, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('u(x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        abs_err = np.abs(u_pred - u_true)
        ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=2)
        ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
        ax2.set_title('1D Absolute Error Distribution', fontsize=11, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('$|u_{pred} - u_{true}|$')
        ax2.grid(True, alpha=0.3)
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
            fontsize=9.5,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_1d_transient_interactive(
    model: torch.nn.Module,
    x_range: tuple = (0, 1),
    t_range: tuple = (0, 1),
    n_points: int = 200,
    exact_func: Optional[Callable] = None,
    title: str = "1D Transient Solution",
    save_path: Optional[str] = None,
):
    """
    一维含时问题：交互式双子图（滑块控制 + 曲线对比 + 绝对误差曲线 + 动态 L2 指标）。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param t_range: t 轴范围 (t_min, t_max)
    :type t_range: tuple
    :param n_points: 采样点数
    :type n_points: int
    :param exact_func: 真实解函数
    :type exact_func: Optional[Callable]
    :param title: 图标题前缀
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    x_min, x_max = x_range
    t_min, t_max = t_range
    def _update_plot(t_val):
        plt.close('all')
        data = prepare_transient_1d_data(model, x_range, t_val, n_points, exact_func)
        x = data['x']
        u_pred = data['u_pred']
        u_true = data['u_true']
        if u_true is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=2)
            ax1.plot(x, u_true, 'r--', label='Exact', linewidth=2)
            ax1.set_title(f'1D Solution Profile (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax1.set_xlabel('x')
            ax1.set_ylabel('u(x, t)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            abs_err = np.abs(u_pred - u_true)
            ax2.plot(x, abs_err, 'm-', label='Absolute Error', linewidth=2)
            ax2.fill_between(x, abs_err, color='magenta', alpha=0.15)
            ax2.set_title(f'1D Absolute Error (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax2.set_xlabel('x')
            ax2.set_ylabel('$|u_{pred} - u_{true}|$')
            ax2.grid(True, alpha=0.3)
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
                fontsize=9.5,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x, u_pred, 'b-', label=f'PINN (t={t_val:.3f})', linewidth=2)
            ax.set_title(f'{title} (t = {t_val:.3f})', fontsize=11, fontweight='bold')
            ax.set_xlabel('x')
            ax.set_ylabel('u(x, t)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    slider = widgets.FloatSlider(
        value=t_min,
        min=t_min,
        max=t_max,
        step=(t_max - t_min) / 100,
        description='Time (t):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    interactive_plot = widgets.interactive(_update_plot, t_val=slider)
    display(interactive_plot)
def plot_1d_time_slice(
    model: torch.nn.Module,
    x_val: float,
    t_range: tuple = (0, 1),
    n_points: int = 200,
    exact_func: Optional[Callable] = None,
    title: str = "Time Evolution at Fixed x",
    save_path: Optional[str] = None,
):
    """
    绘制固定 x 位置处 u 随时间变化的曲线（Notebook 版本）。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_val: 固定的 x 值
    :type x_val: float
    :param t_range: t 轴范围 (t_min, t_max)
    :type t_range: tuple
    :param n_points: 采样点数
    :type n_points: int
    :param exact_func: 真实解函数
    :type exact_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    data = prepare_1d_time_slice_data(model, x_val, t_range, n_points, exact_func)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_1d_time_slice(ax, data['t'], data['u_pred'], x_val, data['u_true'], title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_comparison_1d_multiple(
    models,
    labels,
    x_range=(0, 1),
    n_points=200,
    true_func=None,
    title='Comparison',
    save_path=None,
):
    """
    比较多个模型的预测结果（一维）。

    :param models: 模型列表
    :type models: list
    :param labels: 每个模型对应的标签列表
    :type labels: list
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param n_points: 采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    x = torch.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
    x_np = x.numpy().flatten()
    predictions = {}
    for model, label in zip(models, labels):
        model.eval()
        with torch.no_grad():
            predictions[label] = model(x).numpy().flatten()
    u_true = None
    if true_func is not None:
        u_true = true_func(x).numpy().flatten()
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_1d_multiple_models(ax, x_np, predictions, u_true, title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_2d_solution(
    model,
    x_range=(0, 1),
    y_range=(0, 1),
    n_points=100,
    true_func=None,
    title='2D Steady Solution',
    save_path=None,
):
    """
    二维稳态问题：双子图（3D 预测曲面 + 2D 绝对误差云图 + 全局 L2 指标）。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)
    :type y_range: tuple
    :param n_points: 每个维度的采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    data = prepare_2d_data(model, x_range, y_range, n_points, true_func)
    X, Y = data['X'], data['Y']
    Z_pred = data['Z_pred']
    if true_func is not None:
        Z_true = data['Z_true']
        fig = plt.figure(figsize=(15, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        surf = ax1.plot_surface(
            X, Y, Z_pred,
            cmap='viridis',
            edgecolor='none',
            alpha=0.95,
            antialiased=True
        )
        ax1.set_title('3D Predicted Solution $u_{pred}(x, y)$', fontsize=11, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('u')
        ax1.view_init(elev=30, azim=-60)
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
        ax2 = fig.add_subplot(1, 2, 2)
        abs_error = np.abs(Z_pred - Z_true)
        cf = ax2.contourf(X, Y, abs_error, levels=50, cmap='inferno')
        fig.colorbar(cf, ax=ax2, label='Absolute Error $|u_{pred} - u_{true}|$')
        ax2.set_title('2D Absolute Error Distribution', fontsize=11, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_aspect('equal')
        l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
        text_str = (
            "$\mathbf{Global\ Metrics}$\n"
            f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
            f"• Max Abs Error: {abs_error.max():.3e}\n"
            f"• Mean Abs Error: {abs_error.mean():.3e}"
        )
        ax2.text(
            0.04, 0.96, text_str,
            transform=ax2.transAxes,
            fontsize=9.5,
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.88, edgecolor='#cccccc')
        )
    else:
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
        ax1.set_title('3D Predicted Solution', fontsize=11, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('u')
        ax1.view_init(elev=30, azim=-60)
        fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_3d_surface(
    model,
    x_range=(0, 1),
    y_range=(0, 1),
    n_points=100,
    true_func=None,
    title='Predicted Solution Surface',
    save_path=None,
):
    """
    三维曲面图（Notebook 版本）。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)
    :type y_range: tuple
    :param n_points: 每个维度的采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    data = prepare_2d_data(model, x_range, y_range, n_points, true_func)
    if true_func is not None:
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        draw_3d_compare(ax1, ax2, data['X'], data['Y'], data['Z_pred'], data['Z_true'], title)  
        print(f"Max error: {data['error'].max():.2e}, Mean error: {data['error'].mean():.2e}")
    else:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        draw_3d_surface(ax, data['X'], data['Y'], data['Z_pred'], title=title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_2d_transient_interactive(
    model: torch.nn.Module,
    x_range: tuple = (0, 1),
    y_range: tuple = (0, 1),
    t_range: tuple = (0, 1),
    n_points: int = 80,
    exact_func: Optional[Callable] = None,
    title: str = "2D Transient Solution",
    save_path: Optional[str] = None,
):
    """
    二维含时问题的交互式可视化（带时间滑块）。

    - 左图：当前时刻 t 的 PINN 预测值 3D 曲面图
    - 右图：当前时刻 t 的 2D 绝对误差分布云图

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)
    :type y_range: tuple
    :param t_range: t 轴范围 (t_min, t_max)
    :type t_range: tuple
    :param n_points: 每个维度的采样点数
    :type n_points: int
    :param exact_func: 真实解函数
    :type exact_func: Optional[Callable]
    :param title: 图标题（保留备用）
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    t_min, t_max = t_range
    def _update_plot(t_val):
        plt.close('all')
        data = prepare_transient_2d_data(model, x_range, y_range, t_val, n_points, exact_func)
        X, Y = data['X'], data['Y']
        Z_pred = data['Z_pred']
        if exact_func is not None:
            fig = plt.figure(figsize=(12.5, 6))
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            surf = ax1.plot_surface(
                X, Y, Z_pred, 
                cmap='viridis', 
                edgecolor='none', 
                alpha=0.95,
                antialiased=True
            )
            ax1.set_title(f'3D Predicted Solution $u_{{pred}}(x, y, t)$ (t = {t_val:.3f})', fontsize=11)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.08)
            ax2 = fig.add_subplot(1, 2, 2)
            Z_true = data['Z_true']
            error = np.abs(Z_pred - Z_true)
            cf = ax2.contourf(X, Y, error, levels=50, cmap='inferno')
            fig.colorbar(cf, ax=ax2, label='Absolute Error $|u_{pred} - u_{true}|$')
            ax2.set_title(f'2D Absolute Error Distribution (t = {t_val:.3f})', fontsize=11)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_aspect('equal')
            l2_rel_err = np.linalg.norm(Z_pred - Z_true) / (np.linalg.norm(Z_true) + 1e-8)
            max_abs_err = error.max()
            mean_abs_err = error.mean()
            text_str = (
                f"$\mathbf{{Metrics\ at\ t={t_val:.3f}}}$\n"
                f"• Rel $L_2$ Error: {l2_rel_err:.3e} ({l2_rel_err * 100:.2f}%)\n"
                f"• Max Abs Error: {max_abs_err:.3e}\n"
                f"• Mean Abs Error: {mean_abs_err:.3e}"
            )
            ax2.text(
                0.04, 0.96, text_str,
                transform=ax2.transAxes,
                fontsize=9.5,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    alpha=0.88,
                    edgecolor='#cccccc',
                    linewidth=1
                )
            )
        else:
            fig = plt.figure(figsize=(8, 6))
            ax1 = fig.add_subplot(1, 1, 1, projection='3d')
            surf = ax1.plot_surface(X, Y, Z_pred, cmap='viridis', edgecolor='none', alpha=0.95)
            ax1.set_title(f'3D Predicted Solution (t = {t_val:.3f})', fontsize=11)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('u')
            ax1.view_init(elev=30, azim=-60)
            fig.colorbar(surf, ax=ax1, shrink=0.6, aspect=10)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    slider = widgets.FloatSlider(
        value=t_min,
        min=t_min,
        max=t_max,
        step=(t_max - t_min) / 100,
        description='Time (t):',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    interactive_plot = widgets.interactive(_update_plot, t_val=slider)
    display(interactive_plot)
def plot_1d_transient_3d(
    model: Optional[torch.nn.Module] = None,
    x_range: tuple = (0, 1),
    t_range: tuple = (0, 1),
    n_x: int = 60,
    n_t: int = 60,
    exact_func: Optional[Callable] = None,
    title: str = "1D Transient 3D Evolution",
    save_path: Optional[str] = None,
):
    """
    一维含时问题的 3D 时空演化曲面图（Notebook 版本）。
    适用于展示 u(x,t) 在整个时空域上的变化。

    :param model: 训练好的神经网络模型，为 None 时仅绘制解析解
    :type model: Optional[torch.nn.Module]
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param t_range: t 轴范围 (t_min, t_max)
    :type t_range: tuple
    :param n_x: x 方向采样点数
    :type n_x: int
    :param n_t: t 方向采样点数
    :type n_t: int
    :param exact_func: 真实解函数
    :type exact_func: Optional[Callable]
    :param title: 图标题（保留备用）
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    x_min, x_max = x_range
    t_min, t_max = t_range
    x_lin = torch.linspace(x_min, x_max, n_x)
    t_lin = torch.linspace(t_min, t_max, n_t)
    X, T = torch.meshgrid(x_lin, t_lin, indexing='ij')
    pts = torch.cat([X.flatten().reshape(-1, 1), T.flatten().reshape(-1, 1)], dim=1)
    if model is not None:
        model.eval()
        with torch.no_grad():
            Z_pred = model(pts).numpy().reshape(n_x, n_t)
    else:
        Z_pred = None
    if exact_func is not None:
        Z_true = _safe_call_exact_func(exact_func, pts)
        if Z_true is not None:
            Z_true = Z_true.reshape(n_x, n_t)
    else:
        Z_true = None
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    display_Z = Z_pred if Z_pred is not None else Z_true
    if display_Z is not None:
        surf = ax1.plot_surface(X.numpy(), T.numpy(), display_Z, cmap='viridis', edgecolor='none', alpha=0.85)
        ax1.set_title('PINN Prediction' if Z_pred is not None else 'Analytical Solution')
        fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)
    else:
        ax1.text(0.5, 0.5, "No data", ha='center', va='center')
        ax1.set_title('No data')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    ax1.view_init(elev=25, azim=-60)
    if Z_pred is not None and Z_true is not None:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        surf2 = ax2.plot_surface(X.numpy(), T.numpy(), Z_true, cmap='plasma', edgecolor='none', alpha=0.85)
        ax2.set_title('Exact Solution')
        fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        ax2.view_init(elev=25, azim=-60)
        error = np.abs(Z_pred - Z_true)
        print(f"Max error: {error.max():.2e}, Mean error: {error.mean():.2e}")
    elif Z_pred is not None and Z_true is None:
        ax2 = fig.add_subplot(1, 2, 2)
        cp = ax2.contourf(X.numpy(), T.numpy(), Z_pred, levels=30, cmap='viridis')
        ax2.set_title('Contour View')
        fig.colorbar(cp, ax=ax2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
def plot_2d_transient_slices(
    model: Optional[torch.nn.Module] = None,
    x_range: tuple = (0, 1),
    y_range: tuple = (0, 1),
    t_range: tuple = (0, 1),
    n_points: int = 50,
    n_t: int = 4,
    exact_func: Optional[Callable] = None,
    title: str = "2D Transient Slices",
    save_path: Optional[str] = None,
):
    """
    二维含时问题的多时间切片云图（Notebook 版本）。

    :param model: 训练好的神经网络模型，为 None 时仅绘制解析解
    :type model: Optional[torch.nn.Module]
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)
    :type y_range: tuple
    :param t_range: t 轴范围 (t_min, t_max)
    :type t_range: tuple
    :param n_points: 每个空间维度的采样点数
    :type n_points: int
    :param n_t: 时间切片数量
    :type n_t: int
    :param exact_func: 真实解函数
    :type exact_func: Optional[Callable]
    :param title: 图标题
    :type title: str
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    """
    t_values = np.linspace(t_range[0], t_range[1], n_t)
    fig, axes = plt.subplots(2, n_t, figsize=(3*n_t, 6))
    if n_t == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for idx, t_val in enumerate(t_values):
        data = prepare_transient_2d_data(model, x_range, y_range, t_val, n_points, exact_func)
        Z_display = data['Z_pred'] if data['Z_pred'] is not None else data['Z_true']
        if Z_display is not None:
            cp1 = axes[0, idx].contourf(data['X'], data['Y'], Z_display, levels=30, cmap='viridis')
            axes[0, idx].set_title(f't={t_val:.2f}')
            fig.colorbar(cp1, ax=axes[0, idx], shrink=0.6)
        else:
            axes[0, idx].text(0.5, 0.5, "No data", ha='center', va='center')
            axes[0, idx].set_title(f't={t_val:.2f}')
        axes[0, idx].set_xlabel('x')
        axes[0, idx].set_ylabel('y')
        if idx == 0:
            axes[0, idx].set_ylabel('PINN' if data['Z_pred'] is not None else 'Analytical')
        if data['Z_true'] is not None and data['Z_pred'] is not None:
            cp2 = axes[1, idx].contourf(data['X'], data['Y'], data['Z_true'], levels=30, cmap='plasma')
            axes[1, idx].set_xlabel('x')
            axes[1, idx].set_ylabel('y')
            fig.colorbar(cp2, ax=axes[1, idx], shrink=0.6)
            if idx == 0:
                axes[1, idx].set_ylabel('Exact')
        else:
            axes[1, idx].axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# 统一入口
def plot_solution(
    model,
    x_range,
    y_range=None,
    n_points=100,
    true_func=None,
    plot_type='auto',
    title=None,
    save_path=None,
    t_range=None,
):
    """
    统一的解可视化端口，根据参数自动选择一维、二维或三维绘图。

    :param model: 训练好的神经网络模型
    :type model: torch.nn.Module
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)，可选
    :type y_range: Optional[tuple]
    :param n_points: 采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param plot_type: 绘图类型，支持 'auto', '1d', '2d', '3d', 'transient_1d', 'transient_2d', 'time_slice'
    :type plot_type: str
    :param title: 图标题
    :type title: Optional[str]
    :param save_path: 图片保存路径，为 None 时不保存
    :type save_path: Optional[str]
    :param t_range: t 轴范围 (t_min, t_max)，瞬态绘图需要
    :type t_range: Optional[tuple]
    """
    if y_range is None or plot_type == '1d':
        if title is None:
            title = 'Predicted vs True Solution'
        plot_1d_solution(model, x_range, n_points, true_func, title, save_path)
    elif plot_type == '3d':
        if title is None:
            title = 'Predicted Solution Surface'
        plot_3d_surface(model, x_range, y_range, n_points, true_func, title, save_path)
    elif plot_type == 'transient_1d':
        if t_range is None:
            raise ValueError("transient_1d 需要 t_range 参数")
        if title is None:
            title = '1D Transient Solution'
        plot_1d_transient_interactive(model, x_range, t_range, n_points, true_func, title, save_path)
    elif plot_type == 'transient_2d':
        if t_range is None:
            raise ValueError("transient_2d 需要 t_range 参数")
        if title is None:
            title = '2D Transient Solution'
        plot_2d_transient_interactive(model, x_range, y_range, t_range, n_points, true_func, title, save_path)
    elif plot_type == 'time_slice':
        if t_range is None:
            raise ValueError("time_slice 需要 t_range 参数")
        if title is None:
            title = 'Time Evolution at Fixed x'
        x_val = x_range[0] if isinstance(x_range, tuple) and len(x_range) >= 1 else 0.5
        plot_1d_time_slice(model, x_val, t_range, n_points, true_func, title, save_path)
    else:
        if title is None:
            title = 'Predicted Solution'
        plot_2d_solution(model, x_range, y_range, n_points, true_func, title, save_path)
def quick_plot_from_trainer(
    trainer,
    x_range=(0, 1),
    y_range=None,
    n_points=200,
    true_func=None,
    save_dir=None,
    plot_type='auto',
    t_range=None,
):
    """
    从训练器快速生成损失曲线和解的对比图。

    :param trainer: PINNTrainer 实例
    :type trainer: PINNTrainer
    :param x_range: x 轴范围 (x_min, x_max)
    :type x_range: tuple
    :param y_range: y 轴范围 (y_min, y_max)，可选
    :type y_range: Optional[tuple]
    :param n_points: 采样点数
    :type n_points: int
    :param true_func: 真实解函数
    :type true_func: Optional[Callable]
    :param save_dir: 图片保存目录
    :type save_dir: Optional[str]
    :param plot_type: 绘图类型
    :type plot_type: str
    :param t_range: t 轴范围 (t_min, t_max)，瞬态绘图需要
    :type t_range: Optional[tuple]
    """
    history = trainer.get_loss_history()
    if save_dir:
        plot_loss_history(history, save_path=f"{save_dir}/loss.png")
    else:
        plot_loss_history(history)
    model = trainer.model
    plot_solution(
        model,
        x_range,
        y_range,
        n_points,
        true_func,
        save_path=f"{save_dir}/solution.png" if save_dir else None,
        plot_type=plot_type,
        t_range=t_range,
    )

# 独立测试
if __name__ == "__main__":
    print("=" * 70)
    print("可视化模块独立测试（依赖 plotting_core）")
    print("=" * 70)
    # ---------- 模拟数据 ----------
    class MockModel:
        def eval(self): pass
        def __call__(self, x):
            if x.shape[1] == 1:
                return torch.sin(2 * torch.pi * x)
            elif x.shape[1] == 2:
                return torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
            return torch.zeros_like(x[:, 0:1])
    class Mock1DTransientModel:
        """一维含时测试专用"""
        def eval(self): pass
        def __call__(self, x):
            if x.shape[1] == 2:
                return torch.sin(torch.pi * x[:, 0:1]) * torch.exp(-x[:, 1:2])
            return torch.zeros_like(x[:, 0:1])
    class Mock2DTransientModel:
        """二维含时测试专用"""
        def eval(self): pass
        def __call__(self, x):
            if x.shape[1] == 3:
                return torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2]) * torch.exp(-x[:, 2:3])
            return torch.zeros_like(x[:, 0:1])
    def true_func_1d(x):
        if isinstance(x, (float, int)): return np.sin(2 * np.pi * x)
        return torch.sin(2 * torch.pi * x)
    def true_func_2d(x):
        if torch.is_tensor(x): return torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
        else: return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) if isinstance(x, (list, tuple)) else 0.0
    def true_func_1d_transient(x, t):
        return np.sin(np.pi * x) * np.exp(-t)
    def true_func_2d_transient(x, y, t):
        return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t)
    history = {
        'total_loss': [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005],
        'pde_loss': [0.8, 0.4, 0.15, 0.08, 0.04, 0.015, 0.008, 0.004],
        'bc_loss': [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001],
    }
    model = MockModel()
    model_1d_transient = Mock1DTransientModel()
    model_2d_transient = Mock2DTransientModel()
    # ---------- 测试 ----------
    print("\n[测试 1] plot_loss_history")
    plot_loss_history(history)
    print("  ✅")
    print("\n[测试 2] plot_1d_solution")
    plot_1d_solution(model, true_func=true_func_1d)
    print("  ✅")
    print("\n[测试 3] plot_2d_solution")
    plot_2d_solution(model, true_func=true_func_2d)
    print("  ✅")
    print("\n[测试 4] plot_3d_surface")
    plot_3d_surface(model, true_func=true_func_2d)
    print("  ✅")
    print("\n[测试 5] plot_comparison_1d_multiple")
    model2 = MockModel()
    plot_comparison_1d_multiple([model, model2], ["Model 1", "Model 2"], true_func=true_func_1d)
    print("  ✅")
    print("\n[测试 6] plot_1d_time_slice")
    plot_1d_time_slice(model_1d_transient, x_val=0.5, t_range=(0, 1), exact_func=true_func_1d_transient)
    print("  ✅")
    print("\n[测试 7] plot_1d_transient_3d")
    plot_1d_transient_3d(model_1d_transient, x_range=(0, 1), t_range=(0, 1.5), exact_func=true_func_1d_transient)
    print("  ✅")
    print("\n[测试 8] plot_2d_transient_slices")
    plot_2d_transient_slices(model_2d_transient, x_range=(0, 1), y_range=(0, 1), t_range=(0, 0.5), exact_func=true_func_2d_transient)
    print("  ✅")
    print("\n" + "=" * 70)
    print("所有测试完成！")
    print("=" * 70)
