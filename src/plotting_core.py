"""
plotting_core.py - 纯绘图逻辑模块

本模块包含所有绘图函数的"核心逻辑"，不依赖任何显示后端（如 plt.show、ipywidgets、Qt 等）。
所有函数都接受 matplotlib.axes.Axes 作为第一个参数，在给定的坐标系上绘制图形。

适用场景：
    - Jupyter Notebook: 创建 Figure 和 Axes，调用本模块函数，然后 plt.show()
    - Qt 界面: 将 FigureCanvasQTAgg 的 Axes 传入，调用本模块函数，然后 canvas.draw()
    - 保存图片: 创建 Figure 和 Axes，调用本模块函数，然后 fig.savefig()

这样可以确保绘图逻辑在 Notebook 和 Qt 中完全一致，只需改变显示方式。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Dict, List, Tuple, Callable

# ============================================================================
# 0. 辅助工具函数
# ============================================================================
def _safe_call_exact_func(exact_func, pts):
    """
    安全调用 exact_func，自动适配张量/标量/numpy 数组。
    
    exact_func 可能来自：
        - sympy.lambdify: 接受 numpy 数组 (N, dim)，返回 numpy 数组 (N,)
        - scipy 插值: 接受 numpy 数组 (N, dim)，返回 numpy 数组 (N,)
        - 自定义函数: 接受 torch 张量或标量
    
    参数:
        exact_func: 可调用对象
        pts: torch.Tensor (N, dim) 点集
    
    返回:
        numpy.ndarray (N,) 或 (N, 1)
    """
    if exact_func is None: return None
    def _to_float_array(arr):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            if arr.dtype == object:
                try:
                    return np.array([float(v) if hasattr(v, 'evalf') else float(v) for v in arr.flatten()])
                except Exception:
                    try:
                        return np.array([float(v.evalf()) if hasattr(v, 'evalf') else float(v) for v in arr.flatten()])
                    except Exception:
                        return None
            return arr
        if torch.is_tensor(arr):
            return arr.detach().cpu().numpy().flatten()
        try:
            return np.array([float(arr)])
        except Exception:
            return None
    try:
        result = exact_func(pts)
        converted = _to_float_array(result)
        if converted is not None:
            return converted.flatten()
    except Exception:
        pass
    try:
        pts_np = pts.detach().cpu().numpy()
        result = exact_func(pts_np)
        converted = _to_float_array(result)
        if converted is not None:
            return converted.flatten()
    except Exception:
        pass
    n = pts.shape[0]
    results = []
    for i in range(n):
        pt_i = pts[i].detach().cpu().numpy()
        try:
            val = exact_func(pt_i)
        except Exception:
            if pt_i.ndim == 1 and len(pt_i) == 1:
                val = exact_func(pt_i[0])
            else:
                val = exact_func(*pt_i)
        try:
            if hasattr(val, 'evalf'):
                results.append(float(val.evalf()))
            else:
                results.append(float(val))
        except Exception:
            results.append(np.nan)
    return np.asarray(results).flatten()
def _to_numpy(x):
    """将 torch.Tensor 或 numpy 数组转为 numpy 数组"""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.array(x)
    return x
def _get_label_dim(has_t: bool, dim: int) -> str:
    """根据维度生成标签"""
    if dim == 1 and not has_t:
        return 'x'
    elif dim == 1 and has_t:
        return 'x, t'
    elif dim == 2 and not has_t:
        return 'x, y'
    elif dim == 2 and has_t:
        return 'x, y, t'
    return ''
# ============================================================================
# 1. 数据准备函数
# ============================================================================
def prepare_1d_data(
    model: Optional[torch.nn.Module],
    x_range: Tuple[float, float],
    n_points: int = 200,
    exact_func: Optional[Callable] = None,
) -> Dict:
    """准备一维稳态问题的绘图数据。"""
    x_min, x_max = x_range
    x_test = torch.linspace(x_min, x_max, n_points).reshape(-1, 1)
    x_np = x_test.numpy().flatten()
    u_pred = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            u_pred = model(x_test).numpy().flatten()    
    result = {'x': x_np, 'u_pred': u_pred}
    u_true = None
    if exact_func is not None:
        u_true = _safe_call_exact_func(exact_func, x_test)
        if u_true is not None:
            u_true = np.asarray(u_true).flatten()
            if len(u_true) != len(x_np):
                if len(u_true) == 1:
                    u_true = np.full_like(x_np, u_true[0])
                else:
                    u_true = np.array([_safe_call_exact_func(exact_func, x_test[i:i+1]) for i in range(len(x_np))]).flatten()
    result['u_true'] = u_true
    if u_pred is not None and u_true is not None:
        result['error'] = np.abs(u_pred - u_true)
    else:
        result['error'] = None
    return result
def prepare_2d_data(
    model: Optional[torch.nn.Module],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int = 80,
    exact_func: Optional[Callable] = None,
) -> Dict:
    """准备二维稳态问题的绘图数据。"""
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_lin = torch.linspace(x_min, x_max, n_points)
    y_lin = torch.linspace(y_min, y_max, n_points)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    pts = torch.cat([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1, 1)], dim=1)
    # 1. PINN 网络预测
    Z_pred = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            Z_pred = model(pts).numpy().reshape(n_points, n_points)  
    result = {
        'X': X.numpy(),
        'Y': Y.numpy(),
        'Z_pred': Z_pred,
    }
    # 2. 精确解评估
    Z_true = None
    if exact_func is not None:
        Z_true = _safe_call_exact_func(exact_func, pts)
        if Z_true is not None:
            Z_true = Z_true.reshape(n_points, n_points)    
    result['Z_true'] = Z_true
    # 3. 误差计算
    if Z_pred is not None and Z_true is not None:
        result['error'] = np.abs(Z_pred - Z_true)
    else:
        result['error'] = None
    return result
def prepare_transient_1d_data(
    model: Optional[torch.nn.Module],
    x_range: Tuple[float, float],
    t_val: float,
    n_points: int = 200,
    exact_func: Optional[Callable] = None,
) -> Dict:
    """
    准备一维含时问题的绘图数据。
    
    返回:
        'x': np.ndarray,
        'u_pred': np.ndarray,
        'u_true': np.ndarray or None,
        'error': np.ndarray or None,
    """
    x_min, x_max = x_range
    x_test = torch.linspace(x_min, x_max, n_points).reshape(-1, 1)
    t_test = torch.full_like(x_test, t_val)
    pts = torch.cat([x_test, t_test], dim=1)  # 拼接为 (N, 2) 的输入 tensor [x, t]
    x_np = x_test.numpy().flatten()
    # 1. PINN 网络预测
    u_pred = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            u_pred = model(pts).numpy().flatten()     
    result = {
        'x': x_np,
        't_val': t_val,
        'u_pred': u_pred,
    }
    # 2. 精确/解析解评估
    u_true = None
    if exact_func is not None:
        u_true = _safe_call_exact_func(exact_func, pts)
        if u_true is not None:
            u_true = u_true.flatten()      
    result['u_true'] = u_true
    # 3. 误差计算（两者皆存在时才计算）
    if u_pred is not None and u_true is not None:
        result['error'] = np.abs(u_pred - u_true)
    else:
        result['error'] = None
    return result
def prepare_transient_2d_data(
    model: Optional[torch.nn.Module],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    t_val: float,
    n_points: int = 80,
    exact_func: Optional[Callable] = None,
) -> Dict:
    """
    准备二维含时问题的绘图数据。
    
    返回:
        'X': np.ndarray,    # 网格 X
        'Y': np.ndarray,    # 网格 Y
        'Z_pred': np.ndarray,
        'Z_true': np.ndarray or None,
        'error': np.ndarray or None,
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_lin = torch.linspace(x_min, x_max, n_points)
    y_lin = torch.linspace(y_min, y_max, n_points)
    X, Y = torch.meshgrid(x_lin, y_lin, indexing='ij')
    t_tensor = torch.full((n_points * n_points, 1), t_val)
    pts = torch.cat([X.flatten().reshape(-1, 1), Y.flatten().reshape(-1, 1), t_tensor], dim=1) # (N*N, 3) [x, y, t]
    # 1. PINN 网络预测
    Z_pred = None
    if model is not None:
        model.eval()
        with torch.no_grad():
            Z_pred = model(pts).numpy().reshape(n_points, n_points)     
    result = {
        'X': X.numpy(),
        'Y': Y.numpy(),
        't_val': t_val,
        'Z_pred': Z_pred,
    }
    # 2. 精确/解析解评估
    Z_true = None
    if exact_func is not None:
        Z_true = _safe_call_exact_func(exact_func, pts)
        if Z_true is not None:
            Z_true = Z_true.reshape(n_points, n_points)      
    result['Z_true'] = Z_true
    # 3. 误差计算
    if Z_pred is not None and Z_true is not None:
        result['error'] = np.abs(Z_pred - Z_true)
    else:
        result['error'] = None
    return result
def prepare_1d_time_slice_data(
    model: torch.nn.Module,
    x_val: float,
    t_range: Tuple[float, float],
    n_points: int = 200,
    exact_func: Optional[Callable] = None,
) -> Dict:
    """准备固定 x 处 u 随时间变化的数据"""
    t_min, t_max = t_range
    t_test = torch.linspace(t_min, t_max, n_points).reshape(-1, 1)
    x_vals = torch.full((n_points, 1), x_val)
    xt = torch.cat([x_vals, t_test], dim=1)
    model.eval()
    with torch.no_grad():
        u_pred = model(xt).numpy().flatten()
    t_np = t_test.numpy().flatten()
    result = {'t': t_np, 'u_pred': u_pred}
    if exact_func is not None:
        u_true = _safe_call_exact_func(exact_func, xt)
        if u_true is not None:
            result['u_true'] = u_true
            result['error'] = np.abs(u_pred - u_true)
        else:
            result['u_true'] = None
            result['error'] = None
    else:
        result['u_true'] = None
        result['error'] = None
    return result
# ============================================================================
# 2. 绘图函数
# ============================================================================
def draw_loss_curve(
    ax: plt.Axes,
    history: Dict[str, List[float]],
    log_scale: bool = True,
    title: str = "Training Loss History",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    colors: Optional[Dict[str, str]] = None,
):
    """
    在给定的 Axes 上绘制损失曲线。
    
    参数:
        ax: matplotlib Axes
        history: 包含 'total_loss', 'pde_loss', 'bc_loss' 的字典
        log_scale: 是否使用对数 Y 轴
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        colors: 自定义颜色字典，如 {'total_loss': 'b', 'pde_loss': 'r', 'bc_loss': 'g'}
    """
    colors = colors or {
        'total_loss': 'b',
        'pde_loss': 'r',
        'bc_loss': 'g',
        'ic_loss': 'orange',
    }
    epochs = range(1, len(history['total_loss']) + 1)
    # 绘制总损失
    ax.plot(epochs, history['total_loss'], color=colors['total_loss'], label='Total Loss', linewidth=2)
    # 绘制 PDE 损失
    if 'pde_loss' in history and history['pde_loss']:
        ax.plot(epochs, history['pde_loss'], color=colors['pde_loss'], label='PDE Loss', linestyle='--', linewidth=1.5)
    # 绘制 BC 损失
    if 'bc_loss' in history and history['bc_loss']:
        ax.plot(epochs, history['bc_loss'], color=colors['bc_loss'], label='BC Loss', linestyle='-.', linewidth=1.5)
    # 绘制 IC 损失
    if 'ic_loss' in history and history['ic_loss']:
        ax.plot(epochs, history['ic_loss'], color=colors.get('ic_loss', 'orange'), label='IC Loss', linestyle=':', linewidth=1.5)
    if log_scale:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
def draw_1d_solution(
    ax: plt.Axes,
    x: np.ndarray,
    u_pred: Optional[np.ndarray],
    u_true: Optional[np.ndarray] = None,
    title: str = "1D Solution",
    xlabel: str = "x",
    ylabel: str = "u(x)",
    show_error: bool = True,
):
    """
    在给定的 Axes 上绘制一维解（预测 vs 真实）。
    
    参数:
        ax: matplotlib Axes
        x: x 坐标数组 (N,)
        u_pred: 预测值数组 (N,)
        u_true: 真实解数组 (N,)，可选
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        show_error: 是否在图上显示误差
    """
    # 1. 绘制 PINN 预测
    if u_pred is not None:
        ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
    # 2. 绘制解析解（无 PINN 时用绿实线展示，有 PINN 时用红虚线对比）
    if u_true is not None:
        style = 'r--' if u_pred is not None else 'g-'
        label_text = 'Exact Solution' if u_pred is not None else 'Analytical Solution'
        ax.plot(x, u_true, style, label=label_text, linewidth=2)
    # 3. 计算误差框（仅在二者均有效时计算）
    if u_pred is not None and u_true is not None and show_error:
        error = np.abs(u_pred - u_true)
        ax.text(0.05, 0.95, f"Max error: {error.max():.2e}\nMean error: {error.mean():.2e}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
def draw_1d_transient_slice(
    ax: plt.Axes,
    x: np.ndarray,
    u_pred: Optional[np.ndarray],
    t_val: float,
    u_true: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    xlabel: str = "x",
    ylabel: str = "u(x, t)",
    show_error: bool = True,
):
    """
    在给定的 Axes 上绘制一维含时问题的切片。
    
    参数:
        ax: matplotlib Axes
        x: x 坐标数组 (N,)
        u_pred: 预测值数组 (N,)
        t_val: 当前时间值
        u_true: 真实解数组 (N,)，可选
        title: 图标题
        xlabel: X 轴标签
        show_error: 是否显示误差
    """
    if title is None:
        title = f"1D Transient Solution (t = {t_val:.2f})"
    # 1. 绘制 PINN 预测
    if u_pred is not None:
        ax.plot(x, u_pred, 'b-', label='PINN Prediction', linewidth=2)
    # 2. 绘制解析解（无预测值时采用绿实线显示，作为唯一数据来源）
    if u_true is not None:
        style = 'r--' if u_pred is not None else 'g-'
        label_text = 'Exact Solution' if u_pred is not None else 'Analytical Solution'
        ax.plot(x, u_true, style, label=label_text, linewidth=2)
    # 3. 计算并展示误差框
    if u_pred is not None and u_true is not None and show_error:
        error = np.abs(u_pred - u_true)
        ax.text(
            0.05, 0.95,
            f"t = {t_val:.2f}\nMax error: {error.max():.2e}\nMean error: {error.mean():.2e}",
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'u(x, t={t_val:.3f})')
    ax.set_title(f'{title} (t={t_val:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
def draw_1d_time_slice(
    ax: plt.Axes,
    t: np.ndarray,
    u_pred: Optional[np.ndarray],
    x_val: float,
    u_true: Optional[np.ndarray] = None,
    title: str = "Time Evolution",
    xlabel: str = "t",
    ylabel: str = "u(x=const, t)",
    show_error: bool = True,
):
    """在给定的 Axes 上绘制固定 x 位置处 u 随时间变化的曲线。"""
    if u_pred is not None:
        ax.plot(t, u_pred, 'b-', label=f'PINN (x={x_val:.2f})', linewidth=2)
    if u_true is not None:
        style = 'r--' if u_pred is not None else 'g-'
        label = 'Exact' if u_pred is not None else 'Analytical Solution'
        ax.plot(t, u_true, style, label=label, linewidth=2)
        if u_pred is not None and show_error:
            error = np.abs(u_pred - u_true)
            ax.text(0.05, 0.95, f"Max error: {error.max():.2e}\nMean error: {error.mean():.2e}",
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (x={x_val:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
def draw_1d_multiple_models(
    ax: plt.Axes,
    x: np.ndarray,
    predictions: Dict[str, np.ndarray],
    u_true: Optional[np.ndarray] = None,
    title: str = "Model Comparison",
    xlabel: str = "x",
    ylabel: str = "u(x)",
):
    """
    在给定的 Axes 上绘制多个模型的预测对比。
    
    参数:
        ax: matplotlib Axes
        x: x 坐标数组 (N,)
        predictions: 字典 {label: u_pred}
        u_true: 真实解，可选
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for idx, (label, u_pred) in enumerate(predictions.items()):
        ax.plot(x, u_pred, '--', color=colors[idx], label=label, linewidth=1.5)
    
    if u_true is not None:
        ax.plot(x, u_true, 'k-', label='Exact', linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
def draw_2d_contour(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z: Optional[np.ndarray],
    levels: int = 50,
    cmap: str = 'viridis',
    title: str = "2D Solution",
    xlabel: str = "x",
    ylabel: str = "y",
    colorbar_label: str = "u(x,y)",
    show_colorbar: bool = True,
    fig: Optional[plt.Figure] = None,
):
    """
    在给定的 Axes 上绘制二维伪彩色图（等高线填充）。
    
    参数:
        ax: matplotlib Axes
        X, Y: 网格坐标 (N, N)
        Z: 解值 (N, N)
        levels: 等高线层数
        cmap: 颜色映射
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        colorbar_label: 颜色条标签
        show_colorbar: 是否显示颜色条
        fig: 用于添加颜色条的 Figure（如果为 None，尝试从 ax 获取）
    """
    if Z is None:
        ax.text(0.5, 0.5, "无有效图像数据", ha='center', va='center')
        ax.set_title(title)
        return ax
    cp = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    if show_colorbar:
        if fig is None:
            fig = ax.figure
        if fig is not None:
            fig.colorbar(cp, ax=ax, label=colorbar_label)
    return ax
def draw_2d_error(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    error: np.ndarray,
    levels: int = 50,
    cmap: str = 'hot',
    title: str = "Absolute Error",
    xlabel: str = "x",
    ylabel: str = "y",
    show_colorbar: bool = True,
    fig: Optional[plt.Figure] = None,
):
    """
    在给定的 Axes 上绘制误差图。
    """
    cp = ax.contourf(X, Y, error, levels=levels, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='box')
    if show_colorbar:
        if fig is None:
            fig = ax.figure
        if fig is not None:
            fig.colorbar(cp, ax=ax, label='Absolute Error')
    return ax
def draw_2d_with_error(
    ax_pred: plt.Axes,
    ax_error: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    Z_pred: np.ndarray,
    Z_true: np.ndarray,
    title: str = "2D Solution",
    levels: int = 50,
    cmap: str = 'viridis',
):
    """
    在给定的两个 Axes 上分别绘制预测解和误差图。
    
    参数:
        ax_pred: 用于绘制预测解的 Axes
        ax_error: 用于绘制误差的 Axes
        X, Y: 网格坐标
        Z_pred: 预测解
        Z_true: 真实解
        title: 图标题
        levels: 等高线层数
        cmap: 颜色映射
    """
    draw_2d_contour(ax_pred, X, Y, Z_pred, levels, cmap, title)
    error = np.abs(Z_pred - Z_true)
    ax_error.text(0.05, 0.95, f"Max error: {error.max():.2e}\nMean error: {error.mean():.2e}",
                  transform=ax_error.transAxes, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    draw_2d_error(ax_error, X, Y, error, levels, cmap='hot', title="Absolute Error")
    return ax_pred, ax_error
def draw_3d_surface(
    ax: Axes3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    cmap: str = 'viridis',
    title: str = "3D Surface",
    xlabel: str = "x",
    ylabel: str = "y",
    zlabel: str = "u(x,y)",
    alpha: float = 0.9,
    show_colorbar: bool = True,
    fig: Optional[plt.Figure] = None,
    elev: float = 25,
    azim: float = -60,
):
    """
    在给定的 3D Axes 上绘制曲面图。
    
    参数:
        ax: 3D Axes
        X, Y: 网格坐标
        Z: 解值
        cmap: 颜色映射
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        zlabel: Z 轴标签
        alpha: 透明度
        show_colorbar: 是否显示颜色条
        fig: 用于添加颜色条的 Figure
        elev, azim: 视角角度
    """
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none', alpha=alpha)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    if show_colorbar:
        if fig is None:
            fig = ax.figure
        if fig is not None:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    return ax
def draw_3d_compare(
    ax_pred: Axes3D,
    ax_true: Axes3D,
    X: np.ndarray,
    Y: np.ndarray,
    Z_pred: np.ndarray,
    Z_true: np.ndarray,
    title: str = "3D Comparison",
    cmap_pred: str = 'viridis',
    cmap_true: str = 'plasma',
):
    """
    在给定的两个 3D Axes 上分别绘制预测和真实解。
    """
    draw_3d_surface(ax_pred, X, Y, Z_pred, cmap_pred, f"{title} - PINN")
    draw_3d_surface(ax_true, X, Y, Z_true, cmap_true, f"{title} - Exact")
    return ax_pred, ax_true
def draw_1d_transient_3d(
    ax: Axes3D,
    X: np.ndarray,   # x 网格 (nx, nt)
    T: np.ndarray,   # t 网格 (nx, nt)
    Z: np.ndarray,   # u 值 (nx, nt)
    cmap: str = 'viridis',
    title: str = "1D Transient Evolution",
    xlabel: str = "x",
    ylabel: str = "t",
    zlabel: str = "u(x,t)",
    elev: float = 25,
    azim: float = -60,
):
    """
    在给定的 3D Axes 上绘制一维含时问题的时空演化曲面。
    """
    surf = ax.plot_surface(X, T, Z, cmap=cmap, edgecolor='none', alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    return ax
def draw_2d_transient_slices(
    axes: List[plt.Axes],
    X: np.ndarray,
    Y: np.ndarray,
    slices: List[np.ndarray],
    t_values: List[float],
    cmap: str = 'viridis',
    title: str = "2D Transient Slices",
):
    """
    在给定的 Axes 列表上绘制多个时间切片。
    
    参数:
        axes: Axes 列表，长度应等于 len(t_values)
        X, Y: 网格坐标
        slices: 每个时间点的 Z 值列表
        t_values: 对应的时间值列表
        cmap: 颜色映射
        title: 总标题
    """
    if len(axes) != len(t_values):
        raise ValueError("axes 和 t_values 长度必须相同")
    for ax, Z, t_val in zip(axes, slices, t_values):
        if Z is not None:
            cp = ax.contourf(X, Y, Z, levels=30, cmap=cmap)
            ax.set_title(f't={t_val:.2f}')
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(f't={t_val:.2f} (missing)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='box')
    return axes
# ============================================================================
# 3. 子图辅助与整合
# ============================================================================
def create_subplots(
    nrows: int,
    ncols: int,
    figsize: Tuple[float, float] = None,
    sharex: bool = False,
    sharey: bool = False,
    subplot_kw: Optional[Dict] = None,
    **fig_kwargs
):
    """
    创建 Figure 和 Axes 网格（便于统一管理子图）。
    
    返回:
        (fig, axes): Figure 和 Axes 数组
    """
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    subplot_kw = subplot_kw or {}
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey, subplot_kw=subplot_kw, **fig_kwargs)
    # 确保 axes 是二维数组，便于统一索引
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.array(axes).reshape(nrows, ncols)
    return fig, axes
def draw_complete_solution(
    ax: plt.Axes,
    model: Optional[torch.nn.Module],
    problem_config: Dict,
    exact_func: Optional[Callable] = None,
    plot_type: str = 'auto',
    **kwargs
):
    """
    根据问题配置自动绘制完整解。
    
    参数:
        ax: matplotlib Axes
        model: 神经网络模型
        problem_config: 包含 dimension, has_t, domain 等
        exact_func: 真实解函数
        plot_type: 'auto', '1d', '2d', '3d'
        `**kwargs`: 传递给具体绘图函数的参数
    """
    dim = problem_config.get('dimension', 1)
    has_t = problem_config.get('has_t', False)
    domain = problem_config.get('domain', {})
    if dim == 1 and not has_t:
        # 一维稳态
        x_range = domain.get('x', (0, 1))
        data = prepare_1d_data(model, x_range,  n_points=kwargs.get('n_points', 200), exact_func=exact_func)
        draw_1d_solution(ax, data['x'], data['u_pred'], data['u_true'], title=kwargs.get('title', '1D Solution'))
    elif dim == 1 and has_t:
        # 一维含时
        t_val = kwargs.get('t_val', 0.5)
        x_range = domain.get('x', (0, 1))
        data = prepare_transient_1d_data(model, x_range, t_val, n_points=kwargs.get('n_points', 200), exact_func=exact_func)
        draw_1d_transient_slice(ax, data['x'], data['u_pred'], t_val, data['u_true'], title=kwargs.get('title', '1D Transient'))
    elif dim == 2 and not has_t:
        # 二维稳态
        x_range = domain.get('x', (0, 1))
        y_range = domain.get('y', (0, 1))
        data = prepare_2d_data(model, x_range, y_range, n_points=kwargs.get('n_points', 80), exact_func=exact_func)
        # 如果 Z_pred 为 None，自动取 Z_true 进行绘图
        display_Z = data['Z_pred'] if data['Z_pred'] is not None else data['Z_true']
        draw_2d_contour(ax, data['X'], data['Y'], display_Z, title=kwargs.get('title', '2D Solution'))
    else:
        raise ValueError(f"不支持的配置: dim={dim}, has_t={has_t}")
    return ax
