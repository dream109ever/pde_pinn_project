"""
src/__init__.py

本模块将 src 包中的核心功能提升到顶层命名空间，对外提供简洁一致的接口。
使用者只需通过 `from src import *` 
即可获得所有关键组件，无需关心内部模块的具体组织方式。

公开接口（即 __all__ 列表中的符号）：
    - InputParser        : 输入数据解析器（系数、源项、边界条件）
    - LossGenerator      : PINN 损失函数生成器
    - AnalyticalSolverHub: 解析解/基准解生成器
    - solve_pde          : 主求解入口

    - build_model        : 自动构建神经网络（含复杂度分析）
    
    - DomainSampler      : 采样点生成器

    - PINNTrainer        : PINN 训练器

    - quick_plot_from_trainer: 快速绘制训练结果

此设计便于模块化开发与重构，同时降低外部代码的耦合度。
"""

# 从各个子模块导入需要暴露的公共接口
from .function_factory import (
    InputParser,
    LossGenerator,
    AnalyticalSolverHub,
    solve_pde,
)

from .network_factory import build_model

from .data_utils import DomainSampler

from .trainer import PINNTrainer

from .plotting_core import (
    draw_loss_curve,
    draw_1d_solution,
    draw_2d_contour,
    draw_3d_surface,
    draw_1d_transient_3d,
    draw_2d_transient_slices,
    prepare_1d_data,
    prepare_2d_data,
    prepare_transient_1d_data,
    prepare_transient_2d_data,
    draw_1d_time_slice,
    prepare_1d_time_slice_data,
)

from src.plotting_qt import (
    Steady1DPlotWidget,
    Transient1DPlotWidget,
    Steady2DPlotWidget,
    Transient2DPlotWidget
)

from .visualization import (
    plot_loss_history,
    plot_1d_solution,
    plot_2d_solution,
    plot_3d_surface,
    plot_comparison_1d_multiple,
    plot_1d_transient_interactive,
    plot_2d_transient_interactive,
    plot_solution,
    quick_plot_from_trainer,
    plot_1d_time_slice,
)

__all__ = [
    "InputParser",
    "LossGenerator",
    "AnalyticalSolverHub",
    "solve_pde",

    "build_model",

    "DomainSampler",

    "PINNTrainer",
    
    "draw_loss_curve",
    "draw_1d_solution",
    "draw_2d_contour",
    "draw_3d_surface",
    "draw_1d_transient_3d",
    "draw_2d_transient_slices",
    "prepare_1d_data",
    "prepare_2d_data",
    "prepare_transient_1d_data",
    "prepare_transient_2d_data",
    "draw_1d_time_slice",
    "prepare_1d_time_slice_data",

    "Steady1DPlotWidget",
    "Transient1DPlotWidget",
    "Steady2DPlotWidget",
    "Transient2DPlotWidget",

    "plot_loss_history",
    "plot_1d_solution",
    "plot_2d_solution",
    "plot_3d_surface",
    "plot_comparison_1d_multiple",
    "plot_1d_transient_interactive",
    "plot_2d_transient_interactive",
    "plot_solution",
    "quick_plot_from_trainer",
    "plot_1d_time_slice",
]
