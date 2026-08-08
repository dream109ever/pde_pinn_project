# src/__init__.py
"""
PINN 偏微分方程求解器核心模块。

本模块将 src 包中的核心功能提升到顶层命名空间，对外提供简洁一致的接口。
使用者只需通过 ``from src import *`` 即可获得所有关键组件，
无需关心内部模块的具体组织方式。

:mod:`src` 包提供以下核心功能：
    - PDE 输入解析与损失函数生成 (:mod:`function_factory`)
    - 神经网络自动构建与复杂度分析 (:mod:`network_factory`)
    - 采样点生成 (:mod:`data_utils`)
    - PINN 训练管理 (:mod:`trainer`)
    - 绘图核心逻辑与 Qt/Notebook 可视化 (:mod:`plotting_core`, :mod:`visualization`, :mod:`plotting_qt`)
"""

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
    prepare_1d_data,
    prepare_2d_data,
    prepare_transient_1d_data,
    prepare_transient_2d_data,
    prepare_1d_time_slice_data,
    draw_loss_curve,
    draw_1d_solution,
    draw_1d_transient_slice,
    draw_1d_time_slice,
    draw_1d_multiple_models, 
    draw_2d_contour,
    draw_2d_error,
    draw_2d_with_error, 
    draw_3d_surface,
    draw_3d_compare, 
    draw_1d_transient_3d,
    draw_2d_transient_slices,
    create_subplots,
    draw_complete_solution,
)

from .visualization import (
    plot_loss_history,
    plot_1d_solution,
    plot_1d_transient_interactive,
    plot_1d_time_slice,
    plot_comparison_1d_multiple,
    plot_2d_solution,
    plot_3d_surface,
    plot_2d_transient_interactive,
    plot_1d_transient_3d, 
    plot_2d_transient_slices,
    plot_solution,
    quick_plot_from_trainer,
)

from src.plotting_qt import (
    LossPlotWidget,
    Steady1DPlotWidget,
    Transient1DPlotWidget,
    Steady2DPlotWidget,
    Transient2DPlotWidget
)

__all__ = [
    "InputParser",
    "LossGenerator",
    "AnalyticalSolverHub",
    "solve_pde",

    "build_model",

    "DomainSampler",

    "PINNTrainer",
    
    "prepare_1d_data",
    "prepare_2d_data",
    "prepare_transient_1d_data",
    "prepare_transient_2d_data",
    "prepare_1d_time_slice_data",
    "draw_loss_curve",
    "draw_1d_solution",
    "draw_1d_transient_slice",
    "draw_1d_time_slice",
    "draw_1d_multiple_models",
    "draw_2d_contour",
    "draw_2d_error",
    "draw_2d_with_error",
    "draw_3d_surface",
    "draw_3d_compare",
    "draw_1d_transient_3d",
    "draw_2d_transient_slices",
    "create_subplots",
    "draw_complete_solution",

    "plot_loss_history",
    "plot_1d_solution",
    "plot_1d_transient_interactive",
    "plot_1d_time_slice",
    "plot_comparison_1d_multiple",
    "plot_2d_solution",
    "plot_3d_surface",
    "plot_2d_transient_interactive",
    "plot_1d_transient_3d",
    "plot_2d_transient_slices",
    "plot_solution",
    "quick_plot_from_trainer",

    "LossPlotWidget",
    "Steady1DPlotWidget",
    "Transient1DPlotWidget",
    "Steady2DPlotWidget",
    "Transient2DPlotWidget",
]
