"""
src/__init__.py

本模块将 src 包中的核心功能提升到顶层命名空间，对外提供简洁一致的接口。
使用者只需通过 `from src import build_residual, build_net, ...` 即可获得所有关键组件，
无需关心内部模块的具体组织方式。

公开接口（即 __all__ 列表中的符号）：
    - build_residual   : 根据配置构建偏微分方程残差函数（来自 function_factory）
    - build_net        : 构建神经网络模型（来自 network_factory）
    - train            : 执行训练主循环（来自 trainer）
    - get_train_data   : 生成内部点和边界点数据（来自 data_utils）
    - plot_loss_curve  : 绘制训练损失曲线（来自 visualization）

此设计便于模块化开发与重构，同时降低外部代码的耦合度。
"""

# 从各个子模块导入需要暴露的公共接口
from .function_factory import PDEConfig, ConditionConfig, BoundaryCondition, parse_expression
from .network_factory import build_model, build_network
from .trainer import PINNTrainer
from .data_utils import DomainSampler
from .visualization import quick_plot_from_trainer
from .gui import run_gui

__all__ = [
    "PDEConfig", "ConditionConfig", "BoundaryCondition", "parse_expression", 
    "build_model", "build_network", 
    "PINNTrainer",
    "DomainSampler",
    "quick_plot_from_trainer",
    "run_gui",
]
