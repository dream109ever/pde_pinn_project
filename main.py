import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

# # 创建方程和条件配置
# pde = PDEConfig({'A': 1, 'C': 1, 'G': lambda x,y: torch.sin(torch.pi*x)*torch.sin(torch.pi*y)}, has_t=False)
# bc1 = BoundaryCondition('x=0', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}))
# bc2 = BoundaryCondition('x=1', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}))
# bc3 = BoundaryCondition('y=0', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}))
# bc4 = BoundaryCondition('y=1', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}))
# conditions = [bc1, bc2, bc3, bc4]

# model = build_model(pde, conditions, base_config={'input_dim': 2, 'output_dim': 1, 'hidden_dims' : [24, 24]})
# # 创建训练器
# trainer = PINNTrainer(model, pde, conditions, lr=1e-3)
# # 定义采样器
# sampler = DomainSampler(x_range=(0,1), y_range=(0,1))
# # 训练

# trainer.train(100, sampler, batch_size=5000, n_boundary_per_edge=200)
# trainer.train(200, sampler, batch_size=2000, n_boundary_per_edge=200)
# trainer.train(200, sampler, batch_size=2000, n_boundary_per_edge=200)
# def true_func(points):
#     x = points[:, 0:1]
#     y = points[:, 1:2]
#     return -torch.sin(torch.pi * x) * torch.sin(torch.pi * y) / (2 * torch.pi**2)

# quick_plot_from_trainer(trainer, x_range=(0, 1), y_range=(0, 1), n_points=300, true_func=true_func, plot_type='3d')

run_gui()