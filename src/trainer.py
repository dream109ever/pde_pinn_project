import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from .function_factory import PDEConfig, BoundaryCondition, get_pde_loss
from .data_utils import DomainSampler
from .network_factory import ComplexityAnalyzer, NetworkConfigGenerator, NetworkFactory

class PINNTrainer:
    """
    物理信息神经网络训练器
    """
    def __init__(self, model, pde_config: PDEConfig, conditions: list[BoundaryCondition],
                 optimizer='adam', lr=1e-3, scheduler='plateau', scheduler_patience=500):
        """
        参数:
            model: 神经网络模型
            pde_config: PDEConfig 实例
            conditions: BoundaryCondition 列表（包含初始和边界条件）
            optimizer: 优化器名称或实例
            lr: 学习率
            scheduler: 学习率调度器类型
            scheduler_patience: 调度器耐心值
        """
        self.model = model
        self.pde_config = pde_config
        self.conditions = conditions
        self.loss_fn = get_pde_loss(pde_config, conditions)
        self._stop_training = False
        if isinstance(optimizer, str):
            opt_cls = {'adam': optim.Adam, 'sgd': optim.SGD}[optimizer.lower()]
            self.optimizer = opt_cls(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        self.scheduler = None
        if scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=scheduler_patience, factor=0.5)
        self.history = {'total_loss': [], 'pde_loss': [], 'bc_loss': []}
    def train_step(self, interior_pts, boundary_pts_list):
        """单步训练：前向、损失、反向、更新"""
        self.optimizer.zero_grad()
        total_loss, pde_loss, bc_loss = self.loss_fn(self.model, interior_pts, boundary_pts_list)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), pde_loss.item(), bc_loss.item()
    def stop(self):
        """请求停止训练"""
        self._stop_training = True
    def train(self, n_epochs, sampler: DomainSampler, batch_size=256, n_boundary_per_edge=50,
              verbose=True, eval_interval=100, early_stop_patience=None, callback=None):
        """
        主训练循环。
        参数:
            n_epochs: 总轮数
            sampler: DomainSampler 实例
            batch_size: 内部点批量大小
            n_boundary_per_edge: 每条边采样点数（用于边界）
            verbose: 是否打印进度
            eval_interval: 打印间隔
            early_stop_patience: 早停耐心值
        """
        best_loss = float('inf')
        patience_counter = 0
        # 1. 采样内部点
        interior_pts = sampler.sample_interior(batch_size)
        # 2. 为每个条件采样对应的点（顺序与 self.conditions 严格一致）
        boundary_pts_list = []
        for bc in self.conditions:
            if bc.is_initial:
                pts = sampler.sample_initial(batch_size // 2)
            else:
                info = bc.get_location_info()
                if info['type'] == 'string':
                    axis = info['axis']
                    value = info['value']
                    pts = sampler.sample_boundary_by_axis(axis, value, n_boundary_per_edge)
                elif info['type'] == 'function':
                    all_pts, _ = sampler.sample_boundary(n_boundary_per_edge * 4)
                    mask = info['func'](all_pts)
                    pts = all_pts[mask]
                    if len(pts) == 0:
                        pts, _ = sampler.sample_boundary(n_boundary_per_edge)
                else:
                    raise ValueError(f"Unsupported location type: {info['type']}")
            boundary_pts_list.append(pts)
        for epoch in (tqdm(range(n_epochs)) if verbose else range(n_epochs)):
            if self._stop_training:
                if verbose:
                    print("Training stopped by user request.")
                break
            # 3. 执行一步训练
            total_loss, pde_loss, bc_loss = self.train_step(interior_pts, boundary_pts_list)
            # 4. 记录历史
            self.history['total_loss'].append(total_loss)
            self.history['pde_loss'].append(pde_loss)
            self.history['bc_loss'].append(bc_loss)
            if callback is not None:
                callback(epoch, total_loss, pde_loss, bc_loss)
            # 5. 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(total_loss)
            # 6. 早停检查
            if early_stop_patience is not None:
                if total_loss < best_loss - 1e-8:
                    best_loss = total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
            # 7. 打印信息
            if verbose and (epoch % eval_interval == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:5d} | Total Loss: {total_loss:.3e} | pde_loss: {pde_loss:.3e} | bc_loss: {bc_loss:.3e}")
        if verbose:
            print("Training finished.")
    def get_loss_history(self):
        return self.history
    def evaluate(self, x_test):
        self.model.eval()
        with torch.no_grad():
            u_pred = self.model(x_test)
        return u_pred
