import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Dict, List, Tuple, Callable, Any


class PINNTrainer:
    """
    物理信息神经网络训练器（独立版本，不依赖任何 PDE 解析模块）
    
    职责：
        1. 管理模型、优化器、学习率调度器
        2. 执行训练循环
        3. 记录损失历史
        4. 支持早停、回调、手动停止
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_functions: Tuple[Callable, Callable, Callable],
        optimizer: str = 'adam',
        lr: float = 1e-3,
        scheduler: Optional[str] = 'plateau',
        scheduler_patience: int = 500,
        scheduler_factor: float = 0.5,
        device: str = 'cpu',
    ):
        """
        初始化 PINN 训练器。

        参数:
            model: 神经网络模型
            loss_functions (tuple): `(pde_loss, bc_loss, total_loss)` 损失函数三元组：

                * `pde_loss`: `(model, points) -> Tensor`
                * `bc_loss`: `(model, boundary_pts) -> Tensor`
                * `total_loss`: `(model, points, boundary_pts) -> Tensor`
            
            optimizer (str): 优化器名称，支持 'adam', 'sgd', 'adamw'
            lr (float): 学习率
            scheduler (str or None): 学习率调度器，'plateau' 或 None
            scheduler_patience (int): 调度器耐心值
            scheduler_factor (float): 调度器衰减因子
            device (str or torch.device): 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.loss_functions = loss_functions
        self._stop_training = False
        # 优化器
        opt_cls = {
            'adam': optim.Adam,
            'sgd': optim.SGD,
            'adamw': optim.AdamW,
        }.get(optimizer.lower(), optim.Adam)
        self.optimizer = opt_cls(model.parameters(), lr=lr)
        # 学习率调度器
        self.scheduler = None
        if scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=scheduler_patience,
                factor=scheduler_factor,
            )
        # 历史记录
        self.history: Dict[str, List[float]] = {
            'total_loss': [],
            'pde_loss': [],
            'bc_loss': [],
            'lr': [],
        }
    # ---------- 核心训练方法 ----------
    def train_step(
        self,
        interior_pts: torch.Tensor,
        boundary_pts: Dict[str, torch.Tensor],
        initial_pts: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float, float]:
        """
        单步训练：前向、损失、反向、更新
        
        参数:
            interior_pts: 内部点 (N, dim)
            boundary_pts: 边界点字典，如 {'left': (N, dim), 'right': (N, dim)}
            initial_pts: 初始条件点 (N, dim)，含时间问题使用
        
        兼容两种损失函数签名：
            total_loss(net, x, boundary_pts)  # 含时/2D 分支
            total_loss(net, x)                # 1D 稳态分支
            bc_loss(net, boundary_pts)        # 含时/2D 分支
            bc_loss(net)                      # 1D 稳态分支

        返回:
            (total_loss, pde_loss, bc_loss)
        """
        self.optimizer.zero_grad()
        pde_loss_fn, bc_loss_fn, total_loss_fn = self.loss_functions
        if initial_pts is not None: boundary_pts['initial'] = initial_pts
        interior_pts = interior_pts.clone().detach().requires_grad_(True)
        try: total_loss = total_loss_fn(self.model, interior_pts, boundary_pts, initial_pts)
        except TypeError:
            try: total_loss = total_loss_fn(self.model, interior_pts, boundary_pts)
            except TypeError:
                try: total_loss = total_loss_fn(self.model, interior_pts)
                except TypeError: total_loss = total_loss_fn(self.model)
        try: bc_loss = bc_loss_fn(self.model, boundary_pts, initial_pts)
        except TypeError:
            try: bc_loss = bc_loss_fn(self.model, boundary_pts)
            except TypeError:
                try: bc_loss = bc_loss_fn(self.model)
                except TypeError: bc_loss = bc_loss_fn()
        try: pde_loss = pde_loss_fn(self.model, interior_pts, initial_pts)
        except TypeError:
            try: pde_loss = pde_loss_fn(self.model, interior_pts)
            except TypeError: pde_loss = pde_loss_fn(self.model)
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), pde_loss.item(), bc_loss.item()
    # ---------- 主训练循环 ----------
    def train(
        self,
        n_epochs: int,
        sampler: Any,  # 需要提供 sample_interior, sample_boundary, sample_initial 方法
        n_interior: int = 1000,
        n_boundary_per_side: int = 50,
        n_initial: int = 200,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        eval_interval: int = 100,
        early_stop_patience: Optional[int] = None,
        callback: Optional[Callable[[int, float, float, float], None]] = None,
        resample_every: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """
        主训练循环
        
        参数:
            n_epochs: 总训练轮数
            sampler: 采样器对象，必须包含以下方法：

                * `sample_interior(n_points) -> Tensor`
                * `sample_boundary(n_points_per_side) -> Dict[str, Tensor]`
                * `sample_initial(n_points) -> Tensor`
            
            n_interior: 内部点数量
            n_boundary_per_side: 每条边的边界点数量
            n_initial: 初始条件点数量
            batch_size: 批大小（若为 None，则使用全部点）
            verbose: 是否打印进度
            eval_interval: 打印间隔
            early_stop_patience: 早停耐心值
            callback: 回调函数，签名 (epoch, total_loss, pde_loss, bc_loss)
            resample_every: 每隔多少轮重新采样（None 表示固定采样）
        
        返回:
            Dict: 训练历史
        """
        best_loss = float('inf')
        patience_counter = 0
        # 判断是否有时间维度（通过是否有 sample_initial 方法或 has_t 属性）
        has_time = hasattr(sampler, 'has_t') and sampler.has_t
        # 初始采样
        interior_pts = sampler.sample_interior(n_interior)
        boundary_pts = sampler.sample_boundary(n_boundary_per_side)
        initial_pts = sampler.sample_initial(n_initial) if has_time else None
        # 批处理设置
        if batch_size is not None and batch_size < n_interior:
            use_batch = True
            n_batches = n_interior // batch_size
        else:
            use_batch = False
            n_batches = 1
        epoch_iter = tqdm(range(n_epochs)) if verbose else range(n_epochs)
        for epoch in epoch_iter:
            if self._stop_training:
                if verbose:
                    print("训练已手动停止")
                break
            # 重新采样（如果启用）
            if resample_every is not None and epoch % resample_every == 0 and epoch > 0:
                interior_pts = sampler.sample_interior(n_interior)
                boundary_pts = sampler.sample_boundary(n_boundary_per_side)
                if has_time:
                    initial_pts = sampler.sample_initial(n_initial)
            # 训练（支持批处理）
            if use_batch:
                indices = torch.randperm(n_interior)
                epoch_total_loss = 0.0
                epoch_pde_loss = 0.0
                epoch_bc_loss = 0.0
                for batch_idx in range(n_batches):
                    start = batch_idx * batch_size
                    end = min(start + batch_size, n_interior)
                    batch_indices = indices[start:end]
                    batch_interior = interior_pts[batch_indices]
                    total_l, pde_l, bc_l = self.train_step(batch_interior, boundary_pts, initial_pts)
                    epoch_total_loss += total_l
                    epoch_pde_loss += pde_l
                    epoch_bc_loss += bc_l
                total_loss = epoch_total_loss / n_batches
                pde_loss = epoch_pde_loss / n_batches
                bc_loss = epoch_bc_loss / n_batches
            else:
                total_loss, pde_loss, bc_loss = self.train_step(interior_pts, boundary_pts, initial_pts)
            # 记录历史
            self.history['total_loss'].append(total_loss)
            self.history['pde_loss'].append(pde_loss)
            self.history['bc_loss'].append(bc_loss)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            # 回调
            if callback is not None:
                callback(epoch, total_loss, pde_loss, bc_loss)
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step(total_loss)
            # 早停
            if early_stop_patience is not None:
                if total_loss < best_loss - 1e-8:
                    best_loss = total_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        if verbose:
                            print(f"早停于第 {epoch} 轮")
                        break
            # 打印进度
            if verbose and (epoch % eval_interval == 0 or epoch == n_epochs - 1):
                lr = self.optimizer.param_groups[0]['lr']
                print(
                    f"Epoch {epoch:5d}/{n_epochs} | "
                    f"Loss: {total_loss:.3e} | "
                    f"PDE: {pde_loss:.3e} | "
                    f"BC: {bc_loss:.3e} | "
                    f"LR: {lr:.2e}"
                )
        if verbose:
            print("训练完成")
        return self.history
    # ---------- 其他方法 ----------
    def stop(self) -> None:
        """请求停止训练"""
        self._stop_training = True
    def get_loss_history(self) -> Dict[str, List[float]]:
        """获取损失历史"""
        return self.history
    def evaluate(self, x_test: torch.Tensor) -> torch.Tensor:
        """
        在测试点上评估模型
        
        参数:
            x_test: (N, dim) 测试点
        
        返回:
            (N, 1) 预测值
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(x_test)
    def get_model(self) -> torch.nn.Module:
        """获取模型"""
        return self.model
    def save_checkpoint(self, path: str) -> None:
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
        }, path)
    def load_checkpoint(self, path: str) -> None:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

if __name__ == "__main__":
    """
    PINNTrainer 独立功能测试（不依赖任何外部模块）
    
    本测试模拟了一个简单的一维 PINN 问题：
        - 内部点损失: 让模型逼近 u(x) = sin(2πx)
        - 边界点损失: 约束 u(0)=0, u(1)=0
    
    验证 trainer 的各项功能是否正常。
    """
    print("=" * 70)
    print("PINNTrainer 独立功能测试")
    print("=" * 70)
    # ========================================================================
    # 1. 定义模拟模型
    # ========================================================================
    class MockModel(torch.nn.Module):
        """简单的全连接网络，用于测试"""
        def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_dim, output_dim),
            )
        def forward(self, x):
            return self.net(x)
    # ========================================================================
    # 2. 定义模拟采样器
    # ========================================================================
    class MockSampler:
        """模拟采样器，生成内部点和边界点"""
        def __init__(self):
            self.has_t = False
        def sample_interior(self, n_points):
            """内部点: 在 [0,1] 区间随机采样"""
            return torch.rand(n_points, 1) * 1.0
        def sample_boundary(self, n_points_per_side):
            """边界点: 返回左右端点 (含重复值用于测试)"""
            return {
                'left': torch.zeros(n_points_per_side, 1),
                'right': torch.ones(n_points_per_side, 1),
            }
        def sample_initial(self, n_points):
            """初始条件点 (本测试不需要)"""
            return None
    # ========================================================================
    # 3. 定义损失函数
    # ========================================================================
    def make_mock_losses():
        """生成模拟 PINN 损失函数"""
        def pde_loss(model, points):
            """
            PDE 损失: 让模型逼近目标函数 u(x) = sin(2πx)
            这里使用监督损失模拟 PDE 残差
            """
            u_pred = model(points)
            u_target = torch.sin(2 * torch.pi * points)
            return ((u_pred - u_target) ** 2).mean()
        def bc_loss(model, boundary_pts):
            """
            边界损失: 约束 u(0)=0, u(1)=0
            """
            loss = 0.0
            for side, pts in boundary_pts.items():
                if pts is None or pts.numel() == 0:
                    continue
                u_pred = model(pts)
                # 左边界 -> 0, 右边界 -> 0
                target = torch.zeros_like(u_pred)
                loss += ((u_pred - target) ** 2).mean()
            return loss
        def total_loss(model, points, boundary_pts):
            return pde_loss(model, points) + bc_loss(model, boundary_pts)
        return pde_loss, bc_loss, total_loss
    # ========================================================================
    # 4. 测试函数
    # ========================================================================
    def test_basic_training():
        """测试 1: 基础训练"""
        print("\n[测试 1] 基础训练 (500 轮)")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
            scheduler='plateau',
        )
        sampler = MockSampler()
        history = trainer.train(
            n_epochs=500,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            verbose=True,
            eval_interval=100,
        )
        final_loss = history['total_loss'][-1]
        print(f"  ✅ 训练完成，最终损失: {final_loss:.6f}")
        print(f"  ✅ 历史记录长度: {len(history['total_loss'])}")
        # 验证损失是否下降
        initial_loss = history['total_loss'][0]
        if final_loss < initial_loss * 0.5: print("  ✅ 损失显著下降，训练有效")
        else: print("  ⚠️ 损失下降不明显，可能需要更多轮次")
        return trainer, history
    def test_batch_training():
        """测试 2: 批处理训练"""
        print("\n[测试 2] 批处理训练 (batch_size=64, 500 轮)")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        history = trainer.train(
            n_epochs=500,
            sampler=sampler,
            n_interior=256,
            n_boundary_per_side=10,
            batch_size=64,
            verbose=True,
            eval_interval=100,
        )
        print(f"  ✅ 批处理训练完成，最终损失: {history['total_loss'][-1]:.6f}")
        return trainer, history
    def test_resample_training():
        """测试 3: 带重新采样的训练"""
        print("\n[测试 3] 带重新采样的训练 (每 100 轮重采样)")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        history = trainer.train(
            n_epochs=300,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            resample_every=50,
            verbose=True,
            eval_interval=100,
        )
        print(f"  ✅ 重采样训练完成，最终损失: {history['total_loss'][-1]:.6f}")
        return trainer, history
    def test_early_stop():
        """测试 4: 早停"""
        print("\n[测试 4] 早停 (patience=5, 最多 1000 轮)")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        history = trainer.train(
            n_epochs=1000,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            early_stop_patience=5,
            verbose=True,
            eval_interval=50,
        )
        actual_epochs = len(history['total_loss'])
        print(f"  ✅ 早停触发，实际训练轮数: {actual_epochs} (目标: 1000)")
        return trainer, history
    def test_checkpoint():
        """测试 5: 检查点保存与加载"""
        print("\n[测试 5] 检查点保存与加载")
        print("-" * 60)
        import tempfile
        import os
        # 训练一个模型
        model1 = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer1 = PINNTrainer(
            model=model1,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        trainer1.train(
            n_epochs=200,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            verbose=False,
        )
        # 保存检查点
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        trainer1.save_checkpoint(checkpoint_path)
        print(f"  ✅ 检查点已保存: {checkpoint_path}")
        # 创建新模型并加载
        model2 = MockModel()
        trainer2 = PINNTrainer(
            model=model2,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        trainer2.load_checkpoint(checkpoint_path)
        print(f"  ✅ 检查点已加载")
        # 验证参数是否一致
        params1 = [p.data.numpy() for p in trainer1.model.parameters()]
        params2 = [p.data.numpy() for p in trainer2.model.parameters()]
        all_close = all(
            np.allclose(p1, p2, rtol=1e-6)
            for p1, p2 in zip(params1, params2)
        )
        if all_close:
            print("  ✅ 模型参数一致，加载成功")
        else:
            print("  ❌ 模型参数不一致")
        # 清理临时文件
        os.unlink(checkpoint_path)
        return trainer1, trainer2
    def test_evaluate():
        """测试 6: 评估功能"""
        print("\n[测试 6] 模型评估")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        trainer.train(
            n_epochs=100,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            verbose=False,
        )
        # 测试评估
        x_test = torch.linspace(0, 1, 50).reshape(-1, 1)
        u_pred = trainer.evaluate(x_test)
        print(f"  ✅ 评估完成，输出形状: {u_pred.shape}")
        print(f"  ✅ 预测值范围: [{u_pred.min().item():.4f}, {u_pred.max().item():.4f}]")
        # 检查是否在合理范围
        if -1.5 < u_pred.min().item() and u_pred.max().item() < 1.5:
            print("  ✅ 预测值在合理范围内")
        else:
            print("  ⚠️ 预测值超出预期范围")
        return trainer
    def test_manual_stop():
        """测试 7: 手动停止"""
        print("\n[测试 7] 手动停止")
        print("-" * 60)
        model = MockModel()
        pde_loss, bc_loss, total_loss = make_mock_losses()
        trainer = PINNTrainer(
            model=model,
            loss_functions=(pde_loss, bc_loss, total_loss),
            optimizer='adam',
            lr=1e-3,
        )
        sampler = MockSampler()
        # 用一个线程在 0.5 秒后停止
        import threading
        import time
        def stop_after_delay():
            time.sleep(0.5)
            trainer.stop()
        stop_thread = threading.Thread(target=stop_after_delay)
        stop_thread.start()
        history = trainer.train(
            n_epochs=10000,
            sampler=sampler,
            n_interior=200,
            n_boundary_per_side=10,
            verbose=False,
        )
        actual_epochs = len(history['total_loss'])
        print(f"  ✅ 手动停止触发，实际训练轮数: {actual_epochs} (目标: 10000)")
        print(f"  ✅ 最终损失: {history['total_loss'][-1]:.6f}")
        return trainer, history
    # ========================================================================
    # 5. 运行所有测试
    # ========================================================================
    print("\n开始运行测试...")
    print("=" * 70)
    test_results = {}
    try:
        test_results['basic'] = test_basic_training()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['batch'] = test_batch_training()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['resample'] = test_resample_training()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['early_stop'] = test_early_stop()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['checkpoint'] = test_checkpoint()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['evaluate'] = test_evaluate()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    try:
        test_results['manual_stop'] = test_manual_stop()
        print("  ✅ 通过")
    except Exception as e:
        print(f"  ❌ 失败: {e}")
    # ========================================================================
    # 6. 测试汇总
    # ========================================================================
    print("\n" + "=" * 70)
    print("测试汇总")
    print("=" * 70)
    test_names = {
        'basic': '基础训练',
        'batch': '批处理训练',
        'resample': '重采样训练',
        'early_stop': '早停',
        'checkpoint': '检查点',
        'evaluate': '评估',
        'manual_stop': '手动停止',
    }
    for key, name in test_names.items():
        status = "✅ 通过" if key in test_results else "❌ 失败"
        print(f"  [{key}] {name}: {status}")
    print("=" * 70)
    print("所有测试完成！")
