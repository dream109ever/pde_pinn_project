"""
完整 PINN 训练流程脚本（无 GUI / 无可视化）

本脚本演示了从定义问题到训练完成的完整流程：
    1. 定义 PDE 问题（系数、源项、边界条件、定义域）
    2. 解析输入并生成损失函数
    3. 自动构建神经网络
    4. 创建采样器
    5. 训练模型
    6. 评估并输出结果

依赖:
    - src 包中的 function_factory, data_utils, network_factory, trainer

使用方法:
    python run_training.py
"""

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

import torch
import numpy as np
from src import *

# ============================================================================
# 配置区：用户在此定义要解决的问题
# ============================================================================
# ---------- 示例 1: 一维稳态 ODE ----------
# 方程: u'' + u = 0, u(0)=1, u'(0)=0
# 解析解: u(x) = cos(x)
PROBLEM_1D_STEADY = {
    "dimension": 1,
    "order": 2,
    "has_t": False,
    "coeffs": [1, 0, 1],           # u'' + u = 0
    "source_term": "0",
    "domain": {"x": [0, 1]},
    "condition": [
        {"point": 0.0, "value": 1.0, "derivative": 0},
        {"point": 0.0, "value": 0.0, "derivative": 1},
    ],
}
# ---------- 示例 2: 一维含时 PDE（热传导）----------
# 方程: u_t = 0.1*u_xx, u(0,t)=u(1,t)=0, u(x,0)=sin(pi*x)
PROBLEM_1D_TRANSIENT = {
    "dimension": 1,
    "order": 2,
    "has_t": True,
    "coeffs": {"u_t": 1.0, "u_xx": -0.1},
    "source_term": "0",
    "domain": {"x": [0, 1], "t": [0, 0.5]},
    "condition": [
        {"side": "initial", "derivative": 0, "value": "sin(pi*x)"},
        {"side": "left", "type": "dirichlet", "value": "0"},
        {"side": "right", "type": "dirichlet", "value": "0"},
    ],
}
# ---------- 示例 3: 二维稳态 PDE（泊松方程）----------
# 方程: u_xx + u_yy = sin(pi*x)*sin(pi*y), 四边 u=0
PROBLEM_2D_STEADY = {
    "dimension": 2,
    "order": 2,
    "has_t": False,
    "coeffs": {"u_xx": 1.0, "u_yy": 1.0},
    "source_term": "sin(pi*x)*sin(pi*y)",
    "domain": {"x": [0, 1], "y": [0, 1]},
    "condition": [
        {"side": "left", "type": "dirichlet", "value": "0"},
        {"side": "right", "type": "dirichlet", "value": "0"},
        {"side": "bottom", "type": "dirichlet", "value": "0"},
        {"side": "top", "type": "dirichlet", "value": "0"},
    ],
}
# ---------- 示例 4: 二维含时 PDE（热传导）----------
# 方程: u_t = 0.1*(u_xx + u_yy), 四边 u=0, u(x,y,0)=sin(pi*x)*sin(pi*y)
PROBLEM_2D_TRANSIENT = {
    "dimension": 2,
    "order": 2,
    "has_t": True,
    "coeffs": {"u_t": 1.0, "u_xx": 0.1, "u_yy": 0.1},
    "source_term": "0",
    "domain": {"x": [0, 1], "y": [0, 1], "t": [0, 0.5]},
    "condition": [
        {"side": "initial", "derivative": 0, "value": "sin(pi*x)*sin(pi*y)"},
        {"side": "left", "type": "dirichlet", "value": "0"},
        {"side": "right", "type": "dirichlet", "value": "0"},
        {"side": "bottom", "type": "dirichlet", "value": "0"},
        {"side": "top", "type": "dirichlet", "value": "0"},
    ],
}
# ============================================================================
# 辅助函数
# ============================================================================
def print_separator(title: str = "", char: str = "=", length: int = 70):
    """打印分隔线"""
    if title:
        padding = (length - len(title) - 2) // 2
        print(f"{char * padding} {title} {char * (length - padding - len(title) - 2)}")
    else:
        print(char * length)
def print_problem_info(problem: dict):
    """打印问题信息"""
    print_separator("问题信息")
    print(f"  维度: {problem['dimension']}D, 阶数: {problem['order']}, 含时: {problem['has_t']}")
    print(f"  系数: {problem['coeffs']}")
    print(f"  源项: {problem['source_term']}")
    print(f"  定义域: {problem['domain']}")
    print(f"  条件数: {len(problem['condition'])}")
    print_separator()
def print_training_result(history: dict, final_loss: float):
    """打印训练结果摘要"""
    print_separator("训练结果")
    print(f"  最终损失: {final_loss:.6e}")
    print(f"  总轮数: {len(history['total_loss'])}")
    if history['total_loss']:
        initial_loss = history['total_loss'][0]
        reduction = (initial_loss - final_loss) / initial_loss * 100
        print(f"  损失下降: {initial_loss:.6e} → {final_loss:.6e} (减少 {reduction:.1f}%)")
    print_separator()
def evaluate_and_print(model, sampler, problem: dict, n_points: int = 100):
    """
    在定义域内随机采样并评估模型，打印统计信息
    """
    dimension = problem['dimension']
    has_t = problem['has_t']
    # 生成测试点
    if dimension == 1 and not has_t:
        # 1D 稳态：x 采样
        x_test = torch.linspace(0, 1, n_points).reshape(-1, 1)
        points = x_test
        point_labels = [f"x={x:.3f}" for x in x_test.flatten().tolist()]
    elif dimension == 1 and has_t:
        # 1D 含时：网格采样 (x, t)
        x_vals = torch.linspace(0, 1, int(np.sqrt(n_points)))
        t_vals = torch.linspace(0, 0.5, int(np.sqrt(n_points)))
        xx, tt = torch.meshgrid(x_vals, t_vals, indexing='ij')
        points = torch.stack([xx.flatten(), tt.flatten()], dim=1)
        x_flat = xx.flatten()
        t_flat = tt.flatten()
        point_labels = [f"(x={x_flat[i].item():.2f}, t={t_flat[i].item():.2f})" for i in range(len(x_flat))]
    elif dimension == 2 and not has_t:
        # 2D 稳态：网格采样 (x, y)
        x_vals = torch.linspace(0, 1, int(np.sqrt(n_points)))
        y_vals = torch.linspace(0, 1, int(np.sqrt(n_points)))
        xx, yy = torch.meshgrid(x_vals, y_vals, indexing='ij')
        points = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        x_flat = xx.flatten()
        y_flat = yy.flatten()
        point_labels = [f"(x={x_flat[i].item():.2f}, y={y_flat[i].item():.2f})" for i in range(len(x_flat))]
    else:
        # 2D 含时：随机采样少量点展示
        sampler_temp = DomainSampler(
            problem['domain']['x'],
            problem['domain'].get('y', None),
            problem['domain'].get('t', None)
        )
        points = sampler_temp.sample_interior(min(n_points, 50))
        point_labels = [f"点{i+1}" for i in range(points.shape[0])]
    # 评估
    model.eval()
    with torch.no_grad():
        pred = model(points)
    pred_np = pred.numpy().flatten()
    print_separator("模型评估")
    print(f"  测试点数: {points.shape[0]}")
    print(f"  预测值范围: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    print(f"  预测值均值: {pred_np.mean():.4f}")
    print(f"  预测值标准差: {pred_np.std():.4f}")
    print_separator()
    # 打印前 10 个点的值
    print("  前 10 个点的预测值:")
    for i in range(min(10, len(pred_np))):
        label = point_labels[i] if i < len(point_labels) else f"点{i+1}"
        print(f"    {label}: {pred_np[i]:.6f}")
    # 内部随机 10 个点的预测值
    print("\n  内部随机 10 个点的预测值:")
    try:
        random_pts = sampler.sample_interior(10)
        random_pts_np = random_pts.numpy()
        model.eval()
        with torch.no_grad():
            random_pred = model(random_pts).numpy().flatten()
        if problem['dimension'] == 1 and not problem['has_t']:
            labels = [f"x={pt[0]:.3f}" for pt in random_pts_np]
        elif problem['dimension'] == 1 and problem['has_t']: 
            labels = [f"(x={pt[0]:.3f}, t={pt[1]:.3f})" for pt in random_pts_np]
        elif problem['dimension'] == 2 and not problem['has_t']: 
            labels = [f"(x={pt[0]:.3f}, y={pt[1]:.3f})" for pt in random_pts_np]
        else: 
            labels = [f"(x={pt[0]:.3f}, y={pt[1]:.3f}, t={pt[2]:.3f})" for pt in random_pts_np]
        for i in range(min(10, len(random_pred))):
            print(f"    {labels[i]}: {random_pred[i]:.6f}")
    except Exception as e:
        print(f"    无法生成内部随机点: {e}")
    return points, pred_np
# ============================================================================
# 主函数
# ============================================================================
def run_training(problem: dict, 
                 n_epochs: int = 3000,
                 n_interior: int = 2500,
                 n_boundary_per_side: int = 100,
                 n_initial: int = 500,
                 lr: float = 1e-3,
                 device: str = 'cpu',
                 verbose: bool = True) -> dict:
    """
    运行完整训练流程
    
    参数:
        problem: 问题配置字典
        n_epochs: 训练轮数
        n_interior: 内部点数量
        n_boundary_per_side: 每条边边界点数量
        n_initial: 初始条件点数量
        lr: 学习率
        device: 计算设备
        verbose: 是否打印详细信息
    
    返回:
        dict: 包含 model, history, loss_functions, sampler
    """
    dimension = problem['dimension']
    order = problem['order']
    has_t = problem['has_t']
    coeffs = problem['coeffs']
    source_term = problem['source_term']
    domain = problem['domain']
    condition = problem['condition']
    if verbose:
        print_problem_info(problem)
    # ====== 步骤 1: 解析输入并生成损失函数 ======
    if verbose:
        print("[Step 1] 解析输入并生成损失函数...")
    if dimension == 1 and not has_t:
        coeff_funcs = InputParser.parse_coeffs_1d(coeffs, ['x'])
        f, _ = InputParser.parse_source(source_term, ['x'])
        pde_loss, ic_loss, total_loss = LossGenerator.generate(
            dimension=1, has_t=False,
            coeff_funcs=coeff_funcs,
            f=f,
            condition=condition,
            order=order
        )
        loss_functions = (pde_loss, ic_loss, total_loss)
    elif dimension == 1 and has_t:
        parsed = InputParser.parse_coeffs_1d_transient(coeffs, ['x', 't'])
        f, _ = InputParser.parse_source(source_term, ['x', 't'])
        structured = InputParser.parse_conditions(condition, ['x', 't'])
        ic_conds = structured['initial']
        bc_sides = {"left": [], "right": []}
        for bc in structured['boundary']:
            side = bc.get('location_clean')
            if side in bc_sides:
                bc_sides[side].append(bc)
        pde_loss, bc_ic_loss, total_loss = LossGenerator.generate(
            dimension=1, has_t=True,
            c_tt_fn=parsed.get('u_tt', lambda x, t: 0.0),
            c_t_fn=parsed.get('u_t', lambda x, t: 0.0),
            c_xx_fn=parsed.get('u_xx', lambda x, t: 0.0),
            c_x_fn=parsed.get('u_x', lambda x, t: 0.0),
            c_u_fn=parsed.get('u', lambda x, t: 0.0),
            f=f,
            ic_conds=ic_conds,
            bc_sides=bc_sides
        )
        loss_functions = (pde_loss, bc_ic_loss, total_loss)
    elif dimension == 2 and not has_t:
        parsed = InputParser.parse_coeffs_2d(coeffs, ['x', 'y'])
        f, _ = InputParser.parse_source(source_term, ['x', 'y'])
        structured = InputParser.parse_conditions(condition, ['x', 'y'])
        bc_sides = {"left": [], "right": [], "bottom": [], "top": []}
        for bc in structured['boundary']:
            side = bc.get('location_clean')
            if side in bc_sides:
                bc_sides[side].append(bc)
        pde_loss, bc_loss, total_loss = LossGenerator.generate(
            dimension=2, has_t=False,
            c_xx_fn=parsed.get('u_xx', lambda x, y: 0.0),
            c_yy_fn=parsed.get('u_yy', lambda x, y: 0.0),
            c_xy_fn=parsed.get('u_xy', lambda x, y: 0.0),
            c_x_fn=parsed.get('u_x', lambda x, y: 0.0),
            c_y_fn=parsed.get('u_y', lambda x, y: 0.0),
            c_u_fn=parsed.get('u', lambda x, y: 0.0),
            f=f,
            bc_sides=bc_sides
        )
        loss_functions = (pde_loss, bc_loss, total_loss)
    elif dimension == 2 and has_t:
        parsed = InputParser.parse_coeffs_2d_transient(coeffs, ['x', 'y', 't'])
        f, _ = InputParser.parse_source(source_term, ['x', 'y', 't'])
        structured = InputParser.parse_conditions(condition, ['x', 'y', 't'])
        ic_conds = structured['initial']
        bc_sides = {"left": [], "right": [], "bottom": [], "top": []}
        for bc in structured['boundary']:
            side = bc.get('location_clean')
            if side in bc_sides:
                bc_sides[side].append(bc)
        pde_loss, bc_ic_loss, total_loss = LossGenerator.generate(
            dimension=2, has_t=True,
            c_tt_fn=parsed.get('u_tt', lambda x, y, t: 0.0),
            c_t_fn=parsed.get('u_t', lambda x, y, t: 0.0),
            c_xx_fn=parsed.get('u_xx', lambda x, y, t: 0.0),
            c_yy_fn=parsed.get('u_yy', lambda x, y, t: 0.0),
            c_xy_fn=parsed.get('u_xy', lambda x, y, t: 0.0),
            c_x_fn=parsed.get('u_x', lambda x, y, t: 0.0),
            c_y_fn=parsed.get('u_y', lambda x, y, t: 0.0),
            c_u_fn=parsed.get('u', lambda x, y, t: 0.0),
            f=f,
            ic_conds=ic_conds,
            bc_sides=bc_sides
        )
        loss_functions = (pde_loss, bc_ic_loss, total_loss)
    else:
        raise ValueError(f"不支持的配置: dimension={dimension}, has_t={has_t}")
    if verbose:
        print(f"  ✅ 损失函数生成完成 (共 {len(loss_functions)} 个)")
    # ====== 步骤 2: 自动构建网络 ======
    if verbose:
        print("[Step 2] 自动构建神经网络...")
    model = build_model(
        coeffs=coeffs,
        source_term=source_term,
        conditions=condition,
        has_t=has_t,
        dimension=dimension,
        verbose=verbose,
    )
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ 网络构建完成 (参数量: {total_params:,})")
    # ====== 步骤 3: 创建采样器 ======
    if verbose:
        print("[Step 3] 创建采样器...")
    x_range = domain['x']
    y_range = domain.get('y', None)
    t_range = domain.get('t', None)
    sampler = DomainSampler(x_range, y_range, t_range)
    if verbose:
        print(f"  ✅ 采样器创建完成 (维度: {sampler.dim}D)")
    # ====== 步骤 4: 创建训练器 ======
    if verbose:
        print("[Step 4] 创建训练器...")
    trainer = PINNTrainer(
        model=model,
        loss_functions=loss_functions,
        optimizer='adam',
        lr=lr,
        scheduler='plateau',
        device=device,
    )
    if verbose:
        print(f"  ✅ 训练器创建完成 (设备: {device})")
    # ====== 步骤 5: 执行训练 ======
    if verbose:
        print_separator("开始训练")
    history = trainer.train(
        n_epochs=n_epochs,
        sampler=sampler,
        n_interior=n_interior,
        n_boundary_per_side=n_boundary_per_side,
        n_initial=n_initial,
        verbose=verbose,
        eval_interval=max(1, n_epochs // 10),
    )
    final_loss = history['total_loss'][-1] if history['total_loss'] else float('inf')
    if verbose:
        print_training_result(history, final_loss)
    # ====== 步骤 6: 评估 ======
    if verbose:
        evaluate_and_print(model, sampler, problem)
    return {
        'model': model,
        'history': history,
        'loss_functions': loss_functions,
        'sampler': sampler,
        'trainer': trainer,
        'final_loss': final_loss,
    }
# ============================================================================
# 主入口
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PINN PDE 求解器 - 训练脚本')
    parser.add_argument('--problem', type=str, default='1d_steady',
                        choices=['1d_steady', '1d_transient', '2d_steady', '2d_transient'],
                        help='选择要训练的问题类型')
    parser.add_argument('--epochs', type=int, default=3000, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--device', type=str, default='cpu', help='计算设备 (cpu/cuda)')
    parser.add_argument('--quiet', action='store_true', help='安静模式 (不打印详细信息)')
    args = parser.parse_args()
    # 选择问题
    problem_map = {
        '1d_steady': PROBLEM_1D_STEADY,
        '1d_transient': PROBLEM_1D_TRANSIENT,
        '2d_steady': PROBLEM_2D_STEADY,
        '2d_transient': PROBLEM_2D_TRANSIENT,
    }
    problem = problem_map[args.problem]
    print_separator("PINN PDE Solver")
    print(f"  问题: {args.problem}")
    print(f"  轮数: {args.epochs}")
    print(f"  学习率: {args.lr}")
    print(f"  设备: {args.device}")
    print_separator()
    # 运行训练
    result = run_training(
        problem=problem,
        n_epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=not args.quiet,
    )
    print_separator("训练完成！")
    # 使用 solve_pde 进行快速验证（如果可用）
    print("\n[可选] 使用 solve_pde 进行快速验证...")
    try:
        result_verify = solve_pde(
            dimension=problem['dimension'],
            order=problem['order'],
            has_t=problem['has_t'],
            coeffs=problem['coeffs'],
            source_term=problem['source_term'],
            domain=problem['domain'],
            condition=problem['condition'],
        )
        print(f"  精确解表达式: {result_verify['exact_expression']}")
        print(f"  精确解可用: {result_verify['exact_solution'] is not None}")
    except Exception as e:
        print(f"  ⚠️ solve_pde 验证失败: {e}")
    print_separator()
