import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def plot_loss_history(history, figsize=(10, 6), log_scale=True, save_path=None):
    """
    绘制训练损失曲线。

    参数:
        history: 字典
        figsize: 图片尺寸
        log_scale: 是否使用对数坐标
        save_path: 保存路径
    """
    plt.figure(figsize=figsize)
    epochs = range(1, len(history['total_loss']) + 1)
    plt.plot(epochs, history['total_loss'], label='Total Loss', linewidth=2)
    if 'pde_loss' in history and history['pde_loss']:
        plt.plot(epochs, history['pde_loss'], label='PDE Loss', linestyle='--')
    if 'bc_loss' in history and history['bc_loss']:
        plt.plot(epochs, history['bc_loss'], label='BC Loss', linestyle='-.')
    if 'ic_loss' in history and history['ic_loss']:
        plt.plot(epochs, history['ic_loss'], label='IC Loss', linestyle=':')
    if log_scale:
        plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_1d_solution(model, x_range=(0,1), n_points=200, true_func=None,
                     title='Predicted vs True Solution', save_path=None):
    """
    一维问题：绘制模型预测解与真实解（若提供）的对比。
    参数:
        model: 训练好的神经网络
        x_range: (x_min, x_max)
        n_points: 采样点数
        true_func: 真实解函数，接受张量返回张量，可选
        title: 图标题
        save_path: 保存路径
    """
    x = torch.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        u_pred = model(x).numpy().flatten()
    x_np = x.numpy().flatten()
    
    plt.figure(figsize=(8, 5))
    plt.plot(x_np, u_pred, 'b-', label='PINN Prediction', linewidth=2)
    if true_func is not None:
        u_true = true_func(x).numpy().flatten()
        plt.plot(x_np, u_true, 'r--', label='Exact Solution', linewidth=2)
        # 误差
        error = np.abs(u_pred - u_true)
        print(f"Max absolute error: {error.max():.2e}, Mean error: {error.mean():.2e}")
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_2d_solution(model, x_range=(0,1), y_range=(0,1), n_points=100,
                     true_func=None, title='Predicted Solution', save_path=None):
    """
    二维问题：绘制预测解的彩色填充图（等高线或伪彩色）。
    """
    x = torch.linspace(x_range[0], x_range[1], n_points)
    y = torch.linspace(y_range[0], y_range[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    model.eval()
    with torch.no_grad():
        Z_pred = model(points).numpy().reshape(n_points, n_points)
    
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X.numpy(), Y.numpy(), Z_pred, levels=50, cmap='viridis')
    plt.colorbar(cp, label='u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    if true_func is not None:
        Z_true = true_func(points).numpy().reshape(n_points, n_points)
        error = np.abs(Z_pred - Z_true)
        plt.figure(figsize=(8, 6))
        cp_err = plt.contourf(X.numpy(), Y.numpy(), error, levels=50, cmap='hot')
        plt.colorbar(cp_err, label='Absolute Error')
        plt.title('Absolute Error')
        if save_path:
            plt.savefig(save_path.replace('.png', '_error.png'), dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Max error: {error.max():.2e}, Mean error: {error.mean():.2e}")

def plot_3d_surface(model, x_range=(0,1), y_range=(0,1), n_points=100,
                    true_func=None, title='Predicted Solution Surface', save_path=None):
    """
    三维曲面图（适用于二维问题的解）。
    若提供 true_func，则绘制并排的预测解和真实解，并打印误差。
    """
    x = torch.linspace(x_range[0], x_range[1], n_points)
    y = torch.linspace(y_range[0], y_range[1], n_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    model.eval()
    with torch.no_grad():
        Z_pred = model(points).numpy().reshape(n_points, n_points)
    if true_func is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', edgecolor='none')
        fig.colorbar(surf, label='u(x,y)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        return
    Z_true = true_func(points).numpy().reshape(n_points, n_points)
    error = np.abs(Z_pred - Z_true)
    print(f"Max error: {error.max():.2e}, Mean error: {error.mean():.2e}")
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X.numpy(), Y.numpy(), Z_pred, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_title('PINN Prediction')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X.numpy(), Y.numpy(), Z_true, cmap='plasma', edgecolor='none')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_title('Exact Solution')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)
    plt.suptitle(title, fontsize=14)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_comparison_1d_multiple(models, labels, x_range=(0,1), n_points=200,
                                true_func=None, title='Comparison', save_path=None):
    """
    比较多个模型的预测结果（一维）。
    """
    x = torch.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)
    plt.figure(figsize=(8, 5))
    for model, label in zip(models, labels):
        model.eval()
        with torch.no_grad():
            u_pred = model(x).numpy().flatten()
        plt.plot(x.numpy().flatten(), u_pred, '--', label=label, linewidth=1.5)
    if true_func is not None:
        u_true = true_func(x).numpy().flatten()
        plt.plot(x.numpy().flatten(), u_true, 'k-', label='Exact', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def plot_solution(model, x_range, y_range=None, n_points=100, true_func=None,
                  plot_type='auto', title=None, save_path=None):
    """
    统一的解可视化端口，根据参数自动选择一维、二维或三维绘图。

    参数:
        model: 训练好的神经网络
        x_range: (x_min, x_max) 或 (x_min, x_max, t_fixed) 可扩展
        y_range: (y_min, y_max) 若为 None 则绘制一维；否则二维/三维
        n_points: 每个方向的采样点数
        true_func: 真实解函数（仅在一维和二维中可用）
        plot_type: 'auto', '1d', '2d', '3d'
        title: 图标题
        save_path: 保存路径
    """
    if y_range is None or plot_type == '1d':
        if title is None:
            title = 'Predicted vs True Solution'
        plot_1d_solution(model, x_range, n_points, true_func, title, save_path)
    elif plot_type == '3d':
        if title is None:
            title = 'Predicted Solution Surface'
        plot_3d_surface(model, x_range, y_range, n_points, true_func, title, save_path)
    else:
        if title is None:
            title = 'Predicted Solution'
        plot_2d_solution(model, x_range, y_range, n_points, true_func, title, save_path)

def quick_plot_from_trainer(trainer, x_range=(0,1), y_range=None, n_points=200, true_func=None, save_dir=None, plot_type='auto'):
    plot_loss_history(trainer.get_loss_history(), save_path=f"{save_dir}/loss.png" if save_dir else None)
    model = trainer.model
    plot_solution(model, x_range, y_range, n_points, true_func, save_path=f"{save_dir}/solution.png" if save_dir else None, plot_type=plot_type)
