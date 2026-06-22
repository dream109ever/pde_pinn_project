import torch
import torch.nn as nn
import inspect
import sympy as sp
from typing import Union, Callable, Dict
from .function_factory import PDEConfig, ConditionConfig, BoundaryCondition

class ComplexityAnalyzer:
    """
    分析 PDE 及其边界条件的复杂度。
    评分规则：
        - 常数系数：1 分
        - 简单多项式（幂次 ≤ 2）：2 分
        - 三角函数、指数、对数等：3 分起，每多一个特殊函数 +1
        - 多个函数嵌套或组合：累加
        - 边界数量及系数复杂度额外加分
    """
    def __init__(self, weights: Dict[str, float] = None):
        """
        weights: 各部分的权重因子，默认值如下
            'pde_coeff': 1.0       # 方程每一项系数的权重
            'source': 1.5          # 源项的额外权重
            'bc_coeff': 0.8        # 边界条件中 alpha, beta 的权重
            'bc_gamma': 1.0        # 边界条件中 gamma 的权重
            'bc_count': 0.5        # 每个边界条件本身的权重（条件数量越多越复杂）
            'special_func_extra': 0.5  # 每个特殊函数额外加分
        """
        self.weights = weights or {
            'pde_coeff': 1.0,
            'source': 1.5,
            'bc_coeff': 0.8,
            'bc_gamma': 1.0,
            'bc_count': 0.5,
            'special_func_extra': 0.5
        }
    def _expr_complexity(self, expr: Union[int, float, Callable, sp.Expr]) -> float:
        """评估单个表达式（系数或源项）的复杂度分数"""
        if isinstance(expr, (int, float)):
            return 1.0
        if isinstance(expr, sp.Expr):
            return self._sympy_complexity(expr)
        if callable(expr):
            try:
                src = inspect.getsource(expr).strip()
                score = 3.0
                special_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs',
                                 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan',
                                 'besselj', 'bessely', 'gamma', 'erf']
                count = sum(src.count(f) for f in special_funcs)
                score += count * self.weights.get('special_func_extra', 0.5)
                if '**' in src or 'pow' in src:
                    score += 0.5
                if '/' in src:
                    score += 0.3
                return min(score, 20.0)
            except:
                return 5.0
        return 3.0
    def _sympy_complexity(self, expr: sp.Expr) -> float:
        """分析 sympy 表达式的复杂度"""
        if expr.is_Number:
            return 1.0
        atoms = expr.atoms(sp.Function)
        score = 1.0
        for f in atoms:
            name = f.func.__name__
            if name in ['sin', 'cos', 'tan', 'exp', 'log']:
                score += 1.5
            elif name in ['sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan']:
                score += 1.2
            elif name in ['sqrt', 'abs']:
                score += 0.8
            elif name in ['besselj', 'bessely', 'gamma', 'erf']:
                score += 3.0
            else:
                score += 1.0
        for pow_expr in expr.atoms(sp.Pow):
            exp = pow_expr.exp
            if exp.is_Number and exp > 2:
                score += 0.5 * (exp - 2)
        return score
    def compute_complexity(self, equation : PDEConfig, conditions: list[BoundaryCondition]) -> float:
        """
        计算总复杂度分数。
        equation: PDEConfig 实例
        conditions: BoundaryCondition 列表
        """
        total_score = 0.0
        # 1. 方程系数（包括源项）
        pde_weight = self.weights['pde_coeff']
        for term in equation.terms:
            coeff_symbol = term['coeff_symbol']
            raw_coeff = equation.coeffs.get(coeff_symbol, 0)
            complexity = self._expr_complexity(raw_coeff)
            total_score += pde_weight * complexity
        # 源项
        source_raw = equation.coeffs.get(equation.source_symbol, 0)
        source_complexity = self._expr_complexity(source_raw)
        total_score += self.weights['source'] * source_complexity
        if getattr(equation, 'has_t', False):
            total_score += self.weights.get('has_t', 0.3)
        # 2. 边界条件
        bc_weight = self.weights['bc_coeff']
        gamma_weight = self.weights['bc_gamma']
        count_weight = self.weights['bc_count']
        total_score += count_weight * len(conditions)
        for bc in conditions:
            cond = bc.get_condition()
            alpha_raw = cond.coeffs.get('alpha', 0)
            beta_raw = cond.coeffs.get('beta', 0)
            gamma_raw = cond.coeffs.get('gamma', 0)
            alpha_comp = self._expr_complexity(alpha_raw)
            beta_comp = self._expr_complexity(beta_raw)
            gamma_comp = self._expr_complexity(gamma_raw)
            total_score += bc_weight * (alpha_comp + beta_comp)
            total_score += gamma_weight * gamma_comp
            if getattr(bc, 'is_initial', False):
                total_score += self.weights.get('initial_cond', 0.5)
        return total_score

class NetworkConfigGenerator:
    """
    根据复杂度分数生成网络结构配置。
    """
    def __init__(self, base_config=None, mapping=None):
        """
        base_config: 基础配置字典，如 {'batch_norm': False, 'init_method': 'xavier'}
        mapping: 可调用对象，接收分数返回配置字典；若为None，使用内置分段映射。
        """
        self.base_config = base_config or {}
        self.mapping = mapping or self._default_mapping
    def _default_mapping(self, score):
        """默认分段映射，可根据实际调整分数阈值"""
        if score < 30:
            hidden_dims = [16, 16]
            activation = 'tanh'
        elif score < 60:
            hidden_dims = [32, 32, 16]
            activation = 'tanh'
        else:
            hidden_dims = [64, 64, 32, 16]
            activation = 'tanh'
        return {
            'hidden_dims': hidden_dims,
            'activation': activation
        }
    def generate_config(self, score):
        """返回完整的网络配置字典"""
        config = self.mapping(score)
        config.update(self.base_config)
        if 'input_dim' not in config or 'output_dim' not in config:
            raise ValueError("配置中必须包含 input_dim 和 output_dim")
        return config

def build_network(input_dim, output_dim, hidden_dims, activation='tanh', batch_norm=False, init_method='xavier'):
    """
    构建全连接神经网络。

    参数:
        input_dim: 输入特征数
        output_dim: 输出特征数
        hidden_dims: 列表，每个元素为隐藏层神经元数，例如 [64, 64, 32]
        activation: 激活函数类型，支持 'tanh', 'relu', 'sigmoid', 'sin'
        batch_norm: 是否在每层后添加 BatchNorm1d
        init_method: 权重初始化方法，支持 'xavier', 'kaiming', 'normal'

    返回:
        nn.Sequential 模型
    """
    act_dict = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'sin': lambda x: torch.sin(x)  # 自定义 sin 激活
    }
    activation_func = act_dict.get(activation.lower(), nn.Tanh())
    layers = []
    prev_dim = input_dim
    for _, h_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, h_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h_dim))
        layers.append(activation_func)
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    model = nn.Sequential(*layers)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if init_method == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_method == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity=activation.lower())
            elif init_method == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            else:
                pass
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    return model

class NetworkFactory:
    """网络参数工厂，支持从配置字典构建模型"""
    def __init__(self, config):
        """
        config 应包含:
            - input_dim: int
            - output_dim: int
            - hidden_dims: list of int
            - activation: str (default 'tanh')
            - batch_norm: bool (default False)
            - init_method: str (default 'xavier')
        """
        self.config = config
    def build(self):
        return build_network(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            hidden_dims=self.config['hidden_dims'],
            activation=self.config.get('activation', 'tanh'),
            batch_norm=self.config.get('batch_norm', False),
            init_method=self.config.get('init_method', 'xavier')
        )

def build_model(equation : PDEConfig, conditions: list[BoundaryCondition], base_config={'input_dim': 2, 'output_dim': 1}):
    analyzer = ComplexityAnalyzer()
    score = analyzer.compute_complexity(equation, conditions)
    generator = NetworkConfigGenerator(base_config=base_config)
    config = generator.generate_config(score)
    factory = NetworkFactory(config)
    model = factory.build()
    return model

# # 使用示例
# # coeffs 示例：含有 sin 和 exp
# coeffs = {
#     'A': 1.0,
#     'C': 1.0,
#     'G': lambda x, y: torch.sin(torch.pi * x) * torch.exp(-y)  # 可调用
# }
# equation = PDEConfig(coeffs)
# bc = BoundaryCondition('x=0', ConditionConfig({'alpha': 1, 'beta': 0, 'gamma': 0}))
# conditions = [bc]
# analyzer = ComplexityAnalyzer()
# score = analyzer.compute_complexity(equation, conditions)
# print(f"Complexity score: {score:.2f}")
# # 创建生成器（可自定义映射函数）
# generator = NetworkConfigGenerator(base_config={'input_dim': 2, 'output_dim': 1})
# # 假设已有复杂度分数
# config = generator.generate_config(score)
# # 将 config 传给 NetworkFactory
# factory = NetworkFactory(config)
# model = factory.build()
