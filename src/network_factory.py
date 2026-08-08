# src/network_factory.py
"""
神经网络智能构建模块。

提供基于问题复杂度的网络架构自动推荐与构建功能。
包含自定义正弦激活函数、复杂度分析器、网络配置生成器和网络构建工厂。
"""
import re
import torch
import inspect
import sympy as sp
import torch.nn as nn
from typing import Union, Callable, Dict, List, Optional

class SineActivation(nn.Module):
    """
    正弦激活函数（SIREN 架构的核心）。

    在 PINN 中求解高频振荡、强非线性或具有陡峭梯度的物理问题时，
    正弦激活函数能够有效打破传统 Tanh/ReLU 的“光谱偏差”现象，加速高频分量收敛。

    :param omega_0: 正弦函数的角频率缩放因子，默认 30.0
    :type omega_0: float
    """
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)

class ComplexityAnalyzer:
    """
    分析 PDE 方程及其边界条件的复杂度，生成评分用于自动选择网络规模。

    :param weights: 各项权重配置，可选
    :type weights: Optional[Dict[str, float]]
    """
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            'pde_coeff': 1.0,
            'source': 1.5,
            'bc_coeff': 0.8,
            'bc_value': 1.0,
            'bc_count': 1.0,
            'special_func_extra': 2.0,
            'freq_factor_weight': 1.5,
            'derivative_bc_penalty': 8.0,
            'has_t': 1.5,
        }
    # ---------- 表达式复杂度评估 ----------
    def _expr_complexity(self, expr: Union[int, float, str, Callable, sp.Expr]) -> float:
        """
        评估单个表达式的复杂度分数。

        :param expr: 待评估的表达式
        :type expr: Union[int, float, str, Callable, sp.Expr]
        :return: 复杂度分数
        :rtype: float
        """
        if isinstance(expr, str):
            try: expr = float(expr.strip())
            except ValueError: pass
        if isinstance(expr, (int, float)):
            return 1.0
        if isinstance(expr, str):
            return self._string_complexity(expr)
        if isinstance(expr, sp.Expr):
            return self._sympy_complexity(expr)
        if callable(expr):
            return self._callable_complexity(expr)
        return 3.0
    def _string_complexity(self, expr: str) -> float:
        """
        分析字符串表达式的复杂度，包含高频因子提取机制。

        :param expr: 表达式字符串
        :type expr: str
        :return: 复杂度分数
        :rtype: float
        """
        score = 2.0
        special_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'abs', 'sinh', 'cosh', 'tanh']
        # 1. 基础特殊函数统计
        count = sum(expr.count(f) for f in special_funcs)
        score += count * self.weights.get('special_func_extra', 2.0)
        # 2. 正则提取高频频率因子 (例如识别 50*x, pi*x 内部的乘数)
        inner_matches = re.findall(r'(?:sin|cos|tan|sinh|cosh|tanh)\(([^)]+)\)', expr)
        for match in inner_matches:
            nums = re.findall(r'[-+]?\d*\.\d+|\d+', match)
            if nums:
                max_freq = max([abs(float(n)) for n in nums])
                if max_freq > 1.0:
                    score += max_freq * self.weights.get('freq_factor_weight', 1.5)
            if 'pi' in match:
                score += 3.0
        if '**' in expr or 'pow' in expr:
            score += 1.0
        if '/' in expr:
            score += 0.5
        return min(score, 50.0)
    def _sympy_complexity(self, expr: sp.Expr) -> float:
        """
        分析 SymPy 表达式的复杂度。

        :param expr: SymPy 表达式
        :type expr: sp.Expr
        :return: 复杂度分数
        :rtype: float
        """
        if expr.is_Number:
            return 1.0
        score = 1.0
        for f in expr.atoms(sp.Function):
            name = f.func.__name__
            if name in ['sin', 'cos', 'tan', 'exp', 'log']:
                score += 2.5
            elif name in ['sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan']:
                score += 2.0
            elif name in ['sqrt', 'abs']:
                score += 1.0
            else:
                score += 1.5
        expr_str = str(expr)
        return self._string_complexity(expr_str)
    def _callable_complexity(self, expr: Callable) -> float:
        """
        分析自定义可调用对象的复杂度。

        :param expr: 可调用对象
        :type expr: Callable
        :return: 复杂度分数
        :rtype: float
        """
        try: return self._string_complexity(inspect.getsource(expr).strip())
        except: return 6.0
    # ---------- 主计算接口 ----------
    def compute_complexity(
        self,
        coeffs: Union[Dict[str, Union[float, str, Callable]], List[Union[float, str, Callable]], float],
        source_term: Union[float, str, Callable] = 0.0,
        conditions: Optional[List[Dict]] = None,
        has_t: bool = False,
        dimension: int = 1,
    ) -> float:
        """
        计算总复杂度分数。

        :param coeffs: 方程系数
            - 一维 ODE 使用 list，如 [2, 2, 1] 表示 u'' + 2u' + 2u
            - PDE 使用 dict，如 {'u_xx': 1.0, 'u_yy': 1.0}
        :type coeffs: Union[Dict[str, Union[float, str, Callable]], List[Union[float, str, Callable]], float]

        :param source_term: 源项，可为常数、字符串表达式或可调用对象
        :type source_term: Union[float, str, Callable]

        :param conditions: 边界条件列表，每个条件为 dict
            - 格式: {'location': 'left', 'type': 'dirichlet', 'value': 0.0}
            - 或对于 ODE: {'point': 0.0, 'value': 0.0, 'derivative': 0}
        :type conditions: Optional[List[Dict]]

        :param has_t: 是否含时间变量
        :type has_t: bool

        :param dimension: 空间维度，1 或 2
        :type dimension: int

        :return: 复杂度分数
        :rtype: float
        """
        total_score = 0.0
        pde_weight = self.weights['pde_coeff']
        # ---------- 1. 方程本身系数 ----------
        if isinstance(coeffs, dict):
            for key, val in coeffs.items():
                total_score += pde_weight * self._expr_complexity(val)
        elif isinstance(coeffs, list):
            for val in coeffs:
                total_score += pde_weight * self._expr_complexity(val)
        else:
            total_score += pde_weight * self._expr_complexity(coeffs)
        # ---------- 2. 方程源项 ----------
        source_complexity = self._expr_complexity(source_term)
        total_score += self.weights['source'] * source_complexity
        # ---------- 3. 时间项惩罚 ----------
        if has_t:
            total_score += self.weights.get('has_t', 1.5)
        # ---------- 4. 空间维度的指数级放大 ----------
        # 1D -> 1.0倍， 2D -> 2.5倍， 3D -> 5.0倍
        dim_multipliers = {1: 1.0, 2: 2.5, 3: 5.0}
        total_score *= dim_multipliers.get(dimension, 1.0)
        # ---------- 5. 边界条件类型及代价深度分析 ----------
        if conditions:
            bc_weight = self.weights['bc_coeff']
            value_weight = self.weights['bc_value']
            count_weight = self.weights['bc_count']
            total_score += count_weight * len(conditions)
            for cond in conditions:
                for key in ['value', 'coefficient', 'alpha', 'beta', 'gamma']:
                    if key in cond:
                        total_score += bc_weight * self._expr_complexity(cond[key])
                c_type = str(cond.get('type', '')).lower()
                if 'neumann' in c_type or 'robin' in c_type or cond.get('derivative', 0) > 0:
                    total_score += self.weights.get('derivative_bc_penalty', 8.0)
        return total_score

class NetworkConfigGenerator:
    """
    根据复杂度分数，智能化生成与之匹配的网络层级与激活函数配置。

    :param base_config: 基础配置，如 {'batch_norm': False, 'init_method': 'xavier'}
    :type base_config: Optional[Dict]

    :param mapping: 可调用对象，接收分数返回配置字典
    :type mapping: Optional[Callable[[float], Dict]]
    """
    def __init__(self, base_config: Optional[Dict] = None, mapping: Optional[Callable[[float], Dict]] = None,):
        self.base_config = base_config or {}
        self.mapping = mapping or self._default_mapping
    def _default_mapping(self, score: float) -> Dict:
        """
        分段动态拓扑映射表。

        全面放大了隐藏层神经元基数，防止复杂/多维场景下 PINN 发生严重欠拟合。
        """
        if score < 15:
            # 极简问题，如一维低阶常系数稳态 ODE
            hidden_dims = [32, 32]
            activation = 'tanh'
        elif score < 40:
            # 中等难度，如常规一维时变方程或简单二维稳态方程
            hidden_dims = [64, 64, 64]
            activation = 'tanh'
        elif score < 80:
            # 高难度问题，复杂二维方程或低频波动问题
            hidden_dims = [128, 128, 128, 64]
            activation = 'tanh'
        else:
            # 极限/超高频/多维高耦合物理场
            # 大幅拉高网络容量，并自动切换为正弦激活函数(SIREN)打破高频光谱偏差
            hidden_dims = [256, 256, 256, 256, 128]
            activation = 'sin'
        return {
            'hidden_dims': hidden_dims,
            'activation': activation
        }
    def generate_config(self, score: float, input_dim: int, output_dim: int = 1) -> Dict:
        """
        生成完整的网络配置。

        :param score: 复杂度分数
        :type score: float
        :param input_dim: 输入维度
        :type input_dim: int
        :param output_dim: 输出维度，默认为 1
        :type output_dim: int
        :return: 完整网络配置
        :rtype: Dict
        """
        config = self.mapping(score)
        config.update(self.base_config)
        config['input_dim'] = input_dim
        config['output_dim'] = output_dim
        return config

def build_network(
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int],
    activation: str = 'tanh',
    batch_norm: bool = False,
    init_method: str = 'xavier',
    dropout: float = 0.0,
) -> nn.Sequential:
    """
    构建全连接神经网络，无缝支持 Sine 激活函数的注入与权重初始化适配。

    :param input_dim: 输入特征数
    :type input_dim: int
    :param output_dim: 输出特征数
    :type output_dim: int
    :param hidden_dims: 每层隐藏层神经元数，如 [64, 64, 32]
    :type hidden_dims: List[int]
    :param activation: 激活函数，支持 'tanh', 'relu', 'sigmoid', 'sin', 'leaky_relu', 'gelu'
    :type activation: str
    :param batch_norm: 是否添加 BatchNorm1d
    :type batch_norm: bool
    :param init_method: 权重初始化，'xavier' | 'kaiming' | 'normal'
    :type init_method: str
    :param dropout: Dropout 概率，0 表示不使用
    :type dropout: float
    :return: 构建好的模型
    :rtype: nn.Sequential
    """
    act_dict = {
        'tanh': nn.Tanh(),
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'sin': SineActivation(omega_0=30.0),
        'leaky_relu': nn.LeakyReLU(0.1),
        'gelu': nn.GELU(),
    }
    activation_func = act_dict.get(activation.lower(), nn.Tanh())
    layers = []
    prev_dim = input_dim
    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(activation_func)
        prev_dim = h_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    model = nn.Sequential(*layers)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if activation.lower() == 'sin':
                num_input = m.weight.size(1)
                m.weight.data.uniform_(-1.0 / num_input, 1.0 / num_input)
            else:
                if init_method == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif init_method == 'kaiming':
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu' if 'poly' in activation else 'tanh')
                elif init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)
    return model

class NetworkFactory:
    """网络工厂，从配置字典构建模型"""
    def __init__(self, config: Dict):
        """
    网络工厂，从配置字典构建模型。

    :param config: 配置字典，必须包含 input_dim, output_dim, hidden_dims
                   可选 activation, batch_norm, init_method, dropout
    :type config: Dict
    """
        self.config = config 
    def build(self) -> nn.Sequential:
        """
        根据配置构建网络。

        :return: 构建好的模型
        :rtype: nn.Sequential
        """
        return build_network(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            hidden_dims=self.config['hidden_dims'],
            activation=self.config.get('activation', 'tanh'),
            batch_norm=self.config.get('batch_norm', False),
            init_method=self.config.get('init_method', 'xavier'),
            dropout=self.config.get('dropout', 0.0),
        )

def build_model(
    coeffs: Union[Dict, List, float],
    source_term: Union[float, str, Callable] = 0.0,
    conditions: Optional[List[Dict]] = None,
    has_t: bool = False,
    dimension: int = 1,
    input_dim: Optional[int] = None,
    output_dim: int = 1,
    base_config: Optional[Dict] = None,
    verbose: bool = True,
) -> nn.Sequential:
    """
    一站式构建模型：自动分析复杂度，生成配置，构建网络。

    :param coeffs: 系数（dict 或 list 或 float）
    :type coeffs: Union[Dict, List, float]

    :param source_term: 源项
    :type source_term: Union[float, str, Callable]

    :param conditions: 边界条件列表
    :type conditions: Optional[List[Dict]]

    :param has_t: 是否含时间
    :type has_t: bool

    :param dimension: 空间维度
    :type dimension: int

    :param input_dim: 输入维度（若为 None，自动从 dimension 和 has_t 推断）
    :type input_dim: Optional[int]

    :param output_dim: 输出维度，默认为 1
    :type output_dim: int

    :param base_config: 基础网络配置
    :type base_config: Optional[Dict]

    :param verbose: 是否打印复杂度分数
    :type verbose: bool

    :return: 构建好的模型
    :rtype: nn.Sequential
    """
    if input_dim is None:
        input_dim = dimension + (1 if has_t else 0)
    analyzer = ComplexityAnalyzer()
    score = analyzer.compute_complexity(
        coeffs=coeffs,
        source_term=source_term,
        conditions=conditions,
        has_t=has_t,
        dimension=dimension,
    )
    if verbose:
        print(f"\n[NetworkFactory] --- 智能化物理场审计中 ---")
        print(f"[NetworkFactory] 最终完备复杂度总评分: {score:.2f} (维度: {dimension}D, 含时: {has_t})")
    generator = NetworkConfigGenerator(base_config=base_config)
    config = generator.generate_config(score, input_dim, output_dim)
    if verbose:
        print(f"[NetworkFactory] 推荐模型拓扑结构: {config['hidden_dims']}")
        print(f"[NetworkFactory] 适配激活函数类型: '{config['activation']}'")
    factory = NetworkFactory(config)
    return factory.build()

def suggest_network(
    coeffs: Union[Dict, List, float],
    source_term: Union[float, str, Callable] = 0.0,
    conditions: Optional[List[Dict]] = None,
    has_t: bool = False,
    dimension: int = 1,
    input_dim: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """
    仅建议网络配置，不实际构建模型。

    :param coeffs: 系数（dict 或 list 或 float）
    :type coeffs: Union[Dict, List, float]

    :param source_term: 源项
    :type source_term: Union[float, str, Callable]

    :param conditions: 边界条件列表
    :type conditions: Optional[List[Dict]]

    :param has_t: 是否含时间
    :type has_t: bool

    :param dimension: 空间维度
    :type dimension: int

    :param input_dim: 输入维度（若为 None，自动从 dimension 和 has_t 推断）
    :type input_dim: Optional[int]

    :param verbose: 是否打印配置信息
    :type verbose: bool

    :return: 建议的网络配置
    :rtype: Dict
    """
    if input_dim is None:
        input_dim = dimension + (1 if has_t else 0) 
    analyzer = ComplexityAnalyzer()
    score = analyzer.compute_complexity(
        coeffs=coeffs,
        source_term=source_term,
        conditions=conditions,
        has_t=has_t,
        dimension=dimension,
    )
    generator = NetworkConfigGenerator()
    config = generator.generate_config(score, input_dim)
    if verbose:
        print(f"[NetworkFactory] 独立配置建议: {config}")
    return config

if __name__ == "__main__":
    print("=== 用例 1：常规一维低阶常系数稳态 ODE (简单) ===")
    model_1 = build_model(
        coeffs=[1, 0, 1],
        source_term=0.0,
        conditions=[
            {'point': 0.0, 'value': 1.0, 'derivative': 0},
            {'point': 0.0, 'value': 0.0, 'derivative': 1}
        ],
        dimension=1,
        has_t=False
    )
    print("\n=== 用例 2：经典二维泊松方程 (引入多维扩张因子与导数边界惩罚) ===")
    model_2 = build_model(
        coeffs={'u_xx': 1.0, 'u_yy': 1.0},
        source_term='sin(pi*x)*sin(pi*y)',
        conditions=[
            {'side': 'left', 'type': 'dirichlet', 'value': 0.0},
            {'side': 'right', 'type': 'dirichlet', 'value': '0.0'},  
            {'side': 'bottom', 'type': 'neumann', 'value': '0'},   
            {'side': 'top', 'type': 'dirichlet', 'value': '0'}
        ],
        dimension=2,
        has_t=False
    )
    print("\n=== 用例 3：包含 50Hz 超高频振荡项的瞬态波动 PDE ===")
    model_3 = build_model(
        coeffs={'u_xx': 1.0, 'u_yy': 1.0},
        source_term='sin(50.0*pi*x) * cos(30.0*pi*y)',
        conditions=[
            {'side': 'left', 'type': 'neumann', 'value': '0'},
            {'side': 'right', 'type': 'neumann', 'value': '0'}
        ],
        dimension=2,
        has_t=True
    )
