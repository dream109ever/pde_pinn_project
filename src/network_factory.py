import re
import torch
import inspect
import sympy as sp
import torch.nn as nn
from typing import Union, Callable, Dict, List, Optional

# ============================================================================
# 0. 基础自定义激活函数
# ============================================================================
class SineActivation(nn.Module):
    """
    正弦激活函数（SIREN 架构的核心）。
    在 PINN 中求解高频振荡、强非线性或具有陡峭梯度的物理问题时，
    正弦激活函数能够有效打破传统 Tanh/ReLU 的“光谱偏差”现象，加速高频分量收敛。
    """
    def __init__(self, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)
# ============================================================================
# 1. 复杂度分析器
# ============================================================================
class ComplexityAnalyzer:
    """
    分析 PDE 方程及其边界条件的复杂度，生成评分用于自动选择网络规模。
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
        """评估单个表达式的复杂度分数"""
        # 统一规范化：尝试将字符串数字直接转化为数字，消除格式双标
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
        """分析字符串表达式的复杂度，包含高频因子提取机制"""
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
        """分析 SymPy 表达式的复杂度"""
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
        """分析自定义可调用对象的复杂度"""
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

        参数:
            coeffs: 系数
                一维 ODE 使用 list，如 [2, 2, 1] 表示 u'' + 2u' + 2u。

                PDE 使用 dict，如 {'u_xx': 1.0, 'u_yy': 1.0}。
            
            source_term: 源项，可为常数、字符串表达式或可调用对象
            conditions: 边界条件列表，每个条件为 dict

                格式: {'location': 'left', 'type': 'dirichlet', 'value': 0.0}

                或对于 ODE: {'point': 0.0, 'value': 0.0, 'derivative': 0}
            
            has_t: 是否含时间变量
            dimension: 空间维度，1 或 2
        
        返回:
            float: 复杂度分数
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
                # 识别导数边界条件 (Neumann / Robin / 指定了导数阶数)
                c_type = str(cond.get('type', '')).lower()
                if 'neumann' in c_type or 'robin' in c_type or cond.get('derivative', 0) > 0:
                    total_score += self.weights.get('derivative_bc_penalty', 8.0)
        return total_score
# ============================================================================
# 2. 网络配置生成器
# ============================================================================
class NetworkConfigGenerator:
    """
    根据复杂度分数，智能化生成与之匹配的网络层级与激活函数配置。
    """
    def __init__(self, base_config: Optional[Dict] = None, mapping: Optional[Callable[[float], Dict]] = None,):
        """
        参数:
            base_config: 基础配置，如 {'batch_norm': False, 'init_method': 'xavier'}

            mapping: 可调用对象，接收分数返回配置字典
        """
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
        
        参数:
            score: 复杂度分数
            input_dim: 输入维度
            output_dim: 输出维度
        
        返回:
            Dict: 完整网络配置
        """
        config = self.mapping(score)
        config.update(self.base_config)
        config['input_dim'] = input_dim
        config['output_dim'] = output_dim
        return config
# ============================================================================
# 3. 网络构建器
# ============================================================================
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
    
    参数:
        input_dim: 输入特征数
        output_dim: 输出特征数
        hidden_dims: 每层隐藏层神经元数，如 [64, 64, 32]
        activation: 激活函数，支持 'tanh', 'relu', 'sigmoid', 'sin', 'leaky_relu'
        batch_norm: 是否添加 BatchNorm1d
        init_method: 权重初始化 'xavier' | 'kaiming' | 'normal'
        dropout: Dropout 概率，0 表示不使用
    
    返回:
        nn.Sequential: 构建好的模型
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
    # 针对不同激活函数采用配套的最优初始化策略
    def init_weights(m):
        if isinstance(m, nn.Linear):
            if activation.lower() == 'sin':
                # SIREN 专属初始化原则：第一层与后续层的分布宽度需经专门缩放
                # 简单处理采用较窄的均匀分布，防止正弦激活进入饱和死区
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
        参数:
            config: 必须包含 input_dim, output_dim, hidden_dims
                  可选 activation, batch_norm, init_method, dropout
        """
        self.config = config 
    def build(self) -> nn.Sequential:
        return build_network(
            input_dim=self.config['input_dim'],
            output_dim=self.config['output_dim'],
            hidden_dims=self.config['hidden_dims'],
            activation=self.config.get('activation', 'tanh'),
            batch_norm=self.config.get('batch_norm', False),
            init_method=self.config.get('init_method', 'xavier'),
            dropout=self.config.get('dropout', 0.0),
        )
# ============================================================================
# 4. 高层便捷函数
# ============================================================================
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
    
    参数:
        coeffs: 系数（dict 或 list 或 float）
        source_term: 源项
        conditions: 边界条件列表
        has_t: 是否含时间
        dimension: 空间维度
        input_dim: 输入维度（若为 None，自动从 dimension 和 has_t 推断）
        output_dim: 输出维度，默认为 1
        base_config: 基础网络配置
        verbose: 是否打印复杂度分数
    
    返回:
        nn.Sequential: 构建好的模型
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
    
    返回:
        Dict: 建议的网络配置
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
# ============================================================================
# 5. 使用示例
# ============================================================================
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
        source_term='sin(50.0*pi*x) * cos(30.0*pi*y)', # 自动触发正则高频提取引擎
        conditions=[
            {'side': 'left', 'type': 'neumann', 'value': '0'},
            {'side': 'right', 'type': 'neumann', 'value': '0'}
        ],
        dimension=2,
        has_t=True # 附加时间维
    )
