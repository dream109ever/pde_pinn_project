import torch
import sympy
import numpy as np
from scipy.integrate import dblquad

def parse_expression(expr, var_names=('x', 'y')):
    if isinstance(expr, (int, float)):
        return lambda *args: expr
    if callable(expr):
        return expr
    if isinstance(expr, str):
        syms = sympy.symbols(' '.join(var_names))
        if isinstance(syms, sympy.Symbol):
            syms = (syms,)
        sp_expr = sympy.sympify(expr)
        modules = {
            'sin': torch.sin,
            'cos': torch.cos,
            'tan': torch.tan,
            'exp': torch.exp,
            'log': torch.log,
            'sqrt': torch.sqrt,
            'pi': torch.pi,
            'abs': torch.abs,
        }
        func = sympy.lambdify(syms, sp_expr, modules=modules)
        return lambda *args: func(*args)
    raise TypeError(f"Unsupported expression type: {type(expr)}")
class PDEConfig:
    terms = [
        {'deriv': 'u_xx', 'coeff_symbol': 'A'},
        {'deriv': 'u_xy', 'coeff_symbol': 'B'},
        {'deriv': 'u_yy', 'coeff_symbol': 'C'},
        {'deriv': 'u_x',  'coeff_symbol': 'D'},
        {'deriv': 'u_y',  'coeff_symbol': 'E'},
        {'deriv': 'u',    'coeff_symbol': 'F'},
    ]
    source_symbol = 'G'
    def __init__(self, coeffs: dict, has_t: bool):
        self.coeffs = coeffs
        self.has_t = has_t
        self._parsed_coeffs = {}
        for sym, expr in coeffs.items():
            var_names = ('x', 't') if self.has_t else ('x', 'y')
            self._parsed_coeffs[sym] = parse_expression(expr, var_names=var_names)
        self.equation_type, self.equation_params = self._classify_equation()
    def _classify_equation(self):
        def get_const_val(sym):
            val = self.coeffs.get(sym)
            if val is None:
                return 0
            if isinstance(val, (int, float)):
                return val
            return None
        A, B, C = get_const_val('A'), get_const_val('B'), get_const_val('C')
        D, E, F = get_const_val('D'), get_const_val('E'), get_const_val('F')
        G = get_const_val('G')
        if A is not None and C is not None and self.has_t and C != 0:
            A /= C
            if A < 0:
                a = (-A) ** 0.5
                if all(s == 0 for s in [B, D, E, F]):
                    if G == 0:
                        return "Wave1D", {"a": a}                                           # 一维波动 (u_tt - a^2 u_xx = 0)
                    elif G != 0:
                        return "Wave1D_Source", {"a": a, "source": "G(x, t)", "scale": C}   # 一维有源波动 (u_tt - a^2 u_xx = G)
                elif all(s == 0 for s in [B, D, G]):
                    if E is not None and E != 0:
                        b = E / 2.0 / C
                        if F == 0:
                            return "Wave1D_Damped", {"a": a, "b": b}                        # 一维阻尼波动 (u_tt + 2b u_t - a^2 u_xx = 0)
                        elif F is not None and F != 0:
                            c = F / C
                            return "Telegraph", {"a": a, "b": b, "c": c}                    # 电报方程 (u_tt + 2b u_t - a^2 u_xx + c u = 0)
        elif A is not None and E is not None and self.has_t and E != 0:
            A /= E
            if A < 0:
                a = (-A) ** 0.5
                if all(s == 0 for s in [B, C, D, F]):
                    if G == 0:
                        return "Heat1D", {"a": a}                                           # 一维热传导 (u_t - a^2 u_xx = 0)
                    elif G != 0:
                        return "Heat1D_Source", {"a": a, "source": "G(x, t)", "scale": E}   # 一维有源热传导 (u_t - a^2 u_xx = G)
        elif A is not None and C is not None and A == C and A != 0:
            if all(s == 0 for s in [B, D, E]) and F is not None:
                F /= A
                if F == 0:
                    if G == 0:
                        return "Laplace2D", {}                                              # 二维拉普拉斯 (u_xx + u_yy = 0)
                    elif G != 0:
                        return "Poisson2D", {"source": "G(x, y)", "scale": A}               # 二维泊松 (u_xx + u_yy = G)
                elif F > 0:
                    a = F ** 0.5
                    return "Helmholtz", {"a": a}                                            # 亥姆霍兹 (u_xx + u_yy + a^2 u = 0)
        return "General", {}                                                                # 一般方程
    def get_coeff_func(self, symbol):
        return self._parsed_coeffs.get(symbol, lambda x, y: 0.0)
    def get_source_func(self):
        return self.get_coeff_func(self.source_symbol)
    def get_required_derivatives(self):
        return [term['deriv'] for term in self.terms]

class ConditionConfig:
    terms = [
        {'term': 'u',   'coeff_symbol': 'alpha'},
        {'term': 'u_n', 'coeff_symbol': 'beta'},
    ]
    source_symbol = 'gamma'
    def __init__(self, coeffs: dict):
        self.coeffs = coeffs
        self._parsed_coeffs = {}
        for sym, expr in coeffs.items():
            self._parsed_coeffs[sym] = parse_expression(expr)
        self.is_homogeneous = None
        self.condition_type = self._classify()
    def _classify(self):
        def get_const_val(sym):
            val = self.coeffs.get(sym)
            if val is None:
                return 0
            if isinstance(val, (int, float)):
                return val
            return None
        alpha, beta, gamma = get_const_val('alpha'), get_const_val('beta'), get_const_val('gamma')
        if gamma is not None and gamma == 0:
            self.is_homogeneous = True
        else:
            self.is_homogeneous = False
        if alpha is not None and alpha != 0:
            if beta is not None and beta == 0:
                return 'dirichlet'
            elif beta is not None and beta != 0:
                return 'robin'
        elif alpha is not None and alpha == 0:
            if beta is not None and beta != 0:
                return 'neumann'
        return 'undefined'
    def get_coeff_func(self, symbol):
        return self._parsed_coeffs.get(symbol, lambda x, y: 0.0)
    def get_alpha(self):
        return self.get_coeff_func('alpha')
    def get_beta(self):
        return self.get_coeff_func('beta')
    def get_gamma(self):
        return self.get_coeff_func('gamma')

class BoundaryCondition:
    """
    描述一个边界或初始条件，存储和解析位置信息。
    """
    def __init__(self, location, condition_config : ConditionConfig, is_initial=False):
        """
        参数:
            location: 描述边界位置，支持以下形式：
                      - 字符串: 'x=0', 'x=1', 'y=0', 'y=1', 't=0' 等
                      - 可调用函数: 接受坐标张量 (N, dim)，返回布尔掩码 (N,)
            condition_config: ConditionConfig 实例
            is_initial: 布尔值，True 表示初始条件（时间方向），False 表示边界条件（空间方向）
        """
        self.condition = condition_config
        self.is_initial = is_initial
        self.location = location
        if isinstance(location, str):
            self._parse_location_string(location)
            self.location_type = 'string'
        elif callable(location):
            self.location_func = location
            self.location_type = 'function'
            self.axis = None
            self.value = None
        else:
            raise TypeError("location must be str or callable")
    def _parse_location_string(self, loc_str):
        parts = loc_str.strip().split('=')
        if len(parts) != 2:
            raise ValueError("location string must be in format 'axis=value'")
        self.axis = parts[0].strip()
        try:
            self.value = float(parts[1].strip())
        except ValueError:
            raise ValueError("value must be a number")
        self.location_func = None
    def get_location_info(self):
        return {
            'type': self.location_type,
            'axis': getattr(self, 'axis', None),
            'value': getattr(self, 'value', None),
            'func': getattr(self, 'location_func', None)
        }
    def get_condition(self):
        """返回 ConditionConfig 实例"""
        return self.condition
    def __repr__(self):
        return (f"BoundaryCondition(location={self.location_type}, "
                f"axis={getattr(self, 'axis', None)}, value={getattr(self, 'value', None)}, "
                f"is_initial={self.is_initial})")

def get_pde_loss(equation: PDEConfig, conditions: list[BoundaryCondition]):
    """
    返回一个损失函数，该函数可以计算给定网络和采样点的总损失。

    参数:
        equation: PDEConfig 对象，包含方程的系数和所需导数。
        conditions: BoundaryCondition 列表，每个元素描述一类边界/初始条件。

    返回:
        loss_fn: 可调用对象，签名 loss_fn(net, interior_pts, boundary_pts_list) -> torch.Tensor
                 boundary_pts_list 是与 conditions 等长的张量列表，每个张量是该类条件对应的采样点。
    """
    required_derivs = equation.get_required_derivatives()
    source_func = equation.get_source_func()
    # 预存储每个条件的系数函数和类型
    bc_info = []
    for condion in conditions:
        cond = condion.get_condition()
        bc_info.append({
            'type': cond.condition_type,
            'alpha': cond.get_alpha(),
            'beta': cond.get_beta(),
            'gamma': cond.get_gamma()
        })
    def loss_fn(net, interior_pts : torch.Tensor, boundary_pts_list : list[torch.Tensor]):
        # 1. PDE 残差损失
        x_int = interior_pts.clone().detach().requires_grad_(True)
        u_int = net(x_int)
        derivs = {}
        grad_u = torch.autograd.grad(u_int, x_int, grad_outputs=torch.ones_like(u_int), create_graph=True)[0]
        if 'u_x' in required_derivs:
            derivs['u_x'] = grad_u[..., 0:1]
        if 'u_y' in required_derivs:
            derivs['u_y'] = grad_u[..., 1:2] if grad_u.shape[-1] > 1 else torch.zeros_like(grad_u[..., 0:1])
        if 'u_xx' in required_derivs:
            u_x = derivs['u_x']
            u_xx = torch.autograd.grad(u_x, x_int, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][..., 0:1]
            derivs['u_xx'] = u_xx
        if 'u_yy' in required_derivs:
            u_y = derivs['u_y']
            u_yy = torch.autograd.grad(u_y, x_int, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][..., 1:2]
            derivs['u_yy'] = u_yy
        if 'u_xy' in required_derivs:
            u_x = derivs['u_x']
            u_xy = torch.autograd.grad(u_x, x_int, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][..., 1:2]
            derivs['u_xy'] = u_xy
        residual = 0.0
        for term in equation.terms:
            deriv = term['deriv']
            coeff_symbol = term['coeff_symbol']
            coeff_func = equation.get_coeff_func(coeff_symbol)
            coeff_val = coeff_func(x_int[..., 0], x_int[..., 1])
            if deriv == 'u':
                term_val = u_int
            else:
                term_val = derivs.get(deriv, torch.zeros_like(u_int))
            residual = residual + coeff_val * term_val
        source_val = source_func(x_int[..., 0], x_int[..., 1])
        residual = residual - source_val
        pde_loss = torch.mean(residual ** 2)
        # 2. 边界/初始条件损失
        bc_loss = 0.0
        for idx, cond_info in enumerate(bc_info):
            pts = boundary_pts_list[idx]
            if pts is None or pts.numel() == 0:
                continue
            pts.requires_grad_(True)
            u_pred = net(pts)
            alpha = cond_info['alpha'](pts[..., 0], pts[..., 1])
            beta = cond_info['beta'](pts[..., 0], pts[..., 1])
            gamma = cond_info['gamma'](pts[..., 0], pts[..., 1])
            grad_u = torch.autograd.grad(u_pred, pts, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
            normal_deriv = grad_u[..., 0]
            residual = alpha * u_pred + beta * normal_deriv - gamma
            bc_loss = bc_loss + torch.mean(residual ** 2)
        total_loss = pde_loss + bc_loss
        return total_loss, pde_loss, bc_loss
    return loss_fn

def generate_analytical_solution(equation: PDEConfig, conditions: list[BoundaryCondition]):
    """
    返回解析解函数（若可解），否则返回 None。
    目前仅支持：二维拉普拉斯/泊松/亥姆霍兹，矩形域 [0,1]x[0,1] 且齐次 Dirichlet 边界条件。
    返回的函数接受坐标点 (N,2) 返回解张量。
    """
    # 1. 检查是否为支持的类型
    eq_type = equation.equation_type
    if eq_type not in ['Laplace2D', 'Poisson2D', 'Helmholtz']:
        return None
    # 2. 检查边界条件是否为齐次 Dirichlet（假设边界在 x=0, x=1, y=0, y=1 均为 u=0）
    #    这里简化为检查所有条件的 condition_type 是否为 'dirichlet' 且 gamma=0, alpha=1, beta=0
    #    我们只检查条件数量是否为4（四条边），且都满足。
    if len(conditions) != 4:
        return None
    for bc in conditions:
        cond = bc.get_condition()
        if cond.condition_type != 'dirichlet':
            return None
        # 获取 gamma 的数值（如果不是常数则无法判断）
        gamma_func = cond.get_gamma()
        # 尝试在原点求值看是否为0（简化）
        try:
            gamma_val = gamma_func(0.0, 0.0)
        except:
            gamma_val = None
        if gamma_val != 0:
            return None
        # 检查 alpha 是否为 1，beta 是否为 0（但 ConditionConfig 的 classify 已经要求 dirichlet 时 alpha!=0, beta=0）
        # 这里可以忽略，因为我们信任 classify
    # 3. 获取源项函数（用于泊松和亥姆霍兹）
    source_func = equation.get_source_func()
    # 定义域 [0,1] x [0,1]
    x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0
    Lx = x_max - x_min
    Ly = y_max - y_min
    # 4. 根据方程类型计算傅里叶系数
    N_terms = 6  # 取前6项
    coeffs = {}
    # 定义基函数 sin(m*pi*(x-x_min)/Lx) * sin(n*pi*(y-y_min)/Ly)
    # 归一化因子：对于归一化的傅里叶级数，系数需要乘以 2/Lx * 2/Ly
    # 但为了简化，我们直接使用标准的 sin 级数，并利用正交性。
    def f_func(x, y):
        # 将 torch 张量转换为 numpy，以便 scipy 积分
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        # 如果源项函数接受 torch 张量，我们转换为 torch 后调用再转回 numpy
        # 但为了稳健，我们直接使用原始函数，假设它接受 numpy 数组
        if hasattr(source_func, '__call__'):
            # 尝试用 torch 张量计算，然后转为 numpy
            try:
                x_t = torch.tensor(x, dtype=torch.float32)
                y_t = torch.tensor(y, dtype=torch.float32)
                val = source_func(x_t, y_t)
                return val.detach().cpu().numpy()
            except:
                # 如果失败，尝试直接用 numpy 调用
                return source_func(x, y)
        else:
            # 常数源项
            return source_func
    # 计算傅里叶系数
    for m in range(1, N_terms+1):
        for n in range(1, N_terms+1):
            # 计算系数 c_{mn} = (2/Lx)*(2/Ly) * ∫∫ f(x,y) sin(mπx/Lx) sin(nπy/Ly) dx dy
            # 我们使用数值积分
            def integrand(x, y):
                return f_func(x, y) * np.sin(m * np.pi * (x - x_min) / Lx) * np.sin(n * np.pi * (y - y_min) / Ly)
            # 进行二重积分
            try:
                # 注意：dblquad 返回 (积分值, 误差估计)
                integral, _ = dblquad(integrand, x_min, x_max, lambda x: y_min, lambda x: y_max)
            except Exception as e:
                print(f"积分失败 (m={m}, n={n}): {e}")
                integral = 0.0
            # 归一化因子 (2/Lx)*(2/Ly)
            coeff = integral * (2.0 / Lx) * (2.0 / Ly)
            # 对于泊松方程：c_{mn} = - coeff / (π² (m²/Lx² + n²/Ly²))
            # 对于拉普拉斯：源项为0，所以系数为0，解为0
            if eq_type == 'Poisson2D':
                coeff = -coeff / (np.pi**2 * (m**2 / Lx**2 + n**2 / Ly**2))
            elif eq_type == 'Helmholtz':
                # 亥姆霍兹：∇²u + k²u = f，这里 k² 从 equation_params 获取
                # 我们需要从 equation_params 获取 k（或 a）
                params = equation.equation_params
                k = params.get('a', 1.0)  # 假设默认 k=1
                denom = np.pi**2 * (m**2 / Lx**2 + n**2 / Ly**2) - k**2
                coeff = -coeff / denom
            else:  # Laplace2D
                coeff = 0.0  # 源项为0，解为0
            if abs(coeff) > 1e-12:
                coeffs[(m, n)] = coeff
    # 5. 构造解析解函数
    def analytical_solution(points):
        """
        points: torch.Tensor 形状 (N, 2) 或 (N, 3) 但只取前两维 (x,y)
        """
        if isinstance(points, torch.Tensor):
            x = points[..., 0].detach().cpu().numpy()
            y = points[..., 1].detach().cpu().numpy()
        else:
            x = points[:, 0]
            y = points[:, 1]
        u = np.zeros_like(x)
        for (m, n), coeff in coeffs.items():
            u += coeff * np.sin(m * np.pi * (x - x_min) / Lx) * np.sin(n * np.pi * (y - y_min) / Ly)
        return torch.tensor(u, dtype=torch.float32).reshape(-1, 1)
    return analytical_solution


# # PDE 配置（泊松方程：u_xx + u_yy = sin(pi*x)*sin(pi*y)）
# pde = PDEConfig({
#     'A': 1.0,
#     'C': 1.0,
#     'G': lambda x, y: torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
# })
# # 边界条件：Dirichlet u=0 在四条边上
# bc1 = BoundaryCondition('x=0', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}), is_initial=False)
# bc2 = BoundaryCondition('x=1', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}), is_initial=False)
# bc3 = BoundaryCondition('y=0', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}), is_initial=False)
# bc4 = BoundaryCondition('y=1', ConditionConfig({'alpha':1, 'beta':0, 'gamma':0}), is_initial=False)
# conditions = [bc1, bc2, bc3, bc4]
# loss_fn = get_pde_loss(pde, conditions)
