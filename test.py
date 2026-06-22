# ==========================================
# FILE: function_factory.py
# ==========================================
import torch
import sympy

def parse_expression(expr, var_names=('x', 'y')):
    """
    动态解析表达式，支持任意维度变量 (e.g., 'x', 'y', 't', 'z')
    """
    if isinstance(expr, (int, float)):
        return lambda *args: expr
    if callable(expr):
        return expr
    if isinstance(expr, str):
        # 动态生成 Sympy 变量
        syms = sympy.symbols(' '.join(var_names))
        if isinstance(syms, sympy.Symbol):
            syms = (syms,)
        
        sp_expr = sympy.sympify(expr)
        func = sympy.lambdify(syms, sp_expr, modules='torch')
        return lambda *args: func(*args)
    
    raise TypeError(f"Unsupported expression type: {type(expr)}")

# ---------------------------------------------------------
# 1. 纯粹的物理数学定义层 (完全解耦 PyTorch 训练逻辑)
# ---------------------------------------------------------

class PDEConfig:
    terms = [
        {'deriv': 'u_xx', 'coeff_symbol': 'A'},
        {'deriv': 'u_xy', 'coeff_symbol': 'B'},
        {'deriv': 'u_yy', 'coeff_symbol': 'C'},
        {'deriv': 'u_x',  'coeff_symbol': 'D'},
        {'deriv': 'u_y',  'coeff_symbol': 'E'},
        {'deriv': 'u',    'coeff_symbol': 'F'},
        {'deriv': 'u_t',  'coeff_symbol': 'E_t'},  # 留出时间一阶导扩展
        {'deriv': 'u_tt', 'coeff_symbol': 'C_t'},  # 留出时间二阶导扩展
    ]
    source_symbol = 'G'

    def __init__(self, coeffs: dict, has_t: bool, var_names=None):
        self.coeffs = coeffs
        self.has_t = has_t
        # 动态推断变量维度，未来扩展3D只需传 ['x', 'y', 'z', 't']
        if var_names is None:
            self.var_names = ('x', 't') if has_t and 'C' not in coeffs else ('x', 'y') 
            if has_t and 'y' not in self.var_names and len(coeffs) > 7: # 兼容老UI的临时处理
                self.var_names = ('x', 'y', 't')
        else:
            self.var_names = tuple(var_names)

        self._parsed_coeffs = {}
        for sym, expr in coeffs.items():
            self._parsed_coeffs[sym] = parse_expression(expr, self.var_names)
       
        self.equation_type, self.equation_params = self._classify_equation()

    def _classify_equation(self):
        # (保持原有的分类逻辑不变)
        def get_const_val(sym):
            val = self.coeffs.get(sym)
            if val is None: return 0
            if isinstance(val, (int, float)): return val
            return None
            
        A, B, C = get_const_val('A'), get_const_val('B'), get_const_val('C')
        D, E, F = get_const_val('D'), get_const_val('E'), get_const_val('F')
        G = get_const_val('G')
        
        if A is not None and C is not None and self.has_t and C != 0:
            A /= C
            if A < 0:
                a = (-A) ** 0.5
                if all(s == 0 for s in [B, D, E, F]):
                    return "Wave1D", {"a": a} if G == 0 else "Wave1D_Source"
                elif all(s == 0 for s in [B, D, G]):
                    if E is not None and E != 0:
                        b = E / 2.0 / C
                        return "Wave1D_Damped" if F == 0 else "Telegraph"
        elif A is not None and E is not None and self.has_t and E != 0:
            A /= E
            if A < 0:
                a = (-A) ** 0.5
                if all(s == 0 for s in [B, C, D, F]):
                    return "Heat1D" if G == 0 else "Heat1D_Source"
        elif A is not None and C is not None and A == C and A != 0:
            if all(s == 0 for s in [B, D, E]) and F is not None:
                F /= A
                if F == 0:
                    return "Laplace2D" if G == 0 else "Poisson2D"
                elif F > 0:
                    a = F ** 0.5
                    return "Helmholtz", {"a": a}
        return "General", {}

    def get_coeff_func(self, symbol):
        return self._parsed_coeffs.get(symbol, lambda *args: 0.0)

    def get_source_func(self):
        return self.get_coeff_func(self.source_symbol)

    def get_required_derivatives(self):
        return [term['deriv'] for term in self.terms]


class ConditionConfig:
    # 扩展了 delta 用于表示时间导数（初始速度）
    terms = [
        {'term': 'u',   'coeff_symbol': 'alpha'},
        {'term': 'u_n', 'coeff_symbol': 'beta'},
        {'term': 'u_t', 'coeff_symbol': 'delta'}, # 解决问题：新增初始速度支持
    ]
    source_symbol = 'gamma'

    def __init__(self, coeffs: dict, var_names=None):
        self.coeffs = coeffs
        self.var_names = tuple(var_names) if var_names else ('x', 'y') # 默认后备
        self._parsed_coeffs = {}
        for sym, expr in coeffs.items():
            self._parsed_coeffs[sym] = parse_expression(expr, self.var_names)
        
        self.is_homogeneous = self._classify()

    def set_var_names(self, var_names):
        """支持由外部统一注入维度变量名"""
        self.var_names = tuple(var_names)
        for sym, expr in self.coeffs.items():
            self._parsed_coeffs[sym] = parse_expression(expr, self.var_names)

    def _classify(self):
        gamma = self.coeffs.get('gamma', 0)
        alpha, beta = self.coeffs.get('alpha', 0), self.coeffs.get('beta', 0)
        
        if gamma == 0: self.is_homogeneous = True
        else: self.is_homogeneous = False
        
        if alpha != 0 and beta == 0: self.condition_type = 'dirichlet'
        elif alpha != 0 and beta != 0: self.condition_type = 'robin'
        elif alpha == 0 and beta != 0: self.condition_type = 'neumann'
        else: self.condition_type = 'undefined'
        return self.is_homogeneous

    def get_coeff_func(self, symbol):
        return self._parsed_coeffs.get(symbol, lambda *args: 0.0)
    
    def get_alpha(self): return self.get_coeff_func('alpha')
    def get_beta(self): return self.get_coeff_func('beta')
    def get_delta(self): return self.get_coeff_func('delta') # 获取初始速度系数
    def get_gamma(self): return self.get_coeff_func('gamma')


class BoundaryCondition:
    """描述一个边界或初始条件的位置映射"""
    def __init__(self, location, condition_config: ConditionConfig, is_initial=False):
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
        try: self.value = float(parts[1].strip())
        except ValueError: raise ValueError("value must be a number")
        self.location_func = None

    def get_location_info(self):
        return {
            'type': self.location_type, 'axis': getattr(self, 'axis', None),
            'value': getattr(self, 'value', None), 'func': getattr(self, 'location_func', None)
        }

    def get_condition(self):
        return self.condition


# ---------------------------------------------------------
# 2. 行为解耦建造者 (Builders)
# ---------------------------------------------------------

class PINNLossBuilder:
    """
    解决功能耦合问题：专门负责将数学公式转换为 PyTorch 自动求导计算图。
    支持任意维度、任意组合形式的高阶导数解析。
    """
    def __init__(self, equation: PDEConfig, conditions: list[BoundaryCondition]):
        self.equation = equation
        self.conditions = conditions
        
        # 统一维度的命名空间映射
        self.var_names = equation.var_names
        for bc in self.conditions:
            bc.get_condition().set_var_names(self.var_names)
            
        self.var_idx = {name: idx for idx, name in enumerate(self.var_names)}

    def _compute_dynamic_derivatives(self, u, x_tensor, required_derivs):
        """底层扩展核心：基于变量名动态计算偏导数图"""
        derivs = {}
        if not required_derivs:
            return derivs
            
        # 1. 识别并提取所需的所有一阶导数基础
        first_order_needed = set()
        for d in required_derivs:
            if d == 'u': continue
            parts = d.split('_')
            if len(parts) == 2:
                for var in parts[1]:  # e.g., 'xy' -> 'x', 'y'
                    first_order_needed.add(var)

        if not first_order_needed:
            return derivs

        # 一次性计算出全部一阶梯度
        grad_u = torch.autograd.grad(u, x_tensor, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        
        for var in first_order_needed:
            if var in self.var_idx:
                idx = self.var_idx[var]
                derivs[f"u_{var}"] = grad_u[..., idx:idx+1]
            else:
                derivs[f"u_{var}"] = torch.zeros_like(u)

        # 2. 动态展开二阶导数 (如 u_xx, u_xy)
        for d in required_derivs:
            parts = d.split('_')
            if len(parts) == 2 and len(parts[1]) == 2:
                v1, v2 = parts[1][0], parts[1][1]
                u_v1 = derivs.get(f"u_{v1}")
                if u_v1 is not None and v2 in self.var_idx:
                    idx2 = self.var_idx[v2]
                    grad_u_v1 = torch.autograd.grad(u_v1, x_tensor, grad_outputs=torch.ones_like(u_v1), create_graph=True)[0]
                    derivs[d] = grad_u_v1[..., idx2:idx2+1]
                    
        return derivs

    def build(self):
        """返回构建好的损失函数，签名符合 Trainer 要求"""
        required_derivs = self.equation.get_required_derivatives()
        source_func = self.equation.get_source_func()

        def loss_fn(net, interior_pts: torch.Tensor, boundary_pts_list: list[torch.Tensor]):
            # --- 1. PDE 内部残差计算 ---
            x_int = interior_pts.clone().detach().requires_grad_(True)
            u_int = net(x_int)
            
            # 动态求导
            derivs = self._compute_dynamic_derivatives(u_int, x_int, required_derivs)
            
            residual = 0.0
            args_int = [x_int[..., i] for i in range(x_int.shape[-1])] # 动态解包维度
            
            for term in self.equation.terms:
                deriv_name = term['deriv']
                coeff_func = self.equation.get_coeff_func(term['coeff_symbol'])
                coeff_val = coeff_func(*args_int)
                
                if deriv_name == 'u': term_val = u_int
                else: term_val = derivs.get(deriv_name, torch.zeros_like(u_int))
                
                residual = residual + coeff_val * term_val
                
            source_val = source_func(*args_int)
            residual = residual - source_val
            pde_loss = torch.mean(residual ** 2)

            # --- 2. 边界/初始条件计算 ---
            bc_loss = 0.0
            for idx, condion in enumerate(self.conditions):
                pts = boundary_pts_list[idx]
                if pts is None or pts.numel() == 0: continue
                pts.requires_grad_(True)
                u_pred = net(pts)
                
                cond = condion.get_condition()
                args_bc = [pts[..., i] for i in range(pts.shape[-1])]
                
                alpha = cond.get_alpha()(*args_bc)
                beta = cond.get_beta()(*args_bc)
                delta = cond.get_delta()(*args_bc) # 处理初始速度
                gamma = cond.get_gamma()(*args_bc)

                grad_u = torch.autograd.grad(u_pred, pts, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
                
                # 智能识别法向导数/空间导数
                normal_deriv = 0.0
                if beta is not None:
                    info = condion.get_location_info()
                    if info['type'] == 'string' and info['axis'] in self.var_idx:
                        # 基于实际边界轴线提取导数（修正了原版硬编码[..., 0]的Bug）
                        axis_idx = self.var_idx[info['axis']]
                        normal_deriv = grad_u[..., axis_idx]
                    else:
                        normal_deriv = grad_u[..., 0] # 默认回退
                        
                # 处理初始速度/时间导数
                time_deriv = 0.0
                if delta is not None and 't' in self.var_idx:
                    t_idx = self.var_idx['t']
                    time_deriv = grad_u[..., t_idx]

                # α * u + β * u_n + δ * u_t = γ
                residual_bc = alpha * u_pred + beta * normal_deriv + delta * time_deriv - gamma
                bc_loss = bc_loss + torch.mean(residual_bc ** 2)

            total_loss = pde_loss + bc_loss
            return total_loss, pde_loss, bc_loss

        return loss_fn


class AnalyticalSolverBuilder:
    """
    专门给未来拓展留出的解析解处理层
    将 PDEConfig 转换为纯符号运算 (Sympy)
    """
    def __init__(self, equation: PDEConfig, conditions: list[BoundaryCondition]):
        self.equation = equation
        self.conditions = conditions
        
    def generate_analytical_solution(self):
        # TODO: 使用 sympy 进行解析求解的逻辑放入这里
        # 通过 self.equation.coeffs 可以提取解析用的系数
        return None

# ---------------------------------------------------------
# 3. 对外统一暴露接口 (向后兼容)
# ---------------------------------------------------------

def get_pde_loss(equation: PDEConfig, conditions: list[BoundaryCondition]):
    """
    对外接口保持不变：包装了新的 Builder 架构，确保外部 `trainer.py` 和 `gui.py` 零修改即可直接运行。
    """
    builder = PINNLossBuilder(equation, conditions)
    return builder.build()

def generate_analytical_solution(equation: PDEConfig, conditions: list[BoundaryCondition]):
    """向后兼容接口"""
    builder = AnalyticalSolverBuilder(equation, conditions)
    return builder.generate_analytical_solution()