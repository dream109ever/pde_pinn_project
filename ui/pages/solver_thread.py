from PyQt5.QtCore import QThread, pyqtSignal
from src.function_factory import solve_pde

class SolverThread(QThread):
    """
    基于 function_factory 计算引擎的后台求解线程
    将前端界面传入的 problem_config 转换为标准参数，直接调用 solve_pde 顶层接口
    """
    # 信号定义：状态日志、求解完成、错误提示
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, problem_config: dict):
        super().__init__()
        self.config = problem_config

    def run(self):
        try:
            self.log_signal.emit("正在解析方程配置，准备接入 function_factory 核心引擎...")

            # 1. 提取并规范化 GUI 传入的参数
            dimension = int(self.config.get('dimension', 1))
            has_t = bool(self.config.get('has_t', False))
            order = int(self.config.get('order', 2))
            
            # coeffs：1D 稳态为 list (如 [c0, c1, c2])，其余 PDE 为 dict (如 {"u_xx": 1.0, ...})
            coeffs = self.config.get('coeffs')
            if coeffs is None:
                coeffs = [1, 0, 1] if (dimension == 1 and not has_t) else {"u_xx": 1.0}

            source_term = str(self.config.get('source_term', '0'))
            
            # 边界/初始条件列表 (兼容 conditions 与 condition 键名)
            condition = self.config.get('conditions', self.config.get('condition', []))

            # 默认定义域处理
            domain = self.config.get('domain')
            if not domain:
                domain = {"x": [0.0, 1.0]}
                if dimension == 2:
                    domain["y"] = [0.0, 1.0]
                if has_t:
                    domain["t"] = [0.0, 1.0]

            self.log_signal.emit(f"参数解析完成: 维度={dimension}D | 含时={has_t} | 阶数={order}")
            self.log_signal.emit("正在调用 solve_pde 执行精确解/数值基准解计算...")

            # 2. 直接调用 function_factory 的顶层函数求解
            res_dict = solve_pde(
                dimension=dimension,
                order=order,
                has_t=has_t,
                coeffs=coeffs,
                source_term=source_term,
                domain=domain,
                condition=condition
            )

            # 3. 提取引擎返回结果
            exact_func = res_dict.get("exact_solution")
            exact_expr = res_dict.get("exact_expression")
            loss_funcs = res_dict.get("loss_functions", [])

            self.log_signal.emit("解结构计算完成，正在生成前端响应...")

            if exact_func is None and exact_expr is None:
                self.error_signal.emit("引擎未能获得精确解或数值基准解，请检查定解条件是否完整。")
                return

            # 确定自变量列表，便于后续 PlotWindow 绘图
            var_symbols = ['x']
            if dimension == 2:
                var_symbols.append('y')
            if has_t:
                var_symbols.append('t')

            # 打包返回结果
            result_data = {
                'exact_expr': exact_expr if exact_expr else "数值/级数近似基准解",
                'exact_solution': exact_func,    # 可直接调用的 Python/NumPy 函数
                'loss_functions': loss_funcs,    # [pde_loss, bc_loss, total_loss] 供 PINN 训练使用
                'var_symbols': var_symbols,
                'dimension': dimension,
                'has_t': has_t,
                'domain': domain,
                'config': self.config
            }

            self.finished_signal.emit(result_data)

        except Exception as e:
            self.error_signal.emit(f"核心引擎求解出错: {str(e)}")
