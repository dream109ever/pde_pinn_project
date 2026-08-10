# ui/pages/plot_window_page.py
"""
绘图窗口模块。

提供 PDE/ODE 解函数的可视化弹窗，对接 plotting_qt 控件库，
支持 1D/2D 稳态与含时问题的图形展示，以及任意常数 C1, C2 的滑块调节。
"""
import re
import numpy as np
import sympy as sp
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QSlider, QGroupBox
from PyQt5.QtCore import Qt
import matplotlib
import matplotlib.pyplot as plt
from ui.theme_manager import ThemeManager
from src.plotting_qt import Steady1DPlotWidget, Transient1DPlotWidget, Steady2DPlotWidget, Transient2DPlotWidget
from .base_widgets import BaseDialog
matplotlib.use('Qt5Agg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

def parse_css_color_for_mpl(css_color: str):
    """
    将 QSS/CSS 格式的颜色转换为 Matplotlib 支持的格式。

    :param css_color: CSS 颜色字符串，如 'rgba(255, 255, 255, 0.9)' 或 '#FFFFFF'
    :type css_color: str
    :return: Matplotlib 可识别的颜色格式（元组或十六进制字符串）
    :rtype: tuple or str
    """
    if not css_color or not isinstance(css_color, str):
        return '#888888'
    css_color = css_color.strip()
    rgba_match = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)', css_color)
    if rgba_match:
        r = int(rgba_match.group(1)) / 255.0
        g = int(rgba_match.group(2)) / 255.0
        b = int(rgba_match.group(3)) / 255.0
        a = float(rgba_match.group(4)) if rgba_match.group(4) is not None else 1.0
        return (r, g, b, a)
    return css_color

class PlotWindow(BaseDialog):
    """
    可视化弹窗：对接 plotting_qt 控件库，支持 C1, C2 任意常数拖动与主题切换。

    :param result_data: 求解结果数据字典
    :type result_data: dict
    :param parent: 父控件
    :type parent: Optional[QWidget]
    """
    def __init__(self, result_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PDE / ODE 解函数可视化")
        self.resize(850, 680)
        self.result_data = result_data
        self.sliders = {}
        self.plot_widget = None
        self.mode_combo = None
        dimension = result_data.get('dimension', 1)
        has_t = result_data.get('has_t', False)
        if dimension == 1 and not has_t:
            c_symbols = self.result_data.get('c_symbols', [])
            if not c_symbols:
                raw_rhs = self.result_data.get('raw_rhs')
                if raw_rhs is not None:
                    try:
                        if isinstance(raw_rhs, str): expr = sp.sympify(raw_rhs)
                        elif isinstance(raw_rhs, sp.Expr): expr = raw_rhs
                        else: expr = None
                        if expr is not None and isinstance(expr, sp.Expr):
                            var_syms = {sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('t')}
                            c_syms = [str(s) for s in expr.free_symbols if s not in var_syms]
                            c_syms.sort(key=lambda s: s)
                            if c_syms: self.result_data['c_symbols'] = c_syms
                    except Exception as e:
                        print(f"[PlotWindow] 提取 C 符号失败: {e}")
        self.init_ui()
        self._apply_mpl_theme()
    def init_ui(self):
        """初始化用户界面布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)
        dimension = self.result_data.get('dimension', 1)
        has_t = self.result_data.get('has_t', False)
        # 1. 选择对应的 plotting_qt 绘图控件
        if dimension == 1 and not has_t:
            self.plot_widget = Steady1DPlotWidget(self)
        elif dimension == 1 and has_t:
            self.plot_widget = Transient1DPlotWidget(self)
        elif dimension == 2 and not has_t:
            self.plot_widget = Steady2DPlotWidget(self)
        else:
            self.plot_widget = Transient2DPlotWidget(self)
        main_layout.addWidget(self.plot_widget, stretch=1)
        # 2. 构建任意常数 (C1, C2...) 控制滑动条卡片
        c_symbols = self.result_data.get('c_symbols', [])
        if c_symbols:
            self.slider_group = QGroupBox("调整解中的任意常数 (C1, C2...)")
            slider_box = QVBoxLayout(self.slider_group)
            slider_box.setContentsMargins(10, 15, 10, 10)
            for sym_name in c_symbols:
                h_layout = QHBoxLayout()
                sym_lbl = QLabel(f"{sym_name}:")
                sym_lbl.setFixedWidth(30)
                val_lbl = QLabel("0.0")
                val_lbl.setFixedWidth(40)
                val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                slider = QSlider(Qt.Horizontal)
                slider.setRange(-100, 100)
                slider.setValue(0)
                slider.valueChanged.connect(lambda val, s_name=sym_name, l_lbl=val_lbl: self.on_slider_change(s_name, val, l_lbl))
                h_layout.addWidget(sym_lbl)
                h_layout.addWidget(slider, stretch=1)
                h_layout.addWidget(val_lbl)
                slider_box.addLayout(h_layout)
                self.sliders[sym_name] = slider
            main_layout.addWidget(self.slider_group)
        self.refresh_plot()
    def get_exact_callable(self):
        """
        将 SymPy 表达式/等式/字符串转换为 NumPy/PyTorch 万能兼容的 Python 函数。

        :return: 可调用的函数，接受坐标参数并返回数值解
        :rtype: Optional[Callable]
        """
        dimension = self.result_data.get('dimension', 1)
        has_t = self.result_data.get('has_t', False)
        if has_t or dimension == 2:
            exact_sol = self.result_data.get('exact_solution')
            if callable(exact_sol):
                try:
                    if dimension == 1:
                        test_result = exact_sol(np.array([0.5]), np.array([0.1]))
                    elif has_t and dimension == 2:
                        test_result = exact_sol(np.array([0.5]), np.array([0.5]), np.array([0.1]))
                    else:
                        test_result = exact_sol(np.array([0.5]), np.array([0.5]))
                    if isinstance(test_result, np.ndarray):
                        return exact_sol
                except Exception as e:
                    print(f"[PlotWindow] 含时 exact_solution 测试失败: {e}")
        # 1. 多 Key 兼容获取解析解表达式
        raw_rhs = (
            self.result_data.get('raw_rhs') or 
            self.result_data.get('exact_solution') or 
            self.result_data.get('rhs') or 
            self.result_data.get('solution')
        )
        if raw_rhs is None:
            print("[PlotWindow 提示] result_data 中未找到任何解析解表达式 Key！")
            return None
        def _to_float_array(val):
            """将各种类型的值转换为浮点数数组。"""
            if val is None:
                return None
            try:
                return np.asarray(val, dtype=float)
            except (TypeError, ValueError):
                pass
            arr = np.asarray(val)
            def _convert_item(item):
                if hasattr(item, 'evalf'):
                    try:
                        item = item.evalf()
                    except Exception:
                        pass
                try:
                    return float(item)
                except (TypeError, ValueError):
                    return np.nan
            return np.vectorize(_convert_item, otypes=[float])(arr)
        is_sympy_obj = isinstance(raw_rhs, (sp.Basic, sp.core.function.FunctionClass))
        if callable(raw_rhs) and not is_sympy_obj:
            def callable_wrapper(*args):
                new_args = []
                for arg in args:
                    if hasattr(arg, 'detach'):
                        arg = arg.detach().cpu().numpy()
                    new_args.append(arg)
                try:
                    val = raw_rhs(*new_args)
                except TypeError:
                    if len(new_args) == 1 and hasattr(new_args[0], 'shape') and new_args[0].ndim >= 2:
                        pts = new_args[0]
                        unpacked = [pts[..., i] for i in range(pts.shape[-1])]
                        val = raw_rhs(*unpacked)
                    else:
                        raise
                return _to_float_array(val)
            return callable_wrapper
        # 2. 如果是 SymPy 等式 Eq(u(x), expr)，自动提取右半部分 expr
        if isinstance(raw_rhs, sp.Eq):
            raw_rhs = raw_rhs.rhs
        # 3. 如果是字符串，自动转换为 SymPy 表达式
        if isinstance(raw_rhs, str):
            try:
                raw_rhs = sp.sympify(raw_rhs)
            except Exception as e:
                print(f"[PlotWindow 错误] 字符串解析为 SymPy 表达式失败: {e}")
                return None
        # 4. 替换 C1, C2 等待定常数（兼容 Symbol 和 字符串 key）
        subs_dict = {}
        for sym_name, slider in self.sliders.items():
            val = slider.value() / 10.0
            subs_dict[sp.Symbol(sym_name)] = val
            subs_dict[str(sym_name)] = val
        try:
            expr_curr = raw_rhs.subs(subs_dict)
        except Exception as e:
            print(f"[PlotWindow 错误] 常数替换失败: {e}")
            return None
        dimension = self.result_data.get('dimension', 1)
        has_t = self.result_data.get('has_t', False)
        x_sym, y_sym, t_sym = sp.Symbol('x'), sp.Symbol('y'), sp.Symbol('t')
        allowed_syms = {x_sym}
        if dimension == 2: 
            allowed_syms.add(y_sym)
        if has_t: 
            allowed_syms.add(t_sym)
        if hasattr(expr_curr, 'free_symbols'):
            extra_syms = expr_curr.free_symbols - allowed_syms
            if extra_syms:
                print(f"[PlotWindow 提示] 检测到未绑定的未定义自由符号: {extra_syms}，已自动默认替换为 1.0")
                expr_curr = expr_curr.subs({s: 1.0 for s in extra_syms})
        try:
            # 5. 构造万能包装函数：同时兼容 多个数组输入 和 单个 (N, D) Tensor/pts 输入
            if dimension == 1 and not has_t:
                fn = sp.lambdify(x_sym, expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    arg0 = args[0]
                    if hasattr(arg0, 'detach'): arg0 = arg0.detach().cpu().numpy()
                    val = fn(arg0)
                    return np.full_like(arg0, val) if np.isscalar(val) else val
                return wrapper
            elif dimension == 1 and has_t:
                fn = sp.lambdify((x_sym, t_sym), expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    if len(args) == 1:
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, t = pts[..., 0], pts[..., 1]
                    else:
                        x = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        t = args[1].detach().cpu().numpy() if hasattr(args[1], 'detach') else args[1]
                    val = fn(x, t)
                    return np.full_like(x, val) if np.isscalar(val) else val
                return wrapper
            elif dimension == 2 and not has_t:
                fn = sp.lambdify((x_sym, y_sym), expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    if len(args) == 1:
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, y = pts[..., 0], pts[..., 1]
                    else:
                        x = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        y = args[1].detach().cpu().numpy() if hasattr(args[1], 'detach') else args[1]
                    val = fn(x, y)
                    return np.full_like(x + y, val) if np.isscalar(val) else val
                return wrapper
            else:
                fn = sp.lambdify((x_sym, y_sym, t_sym), expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    if len(args) == 1:
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, y, t = pts[..., 0], pts[..., 1], pts[..., 2]
                    else:
                        x = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        y = args[1].detach().cpu().numpy() if hasattr(args[1], 'detach') else args[1]
                        t = args[2].detach().cpu().numpy() if hasattr(args[2], 'detach') else args[2]
                    val = fn(x, y, t)
                    return np.full_like(x + y, val) if np.isscalar(val) else val
                return wrapper
        except Exception as e:
            print(f"[PlotWindow 错误] lambdify 转换失败: {e}")
            return None
    def refresh_plot(self):
        """把参数与包装好的 exact_func 喂给 plotting_qt 控件。"""
        domain = self.result_data.get('domain', {})
        x_range = tuple(domain.get('x') or [0.0, 1.0])
        y_range = tuple(domain.get('y') or [0.0, 1.0])
        t_range = tuple(domain.get('t') or [0.0, 1.0])
        model = self.result_data.get('model', None)
        exact_func = self.get_exact_callable()
        dimension = self.result_data.get('dimension', 1)
        has_t = self.result_data.get('has_t', False)
        if dimension == 1 and not has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, exact_func=exact_func, true_func=exact_func)
        elif dimension == 1 and has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, t_range=t_range, exact_func=exact_func, true_func=exact_func)
        elif dimension == 2 and not has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, y_range=y_range, exact_func=exact_func, true_func=exact_func)
        else:
            self.plot_widget.set_data(model=model, x_range=x_range, y_range=y_range, t_range=t_range, exact_func=exact_func, true_func=exact_func)
        self.apply_theme()
    def on_slider_change(self, sym_name, val, label_widget):
        """
        滑块值变化时的响应。

        :param sym_name: 常数符号名称
        :type sym_name: str
        :param val: 滑块原始值 (-100 到 100)
        :type val: int
        :param label_widget: 显示数值的标签控件
        :type label_widget: QLabel
        """
        real_val = val / 10.0
        label_widget.setText(f"{real_val:.1f}")
        self.refresh_plot()
    def apply_theme(self):
        """应用主题样式到对话框和绘图控件。"""
        super().apply_theme()
        self._apply_mpl_theme()
    def _apply_mpl_theme(self):
        """适配 Matplotlib 画布与图表元素，响应主题切换。"""
        if not hasattr(self, 'plot_widget') or self.plot_widget is None:
            return
        theme = ThemeManager.instance().current
        text_color = theme.text_primary
        mpl_border_color = parse_css_color_for_mpl(getattr(theme, 'btn_border', '#888888'))
        mpl_text_color = parse_css_color_for_mpl(text_color)
        mpl_card_bg = parse_css_color_for_mpl(theme.card_bg)
        if hasattr(self.plot_widget, 'figure'):
            fig = self.plot_widget.figure
            fig.patch.set_facecolor(mpl_card_bg)
            for ax in fig.axes:
                ax.set_facecolor(mpl_card_bg)
                ax.xaxis.label.set_color(mpl_text_color)
                ax.yaxis.label.set_color(mpl_text_color)
                if hasattr(ax, 'zaxis'):
                    ax.zaxis.label.set_color(mpl_text_color)
                ax.title.set_color(mpl_text_color)
                ax.tick_params(colors=mpl_text_color, labelcolor=mpl_text_color)
                for spine in ax.spines.values():
                    spine.set_color(mpl_border_color)
                legend = ax.get_legend()
                if legend:
                    legend.get_frame().set_facecolor(mpl_card_bg)
                    legend.get_frame().set_edgecolor(mpl_border_color)
                    for t in legend.get_texts():
                        t.set_color(mpl_text_color)
            self.plot_widget.canvas.draw()
