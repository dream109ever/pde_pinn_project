import re
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QWidget, QMessageBox, QGroupBox
from PyQt5.QtCore import Qt
from ui.theme_manager import ThemeManager
from src.plotting_qt import (
    Steady1DPlotWidget,
    Transient1DPlotWidget,
    Steady2DPlotWidget,
    Transient2DPlotWidget
)
# 全局配置 Matplotlib 字体，彻底解决中文乱码与负号显示为方框的问题
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.default'] = 'regular'

def parse_css_color_for_mpl(css_color: str):
    """将 QSS/CSS 格式的颜色（如 'rgba(255, 255, 255, 0.9)' 或 '#FFFFFF'）转换为 Matplotlib 支持的格式"""
    if not css_color or not isinstance(css_color, str):
        return '#888888'  # 默认兜底色
    css_color = css_color.strip()
    # 匹配 rgba(r, g, b, a) 或 rgb(r, g, b)
    rgba_match = re.match(r'rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*([\d.]+)\s*)?\)', css_color)
    if rgba_match:
        r = int(rgba_match.group(1)) / 255.0
        g = int(rgba_match.group(2)) / 255.0
        b = int(rgba_match.group(3)) / 255.0
        a = float(rgba_match.group(4)) if rgba_match.group(4) is not None else 1.0
        return (r, g, b, a)
    # 普通 Hex 颜色 (#FFF / #FFFFFF) 或 Matplotlib 内置颜色名称直接返回
    return css_color

class PlotWindow(QDialog):
    """可视化弹窗：对接 plotting_qt 控件库，支持 C1, C2 任意常数拖动与主题切换"""
    def __init__(self, result_data: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("PDE / ODE 解函数可视化")
        self.resize(850, 680)
        self.result_data = result_data
        self.sliders = {}

        self.init_ui()

        # 绑定主题切换信号
        self.apply_theme()
        ThemeManager.instance().theme_changed.connect(lambda _: self.apply_theme())

    def init_ui(self):
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
            # 改用原生 QGroupBox，不使用任何 ID 选择器
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

                # 闭包绑定滑块改变事件
                slider.valueChanged.connect(
                    lambda val, s_name=sym_name, l_lbl=val_lbl: self.on_slider_change(s_name, val, l_lbl)
                )

                h_layout.addWidget(sym_lbl)
                h_layout.addWidget(slider, stretch=1)
                h_layout.addWidget(val_lbl)
                slider_box.addLayout(h_layout)

                self.sliders[sym_name] = slider

            main_layout.addWidget(self.slider_group)

        # 初次渲染图像
        self.refresh_plot()

    def get_exact_callable(self):
        """将 SymPy 表达式/等式/字符串转换为 NumPy/PyTorch 万能兼容的 Python 函数"""
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
            if val is None:
                return None
            try:
                # 尝试直接转换为标准的 float 数组
                return np.asarray(val, dtype=float)
            except (TypeError, ValueError):
                pass

            # 如果数组内混入了 SymPy 表达式/对象 (dtype=object)，逐个元素求值并转 float
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
                    if len(args) == 1:  # 传入 pts Tensor/Array (N, 2)
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, t = pts[..., 0], pts[..., 1]
                    else:  # 传入 (x, t) 两个独立参数
                        x = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        t = args[1].detach().cpu().numpy() if hasattr(args[1], 'detach') else args[1]
                    val = fn(x, t)
                    return np.full_like(x, val) if np.isscalar(val) else val
                return wrapper

            elif dimension == 2 and not has_t:
                fn = sp.lambdify((x_sym, y_sym), expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    if len(args) == 1:  # 传入 pts Tensor/Array (N, 2)
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, y = pts[..., 0], pts[..., 1]
                    else:  # 传入 (X, Y) 网格矩阵
                        x = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        y = args[1].detach().cpu().numpy() if hasattr(args[1], 'detach') else args[1]
                    val = fn(x, y)
                    return np.full_like(x + y, val) if np.isscalar(val) else val
                return wrapper

            else:  # 2D 含时
                fn = sp.lambdify((x_sym, y_sym, t_sym), expr_curr, modules=['numpy', 'math'])
                def wrapper(*args):
                    if len(args) == 1:  # 传入 pts Tensor/Array (N, 3)
                        pts = args[0].detach().cpu().numpy() if hasattr(args[0], 'detach') else args[0]
                        x, y, t = pts[..., 0], pts[..., 1], pts[..., 2]
                    else:  # 传入 (X, Y, t)
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
        """把参数与包装好的 exact_func 喂给 plotting_qt 控件"""
        domain = self.result_data.get('domain', {})
        x_range = tuple(domain.get('x') or [0.0, 1.0])
        y_range = tuple(domain.get('y') or [0.0, 1.0])
        t_range = tuple(domain.get('t') or [0.0, 1.0])

        model = self.result_data.get('model', None)
        exact_func = self.get_exact_callable()

        dimension = self.result_data.get('dimension', 1)
        has_t = self.result_data.get('has_t', False)

        # 同时传入 exact_func 和 true_func，做双保险适配
        if dimension == 1 and not has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, exact_func=exact_func, true_func=exact_func)
        elif dimension == 1 and has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, t_range=t_range, exact_func=exact_func, true_func=exact_func)
        elif dimension == 2 and not has_t:
            self.plot_widget.set_data(model=model, x_range=x_range, y_range=y_range, exact_func=exact_func, true_func=exact_func)
        else:
            self.plot_widget.set_data(model=model, x_range=x_range, y_range=y_range, t_range=t_range, exact_func=exact_func, true_func=exact_func)

        # 主题色彩适配
        self.apply_theme()

    def on_slider_change(self, sym_name, val, label_widget):
        real_val = val / 10.0
        label_widget.setText(f"{real_val:.1f}")
        self.refresh_plot()

    def apply_theme(self):
        """跟随全局主题同步更新 QSS 样式与 Matplotlib 图表颜色的暗亮适配"""
        theme = ThemeManager.instance().current
        bg_color = theme.card_bg if theme.is_dark else "#FFFFFF"
        text_color = theme.text_primary

        self.setStyleSheet(f"""
            QDialog {{ 
                background-color: {bg_color}; 
            }}
            QGroupBox {{
                color: {text_color};
                font-weight: bold;
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
                background-color: {theme.card_bg};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 4px;
                background-color: {theme.card_bg};
            }}
            QLabel {{ 
                color: {text_color}; 
                background: transparent; 
            }}
            QSlider::groove:horizontal {{
                height: 4px;
                background: {theme.btn_border};
                border-radius: 2px;
            }}
            QSlider::sub-page:horizontal {{
                background: {theme.btn_hover_bg};
                border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {theme.btn_hover_bg};
                width: 12px;
                height: 12px;
                margin: -4px 0;
                border-radius: 6px;
            }}
        """)

        # 适配 Matplotlib 画布与图表元素
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
