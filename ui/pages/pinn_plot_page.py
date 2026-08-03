# ui/pages/pinn_plot_page.py
import sys
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QMessageBox, QGroupBox, QFormLayout, QSplitter, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .base_widgets import BasePage
from .preview_label import PreviewLabel

import sympy as sp
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10

# 导入核心算法库
from src import *

class PINNTrainerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, float, object, object)  # epoch, loss, model, exact_func
    finished_signal = pyqtSignal()

    def __init__(self, problem_config: dict, total_epochs: int = 1500, chunk_epochs: int = 50):
        super().__init__()
        self.problem_config = problem_config
        self.total_epochs = total_epochs
        self.chunk_epochs = chunk_epochs
        self._is_running = True

    def stop(self):
        self._is_running = False

    def _build_loss_functions(self, config):
        dimension = config["dimension"]
        has_t = config["has_t"]
        order = config.get("order", 2)
        coeffs = config.get("coeffs", {})
        source_term = config.get("source_term", "0")
        condition = config.get("condition", [])

        if dimension == 1 and not has_t:
            coeff_funcs = InputParser.parse_coeffs_1d(coeffs, ["x"])
            f, _ = InputParser.parse_source(source_term, ["x"])
            return LossGenerator.generate(
                dimension=1, has_t=False,
                coeff_funcs=coeff_funcs, f=f,
                condition=condition, order=order
            )
        elif dimension == 1 and has_t:
            parsed = InputParser.parse_coeffs_1d_transient(coeffs, ["x", "t"])
            f, _ = InputParser.parse_source(source_term, ["x", "t"])
            structured = InputParser.parse_conditions(condition, ["x", "t"])
            return LossGenerator.generate(
                dimension=1, has_t=True,
                c_tt_fn=parsed.get("u_tt", lambda x, t: 0.0),
                c_t_fn=parsed.get("u_t", lambda x, t: 0.0),
                c_xx_fn=parsed.get("u_xx", lambda x, t: 0.0),
                c_x_fn=parsed.get("u_x", lambda x, t: 0.0),
                c_u_fn=parsed.get("u", lambda x, t: 0.0),
                f=f,
                ic_conds=structured["initial"],
                bc_sides={
                    "left": [b for b in structured["boundary"] if b.get("location_clean") == "left"],
                    "right": [b for b in structured["boundary"] if b.get("location_clean") == "right"]
                }
            )
        elif dimension == 2 and not has_t:
            parsed = InputParser.parse_coeffs_2d(coeffs, ["x", "y"])
            f, _ = InputParser.parse_source(source_term, ["x", "y"])
            structured = InputParser.parse_conditions(condition, ["x", "y"])
            return LossGenerator.generate(
                dimension=2, has_t=False,
                c_xx_fn=parsed.get("u_xx", lambda x, y: 0.0),
                c_yy_fn=parsed.get("u_yy", lambda x, y: 0.0),
                c_xy_fn=parsed.get("u_xy", lambda x, y: 0.0),
                c_x_fn=parsed.get("u_x", lambda x, y: 0.0),
                c_y_fn=parsed.get("u_y", lambda x, y: 0.0),
                c_u_fn=parsed.get("u", lambda x, y: 0.0),
                f=f,
                bc_sides={
                    s: [b for b in structured["boundary"] if b.get("location_clean") == s]
                    for s in ["left", "right", "bottom", "top"]
                }
            )
        elif dimension == 2 and has_t:
            parsed = InputParser.parse_coeffs_2d_transient(coeffs, ["x", "y", "t"])
            f, _ = InputParser.parse_source(source_term, ["x", "y", "t"])
            structured = InputParser.parse_conditions(condition, ["x", "y", "t"])
            return LossGenerator.generate(
                dimension=2, has_t=True,
                c_tt_fn=parsed.get("u_tt", lambda x, y, t: 0.0),
                c_t_fn=parsed.get("u_t", lambda x, y, t: 0.0),
                c_xx_fn=parsed.get("u_xx", lambda x, y, t: 0.0),
                c_yy_fn=parsed.get("u_yy", lambda x, y, t: 0.0),
                c_xy_fn=parsed.get("u_xy", lambda x, y, t: 0.0),
                c_x_fn=parsed.get("u_x", lambda x, y: 0.0),
                c_y_fn=parsed.get("u_y", lambda x, y, t: 0.0),
                c_u_fn=parsed.get("u", lambda x, y, t: 0.0),
                f=f,
                ic_conds=structured["initial"],
                bc_sides={
                    s: [b for b in structured["boundary"] if b.get("location_clean") == s]
                    for s in ["left", "right", "bottom", "top"]
                }
            )

    def run(self):
        try:
            config = self.problem_config
            self.log_signal.emit("初始化求解器...")

            # 1. 构建损失函数
            loss_functions = self._build_loss_functions(config)

            # 2. 构建模型
            model = build_model(
                coeffs=config.get("coeffs", {}),
                source_term=config.get("source_term", "0"),
                conditions=config.get("condition", []),
                has_t=config.get("has_t", False),
                dimension=config.get("dimension", 1),
                verbose=False,
            )

            # 3. 采样器
            domain = config.get("domain", {"x": [0, 1]})
            sampler = DomainSampler(
                domain.get("x", [0, 1]),
                domain.get("y"),
                domain.get("t")
            )

            # 4. 训练器
            trainer = PINNTrainer(
                model=model,
                loss_functions=loss_functions,
                optimizer="adam",
                lr=1e-3,
                scheduler="plateau",
                device="cpu",
            )

            self.log_signal.emit("组件就绪，开始训练...")

            # 5. 精确解
            result = solve_pde(
                dimension=config["dimension"],
                order=config.get("order", 2),
                has_t=config["has_t"],
                coeffs=config.get("coeffs", {}),
                source_term=config.get("source_term", "0"),
                domain=domain,
                condition=config.get("condition", []),
            )
            exact_func = result.get("exact_solution")
            # 6. 分块训练
            current_epoch = 0
            while current_epoch < self.total_epochs and self._is_running:
                step = min(self.chunk_epochs, self.total_epochs - current_epoch)
                history = trainer.train(
                    n_epochs=step,
                    sampler=sampler,
                    n_interior=1000,
                    n_boundary_per_side=50,
                    n_initial=200,
                    verbose=False
                )
                current_epoch += step
                latest_loss = history['total_loss'][-1]
                self.log_signal.emit(f"Epoch [{current_epoch:04d}/{self.total_epochs}] | Loss: {latest_loss:.4e}")
                self.progress_signal.emit(current_epoch, latest_loss, model, exact_func)

            if not self._is_running:
                self.log_signal.emit("[后台线程] 已安全中断。")
            else:
                self.log_signal.emit("[后台线程] 训练全部完成！")
        except Exception as e:
            self.log_signal.emit(f"错误: {e}")
        finally:
            self.finished_signal.emit()


class PinnPlotPage(BasePage):
    """PINN 求解计算、网络配置与实时动态绘图页（借鉴 gui_test.py 架构）"""

    def __init__(self, back_to_input_cb, parent=None):
        super().__init__(parent)
        self.back_to_input_cb = back_to_input_cb
        self.problem_config = None
        self.exact_func = None
        self.exact_expr = None
        self.trainer_thread = None
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 标题
        main_layout.addWidget(QLabel("<h2>PINN 求解与实时绘图</h2>"))

        # 主分割器
        splitter = QSplitter(Qt.Horizontal)

        # ---------- 左侧：网络配置 + 日志 ----------
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 网络参数组
        self.network_group = QGroupBox("网络参数配置")
        net_form = QFormLayout(self.network_group)

        self.num_layers_spin = QSpinBox()
        self.num_layers_spin.setRange(1, 10)
        self.num_layers_spin.setValue(3)
        net_form.addRow("隐藏层数量:", self.num_layers_spin)

        self.neurons_per_layer = QLineEdit("64, 64, 64")
        net_form.addRow("每层神经元数 (逗号分隔):", self.neurons_per_layer)

        self.activation_combo = QComboBox()
        self.activation_combo.addItems(["tanh", "relu", "sigmoid", "sin", "leaky_relu", "gelu"])
        net_form.addRow("激活函数:", self.activation_combo)

        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(["adam", "sgd", "adamw"])
        net_form.addRow("优化器:", self.optimizer_combo)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(1e-3)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setDecimals(6)
        net_form.addRow("学习率:", self.lr_spin)

        # self.batch_norm_check = QCheckBox("使用 BatchNorm")
        # net_form.addRow("", self.batch_norm_check)

        left_layout.addWidget(self.network_group)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.solve_btn = QPushButton("开始训练")
        self.solve_btn.setMinimumHeight(40)
        self.solve_btn.clicked.connect(self.start_solving)
        btn_layout.addWidget(self.solve_btn)

        self.stop_btn = QPushButton("停止训练")
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_solving)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

        # 日志
        left_layout.addWidget(QLabel("<b>训练日志:</b>"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)

        # 解析解展示
        left_layout.addWidget(QLabel("<b>解析解 (LaTeX):</b>"))
        self.result_latex = PreviewLabel(self, font_size=9)
        self.result_latex.set_latex(r"\text{等待配置...}")
        left_layout.addWidget(self.result_latex)

        left_layout.addStretch()
        splitter.addWidget(left_widget)

        # ---------- 右侧：QStackedWidget 4个绘图控件 ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.stacked_widget = QStackedWidget()
        self.widget_1d_steady = Steady1DPlotWidget(orientation='vertical')
        self.widget_1d_transient = Transient1DPlotWidget(orientation='vertical')
        self.widget_2d_steady = Steady2DPlotWidget(orientation='vertical')
        self.widget_2d_transient = Transient2DPlotWidget(orientation='vertical')

        self.stacked_widget.addWidget(self.widget_1d_steady)      # index 0
        self.stacked_widget.addWidget(self.widget_1d_transient)   # index 1
        self.stacked_widget.addWidget(self.widget_2d_steady)      # index 2
        self.stacked_widget.addWidget(self.widget_2d_transient)   # index 3

        right_layout.addWidget(self.stacked_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])

        main_layout.addWidget(splitter)

        # 底部导航
        bottom_nav = QHBoxLayout()
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.back_to_input_cb)
        bottom_nav.addWidget(back_btn)
        bottom_nav.addStretch()
        main_layout.addLayout(bottom_nav)

    # ========== 接收配置 ==========

    def set_problem_config(self, config: dict):
        """
        接收配置（兼容旧格式 result_data 或纯 config）
        """
        self.problem_config = config
        self.exact_func = config.get("_exact_func", None)
        self.exact_expr = None
        auto_net = None

        # 若 exact_func 存在，存入 problem_config 供训练线程使用
        if self.exact_func is not None:
            self.problem_config["_exact_func"] = self.exact_func

        self.log_text.clear()
        self.log_text.append("已接收方程配置。")

        # 自动切换绘图控件
        dim = self.problem_config.get("dimension", 1)
        has_t = self.problem_config.get("has_t", False)
        index = { (1, False): 0, (1, True): 1, (2, False): 2, (2, True): 3 }.get((dim, has_t), 0)
        self.stacked_widget.setCurrentIndex(index)

        # 显示解析解（若有）
        if self.exact_expr is not None:
            self.result_latex.set_latex(sp.latex(self.exact_expr))
            self.log_text.append(f"解析解: {self.exact_expr}")
        else:
            self.result_latex.set_latex(r"\text{无解析解，使用纯PINN训练}")
            self.log_text.append("未找到闭式解析解。")

        # 自动填入网络结构（仅当 auto_net 存在时）
        if auto_net is not None:
            try:
                linear_layers = [m for m in auto_net.children() if isinstance(m, torch.nn.Linear)]
                if linear_layers:
                    self.num_layers_spin.setValue(max(len(linear_layers) - 1, 1))
                    dims = [m.out_features for m in linear_layers[:-1]]
                    self.neurons_per_layer.setText(", ".join(str(d) for d in dims))
                    self.log_text.append("自动填充网络结构。")
            except Exception:
                pass

    # ========== 训练控制 ==========

    def start_solving(self):
        if not self.problem_config:
            QMessageBox.warning(self, "错误", "未检测到有效的方程配置！")
            return

        try:
            hidden_dims = [int(x.strip()) for x in self.neurons_per_layer.text().split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "错误", "每层神经元数必须为逗号分隔的整数！")
            return

        if not hidden_dims:
            QMessageBox.warning(self, "错误", "请至少指定一层隐藏层！")
            return

        self.solve_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.log_text.append("启动训练线程...")

        # 创建训练线程（与 gui_test.py 一致）
        self.trainer_thread = PINNTrainerThread(
            problem_config=self.problem_config,
            total_epochs=3000,
            chunk_epochs=50
        )
        self.trainer_thread.log_signal.connect(self.log_text.append)
        self.trainer_thread.progress_signal.connect(self.update_plot_with_model)
        self.trainer_thread.finished_signal.connect(self.on_training_finished)
        self.trainer_thread.start()

    def stop_solving(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.log_text.append("正在中止线程...")
            self.trainer_thread.stop()
            self.stop_btn.setEnabled(False)

    def on_training_finished(self):
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append("训练线程结束。")

    # ========== 实时更新绘图 ==========

    def update_plot_with_model(self, epoch: int, loss_val: float, model: torch.nn.Module, exact_func):
        """每 chunk 更新一次当前显示的绘图控件"""
        if not self.problem_config:
            return

        dim = self.problem_config.get("dimension", 1)
        has_t = self.problem_config.get("has_t", False)
        domain = self.problem_config.get("domain", {"x": [0, 1]})

        model.eval()

        if dim == 1 and not has_t:
            def true_func_1d(x):
                if exact_func is not None:
                    return torch.tensor([exact_func(xi.item()) for xi in x], dtype=torch.float32).reshape(-1, 1)
                return None
            self.widget_1d_steady.set_data(
                model,
                x_range=tuple(domain.get("x", [0, 1])),
                true_func=true_func_1d
            )

        elif dim == 1 and has_t:
            self.widget_1d_transient.set_data(
                model,
                x_range=tuple(domain.get("x", [0, 1])),
                t_range=tuple(domain.get("t", [0, 1])),
                exact_func=exact_func
            )

        elif dim == 2 and not has_t:
            def true_func_2d(pts):
                if exact_func is not None:
                    return torch.tensor(
                        [exact_func(xi.item(), yi.item()) for xi, yi in pts],
                        dtype=torch.float32
                    ).reshape(-1, 1)
                return None
            self.widget_2d_steady.set_data(
                model,
                x_range=tuple(domain.get("x", [0, 1])),
                y_range=tuple(domain.get("y", [0, 1])),
                true_func=true_func_2d
            )

        elif dim == 2 and has_t:
            self.widget_2d_transient.set_data(
                model,
                x_range=tuple(domain.get("x", [0, 1])),
                y_range=tuple(domain.get("y", [0, 1])),
                t_range=tuple(domain.get("t", [0, 1])),
                exact_func=exact_func
            )
