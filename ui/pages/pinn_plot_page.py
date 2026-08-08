# ui/pages/pinn_plot_page.py
import json
import torch
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QMessageBox, QGroupBox, QFormLayout, QSplitter, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QLineEdit, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib as mpl
from src import *
from ui.theme_manager import ThemeManager
from .base_widgets import BasePage, PreviewLabel, SolverThread
mpl.use('Agg')
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11

class PINNTrainerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, float, object, object)
    loss_signal = pyqtSignal(int, float, float, float)
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
            total_epochs = config.get("_user_epochs", self.total_epochs)
            self.log_signal.emit("初始化求解器...")
            # 1. 构建损失函数
            loss_functions = self._build_loss_functions(config)
            # 2. 构建模型
            network_config = config.get("_network_config")
            if network_config:
                from src.network_factory import build_network
                model = build_network(
                    input_dim=network_config["input_dim"],
                    output_dim=network_config["output_dim"],
                    hidden_dims=network_config["hidden_dims"],
                    activation=network_config.get("activation", "tanh"),
                    init_method=network_config.get("init_method", "xavier"),
                )
            else:
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
            # 5. 从 config 中获取精确解（由 SolverThread 事先存入）
            exact_func = config.get("_exact_func", None)
            # 6. 分块训练
            current_epoch = 0
            loss_history = {'total': [], 'pde': [], 'bc': []}
            # 训练回调：每5步发送一次损失数据
            def loss_callback(epoch, total_loss, pde_loss, bc_loss, total_epochs=total_epochs):
                loss_history['total'].append(total_loss)
                loss_history['pde'].append(pde_loss)
                loss_history['bc'].append(bc_loss)
                if epoch % 5 == 0 or epoch == total_epochs - 1:
                    self.loss_signal.emit(epoch, total_loss, pde_loss, bc_loss)
            while current_epoch < total_epochs and self._is_running:
                step = min(self.chunk_epochs, total_epochs - current_epoch)
                history = trainer.train(
                    n_epochs=step,
                    sampler=sampler,
                    n_interior=1000,
                    n_boundary_per_side=50,
                    n_initial=200,
                    verbose=False,
                    callback=loss_callback
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
    """PINN 求解计算、网络配置与实时动态绘图页"""

    def __init__(self, back_to_input_cb, parent=None):
        super().__init__(parent)
        self.back_to_input_cb = back_to_input_cb
        self.problem_config = None
        self.exact_func = None
        self.exact_expr = None
        self.solver_thread = None
        self.trainer_thread = None
        self.solved_result = None
        self._current_config_hash = None
        self._is_training = False
        self._loss_history = {'total': [], 'pde': [], 'bc': []}
        self.init_ui()
        ThemeManager.instance().theme_changed.connect(lambda _: self.apply_theme())
        self.apply_theme()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(QLabel("PINN 求解与实时绘图"))
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
        self.total_epochs_spin = QSpinBox()
        self.total_epochs_spin.setRange(100, 100000)
        self.total_epochs_spin.setValue(3000)
        self.total_epochs_spin.setSingleStep(500)
        net_form.addRow("总训练轮数:", self.total_epochs_spin)
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
        left_layout.addWidget(QLabel("<b>训练日志:</b>"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        left_layout.addWidget(self.log_text)
        left_layout.addWidget(QLabel("<b>解析解 (LaTeX):</b>"))
        self.result_latex = PreviewLabel(self, font_size=9)
        self.result_latex.set_latex(r"\text{等待配置...}")
        left_layout.addWidget(self.result_latex)
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        # ---------- 右侧：上下布局（损失曲线 + 绘图控件） ----------
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        # 右侧内部垂直分割器（损失曲线 40% | 绘图控件 60%）
        right_splitter = QSplitter(Qt.Vertical)
        # ---- 上半部分：损失曲线 ----
        loss_group = QGroupBox("损失函数曲线")
        loss_layout = QVBoxLayout(loss_group)
        self.loss_plot_widget = LossPlotWidget()
        loss_layout.addWidget(self.loss_plot_widget)
        right_splitter.addWidget(loss_group)
        # ---- 下半部分：绘图控件 ----
        plot_group = QGroupBox("预测解与精确解对比")
        plot_layout = QVBoxLayout(plot_group)
        self.stacked_widget = QStackedWidget()
        self.widget_1d_steady = Steady1DPlotWidget(mode='overlay')
        self.widget_1d_transient = Transient1DPlotWidget(mode='overlay')
        self.widget_2d_steady = Steady2DPlotWidget(mode='overlay')
        self.widget_2d_transient = Transient2DPlotWidget(mode='overlay')
        self.stacked_widget.addWidget(self.widget_1d_steady)      # index 0
        self.stacked_widget.addWidget(self.widget_1d_transient)   # index 1
        self.stacked_widget.addWidget(self.widget_2d_steady)      # index 2
        self.stacked_widget.addWidget(self.widget_2d_transient)   # index 3
        plot_layout.addWidget(self.stacked_widget)
        right_splitter.addWidget(plot_group)
        right_splitter.setSizes([300, 450])
        right_layout.addWidget(right_splitter)
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
        """接收上一页传递的纯配置字典。"""
        # ===== 1. 计算新配置的哈希值 =====
        try:
            key_fields = {
                "dimension": config.get("dimension"),
                "has_t": config.get("has_t"),
                "order": config.get("order"),
                "coeffs": json.dumps(config.get("coeffs", {}), sort_keys=True),
                "source_term": config.get("source_term", ""),
                "domain": json.dumps(config.get("domain", {}), sort_keys=True),
                "condition": json.dumps(config.get("condition", []), sort_keys=True),
            }
            new_hash = hash(json.dumps(key_fields, sort_keys=True))
        except:
            new_hash = None
        # ===== 2. 判断是否是首次设置 =====
        is_first_time = (self._current_config_hash is None)
        # ===== 3. 判断配置是否真正改变 =====
        if is_first_time:
            config_changed = False
        else:
            config_changed = (self._current_config_hash != new_hash)
        # ===== 4. 如果配置改变，停止旧线程 =====
        if config_changed:
            if self.solver_thread is not None and self.solver_thread.isRunning():
                self.solver_thread.terminate()
                self.solver_thread = None
            if self.trainer_thread is not None and self.trainer_thread.isRunning():
                self.trainer_thread.stop()
                self.trainer_thread = None
            self._is_training = False
            self.network_group.setEnabled(True)
            self.total_epochs_spin.setEnabled(True)
            self.solve_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._loss_history = {'total': [], 'pde': [], 'bc': []}
            self._current_config_hash = new_hash
            self.result_latex.set_latex(r"\text{等待求解...}")
            self.loss_plot_widget.update_plot({})
            self.log_text.clear()
            self.log_text.append("检测到方程配置已更改，已自动停止旧训练线程。")
            self.log_text.append("请点击「开始训练」使用新配置重新训练。")
        else:
            self.log_text.clear()
            if is_first_time:
                self.log_text.append("首次接收方程配置。")
                self._current_config_hash = new_hash
            else:
                self.log_text.append("返回绘制页，训练数据保留。")

            if self._loss_history and any(len(v) > 0 for v in self._loss_history.values()):
                self.loss_plot_widget.update_plot(self._loss_history)
                self.log_text.append(f"已恢复损失曲线 (共 {len(self._loss_history['total'])} 步)")
        # ===== 5. 存储配置 =====
        self.problem_config = config
        # ===== 6. 自动切换绘图控件 =====
        dim = self.problem_config.get("dimension", 1)
        has_t = self.problem_config.get("has_t", False)
        index = {(1, False): 0, (1, True): 1, (2, False): 2, (2, True): 3}.get((dim, has_t), 0)
        self.stacked_widget.setCurrentIndex(index)
        # ===== 7. 自动填充网络结构 =====
        auto_net = config.get("auto_network", None)
        if auto_net is not None and is_first_time:
            try:
                linear_layers = [m for m in auto_net.children() if isinstance(m, torch.nn.Linear)]
                if linear_layers:
                    self.num_layers_spin.setValue(max(len(linear_layers) - 1, 1))
                    dims = [m.out_features for m in linear_layers[:-1]]
                    self.neurons_per_layer.setText(", ".join(str(d) for d in dims))
                    self.log_text.append("已自动填充网络结构。")
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
        if not self._loss_history or len(self._loss_history.get('total', [])) == 0:
            self.widget_1d_steady.set_data(model=None)
            self.widget_1d_transient.set_data(model=None)
            self.widget_2d_steady.set_data(model=None)
            self.widget_2d_transient.set_data(model=None)
        dim = self.problem_config.get("dimension", 1)
        has_t = self.problem_config.get("has_t", False)
        input_dim = dim + (1 if has_t else 0)
        network_config = {
            "input_dim": input_dim,
            "output_dim": 1,
            "hidden_dims": hidden_dims,
            "activation": self.activation_combo.currentText(),
            "init_method": "xavier",
        }
        self._is_training = True
        if self._current_config_hash is None or not self._loss_history:
            self._loss_history = {'total': [], 'pde': [], 'bc': []}
            self.loss_plot_widget.update_plot({})
        self.problem_config["_network_config"] = network_config
        self.problem_config["_user_epochs"] = self.total_epochs_spin.value()
        self.network_group.setEnabled(False)
        self.total_epochs_spin.setEnabled(False)
        self.solve_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.log_text.clear()
        self.log_text.append("启动训练线程...")
        self.solver_thread = SolverThread(self.problem_config)
        self.solver_thread.log_signal.connect(self.log_text.append)
        self.solver_thread.finished_signal.connect(self.on_solve_finished)
        self.solver_thread.error_signal.connect(self.on_solve_error)
        self.solver_thread.start()
    def on_solve_finished(self, result_data):
        """求解完成，保存结果并启动训练"""
        self.log_text.append("✅ 求解完成，开始训练...")
        self.solved_result = result_data
        # 更新精确解显示
        self.exact_expr = result_data.get('exact_expr')
        if self.exact_expr:
            self.result_latex.set_latex(str(self.exact_expr))
        else:
            self.result_latex.set_latex(r"\text{无解析解，使用纯PINN训练}")
        exact_func = result_data.get('exact_solution')
        if exact_func is not None:
            self.problem_config["_exact_func"] = exact_func
        self.trainer_thread = PINNTrainerThread(
            problem_config=self.problem_config,
            total_epochs=self.total_epochs_spin.value(),
            chunk_epochs=50
        )
        self.trainer_thread.log_signal.connect(self.log_text.append)
        self.trainer_thread.progress_signal.connect(self.update_plot_with_model)
        self.trainer_thread.loss_signal.connect(self.update_loss_plot)
        self.trainer_thread.finished_signal.connect(self.on_training_finished)
        self.trainer_thread.start()
    def on_solve_error(self, err_msg):
        """求解失败处理"""
        self._is_training = False
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.network_group.setEnabled(True)
        self.total_epochs_spin.setEnabled(True)
        self.log_text.append(f"求解失败: {err_msg}")
        QMessageBox.warning(self, "求解失败", err_msg)
    def on_training_finished(self):
        """训练完成"""
        self._is_training = False
        self.network_group.setEnabled(True)
        self.total_epochs_spin.setEnabled(True)
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append("✅ 训练完成！")
    def stop_solving(self):
        """停止流程：需要同时停止两个线程"""
        self._is_training = False
        if self.solver_thread and self.solver_thread.isRunning():
            self.solver_thread.terminate()
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.trainer_thread.stop()
        self.network_group.setEnabled(True)
        self.total_epochs_spin.setEnabled(True)
        self.solve_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_text.append("⏹ 流程已中断。")
    def _auto_stop_training(self):
        if self._is_training:
            self._is_training = False
            if self.solver_thread and self.solver_thread.isRunning():
                self.solver_thread.terminate()
                self.solver_thread = None
            if self.trainer_thread and self.trainer_thread.isRunning():
                self.trainer_thread.stop()
                self.trainer_thread = None
            self.network_group.setEnabled(True)
            self.total_epochs_spin.setEnabled(True)
            self.solve_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.log_text.append("⏹ 已自动停止旧训练线程（保留最后一次结果）")
    # ========== 更新损失曲线 ==========
    def update_loss_plot(self, epoch: int, total_loss: float, pde_loss: float, bc_loss: float):
        """每5步更新一次损失曲线（高频）"""
        # 获取当前历史数据
        if not hasattr(self, '_loss_history'):
            self._loss_history = {'total': [], 'pde': [], 'bc': []}
        self._loss_history['total'].append(total_loss)
        self._loss_history['pde'].append(pde_loss)
        self._loss_history['bc'].append(bc_loss)
        # 只保留最近2000个点防止内存溢出
        max_points = 2000
        if len(self._loss_history['total']) > max_points:
            self._loss_history['total'] = self._loss_history['total'][-max_points:]
            self._loss_history['pde'] = self._loss_history['pde'][-max_points:]
            self._loss_history['bc'] = self._loss_history['bc'][-max_points:]
        # 更新损失曲线
        self.loss_plot_widget.update_plot(self._loss_history)
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
        self.apply_theme()
    def apply_theme(self):
        """应用主题颜色到所有 matplotlib 画布和 Qt 控件"""
        try:
            theme = ThemeManager.instance().current
            bg_color = theme.card_bg
            text_color = theme.text_primary
            border_color = getattr(theme, 'btn_border', '#888888')
            # ===== 1. 调用子控件的 apply_theme（BasePlotWidget 负责自己的 QSlider/QLabel/QTabWidget） =====
            for widget in [
                self.widget_1d_steady,
                self.widget_1d_transient,
                self.widget_2d_steady,
                self.widget_2d_transient
            ]:
                if widget is not None and hasattr(widget, 'apply_theme'):
                    widget.apply_theme(theme)
            # ===== 2. Matplotlib 画布主题 =====
            def apply_to_figure(fig):
                if fig is None:
                    return
                fig.patch.set_facecolor(bg_color)
                for ax in fig.axes:
                    ax.set_facecolor(bg_color)
                    ax.tick_params(colors=text_color, labelcolor=text_color)
                    if hasattr(ax, 'title'):
                        ax.title.set_color(text_color)
                    if hasattr(ax, 'xaxis'):
                        ax.xaxis.label.set_color(text_color)
                        ax.yaxis.label.set_color(text_color)
                    if hasattr(ax, 'zaxis'):
                        ax.zaxis.label.set_color(text_color)
                    for spine in ax.spines.values():
                        spine.set_color(border_color)
                    legend = ax.get_legend()
                    if legend:
                        legend.get_frame().set_facecolor(bg_color)
                        legend.get_frame().set_edgecolor(border_color)
                        for t in legend.get_texts():
                            t.set_color(text_color)
                fig.canvas.draw()
            if hasattr(self.loss_plot_widget, 'figure'):
                apply_to_figure(self.loss_plot_widget.figure)
            for widget in [
                self.widget_1d_steady,
                self.widget_1d_transient,
                self.widget_2d_steady,
                self.widget_2d_transient
            ]:
                if widget is not None:
                    if hasattr(widget, 'figure'):
                        apply_to_figure(widget.figure)
                    if hasattr(widget, 'pred_figure'):
                        apply_to_figure(widget.pred_figure)
                    if hasattr(widget, 'err_figure'):
                        apply_to_figure(widget.err_figure)
            # ===== 3. 页面级别 QSS（不属于 BasePlotWidget 的控件，如左侧面板的 QGroupBox） =====
            groupbox_style = f"""
                QGroupBox {{
                    color: {text_color};
                    border: 1px solid {border_color};
                    border-radius: 6px;
                    margin-top: 6px;
                    padding-top: 10px;
                    background-color: {bg_color};
                }}
                QGroupBox::title {{
                    subcontrol-origin: margin;
                    subcontrol-position: top left;
                    left: 10px;
                    padding: 0 4px;
                }}
            """
            if hasattr(self, 'network_group'):
                self.network_group.setStyleSheet(groupbox_style)
                input_style = f"""
                    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                        background-color: {bg_color};
                        color: {text_color};
                        border: 1px solid {border_color};
                        border-radius: 4px;
                        padding: 4px;
                    }}
                """
                for child in self.network_group.findChildren((QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
                    child.setStyleSheet(input_style)
        except Exception:
            pass
