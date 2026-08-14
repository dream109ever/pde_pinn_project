# gui_test.py
"""
PINN Qt 界面集成测试系统（集成 src 算法库管线）

核心架构：
1. 导入 src 核心模块，支持 1D/2D 稳态/含时 4 种 PDE 场景。
2. PINNTrainerThread 独立线程调用 src.PINNTrainer，实现无卡顿后台训练。
3. solve_pde 自动推导解析解，实时计算精确的 Relative L2 Error 指标。
4. 提供【开始训练】与【停止训练】按钮，支持中间安全中断。
5. 全局高 DPI 适配与放大字号，提升可读性。

界面特色：
1. 自定义 QPainter 矢量绘制 PINN Logo，嵌入顶部 Header，零外部图片依赖。
2. Light Blue / Soft Slate 柔和清凉配色，专为科学计算与高对比度图表设计。
3. 卡片化布局 (Card Container) + 优雅的内边距与微调阴影感。
4. 完整集成后台多线程训练与 src 求解管线。
"""

import os
import sys
import torch
import matplotlib as mpl
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)
from src import *

# ===== 1. Matplotlib 字号与高 DPI 适配 =====
mpl.rcParams['font.size'] = 11          # 基础字号
mpl.rcParams['axes.titlesize'] = 13     # 子图标题字号
mpl.rcParams['axes.labelsize'] = 11     # X/Y 轴标签字号
mpl.rcParams['xtick.labelsize'] = 10    # X 轴刻度字号
mpl.rcParams['ytick.labelsize'] = 10    # Y 轴刻度字号
mpl.rcParams['legend.fontsize'] = 10   # 图例字号
# Qt 依赖导入
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QTextEdit, QLabel, QStackedWidget, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QPainterPath


# ===== 2. 淡蓝/冷灰科技风格 QSS 样式表 =====
LIGHT_BLUE_QSS = """
/* 全局背景：柔和冷灰/冰蓝底 */
QMainWindow, QWidget#MainContainer {
    background-color: #f0f4f9;
    color: #1e293b;
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
}

/* 卡片容器：纯白背景 + 微圆角 + 边框 */
QFrame.CardFrame {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

/* 标题文本 */
QLabel#SectionTitle {
    font-weight: 600;
    font-size: 13px;
    color: #334155;
}

/* 下拉选择框 */
QComboBox {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 12px;
    font-size: 13px;
    color: #0f172a;
}
QComboBox:hover {
    border-color: #3b82f6;
    background-color: #ffffff;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}

/* 按钮样式 */
QPushButton#StartBtn {
    background-color: #2563eb;
    color: white;
    font-weight: bold;
    font-size: 13px;
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
}
QPushButton#StartBtn:hover {
    background-color: #1d4ed8;
}
QPushButton#StartBtn:disabled {
    background-color: #94a3b8;
}

QPushButton#StopBtn {
    background-color: #ef4444;
    color: white;
    font-weight: bold;
    font-size: 13px;
    border: none;
    border-radius: 6px;
    padding: 7px 18px;
}
QPushButton#StopBtn:hover {
    background-color: #dc2626;
}
QPushButton#StopBtn:disabled {
    background-color: #cbd5e1;
}

/* 运行日志终端 (浅底深字/清爽对比) */
QTextEdit#LogTerminal {
    background-color: #f8fafc;
    color: #0f172a;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
}
"""

# ===== 3. 用 Qt QPainter 绘制的原生 PINN Logo 控件 =====
class PINNLogoWidget(QWidget):
    """纯代码绘制的矢量 Logo：神经网络节点平滑演变为 PDE 物理波形"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(40, 40)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 1. 绘制背景圆角浅蓝底座
        bg_color = QColor("#e0f2fe")
        painter.setBrush(QBrush(bg_color))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 40, 40, 10, 10)
        # 2. 绘制连续的正弦/物理波形 path
        path = QPainterPath()
        path.moveTo(6, 20)
        path.cubicTo(12, 5, 20, 35, 26, 20)
        path.cubicTo(30, 10, 34, 20, 36, 20)
        pen = QPen(QColor("#0284c7"), 2.5)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawPath(path)
        # 3. 绘制神经网络节点 (散落在波形左侧与关键点)
        node_color = QColor("#2563eb")
        painter.setBrush(QBrush(node_color))
        painter.setPen(Qt.NoPen)
        nodes = [QPointF(6, 20), QPointF(14, 12), QPointF(14, 28), QPointF(26, 20)]
        for pt in nodes:
            painter.drawEllipse(pt, 2.5, 2.5)

# ===== 4. 问题字典与求解逻辑 =====
PROBLEMS = {
    "1d_steady": {
        "dimension": 1, "order": 2, "has_t": False, "coeffs": [1, 0, 1], "source_term": "0",
        "domain": {"x": [0, 1]},
        "condition": [{"point": 0.0, "value": 1.0, "derivative": 0}, {"point": 0.0, "value": 0.0, "derivative": 1}],
    },
    "1d_transient": {
        "dimension": 1, "order": 2, "has_t": True, "coeffs": {"u_t": 1.0, "u_xx": -0.1}, "source_term": "0",
        "domain": {"x": [0, 1], "t": [0, 0.5]},
        "condition": [
            {"side": "initial", "derivative": 0, "value": "sin(pi*x)"},
            {"side": "left", "type": "dirichlet", "value": "0"},
            {"side": "right", "type": "dirichlet", "value": "0"},
        ],
    },
    "2d_steady": {
        "dimension": 2, "order": 2, "has_t": False, "coeffs": {"u_xx": 1.0, "u_yy": 1.0},
        "source_term": "sin(pi*x)*sin(pi*y)", "domain": {"x": [0, 1], "y": [0, 1]},
        "condition": [
            {"side": "left", "type": "dirichlet", "value": "0"}, {"side": "right", "type": "dirichlet", "value": "0"},
            {"side": "bottom", "type": "dirichlet", "value": "0"}, {"side": "top", "type": "dirichlet", "value": "0"},
        ],
    },
    "2d_transient": {
        "dimension": 2, "order": 2, "has_t": True, "coeffs": {"u_t": 1.0, "u_xx": 0.1, "u_yy": 0.1},
        "source_term": "0", "domain": {"x": [0, 1], "y": [0, 1], "t": [0, 0.5]},
        "condition": [
            {"side": "initial", "derivative": 0, "value": "sin(pi*x)*sin(pi*y)"},
            {"side": "left", "type": "dirichlet", "value": "0"}, {"side": "right", "type": "dirichlet", "value": "0"},
            {"side": "bottom", "type": "dirichlet", "value": "0"}, {"side": "top", "type": "dirichlet", "value": "0"},
        ],
    },
}
def build_loss_functions(problem):
    dimension = problem["dimension"]
    has_t = problem["has_t"]
    coeffs = problem["coeffs"]
    source_term = problem["source_term"]
    condition = problem["condition"]
    order = problem["order"]
    if dimension == 1 and not has_t:
        coeff_funcs = InputParser.parse_coeffs_1d(coeffs, ["x"])
        f, _ = InputParser.parse_source(source_term, ["x"])
        return LossGenerator.generate(dimension=1, has_t=False, coeff_funcs=coeff_funcs, f=f, condition=condition, order=order)
    elif dimension == 1 and has_t:
        parsed = InputParser.parse_coeffs_1d_transient(coeffs, ["x", "t"])
        f, _ = InputParser.parse_source(source_term, ["x", "t"])
        structured = InputParser.parse_conditions(condition, ["x", "t"])
        return LossGenerator.generate(
            dimension=1, has_t=True,
            c_tt_fn=parsed.get("u_tt", lambda x, t: 0.0), c_t_fn=parsed.get("u_t", lambda x, t: 0.0),
            c_xx_fn=parsed.get("u_xx", lambda x, t: 0.0), c_x_fn=parsed.get("u_x", lambda x, t: 0.0),
            c_u_fn=parsed.get("u", lambda x, t: 0.0), f=f, ic_conds=structured["initial"],
            bc_sides={"left": [b for b in structured["boundary"] if b.get("location_clean") == "left"],
                      "right": [b for b in structured["boundary"] if b.get("location_clean") == "right"]}
        )
    elif dimension == 2 and not has_t:
        parsed = InputParser.parse_coeffs_2d(coeffs, ["x", "y"])
        f, _ = InputParser.parse_source(source_term, ["x", "y"])
        structured = InputParser.parse_conditions(condition, ["x", "y"])
        return LossGenerator.generate(
            dimension=2, has_t=False,
            c_xx_fn=parsed.get("u_xx", lambda x, y: 0.0), c_yy_fn=parsed.get("u_yy", lambda x, y: 0.0),
            c_xy_fn=parsed.get("u_xy", lambda x, y: 0.0), c_x_fn=parsed.get("u_x", lambda x, y: 0.0),
            c_y_fn=parsed.get("u_y", lambda x, y: 0.0), c_u_fn=parsed.get("u", lambda x, y: 0.0), f=f,
            bc_sides={s: [b for b in structured["boundary"] if b.get("location_clean") == s] for s in ["left", "right", "bottom", "top"]}
        )
    elif dimension == 2 and has_t:
        parsed = InputParser.parse_coeffs_2d_transient(coeffs, ["x", "y", "t"])
        f, _ = InputParser.parse_source(source_term, ["x", "y", "t"])
        structured = InputParser.parse_conditions(condition, ["x", "y", "t"])
        return LossGenerator.generate(
            dimension=2, has_t=True,
            c_tt_fn=parsed.get("u_tt", lambda x, y, t: 0.0), c_t_fn=parsed.get("u_t", lambda x, y, t: 0.0),
            c_xx_fn=parsed.get("u_xx", lambda x, y, t: 0.0), c_yy_fn=parsed.get("u_yy", lambda x, y, t: 0.0),
            c_xy_fn=parsed.get("u_xy", lambda x, y, t: 0.0), c_x_fn=parsed.get("u_x", lambda x, y: 0.0),
            c_y_fn=parsed.get("u_y", lambda x, y, t: 0.0), c_u_fn=parsed.get("u", lambda x, y, t: 0.0), f=f,
            ic_conds=structured["initial"],
            bc_sides={s: [b for b in structured["boundary"] if b.get("location_clean") == s] for s in ["left", "right", "bottom", "top"]}
        )

# ===== 5. 多线程训练器 =====
class PINNTrainerThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, float, object, object)
    finished_signal = pyqtSignal()
    def __init__(self, problem_name: str, total_epochs: int = 1500, chunk_epochs: int = 50):
        super().__init__()
        self.problem_name = problem_name
        self.problem = PROBLEMS[problem_name]
        self.total_epochs = total_epochs
        self.chunk_epochs = chunk_epochs
        self._is_running = True
    def stop(self):
        self._is_running = False
    def run(self):
        self.log_signal.emit(f"🚀 初始化场景 [{self.problem_name}] 核心求解器...")
        result = solve_pde(
            dimension=self.problem["dimension"], order=self.problem["order"],
            has_t=self.problem["has_t"], coeffs=self.problem["coeffs"],
            source_term=self.problem["source_term"], domain=self.problem["domain"],
            condition=self.problem["condition"],
        )
        exact_func = result.get("exact_solution")
        loss_functions = build_loss_functions(self.problem)
        model = build_model(
            coeffs=self.problem["coeffs"], source_term=self.problem["source_term"],
            conditions=self.problem["condition"], has_t=self.problem["has_t"],
            dimension=self.problem["dimension"], verbose=False,
        )
        domain = self.problem["domain"]
        sampler = DomainSampler(domain["x"], domain.get("y"), domain.get("t"))
        trainer = PINNTrainer(
            model=model, loss_functions=loss_functions,
            optimizer="adam", lr=1e-3, scheduler="plateau", device="cpu",
        )
        self.log_signal.emit("✅ 组件就绪，开启 Adam 增量优化流程...")
        current_epoch = 0
        while current_epoch < self.total_epochs and self._is_running:
            step = min(self.chunk_epochs, self.total_epochs - current_epoch)
            history = trainer.train(n_epochs=step, sampler=sampler, n_interior=1000, n_boundary_per_side=50, n_initial=200, verbose=False)
            current_epoch += step
            latest_loss = history['total_loss'][-1]
            self.log_signal.emit(f"Epoch [{current_epoch:04d}/{self.total_epochs}] | Total Loss: {latest_loss:.4e}")
            self.progress_signal.emit(current_epoch, latest_loss, model, exact_func)
        if not self._is_running:
            self.log_signal.emit("⏹ [后台线程] 已安全中断。")
        else:
            self.log_signal.emit("✅ [后台线程] 训练全部完成！")
        self.finished_signal.emit()

# ===== 6. 主测试窗口 =====
class GUITestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PINN Scientific PDE Platform")
        self.resize(1200, 900)
        self.trainer_thread = None
        self.problem_keys = ["1d_steady", "1d_transient", "2d_steady", "2d_transient"]
        # 主界面容器 (设置背景 QSS)
        main_container = QWidget()
        main_container.setObjectName("MainContainer")
        self.setCentralWidget(main_container)
        main_layout = QVBoxLayout(main_container)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)
        # --------------------------------------------------------------------
        # 卡片 1: 顶部 Header (含 Logo + 标题 + 场景选择 + 按钮)
        # --------------------------------------------------------------------
        header_card = QFrame()
        header_card.setProperty("class", "CardFrame")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(12, 10, 12, 10)
        # 放入绘制的 Logo Widget
        self.logo = PINNLogoWidget()
        header_layout.addWidget(self.logo)
        # 应用标题与子标题
        title_box = QVBoxLayout()
        title_main = QLabel("PINN PDE Solver")
        title_main.setStyleSheet("font-weight: 700; font-size: 16px; color: #0f172a;")
        title_sub = QLabel("Physics-Informed Neural Network Scientific Platform")
        title_sub.setStyleSheet("font-size: 11px; color: #64748b;")
        title_box.addWidget(title_main)
        title_box.addWidget(title_sub)
        header_layout.addLayout(title_box)
        header_layout.addSpacing(20)
        # 场景选择下拉框
        self.combo = QComboBox()
        self.combo.addItems([
            "1D Steady (稳态)",
            "1D Transient (含时)",
            "2D Steady (稳态)",
            "2D Transient (含时)"
        ])
        self.combo.setMinimumWidth(180)
        header_layout.addWidget(self.combo)
        header_layout.addStretch()
        # 开始/停止按钮
        self.btn_start = QPushButton("▶ 开始训练")
        self.btn_start.setObjectName("StartBtn")
        self.btn_start.clicked.connect(self.start_training)
        header_layout.addWidget(self.btn_start)
        self.btn_stop = QPushButton("⏹ 停止训练")
        self.btn_stop.setObjectName("StopBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_training)
        header_layout.addWidget(self.btn_stop)
        main_layout.addWidget(header_card)
        # --------------------------------------------------------------------
        # 卡片 2: 中部日志区域
        # --------------------------------------------------------------------
        log_card = QFrame()
        log_card.setProperty("class", "CardFrame")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 8, 12, 10)
        lbl_log = QLabel("训练日志与收敛状态")
        lbl_log.setObjectName("SectionTitle")
        log_layout.addWidget(lbl_log)
        self.log_text = QTextEdit()
        self.log_text.setObjectName("LogTerminal")
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(110)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_card)
        # --------------------------------------------------------------------
        # 卡片 3: 下部绘图交互区域
        # --------------------------------------------------------------------
        plot_card = QFrame()
        plot_card.setProperty("class", "CardFrame")
        plot_layout = QVBoxLayout(plot_card)
        plot_layout.setContentsMargins(12, 8, 12, 12)
        lbl_plot = QLabel("实时解预测分布与误差可视化")
        lbl_plot.setObjectName("SectionTitle")
        plot_layout.addWidget(lbl_plot)
        self.stacked_widget = QStackedWidget()
        self.widget_1d_steady = Steady1DPlotWidget()
        self.widget_1d_transient = Transient1DPlotWidget()
        self.widget_2d_steady = Steady2DPlotWidget()
        self.widget_2d_transient = Transient2DPlotWidget()
        self.stacked_widget.addWidget(self.widget_1d_steady)     # 0
        self.stacked_widget.addWidget(self.widget_1d_transient)  # 1
        self.stacked_widget.addWidget(self.widget_2d_steady)     # 2
        self.stacked_widget.addWidget(self.widget_2d_transient)  # 3
        plot_layout.addWidget(self.stacked_widget)
        main_layout.addWidget(plot_card)
        self.combo.currentIndexChanged.connect(self.on_scenario_changed)
    def append_log(self, msg: str):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    def on_scenario_changed(self, index: int):
        self.stacked_widget.setCurrentIndex(index)
        self.append_log(f"🔄 切换求解场景: {self.combo.currentText()}")
    def start_training(self):
        index = self.combo.currentIndex()
        problem_name = self.problem_keys[index]
        self.btn_start.setEnabled(False)
        self.combo.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.append_log("=" * 60)
        self.append_log(f"▶ 启动后台训练线程 | 问题: [{problem_name}]")
        self.trainer_thread = PINNTrainerThread(problem_name=problem_name, total_epochs=1500, chunk_epochs=50)
        self.trainer_thread.log_signal.connect(self.append_log)
        self.trainer_thread.progress_signal.connect(self.update_plot_with_model)
        self.trainer_thread.finished_signal.connect(self.on_training_finished)
        self.trainer_thread.start()
    def stop_training(self):
        if self.trainer_thread and self.trainer_thread.isRunning():
            self.append_log("⏳ 正在中止线程...")
            self.trainer_thread.stop()
            self.btn_stop.setEnabled(False)
    def on_training_finished(self):
        self.btn_start.setEnabled(True)
        self.combo.setEnabled(True)
        self.btn_stop.setEnabled(False)
    def update_plot_with_model(self, epoch: int, loss_val: float, model: torch.nn.Module, exact_func):
        index = self.combo.currentIndex()
        problem_name = self.problem_keys[index]
        problem = PROBLEMS[problem_name]
        model.eval()
        if index == 0:
            def true_func_1d(x):
                return torch.tensor([exact_func(xi.item()) for xi in x], dtype=torch.float32).reshape(-1, 1) if exact_func else None
            self.widget_1d_steady.set_data(model, x_range=(0, 1), true_func=true_func_1d)
        elif index == 1:
            self.widget_1d_transient.set_data(model, x_range=(0, 1), t_range=tuple(problem["domain"]["t"]), exact_func=exact_func)
        elif index == 2:
            def true_func_2d(pts):
                return torch.tensor([exact_func(xi.item(), yi.item()) for xi, yi in pts], dtype=torch.float32).reshape(-1, 1) if exact_func else None
            self.widget_2d_steady.set_data(model, x_range=(0, 1), y_range=(0, 1), true_func=true_func_2d)
        elif index == 3:
            self.widget_2d_transient.set_data(model, x_range=(0, 1), y_range=(0, 1), t_range=tuple(problem["domain"]["t"]), exact_func=exact_func)

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyleSheet(LIGHT_BLUE_QSS)
    window = GUITestWindow()
    window.show()
    sys.exit(app.exec_())
