from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QTextEdit, QLineEdit, QMessageBox, QSpinBox, QGroupBox
)
from PyQt5.QtCore import Qt
from .base_widgets import ClearableListWidget, InputPage, PreviewLabel, SolverThread
from .plot_window_page import PlotWindow

class SolverPage(InputPage):
    def __init__(self, back_to_menu_cb, parent=None):
        super().__init__(back_to_menu_cb, parent)
        self.current_result = None
        self.solver_thread = None
        self.init_ui()
        self.on_type_changed()
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        # 1. 顶部：问题类型、阶数、源项与定义域设置
        self.top_group = QGroupBox("问题控制参数")
        top_layout = QHBoxLayout(self.top_group)
        top_layout.addWidget(QLabel("<b>问题类型:</b>"))
        self.type_combo = QComboBox()
        self.type_combo.setCursor(Qt.PointingHandCursor)
        self.type_combo.addItems(["1D 稳态", "1D 含时", "2D 稳态", "2D 含时"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        top_layout.addWidget(self.type_combo)
        top_layout.addWidget(QLabel("<b>方程阶数 (Order):</b>"))
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        self.order_spin.valueChanged.connect(self.on_order_changed)
        top_layout.addWidget(self.order_spin)
        top_layout.addWidget(QLabel("<b>源项 f:</b>"))
        self.source_input = QLineEdit("0")
        self.source_input.setPlaceholderText("如 0, sin(pi*x), x**2+y**2")
        self.source_input.textChanged.connect(self.update_preview)
        top_layout.addWidget(self.source_input)
        top_layout.addStretch()
        main_layout.addWidget(self.top_group)
        # 2. 中间两列：方程系数项配置 与 定解条件配置
        config_layout = QHBoxLayout()
        # 左侧：方程系数项
        term_box = QVBoxLayout()
        self.term_label = QLabel("<b>方程系数项列表:</b>")
        term_box.addWidget(self.term_label)
        self.term_list_widget = ClearableListWidget(self)
        term_box.addWidget(self.term_list_widget)
        term_btn_layout = QHBoxLayout()
        add_term_btn = QPushButton("添加系数项")
        del_term_btn = QPushButton("删除选中项")
        add_term_btn.clicked.connect(self.open_add_term_dialog)
        del_term_btn.clicked.connect(self.delete_selected_term)
        term_btn_layout.addWidget(add_term_btn)
        term_btn_layout.addWidget(del_term_btn)
        term_box.addLayout(term_btn_layout)
        # 右侧：定解条件框
        cond_box = QVBoxLayout()
        self.cond_label = QLabel("<b>定解条件列表:</b>")
        cond_box.addWidget(self.cond_label)
        self.cond_list_widget = ClearableListWidget(self)
        cond_box.addWidget(self.cond_list_widget)
        cond_btn_layout = QHBoxLayout()
        add_cond_btn = QPushButton("添加条件")
        del_cond_btn = QPushButton("删除选中条件")
        add_cond_btn.clicked.connect(self.open_add_cond_dialog)
        del_cond_btn.clicked.connect(self.delete_selected_cond)
        cond_btn_layout.addWidget(add_cond_btn)
        cond_btn_layout.addWidget(del_cond_btn)
        cond_box.addLayout(cond_btn_layout)
        config_layout.addLayout(term_box)
        config_layout.addLayout(cond_box)
        main_layout.addLayout(config_layout)
        # 3. 拼接方程预览框
        self.preview_label = QLabel("<b>方程预览:</b>")
        main_layout.addWidget(self.preview_label)
        self.eq_preview = PreviewLabel(self, font_size=9)
        main_layout.addWidget(self.eq_preview)
        # 4. 工作日志框
        self.log_label = QLabel("<b>求解工作日志:</b>")
        main_layout.addWidget(self.log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)
        # 5. 结果显示框
        self.result_label = QLabel("<b>解析解结果输出:</b>")
        main_layout.addWidget(self.result_label)
        self.result_latex = PreviewLabel(self, font_size=9)
        self.result_latex.set_latex(r"\text{等待求解...}")
        main_layout.addWidget(self.result_latex)
        # 6. 底部控制按钮行
        bottom_layout = QHBoxLayout()
        self.solve_btn = QPushButton("开始求解")
        self.solve_btn.clicked.connect(self.start_solving)
        self.plot_btn = QPushButton("绘制解函数")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self.open_plot_window)
        back_btn = QPushButton("返回主菜单")
        back_btn.clicked.connect(self.back_to_menu_cb)
        bottom_layout.addWidget(self.solve_btn)
        bottom_layout.addWidget(self.plot_btn)
        bottom_layout.addWidget(back_btn)
        main_layout.addLayout(bottom_layout)
    def start_solving(self):
        if not self.terms:
            QMessageBox.warning(self, "警告", "请先添加方程系数项！")
            return
        p_type = self.type_combo.currentText()
        order = self.order_spin.value()
        dimension = 1 if "1D" in p_type else 2
        has_t = "含时" in p_type
        # 1. coeffs 格式转换
        if dimension == 1 and not has_t:
            coeffs_list = [0.0] * (order + 1)
            for d in range(order + 1):
                key = "u" if d == 0 else "u" + "'" * d
                if key in self.terms:
                    try:
                        coeffs_list[d] = float(self.terms[key])
                    except ValueError:
                        coeffs_list[d] = self.terms[key]
            coeffs_param = coeffs_list
        else:
            coeffs_dict = {}
            for k, v in self.terms.items():
                try:
                    coeffs_dict[k] = float(v)
                except ValueError:
                    coeffs_dict[k] = v
            coeffs_param = coeffs_dict
        # 2. 构建与 function_factory 完全相符的配置
        problem_config = {
            "dimension": dimension,
            "has_t": has_t,
            "order": order,
            "coeffs": coeffs_param,
            "source_term": self.source_input.text().strip() or "0",
            "conditions": self.conditions,
            "domain": {
                "x": [0.0, 1.0],
                "y": [0.0, 1.0] if dimension == 2 else None,
                "t": [0.0, 1.0] if has_t else None
            }
        }
        self.solve_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.log_text.clear()
        self.result_latex.set_latex(r"\text{正在调用核心引擎计算中...}")
        # 3. 启动后台后台线程求解
        self.solver_thread = SolverThread(problem_config)
        self.solver_thread.log_signal.connect(self.log_text.append)
        self.solver_thread.finished_signal.connect(self.on_solve_finished)
        self.solver_thread.error_signal.connect(self.on_solve_error)
        self.solver_thread.start()
    def on_solve_finished(self, result_data):
        self.solve_btn.setEnabled(True)
        self.current_result = result_data
        self.result_latex.set_latex(str(result_data['exact_expr']))
        self.plot_btn.setEnabled(True)
        QMessageBox.information(self, "成功", "方程求解成功！")
    def on_solve_error(self, err_msg):
        self.solve_btn.setEnabled(True)
        self.plot_btn.setEnabled(False)
        self.result_latex.set_latex(r"\text{无精确解或无法求解}")
        self.log_text.append(f"<font color='red'>{err_msg}</font>")
        QMessageBox.warning(self, "求解失败", err_msg)
    def open_plot_window(self):
        if self.current_result:
            plot_dlg = PlotWindow(self.current_result, self)
            plot_dlg.exec_()
