# ui/pages/equation_input_page.py
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QPushButton,
    QLineEdit, QMessageBox, QSpinBox, QGroupBox, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from .base_widgets import ClearableListWidget, InputPage, PreviewLabel
from src.network_factory import suggest_network

class PinnInputPage(InputPage):
    """第一步：输入方程与定解条件页"""
    equation_configured = pyqtSignal(dict)
    def __init__(self, back_to_menu_cb, parent=None):
        super().__init__(back_to_menu_cb, parent)
        self.init_ui()
        self.on_type_changed()
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        # ===== 1. 顶部：问题控制参数 =====
        self.top_group = QGroupBox("问题控制参数")
        top_main_layout = QVBoxLayout(self.top_group)
        # ---- 第一行：类型、阶数、源项 ----
        row1_layout = QHBoxLayout()
        row1_layout.addWidget(QLabel("<b>问题类型:</b>"))
        self.type_combo = QComboBox()
        self.type_combo.setCursor(Qt.PointingHandCursor)
        self.type_combo.addItems(["1D 稳态", "1D 含时", "2D 稳态", "2D 含时"])
        self.type_combo.currentIndexChanged.connect(self.on_type_changed)
        row1_layout.addWidget(self.type_combo)
        row1_layout.addWidget(QLabel("<b>方程阶数:</b>"))
        self.order_spin = QSpinBox()
        self.order_spin.setRange(1, 6)
        self.order_spin.setValue(2)
        self.order_spin.valueChanged.connect(self.on_order_changed)
        row1_layout.addWidget(self.order_spin)
        row1_layout.addWidget(QLabel("<b>源项 f:</b>"))
        self.source_input = QLineEdit("0")
        self.source_input.setPlaceholderText("如 0, sin(pi*x), x**2+y**2")
        self.source_input.textChanged.connect(self.update_preview)
        self.source_input.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row1_layout.addWidget(self.source_input)
        top_main_layout.addLayout(row1_layout)
        # ---- 第二行：定义域范围 ----
        row2_container = QWidget()
        row2_layout = QHBoxLayout(row2_container)
        row2_layout.setContentsMargins(0, 0, 0, 0)
        row2_layout.setSpacing(15)
        row2_layout.addWidget(QLabel("<b>定义域范围:</b>"))
        def create_axis_range_widget(label_text, min_val, max_val):
            container = QWidget()
            container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(4)
            lbl = QLabel(f"{label_text}:")
            min_input = QLineEdit(min_val)
            min_input.setFixedWidth(50)
            min_input.setAlignment(Qt.AlignCenter)
            sep_lbl = QLabel("--")
            sep_lbl.setStyleSheet("color: gray;")
            sep_lbl.setAlignment(Qt.AlignCenter)
            max_input = QLineEdit(max_val)
            max_input.setFixedWidth(50)
            max_input.setAlignment(Qt.AlignCenter)
            h_layout.addWidget(lbl)
            h_layout.addWidget(min_input)
            h_layout.addWidget(sep_lbl)
            h_layout.addWidget(max_input)
            return container, min_input, max_input
        self.x_widget, self.x_min_input, self.x_max_input = create_axis_range_widget("x", "0", "1")
        self.y_widget, self.y_min_input, self.y_max_input = create_axis_range_widget("y", "0", "1")
        self.t_widget, self.t_min_input, self.t_max_input = create_axis_range_widget("t", "0", "1")
        row2_layout.addWidget(self.x_widget)
        row2_layout.addWidget(self.y_widget)
        row2_layout.addWidget(self.t_widget)
        row2_layout.addStretch()
        row2_container.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        top_main_layout.addWidget(row2_container)
        main_layout.addWidget(self.top_group)
        # 2. 中间：项与边界配置
        config_layout = QHBoxLayout()
        # 左侧项
        term_box = QVBoxLayout()
        term_box.addWidget(QLabel("<b>方程系数项列表:</b>"))
        self.term_list_widget = ClearableListWidget(self)
        term_box.addWidget(self.term_list_widget)
        tb_layout = QHBoxLayout()
        add_t_btn = QPushButton("添加系数项")
        del_t_btn = QPushButton("删除选中项")
        add_t_btn.clicked.connect(self.open_add_term_dialog)
        del_t_btn.clicked.connect(self.delete_selected_term)
        tb_layout.addWidget(add_t_btn)
        tb_layout.addWidget(del_t_btn)
        term_box.addLayout(tb_layout)
        # 右侧条件
        cond_box = QVBoxLayout()
        cond_box.addWidget(QLabel("<b>定解条件列表:</b>"))
        self.cond_list_widget = ClearableListWidget(self)
        cond_box.addWidget(self.cond_list_widget)
        cb_layout = QHBoxLayout()
        add_c_btn = QPushButton("添加条件")
        del_c_btn = QPushButton("删除选中条件")
        add_c_btn.clicked.connect(self.open_add_cond_dialog)
        del_c_btn.clicked.connect(self.delete_selected_cond)
        cb_layout.addWidget(add_c_btn)
        cb_layout.addWidget(del_c_btn)
        cond_box.addLayout(cb_layout)
        config_layout.addLayout(term_box)
        config_layout.addLayout(cond_box)
        main_layout.addLayout(config_layout)
        # 3. 预览
        main_layout.addWidget(QLabel("<b>实时方程预览:</b>"))
        self.eq_preview = PreviewLabel(self, font_size=9)
        main_layout.addWidget(self.eq_preview)
        # 4. 底部按钮
        bottom_layout = QHBoxLayout()
        back_btn = QPushButton("返回")
        back_btn.clicked.connect(self.back_to_menu_cb)
        self.next_btn = QPushButton("下一页")
        self.next_btn.clicked.connect(self.proceed_to_next_step)
        bottom_layout.addWidget(back_btn)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.next_btn)
        main_layout.addLayout(bottom_layout)

    def on_type_changed(self):
        p_type = self.type_combo.currentText()
        has_y = "2D" in p_type
        has_t = "含时" in p_type
        if "1D 稳态" in p_type:
            self.order_spin.setEnabled(True)
        else:
            self.order_spin.setValue(2)
            self.order_spin.setEnabled(False)
        self.x_widget.show()
        self.y_widget.setVisible(has_y)
        self.t_widget.setVisible(has_t)
        self.clear_all_inputs()

    def proceed_to_next_step(self):
        """校验条件，构建标准配置字典并传递给下一页"""
        if not self.terms:
            QMessageBox.warning(self, "警告", "条件不齐全：请先至少添加一项方程系数！")
            return
        p_type = self.type_combo.currentText()
        order = self.order_spin.value()
        dimension = 1 if "1D" in p_type else 2
        has_t = "含时" in p_type
        try:
            x_min = float(self.x_min_input.text().strip() or "0")
            x_max = float(self.x_max_input.text().strip() or "1")
            if x_min >= x_max:
                QMessageBox.warning(self, "错误", "x 范围: 最小值必须小于最大值！")
                return
        except ValueError:
            QMessageBox.warning(self, "错误", "x 范围必须为有效数字！")
            return
        domain = {"x": [x_min, x_max]}
        if dimension == 2:
            try:
                y_min = float(self.y_min_input.text().strip() or "0")
                y_max = float(self.y_max_input.text().strip() or "1")
                if y_min >= y_max:
                    QMessageBox.warning(self, "错误", "y 范围: 最小值必须小于最大值！")
                    return
                domain["y"] = [y_min, y_max]
            except ValueError:
                QMessageBox.warning(self, "错误", "y 范围必须为有效数字！")
                return
        if has_t:
            try:
                t_min = float(self.t_min_input.text().strip() or "0")
                t_max = float(self.t_max_input.text().strip() or "1")
                if t_min >= t_max:
                    QMessageBox.warning(self, "错误", "t 范围: 最小值必须小于最大值！")
                    return
                domain["t"] = [t_min, t_max]
            except ValueError:
                QMessageBox.warning(self, "错误", "t 范围必须为有效数字！")
                return
        if dimension == 1 and not has_t:
            coeffs_param = [0.0] * (order + 1)
            for d in range(order + 1):
                key = "u" if d == 0 else "u" + "'" * d
                coeffs_param[d] = self.terms.get(key, 0.0)
        else:
            coeffs_param = self.terms.copy()
        if dimension == 1 and not has_t:
            if len(self.conditions) != order:
                QMessageBox.warning(self, "警告", f"1D 稳态需要恰好 {order} 个定解条件！")
                return
        else:
            required_slots = self.get_all_possible_slots()
            existing_sides = []
            for c in self.conditions:
                if c.get("side") == "initial":
                    existing_sides.append(f"initial_{c.get('derivative', 0)}")
                else:
                    existing_sides.append(c.get("side"))
            missing = [s for s in required_slots if s not in existing_sides]
            if missing:
                QMessageBox.warning(self, "警告", f"缺少条件: {', '.join(missing)}")
                return
        if dimension == 2 and not has_t:
            allowed_keys = {"u_xx", "u_yy"}
            for key in self.terms.keys():
                if key not in allowed_keys:
                    QMessageBox.warning(self, "错误", 
                        f"二维稳态仅支持 u_xx 和 u_yy 项，不能包含 '{key}'")
                    return
            if "u_xx" not in self.terms or "u_yy" not in self.terms:
                QMessageBox.warning(self, "错误", "二维稳态需要同时包含 u_xx 和 u_yy 两项")
                return
            try:
                v_xx = float(self.terms["u_xx"]) if self.terms["u_xx"] else 0.0
                v_yy = float(self.terms["u_yy"]) if self.terms["u_yy"] else 0.0
                if abs(v_xx - v_yy) > 1e-12:
                    QMessageBox.warning(self, "错误", 
                        f"广义泊松要求 u_xx 和 u_yy 的系数相等！当前: u_xx={v_xx}, u_yy={v_yy}")
                    return
            except ValueError:
                pass
        config = {
            "dimension": dimension,
            "order": order,
            "has_t": has_t,
            "coeffs": coeffs_param,
            "source_term": self.source_input.text().strip() or "0",
            "domain": domain,
            "condition": self.conditions.copy(),
        }
        try:
            net_config = suggest_network(
                coeffs=coeffs_param,
                source_term=self.source_input.text().strip() or "0",
                conditions=self.conditions,
                has_t=has_t,
                dimension=dimension,
                verbose=False,
            )
            config["auto_network_preview"] = net_config
        except:
            pass
        self.equation_configured.emit(config)
