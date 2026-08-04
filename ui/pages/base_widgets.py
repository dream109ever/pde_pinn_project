import re
import math
from typing import Optional
from PyQt5.QtWidgets import QWidget, QSpinBox, QVBoxLayout, QComboBox, QListWidget, QLineEdit, QMessageBox, QDialog, QFormLayout, QDialogButtonBox, QTextEdit
from PyQt5.QtCore import Qt, QPointF, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QLinearGradient, QRadialGradient, QPainterPath
from src.function_factory import solve_pde
from ui.theme_manager import ThemeManager

class BasePage(QWidget):
    """支持全局配色切换的基础页面类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        ThemeManager.instance().theme_changed.connect(lambda _: self.update_styles())
        QTimer.singleShot(0, self.update_styles)
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0: return
        # 获取当前主题配色数据
        theme = ThemeManager.instance().current
        # 1. 动态渐变底色
        bg_grad = QLinearGradient(0, 0, w, h)
        bg_grad.setColorAt(0.0, theme.bg_start)
        bg_grad.setColorAt(0.5, theme.bg_mid)
        bg_grad.setColorAt(1.0, theme.bg_end)
        painter.fillRect(self.rect(), bg_grad)
        # 2. 顶部局部柔和高光
        radial_light = QRadialGradient(w * 0.3, h * 0.2, max(w, h) * 0.8)
        light_alpha = 70 if theme.is_dark else 170
        radial_light.setColorAt(0.0, QColor(255, 255, 255, light_alpha))
        radial_light.setColorAt(1.0, QColor(255, 255, 255, 0))
        painter.fillRect(self.rect(), radial_light)
        # 3. 动态正弦波流线
        wave_configs = [
            (35, 0.007, 0.0, 0.28, 110, 1.5, False),
            (55, 0.005, 1.2, 0.45, 80, 2.0, False),
            (40, 0.009, 2.5, 0.62, 110, 1.2, True),
            (65, 0.004, 4.0, 0.78, 75, 2.2, False),
        ]
        for amp, freq, phase, y_ratio, alpha, pen_w, is_dashed in wave_configs:
            path = QPainterPath()
            base_y = h * y_ratio
            path.moveTo(0, base_y + amp * math.sin(phase))
            step = 8
            for x in range(step, w + step, step):
                y = base_y + amp * math.sin(freq * x + phase)
                path.lineTo(x, y)
            pen = QPen(theme.wave_color)
            pen.setWidthF(pen_w)
            if is_dashed:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        # 4. 散落的流场采样节点
        painter.setBrush(theme.node_color)
        painter.setPen(Qt.NoPen)
        node_positions = [
            (w * 0.15, h * 0.29), (w * 0.32, h * 0.42), (w * 0.55, h * 0.35),
            (w * 0.68, h * 0.65), (w * 0.82, h * 0.50), (w * 0.42, h * 0.72)
        ]
        for nx, ny in node_positions:
            painter.drawEllipse(QPointF(nx, ny), 3.0, 3.0)
        super().paintEvent(event)
    def update_styles(self):
        theme = ThemeManager.instance().current
        if hasattr(self, 'eq_preview'): self.eq_preview.set_color(theme.text_primary)
        if hasattr(self, 'result_latex'): self.result_latex.set_color(theme.text_primary)
        input_bg = "rgba(15, 23, 42, 0.55)" if theme.is_dark else "rgba(255, 255, 255, 0.85)"
        list_bg = "rgba(15, 23, 42, 0.45)" if theme.is_dark else "rgba(255, 255, 255, 0.75)"
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme.card_bg};
            }}
            QLabel {{
                color: {theme.text_primary};
                background: transparent;
            }}
            QGroupBox {{
                color: {theme.text_primary};
                font-weight: bold;
                border: 1px solid {theme.btn_border};
                border-radius: 8px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 4px;
                
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                background-color: {input_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QComboBox {{
                background-color: {input_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 4px 8px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid {theme.btn_border};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: url("ui/icons/arrow_down_disabled.png");
                height: 8px;
                width: 8px;
            }}
            QComboBox::down-arrow:hover {{
                image: url("ui/icons/arrow_down.png");
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.card_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                selection-background-color: {theme.btn_hover_bg};
                selection-color: {theme.btn_hover_text};
            }}
            QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                background: transparent;
                width: 16px;
                border: none;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                border-top-right-radius: 6px;
                border-left: 1px solid {theme.btn_border};
                border-bottom: 1px solid {theme.btn_border};
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                border-bottom-right-radius: 6px;
                border-left: 1px solid {theme.btn_border};
            }}
            QAbstractSpinBox::up-arrow, QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
                image: url("ui/icons/arrow_up_disabled.png");
                height: 8px;
                width: 8px;
            }}
            QAbstractSpinBox::up-arrow:hover {{
                image: url("ui/icons/arrow_up.png");
            }}
            QAbstractSpinBox::down-arrow, QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
                image: url("ui/icons/arrow_down_disabled.png");
                height: 8px;
                width: 8px;
            }}
            QComboBox::down-arrow:on, QAbstractSpinBox::down-arrow:hover {{
                image: url("ui/icons/arrow_down.png");
            }}
            QListWidget, QTextEdit {{
                background-color: {list_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 4px;
            }}
            QListWidget::item {{
                border-radius: 4px;
                padding: 2px 4px;
            }}
            QListWidget::item:selected {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
            }}
            QPushButton {{
                background-color: {theme.btn_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
                border: 1px solid {theme.btn_hover_border};
            }}
            QPushButton:disabled {{
                background-color: rgba(150, 150, 150, 0.15);
                color: {theme.text_secondary};
                border: 1px solid transparent;
            }}
            QScrollArea {{
                background: transparent;
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
            }}
            QScrollArea > QWidget > QWidget {{
                background: transparent;
            }}
        """)

class BaseDialog(QDialog):
    """支持全局配色切换的对话框基类"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.apply_theme()
        ThemeManager.instance().theme_changed.connect(self.apply_theme)
    def apply_theme(self):
        """子类重写此方法以实现自定义样式"""
        theme = ThemeManager.instance().current
        input_bg = "rgba(15, 23, 42, 0.6)" if theme.is_dark else "rgba(255, 255, 255, 0.9)"
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {theme.card_bg};
            }}
            QLabel {{
                color: {theme.text_primary};
                font-weight: bold;
            }}
            QGroupBox {{
                font-weight: bold;
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 8px;
                margin-top: 6px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QLineEdit, QComboBox {{
                background-color: {input_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 5px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid {theme.btn_border};
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
                background: transparent;
            }}
            QComboBox::down-arrow {{
                image: url("ui/icons/arrow_down_disabled.png");
                height: 8px;
                width: 8px;
            }}
            QComboBox::down-arrow:on, QComboBox::down-arrow:hover {{
                image: url("ui/icons/arrow_down.png");
            }}
            QComboBox QAbstractItemView {{
                background-color: {theme.card_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                selection-background-color: {theme.btn_hover_bg};
                selection-color: {theme.btn_hover_text};
            }}
            QPushButton {{
                background-color: {theme.btn_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {theme.btn_hover_bg};
                border: 1px solid {theme.btn_hover_border};
                color: {theme.btn_hover_text};
            }}
            QPushButton:pressed {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
            }}
        """)

class ClearableListWidget(QListWidget):
    """点击列表空白处自动取消高亮选中的 QListWidget"""
    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            self.clearSelection()
        super().mousePressEvent(event)

class AddTermDialog(BaseDialog):
    """添加方程系数项弹窗：仅展示当前未被添加的可用项"""
    def __init__(self, avail_terms: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("添加方程系数项")
        self.resize(320, 150)
        self.selected_term = None
        self.coefficient = "1.0"
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.term_combo = QComboBox()
        self.term_combo.setCursor(Qt.PointingHandCursor)
        for term_key, display_name in avail_terms:
            self.term_combo.addItem(display_name, userData=term_key)
        self.coeff_input = QLineEdit("1.0")
        self.coeff_input.setPlaceholderText("可填数值或表达式，如 1.0, x**2, sin(x)")
        form.addRow("选择微分项:", self.term_combo)
        form.addRow("系数 (Coeff):", self.coeff_input)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def accept(self):
        self.selected_term = self.term_combo.currentData()
        self.coefficient = self.coeff_input.text().strip() or "1.0"
        super().accept()

class AddConditionDialog(BaseDialog):
    """添加定解条件弹窗：按 1D/2D/含时 差异化配置，且位置/侧边严格互斥"""
    def __init__(self, p_type: str, order: int, avail_slots: list, parent=None):
        super().__init__(parent)
        self.p_type = p_type
        self.order = order
        self.avail_slots = avail_slots
        self.setWindowTitle("添加定解条件")
        self.resize(380, 220)
        self.condition_data = None
        self.init_ui()
    def init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        if "1D 稳态" in self.p_type:
            self.point_input = QLineEdit("0.0")
            self.deriv_combo = QComboBox()
            self.deriv_combo.setCursor(Qt.PointingHandCursor)
            for d in range(self.order):
                if d == 0: deriv_label = "u"
                else: deriv_label = "u" + "'" * d
                self.deriv_combo.addItem(f"{d} 阶导数 ({deriv_label})", userData=d)
            self.val_input = QLineEdit("0.0")
            form.addRow("作用点 (x):", self.point_input)
            form.addRow("导数阶数:", self.deriv_combo)
            form.addRow("条件值 (Value):", self.val_input)
        else:
            self.side_combo = QComboBox()
            self.side_combo.setCursor(Qt.PointingHandCursor)
            side_map = {
                "initial_pos": "初始位置 (initial: u(x,0))",
                "initial_vel": "初始速度 (initial: u_t(x,0))",
                "left": "左边界 (left: x=x_min)",
                "right": "右边界 (right: x=x_max)",
                "bottom": "下边界 (bottom: y=y_min)",
                "top": "上边界 (top: y=y_max)"
            }
            for slot in self.avail_slots:
                self.side_combo.addItem(side_map.get(slot, slot), userData=slot)
            self.type_combo = QComboBox()
            self.type_combo.setCursor(Qt.PointingHandCursor)
            self.type_combo.addItems(["dirichlet", "neumann"])
            self.val_input = QLineEdit("0.0")
            self.val_input.setPlaceholderText("如 0.0, sin(pi*x), 1.0")
            form.addRow("边界/初始位置:", self.side_combo)
            form.addRow("条件类型 (Type):", self.type_combo)
            form.addRow("条件值 (Value):", self.val_input)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    def on_accept(self):
        if "1D 稳态" in self.p_type:
            try:
                pt_val = float(self.point_input.text().strip())
            except ValueError:
                QMessageBox.warning(self, "错误", "作用点 x 必须为有效数字！")
                return
            self.condition_data = {
                "point": pt_val,
                "derivative": self.deriv_combo.currentData(),
                "value": self.val_input.text().strip() or "0.0"
            }
        else:
            side_key = self.side_combo.currentData()
            cond_type = self.type_combo.currentText()
            val_str = self.val_input.text().strip() or "0.0"
            if side_key == "initial_pos":
                self.condition_data = {"side": "initial", "derivative": 0, "value": val_str}
            elif side_key == "initial_vel":
                self.condition_data = {"side": "initial", "derivative": 1, "value": val_str}
            else:
                self.condition_data = {"side": side_key, "type": cond_type, "value": val_str}
        self.accept()

class InputPage(BasePage):
    """
    方程输入页面的公共基类
    """
    def __init__(self, back_to_menu_cb, parent=None):
        super().__init__(parent)
        self.back_to_menu_cb = back_to_menu_cb
        self.terms = {}       # 方程项 {key: coeff}
        self.conditions = []  # 定解条件列表
        # ---- 公共控件 ----
        self.type_combo: Optional[QComboBox] = None
        self.order_spin: Optional[QSpinBox] = None
        self.source_input: Optional[QLineEdit] = None
        self.term_list_widget: Optional[QListWidget] = None
        self.cond_list_widget: Optional[QListWidget] = None
        # ---- 定义域输入框 ----
        self.x_min_input: Optional[QLineEdit] = None
        self.x_max_input: Optional[QLineEdit] = None
        self.y_min_input: Optional[QLineEdit] = None
        self.y_max_input: Optional[QLineEdit] = None
        self.t_min_input: Optional[QLineEdit] = None
        self.t_max_input: Optional[QLineEdit] = None

    # 抽象方法（子类必须实现）
    def init_ui(self):
        """子类必须重写：创建自己的 UI 布局"""
        raise NotImplementedError("子类必须实现 init_ui()")

    # 公共方法（供子类调用）
    def get_all_possible_terms(self) -> list:
        """根据问题类型与阶数，返回所有合法的 (key, display_name)"""
        p_type = self.type_combo.currentText()
        order = self.order_spin.value()
        if "1D 稳态" in p_type:
            terms = []
            for d in range(order, -1, -1):
                if d == 0:
                    terms.append(("u", "u (零阶项)"))
                else:
                    primes = "'" * d
                    terms.append((f"u{primes}", f"u{primes} ({d}阶导数)"))
            return terms
        elif "1D 含时" in p_type:
            return [
                ("u_tt", "u_tt (二阶时间)"),
                ("u_t", "u_t (一阶时间)"),
                ("u_xx", "u_xx (二阶空间)"),
                ("u_x", "u_x (一阶空间)"),
                ("u", "u (零阶项)")
            ]
        elif "2D 稳态" in p_type:
            return [
                ("u_xx", "u_xx (x方向二阶)"),
                ("u_yy", "u_yy (y方向二阶)"),
            ]
        else:
            return [
                ("u_tt", "u_tt (二阶时间)"),
                ("u_t", "u_t (一阶时间)"),
                ("u_xx", "u_xx (x方向二阶)"),
                ("u_yy", "u_yy (y方向二阶)"),
                ("u_xy", "u_xy (xy混合二阶)"),
                ("u_x", "u_x (x方向一阶)"),
                ("u_y", "u_y (y方向一阶)"),
                ("u", "u (零阶项)")
            ]

    def get_all_possible_slots(self) -> list:
        """根据问题类型返回合法的定解条件侧边/位置列表"""
        p_type = self.type_combo.currentText()
        if "1D 稳态" in p_type:
            return []
        elif "1D 含时" in p_type:
            return ["initial_pos", "initial_vel", "left", "right"]
        elif "2D 稳态" in p_type:
            return ["left", "right", "bottom", "top"]
        else:
            return ["initial", "left", "right", "bottom", "top"]

    def on_type_changed(self):
        p_type = self.type_combo.currentText()
        has_y = "2D" in p_type
        has_t = "含时" in p_type
        if "1D 稳态" in p_type:
            self.order_spin.setEnabled(True)
        else:
            self.order_spin.setValue(2)
            self.order_spin.setEnabled(False)
        self.clear_all_inputs()

    def on_order_changed(self):
        self.clear_all_inputs()

    def clear_all_inputs(self):
        self.terms.clear()
        self.conditions.clear()
        self.term_list_widget.clear()
        self.cond_list_widget.clear()
        self.update_preview()

    def update_preview(self):
        if not self.terms:
            self.eq_preview.set_latex(r"\text{未构建方程}")
            return
        parts = [f"({v}) * {k}" for k, v in self.terms.items()]
        src = self.source_input.text().strip() or "0"
        self.eq_preview.set_latex(" + ".join(parts) + f" = {src}")

    def open_add_term_dialog(self):
        all_terms = self.get_all_possible_terms()
        avail_terms = [item for item in all_terms if item[0] not in self.terms]
        if not avail_terms:
            QMessageBox.information(self, "提示", "当前问题类型的可选微分项已全部添加！")
            return
        dlg = AddTermDialog(avail_terms, self)
        if dlg.exec_():
            term_key = dlg.selected_term
            coeff = dlg.coefficient
            self.terms[term_key] = coeff
            self.term_list_widget.addItem(f"{term_key}  [系数: {coeff}]")
            self.update_preview()

    def delete_selected_term(self):
        row = self.term_list_widget.currentRow()
        if row >= 0:
            item_text = self.term_list_widget.item(row).text()
            term_key = item_text.split("  [")[0]
            self.terms.pop(term_key, None)
            self.term_list_widget.takeItem(row)
            self.update_preview()

    def open_add_cond_dialog(self):
        p_type = self.type_combo.currentText()
        order = self.order_spin.value()
        if "1D 稳态" in p_type:
            if len(self.conditions) >= order:
                QMessageBox.warning(self, "提示", f"当前 1D 稳态方程阶数为 {order} 阶，最多只能添加 {order} 个定解条件！")
                return
            dlg = AddConditionDialog(p_type, order, [], self)
            if dlg.exec_():
                cond = dlg.condition_data
                for exist in self.conditions:
                    if exist.get('point') == cond['point'] and exist.get('derivative') == cond['derivative']:
                        QMessageBox.warning(self, "冲突", f"点 x={cond['point']} 的 {cond['derivative']} 阶导数条件已存在！")
                        return
                self.conditions.append(cond)
                self.cond_list_widget.addItem(f"点 x={cond['point']}, {cond['derivative']}阶导数 = {cond['value']}")
        else:
            all_slots = self.get_all_possible_slots()
            used_slots = []
            for c in self.conditions:
                if c.get("side") == "initial":
                    used_slots.append(f"initial_{c.get('derivative', 0)}")
                else:
                    used_slots.append(c.get("side"))
            avail_slots = [s for s in all_slots if s not in used_slots and f"initial_{s}" not in used_slots]
            if not avail_slots:
                QMessageBox.information(self, "提示", "所有边界/初始条件已设定完毕！")
                return
            dlg = AddConditionDialog(p_type, order, avail_slots, self)
            if dlg.exec_():
                cond = dlg.condition_data
                self.conditions.append(cond)
                if cond.get('side') == 'initial':
                    label_name = "初始位置" if cond.get('derivative') == 0 else "初始速度"
                    display_str = f"[{label_name}] 值: {cond['value']}"
                else:
                    display_str = f"[{cond['side']}] 类型: {cond.get('type', 'initial')} | 值: {cond['value']}"
                self.cond_list_widget.addItem(display_str)

    def delete_selected_cond(self):
        row = self.cond_list_widget.currentRow()
        if row >= 0:
            self.conditions.pop(row)
            self.cond_list_widget.takeItem(row)

    def read_domain(self):
        p_type = self.type_combo.currentText()
        dimension = 1 if "1D" in p_type else 2
        has_t = "含时" in p_type
        domain = {}
        try:
            x_min = float(self.x_min_input.text().strip() or "0")
            x_max = float(self.x_max_input.text().strip() or "1")
            if x_min >= x_max:
                QMessageBox.warning(self, "错误", "x 范围: 最小值必须小于最大值！")
                return None
            domain["x"] = [x_min, x_max]
        except ValueError:
            QMessageBox.warning(self, "错误", "x 范围必须为有效数字！")
            return None
        if dimension == 2:
            try:
                y_min = float(self.y_min_input.text().strip() or "0")
                y_max = float(self.y_max_input.text().strip() or "1")
                if y_min >= y_max:
                    QMessageBox.warning(self, "错误", "y 范围: 最小值必须小于最大值！")
                    return None
                domain["y"] = [y_min, y_max]
            except ValueError:
                QMessageBox.warning(self, "错误", "y 范围必须为有效数字！")
                return None
        if has_t:
            try:
                t_min = float(self.t_min_input.text().strip() or "0")
                t_max = float(self.t_max_input.text().strip() or "1")
                if t_min >= t_max:
                    QMessageBox.warning(self, "错误", "t 范围: 最小值必须小于最大值！")
                    return None
                domain["t"] = [t_min, t_max]
            except ValueError:
                QMessageBox.warning(self, "错误", "t 范围必须为有效数字！")
                return None
        return domain

    def get_config(self):
        """构建标准配置字典"""
        p_type = self.type_combo.currentText()
        order = self.order_spin.value()
        dimension = 1 if "1D" in p_type else 2
        has_t = "含时" in p_type
        if dimension == 1 and not has_t:
            coeffs_param = [0.0] * (order + 1)
            for d in range(order + 1):
                key = "u" if d == 0 else "u" + "'" * d
                coeffs_param[d] = self.terms.get(key, 0.0)
        else:
            coeffs_param = self.terms.copy()
        return {
            "dimension": dimension,
            "order": order,
            "has_t": has_t,
            "coeffs": coeffs_param,
            "source_term": self.source_input.text().strip() or "0",
            "domain": self.read_domain(),
            "condition": self.conditions.copy(),
        }

class PreviewLabel(QTextEdit):
    """基于 QTextEdit 的公式/文本显示控件，支持富文本、滚动、复制。"""
    def __init__(self, parent=None, font_size=9, text_color='#FFFFFF'):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setLineWrapMode(QTextEdit.NoWrap)
        self.default_font_size = font_size
        self.default_text_color = text_color
        self._current_text = ""
        self._current_html = ""
        self._apply_style(text_color, font_size)
    def _apply_style(self, color: str, size: int):
        """应用样式到当前文本"""
        self.setStyleSheet(f"""
            QTextEdit {{
                font-size: {size}pt;
                color: {color};
                background: transparent;
                border: none;
                padding: 2px 4px;
            }}
        """)
    def set_color(self, color_hex: str):
        """更新文字颜色"""
        self.default_text_color = color_hex
        self._apply_style(color_hex, self.default_font_size)
        if self._current_html:
            self.setHtml(self._current_html)
        elif self._current_text:
            self.setText(self._current_text)
    def set_latex(self, latex_str: str):
        """设置显示文本，实际不做 LaTeX 转换。"""
        if latex_str is None:
            latex_str = ""
        self._current_text = latex_str
        if latex_str.strip().startswith('<') and ('</' in latex_str or '<br' in latex_str.lower()):
            self._current_html = latex_str
            self.setHtml(latex_str)
            return
        text = latex_str
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        unicode_map = {
            r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
            r'\epsilon': 'ε', r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ',
            r'\iota': 'ι', r'\kappa': 'κ', r'\lambda': 'λ', r'\mu': 'μ',
            r'\nu': 'ν', r'\xi': 'ξ', r'\omicron': 'ο', r'\pi': 'π',
            r'\rho': 'ρ', r'\sigma': 'σ', r'\tau': 'τ', r'\upsilon': 'υ',
            r'\phi': 'φ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
            r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ',
            r'\Xi': 'Ξ', r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ',
            r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
            r'\infty': '∞', r'\partial': '∂', r'\nabla': '∇',
            r'\sum': 'Σ', r'\int': '∫', r'\oint': '∮',
            r'\prod': 'Π', r'\sqrt': '√',
            r'\rightarrow': '→', r'\leftarrow': '←',
            r'\Rightarrow': '⇒', r'\Leftarrow': '⇐',
            r'\leftrightarrow': '↔', r'\Leftrightarrow': '⇔',
            r'\times': '×', r'\cdot': '·', r'\pm': '±', r'\mp': '∓',
            r'\neq': '≠', r'\leq': '≤', r'\geq': '≥', r'\approx': '≈',
            r'\equiv': '≡', r'\propto': '∝',
            r'\forall': '∀', r'\exists': '∃', r'\emptyset': '∅',
            r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\supset': '⊃',
            r'\cup': '∪', r'\cap': '∩',
            r'\prime': '′', r'\doubleprime': '″',
            r'\text': '', r'\mathrm': '', r'\mathbf': '',
            r'\left': '', r'\right': '',
        }
        for cmd, uni in unicode_map.items():
            text = text.replace(cmd, uni)
        def convert_frac(match):
            num = match.group(1).strip('{}')
            den = match.group(2).strip('{}')
            return f"({num})/({den})"
        text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', convert_frac, text)
        text = re.sub(r'(\w+?)_\{([^}]*)\}', r'\1<sub>\2</sub>', text)
        text = re.sub(r'(\w+?)_(\w+)', r'\1<sub>\2</sub>', text)
        text = re.sub(r'(\w+?)\^\{([^}]*)\}', r'\1<sup>\2</sup>', text)
        text = re.sub(r'(\w+?)\^(\w+)', r'\1<sup>\2</sup>', text)
        def clean_braces(match):
            content = match.group(1)
            if '<' in content and '>' in content:
                return f"{{{content}}}"
            return content
        text = re.sub(r'\{([^{}]*)\}', clean_braces, text)
        func_names = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'sinh', 'cosh', 'tanh', 'coth', 'arcsin', 'arccos', 'arctan', 
                      'log', 'ln', 'lg', 'exp', 'sqrt', 'max', 'min', 'sup', 'inf', 'lim']
        text = re.sub(r'\\sum_\{([^}]*)\}\^\{([^}]*)\}', r'Σ<sub>\1</sub><sup>\2</sup>', text)
        text = re.sub(r'\\int_\{([^}]*)\}\^\{([^}]*)\}', r'∫<sub>\1</sub><sup>\2</sup>', text)
        text = re.sub(r'\s+', ' ', text).strip()
        html = f'<span style="font-family: \'Times New Roman\', \'Microsoft YaHei\', sans-serif;">{text}</span>'
        self._current_html = html
        self.setHtml(html)

class SolverThread(QThread):
    """
    基于 function_factory 计算引擎的后台求解线程，
    可将前端界面传入的 problem_config 转换为标准参数，并直接调用 solve_pde 顶层接口求解。
    """
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    def __init__(self, problem_config: dict):
        super().__init__()
        self.config = problem_config
    def run(self):
        try:
            self.log_signal.emit("正在解析方程配置，准备接入 function_factory 核心引擎...")
            dimension = int(self.config.get('dimension', 1))
            has_t = bool(self.config.get('has_t', False))
            order = int(self.config.get('order', 2))
            coeffs = self.config.get('coeffs')
            if coeffs is None:
                coeffs = [1, 0, 1] if (dimension == 1 and not has_t) else {"u_xx": 1.0}
            source_term = str(self.config.get('source_term', '0'))
            condition = self.config.get('conditions', self.config.get('condition', []))
            domain = self.config.get('domain')
            if not domain:
                domain = {"x": [0.0, 1.0]}
                if dimension == 2:
                    domain["y"] = [0.0, 1.0]
                if has_t:
                    domain["t"] = [0.0, 1.0]
            self.log_signal.emit(f"参数解析完成: 维度={dimension}D | 含时={has_t} | 阶数={order}")
            self.log_signal.emit("正在调用 solve_pde 执行精确解/数值基准解计算...")
            res_dict = solve_pde(
                dimension=dimension,
                order=order,
                has_t=has_t,
                coeffs=coeffs,
                source_term=source_term,
                domain=domain,
                condition=condition
            )
            exact_func = res_dict.get("exact_solution")
            exact_expr = res_dict.get("exact_expression")
            loss_funcs = res_dict.get("loss_functions", [])
            self.log_signal.emit("解结构计算完成，正在生成前端响应...")
            if exact_func is None and exact_expr is None:
                self.error_signal.emit("引擎未能获得精确解或数值基准解，请检查定解条件是否完整。")
                return
            var_symbols = ['x']
            if dimension == 2:
                var_symbols.append('y')
            if has_t:
                var_symbols.append('t')
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
