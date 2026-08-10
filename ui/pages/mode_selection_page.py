# ui/pages/mode_selection_page.py
"""
模式选择页面模块。

提供 PINN 求解器的主页入口，包含精确解析解模式和 PINN 神经网络模式两个核心入口按钮，
以及软件偏好设置和退出功能。
"""
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QApplication, QToolButton
from ui.theme_manager import ThemeManager
from .base_widgets import BasePage
from .settings_dialog_page import SettingsDialog

class ModeSelectionPage(BasePage):
    """
    极简主页：继承 BasePage 统一背景，保持精细调节的字号与排版，支持动态主题与设置。

    :signal mode_selected: 模式选择信号，携带模式名称字符串 ("exact" 或 "pinn")
    """
    mode_selected = pyqtSignal(str)
    def __init__(self, parent=None):
        """
        :param parent: 父控件
        :type parent: Optional[QWidget]
        """
        super().__init__(parent)
        self.init_ui()
        self.update_styles()
        ThemeManager.instance().theme_changed.connect(lambda _: self.update_styles())
    def init_ui(self):
        """初始化用户界面布局。"""
        self.setObjectName("ModeSelectionPage")
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 10)
        # --- 顶栏：右上角偏好设置按钮 ---
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.settings_btn = QToolButton()
        self.settings_btn.setText("⚙")
        self.settings_btn.setToolTip("软件偏好设置")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.clicked.connect(self.open_settings)
        top_bar.addWidget(self.settings_btn)
        main_layout.addLayout(top_bar)
        # --- 顶部：标题 ---
        main_layout.addStretch(1)
        self.title = QLabel("PDE 科学计算平台")
        self.title.setFont(QFont("Microsoft YaHei", 25, QFont.Bold))
        self.title.setStyleSheet("color: #0F172A;")
        self.title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.title)
        main_layout.addStretch(1)
        # --- 中间：核心按钮 ---
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(20)
        btn_layout.setAlignment(Qt.AlignCenter)
        self.exact_btn = QPushButton("精确解析解模式")
        self.exact_btn.setProperty("class", "mode-btn")
        self.exact_btn.setFixedWidth(360)
        self.exact_btn.setCursor(Qt.PointingHandCursor)
        self.exact_btn.clicked.connect(lambda: self.mode_selected.emit("exact"))
        self.pinn_btn = QPushButton("PINN 神经网络模式")
        self.pinn_btn.setProperty("class", "mode-btn")
        self.pinn_btn.setFixedWidth(360)
        self.pinn_btn.setCursor(Qt.PointingHandCursor)
        self.pinn_btn.clicked.connect(lambda: self.mode_selected.emit("pinn"))
        btn_layout.addWidget(self.exact_btn)
        btn_layout.addWidget(self.pinn_btn)
        main_layout.addLayout(btn_layout)
        main_layout.addStretch(2)
        # --- 底部状态栏 ---
        bottom_layout = QHBoxLayout()
        info_layout = QVBoxLayout()
        info_layout.setSpacing(2)
        self.version_label = QLabel("version：1.2.0")
        self.author_label = QLabel("by：dream109ever")
        info_layout.addWidget(self.version_label)
        info_layout.addWidget(self.author_label)
        self.exit_btn = QPushButton("退出")
        self.exit_btn.setObjectName("exitBtn")
        self.exit_btn.setCursor(Qt.PointingHandCursor)
        self.exit_btn.clicked.connect(QApplication.quit)
        bottom_layout.addLayout(info_layout)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.exit_btn, alignment=Qt.AlignRight | Qt.AlignBottom)
        main_layout.addLayout(bottom_layout)
    def open_settings(self):
        """打开设置弹窗。"""
        dialog = SettingsDialog(self)
        dialog.exec_()
    def update_styles(self):
        """
        更新页面样式，响应主题切换。

        覆盖基类方法，针对主页特有控件进行样式更新。
        """
        theme = ThemeManager.instance().current
        self.title.setStyleSheet(f"color: {theme.text_primary}; background: transparent;")
        self.version_label.setStyleSheet(f"color: {theme.text_secondary}; font-size: 10px; background: transparent;")
        self.author_label.setStyleSheet(f"color: {theme.text_secondary}; font-size: 10px; background: transparent;")
        self.settings_btn.setStyleSheet(f"""
            QToolButton {{
                background-color: {theme.btn_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 15px;
                font-size: 16px;
            }}
            QToolButton:hover {{
                background-color: {theme.btn_hover};
            }}
        """)
        self.setStyleSheet(f"""
            QLabel {{
                background: transparent;
            }}
            QPushButton.mode-btn {{
                background-color: {theme.btn_bg};
                color: {theme.text_primary};
                border: 1px solid {theme.btn_border};
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 16px;
            }}
            QPushButton.mode-btn:hover {{
                background-color: {theme.btn_hover_bg};
                border: 1px solid {theme.btn_hover_border};
                color: {theme.btn_hover_text};
            }}
            QPushButton.mode-btn:pressed {{
                background-color: {theme.btn_hover_bg};
                color: {theme.btn_hover_text};
            }}
            QPushButton#exitBtn {{
                background-color: {theme.btn_bg};
                color: {theme.text_secondary};
                border: 1px solid {theme.btn_border};
                border-radius: 4px;
                padding: 5px 14px;
                font-size: 11px;
            }}
            QPushButton#exitBtn:hover {{
                background-color: #EF4444;
                color: #FFFFFF;
                border: 1px solid #EF4444;
            }}
        """)
