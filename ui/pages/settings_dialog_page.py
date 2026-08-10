# ui/pages/settings_dialog_page.py
"""
设置对话框模块。

提供全局软件偏好设置弹窗，包含外观主题切换和高级设置占位功能。
"""
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QGroupBox, QPushButton, QFormLayout
from PyQt5.QtCore import Qt
from .base_widgets import BaseDialog
from ui.theme_manager import ThemeManager

class SettingsDialog(BaseDialog):
    """全局设置弹窗。"""
    def __init__(self, parent=None):
        """
        :param parent: 父控件
        :type parent: Optional[QWidget]
        """
        super().__init__(parent)
        self.setWindowTitle("软件偏好设置")
        self.setFixedSize(420, 320)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.init_ui()
        self.load_settings()
    def init_ui(self):
        """初始化用户界面布局。"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        # 1. 外观板块
        style_group = QGroupBox("外观与主题")
        style_layout = QFormLayout(style_group)
        self.theme_combo = QComboBox()
        self.theme_combo.setCursor(Qt.PointingHandCursor)
        tm = ThemeManager.instance()
        for key, theme_obj in tm.THEMES.items(): self.theme_combo.addItem(theme_obj.name, userData=key)
        # 默认选中当前主题
        current_idx = self.theme_combo.findData(tm._current_theme_key)
        if current_idx >= 0: self.theme_combo.setCurrentIndex(current_idx)
        self.theme_combo.currentIndexChanged.connect(self.on_theme_changed)
        style_layout.addRow("界面主题色系:", self.theme_combo)
        # 2. 高级设置（占位，显示"功能开发中"）
        advanced_group = QGroupBox("高级设置")
        advanced_group.setEnabled(False)
        advanced_layout = QVBoxLayout(advanced_group)
        placeholder = QLabel("功能开发中")
        placeholder.setStyleSheet("color: gray; font-style: italic;")
        placeholder.setAlignment(Qt.AlignCenter)
        advanced_layout.addWidget(placeholder)
        # 底部确定按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        close_btn = QPushButton("完成")
        close_btn.setFixedWidth(90)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        main_layout.addWidget(style_group)
        main_layout.addWidget(advanced_group)
        main_layout.addLayout(btn_layout)
    def on_theme_changed(self, index):
        """
        主题切换事件响应。

        :param index: 当前选中的索引
        :type index: int
        """
        theme_key = self.theme_combo.itemData(index)
        ThemeManager.instance().set_theme(theme_key)
    def load_settings(self):
        """加载当前配置到界面。"""
        tm = ThemeManager.instance()
        current_idx = self.theme_combo.findData(tm._current_theme_key)
        if current_idx >= 0:
            self.theme_combo.setCurrentIndex(current_idx)
