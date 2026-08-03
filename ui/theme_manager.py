from dataclasses import dataclass
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QColor
from .app_config import AppConfig

@dataclass
class ThemePalette:
    name: str
    is_dark: bool
    # 背景渐变三配色
    bg_start: QColor
    bg_mid: QColor
    bg_end: QColor
    # 正弦波与节点透明叠加色 (RGBA)
    wave_color: QColor
    node_color: QColor
    # UI 控件配色
    card_bg: str       
    text_primary: str  
    text_secondary: str
    btn_bg: str        
    btn_hover: str     
    btn_border: str    
    # 鼠标悬停状态的专属高亮配色
    btn_hover_bg: str
    btn_hover_text: str
    btn_hover_border: str

class ThemeManager(QObject):
    """单例主题管理器：负责主题切换与 QSS 样式生成"""
    theme_changed = pyqtSignal(str)
    _instance = None
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def __init__(self):
        super().__init__()
        # 预设多套色系（可扩充）
        self.THEMES = {
            "sky_blue": ThemePalette(
                name="冰川蓝 (默认)",
                is_dark=False,
                bg_start=QColor("#F0F9FF"),
                bg_mid=QColor("#E0F2FE"),
                bg_end=QColor("#BAE6FD"),
                wave_color=QColor(255, 255, 255, 140),
                node_color=QColor(255, 255, 255, 180),
                card_bg="rgba(255, 255, 255, 0.82)",  
                text_primary="#0F172A",           
                text_secondary="#475569",          
                btn_bg="rgba(255, 255, 255, 0.65)",
                btn_hover="rgba(255, 255, 255, 0.95)",
                btn_border="rgba(186, 230, 253, 0.8)",
                btn_hover_bg="#0284C7",
                btn_hover_text="#FFFFFF",
                btn_hover_border="#0284C7"
            ),
            "dark_obsidian": ThemePalette(
                name="深海曜石 (暗黑模式)",
                is_dark=True,
                bg_start=QColor("#0F172A"),
                bg_mid=QColor("#1E293B"),
                bg_end=QColor("#334155"),
                wave_color=QColor(56, 189, 248, 40),
                node_color=QColor(56, 189, 248, 90),
                card_bg="rgba(30, 41, 59, 0.75)",
                text_primary="#F8FAFC",
                text_secondary="#94A3B8",
                btn_bg="rgba(51, 65, 85, 0.6)",
                btn_hover="rgba(71, 85, 105, 0.8)",
                btn_border="rgba(100, 116, 139, 0.5)",
                btn_hover_bg="#38BDF8",
                btn_hover_text="#0F172A",
                btn_hover_border="#38BDF8"
            ),
            "emerald_mint": ThemePalette(
                name="薄荷翡翠 (清新)",
                is_dark=False,
                bg_start=QColor("#F0FDF4"),
                bg_mid=QColor("#DCFCE7"),
                bg_end=QColor("#A7F3D0"),
                wave_color=QColor(255, 255, 255, 140),
                node_color=QColor(255, 255, 255, 180),
                card_bg="rgba(255, 255, 255, 0.82)",
                text_primary="#064E3B",
                text_secondary="#047857",
                btn_bg="rgba(255, 255, 255, 0.65)",
                btn_hover="rgba(255, 255, 255, 0.95)",
                btn_border="rgba(167, 243, 208, 0.8)",
                btn_hover_bg="#059669",
                btn_hover_text="#FFFFFF",
                btn_hover_border="#059669"
            ),
            "cyber_purple": ThemePalette(
                name="幻境紫罗兰",
                is_dark=False,
                bg_start=QColor("#FAF5FF"),
                bg_mid=QColor("#F3E8FF"),
                bg_end=QColor("#E9D5FF"),
                wave_color=QColor(255, 255, 255, 140),
                node_color=QColor(255, 255, 255, 180),
                card_bg="rgba(255, 255, 255, 0.82)",
                text_primary="#3B0764",
                text_secondary="#6B21A8",
                btn_bg="rgba(255, 255, 255, 0.65)",
                btn_hover="rgba(255, 255, 255, 0.95)",
                btn_border="rgba(233, 213, 255, 0.8)",
                btn_hover_bg="#7C3AED",
                btn_hover_text="#FFFFFF",
                btn_hover_border="#7C3AED"
            )
        }
        self._current_theme_key = "sky_blue"
    @property
    def current(self) -> ThemePalette:
        return self.THEMES[self._current_theme_key]
    def set_theme(self, theme_key: str):
        if theme_key in self.THEMES and theme_key != self._current_theme_key:
            self._current_theme_key = theme_key
            AppConfig.instance().set("theme", theme_key)
            self.theme_changed.emit(theme_key)
    def load_from_config(self):
        theme_key = AppConfig.instance().get_str("theme", "sky_blue")
        if theme_key in self.THEMES and theme_key != self._current_theme_key:
            self._current_theme_key = theme_key
            self.theme_changed.emit(theme_key)
