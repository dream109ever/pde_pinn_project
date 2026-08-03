from PyQt5.QtWidgets import QTextEdit
from PyQt5.QtCore import Qt

class PreviewLabel(QTextEdit):
    """基于 QTextEdit 的公式显示控件，支持富文本、滚动、复制"""
    def __init__(self, parent=None, font_size=None, text_color='#FFFFFF'):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setLineWrapMode(QTextEdit.NoWrap)

    def set_color(self, color_hex: str):
        """更新文字颜色"""
        self.setStyleSheet(f"font-size: {self.font().pointSize()}pt; color: {color_hex};")

    def set_latex(self, latex_str: str):
        """设置显示文本，兼容旧接口"""
        # 将 LaTeX 符号转换为 Unicode/HTML
        text = latex_str
        # 常见 LaTeX 转 Unicode
        replacements = {
            r'\sum': 'Σ',
            r'\int': '∫',
            r'\alpha': 'α',
            r'\beta': 'β',
            r'\gamma': 'γ',
            r'\pi': 'π',
            r'\infty': '∞',
            r'\cdot': '·',
            r'\times': '×',
            r'\rightarrow': '→',
            r'\left': '',
            r'\right': '',
            r'\text': '',
            r'\partial': '∂',
            r'\sqrt': '√',
            r'\frac': '',
            r'\sin': 'sin',
            r'\cos': 'cos',
            r'\tan': 'tan',
            r'\exp': 'exp',
            r'\log': 'log',
            r'\ln': 'ln',
            r'\theta': 'θ',
            r'\lambda': 'λ',
            r'\mu': 'μ',
            r'\sigma': 'σ',
            r'\omega': 'ω',
        }
        for latex, unicode_char in replacements.items():
            text = text.replace(latex, unicode_char)
        # 去掉剩余的 $ 和 {}
        text = text.replace('$', '').replace('{', '').replace('}', '')
        # 用 HTML 上标下标替换 _ 和 ^
        # 简单处理：用 HTML 格式
        html_text = self._convert_to_html(text)
        self.setHtml(html_text)

    def _convert_to_html(self, text: str) -> str:
        """将纯文本中的 _ 和 ^ 转换为 HTML 上下标"""
        # 简单转换：处理 u_xx 和 u^2 格式
        import re
        # 处理下标 _ 例如 u_xx → u<sub>xx</sub>
        text = re.sub(r'(\w+?)_(\w+)', r'\1<sub>\2</sub>', text)
        # 处理上标 ^ 例如 u^2 → u<sup>2</sup>
        text = re.sub(r'(\w+?)\^(\w+)', r'\1<sup>\2</sup>', text)
        return f'<span style="font-family: \'Microsoft YaHei\', \'SimHei\', sans-serif;">{text}</span>'
