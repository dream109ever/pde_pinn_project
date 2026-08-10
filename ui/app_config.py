# ui/app_config.py
"""
全局配置管理模块。

提供应用程序配置的持久化存储与读取功能，采用单例模式管理配置数据。
配置以 JSON 格式保存在用户目录下的 `MyPINN/config.json` 文件中。
"""
import os
import json
from PyQt5.QtCore import QObject, pyqtSignal

class AppConfig(QObject):
    """
    全局配置管理器（单例模式）。

    负责应用程序配置的加载、保存和读取。配置变更时会发出 config_changed 信号，
    便于其他模块响应配置更新。

    :signal config_changed: 配置变更信号
    """
    config_changed = pyqtSignal()
    _instance = None
    @classmethod
    def instance(cls):
        """
        获取 AppConfig 的单例实例。

        :return: AppConfig 单例对象
        :rtype: AppConfig
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    def __init__(self):
        super().__init__()
        self._config = {}
        appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
        config_dir = os.path.join(appdata, 'MyPINN')
        if not os.path.exists(config_dir):
            os.makedirs(config_dir, exist_ok=True)
        self._config_file = os.path.join(config_dir, 'config.json')
        self.DEFAULTS = {
            "theme": "sky_blue",
        }
        self._load()
    def _load(self):
        """从 JSON 文件加载配置，若文件不存在或读取失败则使用默认值。"""
        if os.path.exists(self._config_file):
            try:
                with open(self._config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for key in self.DEFAULTS:
                    if key in data:
                        self._config[key] = data[key]
            except Exception as e:
                print(f"[AppConfig] 读取配置文件失败: {e}")
                self._config = {}
        for key, default_val in self.DEFAULTS.items():
            if key not in self._config:
                self._config[key] = default_val
        self._save()
    def _save(self):
        """将当前配置保存到 JSON 文件。"""
        try:
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[AppConfig] 保存配置文件失败: {e}")
    def get(self, key, default=None):
        """
        读取配置项。

        :param key: 配置键名
        :type key: str
        :param default: 键不存在时返回的默认值，若为 None 则使用 DEFAULTS 中的值
        :type default: Any
        :return: 配置值
        :rtype: Any
        """
        if default is None and key in self.DEFAULTS:
            default = self.DEFAULTS[key]
        return self._config.get(key, default)
    def set(self, key, value):
        """
        保存配置项（立即写入文件并发出变更信号）。

        :param key: 配置键名
        :type key: str
        :param value: 配置值
        :type value: Any
        """
        self._config[key] = value
        self._save()
        self.config_changed.emit()
    def get_int(self, key, default=None):
        """
        以整数类型读取配置项。

        :param key: 配置键名
        :type key: str
        :param default: 默认值
        :type default: Optional[Any]
        :return: 整数值
        :rtype: int
        """
        return int(self.get(key, default))
    def get_float(self, key, default=None):
        """
        以浮点数类型读取配置项。

        :param key: 配置键名
        :type key: str
        :param default: 默认值
        :type default: Optional[Any]
        :return: 浮点数值
        :rtype: float
        """
        return float(self.get(key, default))
    def get_bool(self, key, default=None):
        """
        以布尔类型读取配置项。

        :param key: 配置键名
        :type key: str
        :param default: 默认值
        :type default: Optional[Any]
        :return: 布尔值
        :rtype: bool
        """
        val = self.get(key, default)
        if isinstance(val, str): return val.lower() == 'true'
        return bool(val)
    def get_str(self, key, default=None):
        """
        以字符串类型读取配置项。

        :param key: 配置键名
        :type key: str
        :param default: 默认值
        :type default: Optional[Any]
        :return: 字符串值
        :rtype: str
        """
        return str(self.get(key, default))

def config():
    """
    获取 AppConfig 单例实例的便捷函数。

    :return: AppConfig 单例对象
    :rtype: AppConfig
    """
    return AppConfig.instance()
