# ui/app_config.py
import os
import json
from PyQt5.QtCore import QObject, pyqtSignal

class AppConfig(QObject):
    """全局配置管理器（单例）"""
    config_changed = pyqtSignal()
    _instance = None
    @classmethod
    def instance(cls):
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
        """从文件加载配置"""
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
        """保存当前配置到文件"""
        try:
            with open(self._config_file, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"[AppConfig] 保存配置文件失败: {e}")
    def get(self, key, default=None):
        """读取配置项，如果不存在则返回默认值"""
        if default is None and key in self.DEFAULTS:
            default = self.DEFAULTS[key]
        return self._config.get(key, default)
    def set(self, key, value):
        """保存配置项（立即写入文件）"""
        self._config[key] = value
        self._save()
        self.config_changed.emit()
    def get_int(self, key, default=None):
        return int(self.get(key, default))
    def get_float(self, key, default=None):
        return float(self.get(key, default))
    def get_bool(self, key, default=None):
        val = self.get(key, default)
        if isinstance(val, str): return val.lower() == 'true'
        return bool(val)
    def get_str(self, key, default=None):
        return str(self.get(key, default))

# 便捷函数
def config():
    return AppConfig.instance()
