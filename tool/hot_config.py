import os, json, time
from typing import Any, Dict

class hot_config:

    def __init__(self, path: str, defaults: Dict[str, Any] | None = None):
        self.path = path
        self.defaults = defaults or {}
        self._mtime = 0.0
        self.data: Dict[str, Any] = {}
        self.ensure_exists()
        self.reload(force=True)

    def ensure_exists(self):
        if not os.path.exists(self.path):
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.defaults, f, indent=2, ensure_ascii=False)

    def reload(self, force: bool=False) -> bool:
        try:
            m = os.path.getmtime(self.path)
        except FileNotFoundError:
            self.ensure_exists()
            m = os.path.getmtime(self.path)
        if force or (m > self._mtime):
            with open(self.path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
            self._mtime = m
            return True
        return False

    def get(self, key: str, default: Any=None) -> Any:
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.data)
