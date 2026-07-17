"""配置加载：读取 settings.yaml，支持环境变量占位符替换。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"

# Real-data credentials are kept in the repository-local .env file.  Loading
# it here keeps command-line runs and IDE runs consistent; existing process
# environment variables still take precedence.
load_dotenv(_PROJECT_ROOT / ".env", override=False)

_ENV_PATTERN = re.compile(r"\$\{([A-Z_]+)(?::([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    """递归替换 ${VAR:default} 占位符为环境变量值。"""
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            var, default = m.group(1), m.group(2) or ""
            return os.environ.get(var, default)
        return _ENV_PATTERN.sub(_repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_config(path: Path | str | None = None) -> dict:
    """加载 yaml 配置并展开环境变量占位符。"""
    p = Path(path) if path else _CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_env(raw)


class Config:
    """单例式配置访问器。"""
    _data: dict | None = None

    @classmethod
    def load(cls, path: Path | str | None = None) -> dict:
        cls._data = load_config(path)
        return cls._data

    @classmethod
    def get(cls) -> dict:
        if cls._data is None:
            cls.load()
        return cls._data  # type: ignore

    @classmethod
    def datasource(cls) -> dict:
        return cls.get()["datasource"]

    @classmethod
    def cache(cls) -> dict:
        return cls.get()["cache"]

    @classmethod
    def universe(cls) -> dict:
        return cls.get()["universe"]

    @classmethod
    def fetch(cls) -> dict:
        return cls.get()["fetch"]
