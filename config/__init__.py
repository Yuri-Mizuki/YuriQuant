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

    @classmethod
    def discipline(cls) -> dict:
        """L2 段落契约：冻结日历（train/valid/test 边界单一真源）。

        Returns:
            {"begin", "train_end", "valid_end"}（8 位整型日期）；
            test 段 = valid_end 之后，只允许最终验证与上线后监控。
        """
        d = cls.get().get("discipline") or {}
        return {
            "begin": int(d.get("begin", 20220101)),
            "train_end": int(d.get("train_end", 20231231)),
            "valid_end": int(d.get("valid_end", 20241231)),
        }

    @classmethod
    def costs(cls) -> dict:
        """交易成本（全项目**唯一真源**，缺省值与 settings.yaml 的 `costs` 段一致）。

        2026-09-10 收敛：此前费率分散在 `backtest` 段与 `model_portfolio` 段两处，
        差 2~3 倍、互相不可比。现在引擎缺省费率（backtest/engine.py）与生产管线
        （scripts/run_model_portfolio.default_costs）都经由本方法取值。

        Returns:
            {commission_rate, commission_min, stamp_duty, slippage_bp,
             short_borrow_rate, short_margin_ratio}

        新增消费方请走这里，不要在别处硬编码费率字面量——否则 config 改动后会漂移。
        """
        c = cls.get().get("costs") or {}
        return {
            "commission_rate": float(c.get("commission_rate", 0.0003)),
            "commission_min": float(c.get("commission_min", 5.0)),
            "stamp_duty": float(c.get("stamp_duty", 0.001)),
            "slippage_bp": float(c.get("slippage_bp", 10.0)),
            "short_borrow_rate": float(c.get("short_borrow_rate", 0.08)),
            "short_margin_ratio": float(c.get("short_margin_ratio", 1.0)),
        }

    @classmethod
    def benchmarks(cls) -> dict:
        """对照基准指数（各报告/策略的"与谁比"**唯一真源**）。

        2026-09-11 收敛：此前指数代码散落在 scripts 里（`rolling_grid_alla.BENCH_INDEX`
        = 000001.SH、`jq_style_report.BENCH_INDEX` = 399317.SZ、
        `run_etf_rotation.BENCH` = 000300.SH），加上 `backtest.benchmark` 与
        `model_portfolio.benchmark`，共 5 处字面量、改一个要满仓找。
        ⚠️ **三者取值本就不同**（用途不同，非同一概念的重复），故这里按**用途**分键，
        不强行合并成一个值——合并会静默改变某条链路的对照基准。

        Returns:
            {"default": 引擎缺省对照（backtest.benchmark）,
             "all_a": 全A组合对照（上证指数，rolling_grid_alla / model_portfolio）,
             "etf": ETF 轮动对照（沪深300）,
             "report_a_share": 全A报告图表对照（国证A指，中证全指行情缺失的替代）,
             "report_a_share_label": 上者的展示名}

        新增消费方请走这里，不要在别处硬编码指数字面量。
        """
        b = cls.get().get("benchmarks") or {}
        bt = cls.get().get("backtest") or {}
        mp = cls.get().get("model_portfolio") or {}
        default = str(b.get("default") or bt.get("benchmark") or "000300.SH")
        return {
            "default": default,
            "all_a": str(b.get("all_a") or mp.get("benchmark") or "000001.SH"),
            "etf": str(b.get("etf") or "000300.SH"),
            "report_a_share": str(b.get("report_a_share") or "399317.SZ"),
            "report_a_share_label": str(
                b.get("report_a_share_label") or "国证A指(399317)≈中证全指"),
        }

    @classmethod
    def monitoring(cls) -> dict:
        """生产化监控阈值（monitoring/ 包单一真源，缺省值与 settings.yaml 一致）。"""
        m = cls.get().get("monitoring") or {}
        return {
            "window": int(m.get("window", 60)),
            "window_long": int(m.get("window_long", 252)),
            "max_stale_days": int(m.get("max_stale_days", 7)),
            "min_coverage": float(m.get("min_coverage", 0.5)),
            "warn_ic_retention": float(m.get("warn_ic_retention", 0.5)),
            "min_monotonicity": float(m.get("min_monotonicity", 0.5)),
            "min_t_nw_recent": float(m.get("min_t_nw_recent", 1.0)),
            "ledger_root": str(m.get("ledger_root", "reports/monitoring")),
        }
